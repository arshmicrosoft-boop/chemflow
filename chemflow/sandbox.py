"""
The Scientific Sandbox.

The sandbox gives the research agent a controlled environment: it may
investigate freely inside the rules, but it cannot make unsupported claims or
step outside its limits. Concretely it enforces

* dataset boundaries -- the agent only sees labels it is entitled to see; the
  unlabeled pool is revealed only through budgeted virtual assays and the
  hold-out set is locked behind a counted, purpose-logged gate,
* compute and runtime limits -- per-experiment and per-run time limits and a
  total experiment budget,
* reproducibility -- every experiment is seeded and records its full
  specification, data fingerprint and software versions,
* experiment logging -- every action, including blocked and failed ones, is
  written to research memory,
* evidence requirements, confidence thresholds and independent validation --
  a finding is only certified as validated when all of them are met, and
* human approval -- real-world laboratory experiments are never executed;
  they are queued as approval requests for a human to decide.
"""
import concurrent.futures
import hashlib
import json
import os
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import deepchem as dc
from chemflow import data as cfdata
from chemflow.config import ResearchObjective, SandboxPolicy
from chemflow.memory import ResearchMemory


class SandboxViolation(Exception):
    """Raised when an action would break a sandbox rule."""


class DataVault:
    """Owns the dataset and enforces dataset boundaries.

    The data is partitioned once per session and the partition is stored in
    memory, so a resumed session sees exactly the same boundaries:

    * ``labeled`` -- initially known measurements the agent can train on,
    * ``pool`` -- molecules whose labels are hidden; revealed only through a
      budgeted virtual assay,
    * ``validation`` -- used by the agent to evaluate itself,
    * ``holdout`` -- locked set for independent validation of claims.
    """

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 memory: ResearchMemory, session_dir: str):
        self.objective = objective
        self.policy = policy
        self.memory = memory
        self.cache_dir = os.path.join(session_dir, "features")
        os.makedirs(self.cache_dir, exist_ok=True)

        df = pd.read_csv(objective.dataset)
        for col in (objective.smiles_column, objective.target_column):
            if col not in df.columns:
                raise ValueError("Column %r not found in %s" %
                                 (col, objective.dataset))
        df = df[[objective.smiles_column, objective.target_column]].copy()
        df.columns = ["smiles", "y"]
        df["smiles"] = df["smiles"].astype(str).str.strip()
        ok = df["smiles"].map(cfdata.valid_smiles) & df["y"].notna()
        if (~ok).any():
            memory.log("data_rows_dropped",
                       count=int((~ok).sum()),
                       reason="invalid SMILES or missing target")
        df = df[ok].drop_duplicates("smiles").reset_index(drop=True)
        if len(df) < 50:
            raise ValueError("ChemFlow needs at least 50 valid molecules")
        self.df = df
        self.smiles = df["smiles"].tolist()
        self._index = {s: i for i, s in enumerate(self.smiles)}
        self._features: Dict[str, np.ndarray] = {}

        partition = memory.get_state("partition")
        if partition is None:
            partition = self._make_partition()
            memory.set_state("partition", partition)
            memory.log("data_partitioned",
                       **{k: len(v) for k, v in partition.items()})
        self.partition = {
            k: np.array([self._index[s] for s in v], dtype=int)
            for k, v in partition.items()
        }

        self.candidates: Optional[List[str]] = None
        if objective.candidates:
            cdf = pd.read_csv(objective.candidates)
            cand = cdf[objective.smiles_column].astype(str).str.strip()
            self.candidates = [s for s in cand if cfdata.valid_smiles(s)]

    def _make_partition(self) -> Dict[str, List[str]]:
        p = self.policy
        rest = 1 - p.validation_fraction - p.holdout_fraction
        rest_idx, valid_idx, hold_idx = cfdata.scaffold_split(
            self.smiles, (rest, p.validation_fraction, p.holdout_fraction),
            seed=p.seed)
        rng = np.random.RandomState(p.seed)
        rest_idx = rng.permutation(rest_idx)
        n_lab = int(round(len(self.smiles) * p.initial_labeled_fraction))
        names = (("labeled", rest_idx[:n_lab]), ("pool", rest_idx[n_lab:]),
                 ("validation", valid_idx), ("holdout", hold_idx))
        return {k: [self.smiles[i] for i in v] for k, v in names}

    # -- features --------------------------------------------------------
    def features(self, featurizer: str) -> np.ndarray:
        """All-molecule feature matrix for a featurizer, cached on disk."""
        if featurizer not in self._features:
            path = os.path.join(self.cache_dir, featurizer + ".npy")
            if os.path.exists(path):
                X = np.load(path)
            else:
                X = cfdata.featurize(featurizer, self.smiles)
                np.save(path, X)
            self._features[featurizer] = X
        return self._features[featurizer]

    # -- bounded views ---------------------------------------------------
    def acquired_smiles(self) -> List[str]:
        return [d["smiles"] for d in self.memory.datapoints()]

    def training_indices(self) -> np.ndarray:
        """Initially labeled molecules plus labels the agent has acquired."""
        held_out = set(self.partition["validation"]) | set(
            self.partition["holdout"])
        acquired = [
            self._index[s]
            for s in self.acquired_smiles()
            if s in self._index and self._index[s] not in held_out
        ]
        return np.concatenate([self.partition["labeled"],
                               np.array(acquired, dtype=int)])

    def training_set(self, featurizer: str) -> dc.data.NumpyDataset:
        ds = self._dataset(featurizer, self.training_indices())
        # Laboratory results for molecules outside the dataset (approved
        # candidate proposals) are featurized on the fly.
        extra = [
            d for d in self.memory.datapoints(source="lab")
            if d["smiles"] not in self._index
        ]
        if not extra:
            return ds
        X = cfdata.featurize(featurizer, [d["smiles"] for d in extra])
        y = np.array([d["value"] for d in extra]).reshape(-1, 1)
        return dc.data.NumpyDataset(
            X=np.vstack([ds.X, X]),
            y=np.vstack([ds.y, y]),
            ids=np.concatenate([ds.ids, [d["smiles"] for d in extra]]))

    def validation_set(self, featurizer: str) -> dc.data.NumpyDataset:
        return self._dataset(featurizer, self.partition["validation"])

    def unlabeled_pool(self) -> np.ndarray:
        acquired = set(self.acquired_smiles())
        return np.array([
            i for i in self.partition["pool"]
            if self.smiles[i] not in acquired
        ],
                        dtype=int)

    def pool_features(self, featurizer: str) -> Tuple[np.ndarray, List[str]]:
        idx = self.unlabeled_pool()
        return self.features(featurizer)[idx], [self.smiles[i] for i in idx]

    def _dataset(self, featurizer: str,
                 idx: np.ndarray) -> dc.data.NumpyDataset:
        X = self.features(featurizer)[idx]
        y = self.df["y"].values[idx].astype(np.float64)
        return dc.data.NumpyDataset(X=X,
                                    y=y.reshape(-1, 1),
                                    ids=np.array(
                                        [self.smiles[i] for i in idx]))

    def labels_for(self, indices: np.ndarray) -> np.ndarray:
        return self.df["y"].values[indices].astype(np.float64)

    # Locked views; only the Sandbox calls these.
    def _holdout_set(self, featurizer: str) -> dc.data.NumpyDataset:
        return self._dataset(featurizer, self.partition["holdout"])

    def _reveal(self, smiles: List[str]) -> List[float]:
        pool = set(self.smiles[i] for i in self.partition["pool"])
        for s in smiles:
            if s not in pool:
                raise SandboxViolation(
                    "%s is outside the assayable pool" % s)
        return [float(self.df["y"].values[self._index[s]]) for s in smiles]

    def fingerprint(self) -> str:
        """Hash of the current training data, recorded for reproducibility."""
        idx = np.sort(self.training_indices())
        h = hashlib.sha256()
        h.update(json.dumps([self.smiles[i] for i in idx]).encode())
        h.update(self.labels_for(idx).tobytes())
        return h.hexdigest()[:16]


class Sandbox:
    """Enforces the :class:`SandboxPolicy` around everything the agent does."""

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 memory: ResearchMemory, vault: DataVault):
        self.objective = objective
        self.policy = policy
        self.memory = memory
        self.vault = vault
        self.run_started = time.time()
        self.versions = {
            "python": platform.python_version(),
            "deepchem": dc.__version__,
            "numpy": np.__version__,
        }
        try:
            import sklearn
            import rdkit
            self.versions["sklearn"] = sklearn.__version__
            self.versions["rdkit"] = rdkit.__version__
        except ImportError:
            pass

    # -- limits ----------------------------------------------------------
    def runtime_used(self) -> float:
        return time.time() - self.run_started

    def runtime_exhausted(self) -> bool:
        return self.runtime_used() >= self.policy.max_runtime_seconds

    def experiments_used(self) -> int:
        return len([
            e for e in self.memory.experiments()
            if e["status"] not in ("blocked",)
        ])

    def assay_used(self) -> int:
        return len(self.memory.datapoints(source="virtual_assay"))

    def assay_remaining(self) -> int:
        return max(0, self.policy.assay_budget - self.assay_used())

    def holdout_used(self) -> int:
        return self.memory.get_state("holdout_uses", 0)

    def holdout_remaining(self) -> int:
        return max(0, self.policy.max_holdout_evaluations - self.holdout_used())

    def check(self, kind: str, spec: Dict[str, Any]) -> None:
        """Raise SandboxViolation if the experiment is not permitted."""
        p = self.policy
        if kind not in p.allowed_experiment_kinds:
            raise SandboxViolation("experiment kind %r is not allowed" % kind)
        if self.experiments_used() >= p.max_experiments_total:
            raise SandboxViolation("experiment budget exhausted")
        if self.runtime_exhausted():
            raise SandboxViolation("runtime budget exhausted")
        if "featurizer" in spec and spec["featurizer"] not in \
                p.allowed_featurizers:
            raise SandboxViolation("featurizer %r is not allowed" %
                                   spec["featurizer"])
        if "model" in spec and spec["model"] not in p.allowed_models:
            raise SandboxViolation("model %r is not allowed" % spec["model"])
        if kind == "virtual_assay" and self.assay_remaining() <= 0:
            raise SandboxViolation("virtual assay budget exhausted")
        if kind == "holdout_validation" and self.holdout_remaining() <= 0:
            raise SandboxViolation("hold-out evaluation budget exhausted")

    # -- execution -------------------------------------------------------
    def execute(self, hypothesis_id: Optional[str], kind: str,
                spec: Dict[str, Any], fn: Callable[[int, str], Dict[str, Any]],
                expected: Optional[Dict[str, Any]] = None,
                seed: Optional[int] = None) -> Dict[str, Any]:
        """Run ``fn(seed, experiment_id)`` as a logged, seeded, time-limited
        experiment.

        Returns the stored experiment record. Failures, timeouts and blocked
        experiments are recorded too, so research memory keeps failures as
        well as successes.
        """
        if seed is None:
            seed = self.policy.seed + len(self.memory.experiments())
        full_spec = dict(spec,
                         data_fingerprint=self.vault.fingerprint(),
                         versions=self.versions)
        eid = self.memory.add_experiment(hypothesis_id, kind, full_spec, seed,
                                         expected)
        try:
            self.check(kind, spec)
        except SandboxViolation as e:
            self.memory.update_experiment(eid, status="blocked", error=str(e))
            self.memory.log("experiment_blocked", id=eid, reason=str(e))
            raise

        self.memory.update_experiment(eid,
                                      status="running",
                                      started=time.time())
        self.memory.log("experiment_started", id=eid, kind=kind, spec=spec)
        np.random.seed(seed)
        limit = min(self.policy.max_experiment_seconds,
                    max(1.0,
                        self.policy.max_runtime_seconds - self.runtime_used()))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn, seed, eid)
        try:
            result = future.result(timeout=limit)
            status, error = "completed", None
        except concurrent.futures.TimeoutError:
            # Python threads cannot be killed; the result is discarded.
            result, status = None, "timeout"
            error = "exceeded %.0fs time limit" % limit
        except Exception as e:  # noqa: B902 - experiments may fail any way
            result, status, error = None, "failed", "%s: %s" % (
                type(e).__name__, e)
        finally:
            pool.shutdown(wait=False)
        self.memory.update_experiment(eid,
                                      status=status,
                                      result=result,
                                      error=error,
                                      finished=time.time())
        self.memory.log("experiment_finished",
                        id=eid,
                        status=status,
                        error=error)
        return self.memory.experiment(eid)

    # -- gated data access ----------------------------------------------
    def holdout(self, featurizer: str, purpose: str) -> dc.data.NumpyDataset:
        """Counted, logged access to the locked hold-out set."""
        if self.holdout_remaining() <= 0:
            raise SandboxViolation("hold-out evaluation budget exhausted")
        self.memory.set_state("holdout_uses", self.holdout_used() + 1)
        self.memory.log("holdout_accessed", purpose=purpose)
        return self.vault._holdout_set(featurizer)

    def assay(self, smiles: List[str], experiment_id: Optional[str]) -> List[
            float]:
        """Reveal hidden pool labels (a virtual experiment) within budget."""
        if len(smiles) > self.assay_remaining():
            raise SandboxViolation(
                "assay request of %d exceeds remaining budget %d" %
                (len(smiles), self.assay_remaining()))
        values = self.vault._reveal(smiles)
        self.memory.add_datapoints(smiles, values, "virtual_assay",
                                   experiment_id)
        return values

    # -- claims ----------------------------------------------------------
    def certify(self, evidence: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Decide what a body of evidence entitles the agent to claim.

        ``evidence`` may contain ``supporting_experiments`` (list of ids),
        ``meets_confidence``, ``reproducible`` and ``independent_validation``
        (each True, False or None for "not yet tested").

        Returns ``(status, reasons)`` where status is "validated",
        "provisional" or "rejected".
        """
        reasons = []
        for key, label in (("meets_confidence", "confidence threshold"),
                           ("reproducible", "reproducibility check"),
                           ("independent_validation",
                            "independent validation")):
            if evidence.get(key) is False:
                reasons.append("failed %s" % label)
        if reasons:
            return "rejected", reasons
        n = len(evidence.get("supporting_experiments", []))
        if n < self.policy.min_supporting_experiments:
            reasons.append("needs %d supporting experiments, has %d" %
                           (self.policy.min_supporting_experiments, n))
        for key, label in (("meets_confidence", "confidence threshold"),
                           ("reproducible", "reproducibility check"),
                           ("independent_validation",
                            "independent validation")):
            if evidence.get(key) is None:
                reasons.append("%s not yet established" % label)
        if reasons:
            return "provisional", reasons
        return "validated", ["all evidence requirements met"]

    # -- real-world experiments -----------------------------------------
    def request_lab_experiment(self, description: str,
                               payload: Dict[str, Any]) -> str:
        """Queue a real-world experiment for human approval.

        ChemFlow never executes laboratory work. Approved requests are a
        protocol for humans to carry out; their measured results can be
        ingested back with :meth:`ResearchAgent.ingest_lab_results`.
        """
        if "lab_proposal" not in self.policy.allowed_experiment_kinds:
            raise SandboxViolation("lab proposals are not allowed")
        if len(self.memory.approvals()) >= self.policy.max_lab_proposals:
            raise SandboxViolation("lab proposal limit reached")
        return self.memory.request_approval("lab_experiment", description,
                                            payload)
