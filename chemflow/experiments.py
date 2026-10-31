"""
The Experiment Engine: computational experiments run with DeepChem.

Each method returns a function ``fn(seed, experiment_id) -> result`` that the
:class:`~chemflow.sandbox.Sandbox` executes under its limits. Experiment
kinds:

* ``train_evaluate`` -- train a model configuration and evaluate it on the
  validation set with a bootstrap confidence interval.
* ``replicate`` -- repeat a training run with a different seed to check
  reproducibility.
* ``sar_test`` -- statistically test a structure-activity hypothesis on the
  training or validation data.
* ``holdout_validation`` -- independent test of a model or SAR claim on the
  locked hold-out set.
* ``virtual_assay`` -- reveal hidden labels for selected pool molecules (a
  virtual experiment) and let the learning engine learn from them.
"""
from typing import Any, Callable, Dict, List

import numpy as np
from scipy.stats import mannwhitneyu

from chemflow.config import ResearchObjective, SandboxPolicy
from chemflow import data as cfdata
from chemflow.hypotheses import FUNCTIONAL_GROUPS, group_presence, effect_size
from chemflow.learning import LearningEngine
from chemflow.sandbox import DataVault, Sandbox

Experiment = Callable[[int, str], Dict[str, Any]]


def sar_statistics(smiles: List[str], y: np.ndarray, group: str,
                   direction: str) -> Dict[str, Any]:
    """One-sided Mann-Whitney tests of a group effect in both directions."""
    mask = group_presence(smiles)[group]
    with_, without = y[mask], y[~mask]
    out = {
        "n_with": int(mask.sum()),
        "n_without": int((~mask).sum()),
        "effect_size": effect_size(y, mask),
        "mean_with": float(with_.mean()) if len(with_) else None,
        "mean_without": float(without.mean()) if len(without) else None,
        "p_value": None,
        "p_opposite": None,
    }
    if len(with_) < 3 or len(without) < 3:
        return out
    alt, opp = ("greater", "less") if direction == "higher" else ("less",
                                                                 "greater")
    out["p_value"] = float(mannwhitneyu(with_, without, alternative=alt).pvalue)
    out["p_opposite"] = float(
        mannwhitneyu(with_, without, alternative=opp).pvalue)
    return out


class ExperimentEngine:

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 sandbox: Sandbox, vault: DataVault, learning: LearningEngine):
        self.objective = objective
        self.policy = policy
        self.sandbox = sandbox
        self.vault = vault
        self.learning = learning
        # Fitted models from this process, keyed by experiment id, so the
        # learning engine can persist a new champion without refitting.
        self.fitted: Dict[str, Any] = {}

    # -- model experiments ----------------------------------------------
    def train_evaluate(self, hypothesis: Dict[str, Any]) -> Experiment:
        p = hypothesis["params"]

        def run(seed: int, eid: str) -> Dict[str, Any]:
            model, result = self.learning.fit_and_evaluate(
                p["featurizer"], p["model"], p["hyperparameters"], seed)
            self.fitted[eid] = model
            return result

        return run

    def replicate(self, hypothesis: Dict[str, Any]) -> Experiment:
        # Same configuration, different seed (chosen by the sandbox).
        return self.train_evaluate(hypothesis)

    def holdout_validation(self, hypothesis: Dict[str, Any]) -> Experiment:
        p = hypothesis["params"]
        if hypothesis["kind"] == "sar":
            return self._sar_holdout(hypothesis)

        def run(seed: int, eid: str) -> Dict[str, Any]:
            train = self.vault.training_set(p["featurizer"])
            model = cfdata.build_model(p["model"], self.objective.task,
                                       p["hyperparameters"], seed)
            model.fit(train)
            hold = self.sandbox.holdout(p["featurizer"],
                                        "validate %s" % hypothesis["id"])
            y_pred = cfdata.predict(model, hold.X, self.objective.task)
            y_true = hold.y.reshape(-1)
            metric = self.objective.primary_metric
            lo, hi = cfdata.bootstrap_interval(metric, y_true, y_pred,
                                               self.policy.confidence_level,
                                               seed=seed)
            return {
                "metric": cfdata.score(metric, y_true, y_pred),
                "ci_low": lo,
                "ci_high": hi,
                "n_train": len(train),
                "n_holdout": len(hold),
            }

        return run

    # -- structure-activity experiments ---------------------------------
    def sar_test(self, hypothesis: Dict[str, Any], split: str) -> Experiment:
        p = hypothesis["params"]
        featurizer = self.policy.allowed_featurizers[0]

        def run(seed: int, eid: str) -> Dict[str, Any]:
            if split == "training":
                ds = self.vault.training_set(featurizer)
            elif split == "validation":
                ds = self.vault.validation_set(featurizer)
            else:
                raise ValueError("unknown split %s" % split)
            stats = sar_statistics(list(ds.ids), ds.y.reshape(-1),
                                   p["group"], p["direction"])
            return dict(stats, split=split)

        return run

    def _sar_holdout(self, hypothesis: Dict[str, Any]) -> Experiment:
        p = hypothesis["params"]
        featurizer = self.policy.allowed_featurizers[0]

        def run(seed: int, eid: str) -> Dict[str, Any]:
            hold = self.sandbox.holdout(featurizer,
                                        "validate %s" % hypothesis["id"])
            stats = sar_statistics(list(hold.ids), hold.y.reshape(-1),
                                   p["group"], p["direction"])
            return dict(stats, split="holdout")

        return run

    # -- virtual experiments --------------------------------------------
    def virtual_assay(self, hypothesis: Dict[str, Any]) -> Experiment:
        """Select pool molecules, reveal their labels and learn from them."""
        p = hypothesis["params"]

        def run(seed: int, eid: str) -> Dict[str, Any]:
            champ = self.learning.champion
            if champ is None:
                raise RuntimeError("virtual assays need a champion model")
            X, smiles = self.vault.pool_features(champ["featurizer"])
            if len(smiles) == 0:
                raise RuntimeError("the unlabeled pool is empty")
            unc = self.learning.uncertainty(champ["featurizer"], X, seed)
            candidates = np.arange(len(smiles))
            if p["strategy"] == "group":
                mask = group_presence(smiles)[p["group"]]
                candidates = candidates[mask]
                if len(candidates) == 0:
                    raise RuntimeError("no pool molecules contain a %s" %
                                       p["group"])
            n = min(self.policy.assay_batch_size,
                    self.sandbox.assay_remaining(), len(candidates))
            chosen = candidates[np.argsort(-unc[candidates])[:n]]
            picked = [smiles[i] for i in chosen]
            values = self.sandbox.assay(picked, eid)
            update = self.learning.retrain(
                seed, "learned from %d assay results (%s)" % (n, eid))
            return {
                "strategy": p["strategy"],
                "group": p.get("group"),
                "acquired": n,
                "smiles": picked,
                "values": values,
                "mean_uncertainty": float(unc[chosen].mean()),
                "metric_before": update["previous_metric"],
                "metric_after": update["metric"],
                "metric": update["metric"],
                "model_update_accepted": update["accepted"],
                "n_train": update["n_train"],
            }

        return run


__all__ = ["ExperimentEngine", "sar_statistics", "FUNCTIONAL_GROUPS"]
