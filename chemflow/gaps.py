"""
Knowledge-Gap Detection: find unanswered questions and under-explored areas.

The detector inspects research memory and the data boundaries and reports
gaps such as

* the research question itself is still open (no validated predictor),
* hypotheses that were proposed but never tested,
* findings that ended inconclusive because the data lacked power,
* chemical groups that are common among unseen molecules but rare in the
  training data,
* chemical groups where the champion model's errors concentrate, and
* regions of chemical space where the model is most uncertain.

Gaps that more data could close come with a suggested acquisition, which the
agent can turn into an acquisition hypothesis.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from chemflow.config import ResearchObjective, SandboxPolicy
from chemflow.hypotheses import group_presence
from chemflow.learning import LearningEngine
from chemflow.memory import ResearchMemory
from chemflow.sandbox import DataVault, Sandbox


class KnowledgeGapDetector:

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 memory: ResearchMemory, vault: DataVault, sandbox: Sandbox,
                 learning: LearningEngine):
        self.objective = objective
        self.policy = policy
        self.memory = memory
        self.vault = vault
        self.sandbox = sandbox
        self.learning = learning

    def _gap(self, kind: str, description: str, severity: float,
             suggestion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "kind": kind,
            "description": description,
            "severity": round(float(severity), 3),
            "suggestion": suggestion,
        }

    def detect(self) -> List[Dict[str, Any]]:
        gaps = []
        champ = self.learning.champion
        validated_models = [
            h for h in self.memory.hypotheses(status="validated", kind="model")
        ]
        if not validated_models:
            if champ is None:
                text = "No predictive model has been evaluated yet."
            else:
                text = ("No model is validated yet; best validation %s is "
                        "%.3f against a success threshold of %s." %
                        (self.objective.primary_metric, champ["metric"],
                         self.objective.success_threshold))
            gaps.append(self._gap("open_question", text, 1.0))

        untested = self.memory.hypotheses(status="proposed")
        untested = [h for h in untested if h["kind"] != "acquisition"]
        if untested:
            gaps.append(
                self._gap("untested_hypotheses",
                          "%d hypotheses have not been tested yet." %
                          len(untested), min(1.0, len(untested) / 20)))

        pool_smiles = [self.vault.smiles[i] for i in self.vault.unlabeled_pool()]
        pool_groups = group_presence(pool_smiles) if pool_smiles else {}
        for h in self.memory.hypotheses(status="inconclusive", kind="sar"):
            group = h["params"]["group"]
            n_pool = int(pool_groups.get(group, np.zeros(0)).sum())
            gaps.append(
                self._gap(
                    "inconclusive_finding",
                    "Effect of %s on %s is inconclusive; %d unmeasured "
                    "pool molecules contain one." %
                    (group, self.objective.target_column, n_pool), 0.6,
                    {"strategy": "group", "group": group}
                    if n_pool else None))

        train_ids = list(self.vault.training_set(
            self.policy.allowed_featurizers[0]).ids)
        train_groups = group_presence(train_ids)
        unseen = pool_smiles + [
            self.vault.smiles[i] for i in self.vault.partition["validation"]
        ]
        if unseen:
            unseen_groups = group_presence(unseen)
            for group, mask in unseen_groups.items():
                f_unseen = mask.mean()
                n_train = int(train_groups[group].sum())
                f_train = n_train / max(1, len(train_ids))
                if f_unseen >= 0.05 and n_train < self.policy.min_group_size \
                        and f_unseen > 2 * f_train:
                    gaps.append(
                        self._gap(
                            "underrepresented_chemistry",
                            "%s appears in %.0f%% of unseen molecules but "
                            "only %d training molecules." %
                            (group, 100 * f_unseen, n_train), 0.5,
                            {"strategy": "group", "group": group}))

        gaps += self._error_hotspots(champ)

        if champ is not None and self.sandbox.assay_remaining() > 0 and \
                pool_smiles:
            gaps.append(
                self._gap(
                    "uncertain_region",
                    "%d unmeasured molecules remain; %d virtual assays left "
                    "to reduce model uncertainty." %
                    (len(pool_smiles), self.sandbox.assay_remaining()),
                    0.4 * self.sandbox.assay_remaining() /
                    self.policy.assay_budget + 0.2,
                    {"strategy": "uncertainty"}))

        for g in gaps:
            if g["suggestion"] and g["suggestion"].get("strategy") == "group":
                if not pool_groups.get(g["suggestion"]["group"],
                                       np.zeros(0)).any():
                    g["suggestion"] = None
        gaps.sort(key=lambda g: g["severity"], reverse=True)
        self.memory.set_state("knowledge_gaps", gaps)
        self.memory.log("gaps_detected", count=len(gaps))
        return gaps

    def _error_hotspots(self, champ: Optional[Dict[str,
                                                    Any]]) -> List[Dict]:
        if champ is None or self.objective.task != "regression":
            return []
        runs = [
            e for e in self.memory.experiments(
                hypothesis_id=champ["hypothesis_id"], status="completed")
            if e["result"] and "valid_residuals" in e["result"]
        ]
        if not runs:
            return []
        residuals = np.abs(np.array(runs[-1]["result"]["valid_residuals"]))
        valid_smiles = [
            self.vault.smiles[i] for i in self.vault.partition["validation"]
        ]
        overall = residuals.mean()
        out = []
        for group, mask in group_presence(valid_smiles).items():
            if mask.sum() < 5:
                continue
            ratio = residuals[mask].mean() / max(overall, 1e-9)
            if ratio >= 1.5:
                out.append(
                    self._gap(
                        "error_hotspot",
                        "Champion errors on molecules with a %s are %.1fx "
                        "the average (n=%d)." % (group, ratio, mask.sum()),
                        min(0.9, 0.3 + 0.2 * ratio),
                        {"strategy": "group", "group": group}))
        return out
