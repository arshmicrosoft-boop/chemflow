"""
The Evaluation Engine: self-evaluation of hypotheses and of the agent itself.

For every hypothesis the engine gathers the experiments run so far, turns
them into an evidence summary (confidence, reproducibility, independent
validation) and asks the sandbox what that evidence entitles the agent to
claim. It also decides the next step a hypothesis needs, and checks the
agent's own predictions: before each experiment the selector records what it
expects; afterwards the engine records how far off it was.
"""
from typing import Any, Dict, List, Optional, Tuple

from chemflow.config import ResearchObjective, SandboxPolicy
from chemflow.memory import ResearchMemory
from chemflow.sandbox import Sandbox

STATUS_FROM_CERTIFICATE = {
    "validated": "validated",
    "provisional": "provisional",
    "rejected": "refuted",
}


class EvaluationEngine:

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 memory: ResearchMemory, sandbox: Sandbox):
        self.objective = objective
        self.policy = policy
        self.memory = memory
        self.sandbox = sandbox

    # -- helpers ---------------------------------------------------------
    def meets(self, value: float) -> bool:
        t = self.objective.success_threshold
        return value <= t if self.objective.lower_is_better else value >= t

    def bound(self, result: Dict[str, Any]) -> float:
        """The pessimistic end of a metric's confidence interval."""
        return (result["ci_high"]
                if self.objective.lower_is_better else result["ci_low"])

    def _completed(self, hid: str, kind: str,
                   split: Optional[str] = None) -> List[Dict[str, Any]]:
        out = self.memory.experiments(hypothesis_id=hid,
                                      kind=kind,
                                      status="completed")
        if split is not None:
            out = [e for e in out if e["result"].get("split") == split]
        return out

    def failures(self, hid: str) -> int:
        return len([
            e for e in self.memory.experiments(hypothesis_id=hid)
            if e["status"] in ("failed", "timeout")
        ])

    # -- evidence --------------------------------------------------------
    def model_evidence(self, h: Dict[str, Any]) -> Dict[str, Any]:
        runs = self._completed(h["id"], "train_evaluate")
        reps = self._completed(h["id"], "replicate")
        hold = self._completed(h["id"], "holdout_validation")
        ev: Dict[str, Any] = {
            "supporting_experiments": [],
            "meets_confidence": None,
            "reproducible": None,
            "independent_validation": None,
        }
        if not runs:
            return ev
        first = runs[0]["result"]
        ev["metric"] = first["metric"]
        ev["ci"] = [first["ci_low"], first["ci_high"]]
        if not self.meets(first["metric"]):
            ev["meets_confidence"] = False
        elif self.meets(self.bound(first)):
            ev["meets_confidence"] = True
        ev["supporting_experiments"] = [
            e["id"] for e in runs + reps if self.meets(e["result"]["metric"])
        ]
        if reps:
            diff = abs(reps[0]["result"]["metric"] - first["metric"])
            ev["replicate_metric"] = reps[0]["result"]["metric"]
            ev["reproducible"] = diff <= self.policy.reproducibility_tolerance
        if hold:
            ev["holdout_metric"] = hold[0]["result"]["metric"]
            ev["independent_validation"] = self.meets(
                hold[0]["result"]["metric"])
        return ev

    def _sar_outcome(self, stats: Dict[str, Any],
                     alpha: float) -> Optional[bool]:
        """True: supported. False: contradicted. None: not enough power."""
        if stats["p_value"] is None:
            return None
        if stats["p_value"] < alpha and \
                abs(stats["effect_size"]) >= self.policy.min_effect_size:
            return True
        if stats["p_opposite"] < alpha:
            return False
        return None

    def sar_evidence(self, h: Dict[str, Any]) -> Dict[str, Any]:
        train = self._completed(h["id"], "sar_test", "training")
        valid = self._completed(h["id"], "sar_test", "validation")
        hold = self._completed(h["id"], "holdout_validation")
        n_sar = max(1, len(self.memory.hypotheses(kind="sar")))
        ev: Dict[str, Any] = {
            "supporting_experiments": [],
            "meets_confidence": None,
            "reproducible": None,
            "independent_validation": None,
        }
        if train:
            # Bonferroni correction: the training screen looked at many
            # groups, so the discovery test must clear a stricter bar.
            s = train[0]["result"]
            outcome = self._sar_outcome(s, self.policy.significance_alpha /
                                        n_sar)
            ev["discovery"] = s
            ev["meets_confidence"] = False if outcome is None else outcome
            if outcome:
                ev["supporting_experiments"].append(train[0]["id"])
        if valid:
            s = valid[0]["result"]
            outcome = self._sar_outcome(s, self.policy.significance_alpha)
            ev["replication"] = s
            ev["reproducible"] = outcome
            if outcome:
                ev["supporting_experiments"].append(valid[0]["id"])
        if hold:
            s = hold[0]["result"]
            ev["holdout"] = s
            ev["independent_validation"] = self._sar_outcome(
                s, self.policy.significance_alpha)
        return ev

    # -- assessment ------------------------------------------------------
    def next_step(self, h: Dict[str, Any]) -> Optional[Tuple[str, Dict]]:
        """The next experiment a hypothesis needs, or None if it is done."""
        if h["status"] in ("refuted", "validated", "inconclusive",
                           "abandoned", "tested"):
            return None
        if self.failures(h["id"]) >= 2:
            return None
        if h["kind"] == "acquisition":
            return ("virtual_assay", {})
        if h["kind"] == "model":
            ev = self.model_evidence(h)
            if "metric" not in ev:
                return ("train_evaluate", {})
            if ev["meets_confidence"] is False:
                return None
            if ev["reproducible"] is None:
                return ("replicate", {})
            if ev["reproducible"] and ev["meets_confidence"] and \
                    ev["independent_validation"] is None:
                return ("holdout_validation", {})
            return None
        if h["kind"] == "sar":
            ev = self.sar_evidence(h)
            if "discovery" not in ev:
                return ("sar_test", {"split": "training"})
            if not ev["meets_confidence"]:
                return None
            if "replication" not in ev:
                return ("sar_test", {"split": "validation"})
            if ev["reproducible"] and ev["independent_validation"] is None \
                    and "holdout" not in ev:
                return ("holdout_validation", {})
            return None
        return None

    def assess(self, h: Dict[str, Any]) -> Optional[str]:
        """Update a hypothesis' status and finding from its evidence."""
        h = self.memory.hypothesis(h["id"])
        if h["kind"] == "acquisition":
            done = self._completed(h["id"], "virtual_assay")
            if done or self.failures(h["id"]) >= 2:
                self.memory.set_hypothesis_status(
                    h["id"], "tested" if done else "abandoned")
            return None
        ev = (self.model_evidence(h)
              if h["kind"] == "model" else self.sar_evidence(h))
        if not ev["supporting_experiments"] and ev["meets_confidence"] is None \
                and "metric" not in ev and "discovery" not in ev:
            if self.failures(h["id"]) >= 2:
                self.memory.set_hypothesis_status(h["id"], "abandoned")
            return None
        certificate, reasons = self.sandbox.certify(ev)
        status = STATUS_FROM_CERTIFICATE[certificate]
        if status == "provisional" and self.next_step(
                dict(h, status="provisional")) is None:
            # Nothing more can be done within the sandbox: the evidence is
            # neither strong enough to validate nor to refute.
            status = "inconclusive"
        self.memory.set_hypothesis_status(h["id"], status)
        ev["certificate_reasons"] = reasons
        confidence = self.confidence(h, ev)
        claim = self.claim(h, ev, status)
        finding_status = "rejected" if status == "refuted" else status
        self.memory.upsert_finding(h["id"], claim, finding_status, confidence,
                                   ev)
        return status

    def confidence(self, h: Dict[str, Any], ev: Dict[str, Any]) -> float:
        """A simple 0-1 score: share of evidence requirements satisfied."""
        checks = [
            ev.get("meets_confidence"),
            ev.get("reproducible"),
            ev.get("independent_validation"),
            len(ev["supporting_experiments"]) >=
            self.policy.min_supporting_experiments,
        ]
        return sum(1 for c in checks if c is True) / len(checks)

    def claim(self, h: Dict[str, Any], ev: Dict[str, Any], status: str) -> str:
        metric = self.objective.primary_metric
        if h["kind"] == "model":
            p = h["params"]
            text = "%s on %s features reached validation %s = %.3f" % (
                p["model"], p["featurizer"], metric, ev["metric"])
            if "ci" in ev:
                text += " (%d%% CI %.3f to %.3f)" % (
                    round(self.policy.confidence_level * 100), ev["ci"][0],
                    ev["ci"][1])
            if "holdout_metric" in ev:
                text += "; hold-out %s = %.3f" % (metric, ev["holdout_metric"])
            return text
        p = h["params"]
        d = ev.get("discovery", {})
        text = "Molecules with a %s: %s %s (d = %.2f, n = %d)" % (
            p["group"], p["direction"], self.objective.target_column,
            d.get("effect_size", 0.0), d.get("n_with", 0))
        if "replication" in ev:
            text += "; validation d = %.2f" % ev["replication"]["effect_size"]
        if "holdout" in ev:
            text += "; hold-out d = %.2f" % ev["holdout"]["effect_size"]
        return text

    # -- self-evaluation of the agent's own expectations ----------------
    def record_surprise(self, experiment: Dict[str, Any]) -> Optional[float]:
        expected = experiment.get("expected") or {}
        result = experiment.get("result") or {}
        if "metric" not in expected or "metric" not in result:
            return None
        error = result["metric"] - expected["metric"]
        log = self.memory.get_state("prediction_errors", [])
        log.append({
            "experiment_id": experiment["id"],
            "expected": expected["metric"],
            "actual": result["metric"],
            "error": error,
        })
        self.memory.set_state("prediction_errors", log)
        return error
