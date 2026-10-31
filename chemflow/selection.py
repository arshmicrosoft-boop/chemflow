"""
Experiment Selection: decide which experiment is most useful next.

Every open hypothesis proposes its next experiment (via the evaluation
engine). The selector scores each candidate by its expected scientific value:

* model training runs are scored by *expected improvement* over the current
  champion, estimated from results of related configurations, plus a bonus
  for untested model families or featurizations (exploration),
* replication and hold-out validation are scored by how close the
  hypothesis is to becoming a validated claim -- hold-out access is a scarce
  resource and is spent on the strongest candidates,
* structure-activity tests are scored by the strength of the evidence that
  motivated them,
* virtual assays are scored by the remaining assay budget and by how much
  earlier acquisitions actually improved the model (diminishing returns).

The selector also records what it expects each experiment to produce, so the
evaluation engine can later check how well the agent predicts its own
experiments.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from chemflow.config import ResearchObjective, SandboxPolicy
from chemflow.evaluation import EvaluationEngine
from chemflow.learning import LearningEngine
from chemflow.memory import ResearchMemory
from chemflow.sandbox import Sandbox


@dataclass
class Action:
    kind: str
    hypothesis_id: str
    utility: float
    rationale: str
    spec: Dict[str, Any] = field(default_factory=dict)
    expected: Optional[Dict[str, Any]] = None


class ExperimentSelector:

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 memory: ResearchMemory, sandbox: Sandbox,
                 evaluation: EvaluationEngine, learning: LearningEngine):
        self.objective = objective
        self.policy = policy
        self.memory = memory
        self.sandbox = sandbox
        self.evaluation = evaluation
        self.learning = learning

    # -- estimates -------------------------------------------------------
    def _results(self) -> List[Dict[str, Any]]:
        out = []
        for e in self.memory.experiments(kind="train_evaluate",
                                         status="completed"):
            h = self.memory.hypothesis(e["hypothesis_id"])
            out.append(dict(h["params"], metric=e["result"]["metric"]))
        return out

    def expected_metric(self, params: Dict[str, Any],
                        results: List[Dict[str, Any]]) -> (float, float):
        """Mean and spread of the metric predicted for a configuration."""
        if not results:
            return self.objective.success_threshold, 0.2
        all_m = np.array([r["metric"] for r in results])
        same_model = [
            r["metric"] for r in results if r["model"] == params["model"]
        ]
        same_feat = [
            r["metric"]
            for r in results
            if r["featurizer"] == params["featurizer"]
        ]
        parts = [np.mean(x) for x in (same_model, same_feat) if x]
        mu = float(np.mean(parts)) if parts else float(all_m.mean())
        sigma = float(max(all_m.std(), 0.02))
        return mu, sigma

    def expected_improvement(self, mu: float, sigma: float,
                             best: Optional[float]) -> float:
        if best is None:
            return sigma
        gain = (best - mu) if self.objective.lower_is_better else (mu - best)
        z = gain / sigma
        return float(gain * norm.cdf(z) + sigma * norm.pdf(z))

    def assay_value(self) -> float:
        """Average validation-metric gain per past acquisition round."""
        gains = []
        for e in self.memory.experiments(kind="virtual_assay",
                                         status="completed"):
            r = e["result"]
            gain = r["metric_after"] - r["metric_before"]
            gains.append(-gain if self.objective.lower_is_better else gain)
        return float(np.mean(gains[-3:])) if gains else 0.05

    # -- candidates ------------------------------------------------------
    def candidates(self) -> List[Action]:
        results = self._results()
        champion = self.learning.champion
        best = None if champion is None else champion["metric"]
        tested_pairs = {(r["featurizer"], r["model"]) for r in results}
        actions = []
        for h in self.memory.hypotheses():
            step = self.evaluation.next_step(h)
            if step is None:
                continue
            kind, spec = step
            action = self._score(h, kind, spec, results, best, tested_pairs)
            if action is not None:
                actions.append(action)
        actions.sort(key=lambda a: a.utility, reverse=True)
        return actions

    def _score(self, h: Dict[str, Any], kind: str, spec: Dict[str, Any],
               results: List[Dict[str, Any]], best: Optional[float],
               tested_pairs: set) -> Optional[Action]:
        p = h["params"]
        hid = h["id"]
        if kind == "train_evaluate":
            mu, sigma = self.expected_metric(p, results)
            ei = self.expected_improvement(mu, sigma, best)
            novelty = 0.3 if (p["featurizer"], p["model"]) not in \
                tested_pairs else 0.0
            return Action(
                kind, hid, 0.5 + 10 * ei + novelty,
                "expected %s %.3f, expected improvement %.3f%s" %
                (self.objective.primary_metric, mu, ei,
                 ", untested model/featurizer" if novelty else ""),
                {"featurizer": p["featurizer"], "model": p["model"]},
                {"metric": mu})
        ev = self.evaluation.model_evidence(h) if h["kind"] == "model" \
            else None
        is_champion = (self.learning.champion is not None and
                       self.learning.champion["hypothesis_id"] == hid)
        if kind == "replicate":
            closeness = 1.0 if is_champion else max(
                0.0, 1 - abs(ev["metric"] - (best or ev["metric"])) * 5)
            return Action(kind, hid, 0.6 + 0.4 * closeness,
                          "check reproducibility of %s = %.3f" %
                          (self.objective.primary_metric, ev["metric"]),
                          {"featurizer": p["featurizer"],
                           "model": p["model"]}, {"metric": ev["metric"]})
        if kind == "holdout_validation":
            if h["kind"] == "model":
                near_best = best is not None and abs(ev["metric"] - best) <= \
                    self.policy.reproducibility_tolerance
                if not (is_champion or near_best):
                    # Do not spend scarce hold-out access on weaker models.
                    return None
                return Action(kind, hid, 0.95,
                              "independent validation of a strong model",
                              {"featurizer": p["featurizer"],
                               "model": p["model"]}, {"metric": ev["metric"]})
            if self.sandbox.holdout_remaining() <= 1:
                return None  # keep the last hold-out use for models
            return Action(kind, hid, 0.55,
                          "independent validation of a replicated SAR effect")
        if kind == "sar_test":
            if spec["split"] == "training":
                return Action(kind, hid, 0.4 + 0.3 * h["prior"],
                              "test SAR effect motivated by %s" % h["origin"],
                              spec)
            return Action(kind, hid, 0.6,
                          "replicate SAR effect on independent molecules",
                          spec)
        if kind == "virtual_assay":
            if self.learning.champion is None or \
                    self.sandbox.assay_remaining() <= 0:
                return None
            remaining = self.sandbox.assay_remaining() / \
                self.policy.assay_budget
            value = self.assay_value()
            utility = 0.5 + 0.3 * remaining + 5 * max(value, 0.0)
            if p.get("strategy") == "group":
                utility += 0.05
            expected = None
            if best is not None:
                sign = -1 if self.objective.lower_is_better else 1
                expected = {"metric": best + sign * max(value, 0.0)}
            return Action(kind, hid, utility,
                          "%d assays left; past rounds gained %.3f %s" %
                          (self.sandbox.assay_remaining(), value,
                           self.objective.primary_metric), {}, expected)
        return None

    def select(self) -> Optional[Action]:
        actions = self.candidates()
        return actions[0] if actions else None
