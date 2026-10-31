"""
Configuration objects for ChemFlow research sessions.

A session is described by two things:

* a :class:`ResearchObjective` -- the scientific question, the data it is
  allowed to use and what "success" means, and
* a :class:`SandboxPolicy` -- the rules, limits and conditions the research
  agent must operate under.

Both are plain dataclasses that round-trip through JSON so a session can be
stored on disk and resumed later.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

REGRESSION_METRICS = ("r2", "rmse", "mae")
CLASSIFICATION_METRICS = ("roc_auc", "accuracy")
# Metrics where a lower value is better.
LOWER_IS_BETTER = ("rmse", "mae")


@dataclass
class ResearchObjective:
    """The scientific question a session investigates.

    Parameters
    ----------
    name: str
        Short name of the objective.
    question: str
        The research question in plain language.
    dataset: str
        Path to a CSV file with a SMILES column and a measured target column.
    smiles_column: str
        Name of the column holding SMILES strings.
    target_column: str
        Name of the column holding the measured property.
    task: str
        Either "regression" or "classification".
    primary_metric: str
        Metric used to judge models ("r2", "rmse", "mae", "roc_auc" or
        "accuracy").
    success_threshold: float
        A model only counts as a reliable predictor if the lower confidence
        bound of its metric clears this threshold (upper bound for
        lower-is-better metrics).
    desirable_direction: str
        "high" or "low": which values of the target are scientifically
        desirable. Used when proposing candidate molecules.
    candidates: str, optional
        Optional CSV of unlabeled molecules (same SMILES column) to rank and
        propose for real-world testing. Proposals always need human approval.
    """
    name: str
    question: str
    dataset: str
    smiles_column: str
    target_column: str
    task: str = "regression"
    primary_metric: str = "r2"
    success_threshold: float = 0.7
    desirable_direction: str = "high"
    candidates: Optional[str] = None

    def validate(self) -> None:
        if self.task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        allowed = (REGRESSION_METRICS
                   if self.task == "regression" else CLASSIFICATION_METRICS)
        if self.primary_metric not in allowed:
            raise ValueError("primary_metric for %s must be one of %s" %
                             (self.task, allowed))
        if self.desirable_direction not in ("high", "low"):
            raise ValueError("desirable_direction must be 'high' or 'low'")

    @property
    def lower_is_better(self) -> bool:
        return self.primary_metric in LOWER_IS_BETTER


@dataclass
class SandboxPolicy:
    """Rules, limits and conditions enforced by the :class:`Sandbox`.

    The agent is free to investigate inside these bounds but cannot make
    claims that the evidence does not support.
    """
    # Evidence requirements: a finding needs at least this many independent
    # supporting experiments before it can be validated.
    min_supporting_experiments: int = 2
    # Confidence thresholds.
    confidence_level: float = 0.95
    significance_alpha: float = 0.05
    min_effect_size: float = 0.3
    min_group_size: int = 15
    # Reproducibility: a replicate run with a different seed must agree with
    # the original within this tolerance on the primary metric.
    reproducibility_tolerance: float = 0.05
    # Independent validation: number of times the locked hold-out set may be
    # queried during a session. Limits overfitting to the hold-out set.
    max_holdout_evaluations: int = 12
    # Dataset boundaries.
    initial_labeled_fraction: float = 0.3
    validation_fraction: float = 0.15
    holdout_fraction: float = 0.15
    # Virtual-assay budget: number of hidden labels the agent may reveal.
    assay_budget: int = 200
    assay_batch_size: int = 25
    # Compute and runtime limits.
    max_cycles_per_run: int = 25
    max_runtime_seconds: float = 1800.0
    max_experiment_seconds: float = 300.0
    max_experiments_total: int = 200
    # What the agent is allowed to run.
    allowed_experiment_kinds: List[str] = field(default_factory=lambda: [
        "train_evaluate", "replicate", "sar_test", "virtual_assay",
        "holdout_validation", "lab_proposal"
    ])
    allowed_featurizers: List[str] = field(
        default_factory=lambda: ["ecfp", "rdkit_descriptors", "maccs"])
    allowed_models: List[str] = field(
        default_factory=lambda: ["random_forest", "mlp", "ridge", "gbm"])
    # Real-world experiments always require human approval. This cannot be
    # switched off; it is kept as a field so it is visible in session files.
    require_human_approval_for_lab: bool = True
    max_lab_proposals: int = 10
    seed: int = 0

    def validate(self) -> None:
        if not self.require_human_approval_for_lab:
            raise ValueError("require_human_approval_for_lab cannot be "
                             "disabled: real-world experiments always need "
                             "human approval.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        fracs = (self.initial_labeled_fraction + self.validation_fraction +
                 self.holdout_fraction)
        if fracs >= 1:
            raise ValueError("labeled + validation + holdout fractions must "
                             "leave room for the unlabeled pool")
        if self.min_supporting_experiments < 1:
            raise ValueError("min_supporting_experiments must be >= 1")


@dataclass
class SessionConfig:
    objective: ResearchObjective
    sandbox: SandboxPolicy = field(default_factory=SandboxPolicy)

    def validate(self) -> None:
        self.objective.validate()
        self.sandbox.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": asdict(self.objective),
            "sandbox": asdict(self.sandbox)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionConfig":
        config = cls(objective=ResearchObjective(**d["objective"]),
                     sandbox=SandboxPolicy(**d.get("sandbox", {})))
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str) -> "SessionConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
