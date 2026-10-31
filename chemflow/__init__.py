"""
ChemFlow: an autonomous scientific discovery layer built on DeepChem.

DeepChem provides the scientific machine learning (featurizers, models,
datasets, splitters). ChemFlow adds a research agent that investigates a
question over repeated cycles inside a scientific sandbox:

* :mod:`chemflow.sandbox` -- Scientific Sandbox (rules, limits, approvals)
* :mod:`chemflow.memory` -- Research Memory (persistent, resumable sessions)
* :mod:`chemflow.hypotheses` -- Autonomous Hypothesis Generation
* :mod:`chemflow.experiments` -- Computational Experimentation
* :mod:`chemflow.selection` -- Experiment Selection
* :mod:`chemflow.evaluation` -- Self-Evaluation
* :mod:`chemflow.learning` -- Learning Engine (learns from validated data)
* :mod:`chemflow.gaps` -- Knowledge-Gap Detection
* :mod:`chemflow.reporting` -- Research Reporting
* :mod:`chemflow.agent` -- Research Agent coordinating the loop
"""
from chemflow.config import ResearchObjective, SandboxPolicy, SessionConfig
from chemflow.agent import ResearchAgent
from chemflow.sandbox import SandboxViolation

__version__ = "0.1.0"

__all__ = [
    "ResearchAgent", "ResearchObjective", "SandboxPolicy", "SessionConfig",
    "SandboxViolation"
]
