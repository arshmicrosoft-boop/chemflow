"""
Autonomous hypothesis generation.

Hypotheses are testable, falsifiable statements derived from evidence in
research memory and in the data the sandbox allows the agent to see. Three
kinds are generated:

* ``model`` -- "featurization F with model M predicts the target on unseen
  scaffolds at least as well as the success threshold". New configurations
  are proposed around the best-performing results so far.
* ``sar`` -- structure-activity relationships: "molecules containing group G
  have higher/lower target values". Proposed only when the training data
  shows a notable effect, then tested on independent data.
* ``acquisition`` -- "measuring molecules in region R will improve the
  model". Proposed by the knowledge-gap detector.
"""
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem

from chemflow.config import ResearchObjective, SandboxPolicy
from chemflow.data import MODEL_SEARCH_SPACE
from chemflow.memory import ResearchMemory
from chemflow.sandbox import DataVault

# Functional groups used for structure-activity hypotheses.
FUNCTIONAL_GROUPS: Dict[str, str] = {
    "hydroxyl group": "[OX2H][#6]",
    "phenol": "c[OX2H]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2][#6]",
    "ether": "[OD2]([#6])[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "primary amine": "[NX3;H2][#6]",
    "amide": "[CX3](=O)[NX3]",
    "nitro group": "[N+](=O)[O-]",
    "nitrile": "C#N",
    "sulfonamide": "S(=O)(=O)N",
    "halogen": "[F,Cl,Br,I]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "aromatic ring": "a",
    "aromatic nitrogen": "n",
    "fused aromatic ring system": "c1ccc2ccccc2c1",
    "alkene": "C=C",
    "thioether": "[#16X2]([#6])[#6]",
    "phosphorus": "[P]",
}

_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FUNCTIONAL_GROUPS.items()}


def group_presence(smiles: List[str]) -> Dict[str, np.ndarray]:
    """Boolean presence vector of every functional group for each molecule."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return {
        name: np.array([m is not None and m.HasSubstructMatch(p)
                        for m in mols])
        for name, p in _PATTERNS.items()
    }


def effect_size(y: np.ndarray, mask: np.ndarray) -> float:
    """Standardised mean difference (Cohen's d) of y between mask groups."""
    a, b = y[mask], y[~mask]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) +
                      (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


class HypothesisGenerator:
    """Proposes new hypotheses from the current evidence."""

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 memory: ResearchMemory, vault: DataVault):
        self.objective = objective
        self.policy = policy
        self.memory = memory
        self.vault = vault

    def _target(self) -> str:
        return self.objective.target_column

    def _threshold_text(self) -> str:
        op = "<=" if self.objective.lower_is_better else ">="
        return "%s %s %s" % (self.objective.primary_metric, op,
                             self.objective.success_threshold)

    # -- model hypotheses ------------------------------------------------
    def model_hypothesis(self, featurizer: str, model: str, config: int,
                         origin: str, prior: float) -> Optional[str]:
        params = {
            "featurizer": featurizer,
            "model": model,
            "config": config,
            "hyperparameters": MODEL_SEARCH_SPACE[model][config],
        }
        if self.memory.has_hypothesis("model", params):
            return None
        statement = ("%s on %s features (config %d) predicts %s on unseen "
                     "scaffolds with %s" %
                     (model, featurizer, config, self._target(),
                      self._threshold_text()))
        return self.memory.add_hypothesis("model", statement, params, origin,
                                          prior)

    def initial_model_hypotheses(self) -> List[str]:
        out = []
        for featurizer in self.policy.allowed_featurizers:
            for model in self.policy.allowed_models:
                hid = self.model_hypothesis(featurizer, model, 0,
                                            "initial search space", 0.5)
                if hid:
                    out.append(hid)
        return out

    def refine_model_hypotheses(self, top_k: int = 2) -> List[str]:
        """Propose new configurations near the best results so far."""
        scored = []
        for e in self.memory.experiments(kind="train_evaluate",
                                         status="completed"):
            h = self.memory.hypothesis(e["hypothesis_id"])
            scored.append((e["result"]["metric"], h))
        if not scored:
            return []
        sign = -1 if self.objective.lower_is_better else 1
        scored.sort(key=lambda t: sign * t[0], reverse=True)
        out, seen = [], set()
        for metric, h in scored:
            p = h["params"]
            key = (p["featurizer"], p["model"])
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > top_k:
                break
            for config in range(len(MODEL_SEARCH_SPACE[p["model"]])):
                hid = self.model_hypothesis(
                    p["featurizer"], p["model"], config,
                    "refinement of %s (%s=%.3f)" %
                    (h["id"], self.objective.primary_metric, metric), 0.6)
                if hid:
                    out.append(hid)
        return out

    # -- structure-activity hypotheses ----------------------------------
    def sar_hypotheses(self) -> List[str]:
        """Propose SAR hypotheses where the training data shows an effect."""
        ds = self.vault.training_set(self.policy.allowed_featurizers[0])
        y = ds.y.reshape(-1)
        presence = group_presence(list(ds.ids))
        out = []
        for name, mask in presence.items():
            n_with = int(mask.sum())
            if n_with < self.policy.min_group_size or \
                    len(mask) - n_with < self.policy.min_group_size:
                continue
            d = effect_size(y, mask)
            # Exploratory screen at half the required effect size; the test
            # itself applies the full requirement.
            if abs(d) < self.policy.min_effect_size / 2:
                continue
            direction = "higher" if d > 0 else "lower"
            params = {
                "group": name,
                "smarts": FUNCTIONAL_GROUPS[name],
                "direction": direction
            }
            if self.memory.has_hypothesis("sar", params):
                continue
            statement = ("Molecules containing a %s have %s %s than "
                         "molecules without one" %
                         (name, direction, self._target()))
            hid = self.memory.add_hypothesis(
                "sar", statement, params,
                "training-data screen (d=%.2f, n=%d)" % (d, n_with),
                min(1.0, abs(d)))
            out.append(hid)
        return out

    # -- acquisition hypotheses -----------------------------------------
    def acquisition_hypothesis(self, strategy: str, origin: str,
                               prior: float,
                               group: Optional[str] = None) -> Optional[str]:
        params = {"strategy": strategy, "group": group,
                  "round": len(self.memory.hypotheses(kind="acquisition"))}
        if strategy == "uncertainty":
            region = "where the current model is most uncertain"
        else:
            region = "containing a %s" % group
        statement = ("Measuring molecules %s will improve the model's "
                     "%s on unseen scaffolds" %
                     (region, self.objective.primary_metric))
        return self.memory.add_hypothesis("acquisition", statement, params,
                                          origin, prior)

    def open_acquisitions(self) -> List[Dict]:
        return self.memory.hypotheses(status="proposed", kind="acquisition")

    # -- entry point -----------------------------------------------------
    def generate(self) -> List[str]:
        """Generate whatever new hypotheses the current evidence supports."""
        new = []
        if not self.memory.hypotheses(kind="model"):
            new += self.initial_model_hypotheses()
        else:
            new += self.refine_model_hypotheses()
        new += self.sar_hypotheses()
        if new:
            self.memory.log("hypotheses_generated", count=len(new))
        return new
