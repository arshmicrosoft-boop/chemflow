"""
The Learning Engine.

Keeps a *champion* model -- the best configuration found so far -- and
improves it over repeated training cycles as validated data arrives (labels
revealed by virtual assays inside the sandbox, or approved laboratory
results). Every retraining produces a new model version that is evaluated on
the validation set; a version that is worse than its predecessor by more than
the reproducibility tolerance is not promoted, so the model can only learn
from data, not drift on noise.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import deepchem as dc
from chemflow import data as cfdata
from chemflow.config import ResearchObjective, SandboxPolicy
from chemflow.memory import ResearchMemory
from chemflow.sandbox import DataVault


class LearningEngine:

    def __init__(self, objective: ResearchObjective, policy: SandboxPolicy,
                 memory: ResearchMemory, vault: DataVault, session_dir: str):
        self.objective = objective
        self.policy = policy
        self.memory = memory
        self.vault = vault
        self.model_dir = os.path.join(session_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

    # -- champion --------------------------------------------------------
    @property
    def champion(self) -> Optional[Dict[str, Any]]:
        return self.memory.get_state("champion")

    def learning_curve(self) -> List[Dict[str, Any]]:
        return self.memory.get_state("learning_curve", [])

    def better(self, a: float, b: Optional[float]) -> bool:
        if b is None:
            return True
        return a < b if self.objective.lower_is_better else a > b

    def consider(self, hypothesis: Dict[str, Any], experiment: Dict[str, Any],
                 model: Optional[dc.models.SklearnModel]) -> bool:
        """Promote a newly evaluated configuration if it beats the champion."""
        metric = experiment["result"]["metric"]
        current = self.champion
        if current is not None and not self.better(metric, current["metric"]):
            return False
        p = hypothesis["params"]
        version = len(self.learning_curve()) + 1
        champion = {
            "hypothesis_id": hypothesis["id"],
            "featurizer": p["featurizer"],
            "model": p["model"],
            "hyperparameters": p["hyperparameters"],
            "metric": metric,
            "ci": [experiment["result"]["ci_low"],
                   experiment["result"]["ci_high"]],
            "n_train": experiment["result"]["n_train"],
            "version": version,
        }
        self._record_version(champion, "new champion from %s" %
                             experiment["id"], accepted=True)
        if model is not None:
            self._save(model, version)
        self.memory.set_state("champion", champion)
        self.memory.log("champion_changed",
                        hypothesis_id=hypothesis["id"],
                        metric=metric,
                        previous=None if current is None else
                        current["metric"])
        return True

    def _record_version(self, champion: Dict[str, Any], reason: str,
                        accepted: bool) -> None:
        curve = self.learning_curve()
        curve.append({
            "version": len(curve) + 1,
            "cycle": self.memory.cycle,
            "hypothesis_id": champion["hypothesis_id"],
            "model": "%s/%s" % (champion["model"], champion["featurizer"]),
            "n_train": champion["n_train"],
            "metric": champion["metric"],
            "ci": champion["ci"],
            "reason": reason,
            "accepted": accepted,
        })
        self.memory.set_state("learning_curve", curve)

    def _save(self, model: dc.models.SklearnModel, version: int) -> None:
        path = os.path.join(self.model_dir, "champion_v%d.joblib" % version)
        joblib.dump(model.model, path)
        self.memory.set_state("champion_model_path", path)

    def load_champion_model(self) -> Optional[dc.models.SklearnModel]:
        path = self.memory.get_state("champion_model_path")
        if path is None or not os.path.exists(path):
            return None
        return dc.models.SklearnModel(joblib.load(path))

    # -- training --------------------------------------------------------
    def fit_and_evaluate(
            self, featurizer: str, family: str, params: Dict[str, Any],
            seed: int) -> Tuple[dc.models.SklearnModel, Dict[str, Any]]:
        """Train a configuration on the current training data and evaluate it
        on the validation set with a bootstrap confidence interval."""
        train = self.vault.training_set(featurizer)
        valid = self.vault.validation_set(featurizer)
        model = cfdata.build_model(family, self.objective.task, params, seed)
        model.fit(train)
        y_pred = cfdata.predict(model, valid.X, self.objective.task)
        y_true = valid.y.reshape(-1)
        metric = self.objective.primary_metric
        value = cfdata.score(metric, y_true, y_pred)
        lo, hi = cfdata.bootstrap_interval(metric, y_true, y_pred,
                                           self.policy.confidence_level,
                                           seed=seed)
        residuals = (y_true - y_pred).tolist()
        return model, {
            "metric": value,
            "ci_low": lo,
            "ci_high": hi,
            "n_train": len(train),
            "n_valid": len(valid),
            "valid_residuals": residuals,
        }

    def retrain(self, seed: int, reason: str) -> Optional[Dict[str, Any]]:
        """Retrain the champion configuration on all validated data.

        Returns the new learning-curve entry, or None without a champion.
        """
        champ = self.champion
        if champ is None:
            return None
        model, result = self.fit_and_evaluate(champ["featurizer"],
                                              champ["model"],
                                              champ["hyperparameters"], seed)
        candidate = dict(champ,
                         metric=result["metric"],
                         ci=[result["ci_low"], result["ci_high"]],
                         n_train=result["n_train"])
        # Accept unless the new version is clearly worse than the old one.
        worse = (result["metric"] - champ["metric"]) * (
            1 if self.objective.lower_is_better else -1)
        accepted = worse <= self.policy.reproducibility_tolerance
        self._record_version(candidate, reason, accepted)
        entry = self.learning_curve()[-1]
        if accepted:
            candidate["version"] = entry["version"]
            self._save(model, entry["version"])
            self.memory.set_state("champion", candidate)
            self.memory.log("model_retrained",
                            version=entry["version"],
                            metric=result["metric"],
                            n_train=result["n_train"])
        else:
            self.memory.log("model_update_rejected",
                            metric=result["metric"],
                            previous=champ["metric"],
                            reason="validation performance regressed")
        return dict(entry, previous_metric=champ["metric"])

    # -- uncertainty -----------------------------------------------------
    def uncertainty(self, featurizer: str, X: np.ndarray,
                    seed: int) -> np.ndarray:
        """Per-molecule uncertainty from a random-forest ensemble.

        Regression: standard deviation across trees. Classification: entropy
        proxy p(1-p) of the ensemble probability.
        """
        train = self.vault.training_set(featurizer)
        y = train.y.reshape(-1)
        if self.objective.task == "regression":
            rf = RandomForestRegressor(n_estimators=200,
                                       random_state=seed,
                                       n_jobs=-1).fit(train.X, y)
            per_tree = np.stack([t.predict(X) for t in rf.estimators_])
            return per_tree.std(axis=0)
        rf = RandomForestClassifier(n_estimators=200,
                                    random_state=seed,
                                    n_jobs=-1).fit(train.X, y)
        p = rf.predict_proba(X)[:, -1]
        return p * (1 - p)

    def predict(self, smiles: List[str],
                seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Champion predictions and uncertainties for arbitrary molecules."""
        champ = self.champion
        model = self.load_champion_model()
        if champ is None or model is None:
            raise RuntimeError("No champion model available yet")
        X = cfdata.featurize(champ["featurizer"], smiles)
        pred = cfdata.predict(model, X, self.objective.task)
        return pred, self.uncertainty(champ["featurizer"], X, seed)
