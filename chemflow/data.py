"""
DeepChem building blocks used by ChemFlow: featurizers, models and metrics.

ChemFlow does not reimplement any machine learning. Featurization runs
through ``deepchem.feat``, models are wrapped as ``deepchem.models``
objects and datasets are ``deepchem.data.NumpyDataset`` instances. This module
is the registry the research agent draws its experiments from.
"""
import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import deepchem as dc

logger = logging.getLogger(__name__)

# DeepChem featurizers use deprecated RDKit APIs that log on every molecule.
RDLogger.DisableLog("rdApp.*")

FEATURIZERS: Dict[str, Callable[[], dc.feat.Featurizer]] = {
    "ecfp": lambda: dc.feat.CircularFingerprint(size=2048, radius=2),
    "rdkit_descriptors": lambda: dc.feat.RDKitDescriptors(),
    "maccs": lambda: dc.feat.MACCSKeysFingerprint(),
}

# Hyperparameter options explored for each model family. The first entry is
# the default configuration tried first.
MODEL_SEARCH_SPACE: Dict[str, List[Dict[str, Any]]] = {
    "random_forest": [{
        "n_estimators": 300,
        "max_features": "sqrt"
    }, {
        "n_estimators": 300,
        "max_features": 0.3
    }, {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2
    }],
    "mlp": [{
        "hidden_layer_sizes": [256],
        "alpha": 1e-3
    }, {
        "hidden_layer_sizes": [512, 128],
        "alpha": 1e-3
    }, {
        "hidden_layer_sizes": [256],
        "alpha": 1e-1
    }],
    "ridge": [{
        "alpha": 1.0
    }, {
        "alpha": 10.0
    }, {
        "alpha": 0.1
    }],
    "gbm": [{
        "learning_rate": 0.05,
        "max_iter": 300
    }, {
        "learning_rate": 0.1,
        "max_iter": 200,
        "max_leaf_nodes": 15
    }, {
        "learning_rate": 0.03,
        "max_iter": 600
    }],
}


def valid_smiles(smiles: str) -> bool:
    return isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None


def featurize(name: str, smiles: List[str]) -> np.ndarray:
    """Featurize SMILES with a registered DeepChem featurizer.

    Molecules that fail to featurize get a row of NaN which is then replaced
    by zeros; callers are expected to have filtered invalid SMILES already.
    """
    if name not in FEATURIZERS:
        raise KeyError("Unknown featurizer %s" % name)
    raw = FEATURIZERS[name]().featurize(list(smiles))
    width = max(np.shape(x)[0] for x in raw if np.size(x) > 0)
    X = np.full((len(raw), width), np.nan, dtype=np.float64)
    for i, row in enumerate(raw):
        if np.size(row) == width:
            X[i] = row
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip extreme descriptor values so linear and neural models stay stable.
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def build_model(family: str, task: str, params: Dict[str, Any],
                seed: int) -> dc.models.SklearnModel:
    """Build a DeepChem model for the given family and hyperparameters."""
    p = dict(params)
    if family == "random_forest":
        cls = (RandomForestRegressor
               if task == "regression" else RandomForestClassifier)
        est = cls(random_state=seed, n_jobs=-1, **p)
    elif family == "mlp":
        # Feed-forward neural network. Inputs are standardised first.
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        cls = MLPRegressor if task == "regression" else MLPClassifier
        est = make_pipeline(
            StandardScaler(),
            cls(random_state=seed,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=15,
                **p))
    elif family == "ridge":
        if task == "regression":
            est = make_pipeline(StandardScaler(), Ridge(**p))
        else:
            est = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0 / p.get("alpha", 1.0),
                                   max_iter=2000,
                                   random_state=seed))
    elif family == "gbm":
        cls = (HistGradientBoostingRegressor
               if task == "regression" else HistGradientBoostingClassifier)
        est = cls(random_state=seed, **p)
    else:
        raise KeyError("Unknown model family %s" % family)
    return dc.models.SklearnModel(est)


def predict(model: dc.models.SklearnModel, X: np.ndarray,
            task: str) -> np.ndarray:
    pred = model.predict(dc.data.NumpyDataset(X))
    pred = np.asarray(pred)
    if task == "classification" and pred.ndim == 2 and pred.shape[1] == 2:
        return pred[:, 1]
    return pred.reshape(-1)


def score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred)**2)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred >= 0.5))
    raise KeyError("Unknown metric %s" % metric)


def bootstrap_interval(metric: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       confidence: float,
                       n_boot: int = 500,
                       seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if metric == "roc_auc" and len(np.unique(y_true[idx])) < 2:
            continue
        values.append(score(metric, y_true[idx], y_pred[idx]))
    alpha = (1 - confidence) / 2
    return (float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)))


def scaffold_split(smiles: List[str], fractions: Tuple[float, float, float],
                   seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bemis-Murcko scaffold split via DeepChem's ScaffoldSplitter.

    Scaffold splits keep structurally related molecules together, which makes
    validation an honest test of generalisation to new chemistry.
    """
    ds = dc.data.NumpyDataset(X=np.zeros((len(smiles), 1)), ids=smiles)
    train, valid, test = dc.splits.ScaffoldSplitter().split(ds,
                                                            *fractions,
                                                            seed=seed)
    return np.asarray(train), np.asarray(valid), np.asarray(test)
