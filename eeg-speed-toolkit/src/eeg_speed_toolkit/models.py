"""Model construction, fitting, and explicit training-only tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSelectionResult:
    estimator: BaseEstimator
    best_c: float
    validation_score: float
    cv_results: Mapping[str, Any]


def make_logistic_classifier(
    config: Mapping[str, Any],
    c: float | None = None,
    seed: int | None = None,
) -> Pipeline:
    """Construct an unfitted scaler-plus-logistic pipeline."""

    model = config["model"]
    chosen_c = float(model["c_values"][0] if c is None else c)
    if chosen_c <= 0:
        raise ValueError("c must be positive")
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=chosen_c,
            max_iter=int(model["max_iter"]),
            class_weight=model["class_weight"],
            random_state=config["seed"] if seed is None else int(seed),
        )),
    ])


def fit_classifier(
    estimator: BaseEstimator,
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> BaseEstimator:
    """Clone and fit one estimator on exactly the arrays supplied by the caller."""

    x, y = _validate_training_arrays(train_features, train_labels)
    fitted = clone(estimator)
    fitted.fit(x, y)
    return fitted


def tune_logistic_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    config: Mapping[str, Any],
) -> ModelSelectionResult:
    """Select C with group-disjoint CV inside caller-provided training data.

    The returned estimator is refit on all supplied training rows. No external
    validation or test row is accessed.
    """

    x, y = _validate_training_arrays(train_features, train_labels)
    groups = np.asarray(train_groups).astype(str).reshape(-1)
    if len(groups) != len(y):
        raise ValueError("train_groups must match train_labels")
    folds = int(config["split"]["inner_folds"])
    unique = np.unique(groups)
    if folds > len(unique):
        raise ValueError("inner_folds exceeds the number of training groups")
    search = GridSearchCV(
        make_logistic_classifier(config),
        {"model__C": list(config["model"]["c_values"])},
        scoring="balanced_accuracy",
        cv=GroupKFold(n_splits=folds),
        n_jobs=int(config["model"].get("n_jobs", -1)),
        refit=True,
    )
    search.fit(x, y, groups=groups)
    return ModelSelectionResult(
        estimator=search.best_estimator_,
        best_c=float(search.best_params_["model__C"]),
        validation_score=float(search.best_score_),
        cv_results=search.cv_results_,
    )


def _validate_training_arrays(features, labels):
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("features must be a finite trials x features matrix matching labels")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("training labels must contain binary classes 0 and 1")
    return x, y
