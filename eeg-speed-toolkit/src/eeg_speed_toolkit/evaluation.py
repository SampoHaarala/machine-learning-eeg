"""Metrics and permutation tests operating only on supplied predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def classification_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, Any]:
    """Calculate the primary binary classification metrics."""

    truth, prediction = _validate_labels(true_labels, predicted_labels)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
        "n_samples": int(len(truth)),
    }


def per_subject_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    subjects: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Calculate the same metrics separately for every participant."""

    truth, prediction = _validate_labels(true_labels, predicted_labels)
    subjects = np.asarray(subjects).astype(str).reshape(-1)
    if len(subjects) != len(truth):
        raise ValueError("subjects must match labels")
    return {
        subject: classification_metrics(truth[subjects == subject], prediction[subjects == subject])
        for subject in sorted(set(subjects.tolist()), key=_natural_key)
    }


def permutation_pvalue(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    subjects: np.ndarray,
    tasks: np.ndarray,
    repeats: int = 1999,
    seed: int = 20261006,
) -> float:
    """Shuffle labels within participant and task while predictions stay fixed."""

    truth, prediction = _validate_labels(true_labels, predicted_labels)
    subjects = np.asarray(subjects).astype(str).reshape(-1)
    tasks = np.asarray(tasks).astype(str).reshape(-1)
    if not (len(subjects) == len(tasks) == len(truth)):
        raise ValueError("subjects and tasks must match labels")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    observed = balanced_accuracy_score(truth, prediction)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repeats):
        shuffled = truth.copy()
        for subject in np.unique(subjects):
            for task in np.unique(tasks[subjects == subject]):
                index = np.flatnonzero((subjects == subject) & (tasks == task))
                shuffled[index] = rng.permutation(shuffled[index])
        exceedances += balanced_accuracy_score(shuffled, prediction) >= observed
    return float((1 + exceedances) / (repeats + 1))


def _validate_labels(true_labels, predicted_labels):
    truth = np.asarray(true_labels, dtype=int).reshape(-1)
    prediction = np.asarray(predicted_labels, dtype=int).reshape(-1)
    if len(truth) != len(prediction) or not len(truth):
        raise ValueError("true and predicted labels must have equal nonzero length")
    if not set(np.unique(truth)).issubset({0, 1}) or not set(np.unique(prediction)).issubset({0, 1}):
        raise ValueError("labels must be binary 0/1")
    return truth, prediction


def _natural_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)
