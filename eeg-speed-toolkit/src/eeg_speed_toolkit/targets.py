"""Explicit target construction for calibrated and normative speed labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TargetLabels:
    labels: np.ndarray
    thresholds: Mapping[str, Mapping[str, float]]
    description: str


@dataclass(frozen=True)
class FoldTargets:
    train_labels: np.ndarray
    test_labels: np.ndarray
    thresholds: Mapping[str, float]
    description: str


def make_within_subject_task_labels(
    duration: np.ndarray,
    tasks: np.ndarray,
    subjects: np.ndarray,
) -> TargetLabels:
    """Label trials relative to each participant-and-task median.

    This is balanced and useful for scientific association tests, but it uses
    each held-out participant's outcome distribution and therefore requires
    participant calibration in a deployment setting.
    """

    duration, tasks, subjects = _validate_inputs(duration, tasks, subjects)
    labels = np.empty(len(duration), dtype=int)
    thresholds: dict[str, dict[str, float]] = {}
    for subject in sorted(set(subjects.tolist())):
        thresholds[subject] = {}
        for task in sorted(set(tasks[subjects == subject].tolist())):
            mask = (subjects == subject) & (tasks == task)
            if np.count_nonzero(mask) < 2:
                raise ValueError(f"subject {subject}, task {task} needs at least two trials")
            threshold = float(np.median(duration[mask]))
            thresholds[subject][task] = threshold
            labels[mask] = duration[mask] > threshold
    return TargetLabels(
        labels=labels,
        thresholds=thresholds,
        description="0=fast and 1=slow relative to each subject-task median",
    )


def make_training_task_median_labels(
    duration: np.ndarray,
    tasks: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> FoldTargets:
    """Define absolute fast/slow thresholds using outer-training outcomes only."""

    duration = np.asarray(duration, dtype=float).reshape(-1)
    tasks = np.asarray(tasks).astype(str).reshape(-1)
    train = np.asarray(train_indices, dtype=int)
    test = np.asarray(test_indices, dtype=int)
    if len(duration) != len(tasks):
        raise ValueError("duration and tasks must have equal length")
    if set(train.tolist()) & set(test.tolist()):
        raise ValueError("train and test indices overlap")
    thresholds = {}
    for task in sorted(set(tasks[train].tolist())):
        values = duration[train][tasks[train] == task]
        if not len(values):
            raise ValueError(f"training split has no samples for task {task}")
        thresholds[task] = float(np.median(values))
    unseen = sorted(set(tasks[test].tolist()) - set(thresholds))
    if unseen:
        raise ValueError(f"test split contains unseen tasks: {unseen}")
    train_labels = np.asarray([duration[i] > thresholds[tasks[i]] for i in train], dtype=int)
    test_labels = np.asarray([duration[i] > thresholds[tasks[i]] for i in test], dtype=int)
    return FoldTargets(
        train_labels=train_labels,
        test_labels=test_labels,
        thresholds=thresholds,
        description="0=fast and 1=slow relative to task medians learned from outer training data",
    )


def _validate_inputs(duration, tasks, subjects):
    duration = np.asarray(duration, dtype=float).reshape(-1)
    tasks = np.asarray(tasks).astype(str).reshape(-1)
    subjects = np.asarray(subjects).astype(str).reshape(-1)
    if not (len(duration) == len(tasks) == len(subjects)):
        raise ValueError("duration, tasks, and subjects must have equal length")
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("duration must contain positive finite values")
    return duration, tasks, subjects
