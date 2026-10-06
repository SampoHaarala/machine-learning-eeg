"""Index-only split functions; they never train, transform, or save data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class IndexSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray | None = None
    name: str = ""


def leave_one_subject_out_splits(
    subjects: np.ndarray,
    exclude_subjects: Iterable[str] = (),
) -> Iterator[IndexSplit]:
    """Yield one explicit train/test split for every retained participant."""

    subjects = np.asarray(subjects).astype(str).reshape(-1)
    excluded = {str(value) for value in exclude_subjects}
    keep = ~np.isin(subjects, list(excluded))
    for subject in sorted(set(subjects[keep].tolist()), key=_natural_key):
        test = np.flatnonzero(keep & (subjects == subject))
        train = np.flatnonzero(keep & (subjects != subject))
        if not len(train) or not len(test):
            raise ValueError("leave-one-subject-out produced an empty split")
        yield IndexSplit(train=train, validation=np.array([], dtype=int), test=test, name=subject)


def grouped_validation_splits(
    groups: np.ndarray,
    candidate_indices: np.ndarray | None = None,
    n_splits: int = 5,
) -> Iterator[IndexSplit]:
    """Yield train/validation folds with no group overlap.

    Pass the outer-training indices as ``candidate_indices`` to prevent an outer
    test participant from entering hyperparameter selection.
    """

    groups = np.asarray(groups).astype(str).reshape(-1)
    candidates = (
        np.arange(len(groups), dtype=int)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=int)
    )
    unique = np.unique(groups[candidates])
    if n_splits < 2 or n_splits > len(unique):
        raise ValueError("n_splits must be between 2 and the number of candidate groups")
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(candidates))
    for fold, (local_train, local_validation) in enumerate(
        splitter.split(placeholder, groups=groups[candidates])
    ):
        train, validation = candidates[local_train], candidates[local_validation]
        if set(groups[train]) & set(groups[validation]):
            raise AssertionError("group leakage")
        yield IndexSplit(train=train, validation=validation, name=f"group_fold_{fold}")


def subject_partition(
    subjects: np.ndarray,
    validation_subjects: Sequence[str],
    test_subjects: Sequence[str] = (),
    exclude_subjects: Sequence[str] = (),
) -> IndexSplit:
    """Create one caller-specified subject-disjoint train/validation/test split."""

    subjects = np.asarray(subjects).astype(str).reshape(-1)
    validation_names = {str(value) for value in validation_subjects}
    test_names = {str(value) for value in test_subjects}
    excluded = {str(value) for value in exclude_subjects}
    if validation_names & test_names:
        raise ValueError("validation and test subjects overlap")
    available = set(subjects.tolist()) - excluded
    requested = validation_names | test_names
    if not requested <= available:
        raise ValueError(f"unknown or excluded subjects: {sorted(requested - available)}")
    train_names = available - requested
    if not train_names or not validation_names:
        raise ValueError("train and validation must each contain a subject")
    keep = ~np.isin(subjects, list(excluded))
    return IndexSplit(
        train=np.flatnonzero(keep & np.isin(subjects, list(train_names))),
        validation=np.flatnonzero(keep & np.isin(subjects, list(validation_names))),
        test=np.flatnonzero(keep & np.isin(subjects, list(test_names))) if test_names else None,
        name="subject_partition",
    )


def personalized_contiguous_splits(
    tasks: np.ndarray,
    trial_numbers: np.ndarray,
    n_splits: int = 5,
) -> Iterator[IndexSplit]:
    """Hold out one contiguous chunk from every task for a personalized model."""

    tasks = np.asarray(tasks).astype(str).reshape(-1)
    trial_numbers = np.asarray(trial_numbers, dtype=int).reshape(-1)
    if len(tasks) != len(trial_numbers):
        raise ValueError("tasks and trial_numbers must have equal length")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    assignment = np.empty(len(tasks), dtype=int)
    for task in np.unique(tasks):
        index = np.flatnonzero(tasks == task)
        if len(index) < n_splits:
            raise ValueError(f"task {task} has fewer trials than n_splits")
        ordered = index[np.argsort(trial_numbers[index])]
        for fold, chunk in enumerate(np.array_split(ordered, n_splits)):
            assignment[chunk] = fold
    for fold in range(n_splits):
        validation = np.flatnonzero(assignment == fold)
        train = np.flatnonzero(assignment != fold)
        yield IndexSplit(train=train, validation=validation, name=f"contiguous_fold_{fold}")


def _natural_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)
