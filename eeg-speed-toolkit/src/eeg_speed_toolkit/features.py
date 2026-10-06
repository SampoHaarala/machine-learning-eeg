"""Deterministic feature assembly with no model fitting side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dataset import EEGSpeedDataset


MOTOR_GRID = (
    ("FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"),
    ("C5", "C3", "C1", "Cz", "C2", "C4", "C6"),
    ("CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"),
)


def flatten_epochs(epochs: np.ndarray) -> np.ndarray:
    """Convert trials x channels x samples to trials x flat features."""

    values = np.asarray(epochs, dtype=float)
    if values.ndim != 3:
        raise ValueError("epochs must have shape trials x channels x samples")
    return values.reshape(len(values), -1)


def central_surface_laplacian(
    epochs: np.ndarray,
    channel_names: Iterable[str] | None = None,
) -> np.ndarray:
    """Subtract local FC/CP/lateral neighbors from each central-row channel."""

    values = np.asarray(epochs, dtype=float)
    if values.ndim != 3:
        raise ValueError("epochs must have shape trials x channels x samples")
    expected = tuple(name for row in MOTOR_GRID for name in row)
    names = expected if channel_names is None else tuple(channel_names)
    if set(names) != set(expected) or len(names) != len(expected):
        raise ValueError("channel_names must contain the documented 21-channel motor grid")
    lookup = {name: index for index, name in enumerate(names)}
    rows = [[values[:, lookup[name]] for name in row] for row in MOTOR_GRID]
    output = []
    for column in range(7):
        neighbors = [rows[0][column], rows[2][column]]
        if column > 0:
            neighbors.append(rows[1][column - 1])
        if column < 6:
            neighbors.append(rows[1][column + 1])
        output.append(rows[1][column] - np.mean(neighbors, axis=0))
    return np.stack(output, axis=1)


def make_eeg_feature_matrix(dataset: EEGSpeedDataset, name: str) -> np.ndarray:
    """Select one EEG representation and return a finite two-dimensional matrix."""

    if name == "mrcp_full":
        result = flatten_epochs(dataset.mrcp_full)
    elif name == "mrcp_pre":
        result = flatten_epochs(dataset.mrcp_pre)
    elif name == "mrcp_laplacian":
        channel_names = None
        if dataset.manifest and dataset.manifest.get("motor_eeg_names"):
            channel_names = dataset.manifest["motor_eeg_names"]
        result = flatten_epochs(central_surface_laplacian(dataset.mrcp_full, channel_names))
    elif name == "mrcp_frontal":
        if dataset.mrcp_frontal is None:
            raise ValueError("dataset does not contain mrcp_frontal")
        result = flatten_epochs(dataset.mrcp_frontal)
    elif name == "mrcp_posterior":
        if dataset.mrcp_posterior is None:
            raise ValueError("dataset does not contain mrcp_posterior")
        result = flatten_epochs(dataset.mrcp_posterior)
    elif name == "erd":
        result = np.asarray(dataset.erd, dtype=float)
    else:
        raise ValueError(f"unknown feature name: {name}")
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("selected feature matrix must be finite and two-dimensional")
    return result


@dataclass
class ContextEncoder:
    """Small explicit task one-hot encoder for optional non-EEG controls."""

    categories_: tuple[str, ...] | None = None

    def fit(self, tasks: np.ndarray) -> "ContextEncoder":
        values = np.asarray(tasks).astype(str).reshape(-1)
        self.categories_ = tuple(sorted(set(values.tolist())))
        if not self.categories_:
            raise ValueError("tasks cannot be empty")
        return self

    def transform(self, tasks: np.ndarray, reaction_time: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise RuntimeError("fit ContextEncoder before transform")
        tasks = np.asarray(tasks).astype(str).reshape(-1)
        reaction_time = np.asarray(reaction_time, dtype=float).reshape(-1)
        if len(tasks) != len(reaction_time):
            raise ValueError("tasks and reaction_time must have equal length")
        unseen = sorted(set(tasks.tolist()) - set(self.categories_))
        if unseen:
            raise ValueError(f"unseen tasks: {unseen}")
        return np.column_stack([
            *[(tasks == category).astype(float) for category in self.categories_],
            reaction_time,
        ])

    def fit_transform(self, tasks: np.ndarray, reaction_time: np.ndarray) -> np.ndarray:
        return self.fit(tasks).transform(tasks, reaction_time)


def append_context(eeg_features: np.ndarray, context_features: np.ndarray) -> np.ndarray:
    """Concatenate already prepared EEG and context matrices by trial."""

    eeg = np.asarray(eeg_features, dtype=float)
    context = np.asarray(context_features, dtype=float)
    if eeg.ndim != 2 or context.ndim != 2 or len(eeg) != len(context):
        raise ValueError("both matrices must be two-dimensional with equal trial count")
    return np.column_stack([eeg, context])
