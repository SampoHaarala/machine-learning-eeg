"""Dataset container, loading, subsetting, and structural validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EEGSpeedDataset:
    """Arrays used by the reach-speed toolkit; every first axis is a trial."""

    mrcp_full: np.ndarray
    mrcp_pre: np.ndarray
    erd: np.ndarray
    duration: np.ndarray
    reaction_time: np.ndarray
    tasks: np.ndarray
    subjects: np.ndarray
    trials: np.ndarray
    mrcp_frontal: np.ndarray | None = None
    mrcp_posterior: np.ndarray | None = None
    manifest: Mapping[str, Any] | None = None

    @property
    def n_samples(self) -> int:
        return int(len(self.duration))


@dataclass(frozen=True)
class DatasetSummary:
    n_samples: int
    subjects: tuple[str, ...]
    tasks: tuple[str, ...]
    feature_shapes: Mapping[str, tuple[int, ...]]


def load_dataset(path: str | Path) -> EEGSpeedDataset:
    """Load the standalone NPZ contract without fitting or splitting anything."""

    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"])) if "manifest" in data else {}
        optional = {
            name: np.asarray(data[name]) if name in data else None
            for name in ("mrcp_frontal", "mrcp_posterior")
        }
        dataset = EEGSpeedDataset(
            mrcp_full=np.asarray(data["mrcp_full"]),
            mrcp_pre=np.asarray(data["mrcp_pre"]),
            erd=np.asarray(data["erd"]),
            duration=np.asarray(data["duration"]).reshape(-1),
            reaction_time=np.asarray(data["reaction_time"]).reshape(-1),
            tasks=np.asarray(data["tasks"]).astype(str),
            subjects=np.asarray(data["subjects"]).astype(str),
            trials=np.asarray(data["trials"]).astype(str),
            manifest=manifest,
            **optional,
        )
    return dataset


def subset_dataset(dataset: EEGSpeedDataset, indices: np.ndarray) -> EEGSpeedDataset:
    """Return a trial subset; no data are modified in place."""

    index = np.asarray(indices, dtype=int)
    values = {}
    for field in fields(dataset):
        value = getattr(dataset, field.name)
        if field.name == "manifest" or value is None:
            values[field.name] = value
        else:
            values[field.name] = value[index]
    return EEGSpeedDataset(**values)


def validate_dataset(
    dataset: EEGSpeedDataset,
    config: Mapping[str, Any] | None = None,
) -> DatasetSummary:
    """Fail early on inconsistent shapes, nonfinite values, or missing features."""

    if not isinstance(dataset, EEGSpeedDataset):
        raise TypeError("dataset must be EEGSpeedDataset")
    n = dataset.n_samples
    if n < 4:
        raise ValueError("dataset needs at least four trials")
    arrays = {
        "mrcp_full": dataset.mrcp_full,
        "mrcp_pre": dataset.mrcp_pre,
        "erd": dataset.erd,
        "duration": dataset.duration,
        "reaction_time": dataset.reaction_time,
        "tasks": dataset.tasks,
        "subjects": dataset.subjects,
        "trials": dataset.trials,
    }
    if dataset.mrcp_frontal is not None:
        arrays["mrcp_frontal"] = dataset.mrcp_frontal
    if dataset.mrcp_posterior is not None:
        arrays["mrcp_posterior"] = dataset.mrcp_posterior
    for name, value in arrays.items():
        if len(value) != n:
            raise ValueError(f"{name} has {len(value)} trials; expected {n}")
    for name in ("mrcp_full", "mrcp_pre"):
        value = arrays[name]
        if value.ndim != 3:
            raise ValueError(f"{name} must have shape trials x channels x samples")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    if dataset.erd.ndim != 2 or not np.isfinite(dataset.erd).all():
        raise ValueError("erd must be a finite trials x features matrix")
    if not np.isfinite(dataset.duration).all() or np.any(dataset.duration <= 0):
        raise ValueError("duration must contain positive finite seconds")
    if not np.isfinite(dataset.reaction_time).all() or np.any(dataset.reaction_time < 0):
        raise ValueError("reaction_time must contain nonnegative finite seconds")
    if len(set(dataset.trials.tolist())) != n:
        raise ValueError("trial identifiers must be unique")
    if any(not value for value in dataset.tasks) or any(not value for value in dataset.subjects):
        raise ValueError("task and subject identifiers cannot be empty")
    if config is not None:
        feature_name = config["feature"]["name"]
        if feature_name == "mrcp_frontal" and dataset.mrcp_frontal is None:
            raise ValueError("mrcp_frontal is required by config but absent")
        if feature_name == "mrcp_posterior" and dataset.mrcp_posterior is None:
            raise ValueError("mrcp_posterior is required by config but absent")
        if feature_name == "mrcp_laplacian" and dataset.mrcp_full.shape[1] != 21:
            raise ValueError("mrcp_laplacian requires the documented 21-channel motor grid")
    feature_shapes = {
        name: tuple(value.shape)
        for name, value in arrays.items()
        if name.startswith("mrcp") or name == "erd"
    }
    return DatasetSummary(
        n_samples=n,
        subjects=tuple(sorted(set(dataset.subjects.tolist()), key=_natural_key)),
        tasks=tuple(sorted(set(dataset.tasks.tolist()))),
        feature_shapes=feature_shapes,
    )


def _natural_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)
