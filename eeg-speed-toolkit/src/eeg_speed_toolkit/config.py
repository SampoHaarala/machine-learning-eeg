"""Configuration defaults, loading, and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


_DEFAULT = {
    "seed": 20261006,
    "feature": {"name": "mrcp_full", "include_context": False},
    "target": {"mode": "within_subject_task_median"},
    "split": {
        "mode": "leave_one_subject_out",
        "exclude_subjects": ["14"],
        "inner_folds": 5,
    },
    "model": {
        "kind": "logistic",
        "c_values": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "class_weight": "balanced",
        "max_iter": 4000,
        "n_jobs": -1,
    },
    "evaluation": {"permutations": 1999},
}

FEATURE_NAMES = {
    "mrcp_full", "mrcp_pre", "mrcp_laplacian",
    "mrcp_frontal", "mrcp_posterior", "erd",
}


def default_config() -> dict[str, Any]:
    """Return a new, mutable copy of the recommended configuration."""

    return deepcopy(_DEFAULT)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and return a validated, normalized configuration copy."""

    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return validate_config(value)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate public options without mutating the caller's mapping."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    normalized = deepcopy(dict(config))
    required = set(_DEFAULT)
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"config is missing sections: {sorted(missing)}")
    if not isinstance(normalized["seed"], int):
        raise ValueError("seed must be an integer")
    feature = normalized["feature"]
    if feature.get("name") not in FEATURE_NAMES:
        raise ValueError(f"feature.name must be one of {sorted(FEATURE_NAMES)}")
    if not isinstance(feature.get("include_context"), bool):
        raise ValueError("feature.include_context must be boolean")
    if normalized["target"].get("mode") not in {
        "within_subject_task_median", "training_task_median"
    }:
        raise ValueError("unsupported target.mode")
    split = normalized["split"]
    if split.get("mode") not in {
        "leave_one_subject_out", "subject_partition", "personalized_contiguous"
    }:
        raise ValueError("unsupported split.mode")
    if int(split.get("inner_folds", 0)) < 2:
        raise ValueError("split.inner_folds must be at least 2")
    split["exclude_subjects"] = [str(value) for value in split.get("exclude_subjects", [])]
    model = normalized["model"]
    if model.get("kind") != "logistic":
        raise ValueError("only model.kind='logistic' is currently registered")
    c_values = [float(value) for value in model.get("c_values", [])]
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("model.c_values must contain positive numbers")
    model["c_values"] = c_values
    if model.get("class_weight") not in (None, "balanced"):
        raise ValueError("model.class_weight must be null or 'balanced'")
    if int(model.get("max_iter", 0)) < 1:
        raise ValueError("model.max_iter must be positive")
    if int(normalized["evaluation"].get("permutations", -1)) < 0:
        raise ValueError("evaluation.permutations cannot be negative")
    return normalized
