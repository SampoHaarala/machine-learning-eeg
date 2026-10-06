# EEG Speed Toolkit

A standalone, composable Python toolkit for EEG-only classification of relative
reach speed from movement-related cortical potentials.

The package intentionally has no monolithic experiment function. Your code owns
the outer workflow, data persistence, and the decision about which folds to run.
Individual functions cover:

- configuration and dataset validation;
- calibrated or training-only target construction;
- subject-disjoint and personalized contiguous splits;
- MRCP, ERD, central-Laplacian, context, and filter-bank CSP features;
- construction, fitting, and group-aware tuning of logistic classifiers;
- balanced accuracy, macro F1, participant metrics, and permutation testing.

## Install for development

```powershell
python -m pip install -e ".[test]"
```

## Minimal assembly

```python
from eeg_speed_toolkit import (
    load_config, load_dataset, validate_dataset,
    make_eeg_feature_matrix, make_within_subject_task_labels,
    leave_one_subject_out_splits, tune_logistic_classifier,
)

config = load_config("config/example.yaml")
dataset = load_dataset("eeg_speed_features_v2.npz")
validate_dataset(dataset, config)

x = make_eeg_feature_matrix(dataset, config["feature"]["name"])
y = make_within_subject_task_labels(
    dataset.duration, dataset.tasks, dataset.subjects
).labels

fold = next(leave_one_subject_out_splits(dataset.subjects, ["14"]))
selection = tune_logistic_classifier(
    x[fold.train], y[fold.train], dataset.subjects[fold.train], config
)
prediction = selection.estimator.predict(x[fold.test])
```

See `examples/manual_subject_holdout.py` for a complete caller-owned loop and
`output/pdf/eeg_speed_toolkit_guide.pdf` for the function reference.
