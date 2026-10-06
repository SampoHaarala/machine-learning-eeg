"""Example orchestration owned by the caller - deliberately not a library API."""

from pathlib import Path

import numpy as np

from eeg_speed_toolkit import (
    classification_metrics,
    leave_one_subject_out_splits,
    load_config,
    load_dataset,
    make_eeg_feature_matrix,
    make_within_subject_task_labels,
    permutation_pvalue,
    tune_logistic_classifier,
    validate_dataset,
)


def my_training_function(dataset, config):
    """The application controls the loop, outputs, and which folds to run."""

    validate_dataset(dataset, config)
    x = make_eeg_feature_matrix(dataset, config["feature"]["name"])
    target = make_within_subject_task_labels(
        dataset.duration, dataset.tasks, dataset.subjects
    )

    predictions = np.full(dataset.n_samples, -1, dtype=int)
    evaluated = np.zeros(dataset.n_samples, dtype=bool)
    fitted_models = {}
    for split in leave_one_subject_out_splits(
        dataset.subjects, config["split"]["exclude_subjects"]
    ):
        selection = tune_logistic_classifier(
            x[split.train],
            target.labels[split.train],
            dataset.subjects[split.train],
            config,
        )
        predictions[split.test] = selection.estimator.predict(x[split.test])
        evaluated[split.test] = True
        fitted_models[split.name] = selection

    result = classification_metrics(target.labels[evaluated], predictions[evaluated])
    result["permutation_p"] = permutation_pvalue(
        target.labels[evaluated],
        predictions[evaluated],
        dataset.subjects[evaluated],
        dataset.tasks[evaluated],
        repeats=config["evaluation"]["permutations"],
        seed=config["seed"],
    )
    return fitted_models, predictions, result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "example.yaml")
    dataset_path = root.parent / "eeg-emg-prediction" / "data" / "features" / "eeg_speed_features_v2.npz"
    dataset = load_dataset(dataset_path)
    _, _, metrics = my_training_function(dataset, config)
    print(metrics)
