from copy import deepcopy

import numpy as np
import pytest

from eeg_speed_toolkit import (
    EEGSpeedDataset,
    central_surface_laplacian,
    classification_metrics,
    default_config,
    leave_one_subject_out_splits,
    make_eeg_feature_matrix,
    make_training_task_median_labels,
    make_within_subject_task_labels,
    tune_logistic_classifier,
    validate_config,
    validate_dataset,
)


@pytest.fixture
def dataset():
    rng = np.random.default_rng(4)
    subjects = np.repeat(["1", "2", "3", "4"], 8)
    tasks = np.tile(np.repeat(["PG", "UG"], 4), 4)
    duration = np.tile([0.8, 0.9, 1.0, 1.1, 0.85, 0.95, 1.05, 1.15], 4)
    mrcp = rng.normal(size=(32, 21, 10))
    return EEGSpeedDataset(
        mrcp_full=mrcp,
        mrcp_pre=mrcp[:, :, :6],
        erd=rng.normal(size=(32, 12)),
        duration=duration,
        reaction_time=np.linspace(0.2, 0.8, 32),
        tasks=tasks,
        subjects=subjects,
        trials=np.asarray([f"{s}:{i}" for i, s in enumerate(subjects)]),
        mrcp_frontal=rng.normal(size=(32, 15, 10)),
        mrcp_posterior=rng.normal(size=(32, 17, 10)),
        manifest={"motor_eeg_names": [
            "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        ]},
    )


def test_validation_and_feature_selection(dataset):
    config = validate_config(default_config())
    summary = validate_dataset(dataset, config)
    assert summary.n_samples == 32
    assert make_eeg_feature_matrix(dataset, "mrcp_full").shape == (32, 210)
    assert make_eeg_feature_matrix(dataset, "mrcp_laplacian").shape == (32, 70)


def test_invalid_config_is_rejected():
    config = default_config()
    config["feature"]["name"] = "magic"
    with pytest.raises(ValueError, match="feature.name"):
        validate_config(config)


def test_within_subject_labels_are_task_balanced(dataset):
    target = make_within_subject_task_labels(
        dataset.duration, dataset.tasks, dataset.subjects
    )
    for subject in np.unique(dataset.subjects):
        for task in np.unique(dataset.tasks):
            mask = (dataset.subjects == subject) & (dataset.tasks == task)
            assert target.labels[mask].sum() == 2


def test_training_thresholds_do_not_use_test_outcomes(dataset):
    train = np.flatnonzero(dataset.subjects != "4")
    test = np.flatnonzero(dataset.subjects == "4")
    first = make_training_task_median_labels(dataset.duration, dataset.tasks, train, test)
    changed = dataset.duration.copy()
    changed[test] += 100
    second = make_training_task_median_labels(changed, dataset.tasks, train, test)
    assert first.thresholds == second.thresholds


def test_loso_has_no_subject_leakage(dataset):
    folds = list(leave_one_subject_out_splits(dataset.subjects, ["4"]))
    assert len(folds) == 3
    for fold in folds:
        assert not (set(dataset.subjects[fold.train]) & set(dataset.subjects[fold.test]))


def test_laplacian_removes_global_signal():
    epochs = np.ones((3, 21, 5))
    np.testing.assert_allclose(central_surface_laplacian(epochs), 0)


def test_tuning_fits_scaler_on_supplied_training_only(dataset):
    config = default_config()
    config["split"]["inner_folds"] = 3
    config["model"]["n_jobs"] = 1
    x = make_eeg_feature_matrix(dataset, "mrcp_laplacian")
    y = make_within_subject_task_labels(
        dataset.duration, dataset.tasks, dataset.subjects
    ).labels
    train = np.flatnonzero(dataset.subjects != "4")
    result = tune_logistic_classifier(x[train], y[train], dataset.subjects[train], config)
    np.testing.assert_allclose(
        result.estimator.named_steps["scale"].mean_, x[train].mean(axis=0)
    )
    assert result.best_c in config["model"]["c_values"]


def test_metrics_are_plain_serializable_values():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
