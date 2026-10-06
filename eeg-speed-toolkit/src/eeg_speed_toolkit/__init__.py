"""Composable EEG reach-speed modeling tools.

The package intentionally provides no ``train_experiment`` function. Callers
own orchestration, persistence, and the choice of which folds to execute.
"""

from .config import default_config, load_config, validate_config
from .dataset import EEGSpeedDataset, DatasetSummary, load_dataset, subset_dataset, validate_dataset
from .evaluation import classification_metrics, per_subject_metrics, permutation_pvalue
from .features import (
    ContextEncoder,
    append_context,
    central_surface_laplacian,
    flatten_epochs,
    make_eeg_feature_matrix,
)
from .models import ModelSelectionResult, fit_classifier, make_logistic_classifier, tune_logistic_classifier
from .preprocessing import average_reference, baseline_correct, extract_epoch, lowpass_filter, resample_epoch
from .spatial import FilterBankCSP, epoch_covariances
from .splitting import (
    IndexSplit,
    grouped_validation_splits,
    leave_one_subject_out_splits,
    personalized_contiguous_splits,
    subject_partition,
)
from .targets import FoldTargets, TargetLabels, make_training_task_median_labels, make_within_subject_task_labels

__all__ = [
    "ContextEncoder", "DatasetSummary", "EEGSpeedDataset", "FilterBankCSP",
    "FoldTargets", "IndexSplit", "ModelSelectionResult", "TargetLabels",
    "append_context", "average_reference", "baseline_correct",
    "central_surface_laplacian", "classification_metrics", "default_config",
    "epoch_covariances", "extract_epoch", "fit_classifier", "flatten_epochs",
    "grouped_validation_splits", "leave_one_subject_out_splits", "load_config",
    "load_dataset", "lowpass_filter", "make_eeg_feature_matrix",
    "make_logistic_classifier", "make_training_task_median_labels",
    "make_within_subject_task_labels", "per_subject_metrics",
    "permutation_pvalue", "personalized_contiguous_splits", "resample_epoch",
    "subject_partition", "subset_dataset", "tune_logistic_classifier",
    "validate_config", "validate_dataset",
]
