"""Training-only spatial covariance and filter-bank CSP tools."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt


def epoch_covariances(epochs: np.ndarray) -> np.ndarray:
    """Return trace-normalized covariance for every EEG epoch."""

    values = np.asarray(epochs, dtype=float)
    if values.ndim != 3:
        raise ValueError("epochs must have shape trials x channels x samples")
    centered = values - values.mean(axis=-1, keepdims=True)
    covariance = centered @ np.swapaxes(centered, 1, 2)
    covariance /= max(values.shape[-1] - 1, 1)
    trace = np.trace(covariance, axis1=1, axis2=2)
    return covariance / np.maximum(trace[:, None, None], 1e-30)


class FilterBankCSP:
    """Binary filter-bank CSP transformer fitted only on training epochs."""

    def __init__(
        self,
        sampling_rate_hz: float,
        bands=((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0)),
        filters_per_side: int = 2,
        regularization: float = 0.1,
    ):
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.bands = tuple(tuple(map(float, band)) for band in bands)
        self.filters_per_side = int(filters_per_side)
        self.regularization = float(regularization)
        self.filters_: list[np.ndarray] | None = None

    def fit(self, epochs: np.ndarray, labels: np.ndarray) -> "FilterBankCSP":
        values = np.asarray(epochs, dtype=float)
        labels = np.asarray(labels, dtype=int).reshape(-1)
        if values.ndim != 3 or len(values) != len(labels):
            raise ValueError("epochs and labels have incompatible shapes")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("FilterBankCSP requires both binary classes 0 and 1")
        if self.filters_per_side < 1 or not 0 <= self.regularization < 1:
            raise ValueError("invalid filters_per_side or regularization")
        filters = []
        for low, high in self.bands:
            filtered = self._filter(values, low, high)
            covariance = epoch_covariances(filtered)
            class_zero = covariance[labels == 0].mean(axis=0)
            class_one = covariance[labels == 1].mean(axis=0)
            dimension = class_zero.shape[0]
            identity = np.eye(dimension) / dimension
            weight = self.regularization
            class_zero = (1 - weight) * class_zero + weight * identity
            class_one = (1 - weight) * class_one + weight * identity
            _, vectors = eigh(class_one, class_zero + class_one)
            selected = np.r_[
                np.arange(self.filters_per_side),
                np.arange(dimension - self.filters_per_side, dimension),
            ]
            filters.append(vectors[:, selected])
        self.filters_ = filters
        return self

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("fit FilterBankCSP before transform")
        values = np.asarray(epochs, dtype=float)
        if values.ndim != 3:
            raise ValueError("epochs must have shape trials x channels x samples")
        features = []
        for (low, high), filters in zip(self.bands, self.filters_):
            covariance = epoch_covariances(self._filter(values, low, high))
            variance = np.einsum("cf,ncd,df->nf", filters, covariance, filters)
            variance /= np.maximum(variance.sum(axis=1, keepdims=True), 1e-30)
            features.append(np.log(np.maximum(variance, 1e-30)))
        return np.column_stack(features)

    def fit_transform(self, epochs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(epochs, labels).transform(epochs)

    def _filter(self, epochs, low, high):
        if not 0 < low < high < self.sampling_rate_hz / 2:
            raise ValueError(f"invalid CSP band {(low, high)}")
        sos = butter(
            4, [low, high], btype="bandpass", fs=self.sampling_rate_hz, output="sos"
        )
        return sosfiltfilt(sos, epochs, axis=-1)
