"""Array-level EEG operations for callers building their own datasets."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def average_reference(eeg: np.ndarray) -> np.ndarray:
    """Return channel-average-referenced EEG without modifying the input."""

    values = np.asarray(eeg, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("eeg must be channels x samples or trials x channels x samples")
    channel_axis = 0 if values.ndim == 2 else 1
    return values - values.mean(axis=channel_axis, keepdims=True)


def extract_epoch(
    signal: np.ndarray,
    event_time_s: float,
    window_s: tuple[float, float],
    sampling_rate_hz: float,
) -> np.ndarray:
    """Extract a channels x samples epoch using an event time in seconds."""

    values = np.asarray(signal, dtype=float)
    if values.ndim != 2:
        raise ValueError("signal must have shape channels x samples")
    start, stop = window_s
    if not start < stop or sampling_rate_hz <= 0:
        raise ValueError("invalid window or sampling rate")
    first = int(round((event_time_s + start) * sampling_rate_hz))
    last = int(round((event_time_s + stop) * sampling_rate_hz))
    if first < 0 or last > values.shape[-1] or first >= last:
        raise ValueError("epoch is outside the signal")
    return values[:, first:last].copy()


def lowpass_filter(eeg: np.ndarray, sampling_rate_hz: float, cutoff_hz: float = 5.0) -> np.ndarray:
    """Apply the fourth-order zero-phase low-pass used for MRCP extraction."""

    values = np.asarray(eeg, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("eeg must be channels x samples or trials x channels x samples")
    if not 0 < cutoff_hz < sampling_rate_hz / 2:
        raise ValueError("cutoff must lie between zero and Nyquist")
    sos = butter(4, cutoff_hz, btype="lowpass", fs=sampling_rate_hz, output="sos")
    return sosfiltfilt(sos, values, axis=-1)


def baseline_correct(epoch: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Subtract each channel's baseline mean from an epoch."""

    epoch = np.asarray(epoch, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if epoch.ndim != 2 or baseline.ndim != 2 or epoch.shape[0] != baseline.shape[0]:
        raise ValueError("epoch and baseline must be channels x samples with matching channels")
    return epoch - baseline.mean(axis=1, keepdims=True)


def resample_epoch(epoch: np.ndarray, original_hz: int, target_hz: int) -> np.ndarray:
    """Polyphase-resample the final sample axis using integer rates."""

    values = np.asarray(epoch, dtype=float)
    if values.ndim not in (2, 3) or original_hz < 1 or target_hz < 1:
        raise ValueError("invalid epoch shape or sampling rate")
    return resample_poly(values, target_hz, original_hz, axis=-1)
