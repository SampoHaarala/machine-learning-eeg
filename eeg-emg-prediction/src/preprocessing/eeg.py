import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos


class ArtifactError(ValueError):
    pass


def band_filter(x, fs, cfg):
    lo, hi = cfg['bandpass']
    if not 0 < lo < hi < fs / 2:
        raise ValueError(f'Band {lo}–{hi} Hz invalid at {fs} Hz; change config, no silent clipping.')
    sos = butter(cfg['order'], [lo, hi], btype='bandpass', fs=fs, output='sos')
    y = sosfiltfilt(sos, x, axis=-1)
    if cfg.get('notch'):
        if not 0 < cfg['notch_hz'] < fs / 2:
            raise ValueError('Notch frequency must be below Nyquist')
        b, a = iirnotch(cfg['notch_hz'], cfg['notch_q'], fs)
        y = sosfiltfilt(tf2sos(b, a), y, axis=-1)
    return y


def preprocess_eeg(epoch_v, fs, names, cfg):
    """Accept ONLY the permitted EEG window, never a complete post-onset trial.

    Filtering reflects this window at its boundaries. This prevents access to
    future measured samples but introduces edge effects, especially near 1 Hz.
    CAR uses all retained scalp channels before optional feature selection.
    """
    x = np.asarray(epoch_v, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(names) or not np.isfinite(x).all():
        raise ValueError('Expected finite channels x samples in volts')
    if len(names) < 2:
        raise ValueError('Common average reference requires at least two scalp channels')
    x = x - x.mean(axis=0, keepdims=True)
    if np.max(np.abs(x)) > cfg['artifact_uv'] * 1e-6:
        raise ArtifactError('eeg_amplitude_before_filter')
    x = band_filter(x, fs, cfg)
    if np.max(np.abs(x)) > cfg['artifact_uv'] * 1e-6:
        raise ArtifactError('eeg_amplitude_after_filter')
    selected = cfg['selected_channels'] or names
    if len(set(selected)) != len(selected) or not set(selected) <= set(names):
        raise ValueError('Unknown or duplicate selected EEG channel')
    return x[[names.index(n) for n in selected]], list(selected)
