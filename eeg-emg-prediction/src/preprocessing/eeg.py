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


def preprocess_eeg(
    eeg: np.ndarray,
    expected_channels: int = 62,
) -> np.ndarray:
    """
    Validate already preprocessed / ICA-cleaned EEG data.

    No filtering, ICA, rereferencing, or artifact removal is performed here.
    The dataset's preprocessed EEG is used directly for feature extraction.

    Parameters
    ----------
    eeg : np.ndarray
        EEG array with shape (n_channels, n_samples).

    expected_channels : int
        Expected number of EEG channels.

    Returns
    -------
    np.ndarray
        Validated EEG as float64 with shape
        (n_channels, n_samples).
    """

    eeg = np.asarray(eeg, dtype=np.float64)

    # Check dimensions
    if eeg.ndim != 2:
        raise ValueError(
            f"EEG must be 2-D (channels, samples), "
            f"got shape {eeg.shape}"
        )

    # Check channel count
    if eeg.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} EEG channels, "
            f"got {eeg.shape[0]}"
        )

    # Check invalid values
    if not np.all(np.isfinite(eeg)):
        raise ValueError(
            "EEG contains NaN or infinite values."
        )

    return eeg