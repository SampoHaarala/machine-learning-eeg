import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.preprocessing.eeg import band_filter


def target(epoch_v, fs, cfg):
    filtered = band_filter(epoch_v, fs, cfg)
    rectified = np.abs(filtered)
    if cfg['target'] == 'rms':
        return np.sqrt(np.mean(rectified ** 2, axis=-1))
    if not 0 < cfg['envelope_hz'] < fs/2:
        raise ValueError('Invalid envelope cutoff')
    sos = butter(4, cfg['envelope_hz'], btype='lowpass', fs=fs, output='sos')
    envelope = sosfiltfilt(sos, rectified, axis=-1)
    return np.maximum(envelope, 0).mean(axis=-1)


class TargetScaler:
    """All estimates use training targets only; no test-subject fitting.

    balanced_subject_p95: median across training-subject per-muscle p95 values,
    applied identically to everyone, including unseen subjects. It does NOT
    eliminate unseen-subject electrode gain. subject_p95 is within-subject only.
    """
    def __init__(self, method='balanced_subject_p95'):
        self.method = method

    def fit(self, y, subjects):
        subjects = np.asarray(subjects)
        self.fitted_subjects_ = sorted(set(subjects.tolist()))
        self.center_ = np.zeros(y.shape[1])
        self.scales_ = {}
        if self.method in ('balanced_subject_p95', 'subject_p95'):
            q = {s: np.percentile(y[subjects == s], 95, axis=0) for s in self.fitted_subjects_}
            self.scale_ = np.median(list(q.values()), axis=0)
            if self.method == 'subject_p95':
                self.scales_ = {s: np.maximum(v, 1e-12) for s, v in q.items()}
        elif self.method == 'robust':
            self.center_ = np.median(y, axis=0)
            self.scale_ = np.percentile(y, 75, axis=0) - np.percentile(y, 25, axis=0)
        elif self.method == 'none':
            self.scale_ = np.ones(y.shape[1])
        else:
            raise ValueError('Unsupported target scaling')
        self.scale_ = np.maximum(self.scale_, 1e-12)
        return self

    def _scale(self, subjects):
        if not self.scales_:
            return self.scale_
        if subjects is None or any(s not in self.scales_ for s in subjects):
            raise ValueError('subject_p95 needs known subjects with training calibration; no test fallback.')
        return np.asarray([self.scales_[s] for s in subjects])

    def transform(self, y, subjects=None):
        return (y - self.center_) / self._scale(subjects)

    def inverse_transform(self, y, subjects=None):
        return y * self._scale(subjects) + self.center_
