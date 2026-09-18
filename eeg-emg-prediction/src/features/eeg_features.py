import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


def welch_bandpower(epoch, fs, names, cfg):
    nperseg = min(epoch.shape[-1], round(cfg['welch_seconds']*fs))
    if nperseg < 4:
        raise ValueError('Welch segment too short')
    nfft = cfg['nfft'] or nperseg
    if nfft < nperseg:
        raise ValueError('nfft cannot be smaller than segment length')
    f, p = welch(epoch, fs=fs, window='hann', nperseg=nperseg,
                 noverlap=int(nperseg*cfg['overlap']), nfft=nfft,
                 detrend='constant', scaling='density', axis=-1)
    values, feature_names = [], []
    for channel, psd in zip(names, p):
        for band, (lo, hi) in cfg['bands'].items():
            if not 0 <= lo < hi <= f[-1]:
                raise ValueError('Feature band outside PSD frequency range')
            # Interpolate boundaries then integrate density; no empty-bin bands.
            ff = np.r_[lo, f[(f > lo) & (f < hi)], hi]
            power = trapezoid(np.interp(ff, f, psd), ff)
            values.append(np.log(power + cfg['epsilon']))
            feature_names.append(f'{channel}:{band}')
    return np.asarray(values), feature_names


EXTRACTORS = {'welch_bandpower': welch_bandpower}


def extract_features(epoch, fs, names, cfg):
    return EXTRACTORS[cfg['method']](epoch, fs, names, cfg)
