"""Cue is not movement onset. Derived EMG onset is an offline proxy only."""
import csv
import numpy as np
from scipy.signal import butter, sosfilt, lfilter
from src.preprocessing.windows import indices


class EventError(ValueError):
    pass


def read_annotations(path):
    if not path:
        return {}
    with open(path, newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        key = (row['subject'], row['trial_id'])
        if key in result:
            raise ValueError('Duplicate onset annotation')
        result[key] = float(row['onset_s'])
    return result


def align(trial, cfg, annotations=None):
    event = cfg['event']
    if event == 'annotated_onset':
        value = (annotations or {}).get((trial.subject, trial.trial_id))
        if value is None or not np.isfinite(value):
            raise EventError('missing_onset_annotation')
        return value, {'method': 'annotated_onset'}
    if event != 'emg_onset':
        if event not in trial.events:
            raise EventError('missing_event_' + event)
        return trial.events[event], {'method': event, 'movement_onset_verified': False}
    if 'cue' not in trial.events:
        raise EventError('missing_cue')
    c, fs = cfg['onset'], trial.fs
    lo, hi = c['bandpass']
    if not 0 < lo < hi < fs/2:
        raise ValueError('Onset detector bandpass exceeds Nyquist')
    selected = c['muscles'] or trial.emg_names
    ids = [trial.emg_names.index(n) for n in selected]
    filtered = sosfilt(butter(4, [lo, hi], fs=fs, btype='bandpass', output='sos'), trial.emg[ids], axis=-1)
    width = max(1, round(c['rms_ms'] * fs/1000))
    envelope = np.sqrt(lfilter(np.ones(width)/width, [1], filtered ** 2, axis=-1))
    a, b = indices(trial.events['cue'], c['baseline'], fs, envelope.shape[1])
    baseline = envelope[:, a:b]
    median = np.median(baseline, axis=1)
    mad = 1.4826 * np.median(np.abs(baseline - median[:, None]), axis=1)
    threshold = np.maximum.reduce([median + c['mad_multiplier']*mad,
                                   median*c['min_ratio'], np.full_like(median, c['min_amplitude_v'])])
    a, b = indices(trial.events['cue'], c['search'], fs, envelope.shape[1])
    active = envelope > threshold[:, None]
    sustain = max(1, round(c['sustain_ms']*fs/1000))
    onset = None
    for i in range(a, b-sustain+1):
        if np.any(active[:, i:i+sustain].all(axis=1)):
            onset = i
            break
    if onset is None:
        raise EventError('no_emg_onset')
    return onset/fs, {'method': 'emg_onset', 'onset_sample': onset,
                      'muscles': selected, 'threshold_v': threshold.tolist(),
                      'movement_onset_verified': False, '_active': active}


def preactivation_check(info, sample_bounds, cfg):
    active = info.pop('_active', None)
    if active is None:
        return
    a, b = sample_bounds
    fraction = float(np.max(active[:, a:b].mean(axis=1)))
    info['pre_active_fraction'] = fraction
    if cfg['onset']['reject_pre_active'] and fraction > cfg['onset']['active_fraction']:
        raise EventError('pre_eeg_window_muscle_activity')
