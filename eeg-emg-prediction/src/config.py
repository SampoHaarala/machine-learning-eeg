"""One configuration contract for CLI, Python and GUI."""
from pathlib import Path
import hashlib
import json
import yaml


def load_config(path='configs/default.yaml'):
    with open(path, encoding='utf8') as f:
        c = yaml.safe_load(f)
    validate(c)
    return c


def validate(c):
    a, b = c['eeg']['window']
    x, y = c['emg']['window']
    if not a < b < 0 or not 0 <= x < y:
        raise ValueError('EEG must end strictly before onset; EMG must begin at/after onset.')
    for kind in ('eeg', 'emg'):
        lo, hi = c[kind]['bandpass']
        if not 0 < lo < hi or c[kind]['order'] < 1:
            raise ValueError(f'Invalid {kind} filter')
    if c['eeg']['reference'] != 'average':
        raise ValueError('Only average reference is implemented.')
    if c['eeg']['artifact_uv'] <= 0:
        raise ValueError('Artifact threshold must be positive')
    if c['features']['method'] != 'welch_bandpower':
        raise ValueError('Register a feature extractor before selecting that method.')
    if not 0 <= c['features']['overlap'] < 1 or c['features']['epsilon'] <= 0:
        raise ValueError('Invalid Welch overlap/epsilon')
    for lo, hi in c['features']['bands'].values():
        if not c['eeg']['bandpass'][0] <= lo < hi <= c['eeg']['bandpass'][1]:
            raise ValueError('Feature bands must lie within the EEG passband')
    if c['emg']['target'] not in ('rms', 'mean_envelope'):
        raise ValueError('Unknown EMG target')
    if c['emg']['normalization'] not in ('balanced_subject_p95', 'robust', 'none', 'subject_p95'):
        raise ValueError('Unknown target normalization')
    if c['model']['name'] not in ('ridge', 'random_forest', 'mlp'):
        raise ValueError('Unknown model')
    if c['split']['mode'] not in ('subject', 'within_subject'):
        raise ValueError('Unknown split mode')
    return c


def processing_config(c):
    return {k: c[k] for k in ('data', 'alignment', 'eeg', 'emg', 'features')}


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def dump_config(c, path):
    Path(path).write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf8')
