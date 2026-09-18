"""Versioned inference bundle. Load only locally trusted joblib artifacts."""
from pathlib import Path
import json
import platform
import joblib
import numpy as np
from importlib.metadata import version, PackageNotFoundError


def environment():
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'PyYAML', 'matplotlib', 'pandas', 'h5py', 'streamlit'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def save_model(model, path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    joblib.dump({'version': 1, 'model': model}, p/'bundle.joblib', compress=3)
    metadata = {'format_version': 1, 'config': model.config, 'schema': model.schema,
                'feature_names': model.feature_names, 'environment': environment(),
                'output': 'normalized EMG; request units=V for inverse scaling',
                'input': 'raw scalp EEG in volts, channels x samples; only saved pre-event window',
                'target_scaler_training_subjects': model.target_scaler.fitted_subjects_}
    (p/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    if model.config['model']['name'] == 'mlp':
        est = model.estimator
        np.savez(p/'weights.npz', **{f'W{i}': w for i, w in enumerate(est.weights_)},
                 **{f'b{i}': b for i, b in enumerate(est.biases_)})
    return p


def load_model(path):
    p = Path(path)
    if p.is_dir():
        p = p/'bundle.joblib'
    bundle = joblib.load(p)
    if bundle['version'] != 1:
        raise ValueError('Unsupported model format version')
    return bundle['model']
