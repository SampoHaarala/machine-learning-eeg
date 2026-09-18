"""Build immutable features and unnormalized targets. Split/scaling happen later."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from src.config import validate, processing_config, fingerprint
from src.data.loader import iter_trials, files, sha256
from src.data.events import align, read_annotations, preactivation_check, EventError
from src.preprocessing.windows import extract, indices
from src.preprocessing.eeg import preprocess_eeg, ArtifactError
from src.preprocessing.emg import target
from src.features.eeg_features import extract_features


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    metadata: list
    manifest: dict

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reject accidental overwrite. New config => new dataset filename.
        with path.open('xb') as f:
            np.savez_compressed(f, X=self.X, y=self.y,
                                metadata=json.dumps(self.metadata), manifest=json.dumps(self.manifest))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(d['X'], d['y'], json.loads(str(d['metadata'])), json.loads(str(d['manifest'])))


def sample(trial, c, annotations=None):
    onset, event_info = align(trial, c['alignment'], annotations)
    if not 0 <= onset < trial.eeg.shape[1]/trial.fs:
        raise EventError('onset_out_of_bounds')
    if c['alignment']['event'] == 'emg_onset':
        if c['eeg']['window'][1] > -c['alignment']['onset']['guard_ms']/1000 + 1e-9:
            raise ValueError('EEG endpoint violates configured onset guard')
    eeg, eb = extract(trial.eeg, onset, c['eeg']['window'], trial.fs)
    emg, mb = extract(trial.emg, onset, c['emg']['window'], trial.fs)
    preactivation_check(event_info, eb, c['alignment'])
    clean, selected = preprocess_eeg(eeg, trial.fs, trial.eeg_names, c['eeg'])
    X, names = extract_features(clean, trial.fs, selected, c['features'])
    y = target(emg, trial.fs, c['emg'])
    meta = {'subject': trial.subject, 'trial_id': trial.trial_id, 'task': trial.task,
            'block': trial.block, 'event_time_s': onset, 'source': trial.source,
            'eeg_window_s': c['eeg']['window'], 'emg_window_s': c['emg']['window'],
            'eeg_samples': list(eb), 'emg_samples': list(mb), 'alignment': event_info,
            **trial.metadata}
    return X, y, meta, names


def build_dataset(c, progress=None):
    validate(c)
    rows, rejected = [], []
    annotations = read_annotations(c['alignment']['annotations_csv'])
    schema = None
    for i, trial in enumerate(iter_trials(c)):
        current = {'fs': trial.fs, 'eeg_names': trial.eeg_names, 'emg_names': trial.emg_names}
        if schema is None:
            schema = current
        elif current != schema:
            raise ValueError('Inconsistent sample rate/channel ordering; explicitly align before combining.')
        try:
            row = sample(trial, c, annotations)
            rows.append(row)
        except (EventError, ArtifactError) as e:
            rejected.append({'subject': trial.subject, 'trial_id': trial.trial_id, 'reason': str(e)})
        except ValueError as e:
            if str(e) != 'window_out_of_bounds':
                raise
            rejected.append({'subject': trial.subject, 'trial_id': trial.trial_id, 'reason': str(e)})
        if progress:
            progress({'stage': 'build', 'trials_seen': i+1, 'accepted': len(rows), 'rejected': len(rejected)})
    if not rows:
        raise ValueError(f'No usable samples. Rejections: {rejected[:20]}')
    manifest = {'version': 1, 'processing': processing_config(c),
                'processing_hash': fingerprint(processing_config(c)), 'schema': schema,
                'features': rows[0][3], 'rejected': rejected,
                'sources': [{'path': str(p.resolve()), 'sha256': sha256(p)} for p in files(c)],
                'synthetic': all(r[2].get('synthetic', False) for r in rows)}
    if c['alignment']['annotations_csv']:
        manifest['annotations_sha256'] = sha256(c['alignment']['annotations_csv'])
    return Dataset(np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]),
                   [r[2] for r in rows], manifest)
