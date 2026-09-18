"""Synthetic software fixture, deliberately NOT physiological validation."""
from pathlib import Path
import json
import numpy as np
from src.config import load_config, dump_config


def generate(folder, subjects=14, trials=12, channels=62, muscles=13, seed=123):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fs, duration, onset = 1000., 4., 2.
    time = np.arange(round(fs*duration))/fs
    mixing = rng.normal(size=(channels, 4))
    muscle_map = rng.uniform(.2, 1.2, (4, muscles))
    for s in range(1, subjects+1):
        gain = rng.uniform(.9, 1.1)
        for j in range(1, trials+1):
            latent = rng.uniform(.3, 1.5, 4)
            oscillations = np.array([np.sin(2*np.pi*f*time+rng.uniform(0, 6)) for f in (3, 6, 10, 20)])
            eeg = (mixing@(latent[:, None]*oscillations))*4e-6 + rng.normal(0, 1e-6, (channels, len(time)))
            amplitude = (latent@muscle_map)*30e-6*gain
            emg = rng.normal(0, 1e-6, (muscles, len(time)))
            gate = (time >= onset) & (time < onset+1.)
            emg[:, gate] += rng.normal(size=(muscles, int(gate.sum())))*amplitude[:, None]
            names = [f'EEG{i+1:02d}' for i in range(channels)]
            meta = {'subject': str(s), 'trial_id': str(j), 'task': ['PG','UG','WH'][(j-1)%3],
                    'block': str((j-1)//4+1), 'fs': fs, 'unit': 'V', 'eeg_names': names,
                    'emg_names': [f'M{i+1:02d}' for i in range(muscles)],
                    'events': {'cue': 1.7, 'movement_onset': onset, 'touch': 2.6, 'lift': 3.},
                    'extra': {'synthetic': True}}
            file = path/f's{s:02d}_t{j:03d}.npz'
            # Encode before writing so ZIP central directory is complete before IO.
            from io import BytesIO
            import os
            buffer = BytesIO()
            np.savez_compressed(buffer, eeg=eeg, emg=emg, metadata=json.dumps(meta))
            payload = buffer.getvalue()
            with file.open('xb') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            if file.stat().st_size != len(payload):
                raise IOError(f'Incomplete synthetic file after write: {file}')
    return path


def demo_config(default='configs/default.yaml'):
    c = load_config(default)
    c['data'].update(path='data/datasets/synthetic', format='npz', pattern='*.npz', units_confirmed=True)
    c['alignment']['event'] = 'movement_onset'
    c['paths']['processed'] = 'data/processed/synthetic.npz'
    return c
