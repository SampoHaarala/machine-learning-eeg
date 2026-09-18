"""Read-only adapters for the published Grasping Box schema and canonical NPZ.

Internally arrays are channels x samples in volts. Event times are seconds from
trial start. MATLAB frame indices are converted exactly once at the boundary.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat, whosmat


@dataclass
class Trial:
    subject: str
    trial_id: str
    task: str
    block: str
    eeg: np.ndarray
    emg: np.ndarray
    fs: float
    eeg_names: list
    emg_names: list
    events: dict
    source: str
    metadata: dict = field(default_factory=dict)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_mat(path):
    try:
        return loadmat(path, simplify_cells=True)
    except (NotImplementedError, ValueError) as exc:
        import h5py
        if not h5py.is_hdf5(path):
            raise exc
        # MATLAB v7.3: reverse stored dimension order; recursively resolve cells
        # and struct reference arrays without pickle or MATLAB execution.
        with h5py.File(path, 'r') as f:
            def decode(node):
                if isinstance(node, h5py.Group):
                    values = {k: decode(v) for k, v in node.items() if not k.startswith('#')}
                    cls = node.attrs.get('MATLAB_class', b'')
                    if cls == b'struct':
                        arrays = [v for v in values.values() if isinstance(v, np.ndarray) and v.dtype == object]
                        if arrays and all(a.size == arrays[0].size for a in arrays):
                            n = arrays[0].size
                            if n > 1:
                                return [{'{}'.format(k): v.reshape(-1)[i] if isinstance(v, np.ndarray) and v.dtype == object else v
                                         for k, v in values.items()} for i in range(n)]
                    return {k: v.item() if isinstance(v, np.ndarray) and v.size == 1 else v for k, v in values.items()}
                a = node[()]
                if h5py.check_dtype(ref=node.dtype) is not None:
                    result = np.empty(a.shape, dtype=object)
                    for idx in np.ndindex(a.shape):
                        result[idx] = decode(f[a[idx]]) if a[idx] else None
                    return result.T.squeeze()
                if node.attrs.get('MATLAB_class', b'') == b'char':
                    return ''.join(chr(int(v)) for v in a.T.flatten() if v)
                return a.T.squeeze()
            return {k: decode(v) for k, v in f.items() if not k.startswith('#')}


def entries(value):
    if isinstance(value, dict):
        return [value]
    return list(np.asarray(value, dtype=object).reshape(-1))


def labels(value):
    if isinstance(value, dict) and 'labels' in value:
        return [str(x) for x in np.atleast_1d(value['labels']).flatten()]
    return [str(x['labels']) if isinstance(x, dict) else str(x) for x in entries(value)]


def volts(a, unit):
    factors = {'V': 1., 'mV': 1e-3, 'uV': 1e-6}
    if unit not in factors:
        raise ValueError('Signal unit must be V, mV, or uV; never infer units from amplitude.')
    return np.asarray(a, dtype=float) * factors[unit]


def _validate_trial(t):
    for arr, names in ((t.eeg, t.eeg_names), (t.emg, t.emg_names)):
        if arr.ndim != 2 or arr.shape[0] != len(names) or len(set(names)) != len(names):
            raise ValueError(f'{t.source}: invalid channels x samples shape or duplicate labels')
        if not np.isfinite(arr).all():
            raise ValueError(f'{t.source}: nonfinite signal values')
    if t.eeg.shape[1] != t.emg.shape[1] or not np.isfinite(t.fs) or t.fs <= 0:
        raise ValueError('Adapter requires synchronized streams with equal sample count/rate.')
    return t


def load_grasping_box(path, c):
    if not c['data']['units_confirmed']:
        raise ValueError('Confirm source EEG/EMG units in config (data.units_confirmed) before building.')
    root = read_mat(path)['data']
    info, subject = root['info'], root['subject']
    fs = float(info['fs'])
    expected_fs = c['data']['sampling_rate']
    if expected_fs is not None and not np.isclose(fs, expected_fs):
        raise ValueError(f'Sampling rate {fs} differs from configured {expected_fs}')
    eeg_names, emg_names = labels(info['eeg_channels']), labels(info['emg_channels'])
    task_names = labels(info['tasks_names'])
    exclude = {x.upper() for x in c['data']['exclude_channels']}
    keep = [i for i, n in enumerate(eeg_names) if n.upper() not in exclude and 'EOG' not in n.upper()]
    eeg_names = [eeg_names[i] for i in keep]
    for actual, key in ((len(eeg_names), 'expected_eeg_channels'), (len(emg_names), 'expected_emg_channels')):
        expected = c['data'][key]
        if expected is not None and expected != actual:
            raise ValueError(f'{path}: {actual} channels; expected {expected}. Inspect labels, do not silently truncate.')
    base = c['data']['event_index_base']
    for i, tr in enumerate(entries(root['trials'])):
        if c['data']['require_complete'] and not bool(tr['event_complete_trial']):
            continue
        events = {}
        for key, name in [('LEDon_event_frame', 'cue'), ('touch_event_frame', 'touch'), ('lift_event_frame', 'lift')]:
            value = float(tr.get(key, np.nan))
            if np.isfinite(value) and value >= base:
                events[name] = (value - base) / fs
        task = int(tr['task']) - 1
        if task < 0 or task >= len(task_names):
            raise ValueError('Task index outside tasks_names')
        t = Trial(str(subject['id']), str(i + 1), task_names[task], str(tr['block']),
                  volts(tr['eeg'], c['data']['eeg_unit'])[keep], volts(tr['emg'], c['data']['emg_unit']),
                  fs, eeg_names, emg_names, events, str(Path(path).resolve()),
                  {'hand': str(subject.get('hand', 'unknown')), 'synthetic': False,
                   'event_complete': bool(tr.get('event_complete_trial', False))})
        yield _validate_trial(t)


def load_npz(path, c):
    """Canonical format: one independently recorded trial per NPZ; no pickle."""
    import json
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d['metadata']))
        t = Trial(**{k: meta[k] for k in ('subject', 'trial_id', 'task', 'block', 'fs', 'eeg_names', 'emg_names', 'events')},
                  eeg=d['eeg'], emg=d['emg'], source=str(Path(path).resolve()), metadata=meta.get('extra', {}))
    if meta.get('unit') != 'V':
        raise ValueError('Canonical NPZ must be in volts')
    return [_validate_trial(t)]


def files(c):
    return sorted(Path(c['data']['path']).glob(c['data']['pattern']))


def iter_trials(c):
    paths = files(c)
    if not paths:
        raise FileNotFoundError('No input files found; download data or run the synthetic demo.')
    adapter = {'grasping_box': load_grasping_box, 'npz': load_npz}[c['data']['format']]
    seen = set()
    for path in paths:
        for t in adapter(path, c):
            expected_fs = c['data']['sampling_rate']
            if expected_fs is not None and not np.isclose(t.fs, expected_fs):
                raise ValueError('Sampling rate differs from configured rate')
            for actual, key in ((len(t.eeg_names), 'expected_eeg_channels'), (len(t.emg_names), 'expected_emg_channels')):
                if c['data'][key] is not None and actual != c['data'][key]:
                    raise ValueError(f'{key}: expected {c["data"][key]}, got {actual}')
            if any(c['data'][key] and str(value) not in list(map(str, c['data'][key]))
                   for key, value in [('subjects', t.subject), ('trials', t.trial_id), ('tasks', t.task)]):
                continue
            uid = (t.subject, t.trial_id)
            if uid in seen:
                raise ValueError(f'Duplicate subject/trial identity: {uid}')
            seen.add(uid)
            yield t


def inspect_file(path):
    """Inspect actual structure without trusting guessed field names or units."""
    def tree(x, depth=0):
        if depth > 5:
            return '...'
        if isinstance(x, dict):
            return {k: tree(v, depth+1) for k, v in x.items() if not k.startswith('__')}
        if isinstance(x, (list, tuple)) or (isinstance(x, np.ndarray) and x.dtype == object):
            v = entries(x)
            return {'entries': len(v), 'first': tree(v[0], depth+1) if v else None}
        if isinstance(x, np.ndarray):
            return {'shape': list(x.shape), 'dtype': str(x.dtype)}
        return str(x)
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as d:
            return tree(dict(d))
    return tree(read_mat(path))
