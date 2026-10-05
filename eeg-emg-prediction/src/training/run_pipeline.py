from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import re
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.config import dump_config, fingerprint, processing_config, validate
from src.data.loader import sha256
from src.inference.predict import PredictionModel
from src.models.mlp import MLPRegressor
from src.models.random_forest import make_model as forest
from src.models.ridge import make_model as ridge
from src.preprocessing.emg import TargetScaler
from src.storage.model_io import environment, save_model
from src.training.evaluate import metrics, save_plots
from src.training.split import make_split


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')


def output_guard(c, path):
    raw, out = Path(c['data']['path']).resolve(), Path(path).resolve()
    if out == raw or raw in out.parents:
        raise ValueError('Outputs cannot be written inside the raw-data directory')


def resolve_experiment_name(config, name=None):
    candidate = name or config['experiment']['name'] or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', candidate):
        raise ValueError('Experiment name must be a safe filename, not a path')
    return candidate


def initialize_run_directory(config, name):
    out = Path(config['paths']['runs']) / name
    output_guard(config, out)
    output_guard(config, Path(config['paths']['models']) / name)
    if (Path(config['paths']['models']) / name).exists():
        raise FileExistsError('Saved model name already exists')
    out.mkdir(parents=True, exist_ok=False)
    dump_config(config, out / 'config.yaml')
    write_json(out / 'status.json', {'status': 'running'})
    return out


def validate_dataset_compatibility(dataset, config):
    if dataset.manifest['processing_hash'] != fingerprint(processing_config(config)):
        raise ValueError('Processed data/config mismatch: rebuild dataset for this preprocessing configuration')
    if not np.isfinite(dataset.X).all() or not np.isfinite(dataset.y).all():
        raise ValueError('Nonfinite features/targets')


def snapshot_project_sources(run_dir):
    code_root = Path(__file__).resolve().parents[1]
    write_json(run_dir / 'source_hashes.json', {str(p.relative_to(code_root)): sha256(p) for p in code_root.rglob('*.py')})
    import zipfile
    with zipfile.ZipFile(run_dir / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as snapshot:
        for source in code_root.rglob('*.py'):
            snapshot.write(source, str(Path('src') / source.relative_to(code_root)))


def create_experiment_split(dataset, config):
    split, report = make_split(dataset.metadata, config['split'], config['seed'])
    return split, report


def prepare_target_scaler(dataset, split, config):
    subjects = np.array([m['subject'] for m in dataset.metadata])
    norm = config['emg']['normalization']
    if norm == 'subject_p95' and config['split']['mode'] == 'subject':
        raise ValueError('subject_p95 cannot calibrate unseen subjects. Use balanced_subject_p95 or robust.')
    tr, va, te = [split[k] for k in ('train', 'validation', 'test')]
    scaler = TargetScaler(norm).fit(dataset.y[tr], subjects[tr])
    return scaler, subjects, tr, va, te


def train_model_for_split(dataset, split, config, scaler, subjects, tr, va, te, progress, should_stop):
    X = dataset.X
    kind = config['model']['name']
    history = []
    selected = None

    if kind == 'mlp':
        y_target = scaler.transform(dataset.y[tr], subjects[tr])
        y_valid = scaler.transform(dataset.y[va], subjects[va])
        est = MLPRegressor(config['model']['mlp'], config['seed'])
        est.fit(X[tr], y_target, X[va], y_valid, progress, should_stop)
        history = est.history_
        selected = {'best_epoch': est.best_epoch_, **config['model']['mlp']}
        return est, history, selected, y_target, y_valid

    y_target = scaler.transform(dataset.y[tr], subjects[tr])
    y_valid = scaler.transform(dataset.y[va], subjects[va])
    if kind == 'ridge':
        candidates = [{'alpha': a} for a in config['model']['ridge']['alphas']]
    else:
        params = {k: v for k, v in config['model']['random_forest'].items() if k != 'search'}
        candidates = [{**params, **p} for p in ParameterGrid(config['model']['random_forest']['search'])]

    best, est = np.inf, None
    for i, params in enumerate(candidates):
        if should_stop and should_stop():
            raise InterruptedError('Training cancelled')
        candidate = ridge(params['alpha'], config['seed']) if kind == 'ridge' else forest(params, config['seed'])
        candidate.fit(X[tr], y_target)
        score = float(np.mean((candidate.predict(X[va]) - y_valid) ** 2))
        row = {'candidate': i + 1, 'parameters': params, 'validation_mse': score}
        history.append(row)
        if progress:
            progress({'stage': 'train', 'model': kind, 'total': len(candidates), **row})
        if score < best:
            best, est, selected = score, candidate, params

    if est is None:
        raise ValueError('No model candidates')

    return est, history, selected, y_target, y_valid


def evaluate_predictions(dataset, scaler, split, config, est, subjects, tr, va, te, muscles):
    X = dataset.X
    y_target = scaler.transform(dataset.y[tr], subjects[tr])
    pred_val = est.predict(X[va])
    pred_test = est.predict(X[te])
    result = {
        'model': config['model']['name'],
        'selected': None,
        'synthetic': dataset.manifest['synthetic'],
        'comparison_signature': fingerprint({'manifest': dataset.manifest, 'split': split}),
        'target_units': config['emg']['normalization'],
        'selection_criterion': 'validation MSE',
    }

    for key, idx, pred in [('validation', va, pred_val), ('test', te, pred_test)]:
        true = scaler.transform(dataset.y[idx], subjects[idx])
        result[key] = metrics(true, pred, muscles)
        result[key + '_volts'] = metrics(dataset.y[idx], scaler.inverse_transform(pred, subjects[idx]), muscles)
        result[key + '_by_subject'] = {
            s: metrics(true[subjects[idx] == s], pred[subjects[idx] == s], muscles)
            for s in sorted(set(subjects[idx]))
        }
        dummy = np.repeat(y_target.mean(axis=0)[None], len(idx), axis=0)
        result[key + '_mean_baseline'] = metrics(true, dummy, muscles)
        tasks = [m['task'] for m in dataset.metadata]
        task_mean = {
            task: y_target[np.array([tasks[i] == task for i in tr])].mean(axis=0)
            for task in set(tasks[i] for i in tr)
        }
        grip_pred = np.array([task_mean.get(tasks[i], y_target.mean(axis=0)) for i in idx])
        result[key + '_grip_baseline'] = metrics(true, grip_pred, muscles)

    return result, pred_val, pred_test


def save_prediction_artifacts(run_dir, result, dataset, scaler, subjects, va, te, pred_val, pred_test):
    for key, idx, pred in [('validation', va, pred_val), ('test', te, pred_test)]:
        true = scaler.transform(dataset.y[idx], subjects[idx])
        table = pd.DataFrame(result[key]['per_muscle'])
        table.to_csv(run_dir / f'{key}_per_muscle.csv', index=False)
        np.savez_compressed(run_dir / f'{key}_predictions.npz', measured=true, predicted=pred, indices=idx)


def compute_permutation_control(dataset, config, subjects, tr, va, selected, y_target, y_valid, should_stop):
    if config['experiment']['permutation_repeats'] == 0:
        return None
    if config['model']['name'] != 'ridge':
        raise ValueError('Permutation control is currently implemented for Ridge only')
    rng = np.random.default_rng(config['seed'])
    groups = {}
    for j, i in enumerate(tr):
        groups.setdefault((subjects[i], dataset.metadata[i]['task']), []).append(j)
    scores = []
    for _ in range(config['experiment']['permutation_repeats']):
        if should_stop and should_stop():
            raise InterruptedError('Training cancelled')
        yp = y_target.copy()
        for ids in groups.values():
            yp[ids] = y_target[rng.permutation(ids)]
        null = ridge(selected['alpha'], config['seed']).fit(dataset.X[tr], yp)
        scores.append(float(np.mean((null.predict(dataset.X[va]) - y_valid) ** 2)))
    return {
        'validation_permutation_control_mse': scores,
        'permutation_note': 'Diagnostic with fixed selected hyperparameter; not a calibrated hypothesis test',
    }


def finalize_run_artifacts(run_dir, dataset, config, result, scaler, est, history, pred_test, selected, subjects, te):
    write_json(run_dir / 'history.json', history)
    write_json(run_dir / 'features.json', {'names': dataset.manifest['features']})

    importance = None
    if config['model']['name'] != 'mlp':
        model = est.named_steps['regressor']
        importance = np.mean(np.abs(model.coef_), axis=0) if config['model']['name'] == 'ridge' else model.feature_importances_
    save_plots(
        scaler.transform(dataset.y[te], subjects[te]),
        pred_test,
        dataset.manifest['schema']['emg_names'],
        run_dir / 'plots',
        history,
        importance,
        dataset.manifest['features'],
    )

    bundle = PredictionModel(est, scaler, config, dataset.manifest['schema'], dataset.manifest['features'])
    save_model(bundle, run_dir / 'model')
    save_model(bundle, Path(config['paths']['models']) / Path(run_dir).name)
    result['selected'] = selected
    write_json(run_dir / 'metrics.json', result)
    write_json(run_dir / 'status.json', {'status': 'complete'})
    return bundle


def write_failure_status(run_dir, exc, history):
    write_json(run_dir / 'status.json', {
        'status': 'cancelled' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
        'error': str(exc),
    })
    (run_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
    if history:
        write_json(run_dir / 'history.json', history)


def train_experiment(dataset, config, name=None, progress=None, should_stop=None):
    c = copy.deepcopy(validate(config))
    validate_dataset_compatibility(dataset, c)
    run_dir = initialize_run_directory(c, resolve_experiment_name(c, name))
    history = []

    try:
        snapshot_project_sources(run_dir)
        split, report = create_experiment_split(dataset, c)
        write_json(run_dir / 'split.json', report)
        write_json(run_dir / 'dataset_manifest.json', dataset.manifest)
        write_json(run_dir / 'environment.json', environment())

        scaler, subjects, tr, va, te = prepare_target_scaler(dataset, split, c)
        est, history, selected, y_target, y_valid = train_model_for_split(
            dataset,
            split,
            c,
            scaler,
            subjects,
            tr,
            va,
            te,
            progress,
            should_stop,
        )
        if should_stop and should_stop():
            raise InterruptedError('Training cancelled')

        muscles = dataset.manifest['schema']['emg_names']
        result, pred_val, pred_test = evaluate_predictions(
            dataset,
            scaler,
            report,
            c,
            est,
            subjects,
            tr,
            va,
            te,
            muscles,
        )
        result['selected'] = selected
        save_prediction_artifacts(run_dir, result, dataset, scaler, subjects, va, te, pred_val, pred_test)
        write_json(run_dir / 'features.json', {'names': dataset.manifest['features']})

        permutation = compute_permutation_control(
            dataset,
            c,
            subjects,
            tr,
            va,
            selected,
            y_target,
            y_valid,
            should_stop,
        )
        if permutation:
            result.update(permutation)

        bundle = finalize_run_artifacts(run_dir, dataset, c, result, scaler, est, history, pred_test, selected, subjects, te)
        return run_dir, result, bundle
    except BaseException as exc:
        write_failure_status(run_dir, exc, history)
        raise
