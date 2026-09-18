from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import re
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src.config import validate, dump_config, fingerprint, processing_config
from src.data.loader import sha256
from src.training.split import make_split
from src.preprocessing.emg import TargetScaler
from src.models.ridge import make_model as ridge
from src.models.random_forest import make_model as forest
from src.models.mlp import MLPRegressor
from src.inference.predict import PredictionModel
from src.storage.model_io import save_model, environment
from src.training.evaluate import metrics, save_plots


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf8')


def output_guard(c, path):
    raw, out = Path(c['data']['path']).resolve(), Path(path).resolve()
    if out == raw or raw in out.parents:
        raise ValueError('Outputs cannot be written inside the raw-data directory')


def train(dataset, config, name=None, progress=None, should_stop=None):
    c = copy.deepcopy(validate(config))
    if dataset.manifest['processing_hash'] != fingerprint(processing_config(c)):
        raise ValueError('Processed data/config mismatch: rebuild dataset for this preprocessing configuration')
    if not np.isfinite(dataset.X).all() or not np.isfinite(dataset.y).all():
        raise ValueError('Nonfinite features/targets')
    name = name or c['experiment']['name'] or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', name):
        raise ValueError('Experiment name must be a safe filename, not a path')
    out = Path(c['paths']['runs'])/name
    output_guard(c, out)
    output_guard(c, Path(c['paths']['models'])/name)
    if (Path(c['paths']['models'])/name).exists():
        raise FileExistsError('Saved model name already exists')
    out.mkdir(parents=True, exist_ok=False)
    dump_config(c, out/'config.yaml')
    write_json(out/'status.json', {'status': 'running'})
    history = []
    try:
        split, report = make_split(dataset.metadata, c['split'], c['seed'])
        write_json(out/'split.json', report)
        write_json(out/'dataset_manifest.json', dataset.manifest)
        write_json(out/'environment.json', environment())
        code_root = Path(__file__).resolve().parents[1]
        write_json(out/'source_hashes.json', {str(p.relative_to(code_root)): sha256(p) for p in code_root.rglob('*.py')})
        import zipfile
        with zipfile.ZipFile(out/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as snapshot:
            for source in code_root.rglob('*.py'):
                snapshot.write(source, str(Path('src')/source.relative_to(code_root)))
        tr, va, te = [split[k] for k in ('train', 'validation', 'test')]
        subjects = np.array([m['subject'] for m in dataset.metadata])
        norm = c['emg']['normalization']
        if norm == 'subject_p95' and c['split']['mode'] == 'subject':
            raise ValueError('subject_p95 cannot calibrate unseen subjects. Use balanced_subject_p95 or robust.')
        ys = TargetScaler(norm).fit(dataset.y[tr], subjects[tr])
        yt = ys.transform(dataset.y[tr], subjects[tr])
        yv = ys.transform(dataset.y[va], subjects[va])
        X = dataset.X
        kind = c['model']['name']
        if kind == 'mlp':
            est = MLPRegressor(c['model']['mlp'], c['seed'])
            est.fit(X[tr], yt, X[va], yv, progress, should_stop)
            history = est.history_
            selected = {'best_epoch': est.best_epoch_, **c['model']['mlp']}
        else:
            if kind == 'ridge':
                candidates = [{'alpha': a} for a in c['model']['ridge']['alphas']]
            else:
                params = {k: v for k, v in c['model']['random_forest'].items() if k != 'search'}
                candidates = [{**params, **p} for p in ParameterGrid(c['model']['random_forest']['search'])]
            best, est = np.inf, None
            for i, params in enumerate(candidates):
                if should_stop and should_stop():
                    raise InterruptedError('Training cancelled')
                candidate = ridge(params['alpha'], c['seed']) if kind == 'ridge' else forest(params, c['seed'])
                candidate.fit(X[tr], yt)
                score = float(np.mean((candidate.predict(X[va])-yv)**2))
                row = {'candidate': i+1, 'parameters': params, 'validation_mse': score}
                history.append(row)
                if progress:
                    progress({'stage': 'train', 'model': kind, 'total': len(candidates), **row})
                if score < best:
                    best, est, selected = score, candidate, params
            if est is None:
                raise ValueError('No model candidates')
        if should_stop and should_stop():
            raise InterruptedError('Training cancelled')
        muscles = dataset.manifest['schema']['emg_names']
        # Model selection finished before evaluating any held-out test targets.
        result = {'model': kind, 'selected': selected, 'synthetic': dataset.manifest['synthetic'],
                  'comparison_signature': fingerprint({'manifest': dataset.manifest, 'split': report}),
                  'target_units': norm, 'selection_criterion': 'validation MSE'}
        pred_val, pred_test = est.predict(X[va]), est.predict(X[te])
        for key, idx, pred in [('validation', va, pred_val), ('test', te, pred_test)]:
            true = ys.transform(dataset.y[idx], subjects[idx])
            result[key] = metrics(true, pred, muscles)
            result[key+'_volts'] = metrics(dataset.y[idx], ys.inverse_transform(pred, subjects[idx]), muscles)
            result[key+'_by_subject'] = {s: metrics(true[subjects[idx] == s], pred[subjects[idx] == s], muscles)
                                         for s in sorted(set(subjects[idx]))}
            dummy = np.repeat(yt.mean(axis=0)[None], len(idx), axis=0)
            result[key+'_mean_baseline'] = metrics(true, dummy, muscles)
            # Known grip can account for prediction without EEG; fit means on training only.
            tasks = [m['task'] for m in dataset.metadata]
            task_mean = {task: yt[np.array([tasks[i] == task for i in tr])].mean(axis=0)
                         for task in set(tasks[i] for i in tr)}
            grip_pred = np.array([task_mean.get(tasks[i], yt.mean(axis=0)) for i in idx])
            result[key+'_grip_baseline'] = metrics(true, grip_pred, muscles)
            table = pd.DataFrame(result[key]['per_muscle'])
            table.to_csv(out/f'{key}_per_muscle.csv', index=False)
            np.savez_compressed(out/f'{key}_predictions.npz', measured=true, predicted=pred, indices=idx)
        write_json(out/'history.json', history)
        write_json(out/'features.json', {'names': dataset.manifest['features']})
        if c['experiment']['permutation_repeats']:
            # Negative control on validation only: selected Ridge hyperparameter, shuffled
            # training targets within subject AND task, preserving those confounds.
            if kind != 'ridge':
                raise ValueError('Permutation control is currently implemented for Ridge only')
            rng = np.random.default_rng(c['seed'])
            groups = {}
            for j, i in enumerate(tr):
                groups.setdefault((subjects[i], dataset.metadata[i]['task']), []).append(j)
            scores = []
            for _ in range(c['experiment']['permutation_repeats']):
                if should_stop and should_stop():
                    raise InterruptedError('Training cancelled')
                yp = yt.copy()
                for ids in groups.values():
                    yp[ids] = yt[rng.permutation(ids)]
                null = ridge(selected['alpha'], c['seed']).fit(X[tr], yp)
                scores.append(float(np.mean((null.predict(X[va])-yv)**2)))
            result['validation_permutation_control_mse'] = scores
            result['permutation_note'] = 'Diagnostic with fixed selected hyperparameter; not a calibrated hypothesis test'
        importance = None
        if kind != 'mlp':
            model = est.named_steps['regressor']
            importance = np.mean(np.abs(model.coef_), axis=0) if kind == 'ridge' else model.feature_importances_
        save_plots(ys.transform(dataset.y[te], subjects[te]), pred_test, muscles, out/'plots',
                   history, importance, dataset.manifest['features'])
        bundle = PredictionModel(est, ys, c, dataset.manifest['schema'], dataset.manifest['features'])
        save_model(bundle, out/'model')
        save_model(bundle, Path(c['paths']['models'])/name)
        write_json(out/'metrics.json', result)
        write_json(out/'status.json', {'status': 'complete'})
        return out, result, bundle
    except BaseException as exc:
        write_json(out/'status.json', {'status': 'cancelled' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
                                      'error': str(exc)})
        (out/'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        if history:
            write_json(out/'history.json', history)
        raise
