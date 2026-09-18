from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def metrics(y, pred, names):
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError('Invalid prediction shape/values')
    rows = []
    for i, name in enumerate(names):
        error = pred[:, i]-y[:, i]
        ss = np.sum((y[:, i]-np.mean(y[:, i]))**2)
        r2 = 1-np.sum(error**2)/ss if ss > 1e-24 and len(y) > 1 else None
        corr = float(np.corrcoef(y[:, i], pred[:, i])[0, 1]) if (
            len(y) > 1 and np.std(y[:, i]) > 1e-12 and np.std(pred[:, i]) > 1e-12) else None
        rows.append({'muscle': name, 'MAE': float(np.mean(np.abs(error))),
                     'RMSE': float(np.sqrt(np.mean(error**2))),
                     'R2': float(r2) if r2 is not None else None, 'Pearson_r': corr})
    aggregate = {}
    for key in ('MAE', 'RMSE', 'R2', 'Pearson_r'):
        valid = [row[key] for row in rows if row[key] is not None]
        aggregate[key] = float(np.mean(valid)) if valid else None
        aggregate[key+'_defined_muscles'] = len(valid)
    aggregate['global_RMSE'] = float(np.sqrt(np.mean((y-pred)**2)))
    return {'per_muscle': rows, 'macro': aggregate, 'n_samples': len(y)}


def save_plots(y, p, names, folder, history=None, importance=None, feature_names=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cols, rows = 4, int(np.ceil(len(names)/4))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= len(names):
            ax.axis('off'); continue
        ax.scatter(y[:, i], p[:, i], s=9, alpha=.5)
        lo, hi = min(y[:, i].min(), p[:, i].min()), max(y[:, i].max(), p[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=.7)
        ax.set(title=names[i], xlabel='Measured', ylabel='Predicted')
    fig.tight_layout(); fig.savefig(folder/'predicted_vs_true.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.boxplot(p-y, tick_labels=names, showfliers=False)
    ax.tick_params(axis='x', rotation=60); ax.set_ylabel('Prediction error, normalized target')
    fig.tight_layout(); fig.savefig(folder/'errors.png', dpi=140); plt.close(fig)
    table = pd.DataFrame(metrics(y, p, names)['per_muscle']).set_index('muscle')
    fig, ax = plt.subplots(figsize=(11, 4))
    table[['R2', 'Pearson_r']].plot.bar(ax=ax)
    ax.axhline(0, color='black', lw=.6)
    fig.tight_layout(); fig.savefig(folder/'per_muscle.png', dpi=140); plt.close(fig)
    if history and 'epoch' in history[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Training')
        ax.plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], label='Validation')
        ax.set(xlabel='Epoch', ylabel='Data loss'); ax.legend()
        fig.tight_layout(); fig.savefig(folder/'learning_curve.png', dpi=140); plt.close(fig)
    if importance is not None:
        idx = np.argsort(importance)[-20:]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh([feature_names[i] for i in idx], importance[idx])
        ax.set_xlabel('Mean |standardized Ridge coefficient| / RF impurity importance')
        fig.tight_layout(); fig.savefig(folder/'feature_importance.png', dpi=140); plt.close(fig)


def compare_runs(paths, output):
    records, signatures = [], set()
    for path in map(Path, paths):
        m = json.loads((path/'metrics.json').read_text())
        signatures.add(m['comparison_signature'])
        records.append({'experiment': path.name, **m['validation']['macro']})
    if len(signatures) != 1:
        raise ValueError('Cannot compare different datasets/splits/target definitions')
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).set_index('experiment')
    frame.to_csv(out/'model_comparison.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    frame[['MAE', 'RMSE']].plot.bar(ax=ax)
    ax.set_title('Validation comparison — select model before opening test results')
    fig.tight_layout(); fig.savefig(out/'model_comparison.png', dpi=150); plt.close(fig)
    return frame
