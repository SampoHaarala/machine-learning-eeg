import argparse
import json
from pathlib import Path
import numpy as np
from src.config import load_config, dump_config
from src.data.loader import inspect_file, iter_trials
from src.data.dataset_builder import build_dataset, Dataset
from src.data.synthetic import generate, demo_config
from src.data.visualize import plot_trial
from src.training.train import train, output_guard
from src.training.evaluate import compare_runs
from src.storage.model_io import load_model
from src.inference.predict import EEGSample


def main():
    p = argparse.ArgumentParser(description='EEG → future EMG regression')
    p.add_argument('--config', default='configs/default.yaml')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('inspect'); s.add_argument('file')
    sub.add_parser('build')
    s = sub.add_parser('train'); s.add_argument('--name'); s.add_argument('--model', choices=['ridge', 'random_forest', 'mlp'])
    s = sub.add_parser('demo'); s.add_argument('--subjects', type=int, default=14); s.add_argument('--trials', type=int, default=12)
    s = sub.add_parser('plot'); s.add_argument('--subject'); s.add_argument('--trial'); s.add_argument('--output', default='runs/inspection.png')
    s = sub.add_parser('predict'); s.add_argument('model'); s.add_argument('sample'); s.add_argument('--units', default='normalized', choices=['normalized','V']); s.add_argument('--subject')
    s = sub.add_parser('compare'); s.add_argument('runs', nargs='+'); s.add_argument('--output', default='runs/comparison')
    args = p.parse_args()
    if args.command == 'inspect':
        print(json.dumps(inspect_file(args.file), indent=2)); return
    if args.command == 'demo':
        c = demo_config(args.config)
        generate(c['data']['path'], args.subjects, args.trials)
        if args.subjects != 14:
            if args.subjects < 3:
                raise ValueError('Demo requires at least 3 subjects')
            c['split']['counts'] = [args.subjects-2, 1, 1]
        dump_config(c, 'configs/demo.yaml')
        ds = build_dataset(c)
        ds.save(c['paths']['processed'])
        path, _, _ = train(ds, c, name='synthetic_ridge', progress=print)
        print(f'SYNTHETIC SOFTWARE CHECK ONLY: {path}'); return
    if args.command == 'predict':
        model = load_model(args.model)
        with np.load(args.sample, allow_pickle=False) as f:
            sample = EEGSample(f['eeg'], float(f['fs']), f['channel_names'].tolist(), tuple(f['window']))
        print(json.dumps(dict(zip(model.muscle_names, model.predict(sample, units=args.units, subject=args.subject).tolist())), indent=2)); return
    if args.command == 'compare':
        print(compare_runs(args.runs, args.output)); return
    c = load_config(args.config)
    if args.command == 'build':
        output_guard(c, c['paths']['processed'])
        d = build_dataset(c, progress=print)
        d.save(c['paths']['processed']); print(f'Saved {len(d.X)} samples'); return
    if args.command == 'train':
        if args.model:
            c['model']['name'] = args.model
        out, _, _ = train(Dataset.load(c['paths']['processed']), c, args.name, progress=print)
        print(out); return
    if args.command == 'plot':
        if args.subject:
            c['data']['subjects'] = [args.subject]
        if args.trial:
            c['data']['trials'] = [args.trial]
        t = next(iter_trials(c))
        output_guard(c, args.output)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plot_trial(t, c).savefig(args.output, dpi=150)


if __name__ == '__main__':
    main()
