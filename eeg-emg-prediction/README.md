# EEG → future EMG regression

A modular Python/Streamlit project for testing whether pre-event EEG predicts
subsequent multi-muscle EMG amplitude. Includes Ridge, Random Forest and a small
NumPy MLP, subject-separated evaluation, complete model bundles and a synthetic demo.

**Status:** software tested on synthetic signals and MATLAB schema fixtures.
Actual Grasping Box files were unavailable in the build environment. Real-data
scientific validation is still required; no included score answers the research
question. Read [PROGRESS.md](PROGRESS.md) for checkpoint and verification status.

## Install and run

Python 3.10+; tested on Python 3.12. From this directory:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m streamlit run app/app.py
```

`requirements-lock.txt` records tested exact library versions. Use it instead
of requirements.txt to reproduce this environment's package versions. No CUDA,
PyTorch or MATLAB installation is required. The neural network uses NumPy.
Paths are relative to the working directory, so run commands from the project root.

## First run without downloading participant data

```bash
python -m src.cli demo --trials 6
python -m src.cli --config configs/demo.yaml train --model random_forest --name synthetic_forest
python -m src.cli --config configs/demo.yaml train --model mlp --name synthetic_mlp
python -m src.cli compare runs/synthetic_ridge runs/synthetic_forest runs/synthetic_mlp
```

The demo generates 14 synthetic subjects, 62 EEG channels and 13 EMG channels,
builds 248-feature samples, splits subjects 10/2/2 and trains Ridge. Later commands
reuse exactly the same processed data/split for model comparison. It deliberately
embeds a predictable relationship and is **only a software check**. The demo can
use several hundred MB of disk space. Existing raw demo files, processed datasets,
and experiments are never overwritten; use new names/paths for a new experiment.

The distribution may already contain `configs/demo.yaml`, processed synthetic
features and completed example runs. To regenerate raw demo data after extracting
an archive, use the regeneration command in `docs/VERIFICATION.md` rather than
trying to overwrite the included processed dataset. Upload configs/demo.yaml in
the GUI and click Apply to explore the included results.

## Real-data workflow

The intended source is [the Grasping Box dataset](https://osf.io/rsv4z/).
Read [docs/DATASET.md](docs/DATASET.md) before using the real-data adapter.

```bash
python scripts/download_dataset.py --output data/raw
python -m src.cli inspect data/raw/s_1.mat
# Verify source units/labels/frame indices, then edit configs/default.yaml.
python -m src.cli plot --subject 1 --trial 1 --output runs/raw_inspection.png
python -m src.cli build
python -m src.cli train --model ridge --name real_ridge_v1
```

Default YAML deliberately requires source-unit confirmation. The source markers
are cue/touch/lift; they do not directly provide the requested movement onset.
Default EMG-derived onset is an offline proxy, with an external-onset CSV option.
Never report cue- or touch-aligned results as pre-movement decoding without review.

After inspecting the Ridge baseline, train Forest and MLP against the frozen
split. Compare validation results and baselines; only then report held-out results.
Changing EEG/EMG/features/alignment requires rebuilding to a new processed filename.
All important parameters, paths and subject lists are in YAML; model-specific
parameters can be changed without rebuilding features.

## Python API

```python
from src.config import load_config
from src.data.dataset_builder import build_dataset, Dataset
from src.training.train import train
from src.storage.model_io import load_model
from src.inference.predict import EEGSample

config = load_config('configs/default.yaml')
dataset = build_dataset(config)
dataset.save(config['paths']['processed'])
run_path, metrics, model = train(dataset, config, name='ridge_v1')

model = load_model(run_path / 'model')
# eeg_epoch: RAW retained scalp channels × time, in volts, pre-event window only.
# A typed sample checks sampling rate, channel order and timing explicitly.
sample = EEGSample(eeg_epoch, fs, channel_names, (-1.0, -0.1))
activation = model.predict(sample)                 # normalized targets, one per muscle
activation_v = model.predict(sample, units='V')    # inverse-scaled targets
# model.predict(eeg_epoch) adopts the saved channel/fs/window/volts contract.
```

An acquisition system can deliver the same EEGSample without using the dataset
loader. The caller must supply the specified window; the model is not an onset
predictor. For a future online system, evaluate how to schedule those windows.

CLI/GUI inference NPZ format:

```python
import numpy as np
np.savez('sample.npz', eeg=eeg_epoch, fs=fs,
         channel_names=np.array(channel_names), window=np.array([-1., -.1]))
```

```bash
python -m src.cli predict models/trained/ridge_v1 sample.npz
```

## Interface

Nine tabs cover Data, EEG, EMG target, Features, Model & split, Training, Results,
Saved models and Prediction. Read the catalog to select subjects, trial IDs and
grip types, inspect source structure and visualize synchronized traces. Edit
hyperparameters as YAML in the model panel. The sidebar exports the same config
used by the CLI. Builds/training run in a background thread with progress and
cooperative cancellation; one forest candidate finishes before stopping.
Training saves the experiment automatically. Existing experiment names fail
instead of replacing results. Load saved local models to predict uploaded samples
or a selected recorded trial. A separate button saves a model under a new path.

## Project map

```text
configs/             default and synthetic YAML
src/data/            loaders, events, builders, visualization, synthetic fixtures
src/preprocessing/   window boundaries, EEG filters, EMG targets and scaling
src/features/        registry and Welch log-bandpower
src/models/          Ridge, Random Forest, NumPy MLP
src/training/        splitting, training, evaluation and plots
src/storage/         versioned full inference bundles
src/inference/       acquisition-independent EEGSample and prediction
src/cli.py           inspect/build/train/demo/plot/predict/compare
app/app.py           Streamlit interface
scripts/             OSF downloader
runs/<name>/         configuration, splits, metrics, history, plots, model
models/trained/      named saved inference bundles
tests/               leakage, alignment, MAT, model and UI checks
docs/                dataset contract, scientific methods and verification
```

See [docs/METHODS.md](docs/METHODS.md) for temporal filtering tradeoffs, target
normalization on unseen subjects, exact loss conventions and interpretation limits.
New adapters yield Trial objects; new fixed features register in EXTRACTORS.
Learned feature transforms belong in the training-only fitted model pipeline.
