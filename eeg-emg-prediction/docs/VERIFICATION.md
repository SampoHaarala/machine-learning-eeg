# Verification report

## Completed in this build

- 20 automated tests: temporal isolation even with derived onset, EMG not used as
  amplitude features, independent target windows, onset failure handling, artifact
  rejection, subject disjointness, within-subject block grouping, training-only
  target and feature scaling, MATLAB fixtures (classic and v7.3 reference arrays),
  constant-target metrics, all model round trips, MLP MSE/Huber, cancellation,
  output-path guard and Streamlit startup/model selection.
- Full synthetic pipeline: 14 subjects × 6 trials = 84 samples; 62 EEG channels,
  13 muscle outputs, 248 features, 60/12/12 trials in a 10/2/2 subject split.
- All three model families trained with the default architecture/settings, selected
  on validation data, saved/reloaded, and returned 13 finite predictions.
- CLI compilation and Streamlit AppTest passed.

The included experiments are synthetic. Their scores are not evidence of
physiological EEG→EMG predictability or superiority of one model on real people.
The synthetic source intentionally contains a predictable relationship.

## Included examples

- `data/processed/synthetic.npz`: features, targets, metadata and raw-file hashes.
- `data/datasets/example_eeg.npz`: one valid raw pre-event EEG sample, in volts.
- `runs/synthetic_ridge`, `runs/synthetic_forest`, `runs/synthetic_mlp`: complete
  experiment reports and bundles. The Ridge folder also has a raw-signal plot.
- `runs/comparison`: validation comparison table and plot.
- `models/trained/`: the three named models.

The large generated trial collection is omitted from the archive. It can be
recreated deterministically from project root:

```bash
python -c "from src.data.synthetic import generate; generate('data/datasets/demo', subjects=14, trials=6)"
```

This populates the relative path in `configs/demo.yaml`, enabling the GUI's raw
trial catalog and visualization. For inference only, regeneration is unnecessary:

```bash
python -m src.cli predict models/trained/synthetic_ridge data/datasets/example_eeg.npz
```

To rebuild features without replacing the packaged dataset, change
`paths.processed` to a new filename, then run `build`. Use new run names for
training. The existing source paths in archived manifests record the build
machine's original file locations; the source SHA256 hashes remain useful after
relocation. Regenerating raw files does not modify those archival manifests.

## Still required for the scientific milestone

1. Download the authors' actual MATLAB data from OSF and verify units/labels.
2. Run inspection and alignment plots; review EMG-onset detection on real trials
   and use independent onset annotations where possible.
3. Build real trial samples, inspect rejection rates by subject/grip, and freeze
   the subject split and preprocessing choices.
4. Train real-data Ridge and compare to mean/grip baselines; inspect per-subject
   errors and onset/guard sensitivity using validation subjects.
5. Evaluate a frozen final protocol on untouched test subjects, then qualify
   uncertainty and contamination limitations before making a research claim.

Actual OSF access timed out. The downloader and real-data adapter are implemented,
but a genuine real-data run and interactive browser-based end-user testing were
not performed. Automated Streamlit tests verify startup and control changes.
