# Real dataset integration

The matching dataset is Lomele et al. (2026), *High-Density EEG and Multi-Muscle
EMG Dataset during Object Prehension with a sensorized Grasping Box in Humans*.

- Article: https://doi.org/10.1038/s41597-026-07242-y
- Dataset: https://osf.io/rsv4z/ (DOI: https://doi.org/10.17605/OSF.IO/RSV4Z)
- Author analysis code: https://osf.io/g6yta/

The published schema describes one `s_X.mat` per participant, with `data.subject`,
`data.info`, and `data.trials`. The adapter follows that schema, including
`eeg`, `emg`, `LEDon_event_frame`, `touch_event_frame`, `lift_event_frame`, `task`,
and `block`. MATLAB frame indices are one-based by default. Source signals are
already synchronized, and source EMG was low-pass filtered and downsampled.
Use the raw EEG release, not the authors' ICA-processed EEG derivatives.

**Validation status:** the paper's schema was inspected, but repository access
failed in this build environment. The adapter is tested with generated MATLAB
fixtures, including HDF5 references; it has not been run on the authors' actual
files. Do not call the included results a real-data baseline. Unit defaults are
explicit placeholders (`uV` EEG, `mV` EMG), not verified facts about the release.
The builder refuses real MATLAB data until `data.units_confirmed: true` is set.

## Download, inspect, verify

From the project root:

```bash
python scripts/download_dataset.py --output data/raw
python -m src.cli inspect data/raw/s_1.mat
```

Alternatively download the MATLAB files manually from OSF. Match filename case
on your system. The downloader discovers files through the OSF API and skips
existing files; interrupted `.partial` files require review before retrying.
It is provided but could not be network-tested here.

Check source units with the authors' README/import code; then configure
`data.eeg_unit`, `data.emg_unit`, and `data.units_confirmed`. Verify channel labels,
non-EEG exclusions, sampling rate, frame numbering and cue location. The loader
will fail on unexplained channel count mismatches instead of dropping channels.
Inspect at least several raw signal/event plots before training. Confirm signal
units with documentation, not just by looking at plausible amplitude ranges.

## Onset is not a provided movement trigger

The published trial fields contain cue, contact, and lift markers. The pipeline
therefore supports:

- `emg_onset`: first sustained above-threshold EMG activation after the cue.
  Causal bandpass → trailing RMS → per-trial pre-cue median/MAD threshold.
  Threshold combines MAD, baseline ratio and an absolute voltage floor. The
  earliest qualifying selected muscle defines the proxy. Parameters are fixed
  before evaluation; do not tune them using test performance.
- `annotated_onset`: externally reviewed onset seconds from trial start in CSV:
  `subject,trial_id,onset_s`. Prefer independently measured kinematic onset when
  available. Annotating from EMG remains an EMG-derived proxy.
- `cue`, `touch`, `lift`: explicit alternate alignment questions. They are not
  interchangeable with onset. Touch/lift can place movement inside EEG inputs.
- `movement_onset`: a named event for a canonical dataset that actually supplies
  it; included in synthetic fixtures only, not invented for the MATLAB files.

The EMG detector records thresholds and rejects missing onsets. A pre-EEG-window
activation check rejects obvious prior muscle activity. This does not prove the
absence of subtle EMG, cranial-muscle contamination or detector timing error.
The derived timestamp depends on EMG, but EMG amplitudes are never predictors.
This is a retrospectively aligned study, not online prediction of when movement
will begin. Review detector errors, vary the guard interval in training/validation,
and report onset definitions and exclusion counts.

## Another synchronized dataset

Replace the adapter with one yielding `Trial` objects: EEG and EMG in volts,
channels × samples, synchronized equal sample rates, event seconds relative to
trial start, stable channel order and explicit identity fields. For unequal clocks,
first align synchronization pulses and resample with a documented anti-alias
procedure; this version refuses to guess clock offsets or silently resample.

Canonical NPZ stores `eeg`, `emg` numeric arrays and JSON string `metadata`:

```json
{"subject":"1","trial_id":"1","task":"PG","block":"1","fs":1000,
 "unit":"V","eeg_names":["C3","C4"],"emg_names":["FDI"],
 "events":{"cue":2.0,"movement_onset":2.4},"extra":{"synthetic":false}}
```

This example describes the schema only; it is not a real trial. Never store
object arrays requiring pickle in canonical or processed NPZ files.
