# Resumable project status

## Part 1 — Data pipeline: complete, fixture-tested
Published-schema MATLAB and canonical NPZ loaders, MATLAB v7.3 references,
explicit units/channel/rate checks, onset alignment, independent EEG/EMG windows,
artifact rejection, Welch features, processed NPZ, raw plots and provenance.

## Part 2 — Training and inference: complete, synthetic-tested
Ridge, multi-output Random Forest, NumPy MLP (MSE/Huber/Adam/early stopping),
training-only scaling, subject/block splits, baseline controls, evaluation,
plots, saved model bundles, inference, run tracking and CLI.

## Part 3 — Interface and packaging: complete, tested and packaged
Streamlit's nine sections, background jobs/cancellation, YAML config editing,
model management and sample prediction; documentation and 20 automated tests.
Full 62-input-channel/13-output demo passed for all three models.

## Unresolved research validation
The actual raw dataset could not be fetched here. Unit defaults MUST be verified
before enabling real MATLAB builds. Onsets must be reviewed: the published markers
are cue/touch/lift, so the default EMG-derived onset is an offline proxy.
No real-data prediction claims have been established.

## Resume instructions
Open README.md and docs/VERIFICATION.md. If packaging is complete, start with
`python -m pytest -q`, then launch `python -m streamlit run app/app.py`.
To continue the scientific experiment, download OSF RSV4Z and follow the five
real-data steps in docs/VERIFICATION.md. Do not silently reinterpret cue as onset
or fit a scaler on held-out subjects.
