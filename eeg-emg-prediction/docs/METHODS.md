# Methods and limits

## Sample definition and temporal isolation

A sample contains scalp EEG in `[onset−1 s, onset−0.1 s)` and EMG targets in
`[onset, onset+0.5 s)`. Intervals are half-open and converted using sample timestamps.
No EEG sample at or beyond the input endpoint reaches a feature computation.
Raw files are read only. Processed outputs are separate and refuse overwrite.

The EEG pipeline removes identified non-EEG channels, applies CAR across the
retained scalp channels, and uses a fourth-order Butterworth prototype with
forward/backward SOS filtering **inside the extracted input window only**.
A Butterworth bandpass transformation has twice the prototype order; forward/backward
application also squares its magnitude response. It should not be described as
having the same response as a single fourth-order low-pass.

Filtering the entire trial with a zero-phase filter before cropping would allow
post-onset data to influence pre-onset features. Here padding only reflects
permitted samples. This protects time isolation at the expense of edge distortion.
A 0.9-second interval is short for 1 Hz filtering and delta-band estimation.
Longer fully pre-onset windows or a validated causal filtering variant should be
compared on validation subjects before making strong spectral interpretations.

Artifacts above 150 µV (configurable) after CAR, before or after filtering, reject
the entire sample. This simple rule does not detect every artifact or bad channel.
There is no ICA, no learned artifact threshold, and no hidden channel interpolation.
A 40 Hz cutoff attenuates but does not guarantee elimination of 50 Hz noise;
optional notch filtering remains available. Filter cutoffs violating Nyquist
raise an error rather than silently changing the effective pipeline.

## Features and targets

Welch uses a Hann window, constant detrending and density scaling. Each channel's
PSD is integrated between interpolated band boundaries, then log(power+epsilon)
is taken. Features are ordered channel-major, band-minor and named in the bundle.
Default bands: 1–4, 4–8, 8–13, 13–30 Hz. 62 channels yield 248 features. Zero-padding
does not increase physical frequency resolution. `EXTRACTORS` is the extension
point for new fixed feature methods. Learned features must be fitted inside the
training stage, never while building the full dataset.

EMG is filtered independently within the target window, then rectified. RMS is
sqrt(mean(filtered_signal²)), which is identical after rectification. The optional
mean-envelope target rectifies, low-pass smooths and averages (negative smoothing
overshoot is clipped). Target-edge effects are also a limitation. No EMG signal
is concatenated with EEG features.

## Scaling and unseen subjects

Feature StandardScaler fits training trials only. Default target scaling computes
a per-muscle p95 separately in each training subject and uses the median of those
values as one global per-muscle denominator. It gives subjects equal influence
on the denominator, applies the same target definition to unseen subjects, and
never reads their targets to fit a transform.

This **cannot identify or remove a new subject's electrode-contact gain**.
That problem requires independent calibration or stronger assumptions. Global
training-only robust scaling and unscaled volts are alternatives. Per-subject
p95 is supported only when the subject has training samples (within-subject
experiments); unseen subjects are rejected. Do not normalize test subjects using
their own full target distributions. Each model can return normalized or inverse-
scaled volt targets. Results include both sets of metrics.

## Models

| Model | Training objective | Selection |
|---|---|---|
| Ridge | sklearn sum of squared errors + alpha × squared weight norm | lowest validation MSE over configured alphas |
| Random Forest | multi-output squared-error impurity | lowest validation MSE over optional parameter grid |
| MLP | mean squared error or mean Huber + weight_decay/2 × sum(W²) | snapshot with lowest validation MSE; early stopping |

Ridge alpha is in sklearn's sum-of-squares convention; an MSE-based lambda equals
alpha/n_training. RF predictions average tree outputs, not class votes. The MLP
is a small NumPy implementation with ReLU hidden layers, linear outputs, Adam
(beta1=.9, beta2=.999, epsilon=1e-8), coupled L2 on weights, deterministic shuffling
and a training-fitted StandardScaler. Huber delta is expressed in normalized
target units. Biases are not regularized. It is CPU-only and includes no dropout.
Weights are restored from the best validation epoch, including when training
ends before patience is exhausted. Predictions can be negative; reporting leaves
these visible instead of silently improving metrics through clipping.

## Evaluation and interpretation

Default subject counts are 10/2/2, with deterministic random assignment and exact
IDs saved. Explicit subject splits must cover every selected subject once. The
within-subject option partitions whole blocks by default. Choosing trial-level
splits is weaker when adjacent trials are correlated; no claims of cross-subject
generalization apply to either within-subject mode.

MAE, RMSE, R² and Pearson r are reported per muscle, as unweighted muscle means,
and separately per held-out subject. Global RMSE is also reported. Undefined
constant-target R²/correlation are JSON null, with defined-muscle counts, never
manufactured zero scores. Subject-level results help reveal participant imbalance;
no confidence intervals or population significance claims are generated.

Training mean and known-grip mean baselines establish how much can be predicted
without EEG. Optional Ridge negative controls shuffle training targets within
subject and task and evaluate on validation data. Because they reuse a selected
alpha, they are diagnostics, not calibrated permutation p-values. Forest impurity
importance and Ridge coefficient magnitudes are descriptive; correlated features
and scaling affect them. They do not localize causal neural sources.

Choose preprocessing, onset thresholds and model family using training/validation
only. `compare` deliberately compares validation scores. Every run stores test
metrics for reporting; repeatedly inspecting them and modifying the method turns
the test set into another validation set. Prefer a preregistered frozen final test
or nested subject cross-validation for a publishable study. Neither high
correlation nor good regression establishes cortical causation: task cues,
pre-existing activation and artifacts may explain results.

## Reproducibility

Each completed run saves config, split IDs/trial IDs, source file SHA256 hashes,
source-code hashes and snapshots, dependency versions, dataset manifest/exclusions, feature names,
selection history, metrics, predictions, plots and complete inference bundle.
A processed dataset embeds its preprocessing hash; a mismatch blocks training.
The distribution contains a tested dependency lock. Floating point behavior can
still vary across BLAS versions and platforms; fixed seeds are not a universal
bitwise reproducibility guarantee. NumPy MLP and sklearn bundles include fitted
feature and target transforms. Joblib is trusted-local serialization, not a secure
format for arbitrary uploads. MLP weights are additionally saved as plain NPZ.

## References

- Dataset and schema: https://doi.org/10.1038/s41597-026-07242-y
- SciPy filtering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
- Welch: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
- sklearn leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
