"""Generate the standalone EEG Speed Toolkit API guide."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "eeg_speed_toolkit_guide.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#236A8D")
TEAL = colors.HexColor("#2A8C82")
PALE = colors.HexColor("#EAF3F6")
LIGHT = colors.HexColor("#F5F7F9")
MID = colors.HexColor("#D5E0E5")
TEXT = colors.HexColor("#23313A")
MUTED = colors.HexColor("#5D6C74")
ORANGE = colors.HexColor("#D9822B")


class GuideDocument(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=18 * mm,
            title="EEG Speed Toolkit - Composable API Guide",
            author="EEG Speed Toolkit",
            subject="Function reference and assembly guide",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates([PageTemplate(id="guide", frames=[frame], onPage=self._page)])

    def _page(self, canvas, document):
        canvas.saveState()
        if document.page > 1:
            canvas.setStrokeColor(MID)
            canvas.line(18 * mm, A4[1] - 13 * mm, A4[0] - 18 * mm, A4[1] - 13 * mm)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(18 * mm, A4[1] - 10 * mm, "EEG Speed Toolkit")
            canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=28,
    leading=32, textColor=NAVY, alignment=TA_LEFT, spaceAfter=7 * mm,
))
styles.add(ParagraphStyle(
    "CoverSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=13,
    leading=18, textColor=BLUE, spaceAfter=8 * mm,
))
styles.add(ParagraphStyle(
    "H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=19,
    leading=23, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    "H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13,
    leading=16, textColor=BLUE, spaceBefore=4 * mm, spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    "H3x", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10.5,
    leading=13, textColor=TEAL, wordWrap="CJK",
    spaceBefore=3 * mm, spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    "Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2,
    leading=13, textColor=TEXT, spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    "Smallx", parent=styles["BodyText"], fontName="Helvetica", fontSize=8,
    leading=10.5, textColor=MUTED, spaceAfter=1.5 * mm,
))
styles.add(ParagraphStyle(
    "Calloutx", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=9.2,
    leading=13, textColor=NAVY, backColor=PALE, borderColor=TEAL,
    borderWidth=0.7, borderPadding=7, spaceBefore=2 * mm, spaceAfter=4 * mm,
))
styles.add(ParagraphStyle(
    "Warnx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9,
    leading=12.5, textColor=TEXT, backColor=colors.HexColor("#FFF3E3"),
    borderColor=ORANGE, borderWidth=0.7, borderPadding=7,
    spaceBefore=2 * mm, spaceAfter=4 * mm,
))
styles.add(ParagraphStyle(
    "Codex", fontName="Courier", fontSize=7.2, leading=9.2, textColor=TEXT,
    backColor=LIGHT, borderColor=MID, borderWidth=0.5, borderPadding=6,
    spaceBefore=1.5 * mm, spaceAfter=3 * mm,
))


def p(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def code(text):
    return Preformatted(text.strip("\n"), styles["Codex"])


def heading(text, level=1):
    return p(text, {1: "H1x", 2: "H2x", 3: "H3x"}[level])


def function_block(signature, purpose, when, notes=None):
    parts = [
        p(signature, "H3x"),
        p(f"<b>Purpose.</b> {purpose}"),
        p(f"<b>Use it when.</b> {when}"),
    ]
    if notes:
        parts.append(p(f"<b>Important.</b> {notes}", "Smallx"))
    return KeepTogether(parts)


def api_table(rows):
    data = [[p("Object", "Smallx"), p("What it contains", "Smallx")]]
    data.extend([[p(a, "Smallx"), p(b, "Smallx")] for a, b in rows])
    table = Table(data, colWidths=[55 * mm, 115 * mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, MID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


story = []
story += [
    Spacer(1, 20 * mm),
    p("EEG Speed Toolkit", "CoverTitle"),
    p("Composable API guide for EEG-only reach-speed classification", "CoverSubtitle"),
    p(
        "A standalone function library for building your own training function. "
        "No high-level call owns your experiment, filesystem, split loop, or reporting.",
        "Calloutx",
    ),
    Spacer(1, 8 * mm),
    api_table([
        ("Package", "eeg_speed_toolkit"),
        ("Primary target", "Fast versus slow reach completion within participant and grasp task"),
        ("Primary EEG feature", "Time-resolved 0-5 Hz movement-related cortical potential"),
        ("Recommended validation", "Leave one subject out, with group-disjoint inner tuning"),
        ("Python", "3.10 or newer"),
        ("Branch", "feature/eeg-speed-toolkit"),
    ]),
    Spacer(1, 10 * mm),
    p("Version 0.1 - 6 October 2026", "Smallx"),
    PageBreak(),
]

story += [
    heading("1. Design contract"),
    p(
        "The toolkit is intentionally assembled from small operations. A function either validates, "
        "constructs labels, returns indices, creates features, fits one model, or calculates metrics. "
        "No public function combines all of these steps.",
    ),
    p(
        "The application is responsible for the outer loop. This makes the train, validation, and test "
        "boundaries visible in your own code and prevents a short wrapper from hiding leakage-sensitive work.",
        "Calloutx",
    ),
    heading("Recommended call order", 2),
    api_table([
        ("1. Configuration", "load_config() or default_config(), then validate_config()."),
        ("2. Dataset", "load_dataset() or construct EEGSpeedDataset, then validate_dataset()."),
        ("3. Target", "Choose calibrated within-subject labels or outer-training normative labels."),
        ("4. Split", "Request index arrays. Inspect them before touching a model."),
        ("5. Features", "Create one EEG matrix. Fit any learned transform on training rows only."),
        ("6. Model", "Construct or tune one estimator with caller-supplied training arrays."),
        ("7. Prediction", "Call estimator.predict() on validation or test rows."),
        ("8. Evaluation", "Pass predictions and labels to metric and permutation functions."),
    ]),
    heading("What the library never does", 2),
    p(
        "It never creates a run directory, chooses an outer fold, saves a model, loads raw participant "
        "recordings, or calls a monolithic train_experiment routine. Persistence is intentionally outside "
        "the package.",
        "Warnx",
    ),
    PageBreak(),
]

story += [
    heading("2. Dataset and configuration"),
    heading("EEGSpeedDataset", 2),
    p("A frozen dataclass whose first array axis is always trial. Required fields:"),
    api_table([
        ("mrcp_full", "trials x channels x samples, normally -500 to +300 ms around onset"),
        ("mrcp_pre", "trials x channels x samples, normally -500 to 0 ms"),
        ("erd", "trials x fixed ERD features"),
        ("duration", "positive reach duration in seconds"),
        ("reaction_time", "nonnegative cue-to-onset time in seconds"),
        ("tasks / subjects / trials", "one string identifier per trial; trial identifiers must be unique"),
        ("mrcp_frontal / posterior", "optional spatial-control epochs"),
        ("manifest", "optional metadata, including motor channel names"),
    ]),
    function_block(
        "load_dataset(path) -> EEGSpeedDataset",
        "Load the documented NPZ contract. It performs no fitting or filtering.",
        "You already produced eeg_speed_features_v2.npz or another compatible artifact.",
    ),
    function_block(
        "validate_dataset(dataset, config=None) -> DatasetSummary",
        "Check trial counts, ranks, finite values, positive durations, unique identifiers, and feature availability.",
        "Immediately after constructing or loading a dataset, before labels or splits.",
        "Validation is read-only. Passing config adds checks for the selected feature.",
    ),
    function_block(
        "subset_dataset(dataset, indices) -> EEGSpeedDataset",
        "Create an immutable trial subset across every array.",
        "You need a compact dataset object for inspection or a downstream component.",
        "For model fitting, direct matrix indexing is usually clearer.",
    ),
    PageBreak(),
]

story += [
    heading("3. Configuration functions"),
    function_block(
        "default_config() -> dict",
        "Return a new mutable copy of the recommended configuration.",
        "You construct configuration in Python and want safe defaults.",
    ),
    function_block(
        "load_config(path) -> dict",
        "Read YAML, validate it, and return a normalized copy.",
        "Configuration belongs in a versioned YAML file.",
    ),
    function_block(
        "validate_config(config) -> dict",
        "Validate supported feature, target, split, model, and evaluation options without mutating the input.",
        "Your code creates or modifies a mapping before use.",
    ),
    heading("Supported top-level sections", 2),
    api_table([
        ("seed", "Integer used only where the caller passes the configuration onward."),
        ("feature", "name plus include_context flag. The flag documents intent; you still assemble context explicitly."),
        ("target", "within_subject_task_median or training_task_median."),
        ("split", "mode, excluded subjects, and inner group-fold count."),
        ("model", "logistic model, C candidates, class weighting, iterations, and worker count."),
        ("evaluation", "permutation repeat count."),
    ]),
    code("""
config = default_config()
config["feature"]["name"] = "mrcp_laplacian"
config["split"]["exclude_subjects"] = ["14"]
config = validate_config(config)
"""),
    PageBreak(),
]

story += [
    heading("4. Target construction"),
    function_block(
        "make_within_subject_task_labels(duration, tasks, subjects) -> TargetLabels",
        "Create 0=fast and 1=slow labels relative to each participant-and-task median.",
        "Your scientific question concerns trial-to-trial speed differences after removing person and grasp baselines.",
        "This target uses the held-out participant's outcome distribution. It is calibrated, not zero-calibration deployment.",
    ),
    function_block(
        "make_training_task_median_labels(duration, tasks, train_indices, test_indices) -> FoldTargets",
        "Learn one median per task from outer-training outcomes and apply those thresholds to train and test rows.",
        "You want a normative unseen-participant target with no threshold information from the test subject.",
        "Call it separately inside every outer fold.",
    ),
    heading("Returned target objects", 2),
    api_table([
        ("TargetLabels.labels", "One binary label per input trial."),
        ("TargetLabels.thresholds", "Nested subject then task threshold mapping."),
        ("FoldTargets.train_labels", "Labels aligned to train_indices, not the complete dataset."),
        ("FoldTargets.test_labels", "Labels aligned to test_indices."),
        ("description", "Human-readable definition suitable for a report."),
    ]),
    p(
        "Do not calculate one global median before subject-held-out testing. That mixes participant and task "
        "effects and can create deceptively strong pooled performance.",
        "Warnx",
    ),
    PageBreak(),
]

story += [
    heading("5. Splitting functions"),
    function_block(
        "leave_one_subject_out_splits(subjects, exclude_subjects=()) -> iterator[IndexSplit]",
        "Yield explicit outer train/test indices, once per retained subject.",
        "You need subject-independent evaluation.",
    ),
    function_block(
        "grouped_validation_splits(groups, candidate_indices=None, n_splits=5)",
        "Yield train/validation indices with no group overlap.",
        "You are writing custom hyperparameter selection and want inner subject-disjoint folds.",
        "Pass only the outer-training indices as candidate_indices.",
    ),
    function_block(
        "subject_partition(subjects, validation_subjects, test_subjects=(), exclude_subjects=())",
        "Create one caller-specified subject-disjoint train/validation/test partition.",
        "You want a fixed, reviewable split rather than cross-validation.",
    ),
    function_block(
        "personalized_contiguous_splits(tasks, trial_numbers, n_splits=5)",
        "Hold out one contiguous trial chunk from every task.",
        "You train one personalized model and need temporal rather than random generalization.",
    ),
    heading("IndexSplit", 2),
    p(
        "Each split contains train, validation, optional test, and a name. Empty validation arrays in LOSO "
        "are deliberate: outer LOSO defines train/test, while you explicitly choose an inner validation method.",
    ),
    PageBreak(),
]

story += [
    heading("6. EEG preprocessing helpers"),
    function_block(
        "average_reference(eeg) -> ndarray",
        "Subtract the channel average from 2D continuous EEG or 3D epochs.",
        "Your input has not already been average-referenced.",
    ),
    function_block(
        "extract_epoch(signal, event_time_s, window_s, sampling_rate_hz)",
        "Extract a copied channels x samples interval around an event.",
        "You are building the feature dataset from continuous or trial-wise EEG.",
    ),
    function_block(
        "lowpass_filter(eeg, sampling_rate_hz, cutoff_hz=5.0)",
        "Apply the fourth-order zero-phase low-pass used for offline MRCP extraction.",
        "You want MRCP morphology rather than oscillatory EEG.",
        "Zero-phase filtering is offline and noncausal. Do not describe it as a real-time filter.",
    ),
    function_block(
        "baseline_correct(epoch, baseline) -> ndarray",
        "Subtract each channel's baseline mean.",
        "You have separately extracted a pre-cue baseline for the same trial.",
    ),
    function_block(
        "resample_epoch(epoch, original_hz, target_hz) -> ndarray",
        "Polyphase-resample the final time axis.",
        "You want a compact fixed-rate epoch, such as 50 Hz MRCP samples.",
    ),
    code("""
referenced = average_reference(trial_eeg)
baseline = extract_epoch(referenced, cue_s, (-0.9, -0.5), fs)
epoch = extract_epoch(referenced, movement_onset_s, (-0.5, 0.3), fs)
mrcp = lowpass_filter(epoch, fs, cutoff_hz=5.0)
mrcp = baseline_correct(mrcp, lowpass_filter(baseline, fs, 5.0))
mrcp_50hz = resample_epoch(mrcp, int(fs), 50)
"""),
    PageBreak(),
]

story += [
    heading("7. Feature assembly"),
    function_block(
        "flatten_epochs(epochs) -> ndarray",
        "Convert trials x channels x samples into trials x features.",
        "You want a conventional estimator input while preserving all channel-time values.",
    ),
    function_block(
        "central_surface_laplacian(epochs, channel_names=None) -> ndarray",
        "Subtract local FC, CP, and lateral neighbors from seven central channels.",
        "You want to suppress spatially global movement or reference activity.",
        "The input must contain the documented 21-channel motor grid.",
    ),
    function_block(
        "make_eeg_feature_matrix(dataset, name) -> ndarray",
        "Select and flatten mrcp_full, mrcp_pre, mrcp_laplacian, spatial controls, or ERD.",
        "You want a validated fixed feature representation with no learned parameters.",
    ),
    function_block(
        "ContextEncoder.fit / transform / fit_transform",
        "One-hot encode task and append reaction time using training-known task categories.",
        "You are building an explicit non-EEG control or additive baseline.",
        "Fit on outer-training tasks, then transform validation/test tasks.",
    ),
    function_block(
        "append_context(eeg_features, context_features) -> ndarray",
        "Concatenate two prepared matrices by trial.",
        "You explicitly choose to test incremental EEG value beyond context.",
    ),
    p(
        "The recommended primary EEG feature is mrcp_full. Always report mrcp_pre and mrcp_laplacian as "
        "controls, because the observed signal emerged after movement onset and was spatially broad.",
        "Calloutx",
    ),
    PageBreak(),
]

story += [
    heading("8. Training-only spatial filtering"),
    function_block(
        "epoch_covariances(epochs) -> ndarray",
        "Calculate trace-normalized covariance for every epoch.",
        "You are implementing spatial covariance features or inspecting CSP inputs.",
    ),
    function_block(
        "FilterBankCSP(sampling_rate_hz, bands, filters_per_side, regularization)",
        "Create a binary filter-bank common spatial pattern transformer.",
        "You want learned mu/beta spatial filters as an alternative to fixed MRCP features.",
    ),
    function_block(
        "FilterBankCSP.fit(train_epochs, train_labels)",
        "Learn band-specific generalized eigenvectors from training classes only.",
        "Inside an outer split, before touching validation or test epochs.",
    ),
    function_block(
        "FilterBankCSP.transform(epochs) -> ndarray",
        "Apply already learned spatial filters and return log-variance features.",
        "After fit, for training, validation, or test epochs.",
    ),
    code("""
csp = FilterBankCSP(sampling_rate_hz=250)
x_train = csp.fit_transform(raw_epochs[train], y[train])
x_test = csp.transform(raw_epochs[test])
"""),
    p(
        "Never call fit_transform separately on test data. That would estimate class-sensitive spatial "
        "filters from the held-out participant.",
        "Warnx",
    ),
    PageBreak(),
]

story += [
    heading("9. Model functions"),
    function_block(
        "make_logistic_classifier(config, c=None, seed=None) -> Pipeline",
        "Construct an unfitted StandardScaler plus class-weighted logistic classifier.",
        "You already chose C or want a single explicit model.",
    ),
    function_block(
        "fit_classifier(estimator, train_features, train_labels) -> estimator",
        "Clone and fit one estimator on exactly the supplied rows.",
        "You manage validation yourself or fit a final model after selecting settings.",
        "The input estimator is not modified.",
    ),
    function_block(
        "tune_logistic_classifier(train_features, train_labels, train_groups, config)",
        "Select C with group-disjoint inner CV and refit on all supplied outer-training rows.",
        "You want the recommended nested subject-held-out procedure.",
        "The function cannot access outer test arrays because they are not arguments.",
    ),
    heading("ModelSelectionResult", 2),
    api_table([
        ("estimator", "Fitted scaler-plus-logistic pipeline, refit on all supplied training rows."),
        ("best_c", "Selected inverse regularization strength."),
        ("validation_score", "Mean inner balanced accuracy for best C."),
        ("cv_results", "Full scikit-learn model-selection diagnostics."),
    ]),
    p(
        "Call estimator.predict(x) and estimator.predict_proba(x) directly. Thin wrappers would hide the "
        "standard estimator interface without adding safety.",
        "Smallx",
    ),
    PageBreak(),
]

story += [
    heading("10. Evaluation functions"),
    function_block(
        "classification_metrics(true_labels, predicted_labels) -> dict",
        "Return balanced accuracy, macro F1, confusion matrix, and sample count.",
        "After you have predictions for one fold or pooled held-out predictions.",
    ),
    function_block(
        "per_subject_metrics(true_labels, predicted_labels, subjects) -> dict",
        "Calculate the same metrics for every participant.",
        "You need to show whether a pooled result is consistent across people.",
    ),
    function_block(
        "permutation_pvalue(true_labels, predicted_labels, subjects, tasks, repeats, seed)",
        "Shuffle labels within participant and task while keeping predictions fixed.",
        "You need a null that preserves participant and grasp composition.",
        "Use pooled outer-held-out predictions, not training predictions.",
    ),
    p(
        "Balanced accuracy is primary because fold-specific class proportions can differ even when labels "
        "are median-based. Always retain the confusion matrix and participant table.",
        "Calloutx",
    ),
    PageBreak(),
]

story += [
    heading("11. Complete caller-owned assembly"),
    p(
        "The following example is intentionally longer than three lines. Every leakage-sensitive decision "
        "is visible in the application function.",
    ),
    code("""
import numpy as np
from eeg_speed_toolkit import (
    load_dataset, load_config, validate_dataset,
    make_eeg_feature_matrix, make_within_subject_task_labels,
    leave_one_subject_out_splits, tune_logistic_classifier,
    classification_metrics, per_subject_metrics, permutation_pvalue,
)

def my_training_function(dataset, config):
    validate_dataset(dataset, config)
    x = make_eeg_feature_matrix(dataset, config["feature"]["name"])
    target = make_within_subject_task_labels(
        dataset.duration, dataset.tasks, dataset.subjects
    )

    prediction = np.full(dataset.n_samples, -1, dtype=int)
    evaluated = np.zeros(dataset.n_samples, dtype=bool)
    models = {}

    splits = leave_one_subject_out_splits(
        dataset.subjects,
        exclude_subjects=config["split"]["exclude_subjects"],
    )
    for split in splits:
        selection = tune_logistic_classifier(
            x[split.train],
            target.labels[split.train],
            dataset.subjects[split.train],
            config,
        )
        prediction[split.test] = selection.estimator.predict(x[split.test])
        evaluated[split.test] = True
        models[split.name] = selection
"""),
    PageBreak(),
    heading("11. Complete caller-owned assembly - continued"),
    code("""
    truth = target.labels[evaluated]
    predicted = prediction[evaluated]
    result = classification_metrics(truth, predicted)
    result["subjects"] = per_subject_metrics(
        truth, predicted, dataset.subjects[evaluated]
    )
    result["permutation_p"] = permutation_pvalue(
        truth,
        predicted,
        dataset.subjects[evaluated],
        dataset.tasks[evaluated],
        repeats=config["evaluation"]["permutations"],
        seed=config["seed"],
    )
    return models, prediction, result

config = load_config("config/example.yaml")
dataset = load_dataset("eeg_speed_features_v2.npz")
models, prediction, result = my_training_function(dataset, config)
"""),
    p(
        "Your application may replace the model, feature matrix, split iterator, result structure, and "
        "persistence. The toolkit functions remain useful because their contracts are independent.",
        "Calloutx",
    ),
    heading("Using normative labels instead", 2),
    code("""
for split in leave_one_subject_out_splits(dataset.subjects, ["14"]):
    target = make_training_task_median_labels(
        dataset.duration, dataset.tasks, split.train, split.test
    )
    selection = tune_logistic_classifier(
        x[split.train], target.train_labels,
        dataset.subjects[split.train], config,
    )
    prediction[split.test] = selection.estimator.predict(x[split.test])
"""),
    PageBreak(),
]

story += [
    heading("12. Leakage and interpretation checklist"),
    api_table([
        ("Dataset", "All arrays share the same trial order; trial IDs are unique; units and event definitions are documented."),
        ("Target", "State whether labels use held-out participant calibration or outer-training normative thresholds."),
        ("Outer split", "A participant never appears in both train and test for subject-independent claims."),
        ("Inner split", "Regularization and learned spatial filters use only outer-training participants."),
        ("Scaling", "StandardScaler remains inside the estimator pipeline and is fit on training rows."),
        ("CSP", "fit() sees training epochs and labels only; transform() handles held-out epochs."),
        ("Context", "Reaction time and task are reported as controls, not silently concatenated."),
        ("Metrics", "Use pooled held-out balanced accuracy plus per-subject results."),
        ("Null", "Shuffle labels within participant and task."),
        ("Claim", "The successful MRCP window includes early execution. Do not call it pre-movement intention decoding."),
        ("Spatial caveat", "Frontal/posterior controls were strong; retain the central-Laplacian analysis."),
    ]),
    heading("Recommended minimum report", 2),
    p(
        "Report feature window, onset source, target calibration, excluded participants, outer and inner "
        "splits, chosen C distribution, pooled balanced accuracy, macro F1, confusion matrix, participant "
        "scores, permutation p-value, pre-onset control, frontal/posterior controls, and Laplacian control.",
    ),
    PageBreak(),
]

story += [
    heading("13. Quick reference"),
    api_table([
        ("config.py", "default_config, load_config, validate_config"),
        ("dataset.py", "EEGSpeedDataset, DatasetSummary, load_dataset, subset_dataset, validate_dataset"),
        ("targets.py", "TargetLabels, FoldTargets, calibrated and training-only median labels"),
        ("splitting.py", "IndexSplit, LOSO, grouped inner folds, fixed partition, contiguous personalized folds"),
        ("preprocessing.py", "average reference, epoch extraction, low-pass, baseline correction, resampling"),
        ("features.py", "flattening, central Laplacian, EEG selector, context encoder, explicit concatenation"),
        ("spatial.py", "epoch covariance and training-only FilterBankCSP"),
        ("models.py", "logistic pipeline construction, direct fitting, group-aware C selection"),
        ("evaluation.py", "pooled metrics, participant metrics, stratified permutation p-value"),
    ]),
    Spacer(1, 8 * mm),
    p(
        "Core principle: the toolkit supplies testable building blocks; your function remains the visible "
        "owner of the experiment.",
        "Calloutx",
    ),
    heading("Project layout", 2),
    code("""
eeg-speed-toolkit/
  config/example.yaml
  examples/manual_subject_holdout.py
  src/eeg_speed_toolkit/
    config.py       dataset.py       targets.py
    splitting.py    preprocessing.py features.py
    spatial.py      models.py        evaluation.py
  tests/test_toolkit.py
  output/pdf/eeg_speed_toolkit_guide.pdf
"""),
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = GuideDocument(str(OUTPUT))
    document.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
