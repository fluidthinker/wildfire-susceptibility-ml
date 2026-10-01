"""Shared model pipelines, frozen-CV mechanics, and result comparison."""

from collections.abc import Callable
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONTINUOUS_COLUMNS = (
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "annual_precip_mean", "evt_dominant_fraction",
)
ASPECT_COLUMNS = ("aspect_sin_mean", "aspect_cos_mean", "aspect_strength")
CATEGORICAL_COLUMNS = ("evt_dominant_class",)
PREDICTORS = (*CONTINUOUS_COLUMNS, *ASPECT_COLUMNS, *CATEGORICAL_COLUMNS)

# Reporting names preserve the existing stored Average Precision definition.
COMPARISON_METRICS = {
    "roc_auc_oof": "roc_auc_oof",
    "average_precision_oof": "pr_auc_oof",
    "roc_auc_fold_mean": "roc_auc_fold_mean",
    "roc_auc_fold_std": "roc_auc_fold_std",
    "average_precision_fold_mean": "pr_auc_fold_mean",
    "average_precision_fold_std": "pr_auc_fold_std",
}


def validate_comparable_summaries(random: dict, spatial: dict) -> dict:
    """Require matching experiment contracts before comparing validation designs.

    Population identity relies on saved provenance, without reopening row data.
    Missing hashes are reported as unverified rather than treated as a match.

    Args:
        random: Saved random-CV summary for any model family.
        spatial: Saved spatial-CV summary using the same core result schema.

    Returns:
        Verified comparison flags; same_input_hash is null if either is absent.

    Raises:
        ValueError: If required metadata or metrics are absent or incompatible.
    """
    same_fields = (
        "model", "rows", "positives", "negatives", "predictors", "preprocessing",
        "model_parameters", "fold_count", "input_path", "pr_auc_definition", "fold_std_ddof",
    )
    for field in same_fields:
        if field not in random or field not in spatial or random[field] != spatial[field]:
            raise ValueError(f"Comparison contract differs or is missing: {field}")
    for strategy, summary in (("random", random), ("spatial", spatial)):
        for field in ("rows", "positives", "negatives", "fold_count"):
            if type(summary[field]) is not int or summary[field] <= 0:
                raise ValueError(f"{strategy}: {field} must be a positive integer")
        if summary["fold_count"] < 2 or summary["rows"] != summary["positives"] + summary["negatives"]:
            raise ValueError(f"{strategy}: inconsistent population or fold count")
        prevalence = summary.get("positive_prevalence", np.nan)
        if not np.isclose(prevalence, summary["positives"] / summary["rows"], rtol=0, atol=1e-12):
            raise ValueError(f"{strategy}: inconsistent positive prevalence")
        if summary.get("final_test_accessed") is not False:
            raise ValueError(f"{strategy}: final-test exclusion must be explicit")
        if summary.get("all_oof_probabilities_populated") is not True:
            raise ValueError(f"{strategy}: complete OOF coverage must be explicit")
        if summary.get(f"saved_{strategy}_folds_reused") is not True:
            raise ValueError(f"{strategy}: frozen fold reuse must be explicit")
        expected_method = f"frozen {strategy} {summary['fold_count']}-fold CV"
        if summary.get("validation_method") != expected_method:
            raise ValueError(f"{strategy}: unexpected validation method")
        if summary["pr_auc_definition"] != "average_precision_score; not trapezoidal PR area":
            raise ValueError("Comparison requires the established Average Precision definition")
        if summary["fold_std_ddof"] != 1:
            raise ValueError("Fold variability must use sample standard deviation (ddof=1)")
        for field in COMPARISON_METRICS.values():
            value = summary.get(field)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{strategy}: invalid metric {field}")
        fingerprint = summary.get("input_sha256")
        if fingerprint is not None and (
            not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(f"{strategy}: invalid input SHA-256")
    if not np.isclose(random["positive_prevalence"], spatial["positive_prevalence"], rtol=0, atol=1e-12):
        raise ValueError("Positive prevalences differ")
    hashes_available = all(summary.get("input_sha256") is not None for summary in (random, spatial))
    if hashes_available and random["input_sha256"] != spatial["input_sha256"]:
        raise ValueError("Training input SHA-256 fingerprints differ")
    return {
        "same_input_hash": True if hashes_available else None,
        "same_predictors": True, "same_preprocessing": True,
        "same_model_parameters": True, "same_training_population": True,
        "population_validation_basis": "saved population counts, input path, and available SHA-256",
        "final_test_accessed": False,
    }


def validate_fold_metrics(folds: pd.DataFrame, summary: dict) -> None:
    """Reconcile saved fold populations and variability with an experiment summary.

    Combined OOF metrics cannot be reconstructed from fold means; they remain
    authoritative summary values. Floating-point comparisons allow only 1e-12.

    Args:
        folds: Saved fold metric table, with optional spatial block counts.
        summary: Experiment summary already checked for comparability.

    Raises:
        ValueError: If fold identities, counts, metrics, or summary statistics fail.
    """
    required = {"fold", "rows", "positives", "positive_prevalence", "roc_auc", "pr_auc"}
    if not required.issubset(folds.columns) or len(folds) != summary["fold_count"]:
        raise ValueError("Fold table schema or row count is invalid")
    integer_columns = ["fold", "rows", "positives"]
    if "spatial_block_count" in summary:
        if "blocks" not in folds:
            raise ValueError("Spatial fold block counts are missing")
        integer_columns.append("blocks")
    for column in integer_columns:
        if not pd.api.types.is_integer_dtype(folds[column]) or folds[column].isna().any():
            raise ValueError(f"Fold {column} must contain nonmissing integers")
    if set(folds.fold) != set(range(summary["fold_count"])):
        raise ValueError("Fold labels must occur exactly once, from zero to fold_count - 1")
    if (folds.positives <= 0).any() or (folds.positives >= folds.rows).any():
        raise ValueError("Every fold must contain both target classes")
    if folds.rows.sum() != summary["rows"] or folds.positives.sum() != summary["positives"]:
        raise ValueError("Fold populations disagree with the summary")
    if not np.allclose(folds.positive_prevalence, folds.positives / folds.rows, rtol=0, atol=1e-12):
        raise ValueError("Fold prevalences disagree with counts")
    if "spatial_block_count" in summary:
        if (folds.blocks <= 0).any() or folds.blocks.sum() != summary["spatial_block_count"]:
            raise ValueError("Spatial block counts disagree with the summary")
    for metric in ("roc_auc", "pr_auc"):
        values = folds[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid fold metric: {metric}")
        for statistic, value in (("mean", values.mean()), ("std", values.std(ddof=1))):
            if not np.isclose(value, summary[f"{metric}_fold_{statistic}"], rtol=0, atol=1e-12):
                raise ValueError(f"Fold {metric} {statistic} disagrees with summary")


def build_validation_comparison(random: dict, spatial: dict) -> tuple[pd.DataFrame, dict]:
    """Extract two comparable result rows and signed Spatial-minus-Random changes.

    Args:
        random: Random-CV summary; no model-specific parameters are assumed.
        spatial: Spatial-CV summary with a matching experiment contract.

    Returns:
        Comparison table and absolute metric differences (not relative changes).

    Raises:
        ValueError: If experiment summaries are incompatible.
    """
    validate_comparable_summaries(random, spatial)
    records = []
    for label, summary in (("Random CV", random), ("Spatial CV", spatial)):
        record = {"model": summary["model"], "validation_strategy": label}
        record.update({label: summary[source] for label, source in COMPARISON_METRICS.items()})
        records.append(record)
    changes = {
        "roc_auc": spatial["roc_auc_oof"] - random["roc_auc_oof"],
        "average_precision": spatial["pr_auc_oof"] - random["pr_auc_oof"],
    }
    return pd.DataFrame(records), changes


def build_logistic_pipeline() -> Pipeline:
    """Create a fresh baseline whose preprocessing learns only from fitted rows.

    Undefined flat-terrain aspect is replaced with zero before scaling. EVT
    codes are nominal categories; unseen validation categories encode as zeros.
    All other numeric predictors must already be nonmissing. The model uses
    L2 regularization, C=1, no class weighting, and no hyperparameter tuning.

    Returns:
        Unfitted preprocessing and Logistic Regression pipeline.
    """
    aspect = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("continuous", StandardScaler(), list(CONTINUOUS_COLUMNS)),
        ("aspect", aspect, list(ASPECT_COLUMNS)),
        ("vegetation", OneHotEncoder(handle_unknown="ignore", sparse_output=True), list(CATEGORICAL_COLUMNS)),
    ], remainder="drop", sparse_threshold=1.0)
    # In sklearn 1.9, l1_ratio=0 specifies L2 without the deprecated penalty argument.
    model = LogisticRegression(
        solver="lbfgs", C=1.0, l1_ratio=0.0, max_iter=2000, tol=1e-4,
        class_weight=None, fit_intercept=True, random_state=42, warm_start=False,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def build_random_forest_pipeline(
    *, max_depth: int | None = None, min_samples_leaf: int = 1,
) -> Pipeline:
    """Create a forest with nominal EVT encoding and no numeric scaling.

    Undefined flat-terrain aspect is imputed to zero inside each fold. Sparse
    one-hot encoding avoids imposing an ordering on arbitrary EVT codes.

    Args:
        max_depth: Maximum tree depth; None preserves the unlimited baseline.
        min_samples_leaf: Minimum leaf population; 1 preserves the baseline.

    Returns:
        Fresh preprocessing and 300-tree Random Forest pipeline.
    """
    preprocessing = ColumnTransformer([
        ("continuous", "passthrough", list(CONTINUOUS_COLUMNS)),
        ("aspect", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True), list(ASPECT_COLUMNS)),
        ("vegetation", OneHotEncoder(handle_unknown="ignore", sparse_output=True), list(CATEGORICAL_COLUMNS)),
    ], remainder="drop", sparse_threshold=1.0)
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight=None,
        criterion="gini", max_depth=max_depth, min_samples_split=2,
        min_samples_leaf=min_samples_leaf, max_features="sqrt", bootstrap=True,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def calculate_probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Measure ranking performance without choosing a classification threshold.

    Args:
        target: One-dimensional binary labels containing both classes.
        probability: Matching class-1 probabilities.

    Returns:
        ROC-AUC and pr_auc, defined as Average Precision (not trapezoidal area).

    Raises:
        ValueError: If labels or probabilities are invalid or misaligned.
    """
    target = np.asarray(target)
    probability = np.asarray(probability)
    if target.ndim != 1 or probability.shape != target.shape or set(target) != {0, 1}:
        raise ValueError("Metrics require matching one-dimensional probabilities and both binary classes")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return {"roc_auc": float(roc_auc_score(target, probability)),
            "pr_auc": float(average_precision_score(target, probability))}



def fit_validation_fold(
    features: pd.DataFrame, target: pd.Series, validation: pd.Series, fold: int,
    pipeline_factory: Callable[[], Pipeline] = build_logistic_pipeline,
) -> tuple[np.ndarray, dict]:
    """Fit a fresh pipeline on four folds and predict only the held-out fold.

    Args:
        features: Explicit predictor columns, without metadata or target.
        target: Binary target aligned with features.
        validation: True only for this fold's held-out rows.
        fold: Label included in warning diagnostics.
        pipeline_factory: Callable returning a fresh unfitted model pipeline.

    Returns:
        Held-out class-1 probabilities and fit diagnostics, including warnings.

    Raises:
        ValueError: If predictors or fitted class ordering are unexpected.
    """
    if list(features.columns) != list(PREDICTORS):
        raise ValueError("X must contain only the explicit predictor list")
    pipeline = pipeline_factory()
    started = perf_counter()
    # Fit calls each transformer using only the four training folds. Validation
    # categories never enter the encoder vocabulary or numeric scaling estimates.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit_started = perf_counter()
        pipeline.fit(features.loc[~validation], target.loc[~validation])
        fit_runtime_seconds = perf_counter() - fit_started
        if list(pipeline.named_steps["model"].classes_) != [0, 1]:
            raise ValueError("Expected class-1 probability in predict_proba column 1")
        probabilities = pipeline.predict_proba(features.loc[validation])[:, 1]
    messages = [{"category": warning.category.__name__, "message": str(warning.message)} for warning in captured]
    for message in messages:
        print(f"Fold {fold} warning: {message}", flush=True)
    model = pipeline.named_steps["model"]
    diagnostic = {
        "fold": fold,
        "fit_runtime_seconds": round(fit_runtime_seconds, 3),
        "runtime_seconds": round(perf_counter() - started, 3), "warnings": messages,
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in captured),
    }
    if hasattr(model, "n_iter_"):
        diagnostic["iterations"] = int(model.n_iter_[0])
    if isinstance(model, RandomForestClassifier):
        diagnostic["trees"] = len(model.estimators_)
    return probabilities, diagnostic


def validate_training_input(
    data: pd.DataFrame, expected_folds: list[tuple[int, int]], aspect_missing: int,
    fold_column: str = "random_cv_fold", expected_blocks: list[int] | None = None,
) -> None:
    """Enforce the approved training population and selected frozen-fold contract.

    Args:
        data: Raw training-only dataset.
        expected_folds: Frozen (row, positive) counts in fold-label order.
        aspect_missing: Approved missing count for each aspect feature.
        fold_column: Saved fold assignment to validate.
        expected_blocks: Spatial block counts per fold; required for spatial CV.

    Raises:
        ValueError: If schema, keys, target, missingness, or fold counts changed.
    """
    expected_rows = sum(rows for rows, _ in expected_folds)
    expected_positives = sum(positives for _, positives in expected_folds)
    metadata = {"cell_id", "spatial_block_id", "random_cv_fold", "spatial_cv_fold"}
    if set(data.columns) != set(PREDICTORS) | metadata | {"target"}:
        raise ValueError("Unexpected training schema; only approved predictors, target, and metadata are allowed")
    wrong_rows = len(data) != expected_rows
    invalid_keys = data.cell_id.isna().any() or data.cell_id.duplicated().any()
    if wrong_rows or invalid_keys:
        raise ValueError(f"Training requires {expected_rows:,} uniquely keyed, nonmissing cell IDs")
    if data.target.value_counts().to_dict() != {0: expected_rows - expected_positives, 1: expected_positives} or data.target.isna().any():
        raise ValueError("Training binary target counts changed")
    if not pd.api.types.is_integer_dtype(data[fold_column]) or data[fold_column].isna().any():
        raise ValueError("Saved folds must be nonmissing integers")
    if set(data[fold_column]) != set(range(len(expected_folds))):
        raise ValueError("Unexpected saved fold labels")
    counts = data.groupby(fold_column).target.agg(["size", "sum"])
    if list(counts.itertuples(index=False, name=None)) != expected_folds:
        raise ValueError(f"Frozen fold populations changed: {counts}")
    if fold_column == "spatial_cv_fold" and expected_blocks is None:
        raise ValueError("Spatial CV requires expected block counts")
    if expected_blocks is not None:
        validate_spatial_block_assignment(data, fold_column, expected_blocks)
    for column in PREDICTORS:
        expected_missing = aspect_missing if column in ASPECT_COLUMNS else 0
        if data[column].isna().sum() != expected_missing:
            raise ValueError(f"{column}: expected {expected_missing} missing values")
        if not pd.api.types.is_numeric_dtype(data[column]) or np.isinf(data[column].dropna()).any():
            raise ValueError(f"{column}: unexpected nonnumeric or infinite values")
    if not pd.api.types.is_integer_dtype(data.evt_dominant_class):
        raise ValueError("EVT must retain its original integer categorical identifiers")


def validate_fold_partition(
    data: pd.DataFrame, validation: pd.Series, fold: int,
    expected_folds: list[tuple[int, int]],
    fold_column: str = "random_cv_fold", expected_blocks: list[int] | None = None,
) -> None:
    """Check that one saved validation fold and its complement cover training.

    Args:
        data: Validated complete training population.
        validation: Boolean selection derived directly from saved folds.
        fold: Frozen validation label.
        expected_folds: Approved (row, positive) counts for each fold.
        fold_column: Saved fold assignment to validate.
        expected_blocks: Optional expected spatial block counts per fold.

    Raises:
        ValueError: If membership, counts, classes, or disjointness fail.
    """
    if not validation.equals(data[fold_column].eq(fold)):
        raise ValueError("Validation membership differs from saved folds")
    train = data.loc[~validation]
    held_out = data.loc[validation]
    if len(train) + len(held_out) != len(data) or set(train.cell_id) & set(held_out.cell_id):
        raise ValueError(f"Fold {fold}: train/validation coverage or overlap failure")
    if (len(held_out), int(held_out.target.sum())) != expected_folds[fold]:
        raise ValueError(f"Fold {fold}: validation population differs from saved design")
    if expected_blocks is not None:
        if set(train.spatial_block_id) & set(held_out.spatial_block_id):
            raise ValueError(f"Fold {fold}: a held-out spatial block also appears in training")
        if held_out.spatial_block_id.nunique() != expected_blocks[fold]:
            raise ValueError(f"Fold {fold}: spatial block count differs from frozen design")
    if set(train.target) != {0, 1} or set(held_out.target) != {0, 1}:
        raise ValueError(f"Fold {fold}: both classes must occur in each partition")


def run_saved_cv(
    data: pd.DataFrame, features: pd.DataFrame, oof: pd.DataFrame,
    expected_folds: list[tuple[int, int]], pipeline_factory: Callable[[], Pipeline],
    fold_column: str = "random_cv_fold", expected_blocks: list[int] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Populate one OOF probability per row using only the selected saved fold labels.

    Args:
        data: Validated raw training population.
        features: Predictor-only table aligned with data.
        oof: Output storage with an initially missing probability column.
        expected_folds: Approved (row, positive) counts for each fold.
        pipeline_factory: Callable returning a fresh model for every fit.
        fold_column: Saved random or spatial fold column.
        expected_blocks: Required per-fold block counts for spatial validation.

    Returns:
        Fold metrics and diagnostics; OOF storage is filled in place.

    Raises:
        ValueError: If any row receives other than one validation prediction.
    """
    if not data.index.is_unique or not features.index.equals(data.index) or not oof.index.equals(data.index):
        raise ValueError("Training, predictors, and OOF storage must have matching unique indices")
    if fold_column == "spatial_cv_fold" and expected_blocks is None:
        raise ValueError("Spatial CV requires expected block counts")
    if expected_blocks is not None:
        validate_spatial_block_assignment(data, fold_column, expected_blocks)
    assignment_counts = np.zeros(len(data), dtype=np.uint8)
    metrics, diagnostics = [], []
    for fold in range(len(expected_folds)):
        validation = data[fold_column].eq(fold)
        validate_fold_partition(data, validation, fold, expected_folds, fold_column, expected_blocks)
        print(f"Fitting saved {fold_column} fold {fold}...", flush=True)
        probabilities, diagnostic = fit_validation_fold(features, data.target, validation, fold, pipeline_factory)
        oof.loc[validation, "predicted_probability"] = probabilities
        assignment_counts[validation.to_numpy(dtype=bool)] += 1
        labels = data.loc[validation, "target"]
        scores = calculate_probability_metrics(labels.to_numpy(), probabilities)
        metrics.append({"fold": fold, "rows": len(labels), "positives": int(labels.sum()),
                        "positive_prevalence": float(labels.mean()), **scores})
        if expected_blocks is not None:
            metrics[-1]["blocks"] = int(data.loc[validation, "spatial_block_id"].nunique())
            diagnostic["spatial_blocks_disjoint"] = True
        diagnostics.append(diagnostic)
        print(f"Fold {fold}: {scores}; runtime={diagnostic['runtime_seconds']} seconds", flush=True)
    if not np.all(assignment_counts == 1):
        raise ValueError("Every training row must receive exactly one held-out probability")
    return pd.DataFrame(metrics), diagnostics


def validate_oof_predictions(
    oof: pd.DataFrame, data: pd.DataFrame, fold_column: str = "random_cv_fold",
) -> None:
    """Verify complete probability coverage and exact preservation of row identity.

    Args:
        oof: Proposed or read-back out-of-fold predictions.
        data: Original training-only data.
        fold_column: Saved fold column; spatial output also preserves block IDs.

    Raises:
        ValueError: If output schema, keys, or probabilities are invalid.
        AssertionError: If target labels or frozen fold membership changed.
    """
    identity = ["cell_id", "target"]
    if fold_column == "spatial_cv_fold":
        identity.append("spatial_block_id")
    identity.append(fold_column)
    if list(oof.columns) != [*identity, "predicted_probability"] or len(oof) != len(data):
        raise ValueError("Unexpected OOF schema or row count")
    if oof.cell_id.isna().any() or oof.cell_id.duplicated().any():
        raise ValueError("OOF keys must be unique and nonmissing")
    pd.testing.assert_frame_equal(oof[identity], data[identity], check_exact=True)
    probability = oof.predicted_probability.to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("OOF probabilities must all be populated, finite, and in [0,1]")


def publish_results(
    oof: pd.DataFrame, folds: pd.DataFrame, summary: dict, data: pd.DataFrame, output_dir: Path,
    fold_column: str = "random_cv_fold",
) -> None:
    """Stage and verify all three outputs before publishing each final file.

    Args:
        oof: Validated held-out probabilities.
        folds: Fold metrics.
        summary: JSON-compatible summary of the experiment.
        data: Original training rows for read-back validation.
        output_dir: Destination directory for this experiment.
        fold_column: Saved fold column used for read-back validation.

    Raises:
        ValueError: If stored predictions violate their contract.
        AssertionError: If any artifact changes during serialization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / name for name in
                    ("oof_predictions.parquet", "fold_metrics.csv", "summary_metrics.json")]
    partials = [path.with_suffix(path.suffix + ".part") for path in destinations]
    try:
        oof.to_parquet(partials[0], index=False, compression="zstd")
        folds.to_csv(partials[1], index=False)
        partials[2].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        written = pd.read_parquet(partials[0])
        validate_oof_predictions(written, data, fold_column)
        pd.testing.assert_frame_equal(written, oof, check_exact=True)
        pd.testing.assert_frame_equal(pd.read_csv(partials[1], float_precision="round_trip"), folds, check_exact=True)
        if json.loads(partials[2].read_text(encoding="utf-8")) != summary:
            raise ValueError("Summary read-back differs")
        # Each replacement is atomic; these three files are not a filesystem transaction.
        # Summary is published last, after the predictions and fold metrics succeed.
        for partial, destination in zip(partials, destinations):
            partial.replace(destination)
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)


def validate_spatial_block_assignment(
    data: pd.DataFrame, fold_column: str, expected_blocks: list[int],
) -> None:
    """Validate saved whole-block membership without rebuilding spatial blocks.

    Args:
        data: Rows containing saved block IDs and fold assignments.
        fold_column: Saved spatial fold column.
        expected_blocks: Frozen block counts in fold-label order.

    Raises:
        ValueError: If blocks are missing, split across folds, or counts changed.
    """
    if data.spatial_block_id.isna().any() or data[fold_column].isna().any():
        raise ValueError("Spatial block IDs and fold assignments must be nonmissing")
    block_folds = data.groupby("spatial_block_id")[fold_column].nunique()
    counts = data.groupby(fold_column).spatial_block_id.nunique()
    if (len(block_folds) != sum(expected_blocks) or not block_folds.eq(1).all()
            or counts.index.tolist() != list(range(len(expected_blocks)))
            or counts.tolist() != expected_blocks):
        raise ValueError("Saved spatial blocks must belong to one fold and match frozen counts")


# Preserve existing callers while sharing the same OOF mechanics across designs.
run_random_cv = run_saved_cv
