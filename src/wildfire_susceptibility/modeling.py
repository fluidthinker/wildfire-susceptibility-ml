"""Shared Logistic Regression, probability metrics, and CV result comparison."""

from time import perf_counter
import warnings

import numpy as np
import pandas as pd
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
    features: pd.DataFrame, target: pd.Series, validation: pd.Series, fold: int
) -> tuple[np.ndarray, dict]:
    """Fit a fresh pipeline on four folds and predict only the held-out fold.

    Args:
        features: Explicit predictor columns, without metadata or target.
        target: Binary target aligned with features.
        validation: True only for this fold's held-out rows.
        fold: Label included in warning diagnostics.

    Returns:
        Held-out class-1 probabilities and fit diagnostics, including warnings.

    Raises:
        ValueError: If predictors or fitted class ordering are unexpected.
    """
    if list(features.columns) != list(PREDICTORS):
        raise ValueError("X must contain only the explicit predictor list")
    pipeline = build_logistic_pipeline()
    started = perf_counter()
    # Fit calls each transformer using only the four training folds. Validation
    # categories never enter the encoder vocabulary or numeric scaling estimates.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline.fit(features.loc[~validation], target.loc[~validation])
        if list(pipeline.named_steps["model"].classes_) != [0, 1]:
            raise ValueError("Expected class-1 probability in predict_proba column 1")
        probabilities = pipeline.predict_proba(features.loc[validation])[:, 1]
    messages = [{"category": warning.category.__name__, "message": str(warning.message)} for warning in captured]
    for message in messages:
        print(f"Fold {fold} warning: {message}", flush=True)
    return probabilities, {
        "fold": fold, "iterations": int(pipeline.named_steps["model"].n_iter_[0]),
        "runtime_seconds": round(perf_counter() - started, 3), "warnings": messages,
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in captured),
    }
