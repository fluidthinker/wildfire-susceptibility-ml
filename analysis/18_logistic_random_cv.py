"""Evaluate Logistic Regression using the five saved random CV folds only.

Only the raw training-only Parquet is read. No threshold, spatial evaluation,
hyperparameter search, or final-test access is part of this experiment.
"""

# %% Imports
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

from wildfire_susceptibility.modeling import (
    ASPECT_COLUMNS, CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, PREDICTORS,
    build_logistic_pipeline, calculate_probability_metrics,
)


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/processed/modeling/nm_training_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs/modeling/logistic_random_cv"
EXPECTED_ROWS = 252_569
EXPECTED_FOLDS = [(50514, 1317), (50514, 1317), (50514, 1317), (50514, 1317), (50513, 1317)]
METADATA = ["cell_id", "spatial_block_id", "random_cv_fold", "spatial_cv_fold"]
OOF_COLUMNS = ["cell_id", "target", "random_cv_fold", "predicted_probability"]


# %% Helper functions
def validate_training_input(data: pd.DataFrame) -> None:
    """Enforce the approved training population and frozen random-fold contract.

    Args:
        data: Raw training-only dataset created by script 17.

    Raises:
        ValueError: If schema, keys, target, missingness, or fold counts changed.
    """
    if set(data.columns) != set(PREDICTORS) | set(METADATA) | {"target"}:
        raise ValueError("Unexpected training schema; only approved predictors, target, and metadata are allowed")
    wrong_rows = len(data) != EXPECTED_ROWS
    invalid_keys = data.cell_id.isna().any() or data.cell_id.duplicated().any()
    if wrong_rows or invalid_keys:
        raise ValueError("Training requires 252,569 uniquely keyed, nonmissing cell IDs")
    if data.target.value_counts().to_dict() != {0: 245984, 1: 6585} or data.target.isna().any():
        raise ValueError("Training binary target counts changed")
    if not pd.api.types.is_integer_dtype(data.random_cv_fold) or data.random_cv_fold.isna().any():
        raise ValueError("Saved random folds must be nonmissing integers")
    if set(data.random_cv_fold) != set(range(5)):
        raise ValueError("Expected saved random fold labels 0-4")
    counts = data.groupby("random_cv_fold").target.agg(["size", "sum"])
    if list(counts.itertuples(index=False, name=None)) != EXPECTED_FOLDS:
        raise ValueError(f"Frozen random fold populations changed: {counts}")
    for column in PREDICTORS:
        expected_missing = 38 if column in ASPECT_COLUMNS else 0
        if data[column].isna().sum() != expected_missing:
            raise ValueError(f"{column}: expected {expected_missing} missing values")
        if not pd.api.types.is_numeric_dtype(data[column]) or np.isinf(data[column].dropna()).any():
            raise ValueError(f"{column}: unexpected nonnumeric or infinite values")
    if not pd.api.types.is_integer_dtype(data.evt_dominant_class):
        raise ValueError("EVT must retain its original integer categorical identifiers")


def validate_fold_partition(data: pd.DataFrame, validation: pd.Series, fold: int) -> None:
    """Check that one saved validation fold and its complement cover training.

    Args:
        data: Validated complete training population.
        validation: Boolean selection derived directly from saved random folds.
        fold: Frozen validation label.

    Raises:
        ValueError: If membership, counts, classes, or disjointness fail.
    """
    train = data.loc[~validation]
    held_out = data.loc[validation]
    if len(train) + len(held_out) != EXPECTED_ROWS or set(train.cell_id) & set(held_out.cell_id):
        raise ValueError(f"Fold {fold}: train/validation coverage or overlap failure")
    if (len(held_out), int(held_out.target.sum())) != EXPECTED_FOLDS[fold]:
        raise ValueError(f"Fold {fold}: validation population differs from saved design")
    if set(train.target) != {0, 1} or set(held_out.target) != {0, 1}:
        raise ValueError(f"Fold {fold}: both classes must occur in each partition")


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


def run_random_cv(data: pd.DataFrame, features: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Populate one OOF probability per row using only the saved random labels.

    Args:
        data: Validated raw training population.
        features: Predictor-only table aligned with data.
        oof: Output storage with an initially missing probability column.

    Returns:
        Fold metrics and diagnostics; OOF storage is filled in place.

    Raises:
        ValueError: If any row receives other than one validation prediction.
    """
    assignment_counts = np.zeros(len(data), dtype=np.uint8)
    metrics, diagnostics = [], []
    for fold in range(5):
        validation = data.random_cv_fold.eq(fold)
        validate_fold_partition(data, validation, fold)
        print(f"Fitting saved random fold {fold}...", flush=True)
        probabilities, diagnostic = fit_validation_fold(features, data.target, validation, fold)
        oof.loc[validation, "predicted_probability"] = probabilities
        assignment_counts[validation.to_numpy(dtype=bool)] += 1
        labels = data.loc[validation, "target"]
        scores = calculate_probability_metrics(labels.to_numpy(), probabilities)
        metrics.append({"fold": fold, "rows": len(labels), "positives": int(labels.sum()),
                        "positive_prevalence": float(labels.mean()), **scores})
        diagnostics.append(diagnostic)
        print(f"Fold {fold}: {scores}; iterations={diagnostic['iterations']}", flush=True)
    if not np.all(assignment_counts == 1):
        raise ValueError("Every training row must receive exactly one held-out probability")
    return pd.DataFrame(metrics), diagnostics


def validate_oof_predictions(oof: pd.DataFrame, data: pd.DataFrame) -> None:
    """Verify complete probability coverage and exact preservation of row identity.

    Args:
        oof: Proposed or read-back out-of-fold predictions.
        data: Original training-only data.

    Raises:
        ValueError: If output schema, keys, or probabilities are invalid.
        AssertionError: If target labels or frozen fold membership changed.
    """
    if list(oof.columns) != OOF_COLUMNS or len(oof) != EXPECTED_ROWS:
        raise ValueError("Unexpected OOF schema or row count")
    if oof.cell_id.isna().any() or oof.cell_id.duplicated().any():
        raise ValueError("OOF keys must be unique and nonmissing")
    identity = ["cell_id", "target", "random_cv_fold"]
    pd.testing.assert_frame_equal(oof[identity], data[identity], check_exact=True)
    probability = oof.predicted_probability.to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("OOF probabilities must all be populated, finite, and in [0,1]")


def summarize_experiment(oof: pd.DataFrame, folds: pd.DataFrame, diagnostics: list[dict]) -> dict:
    """Summarize combined OOF ranking metrics separately from fold variation.

    Args:
        oof: Complete validated held-out predictions.
        folds: Per-fold probability metrics.
        diagnostics: Iteration counts and captured warnings for every fit.

    Returns:
        JSON-compatible experiment metadata and metrics; std uses ddof=1.
    """
    combined = calculate_probability_metrics(oof.target.to_numpy(), oof.predicted_probability.to_numpy())
    return {
        "model": "LogisticRegression", "validation_method": "frozen random 5-fold CV",
        "rows": len(oof), "positives": int(oof.target.sum()), "negatives": int(oof.target.eq(0).sum()),
        "positive_prevalence": float(oof.target.mean()), "fold_count": 5,
        "roc_auc_oof": combined["roc_auc"], "pr_auc_oof": combined["pr_auc"],
        "roc_auc_fold_mean": float(folds.roc_auc.mean()), "roc_auc_fold_std": float(folds.roc_auc.std(ddof=1)),
        "pr_auc_fold_mean": float(folds.pr_auc.mean()), "pr_auc_fold_std": float(folds.pr_auc.std(ddof=1)),
        "fold_std_ddof": 1, "pr_auc_definition": "average_precision_score; not trapezoidal PR area",
        "predictors": list(PREDICTORS),
        "preprocessing": {"continuous": list(CONTINUOUS_COLUMNS), "aspect": list(ASPECT_COLUMNS),
                          "aspect_imputation": "constant 0 before scaling, pipeline only",
                          "numeric_scaling": "StandardScaler fitted within each training fold",
                          "categorical": list(CATEGORICAL_COLUMNS), "encoding": "OneHotEncoder(handle_unknown='ignore')"},
        "model_parameters": build_logistic_pipeline().named_steps["model"].get_params(),
        "regularization": "L2 (l1_ratio=0), C=1.0", "fold_diagnostics": diagnostics,
        "saved_random_folds_reused": True, "final_test_accessed": False,
        "all_oof_probabilities_populated": True,
        "input_path": INPUT_PATH.relative_to(ROOT).as_posix(),
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }


def publish_results(oof: pd.DataFrame, folds: pd.DataFrame, summary: dict, data: pd.DataFrame) -> None:
    """Stage and verify all three outputs before publishing each final file.

    Args:
        oof: Validated held-out probabilities.
        folds: Fold metrics.
        summary: JSON-compatible summary of the experiment.
        data: Original training rows for read-back validation.

    Raises:
        ValueError: If stored predictions violate their contract.
        AssertionError: If any artifact changes during serialization.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destinations = [OUTPUT_DIR / name for name in
                    ("oof_predictions.parquet", "fold_metrics.csv", "summary_metrics.json")]
    partials = [path.with_suffix(path.suffix + ".part") for path in destinations]
    try:
        oof.to_parquet(partials[0], index=False, compression="zstd")
        folds.to_csv(partials[1], index=False)
        partials[2].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        written = pd.read_parquet(partials[0])
        validate_oof_predictions(written, data)
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


# %% Main workflow
def main() -> None:
    """Run the frozen random-CV experiment and publish its probability evidence."""
    started = perf_counter()
    # STEP 1 - Load only script 17's training artifact and enforce its contract.
    # Final-test source files are never opened by this experiment.
    input_hash = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    data = pd.read_parquet(INPUT_PATH).sort_values("cell_id").reset_index(drop=True)
    validate_training_input(data)

    # STEP 2 - Select the explicit predictors, keeping labels and metadata outside X.
    # EVT fraction is continuous; its dominant class identifier is categorical.
    features = data[list(PREDICTORS)]

    # STEP 3 - Prepare missing OOF storage so incomplete coverage is detectable.
    oof = data[["cell_id", "target", "random_cv_fold"]].copy()
    oof["predicted_probability"] = np.nan

    # STEP 4 - Fit five fresh pipelines using only saved random fold membership.
    # Every validation probability comes from a model that excluded that row.
    folds, diagnostics = run_random_cv(data, features, oof)

    # STEP 5 - Verify complete held-out coverage and unchanged input data.
    validate_oof_predictions(oof, data)
    if hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest() != input_hash:
        raise ValueError("Training input changed during the experiment")

    # STEP 6 - Distinguish combined OOF metrics from mean and sample std across folds.
    # Average Precision evaluates probabilities without selecting a threshold.
    summary = summarize_experiment(oof, folds, diagnostics)
    summary["input_sha256"] = input_hash
    summary["runtime_seconds_before_publication"] = round(perf_counter() - started, 3)

    # STEP 7 - Stage, read back, validate, and publish the three requested artifacts.
    publish_results(oof, folds, summary, data)

    # STEP 8 - Report evidence and warnings without proceeding to another experiment.
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
