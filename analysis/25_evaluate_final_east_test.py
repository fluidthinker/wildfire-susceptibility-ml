"""Evaluate the preselected Random Forest once on the frozen eastern holdout.

Final-test labels serve only contract checks and final ranking metrics. There
is no tuning, threshold selection, CV, geographic splitting, or subsequent fit.
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

from wildfire_susceptibility.modeling import (
    ASPECT_COLUMNS, CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, PREDICTORS,
    build_random_forest_pipeline, calculate_probability_metrics,
)


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
SPLIT_PATH = ROOT / "data/processed/modeling/nm_modeling_splits.parquet"
TRAINING_PATH = ROOT / "data/processed/modeling/nm_training_dataset.parquet"
CV_PATHS = [ROOT / f"outputs/modeling/random_forest_{design}_cv/summary_metrics.json"
            for design in ("random", "spatial")]
OUTPUT_DIR = ROOT / "outputs/modeling/random_forest_final_test"
PREDICTION_PATH = OUTPUT_DIR / "final_test_predictions.parquet"
SUMMARY_PATH = OUTPUT_DIR / "final_test_summary.json"
IDENTITY = ["cell_id", "target", "spatial_block_id"]
FOLD_COLUMNS = ["random_cv_fold", "spatial_cv_fold"]
PREPROCESSING = {
    "continuous": list(CONTINUOUS_COLUMNS), "aspect": list(ASPECT_COLUMNS),
    "aspect_imputation": "constant 0, pipeline only",
    "numeric_scaling": "none; numeric predictors remain unscaled",
    "categorical": list(CATEGORICAL_COLUMNS),
    "encoding": "OneHotEncoder(handle_unknown='ignore')",
}


# %% Helper functions
def fingerprint_inputs(paths: list[Path]) -> dict[str, str]:
    """Hash immutable inputs using bounded reads and repository-relative names.

    Args:
        paths: Existing source and configuration artifacts.

    Returns:
        Relative input paths mapped to SHA-256 digests.
    """
    fingerprints = {}
    for path in paths:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        fingerprints[path.relative_to(ROOT).as_posix()] = digest
    return fingerprints


def validate_source_tables(modeling: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Require compatible unique keys and the established source schemas.

    Args:
        modeling: Complete eligible raw modeling population.
        splits: Script 16's frozen assignments, never reconstructed here.

    Raises:
        ValueError: If schemas, populations, keys, or saved labels differ.
    """
    schemas = [set(PREDICTORS) | {"cell_id", "target", "burned_fraction"},
               {"cell_id", "split", "spatial_block_id", *FOLD_COLUMNS}]
    for table, schema in zip((modeling, splits), schemas):
        if not table.columns.is_unique or set(table.columns) != schema:
            raise ValueError("Unexpected modeling or split schema")
        if len(table) != 313_129 or table.cell_id.isna().any() or not table.cell_id.is_unique:
            raise ValueError("Each source requires 313,129 unique nonmissing cell IDs")
    if set(modeling.cell_id) != set(splits.cell_id):
        raise ValueError("Modeling and split cell ID sets differ")
    if not splits.split.isin(["train", "final_test"]).all() or splits.spatial_block_id.isna().any():
        raise ValueError("Invalid saved split labels or missing spatial blocks")
    if splits.loc[splits.split.eq("final_test"), FOLD_COLUMNS].notna().any().any():
        raise ValueError("Final-test rows must not have CV assignments")


def validate_populations(training: pd.DataFrame, final_test: pd.DataFrame) -> None:
    """Enforce approved counts, spatial isolation, and raw feature contracts.

    Args:
        training: Rows selected solely by saved train membership.
        final_test: Rows selected solely by saved final_test membership.

    Raises:
        ValueError: If population, feature, or isolation contracts fail.
    """
    for data, rows, positives, blocks, missing, label in (
        (training, 252_569, 6_585, 120, 38, "train"),
        (final_test, 60_560, 821, 33, 1, "final_test"),
    ):
        if len(data) != rows or not data.split.eq(label).all():
            raise ValueError(f"Unexpected {label} membership or row count")
        if data.target.isna().any() or data.target.value_counts().to_dict() != {0: rows - positives, 1: positives}:
            raise ValueError(f"Unexpected {label} binary target counts")
        if data.spatial_block_id.nunique() != blocks:
            raise ValueError(f"Unexpected {label} spatial block count")
        # Script 17 established 38 missing training aspects and one held-out
        # missing aspect. Preserve these values for pipeline-only imputation.
        for column in PREDICTORS:
            expected_missing = missing if column in ASPECT_COLUMNS else 0
            if data[column].isna().sum() != expected_missing:
                raise ValueError(f"{label}: unexpected missingness in {column}")
            if not pd.api.types.is_numeric_dtype(data[column]) or np.isinf(data[column].dropna()).any():
                raise ValueError(f"{label}: nonnumeric or infinite {column}")
        if not pd.api.types.is_integer_dtype(data.evt_dominant_class):
            raise ValueError("EVT must retain integer categorical codes")
    if set(training.cell_id) & set(final_test.cell_id):
        raise ValueError("Training and final-test cells overlap")
    if set(training.spatial_block_id) & set(final_test.spatial_block_id):
        raise ValueError("Training and final-test spatial blocks overlap")


def validate_cv_configuration(parameters: dict, fingerprints: dict) -> None:
    """Confirm the fresh shared model matches both completed CV configurations.

    Args:
        parameters: Unmodified shared Random Forest parameters.
        fingerprints: Current input hashes, including script 17's artifact.

    Raises:
        ValueError: If configuration, training provenance, or versions changed.
    """
    versions = {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    for path in CV_PATHS:
        saved = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "model": "RandomForestClassifier", "predictors": list(PREDICTORS),
            "preprocessing": PREPROCESSING, "model_parameters": parameters,
            "input_sha256": fingerprints[TRAINING_PATH.relative_to(ROOT).as_posix()],
            "versions": versions, "final_test_accessed": False,
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"{path.name}: approved CV contract changed: {key}")


def validate_predictions(predictions: pd.DataFrame, final_test: pd.DataFrame) -> None:
    """Check held-out coverage and exact target/block/identity preservation.

    Args:
        predictions: Proposed or read-back probability artifact.
        final_test: Validated rows from frozen final-test membership.

    Raises:
        ValueError: If coverage, schema, or probabilities are invalid.
        AssertionError: If original identities, targets, or blocks changed.
    """
    if list(predictions.columns) != [*IDENTITY, "predicted_probability"]:
        raise ValueError("Unexpected prediction schema")
    if len(predictions) != 60_560 or predictions.cell_id.nunique() != 60_560:
        raise ValueError("Expected exactly 60,560 unique final-test predictions")
    if predictions.cell_id.isna().any() or int(predictions.target.sum()) != 821:
        raise ValueError("Prediction keys or positive counts changed")
    pd.testing.assert_frame_equal(predictions[IDENTITY], final_test[IDENTITY], check_exact=True)
    probability = predictions.predicted_probability.to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be populated, finite, and in [0,1]")


def publish_results(predictions: pd.DataFrame, summary: dict, final_test: pd.DataFrame,
                    input_paths: list[Path], fingerprints: dict) -> None:
    """Stage both artifacts, read back, and verify inputs before publication.

    Args:
        predictions: Validated final-test probabilities.
        summary: JSON-compatible final evaluation evidence.
        final_test: Original held-out identities for read-back checks.
        input_paths: Immutable files hashed before fitting.
        fingerprints: Original SHA-256 values.

    Raises:
        ValueError: If serialization or input fingerprint validation fails.
        AssertionError: If stored predictions differ from memory.
    """
    destinations = [PREDICTION_PATH, SUMMARY_PATH]
    partials = [path.with_suffix(path.suffix + ".part") for path in destinations]
    try:
        predictions.to_parquet(partials[0], index=False, compression="zstd")
        partials[1].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        written = pd.read_parquet(partials[0])
        validate_predictions(written, final_test)
        pd.testing.assert_frame_equal(written, predictions, check_exact=True)
        if json.loads(partials[1].read_text(encoding="utf-8")) != summary:
            raise ValueError("Summary read-back differs")
        if fingerprint_inputs(input_paths) != fingerprints:
            raise ValueError("Immutable inputs changed during final evaluation")
        # Per-file replacement is atomic, not a two-file transaction. Publish
        # the summary last; an incomplete pair requires review, never auto-refit.
        for partial, destination in zip(partials, destinations):
            partial.replace(destination)
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)


# %% Main workflow
def main() -> None:
    """Fit the approved model once, publish its final evaluation, and stop."""
    started = perf_counter()
    # Fail closed on reruns or interrupted runs. An existing directory requires
    # human review; this script never silently repeats the final evaluation.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    # STEP 1 - Load modeling data and frozen split assignments.
    # Script 16 defines membership; hashes also protect CV configuration evidence.
    input_paths = [MODELING_PATH, SPLIT_PATH, TRAINING_PATH, *CV_PATHS,
                   Path(__file__).resolve(), ROOT / "src/wildfire_susceptibility/modeling.py"]
    fingerprints = fingerprint_inputs(input_paths)
    modeling = pd.read_parquet(MODELING_PATH)
    splits = pd.read_parquet(SPLIT_PATH)
    validate_source_tables(modeling, splits)

    # STEP 2 - Validate and separate training from final test.
    # Matching key sets plus a one-to-one join prevent silent row loss.
    joined = modeling.merge(splits, on="cell_id", how="inner", validate="one_to_one")
    if len(joined) != 313_129:
        raise ValueError("Join did not preserve the complete eligible population")
    training = joined.loc[joined.split.eq("train")].sort_values("cell_id").reset_index(drop=True)
    final_test = joined.loc[joined.split.eq("final_test")].sort_values("cell_id").reset_index(drop=True)
    validate_populations(training, final_test)
    approved_training = pd.read_parquet(TRAINING_PATH).sort_values("cell_id").reset_index(drop=True)
    comparison_columns = [*IDENTITY, *PREDICTORS, *FOLD_COLUMNS]
    pd.testing.assert_frame_equal(training[comparison_columns], approved_training[comparison_columns], check_exact=True)

    # STEP 3 - Select the approved predictors and target.
    # The shared ordering matches Scripts 21/22; no metadata enters X.
    train_x = training[list(PREDICTORS)]
    test_x = final_test[list(PREDICTORS)]
    pipeline = build_random_forest_pipeline()
    parameters = pipeline.named_steps["model"].get_params()
    validate_cv_configuration(parameters, fingerprints)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # STEP 4 - Fit one unchanged pipeline on ALL approved training rows.
        # Only training predictors/labels enter preprocessing and model fitting.
        print("Contracts passed. Fitting once on 252,569 training rows...", flush=True)
        fit_started = perf_counter()
        pipeline.fit(train_x, training.target)
        fit_seconds = perf_counter() - fit_started
        if list(pipeline.named_steps["model"].classes_) != [0, 1]:
            raise ValueError("Expected class-1 probability in column 1")

        # STEP 5 - Predict probabilities for the untouched eastern final test.
        print("Fit complete. Predicting the 60,560 eastern final-test rows...", flush=True)
        predictions = final_test[IDENTITY].copy()
        predictions["predicted_probability"] = pipeline.predict_proba(test_x)[:, 1]
        validate_predictions(predictions, final_test)

        # STEP 6 - Calculate ROC-AUC and Average Precision once.
        # No threshold is selected and there is no subsequent fitting step.
        scores = calculate_probability_metrics(final_test.target.to_numpy(), predictions.predicted_probability.to_numpy())

    # STEP 7 - Validate and safely publish predictions and summary.
    summary = {
        "model": "RandomForestClassifier", "evaluation": "eastern final spatial holdout",
        "training_rows": len(training), "training_positives": int(training.target.sum()),
        "final_test_rows": len(final_test), "final_test_positives": int(final_test.target.sum()),
        "final_test_negatives": int(final_test.target.eq(0).sum()),
        "final_test_prevalence": float(final_test.target.mean()),
        "final_test_block_count": int(final_test.spatial_block_id.nunique()),
        "roc_auc": scores["roc_auc"], "average_precision": scores["pr_auc"],
        "average_precision_definition": "sklearn average_precision_score; not trapezoidal PR area",
        "predictors": list(PREDICTORS), "preprocessing": PREPROCESSING,
        "model_parameters": parameters,
        "training_input_path": TRAINING_PATH.relative_to(ROOT).as_posix(),
        "training_input_role": "validation only; fitting rows selected from modeling data using frozen splits",
        "modeling_input_path": MODELING_PATH.relative_to(ROOT).as_posix(),
        "split_input_path": SPLIT_PATH.relative_to(ROOT).as_posix(),
        "training_input_sha256": fingerprints[TRAINING_PATH.relative_to(ROOT).as_posix()],
        "input_sha256": fingerprints,
        "runtime": {"fit_seconds": round(fit_seconds, 3),
                    "seconds_before_publication": round(perf_counter() - started, 3)},
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "final_test_used_for_fitting": False, "final_test_used_for_preprocessing_fit": False,
        "hyperparameters_changed_after_cv": False, "threshold_selected": False,
        "model_selection_basis": "selected before final-test evaluation using training-only cross-validation",
        "configuration_matches_both_cv_runs": True, "prediction_rows": len(predictions),
    }
    publish_results(predictions, summary, final_test, input_paths, fingerprints)

    # STEP 8 - Report the result and STOP. No statewide prediction or mapping.
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Outputs: {PREDICTION_PATH}\n{SUMMARY_PATH}", flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
