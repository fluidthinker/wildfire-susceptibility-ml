"""Post-hoc evaluation of the Script 26 winner on the existing eastern holdout.

The original holdout result was already observed. This is not a new untouched
test, and this script performs one fit without tuning or threshold selection.
"""

# %% Imports
import json
from pathlib import Path
import runpy
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import sklearn

from wildfire_susceptibility.modeling import (
    PREDICTORS, build_random_forest_pipeline, calculate_probability_metrics,
)


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
SPLIT_PATH = ROOT / "data/processed/modeling/nm_modeling_splits.parquet"
TRAINING_PATH = ROOT / "data/processed/modeling/nm_training_dataset.parquet"
TUNING_PATH = ROOT / "outputs/modeling/random_forest_spatial_tuning/tuning_summary.json"
BASELINE_PATH = ROOT / "outputs/modeling/random_forest_final_test/final_test_summary.json"
OUTPUT_DIR = ROOT / "outputs/modeling/random_forest_tuned_east_holdout"
TUNED_PARAMETERS = {"max_depth": 30, "min_samples_leaf": 5}
FIXED_PARAMETERS = {
    "n_estimators": 300, "random_state": 42, "n_jobs": -1,
    "class_weight": None, "criterion": "gini", "bootstrap": True,
    "max_features": "sqrt", "min_samples_split": 2,
}
IDENTITY = ["cell_id", "target", "spatial_block_id"]

# Reuse the existing case-study contracts without copying them or changing
# Script 25. run_path's default name is not __main__, so its fit never runs.
# An explicit file path also works in VS Code cells without sys.path changes.
_baseline_script = runpy.run_path(str(ROOT / "analysis/25_evaluate_final_east_test.py"))
fingerprint_inputs = _baseline_script["fingerprint_inputs"]
validate_source_tables = _baseline_script["validate_source_tables"]
validate_populations = _baseline_script["validate_populations"]
validate_predictions = _baseline_script["validate_predictions"]
PREPROCESSING = _baseline_script["PREPROCESSING"]


# %% Helper functions
def validate_tuning_selection(saved: dict, fingerprints: dict[str, str]) -> None:
    """Verify the already-selected candidate and its training provenance.

    Args:
        saved: Script 26's published tuning summary.
        fingerprints: Current immutable-input hashes.

    Raises:
        ValueError: If the winner, selection design, or training input differs.
    """
    winner = saved.get("selected_candidate", {})
    if any(winner.get(key) != value for key, value in
           {"candidate": "D", **TUNED_PARAMETERS}.items()):
        raise ValueError("Script 26 winner must be Candidate D with depth=30 and leaf=5")
    expected = {
        "selection_metric": "mean Average Precision across 5 spatial folds",
        "candidate_count": 5, "fit_count": 25, "training_rows": 252_569,
        "positives": 6_585, "fixed_parameters": FIXED_PARAMETERS,
        "predictors": list(PREDICTORS), "final_test_accessed": False,
        "saved_spatial_folds_reused": True,
        "input_sha256": fingerprints[TRAINING_PATH.relative_to(ROOT).as_posix()],
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Script 26 selection contract differs: {key}")


def validate_baseline_summary(saved: dict, fingerprints: dict[str, str]) -> None:
    """Ensure the comparison uses the original model on the identical holdout.

    Args:
        saved: Script 25's published metrics and provenance.
        fingerprints: Current immutable-input hashes.

    Raises:
        ValueError: If population, model, metrics, or source identity differs.
    """
    expected = {
        "model": "RandomForestClassifier", "training_rows": 252_569,
        "training_positives": 6_585, "final_test_rows": 60_560,
        "final_test_positives": 821, "final_test_negatives": 59_739,
        "final_test_block_count": 33, "predictors": list(PREDICTORS),
        "preprocessing": PREPROCESSING,
        "model_parameters": build_random_forest_pipeline().named_steps["model"].get_params(),
        "average_precision_definition": "sklearn average_precision_score; not trapezoidal PR area",
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Script 25 comparison contract differs: {key}")
    for path in (MODELING_PATH, SPLIT_PATH, TRAINING_PATH):
        relative = path.relative_to(ROOT).as_posix()
        if saved.get("input_sha256", {}).get(relative) != fingerprints[relative]:
            raise ValueError(f"Baseline source fingerprint differs: {relative}")
    for metric, approximate in (("roc_auc", 0.541339), ("average_precision", 0.014335)):
        value = saved.get(metric)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid baseline {metric}")
        if not np.isclose(value, approximate, rtol=0, atol=0.0000005):
            raise ValueError(f"Unexpected original baseline {metric}")


def build_comparison(baseline: dict, scores: dict) -> tuple[pd.DataFrame, dict]:
    """Compare the two fixed models without using results for model selection.

    Args:
        baseline: Validated original holdout summary.
        scores: Tuned holdout ROC-AUC and shared pr_auc (Average Precision).

    Returns:
        Two comparison rows and tuned-minus-baseline metric differences.
    """
    comparison = pd.DataFrame([
        {"model_version": "baseline", "max_depth": None, "min_samples_leaf": 1,
         "roc_auc": baseline["roc_auc"], "average_precision": baseline["average_precision"]},
        {"model_version": "tuned", **TUNED_PARAMETERS,
         "roc_auc": scores["roc_auc"], "average_precision": scores["pr_auc"]},
    ])
    changes = {
        "roc_auc": scores["roc_auc"] - baseline["roc_auc"],
        "average_precision": scores["pr_auc"] - baseline["average_precision"],
    }
    return comparison, changes


def publish_results(
    predictions: pd.DataFrame, comparison: pd.DataFrame, summary: dict,
    holdout: pd.DataFrame, input_paths: list[Path], fingerprints: dict[str, str],
) -> None:
    """Read back and validate staged evidence before publishing final files.

    Args:
        predictions: Validated held-out probabilities and original identities.
        comparison: Baseline and tuned metric rows.
        summary: JSON-compatible evaluation evidence.
        holdout: Original frozen holdout rows.
        input_paths: Immutable artifacts hashed before the fit.
        fingerprints: Original input hashes.

    Raises:
        ValueError: If metrics, JSON, or input fingerprints change.
        AssertionError: If stored predictions or comparison differ from memory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destinations = [OUTPUT_DIR / name for name in (
        "tuned_holdout_predictions.parquet", "baseline_vs_tuned_holdout.csv",
        "tuned_holdout_summary.json",
    )]
    partials = [path.with_suffix(path.suffix + ".part") for path in destinations]
    try:
        predictions.to_parquet(partials[0], index=False, compression="zstd")
        comparison.to_csv(partials[1], index=False)
        partials[2].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stored = pd.read_parquet(partials[0])
        validate_predictions(stored, holdout)
        pd.testing.assert_frame_equal(stored, predictions, check_exact=True)
        pd.testing.assert_frame_equal(
            pd.read_csv(partials[1], float_precision="round_trip"), comparison, check_exact=True,
        )
        stored_scores = calculate_probability_metrics(
            stored.target.to_numpy(), stored.predicted_probability.to_numpy(),
        )
        if stored_scores != {"roc_auc": summary["roc_auc"], "pr_auc": summary["average_precision"]}:
            raise ValueError("Stored predictions disagree with summary metrics")
        if json.loads(partials[2].read_text(encoding="utf-8")) != summary:
            raise ValueError("Summary read-back differs")
        if fingerprint_inputs(input_paths) != fingerprints:
            raise ValueError("Immutable inputs changed during evaluation")
        # Each replacement is atomic; the three files are not a transaction.
        # Publish the completion summary last and clean partials on failure.
        for temporary, destination in zip(partials, destinations):
            temporary.replace(destination)
    finally:
        for temporary in partials:
            temporary.unlink(missing_ok=True)


# %% Main workflow
def main() -> None:
    """Fit Candidate D once, evaluate the existing holdout, and stop."""
    started = perf_counter()
    # STEP 1 - Load and validate modeling data and the existing frozen splits.
    # Hash the selection/comparison evidence too; neither may change mid-run.
    input_paths = [MODELING_PATH, SPLIT_PATH, TRAINING_PATH, TUNING_PATH, BASELINE_PATH,
                   Path(__file__).resolve(), ROOT / "analysis/25_evaluate_final_east_test.py",
                   ROOT / "src/wildfire_susceptibility/modeling.py"]
    fingerprints = fingerprint_inputs(input_paths)
    modeling = pd.read_parquet(MODELING_PATH)
    splits = pd.read_parquet(SPLIT_PATH)
    validate_source_tables(modeling, splits)

    # STEP 2 - Verify the saved winner; do not select another configuration.
    tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    validate_tuning_selection(tuning, fingerprints)
    pipeline = build_random_forest_pipeline(**TUNED_PARAMETERS)
    parameters = pipeline.named_steps["model"].get_params()
    if any(parameters[key] != value for key, value in {**FIXED_PARAMETERS, **TUNED_PARAMETERS}.items()):
        raise ValueError("Tuned pipeline settings differ from the approved configuration")

    # STEP 3 - Separate populations using only Script 16's saved membership.
    # Compare training with Script 26's artifact; it is validation-only here.
    joined = modeling.merge(splits, on="cell_id", how="inner", validate="one_to_one")
    if len(joined) != 313_129:
        raise ValueError("One-to-one join did not preserve all modeling rows")
    training = joined.loc[joined.split.eq("train")].sort_values("cell_id").reset_index(drop=True)
    holdout = joined.loc[joined.split.eq("final_test")].sort_values("cell_id").reset_index(drop=True)
    validate_populations(training, holdout)
    approved = pd.read_parquet(TRAINING_PATH).sort_values("cell_id").reset_index(drop=True)
    columns = [*IDENTITY, *PREDICTORS, "random_cv_fold", "spatial_cv_fold"]
    pd.testing.assert_frame_equal(training[columns], approved[columns], check_exact=True)
    train_x, holdout_x = training[list(PREDICTORS)], holdout[list(PREDICTORS)]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # STEP 4 - Fit once on ALL training rows with training-only preprocessing.
        # Numeric values stay unscaled; only aspect is imputed, and EVT is nominal.
        print("Contracts passed. Fitting Candidate D once on all 252,569 training rows...", flush=True)
        fit_started = perf_counter()
        pipeline.fit(train_x, training.target)
        fit_seconds = perf_counter() - fit_started
        if list(pipeline.named_steps["model"].classes_) != [0, 1]:
            raise ValueError("Expected class-1 probability in column 1")

        # STEP 5 - Apply the fitted pipeline to the existing eastern holdout.
        predictions = holdout[IDENTITY].copy()
        predictions["predicted_probability"] = pipeline.predict_proba(holdout_x)[:, 1]
        validate_predictions(predictions, holdout)

        # STEP 6 - Calculate probability/ranking metrics without a threshold.
        scores = calculate_probability_metrics(
            holdout.target.to_numpy(), predictions.predicted_probability.to_numpy(),
        )

    # STEP 7 - Read the original baseline strictly for comparison/reporting.
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    validate_baseline_summary(baseline, fingerprints)
    comparison, changes = build_comparison(baseline, scores)

    # STEP 8 - Record provenance, validate staged outputs, and publish safely.
    summary = {
        "model": "RandomForestClassifier",
        "evaluation": "post-hoc evaluation of tuned Random Forest on existing eastern geographic holdout",
        "training_rows": len(training), "training_positives": int(training.target.sum()),
        "holdout_rows": len(holdout), "holdout_positives": int(holdout.target.sum()),
        "holdout_negatives": int(holdout.target.eq(0).sum()),
        "holdout_prevalence": float(holdout.target.mean()),
        "holdout_block_count": int(holdout.spatial_block_id.nunique()),
        "tuned_parameters": TUNED_PARAMETERS, "fixed_parameters": FIXED_PARAMETERS,
        "model_parameters": parameters, "predictors": list(PREDICTORS),
        "preprocessing": PREPROCESSING,
        "roc_auc": scores["roc_auc"], "average_precision": scores["pr_auc"],
        "average_precision_definition": baseline["average_precision_definition"],
        "baseline_roc_auc": baseline["roc_auc"],
        "baseline_average_precision": baseline["average_precision"],
        "change_tuned_minus_baseline": changes,
        "selection_basis": "Candidate D selected in Script 26 using highest mean Average Precision "
                           "across five frozen training-only Spatial-CV folds",
        "final_test_was_previously_observed": True, "threshold_selected": False,
        "additional_tuning_after_script_26": False, "fit_count": 1,
        "all_training_rows_used_for_fitting": True,
        "holdout_used_for_fitting": False, "holdout_used_for_preprocessing_fit": False,
        "frozen_split_membership_reused": True,
        "input_sha256": fingerprints, "input_hashes_verified_before_publication": True,
        "training_artifact_role": "validation only; fit membership comes from frozen split assignments",
        "runtime": {"fit_seconds": round(fit_seconds, 3),
                    "seconds_before_publication": round(perf_counter() - started, 3)},
        "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    publish_results(predictions, comparison, summary, holdout, input_paths, fingerprints)

    # STEP 9 - Report and STOP. No diagnostics, further tuning, or mapping.
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
