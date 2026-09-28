"""Evaluate Random Forest using the five saved spatial CV folds only.

Only the raw training-only Parquet is read. No threshold, random evaluation,
hyperparameter search, or final-test access is part of this experiment.
"""

# %% Imports
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from wildfire_susceptibility.modeling import (
    ASPECT_COLUMNS, CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, PREDICTORS,
    build_random_forest_pipeline, calculate_probability_metrics,
    validate_training_input, run_saved_cv, validate_oof_predictions, publish_results,
)


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/processed/modeling/nm_training_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs/modeling/random_forest_spatial_cv"
EXPECTED_ROWS = 252_569
EXPECTED_FOLDS = [(50435, 1376), (50605, 1321), (50454, 1307), (50408, 1280), (50667, 1301)]
EXPECTED_BLOCKS = [24, 24, 25, 23, 24]
FOLD_COLUMN = "spatial_cv_fold"


# %% Helper functions
def summarize_experiment(oof: pd.DataFrame, folds: pd.DataFrame, diagnostics: list[dict]) -> dict:
    """Summarize combined OOF ranking metrics separately from fold variation.

    Args:
        oof: Complete validated held-out predictions.
        folds: Per-fold probability metrics.
        diagnostics: Tree counts, runtimes, and captured warnings for every fit.

    Returns:
        JSON-compatible experiment metadata and metrics; std uses ddof=1.
    """
    combined = calculate_probability_metrics(oof.target.to_numpy(), oof.predicted_probability.to_numpy())
    return {
        "model": "RandomForestClassifier", "validation_method": "frozen spatial 5-fold CV",
        "spatial_block_count": int(oof.spatial_block_id.nunique()),
        "spatial_block_size_m": 50_000,
        "spatial_blocks_disjoint_all_folds": all(d["spatial_blocks_disjoint"] for d in diagnostics),
        "rows": len(oof), "positives": int(oof.target.sum()), "negatives": int(oof.target.eq(0).sum()),
        "positive_prevalence": float(oof.target.mean()), "fold_count": 5,
        "roc_auc_oof": combined["roc_auc"], "pr_auc_oof": combined["pr_auc"],
        "roc_auc_fold_mean": float(folds.roc_auc.mean()), "roc_auc_fold_std": float(folds.roc_auc.std(ddof=1)),
        "pr_auc_fold_mean": float(folds.pr_auc.mean()), "pr_auc_fold_std": float(folds.pr_auc.std(ddof=1)),
        "fold_std_ddof": 1, "pr_auc_definition": "average_precision_score; not trapezoidal PR area",
        "predictors": list(PREDICTORS),
        "preprocessing": {"continuous": list(CONTINUOUS_COLUMNS), "aspect": list(ASPECT_COLUMNS),
                          "aspect_imputation": "constant 0, pipeline only",
                          "numeric_scaling": "none; numeric predictors remain unscaled",
                          "categorical": list(CATEGORICAL_COLUMNS), "encoding": "OneHotEncoder(handle_unknown='ignore')"},
        "model_parameters": build_random_forest_pipeline().named_steps["model"].get_params(),
        "fold_diagnostics": diagnostics,
        "saved_spatial_folds_reused": True, "final_test_accessed": False,
        "all_oof_probabilities_populated": True,
        "input_path": INPUT_PATH.relative_to(ROOT).as_posix(),
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }


# %% Main workflow
def main() -> None:
    """Run the frozen spatial-CV experiment and publish its probability evidence."""
    started = perf_counter()
    # STEP 1 - Load only script 17's training artifact and enforce its contract.
    # Final-test source files are never opened by this experiment.
    input_hash = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    data = pd.read_parquet(INPUT_PATH).sort_values("cell_id").reset_index(drop=True)
    validate_training_input(
        data, EXPECTED_FOLDS, aspect_missing=38,
        fold_column=FOLD_COLUMN, expected_blocks=EXPECTED_BLOCKS,
    )

    # STEP 2 - Select the explicit predictors, keeping labels and metadata outside X.
    # EVT fraction is continuous; its dominant class identifier is categorical.
    features = data[list(PREDICTORS)]

    # STEP 3 - Prepare missing OOF storage so incomplete coverage is detectable.
    oof = data[["cell_id", "target", "spatial_block_id", "spatial_cv_fold"]].copy()
    oof["predicted_probability"] = np.nan

    # STEP 4 - Fit five fresh pipelines using only saved spatial fold membership.
    # Each forest has 300 trees; numeric values are unscaled and EVT is nominal.
    # Every validation probability comes from a model that excluded that row.
    # Whole frozen 50-km blocks are withheld together; no blocks are recreated.
    folds, diagnostics = run_saved_cv(
        data, features, oof, EXPECTED_FOLDS, build_random_forest_pipeline,
        fold_column=FOLD_COLUMN, expected_blocks=EXPECTED_BLOCKS,
    )

    # STEP 5 - Verify complete held-out coverage and unchanged input data.
    validate_oof_predictions(oof, data, FOLD_COLUMN)
    if hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest() != input_hash:
        raise ValueError("Training input changed during the experiment")

    # STEP 6 - Distinguish combined OOF metrics from mean and sample std across folds.
    # Average Precision evaluates probabilities without selecting a threshold.
    summary = summarize_experiment(oof, folds, diagnostics)
    summary["input_sha256"] = input_hash
    summary["runtime_seconds_before_publication"] = round(perf_counter() - started, 3)

    # STEP 7 - Stage, read back, validate, and publish the three requested artifacts.
    publish_results(oof, folds, summary, data, OUTPUT_DIR, FOLD_COLUMN)

    # STEP 8 - Report evidence and warnings without proceeding to another experiment.
    print(folds.to_string(index=False), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
