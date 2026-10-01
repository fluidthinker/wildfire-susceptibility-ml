"""Select one of five frozen forests using training-only Spatial CV.

This is post-hoc learning: the original eastern test result was already seen.
Only the training artifact is opened; no test result informs this selection.
"""

# %% Imports
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from wildfire_susceptibility.modeling import (
    PREDICTORS, build_random_forest_pipeline, calculate_probability_metrics,
    fit_validation_fold, validate_fold_partition, validate_training_input,
)


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/processed/modeling/nm_training_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs/modeling/random_forest_spatial_tuning"
EXPECTED_FOLDS = [(50435, 1376), (50605, 1321), (50454, 1307), (50408, 1280), (50667, 1301)]
EXPECTED_BLOCKS = [24, 24, 25, 23, 24]
FOLD_COLUMN = "spatial_cv_fold"
# Immutable definitions established before any scores are observed.
CANDIDATES = (("A", None, 1), ("B", None, 5), ("C", None, 10), ("D", 30, 5), ("E", 15, 5))
FIXED_PARAMETERS = {
    "n_estimators": 300, "random_state": 42, "n_jobs": -1,
    "class_weight": None, "criterion": "gini", "bootstrap": True,
    "max_features": "sqrt", "min_samples_split": 2,
}
TIE_BREAK_ORDER = [
    "higher mean Average Precision", "higher mean ROC-AUC",
    "larger min_samples_leaf", "smaller finite max_depth", "lower candidate letter",
]


# %% Helper functions
def evaluate_candidates(data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate the fixed candidates sequentially on whole held-out blocks.

    Args:
        data: Validated training artifact with frozen spatial fold labels.

    Returns:
        Twenty-five fold records and warning/runtime diagnostics. Predictions
        live only for one fold; preprocessing is fitted afresh for every fit.

    Raises:
        ValueError: If a partition or the requested fixed parameters differ.
    """
    features = data[list(PREDICTORS)]
    records, diagnostics = [], []
    for candidate, max_depth, min_samples_leaf in CANDIDATES:
        factory = partial(build_random_forest_pipeline,
                          max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        parameters = factory().named_steps["model"].get_params()
        if any(parameters[key] != value for key, value in FIXED_PARAMETERS.items()):
            raise ValueError("Shared forest defaults differ from frozen tuning parameters")
        for fold in range(5):
            validation = data[FOLD_COLUMN].eq(fold)
            validate_fold_partition(
                data, validation, fold, EXPECTED_FOLDS, FOLD_COLUMN, EXPECTED_BLOCKS,
            )
            print(f"Candidate {candidate}, spatial fold {fold}: fitting...", flush=True)
            probabilities, diagnostic = fit_validation_fold(
                features, data.target, validation, fold, factory,
            )
            labels = data.loc[validation, "target"]
            scores = calculate_probability_metrics(labels.to_numpy(), probabilities)
            records.append({
                "candidate": candidate, "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf, "fold": fold,
                "blocks": int(data.loc[validation, "spatial_block_id"].nunique()),
                "rows": len(labels), "positives": int(labels.sum()),
                "positive_prevalence": float(labels.mean()),
                "roc_auc": scores["roc_auc"], "average_precision": scores["pr_auc"],
                "runtime_seconds": diagnostic["runtime_seconds"],
            })
            diagnostics.append({"candidate": candidate, **diagnostic})
            print(f"Candidate {candidate}, fold {fold}: {scores}; "
                  f"runtime={diagnostic['runtime_seconds']} seconds", flush=True)
    return pd.DataFrame(records), diagnostics


def summarize_candidates(folds: pd.DataFrame) -> pd.DataFrame:
    """Calculate equally weighted fold means and sample standard deviations.

    Args:
        folds: One probability-metric record per candidate and spatial fold.

    Returns:
        Five candidate summaries; standard deviations use ddof=1.
    """
    records = []
    for candidate, max_depth, min_samples_leaf in CANDIDATES:
        subset = folds.loc[folds.candidate.eq(candidate)]
        records.append({
            "candidate": candidate, "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "roc_auc_fold_mean": float(subset.roc_auc.mean()),
            "roc_auc_fold_std": float(subset.roc_auc.std(ddof=1)),
            "average_precision_fold_mean": float(subset.average_precision.mean()),
            "average_precision_fold_std": float(subset.average_precision.std(ddof=1)),
            "total_runtime_seconds": float(subset.runtime_seconds.sum()),
        })
    return pd.DataFrame(records)


def select_candidate(summary: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Select highest mean AP, applying the frozen order only for exact ties.

    Args:
        summary: Five complete candidate summaries at full float precision.

    Returns:
        Summary with one selected flag and whether the best AP was tied.
        CSV stores full round-trip precision; display rounding is not selection.
    """
    # Missing depth means unlimited complexity, so finite depths sort first.
    ranking = summary.assign(depth_order=summary.max_depth.fillna(np.inf)).sort_values(
        ["average_precision_fold_mean", "roc_auc_fold_mean", "min_samples_leaf",
         "depth_order", "candidate"],
        ascending=[False, False, False, True, True],
    )
    best_ap = ranking.iloc[0].average_precision_fold_mean
    tie_required = bool(summary.average_precision_fold_mean.eq(best_ap).sum() > 1)
    selected = summary.copy()
    selected["selected"] = selected.candidate.eq(ranking.iloc[0].candidate)
    return selected, tie_required


def validate_tuning_evidence(folds: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Require complete, finite evidence and a reproducible single winner.

    Args:
        folds: In-memory or read-back fold records.
        summary: In-memory or read-back candidate summaries.

    Raises:
        ValueError: If coverage, candidate definitions, populations, or scores fail.
        AssertionError: If summaries or selection disagree with fold evidence.
    """
    expected_pairs = {(candidate, fold) for candidate, _, _ in CANDIDATES for fold in range(5)}
    if len(folds) != 25 or set(zip(folds.candidate, folds.fold)) != expected_pairs:
        raise ValueError("Expected exactly 25 unique candidate/fold fits")
    for candidate, max_depth, min_samples_leaf in CANDIDATES:
        subset = folds.loc[folds.candidate.eq(candidate)].sort_values("fold")
        depth_matches = subset.max_depth.isna() if max_depth is None else subset.max_depth.eq(max_depth)
        if not depth_matches.all() or not subset.min_samples_leaf.eq(min_samples_leaf).all():
            raise ValueError("Frozen candidate parameters changed")
        if list(zip(subset.rows, subset.positives)) != EXPECTED_FOLDS or subset.blocks.tolist() != EXPECTED_BLOCKS:
            raise ValueError("Frozen held-out populations changed")
        if not np.allclose(subset.positive_prevalence, subset.positives / subset.rows, rtol=0, atol=1e-15):
            raise ValueError("Incorrect fold prevalence")
    scores = folds[["roc_auc", "average_precision"]].to_numpy()
    runtimes = folds.runtime_seconds.to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probability metrics")
    if not np.isfinite(runtimes).all() or (runtimes < 0).any():
        raise ValueError("Invalid runtimes")
    expected, _ = select_candidate(summarize_candidates(folds))
    pd.testing.assert_frame_equal(summary, expected, check_exact=True)
    if len(summary) != 5 or summary.selected.sum() != 1:
        raise ValueError("Exactly one of five candidates must be selected")


def build_tuning_summary(
    data: pd.DataFrame, summary: pd.DataFrame, diagnostics: list[dict],
    input_hash: str, tie_required: bool,
) -> dict:
    """Record training-only provenance and the selected configuration.

    Args:
        data: Validated training population.
        summary: Candidate metrics with one selected row.
        diagnostics: All 25 fits' warnings and runtimes.
        input_hash: SHA-256 verified unchanged after tuning.
        tie_required: Whether top mean AP values were exactly tied.

    Returns:
        JSON-compatible evidence without any final-test metrics.
    """
    winner = summary.loc[summary.selected].iloc[0]
    return {
        "experiment": "training-only Random Forest hyperparameter tuning using frozen Spatial CV",
        "selection_metric": "mean Average Precision across 5 spatial folds",
        "candidate_count": 5, "fit_count": len(diagnostics),
        "training_rows": len(data), "positives": int(data.target.sum()),
        "negatives": int(data.target.eq(0).sum()),
        "spatial_block_count": int(data.spatial_block_id.nunique()),
        "input_path": INPUT_PATH.relative_to(ROOT).as_posix(), "input_sha256": input_hash,
        "input_sha256_unchanged": True, "fixed_parameters": FIXED_PARAMETERS,
        "candidates": [{"candidate": c, "max_depth": d, "min_samples_leaf": leaf}
                       for c, d, leaf in CANDIDATES],
        "selected_candidate": {
            "candidate": winner.candidate,
            "max_depth": None if pd.isna(winner.max_depth) else int(winner.max_depth),
            "min_samples_leaf": int(winner.min_samples_leaf),
            "mean_average_precision": float(winner.average_precision_fold_mean),
            "std_average_precision": float(winner.average_precision_fold_std),
            "mean_roc_auc": float(winner.roc_auc_fold_mean),
            "std_roc_auc": float(winner.roc_auc_fold_std),
        },
        "tie_break_required": tie_required, "tie_break_order": TIE_BREAK_ORDER,
        "stored_precision": "full round-trip float precision; exact equality for ties",
        "fold_std_ddof": 1, "saved_spatial_folds_reused": True,
        "spatial_blocks_disjoint_all_fits": True, "final_test_accessed": False,
        "post_hoc_context": "The original eastern final-test result was observed before this tuning "
                            "exercise; candidate selection itself used training-only Spatial CV.",
        "predictors": list(PREDICTORS),
        "preprocessing": {"numeric_scaling": "none", "aspect_imputation": "constant 0 inside pipeline",
                          "evt_encoding": "OneHotEncoder(handle_unknown='ignore')",
                          "fit_scope": "four training folds separately for each fit"},
        "fold_diagnostics": diagnostics,
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }


def publish_tuning_results(folds: pd.DataFrame, summary: pd.DataFrame, evidence: dict) -> None:
    """Stage, read back, and validate all evidence before publishing final files.

    Args:
        folds: Twenty-five validated fold records.
        summary: Five validated candidate summaries.
        evidence: JSON-compatible provenance and selection evidence.

    Raises:
        ValueError: If JSON changes during serialization.
        AssertionError: If CSV contents change during serialization.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destinations = [OUTPUT_DIR / name for name in
                    ("candidate_fold_metrics.csv", "candidate_summary.csv", "tuning_summary.json")]
    partials = [path.with_suffix(path.suffix + ".part") for path in destinations]
    try:
        folds.to_csv(partials[0], index=False)
        summary.to_csv(partials[1], index=False)
        partials[2].write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stored_folds = pd.read_csv(partials[0], float_precision="round_trip")
        stored_summary = pd.read_csv(partials[1], float_precision="round_trip")
        pd.testing.assert_frame_equal(stored_folds, folds, check_exact=True)
        pd.testing.assert_frame_equal(stored_summary, summary, check_exact=True)
        validate_tuning_evidence(stored_folds, stored_summary)
        if json.loads(partials[2].read_text(encoding="utf-8")) != evidence:
            raise ValueError("Tuning summary read-back differs")
        # Individual replacements are atomic, not a multi-file transaction.
        # Publish the JSON completion record last; remove partials on failure.
        for temporary, destination in zip(partials, destinations):
            temporary.replace(destination)
    finally:
        for temporary in partials:
            temporary.unlink(missing_ok=True)


# %% Main workflow
def main() -> None:
    """Run the small frozen experiment, publish evidence, and stop."""
    started = perf_counter()
    # STEP 1 - Load and validate the frozen training-only dataset.
    # No final-test data or earlier model results are opened.
    input_hash = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    data = pd.read_parquet(INPUT_PATH).sort_values("cell_id").reset_index(drop=True)
    validate_training_input(data, EXPECTED_FOLDS, aspect_missing=38,
                            fold_column=FOLD_COLUMN, expected_blocks=EXPECTED_BLOCKS)

    # STEP 2 - Use the five immutable candidates defined above before tuning.
    # Depth limits tree growth; leaf size limits how locally a tree can fit.
    print(f"Training: {len(data):,} rows, {int(data.target.sum()):,} positives, "
          f"{data.spatial_block_id.nunique()} blocks. Frozen candidates: {CANDIDATES}", flush=True)

    # STEP 3 - Evaluate all five candidates on the same five saved spatial folds.
    # Sequential fits bound memory; n_jobs=-1 parallelizes trees within a fit.
    folds, diagnostics = evaluate_candidates(data)

    # STEP 4 - Summarize fold means and sample standard deviations.
    # Each geographic fold contributes equally to the selection metric.
    summary = summarize_candidates(folds)

    # STEP 5 - Select by mean AP using the documented exact-tie order.
    summary, tie_required = select_candidate(summary)

    # STEP 6 - Validate evidence and the unchanged input before safe publication.
    validate_tuning_evidence(folds, summary)
    if hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest() != input_hash:
        raise ValueError("Training input changed during tuning")
    evidence = build_tuning_summary(data, summary, diagnostics, input_hash, tie_required)
    evidence["runtime_seconds_before_publication"] = round(perf_counter() - started, 3)
    publish_tuning_results(folds, summary, evidence)

    # STEP 7 - Report the winner and STOP; no automatic final-test evaluation.
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(evidence, indent=2, allow_nan=False), flush=True)
    print(f"Outputs: {OUTPUT_DIR}", flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
