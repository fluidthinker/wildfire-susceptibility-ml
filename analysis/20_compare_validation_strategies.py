"""Compare completed Random and Spatial CV experiments without fitting models.

Only summary JSON and fold CSV artifacts are read. Training, OOF, and final-test
row data are not needed. Change the model paths to reuse the comparison for
another model family publishing the same core schema.
"""

# %% Imports
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Write a standalone figure without opening a GUI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wildfire_susceptibility.modeling import (
    build_validation_comparison,
    validate_comparable_summaries,
    validate_fold_metrics,
)


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_KEY = "logistic"
MODEL_LABEL = "Logistic Regression"
EXPECTED_MODEL = "LogisticRegression"
EXPECTED_POPULATION = {"rows": 252_569, "positives": 6_585, "negatives": 245_984, "fold_count": 5}
MODEL_DIR = ROOT / "outputs/modeling"
RANDOM_SUMMARY_PATH = MODEL_DIR / f"{MODEL_KEY}_random_cv/summary_metrics.json"
SPATIAL_SUMMARY_PATH = MODEL_DIR / f"{MODEL_KEY}_spatial_cv/summary_metrics.json"
RANDOM_FOLDS_PATH = MODEL_DIR / f"{MODEL_KEY}_random_cv/fold_metrics.csv"
SPATIAL_FOLDS_PATH = MODEL_DIR / f"{MODEL_KEY}_spatial_cv/fold_metrics.csv"
OUTPUT_DIR = MODEL_DIR / f"{MODEL_KEY}_validation_comparison"
FIGURE_PATH = ROOT / f"outputs/figures/{MODEL_KEY}_random_vs_spatial_cv.png"


# %% Helper functions
def validate_case_study(summary: dict) -> None:
    """Require the approved New Mexico training population for this analysis.

    Args:
        summary: Saved experiment metadata; no source data are opened.

    Raises:
        ValueError: If the selected model or approved population differs.
    """
    if summary["model"] != EXPECTED_MODEL:
        raise ValueError(f"Expected model {EXPECTED_MODEL}")
    for field, expected in EXPECTED_POPULATION.items():
        if summary[field] != expected:
            raise ValueError(f"Expected {field}={expected}, found {summary[field]}")


def create_comparison_figure(comparison: pd.DataFrame) -> plt.Figure:
    """Plot combined OOF metrics on a full zero-to-one scale.

    Args:
        comparison: Validated comparison rows in Random, Spatial order.

    Returns:
        Figure for publication; the caller closes it after saving.
    """
    figure, axis = plt.subplots(figsize=(10, 6))
    positions = np.arange(2)
    width = 0.32
    for index, color in enumerate(("#356A9A", "#C36A35")):
        row = comparison.iloc[index]
        values = [row.roc_auc_oof, row.average_precision_oof]
        bars = axis.bar(
            positions + (index - 0.5) * width, values, width,
            color=color, label=row.validation_strategy,
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5, fontsize=11)
    axis.set_xticks(positions, ["ROC-AUC", "Average Precision"], fontsize=12)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Combined OOF score", fontsize=11)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E4E4E4", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    figure.suptitle(f"{MODEL_LABEL}: Random vs Spatial Cross-Validation", fontsize=16, y=0.96)
    axis.set_title("Combined out-of-fold performance", fontsize=11, color="#555555", pad=16)
    figure.subplots_adjust(top=0.82, bottom=0.19, left=0.09, right=0.97)
    return figure


def build_comparison_summary(
    comparison: pd.DataFrame, changes: dict, contract: dict, random: dict, spatial: dict
) -> dict:
    """Assemble numeric results and source provenance for later review.

    Args:
        comparison: Validated two-row comparison table.
        changes: Signed changes in combined OOF metrics.
        contract: Evidence from fair-comparison validation.
        random: Original Random-CV summary.
        spatial: Original Spatial-CV summary.

    Returns:
        JSON-compatible report, with no causal interpretation.
    """
    metrics = comparison.drop(columns=["model", "validation_strategy"]).to_dict(orient="records")
    sources = {
        "random_summary": RANDOM_SUMMARY_PATH, "spatial_summary": SPATIAL_SUMMARY_PATH,
        "random_folds": RANDOM_FOLDS_PATH, "spatial_folds": SPATIAL_FOLDS_PATH,
    }
    return {
        "model": random["model"], "training_rows": random["rows"],
        "positives": random["positives"], "negatives": random["negatives"],
        "positive_prevalence": random["positive_prevalence"], "fold_count": random["fold_count"],
        "random_cv": metrics[0], "spatial_cv": metrics[1],
        "change_spatial_minus_random": changes, "comparison_contract": contract,
        "fold_std_ddof": 1, "average_precision_definition": random["pr_auc_definition"],
        "input_path": random["input_path"],
        "input_sha256": {"random_cv": random.get("input_sha256"), "spatial_cv": spatial.get("input_sha256")},
        "source_paths": {name: path.relative_to(ROOT).as_posix() for name, path in sources.items()},
        "figure_path": FIGURE_PATH.relative_to(ROOT).as_posix(),
        "models_retrained": False, "final_test_accessed": False,
    }


def publish_comparison(comparison: pd.DataFrame, summary: dict, figure: plt.Figure) -> None:
    """Stage and verify the CSV, JSON, and PNG before replacing final outputs.

    Args:
        comparison: Validated table to serialize without rounding metrics.
        summary: Comparison metadata with provenance.
        figure: Completed grouped bar chart.

    Raises:
        ValueError: If the staged JSON or PNG fails read-back validation.
        AssertionError: If CSV values change during serialization.
    """
    destinations = [OUTPUT_DIR / "validation_comparison.csv", FIGURE_PATH, OUTPUT_DIR / "comparison_summary.json"]
    partials = [path.with_suffix(path.suffix + ".part") for path in destinations]
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        comparison.to_csv(partials[0], index=False)
        figure.savefig(partials[1], format="png", dpi=240, facecolor="white")
        partials[2].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pd.testing.assert_frame_equal(
            pd.read_csv(partials[0], float_precision="round_trip"), comparison, check_exact=True,
        )
        pixels = plt.imread(partials[1], format="png")
        if pixels.ndim != 3 or min(pixels.shape[:2]) < 1000 or not np.isfinite(pixels).all():
            raise ValueError("Figure read-back failed")
        if json.loads(partials[2].read_text(encoding="utf-8")) != summary:
            raise ValueError("Summary read-back differs")
        # Replacements are atomic per file, not a transaction across all three.
        # Publish the summary last, after both deliverables are ready.
        for partial, destination in zip(partials, destinations):
            partial.replace(destination)
    finally:
        for partial in partials:
            partial.unlink(missing_ok=True)


# %% Main workflow
def main() -> None:
    """Compare completed experiments and publish descriptive evidence only."""
    started = perf_counter()
    # STEP 1 - Read completed summaries; no fitting or row-level data are needed.
    random = json.loads(RANDOM_SUMMARY_PATH.read_text(encoding="utf-8"))
    spatial = json.loads(SPATIAL_SUMMARY_PATH.read_text(encoding="utf-8"))

    # STEP 2 - Require the same model, inputs, and preprocessing for a fair comparison.
    contract = validate_comparable_summaries(random, spatial)
    validate_case_study(random)
    validate_case_study(spatial)

    # STEP 3 - Reconcile fold variability and counts with their saved summaries.
    # Sample std describes variation across folds, not uncertainty in OOF scores.
    random_folds = pd.read_csv(RANDOM_FOLDS_PATH, float_precision="round_trip")
    spatial_folds = pd.read_csv(SPATIAL_FOLDS_PATH, float_precision="round_trip")
    validate_fold_metrics(random_folds, random)
    validate_fold_metrics(spatial_folds, spatial)

    # STEP 4 - Preserve stored OOF scores and calculate signed Spatial-minus-Random changes.
    comparison, changes = build_validation_comparison(random, spatial)
    summary = build_comparison_summary(comparison, changes, contract, random, spatial)

    # STEP 5 - Show combined OOF scores with human-readable metric names and a full scale.
    figure = create_comparison_figure(comparison)
    try:
        # STEP 6 - Validate staged files before publishing the comparison artifacts.
        summary["runtime_seconds_before_publication"] = round(perf_counter() - started, 3)
        publish_comparison(comparison, summary, figure)
    finally:
        plt.close(figure)

    # STEP 7 - Report numeric differences; no causal explanation is inferred.
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
