"""Create the portfolio validation-performance chart from saved model results.

This script does not train or evaluate any model. It reads completed Random
Forest evaluation summaries and creates one presentation-quality figure showing
how ROC-AUC changed across:

1. Random cross-validation
2. Spatial cross-validation
3. The original geographically separate test area

The original baseline geographic-test result is used rather than the later
post-hoc tuned evaluation so the three bars describe the same baseline Random
Forest under increasingly demanding validation.

Output:
    outputs/figures/wildfire-validation-performance.png
"""

# %% Imports

import json
from pathlib import Path

import matplotlib.pyplot as plt


# %% Paths and constants

ROOT = Path(__file__).resolve().parents[1]

MODELING_OUTPUT_DIR = ROOT / "outputs" / "modeling"

OUTPUT_PATH = (
    ROOT / "outputs" / "figures" / "wildfire-validation-performance.png"
)

EXPECTED_MODEL_TOKEN = "randomforest"


# %% Helper functions

def load_json(path: Path) -> dict:
    """Read a JSON object from disk.

    Args:
        path: JSON file to read.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If the JSON root is not an object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def normalized_model_name(summary: dict) -> str:
    """Return a compact model-name string for comparison."""
    return str(summary.get("model", "")).lower().replace("_", "").replace(" ", "")


def find_cv_summary(strategy: str) -> tuple[Path, dict]:
    """Find the saved Random Forest CV summary for one validation strategy.

    The project's CV summaries contain:
        model
        validation_method
        roc_auc_oof

    Args:
        strategy: Either ``random`` or ``spatial``.

    Returns:
        Matching summary path and parsed JSON.

    Raises:
        ValueError: If exactly one matching summary cannot be identified.
    """
    matches: list[tuple[Path, dict]] = []

    for path in MODELING_OUTPUT_DIR.rglob("*.json"):
        try:
            summary = load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue

        model_name = normalized_model_name(summary)
        validation_method = str(
            summary.get("validation_method", "")
        ).lower()

        if (
            EXPECTED_MODEL_TOKEN in model_name
            and strategy in validation_method
            and "roc_auc_oof" in summary
        ):
            matches.append((path, summary))

    if len(matches) != 1:
        found = "\n".join(
            f"  - {path.relative_to(ROOT)}"
            for path, _ in matches
        ) or "  none"

        raise ValueError(
            f"Expected exactly one Random Forest {strategy}-CV summary.\n"
            f"Found:\n{found}"
        )

    return matches[0]


def find_original_geographic_test_summary() -> tuple[Path, dict]:
    """Locate Script 25's original baseline geographic-test summary.

    Tuned/post-hoc outputs are deliberately excluded because the portfolio
    comparison should show the baseline Random Forest under three validation
    designs.

    Returns:
        Original geographic-test summary path and parsed JSON.

    Raises:
        ValueError: If one unambiguous baseline test summary cannot be found.
    """
    candidates: list[tuple[Path, dict]] = []

    geographic_tokens = (
        "final",
        "test",
        "holdout",
        "east",
        "eastern",
    )

    excluded_tokens = (
        "tuned",
        "tuning",
        "posthoc",
        "post_hoc",
        "post-hoc",
    )

    for path in MODELING_OUTPUT_DIR.rglob("*.json"):
        path_text = path.as_posix().lower()

        # Ignore the later tuned/post-hoc evaluation.
        if any(token in path_text for token in excluded_tokens):
            continue

        # The original geographic evaluation should be identifiable from its
        # output path or filename.
        if not any(token in path_text for token in geographic_tokens):
            continue

        try:
            summary = load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue

        model_name = normalized_model_name(summary)

        if EXPECTED_MODEL_TOKEN not in model_name:
            continue

        # CV summaries are not final geographic-test summaries.
        if "roc_auc_oof" in summary:
            continue

        try:
            extract_roc_auc(summary)
        except KeyError:
            continue

        candidates.append((path, summary))

    if len(candidates) != 1:
        found = "\n".join(
            f"  - {path.relative_to(ROOT)}"
            for path, _ in candidates
        ) or "  none"

        raise ValueError(
            "Could not uniquely identify the original baseline geographic-test "
            "summary.\n"
            "Candidate files found:\n"
            f"{found}\n\n"
            "If Script 25 used an unusual output directory name, add one of "
            "its identifying words to geographic_tokens."
        )

    return candidates[0]


def extract_roc_auc(summary: dict) -> float:
    """Extract ROC-AUC from a final-test summary.

    Script 25 may store the final-test metric at the top level or inside a
    small nested result dictionary. Search only for common ROC-AUC field names;
    do not calculate a new metric.

    Args:
        summary: Parsed summary JSON.

    Returns:
        Saved ROC-AUC value.

    Raises:
        KeyError: If no supported ROC-AUC field is present.
        ValueError: If the value is outside [0, 1].
    """
    possible_keys = (
        "roc_auc",
        "roc_auc_score",
        "final_test_roc_auc",
        "holdout_roc_auc",
        "test_roc_auc",
    )

    # First inspect the top level.
    for key in possible_keys:
        if key in summary:
            value = float(summary[key])

            if not 0 <= value <= 1:
                raise ValueError(f"Invalid ROC-AUC value: {value}")

            return value

    # Then inspect one level of nested dictionaries.
    for nested in summary.values():
        if not isinstance(nested, dict):
            continue

        for key in possible_keys:
            if key in nested:
                value = float(nested[key])

                if not 0 <= value <= 1:
                    raise ValueError(f"Invalid ROC-AUC value: {value}")

                return value

    raise KeyError("No supported ROC-AUC field found")


def validate_cv_summaries(
    random_summary: dict,
    spatial_summary: dict,
) -> None:
    """Verify the two CV summaries represent comparable experiments."""
    required_keys = {
        "model",
        "rows",
        "positives",
        "negatives",
        "roc_auc_oof",
    }

    for name, summary in (
        ("Random CV", random_summary),
        ("Spatial CV", spatial_summary),
    ):
        missing = required_keys - summary.keys()

        if missing:
            raise ValueError(
                f"{name} summary missing required fields: {sorted(missing)}"
            )

    if random_summary["model"] != spatial_summary["model"]:
        raise ValueError("Random and spatial CV use different models")

    for key in ("rows", "positives", "negatives"):
        if random_summary[key] != spatial_summary[key]:
            raise ValueError(
                f"Random and spatial CV differ on {key}"
            )

    if "predictors" in random_summary and "predictors" in spatial_summary:
        if random_summary["predictors"] != spatial_summary["predictors"]:
            raise ValueError("Random and spatial CV use different predictors")

    if "input_sha256" in random_summary and "input_sha256" in spatial_summary:
        if random_summary["input_sha256"] != spatial_summary["input_sha256"]:
            raise ValueError("Random and spatial CV use different input data")


def create_chart(
    labels: list[str],
    values: list[float],
) -> None:
    """Create and save the portfolio validation-performance chart."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, values, width=0.62)

    # Print the exact saved metric above each bar.
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title(
        "Model Performance Across Validation Strategies",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )

    ax.text(
        0.5,
        1.01,
        "Random Forest ROC-AUC",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    ax.set_ylabel("ROC-AUC", fontsize=12)

    # ROC-AUC is bounded from 0 to 1. Starting at zero avoids exaggerating
    # the visual differences between validation strategies.
    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        linewidth=0.8,
        alpha=0.3,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Plain-language context for portfolio readers.
    fig.text(
        0.5,
        0.015,
        (
            "Performance dropped as the model was tested on increasingly "
            "geographically separate data."
        ),
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=[0.04, 0.06, 0.98, 0.94])

    temporary_path = OUTPUT_PATH.with_name(
        OUTPUT_PATH.stem + ".tmp.png"
    )

    try:
        fig.savefig(
            temporary_path,
            dpi=200,
            bbox_inches="tight",
        )
        temporary_path.replace(OUTPUT_PATH)
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)


# %% Main workflow

def main() -> None:
    """Load completed evaluation results and create the portfolio chart."""

    # STEP 1 — Load the saved Random Forest random-CV result.
    # No model is retrained and no metric is recalculated.
    random_path, random_summary = find_cv_summary("random")

    # STEP 2 — Load the saved Random Forest spatial-CV result.
    spatial_path, spatial_summary = find_cv_summary("spatial")

    # STEP 3 — Confirm that the two CV experiments are directly comparable.
    validate_cv_summaries(random_summary, spatial_summary)

    # STEP 4 — Locate Script 25's original geographic-test result.
    # Later tuned/post-hoc evaluations are deliberately excluded.
    test_path, test_summary = find_original_geographic_test_summary()

    # STEP 5 — Extract the three saved baseline Random Forest ROC-AUC values.
    random_roc_auc = float(random_summary["roc_auc_oof"])
    spatial_roc_auc = float(spatial_summary["roc_auc_oof"])
    geographic_test_roc_auc = extract_roc_auc(test_summary)

    labels = [
        "Random CV",
        "Spatial CV",
        "Separate Geographic Test",
    ]

    values = [
        random_roc_auc,
        spatial_roc_auc,
        geographic_test_roc_auc,
    ]

    # STEP 6 — Create the presentation figure.
    create_chart(labels, values)

    # STEP 7 — Report exactly which saved artifacts supplied the figure.
    print("\nVALIDATION PERFORMANCE CHART")
    print("----------------------------")
    print(
        f"Random CV:              {random_roc_auc:.6f}  "
        f"({random_path.relative_to(ROOT)})"
    )
    print(
        f"Spatial CV:             {spatial_roc_auc:.6f}  "
        f"({spatial_path.relative_to(ROOT)})"
    )
    print(
        f"Separate geographic test: {geographic_test_roc_auc:.6f}  "
        f"({test_path.relative_to(ROOT)})"
    )
    print()
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("\nNo model was trained and no evaluation metric was recalculated.")


# %% Run script

if __name__ == "__main__":
    main()
# %%
