"""Describe environmental differences in the existing eastern geographic holdout.

The holdout is a selected geographic band, not all of eastern New Mexico.
No models, predictions, significance tests, or causal explanations are produced.
"""

# %% Imports
import json
from pathlib import Path
import runpy
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from wildfire_susceptibility.modeling import ASPECT_COLUMNS, CONTINUOUS_COLUMNS


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
SPLIT_PATH = ROOT / "data/processed/modeling/nm_modeling_splits.parquet"
OUTPUT_DIR = ROOT / "outputs/diagnostics/east_holdout"
FIGURE_DIR = ROOT / "outputs/figures"
NUMERIC_PREDICTORS = (*CONTINUOUS_COLUMNS, *ASPECT_COLUMNS)
PRIMARY = ("elevation_mean", "slope_mean", "annual_precip_mean")
AXIS_LABELS = ("Mean elevation (m)", "Mean slope (degrees)", "Mean annual precipitation (mm)")
POPULATIONS = ("training", "eastern_holdout")
DISPLAY_LABELS = ("Training", "Eastern geographic holdout")
COLORS = ("#286A91", "#CD6634")

# Follow Script 27's reuse of case-study contracts. The default run_path name
# is not __main__: Script 25's workflow and model fit are never executed.
_contracts = runpy.run_path(str(ROOT / "analysis/25_evaluate_final_east_test.py"))
validate_source_tables = _contracts["validate_source_tables"]
validate_populations = _contracts["validate_populations"]
fingerprint_inputs = _contracts["fingerprint_inputs"]


# %% Helper functions
def summarize_numeric(populations: dict[str, pd.DataFrame], columns: tuple) -> pd.DataFrame:
    """Describe each feature using available observations without imputation.

    Args:
        populations: Named training/holdout tables, optionally positive-only.
        columns: Numeric features to summarize; EVT class codes are excluded.

    Returns:
        Long-form statistics with population rows, valid counts and missingness.
        Standard deviations use ddof=1; quantiles use linear interpolation.
    """
    records = []
    for population, data in populations.items():
        for column in columns:
            # Accumulate in float64 even for float32 source columns; the
            # source table itself and its missing values remain unchanged.
            values = data[column].dropna().astype("float64")
            quantiles = values.quantile([0.05, 0.25, 0.75, 0.95])
            records.append({
                "predictor": column, "population": population, "rows": len(data),
                "nonmissing_count": len(values), "missing_count": int(data[column].isna().sum()),
                "mean": float(values.mean()), "median": float(values.median()),
                "std": float(values.std(ddof=1)), "p05": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]), "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
            })
    return pd.DataFrame(records)


def add_standardized_mean_difference(statistics: pd.DataFrame) -> pd.DataFrame:
    """Express the holdout-minus-training mean difference in pooled SD units.

    Args:
        statistics: Two available-case summaries per numeric predictor.

    Returns:
        Summary with repeated SMD and pooled SD for each population row.
        Pooling weights sample variances by nonmissing degrees of freedom:
        sqrt(((n_train-1)*s_train^2 + (n_holdout-1)*s_holdout^2)/(n_train+n_holdout-2)).
        An undefined SMD (zero pooled SD) is left missing, never imputed.
    """
    result = statistics.copy()
    for predictor, group in statistics.groupby("predictor", sort=False):
        paired = group.set_index("population")
        train, holdout = paired.loc["training"], paired.loc["eastern_holdout"]
        degrees = train.nonmissing_count + holdout.nonmissing_count - 2
        variance_sum = ((train.nonmissing_count - 1) * train["std"] ** 2
                        + (holdout.nonmissing_count - 1) * holdout["std"] ** 2)
        pooled = np.sqrt(variance_sum / degrees)
        smd = (holdout["mean"] - train["mean"]) / pooled if pooled > 0 else np.nan
        mask = result.predictor.eq(predictor)
        result.loc[mask, "pooled_standard_deviation"] = pooled
        result.loc[mask, "standardized_mean_difference"] = smd
    return result


def compare_evt(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare the union of observed nominal EVT codes, including absent classes.

    Args:
        populations: Validated training and eastern holdout tables.

    Returns:
        Counts, within-population proportions, and holdout-minus-training
        differences. Zero denotes absence, not a missing environmental value.
    """
    counts = pd.concat({name: data.evt_dominant_class.value_counts()
                        for name, data in populations.items()}, axis=1).fillna(0).astype("int64")
    counts = counts.sort_index().rename_axis("evt_dominant_class")
    result = pd.DataFrame(index=counts.index)
    for population in POPULATIONS:
        result[f"{population}_count"] = counts[population]
        result[f"{population}_proportion"] = counts[population] / len(populations[population])
    result["difference_in_proportion"] = result.eastern_holdout_proportion - result.training_proportion
    if not np.allclose([result.training_proportion.sum(), result.eastern_holdout_proportion.sum()], 1):
        raise ValueError("EVT proportions must sum to one within each population")
    return result.reset_index()


def plot_continuous(populations: dict[str, pd.DataFrame]) -> plt.Figure:
    """Plot comparable distributions with common full-range bins per predictor.

    Args:
        populations: Validated original populations; no sampling or imputation.

    Returns:
        Three-panel figure; each density integrates to one independently.
    """
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    for axis, predictor, label in zip(axes, PRIMARY, AXIS_LABELS):
        combined = pd.concat([data[predictor] for data in populations.values()]).dropna()
        # Shared edges retain the entire observed range. Forty-five bins are
        # a display choice, not a scientific threshold or a resampling rule.
        edges = np.linspace(combined.min(), combined.max(), 46)
        for name, display, color in zip(POPULATIONS, DISPLAY_LABELS, COLORS):
            axis.hist(populations[name][predictor].dropna(), bins=edges, density=True,
                      histtype="step", linewidth=2, color=color, label=display)
        axis.set(xlabel=label, ylabel="Probability density")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Environmental distributions | Training vs eastern geographic holdout", fontsize=14)
    return figure


def plot_evt(evt: pd.DataFrame) -> plt.Figure:
    """Show the ten most frequent combined-population codes and remaining mass.

    Args:
        evt: Complete composition table with numeric class identifiers.

    Returns:
        Grouped proportion chart; ties in combined counts use ascending code.
    """
    ranked = evt.assign(combined=evt.training_count + evt.eastern_holdout_count).sort_values(
        ["combined", "evt_dominant_class"], ascending=[False, True],
    )
    top = ranked.head(10)
    labels = top.evt_dominant_class.astype(str).tolist()
    columns = ["training_proportion", "eastern_holdout_proportion"]
    proportions = top[columns].to_numpy()
    if len(ranked) > 10:
        labels.append("Other")
        proportions = np.vstack([proportions, ranked.iloc[10:][columns].sum().to_numpy()])
    if not np.allclose(proportions.sum(axis=0), 1):
        raise ValueError("Top-ten plus Other must preserve population proportions")
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    positions = np.arange(len(labels))
    for index, (label, color) in enumerate(zip(DISPLAY_LABELS, COLORS)):
        axis.barh(positions + (index - 0.5) * 0.36, proportions[:, index],
                  height=0.36, label=label, color=color)
    axis.set(yticks=positions, yticklabels=labels, xlabel="Proportion of population cells",
             ylabel="LANDFIRE EVT class code", title="EVT composition | Top 10 combined-frequency classes + Other")
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def build_summary(populations: dict, continuous: pd.DataFrame, evt: pd.DataFrame,
                  positive: pd.DataFrame, prevalence: dict) -> dict:
    """Collect descriptive evidence without assigning a cause to model behavior.

    Args:
        populations: Original validated population tables.
        continuous: Numeric statistics and descriptive SMDs.
        evt: Complete nominal-code composition comparison.
        positive: Numeric summaries restricted to target==1 cells.
        prevalence: Population prevalence and their ratio.

    Returns:
        JSON-compatible diagnostic evidence with explicit methodology.
    """
    shifts = continuous.loc[continuous.population.eq("training"),
                            ["predictor", "standardized_mean_difference"]].copy()
    shifts["absolute_smd"] = shifts.standardized_mean_difference.abs()
    shifts = shifts.sort_values(["absolute_smd", "predictor"], ascending=[False, True])
    differences = evt.assign(absolute_difference=evt.difference_in_proportion.abs()).sort_values(
        ["absolute_difference", "evt_dominant_class"], ascending=[False, True],
    )
    unseen = evt.loc[evt.training_count.eq(0), "evt_dominant_class"].tolist()
    absent = evt.loc[evt.eastern_holdout_count.eq(0), "evt_dominant_class"].tolist()
    notes = []
    for predictor in PRIMARY:
        rows = continuous.loc[continuous.predictor.eq(predictor)].set_index("population")
        direction = "lower" if rows.loc["eastern_holdout", "mean"] < rows.loc["training", "mean"] else "higher"
        notes.append(f"The eastern geographic holdout had {direction} mean {predictor} than training.")
    notes.extend([
        "EVT frequencies differ between the two populations; codes are treated as nominal categories.",
        "No holdout EVT classes were absent from training." if not unseen else
        f"{len(unseen)} holdout EVT classes were absent from training; their one-hot block would encode as zeros.",
        "The positive-class prevalence was lower in the holdout." if prevalence["prevalence_ratio"] < 1 else
        "The positive-class prevalence was not lower in the holdout.",
        "These summaries describe a selected eastern geographic holdout band, not all of eastern New Mexico.",
        "Environmental and prevalence differences do not establish the cause of poorer model performance.",
    ])
    # DataFrame JSON conversion maps undefined SMD to null, without modifying data.
    return {
        "training_rows": len(populations["training"]), "holdout_rows": len(populations["eastern_holdout"]),
        "training_positives": int(populations["training"].target.sum()),
        "holdout_positives": int(populations["eastern_holdout"].target.sum()),
        "holdout_negatives": int(populations["eastern_holdout"].target.eq(0).sum()),
        "training_block_count": int(populations["training"].spatial_block_id.nunique()),
        "holdout_block_count": int(populations["eastern_holdout"].spatial_block_id.nunique()),
        **prevalence,
        "continuous_predictors": list(NUMERIC_PREDICTORS),
        "largest_absolute_smd_predictors": json.loads(shifts.head(5).to_json(orient="records", double_precision=15)),
        "smd_definition": "(holdout_mean-training_mean)/sqrt(((n_train-1)*sd_train^2 + "
                          "(n_holdout-1)*sd_holdout^2)/(n_train+n_holdout-2)); available-case n; sample SD",
        "smd_interpretation": "Descriptive mean separation only; no significance tests or scientific cutoffs",
        "missingness_policy": "Available-case statistics; aspect missingness preserved; no imputation",
        "missing_counts": {name: data[list(NUMERIC_PREDICTORS)].isna().sum().to_dict()
                           for name, data in populations.items()},
        "evt_class_count_training": int(evt.training_count.gt(0).sum()),
        "evt_class_count_holdout": int(evt.eastern_holdout_count.gt(0).sum()),
        "evt_classes_unseen_in_training": unseen, "evt_classes_absent_in_holdout": absent,
        "top_evt_proportion_differences": differences.head(10).to_dict(orient="records"),
        "evt_label_policy": "Numeric codes retained; no vegetation-name interpretation attempted",
        "positive_class_summary": positive.to_dict(orient="records"),
        "positive_class_definition": "Observed target == 1, not thresholded model true positives",
        "interpretation_notes": notes, "models_fitted": 0, "threshold_selected": False,
    }


def publish_diagnostics(tables: dict, figures: dict, summary: dict,
                        input_paths: list[Path], fingerprints: dict) -> None:
    """Stage and read back all six artifacts before publishing the summary last.

    Args:
        tables: CSV basenames mapped to completed diagnostic tables.
        figures: PNG basenames mapped to Matplotlib figures.
        summary: JSON-compatible evidence and methodology.
        input_paths: Immutable sources and code hashed before analysis.
        fingerprints: Original SHA-256 digests.

    Raises:
        ValueError: If a figure, JSON round trip, or input fingerprint is invalid.
        AssertionError: If table serialization changes any stored value.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    destinations = [*[OUTPUT_DIR / name for name in tables],
                    *[FIGURE_DIR / name for name in figures], OUTPUT_DIR / "diagnostic_summary.json"]
    staged = {path: path.with_suffix(path.suffix + ".part") for path in destinations}
    try:
        for name, table in tables.items():
            path = staged[OUTPUT_DIR / name]
            table.to_csv(path, index=False)
            pd.testing.assert_frame_equal(pd.read_csv(path, float_precision="round_trip"), table, check_exact=True)
        for name, figure in figures.items():
            path = staged[FIGURE_DIR / name]
            figure.savefig(path, format="png", dpi=180, facecolor="white")
            pixels = plt.imread(path, format="png")
            if pixels.ndim != 3 or min(pixels.shape[:2]) < 500 or not np.isfinite(pixels).all():
                raise ValueError(f"Invalid diagnostic figure: {name}")
        summary_path = staged[OUTPUT_DIR / "diagnostic_summary.json"]
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise ValueError("Summary read-back differs")
        if fingerprint_inputs(input_paths) != fingerprints:
            raise ValueError("Immutable inputs changed during diagnostics")
        # Replacement is atomic per file, not a six-file transaction.
        for destination, temporary in staged.items():
            temporary.replace(destination)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for figure in figures.values():
            plt.close(figure)


# %% Main workflow
def main() -> None:
    """Compare the unchanged populations, publish descriptive evidence, and stop."""
    started = perf_counter()
    # STEP 1 - Load the authoritative modeling table and frozen split membership.
    input_paths = [MODELING_PATH, SPLIT_PATH, Path(__file__).resolve(),
                   ROOT / "analysis/25_evaluate_final_east_test.py",
                   ROOT / "src/wildfire_susceptibility/modeling.py"]
    fingerprints = fingerprint_inputs(input_paths)
    modeling, splits = pd.read_parquet(MODELING_PATH), pd.read_parquet(SPLIT_PATH)

    # STEP 2 - Validate population contracts without changing missingness or folds.
    validate_source_tables(modeling, splits)
    joined = modeling.merge(splits, on="cell_id", how="inner", validate="one_to_one")
    if len(joined) != 313_129:
        raise ValueError("Join changed the eligible population")
    training = joined.loc[joined.split.eq("train")].copy()
    holdout = joined.loc[joined.split.eq("final_test")].copy()
    validate_populations(training, holdout)
    populations = dict(zip(POPULATIONS, (training, holdout)))

    # STEP 3 - Compare all nine numeric predictors; EVT codes remain categorical.
    continuous = add_standardized_mean_difference(summarize_numeric(populations, NUMERIC_PREDICTORS))

    # STEP 4 - Compare observed target prevalence without explaining model scores.
    prevalence = {"training_prevalence": float(training.target.mean()),
                  "holdout_prevalence": float(holdout.target.mean())}
    prevalence["prevalence_ratio"] = prevalence["holdout_prevalence"] / prevalence["training_prevalence"]

    # STEP 5 - Compare EVT proportions, retaining codes absent from either side.
    evt = compare_evt(populations)

    # STEP 6 - Describe observed positive cells for the three main predictors only.
    positive_populations = {name: data.loc[data.target.eq(1)] for name, data in populations.items()}
    positive = summarize_numeric(positive_populations, PRIMARY)[
        ["predictor", "population", "rows", "mean", "median", "p05", "p95"]]

    # STEP 7 - Build two compact figures; no geographic map or model is created.
    figures = {"training_vs_east_holdout_environment.png": plot_continuous(populations),
               "training_vs_east_holdout_evt.png": plot_evt(evt)}

    # STEP 8 - Publish tables, figures and descriptive summary after read-back QA.
    summary = build_summary(populations, continuous, evt, positive, prevalence)
    summary["input_sha256"] = fingerprints
    summary["input_hashes_verified_before_publication"] = True
    summary["runtime_seconds_before_publication"] = round(perf_counter() - started, 3)
    summary["versions"] = {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    tables = {"continuous_predictor_comparison.csv": continuous,
              "evt_composition_comparison.csv": evt, "positive_class_comparison.csv": positive}
    publish_diagnostics(tables, figures, summary, input_paths, fingerprints)

    # STEP 9 - Report observed differences and STOP; causation remains untested.
    print(continuous.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Tables: {OUTPUT_DIR}\nFigures: {FIGURE_DIR}", flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
