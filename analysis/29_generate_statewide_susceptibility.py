"""Create model-estimated statewide wildfire susceptibility after evaluation.

This single mapping fit includes the former training and holdout populations.
Its probabilities are visualization outputs, not independent performance
estimates, ignition forecasts, wildfire risk, or validated occurrence rates.
"""

# %% Imports
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn
from geopandas.testing import assert_geodataframe_equal

from wildfire_susceptibility.modeling import (
    ASPECT_COLUMNS, PREDICTORS, build_random_forest_pipeline,
)


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
GRID_PATH = ROOT / "data/processed/grids/nm_analysis_grid_1km.gpkg"
FEATURE_PATHS = [ROOT / f"data/processed/features/nm_{name}_features_1km.parquet"
                 for name in ("terrain", "prism", "landfire")]
TUNING_PATH = ROOT / "outputs/modeling/random_forest_spatial_tuning/tuning_summary.json"
OUTPUT_PATH = ROOT / "data/processed/modeling/nm_statewide_susceptibility.geoparquet"
SUMMARY_PATH = ROOT / "outputs/modeling/statewide_susceptibility/summary.json"
GRID_ROWS = 314_920
FIT_ROWS = 313_129
TUNED_PARAMETERS = {"max_depth": 30, "min_samples_leaf": 5}
FIXED_PARAMETERS = {
    "n_estimators": 300, "random_state": 42, "n_jobs": -1, "class_weight": None,
    "criterion": "gini", "bootstrap": True, "max_features": "sqrt", "min_samples_split": 2,
}


# %% Helper functions
def fingerprint_inputs(paths: list[Path]) -> dict[str, str]:
    """Hash source artifacts with bounded reads for reproducibility.

    Args:
        paths: Immutable input and code files.

    Returns:
        Repository-relative paths mapped to SHA-256 digests.
    """
    result = {}
    for path in paths:
        with path.open("rb") as source:
            result[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


def validate_keys(data: pd.DataFrame, expected_rows: int, label: str) -> None:
    """Require complete unique cell keys before joining or predicting.

    Args:
        data: Input or output table with cell_id.
        expected_rows: Frozen population size.
        label: Context for validation errors.

    Raises:
        ValueError: If columns, row count, or keys are invalid.
    """
    if not data.columns.is_unique or "cell_id" not in data:
        raise ValueError(f"{label}: missing keys or duplicate columns")
    if len(data) != expected_rows or data.cell_id.isna().any() or not data.cell_id.is_unique:
        raise ValueError(f"{label}: expected {expected_rows:,} unique nonmissing cell IDs")


def validate_predictors(data: pd.DataFrame, label: str) -> dict:
    """Enforce existing numeric features while preserving flat-cell aspect NaNs.

    Args:
        data: Labeled population or complete statewide predictor grid.
        label: Context for validation errors.

    Returns:
        Missing-value counts for summary metadata.

    Raises:
        ValueError: If approved predictors, types, or missingness differ.
    """
    if not set(PREDICTORS).issubset(data.columns):
        raise ValueError(f"{label}: missing approved predictors")
    missing = {}
    for column in PREDICTORS:
        values = data[column]
        missing[column] = int(values.isna().sum())
        expected = 39 if column in ASPECT_COLUMNS else 0
        if missing[column] != expected:
            raise ValueError(f"{label}.{column}: {missing[column]} missing values, expected {expected}; no cells excluded")
        if not pd.api.types.is_numeric_dtype(values) or np.isinf(values.dropna()).any():
            raise ValueError(f"{label}.{column}: nonnumeric or infinite values")
    if not pd.api.types.is_integer_dtype(data.evt_dominant_class):
        raise ValueError("EVT class identifiers must remain integer nominal codes")
    return missing


def validate_mapping_population(data: pd.DataFrame) -> None:
    """Validate all eligible labels without constructing a new target or split.

    Args:
        data: Script 15's complete labeled modeling artifact.

    Raises:
        ValueError: If schema, target counts, or existing label rules differ.
    """
    validate_keys(data, FIT_ROWS, "Mapping fit")
    if set(data.columns) != {"cell_id", "target", "burned_fraction", *PREDICTORS}:
        raise ValueError("Unexpected labeled modeling schema")
    if data.target.isna().any() or data.target.value_counts().to_dict() != {0: 305_723, 1: 7_406}:
        raise ValueError("Expected 305,723 negatives and 7,406 positives")
    valid_labels = ((data.burned_fraction.eq(0) & data.target.eq(0))
                    | (data.burned_fraction.between(0.25, 1) & data.target.eq(1)))
    if not valid_labels.all():
        raise ValueError("Labels differ from the established burned-fraction rule")
    validate_predictors(data, "Mapping fit")


def assemble_statewide_features(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join completed features to every authoritative grid cell without filtering.

    Args:
        grid: Original EPSG:5070 grid in stable cell_id order.

    Returns:
        Full grid with the ten approved predictors and unchanged geometry.

    Raises:
        ValueError: If any source has mismatched keys or overlapping fields.
    """
    statewide = grid.copy()
    for path in FEATURE_PATHS:
        features = pd.read_parquet(path)
        validate_keys(features, GRID_ROWS, path.name)
        if set(features.cell_id) != set(grid.cell_id):
            raise ValueError(f"{path.name}: feature keys differ from the authoritative grid")
        if (set(features.columns) & set(statewide.columns)) != {"cell_id"}:
            raise ValueError(f"{path.name}: overlapping feature fields")
        statewide = statewide.merge(features, on="cell_id", how="left", validate="one_to_one", sort=False)
    if set(statewide.columns) != {"cell_id", "geometry", *PREDICTORS}:
        raise ValueError("Statewide assembly must contain exactly the approved predictors")
    return statewide


def validate_grid(grid: gpd.GeoDataFrame) -> None:
    """Verify the saved full-cell geometry without reprojection or repair.

    Args:
        grid: Authoritative statewide geometry.

    Raises:
        ValueError: If CRS, geometry, or complete 1-km cell dimensions differ.
    """
    validate_keys(grid, GRID_ROWS, "Statewide grid")
    if grid.crs is None or grid.crs.to_epsg() != 5070:
        raise ValueError("Expected EPSG:5070; no automatic reprojection")
    geometry = grid.geometry
    if geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all():
        raise ValueError("Grid contains missing, empty, or invalid geometry")
    if not geometry.geom_type.eq("Polygon").all() or geometry.duplicated().any():
        raise ValueError("Expected unique polygon cells")
    bounds = geometry.bounds
    if not (np.allclose(geometry.area, 1_000_000, rtol=0, atol=1e-6)
            and np.allclose(bounds.maxx - bounds.minx, 1000, rtol=0, atol=1e-6)
            and np.allclose(bounds.maxy - bounds.miny, 1000, rtol=0, atol=1e-6)):
        raise ValueError("Grid must retain complete 1000 m square cells")


def validate_output(result: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> None:
    """Check one bounded probability per cell and exact geometry preservation.

    Args:
        result: Proposed or read-back geospatial susceptibility artifact.
        grid: Original authoritative geometry and identifiers.

    Raises:
        ValueError: If schema, coverage, probabilities, CRS, or geometry differ.
        AssertionError: If cell ordering changes.
    """
    validate_keys(result, GRID_ROWS, "Susceptibility output")
    if list(result.columns) != ["cell_id", "susceptibility_probability", "geometry"]:
        raise ValueError("Unexpected susceptibility output schema")
    pd.testing.assert_series_equal(result.cell_id, grid.cell_id, check_exact=True)
    if result.crs != grid.crs or not np.array_equal(result.geometry.to_wkb(), grid.geometry.to_wkb()):
        raise ValueError("Output geometry or CRS changed")
    probability = result.susceptibility_probability.to_numpy()
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Susceptibility probabilities must be finite and in [0,1]")


def publish_results(result: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, summary: dict,
                    paths: list[Path], fingerprints: dict) -> None:
    """Publish only after GeoParquet/JSON read-back and immutable-input checks.

    Args:
        result: Validated statewide probabilities and geometry.
        grid: Original grid for exact identity checks.
        summary: JSON-compatible production-fit metadata.
        paths: Inputs and code hashed before fitting.
        fingerprints: Original SHA-256 values.

    Raises:
        ValueError: If stored artifacts or source hashes fail validation.
        AssertionError: If serialization changes the geospatial table.
    """
    destinations = [OUTPUT_PATH, SUMMARY_PATH]
    partials = [path.with_suffix(path.suffix + ".part") for path in destinations]
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_parquet(partials[0], index=False, compression="zstd")
        partials[1].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stored = gpd.read_parquet(partials[0])
        validate_output(stored, grid)
        assert_geodataframe_equal(stored, result)
        pd.testing.assert_series_equal(stored.susceptibility_probability,
                                       result.susceptibility_probability, check_exact=True)
        if json.loads(partials[1].read_text(encoding="utf-8")) != summary:
            raise ValueError("Summary read-back differs")
        if fingerprint_inputs(paths) != fingerprints:
            raise ValueError("Immutable inputs changed during mapping fit")
        # Atomic per-file replacement; not a multi-file transaction.
        # Publish summary last, and remove partial artifacts after any failure.
        for temporary, destination in zip(partials, destinations):
            temporary.replace(destination)
    finally:
        for temporary in partials:
            temporary.unlink(missing_ok=True)


# %% Main workflow
def main() -> None:
    """Fit once for statewide visualization, publish the surface, and stop."""
    started = perf_counter()
    paths = [MODELING_PATH, GRID_PATH, *FEATURE_PATHS, TUNING_PATH,
             Path(__file__).resolve(), ROOT / "src/wildfire_susceptibility/modeling.py"]
    fingerprints = fingerprint_inputs(paths)
    # STEP 1 - Use every eligible labeled cell now that evaluation is complete.
    labeled = pd.read_parquet(MODELING_PATH).sort_values("cell_id").reset_index(drop=True)
    validate_mapping_population(labeled)

    # STEP 2 - Assemble full-grid predictors from existing processed tables.
    # Target ambiguity does not exclude a cell with valid environmental features.
    grid = gpd.read_file(GRID_PATH, layer="analysis_grid").sort_values("cell_id").reset_index(drop=True)
    validate_grid(grid)
    statewide = assemble_statewide_features(grid)

    # STEP 3 - Preserve geometry and enforce the established missingness contract.
    missing = validate_predictors(statewide, "Statewide")
    assert_geodataframe_equal(statewide[["cell_id", "geometry"]], grid)
    # Exact feature agreement prevents fitting and mapping with different products.
    matched = statewide.set_index("cell_id").loc[labeled.cell_id, list(PREDICTORS)].reset_index()
    # Source EVT uses nullable Int64 while Script 15 stores int64. Both are
    # validated integers without missing codes; require exact values, not the
    # same storage dtype. No feature values are converted or imputed here.
    pd.testing.assert_frame_equal(
        matched, labeled[["cell_id", *PREDICTORS]], check_exact=True, check_dtype=False,
    )

    # STEP 4 - Verify the already-selected settings; never choose a new candidate.
    tuning = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
    winner = tuning["selected_candidate"]
    if any(winner.get(key) != value for key, value in {"candidate": "D", **TUNED_PARAMETERS}.items()):
        raise ValueError("Script 26 must identify Candidate D: depth=30, leaf=5")
    if tuning["fixed_parameters"] != FIXED_PARAMETERS or tuning["predictors"] != list(PREDICTORS):
        raise ValueError("Script 26 fixed parameters or predictors changed")
    pipeline = build_random_forest_pipeline(**TUNED_PARAMETERS)
    parameters = pipeline.named_steps["model"].get_params()
    if any(parameters[key] != value for key, value in {**FIXED_PARAMETERS, **TUNED_PARAMETERS}.items()):
        raise ValueError("Pipeline settings differ from the frozen mapping configuration")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # STEP 5 - Fit preprocessing and forest once on all 313,129 labeled cells.
        print(f"Fitting one mapping model on {len(labeled):,} labeled cells...", flush=True)
        fit_started = perf_counter()
        pipeline.fit(labeled[list(PREDICTORS)], labeled.target)
        fit_seconds = perf_counter() - fit_started
        if list(pipeline.named_steps["model"].classes_) != [0, 1]:
            raise ValueError("Expected class-1 probability in column 1")

        # STEP 6 - Predict every full-grid cell; no performance scores are computed.
        print(f"Predicting model-estimated susceptibility for {len(statewide):,} cells...", flush=True)
        result = grid.copy()
        result["susceptibility_probability"] = pipeline.predict_proba(statewide[list(PREDICTORS)])[:, 1]
        result = result[["cell_id", "susceptibility_probability", "geometry"]]

    # STEP 7 - Validate complete output coverage and exact original geometry.
    validate_output(result, grid)

    # STEP 8 - Safely publish the minimal GeoParquet and interpretation metadata.
    probability = result.susceptibility_probability
    percentiles = probability.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    summary = {
        "model": "RandomForestClassifier", "mapping_fit_rows": len(labeled),
        "mapping_fit_positives": int(labeled.target.sum()), "mapping_fit_negatives": int(labeled.target.eq(0).sum()),
        "statewide_grid_rows": len(grid), "predicted_rows": len(result), "excluded_rows": 0,
        "exclusion_reasons": [], "predicted_cells_outside_labeled_population": len(grid) - len(labeled),
        "predictors": list(PREDICTORS), "model_parameters": parameters,
        "selected_candidate": "D", "tuned_parameters": TUNED_PARAMETERS,
        "preprocessing": {"numeric_scaling": "none", "aspect_imputation": "constant 0 inside pipeline",
                          "evt_encoding": "OneHotEncoder(handle_unknown='ignore')",
                          "fit_population": "all 313129 eligible labeled cells"},
        "statewide_missingness_before_pipeline": missing,
        "probability_min": float(probability.min()), "probability_median": float(probability.median()),
        "probability_mean": float(probability.mean()), "probability_max": float(probability.max()),
        "probability_percentiles": {f"p{int(q * 100):02d}": float(value) for q, value in percentiles.items()},
        "crs": "EPSG:5070", "output_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "quantity": "model-estimated wildfire susceptibility",
        "performance_claim": "None. This refit is for statewide visualization after evaluation was completed.",
        "interpretation": "Not wildfire risk, ignition probability, next-year fire probability, operational hazard, "
                          "or a validated statewide probability of occurrence.",
        "fit_includes_former_holdout": True, "fit_count": 1,
        "tuning_performed": False, "cv_performed": False, "performance_evaluation_performed": False,
        "input_sha256": fingerprints, "input_hashes_verified_before_publication": True,
        "runtime": {"fit_seconds": round(fit_seconds, 3),
                    "seconds_before_publication": round(perf_counter() - started, 3)},
        "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in captured],
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "geopandas": gpd.__version__},
    }
    publish_results(result, grid, summary, paths, fingerprints)

    # STEP 9 - Report the production artifact and STOP before cartography.
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"Summary: {SUMMARY_PATH}", flush=True)
    print(f"Total runtime including publication: {perf_counter() - started:.3f} seconds", flush=True)


# %% Run script
if __name__ == "__main__":
    main()
