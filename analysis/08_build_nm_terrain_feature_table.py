"""Validate spatially batched construction of New Mexico terrain features."""

# %% Imports
import gc
import inspect
from pathlib import Path
from time import perf_counter

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import psutil
import rioxarray  # noqa: F401  # Register the xarray ``.rio`` CRS accessor.
import stackstac
import xarray as xr
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import box, shape
from xrspatial import aspect as xrspatial_aspect
from xrspatial import slope as xrspatial_slope


# %% Case-study and test parameters
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"
STAC_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:5070"
TARGET_EPSG = 5070
TARGET_RESOLUTION_M = 10
ANALYSIS_CELL_SIZE_M = 1_000
EXPECTED_RETAINED_CELL_COUNT = 314_920
EXPECTED_10M_ITEM_COUNT = 46
EXPECTED_GRID_BOUNDS = (-1_234_000, 992_000, -617_000, 1_629_000)
EXPECTED_TRIMMED_SHAPE = (63_700, 61_700)
EXPECTED_RECTANGULAR_SHAPE = (637, 617)
BATCH_SIZE_BLOCKS = 20
CHUNK_SIZE_PIXELS = 2_000
TERRAIN_HALO_M = TARGET_RESOLUTION_M
SOURCE_FILTER_BUFFER_M = 100
MOSAIC_SPLIT_EVERY = 2
ALIGNMENT_TOLERANCE_M = 1e-6
COMPONENT_TOLERANCE = 1e-6

FEATURE_COLUMNS = [
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "aspect_sin_mean", "aspect_cos_mean", "aspect_strength",
]
CHECKPOINT_COLUMNS = [
    "batch_id", "row_index", "column_index", "x_center", "y_center",
    "selected_item_count", *FEATURE_COLUMNS,
]

repo_root = Path(__file__).resolve().parents[1]
boundary_path = repo_root / "data" / "processed" / "boundaries" / "nm_boundary.gpkg"
grid_path = repo_root / "data" / "processed" / "grids" / "nm_analysis_grid_1km.gpkg"
checkpoint_dir = repo_root / "data" / "interim" / "terrain_1km" / "batches"
output_path = repo_root / "data" / "processed" / "features" / "nm_terrain_features_1km.parquet"


# %% Batch helpers
def _batch_id(row_start: int, column_start: int) -> str:
    """Return a deterministic identifier from aggregate-grid start indices."""
    return f"batch_r{row_start:03d}_c{column_start:03d}"


def _batch_spec(
    row_start: int,
    column_start: int,
    rectangular_rows: int,
    rectangular_columns: int,
    grid_bounds: tuple[float, float, float, float],
) -> dict[str, object]:
    """Describe one bounded aggregate-grid batch and its support extent."""
    if not (0 <= row_start < rectangular_rows and 0 <= column_start < rectangular_columns):
        raise ValueError("Batch start indices fall outside the aggregate grid.")
    row_stop = min(row_start + BATCH_SIZE_BLOCKS, rectangular_rows)
    column_stop = min(column_start + BATCH_SIZE_BLOCKS, rectangular_columns)
    interior_bounds = (
        grid_bounds[0] + column_start * ANALYSIS_CELL_SIZE_M,
        grid_bounds[3] - row_stop * ANALYSIS_CELL_SIZE_M,
        grid_bounds[0] + column_stop * ANALYSIS_CELL_SIZE_M,
        grid_bounds[3] - row_start * ANALYSIS_CELL_SIZE_M,
    )
    expanded_bounds = (
        interior_bounds[0] - TERRAIN_HALO_M,
        interior_bounds[1] - TERRAIN_HALO_M,
        interior_bounds[2] + TERRAIN_HALO_M,
        interior_bounds[3] + TERRAIN_HALO_M,
    )
    if not all(
        np.isclose(value / TARGET_RESOLUTION_M, round(value / TARGET_RESOLUTION_M))
        for value in expanded_bounds
    ):
        raise ValueError("Expanded batch bounds do not align with the 10 m grid.")
    return {
        "batch_id": _batch_id(row_start, column_start),
        "row_start": row_start, "row_stop": row_stop,
        "column_start": column_start, "column_stop": column_stop,
        "block_rows": row_stop - row_start,
        "block_columns": column_stop - column_start,
        "interior_bounds": interior_bounds, "expanded_bounds": expanded_bounds,
    }


def _select_batch_items(selected_items: list, expanded_bounds: tuple) -> list:
    """Select globally ordered Items that can contribute to a batch."""
    # The 100 m margin exceeds one target pixel, conservatively protecting
    # bilinear support and small footprint-coordinate differences.
    filter_geometry = gpd.GeoSeries(
        [box(*expanded_bounds).buffer(SOURCE_FILTER_BUFFER_M)], crs=TARGET_CRS
    ).to_crs(STAC_CRS).iloc[0]
    batch_items = []
    for item in selected_items:
        if item.geometry is None:
            raise ValueError(f"3DEP Item has no footprint geometry: {item.id}")
        if shape(item.geometry).intersects(filter_geometry):
            batch_items.append(item)
    if not batch_items:
        raise ValueError("No qualifying 3DEP Items intersect the expanded batch.")
    return batch_items


def _expected_batch_keys(batch: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic row-major block keys for one batch."""
    row_values = np.arange(batch["row_start"], batch["row_stop"], dtype=np.int32)
    column_values = np.arange(batch["column_start"], batch["column_stop"], dtype=np.int32)
    rows, columns = np.meshgrid(row_values, column_values, indexing="ij")
    return rows.ravel(), columns.ravel()


def _validate_feature_values(checkpoint: pd.DataFrame, batch_id: str) -> None:
    """Validate fixed terrain-feature domains for one checkpoint."""
    if (checkpoint["elevation_std"].dropna() < 0).any():
        raise ValueError(f"{batch_id} contains negative elevation_std.")
    if not checkpoint["slope_mean"].dropna().between(0, 90).all():
        raise ValueError(f"{batch_id} slope_mean falls outside [0, 90].")
    if (checkpoint["slope_std"].dropna() < 0).any():
        raise ValueError(f"{batch_id} contains negative slope_std.")
    for name in ("aspect_sin_mean", "aspect_cos_mean"):
        if not checkpoint[name].dropna().between(
            -1 - COMPONENT_TOLERANCE, 1 + COMPONENT_TOLERANCE
        ).all():
            raise ValueError(f"{batch_id} {name} is outside [-1, 1].")
    if not checkpoint["aspect_strength"].dropna().between(
        0, 1 + COMPONENT_TOLERANCE
    ).all():
        raise ValueError(f"{batch_id} aspect_strength is outside [0, 1].")


def _validate_checkpoint(checkpoint: pd.DataFrame, batch: dict[str, object]) -> None:
    """Prove that a checkpoint contains exactly its expected blocks."""
    batch_id = str(batch["batch_id"])
    expected_rows, expected_columns = _expected_batch_keys(batch)
    expected_count = int(batch["block_rows"]) * int(batch["block_columns"])
    if list(checkpoint.columns) != CHECKPOINT_COLUMNS:
        raise ValueError(f"{batch_id} checkpoint has an unexpected schema.")
    if len(checkpoint) != expected_count:
        raise ValueError(f"{batch_id} checkpoint has an unexpected row count.")
    if checkpoint["batch_id"].nunique() != 1 or checkpoint["batch_id"].iloc[0] != batch_id:
        raise ValueError(f"{batch_id} checkpoint contains the wrong batch ID.")
    if not (
        np.array_equal(checkpoint["row_index"], expected_rows)
        and np.array_equal(checkpoint["column_index"], expected_columns)
    ):
        raise ValueError(f"{batch_id} checkpoint has incomplete or unordered keys.")
    if checkpoint.duplicated(["row_index", "column_index"]).any():
        raise ValueError(f"{batch_id} checkpoint contains duplicate block keys.")
    if checkpoint["selected_item_count"].nunique() != 1:
        raise ValueError(f"{batch_id} checkpoint has inconsistent Item counts.")
    _validate_feature_values(checkpoint, batch_id)


def _print_batch_qa(checkpoint: pd.DataFrame, batch: dict[str, object]) -> None:
    """Print feature QA/QC for one computed or restored batch."""
    batch_id = str(batch["batch_id"])
    expected_count = int(batch["block_rows"]) * int(batch["block_columns"])
    print(f"\nBATCH QA/QC: {batch_id}")
    print(f"1-km block dimensions (y, x): {batch['block_rows']}, {batch['block_columns']}")
    print(f"Expected / actual rows: {expected_count:,} / {len(checkpoint):,}")
    print(f"Selected 3DEP Items: {checkpoint['selected_item_count'].iloc[0]:,}")
    for name in ("elevation_mean", "elevation_std", "slope_mean", "slope_std"):
        values = checkpoint[name].dropna()
        print(f"{name} min/max/mean: {values.min():.6f} / {values.max():.6f} / {values.mean():.6f}")
    for name in ("aspect_sin_mean", "aspect_cos_mean", "aspect_strength"):
        values = checkpoint[name].dropna()
        print(f"{name} min/max: {values.min():.6f} / {values.max():.6f}")
    print(f"Rows with any missing feature: {checkpoint[FEATURE_COLUMNS].isna().any(axis=1).sum():,}")
    print(f"Duplicate block keys: {checkpoint.duplicated(['row_index', 'column_index']).sum():,}")


def _stack_batch(signed_items: list, bounds: tuple) -> xr.DataArray:
    """Build a batch stack while isolating stackstac's pandas-3 workaround."""
    original_to_datetime = stackstac.prepare.pd.to_datetime
    if "infer_datetime_format" not in inspect.signature(original_to_datetime).parameters:
        def stackstac_to_datetime(*args, **kwargs):
            kwargs.pop("infer_datetime_format", None)
            return original_to_datetime(*args, **kwargs)
        stackstac.prepare.pd.to_datetime = stackstac_to_datetime
    try:
        return stackstac.stack(
            signed_items, assets=["data"], epsg=TARGET_EPSG,
            resolution=TARGET_RESOLUTION_M, bounds=bounds, snap_bounds=False,
            resampling=Resampling.bilinear,
            chunksize=(1, 1, CHUNK_SIZE_PIXELS, CHUNK_SIZE_PIXELS),
            dtype="float32", fill_value=np.float32(np.nan), rescale=False,
            sortby_date=False, properties=False,
        )
    finally:
        stackstac.prepare.pd.to_datetime = original_to_datetime


def _process_batch(batch: dict, selected_items: list, grid_bounds: tuple) -> tuple[pd.DataFrame, bool]:
    """Load a valid checkpoint or compute one bounded terrain batch."""
    batch_id = str(batch["batch_id"])
    checkpoint_path = checkpoint_dir / f"{batch_id}.parquet"
    if checkpoint_path.exists():
        try:
            checkpoint = pd.read_parquet(checkpoint_path)
            _validate_checkpoint(checkpoint, batch)
        except (OSError, ValueError) as error:
            print(f"Invalid checkpoint will be recomputed: {checkpoint_path}\n{error}")
        else:
            return checkpoint, True

    started_at = perf_counter()
    batch_items = _select_batch_items(selected_items, batch["expanded_bounds"])
    elevation_stack = _stack_batch([pc.sign(item) for item in batch_items], batch["expanded_bounds"])
    elevation_mosaic = stackstac.mosaic(
        elevation_stack.squeeze("band", drop=True), dim="time", reverse=True,
        nodata=np.nan, split_every=MOSAIC_SPLIT_EVERY,
    )
    slope = xrspatial_slope(elevation_mosaic, name="slope", method="planar")
    raw_aspect = xrspatial_aspect(
        elevation_mosaic, name="raw_aspect", method="planar"
    ).where(lambda values: values >= 0)
    aspect_radians = np.deg2rad(raw_aspect)
    aspect_sin = np.sin(aspect_radians).rename("aspect_sin")
    aspect_cos = np.cos(aspect_radians).rename("aspect_cos")

    expanded_shape = (int(batch["block_rows"]) * 100 + 2, int(batch["block_columns"]) * 100 + 2)
    interior_shape = (expanded_shape[0] - 2, expanded_shape[1] - 2)
    if elevation_mosaic.shape != expanded_shape:
        raise ValueError(f"{batch_id} expanded shape is {elevation_mosaic.shape}, expected {expanded_shape}.")

    # Each temporary one-pixel halo prevents false 3x3 edges between batches,
    # then is removed so support pixels never enter final 1-km summaries.
    interiors = {
        "elevation": elevation_mosaic.isel(y=slice(1, -1), x=slice(1, -1)),
        "slope": slope.isel(y=slice(1, -1), x=slice(1, -1)),
        "aspect_sin": aspect_sin.isel(y=slice(1, -1), x=slice(1, -1)),
        "aspect_cos": aspect_cos.isel(y=slice(1, -1), x=slice(1, -1)),
    }
    if any(value.shape != interior_shape for value in interiors.values()):
        raise ValueError(f"{batch_id} halo trimming did not restore its interior.")
    inputs = {name: xr.DataArray(value.data, dims=("y", "x")) for name, value in interiors.items()}
    coarseners = {name: value.coarsen(y=100, x=100, boundary="exact") for name, value in inputs.items()}
    aggregates = xr.Dataset({
        "elevation_mean": coarseners["elevation"].mean(skipna=True),
        "elevation_std": coarseners["elevation"].std(skipna=True, ddof=0),
        "slope_mean": coarseners["slope"].mean(skipna=True),
        "slope_std": coarseners["slope"].std(skipna=True, ddof=0),
        "aspect_sin_mean": coarseners["aspect_sin"].mean(skipna=True),
        "aspect_cos_mean": coarseners["aspect_cos"].mean(skipna=True),
    })
    if not all(dask.is_dask_collection(value.data) for value in aggregates.data_vars.values()):
        raise ValueError(f"{batch_id} aggregates became eager before compute.")
    with dask.config.set(scheduler="single-threaded"):
        loaded = aggregates.compute()

    rows, columns = _expected_batch_keys(batch)
    checkpoint = pd.DataFrame({
        "batch_id": batch_id, "row_index": rows, "column_index": columns,
        "x_center": grid_bounds[0] + (columns + 0.5) * ANALYSIS_CELL_SIZE_M,
        "y_center": grid_bounds[3] - (rows + 0.5) * ANALYSIS_CELL_SIZE_M,
        "selected_item_count": len(batch_items),
    })
    for name in FEATURE_COLUMNS[:-1]:
        checkpoint[name] = loaded[name].to_numpy().ravel()
    checkpoint["aspect_strength"] = np.hypot(checkpoint["aspect_sin_mean"], checkpoint["aspect_cos_mean"])
    checkpoint = checkpoint[CHECKPOINT_COLUMNS]
    _validate_checkpoint(checkpoint, batch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.to_parquet(checkpoint_path, index=False)

    memory_info = psutil.Process().memory_info()
    peak_bytes = getattr(memory_info, "peak_wset", memory_info.rss)
    checkpoint.attrs["elapsed_seconds"] = perf_counter() - started_at
    checkpoint.attrs["peak_memory_gib"] = peak_bytes / 1024**3

    # Release the complete local graph before another batch is constructed.
    del loaded, aggregates, coarseners, inputs, interiors, aspect_cos, aspect_sin
    del aspect_radians, raw_aspect, slope, elevation_mosaic, elevation_stack
    gc.collect()
    return checkpoint, False


# %% Validate prepared inputs and statewide coordinate structure
for prepared_path in (boundary_path, grid_path):
    if not prepared_path.exists():
        raise FileNotFoundError(f"Required prepared dataset not found: {prepared_path}")
boundary = gpd.read_file(boundary_path, layer="state_boundary")
analysis_grid = gpd.read_file(grid_path, layer="analysis_grid")
if len(boundary) != 1 or boundary.crs is None or boundary.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Expected one prepared boundary in {TARGET_CRS}.")
if len(analysis_grid) != EXPECTED_RETAINED_CELL_COUNT:
    raise ValueError(f"Expected {EXPECTED_RETAINED_CELL_COUNT:,} retained cells.")
if analysis_grid.crs is None or analysis_grid.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Prepared analysis grid must use {TARGET_CRS}.")
if not analysis_grid["cell_id"].is_unique:
    raise ValueError("Prepared analysis-grid cell IDs must be unique.")
grid_bounds = tuple(float(value) for value in analysis_grid.total_bounds)
if not np.allclose(grid_bounds, EXPECTED_GRID_BOUNDS, atol=ALIGNMENT_TOLERANCE_M):
    raise ValueError(f"Unexpected retained-grid bounds: {grid_bounds}")
derived_shape = (
    int((grid_bounds[3] - grid_bounds[1]) / TARGET_RESOLUTION_M),
    int((grid_bounds[2] - grid_bounds[0]) / TARGET_RESOLUTION_M),
)
if derived_shape != EXPECTED_TRIMMED_SHAPE or tuple(value // 100 for value in derived_shape) != EXPECTED_RECTANGULAR_SHAPE:
    raise ValueError("The retained extent does not produce the expected aligned grids.")
rectangular_rows, rectangular_columns = EXPECTED_RECTANGULAR_SHAPE


# %% Search once for the validated globally ordered source set
query_geometry = boundary.to_crs(STAC_CRS).geometry.iloc[0].__geo_interface__
items = list(Client.open(PC_STAC_URL).search(collections=[COLLECTION], intersects=query_geometry).items())
items_10m = [item for item in items if item.properties.get("gsd") == 10]
if len(items_10m) != EXPECTED_10M_ITEM_COUNT:
    raise ValueError(f"Expected {EXPECTED_10M_ITEM_COUNT} qualifying Items; found {len(items_10m)}.")
selected_items = sorted(items_10m, key=lambda item: item.id)
if len({item.id for item in selected_items}) != len(selected_items):
    raise ValueError("Qualifying 10 m Items contain duplicate Item IDs.")
if sorted({item.properties.get("proj:code") for item in selected_items}, key=str) != ["EPSG:5498"]:
    raise ValueError("Qualifying Items have an unexpected source CRS.")


# %% Enumerate and inventory every deterministic statewide batch
all_batches = [
    _batch_spec(row, column, rectangular_rows, rectangular_columns, grid_bounds)
    for row in range(0, rectangular_rows, BATCH_SIZE_BLOCKS)
    for column in range(0, rectangular_columns, BATCH_SIZE_BLOCKS)
]
valid_checkpoint_ids = set()
for batch in all_batches:
    checkpoint_path = checkpoint_dir / f"{batch['batch_id']}.parquet"
    if not checkpoint_path.exists():
        continue
    try:
        existing_checkpoint = pd.read_parquet(checkpoint_path)
        _validate_checkpoint(existing_checkpoint, batch)
    except (OSError, ValueError):
        continue
    valid_checkpoint_ids.add(batch["batch_id"])

print("STATEWIDE SPATIAL-BATCH PLAN")
print("----------------------------")
print(f"Rectangular aggregate dimensions: {rectangular_rows} x {rectangular_columns}")
print(f"Total rectangular blocks: {rectangular_rows * rectangular_columns:,}")
print(f"Full batch size: {BATCH_SIZE_BLOCKS} x {BATCH_SIZE_BLOCKS} blocks")
print(f"Total batches: {len(all_batches):,}")
print(f"Existing valid checkpoints: {len(valid_checkpoint_ids):,}")
print(f"Batches requiring computation: {len(all_batches) - len(valid_checkpoint_ids):,}")
print(f"Mosaic split_every: {MOSAIC_SPLIT_EVERY}")


# %% Process or safely reuse every batch without retaining raster graphs
run_started_at = perf_counter()
checkpoint_tables = []
reused_count = 0
computed_count = 0
last_completed_batch = None
for batch_number, batch in enumerate(all_batches, start=1):
    batch_started_at = perf_counter()
    checkpoint, reused = _process_batch(batch, selected_items, grid_bounds)
    checkpoint_tables.append(checkpoint)
    last_completed_batch = batch["batch_id"]
    if reused:
        reused_count += 1
        outcome = "reused"
    else:
        computed_count += 1
        outcome = "computed"

    memory_info = psutil.Process().memory_info()
    peak_bytes = getattr(memory_info, "peak_wset", memory_info.rss)
    print(
        f"[{batch_number:03d}/{len(all_batches):03d}] {batch['batch_id']} "
        f"{outcome}; Items={checkpoint['selected_item_count'].iloc[0]}; "
        f"rows={len(checkpoint)}; elapsed={perf_counter() - batch_started_at:.1f}s; "
        f"RSS={memory_info.rss / 1024**3:.2f} GiB; "
        f"peak={peak_bytes / 1024**3:.2f} GiB"
    )


# %% Combine checkpoints and prove complete rectangular coverage
rectangular_features = pd.concat(checkpoint_tables, ignore_index=True)
expected_rectangular_count = rectangular_rows * rectangular_columns
duplicate_block_count = int(
    rectangular_features.duplicated(["row_index", "column_index"]).sum()
)
expected_keys = pd.MultiIndex.from_product(
    [range(rectangular_rows), range(rectangular_columns)],
    names=["row_index", "column_index"],
)
actual_keys = pd.MultiIndex.from_frame(
    rectangular_features[["row_index", "column_index"]]
)
missing_block_count = len(expected_keys.difference(actual_keys))
extra_block_count = len(actual_keys.difference(expected_keys))
if (
    len(rectangular_features) != expected_rectangular_count
    or duplicate_block_count
    or missing_block_count
    or extra_block_count
):
    raise ValueError("Combined checkpoints do not exactly cover the rectangular grid.")

print("\nRECTANGULAR CHECKPOINT QA")
print("-------------------------")
print(f"Expected / actual rows: {expected_rectangular_count:,} / {len(rectangular_features):,}")
print(f"Duplicate block keys: {duplicate_block_count:,}")
print(f"Missing block keys: {missing_block_count:,}")
print(f"Extra block keys: {extra_block_count:,}")


# %% Map authoritative retained cells to exactly one rectangular block
centroids = analysis_grid.geometry.centroid
retained_mapping = pd.DataFrame(
    {
        "cell_id": analysis_grid["cell_id"].to_numpy(),
        "row_index": np.rint(
            ((grid_bounds[3] - ANALYSIS_CELL_SIZE_M / 2) - centroids.y.to_numpy())
            / ANALYSIS_CELL_SIZE_M
        ).astype(np.int32),
        "column_index": np.rint(
            (centroids.x.to_numpy() - (grid_bounds[0] + ANALYSIS_CELL_SIZE_M / 2))
            / ANALYSIS_CELL_SIZE_M
        ).astype(np.int32),
    }
)
if retained_mapping.duplicated(["row_index", "column_index"]).any():
    raise ValueError("Multiple retained cells map to one aggregate block.")

terrain_features = retained_mapping.merge(
    rectangular_features[["row_index", "column_index", *FEATURE_COLUMNS]],
    on=["row_index", "column_index"],
    how="left",
    validate="one_to_one",
    indicator=True,
)
if not terrain_features["_merge"].eq("both").all():
    raise ValueError("At least one retained cell is missing an aggregate block.")
terrain_features = (
    terrain_features[["cell_id", *FEATURE_COLUMNS]]
    .sort_values("cell_id")
    .reset_index(drop=True)
)
if len(terrain_features) != EXPECTED_RETAINED_CELL_COUNT:
    raise ValueError("Final retained terrain table has an unexpected row count.")
if terrain_features["cell_id"].nunique() != EXPECTED_RETAINED_CELL_COUNT:
    raise ValueError("Final retained terrain table has missing or duplicate cell IDs.")


# %% Validate and report final numeric features without filling missing values
_validate_feature_values(terrain_features, "final terrain table")
if terrain_features["elevation_mean"].notna().any() and not np.isfinite(
    terrain_features.loc[terrain_features["elevation_mean"].notna(), "elevation_mean"]
).all():
    raise ValueError("Valid final elevation means contain a non-finite value.")

print("\nFINAL TERRAIN FEATURE QA/QC")
print("---------------------------")
for feature_name in FEATURE_COLUMNS:
    values = terrain_features[feature_name]
    valid_values = values.dropna()
    print(f"{feature_name}:")
    print(f"  valid count: {len(valid_values):,}")
    print(f"  missing count: {values.isna().sum():,}")
    print(f"  minimum: {valid_values.min():.6f}")
    print(f"  maximum: {valid_values.max():.6f}")
    print(f"  mean: {valid_values.mean():.6f}")
    print(f"  standard deviation: {valid_values.std(ddof=0):.6f}")

cells_with_missing = int(terrain_features[FEATURE_COLUMNS].isna().any(axis=1).sum())
missing_aspect = int(
    terrain_features[["aspect_sin_mean", "aspect_cos_mean"]].isna().all(axis=1).sum()
)
print(f"Cells with any missing terrain feature: {cells_with_missing:,}")
print(f"Cells with undefined/missing aspect summaries: {missing_aspect:,}")
print(f"Total final rows: {len(terrain_features):,}")
print(f"Unique cell_id count: {terrain_features['cell_id'].nunique():,}")
print(f"Duplicate cell_id count: {terrain_features['cell_id'].duplicated().sum():,}")
print("\nFirst five final rows:")
print(terrain_features.head().to_string(index=False))


# %% Write and independently verify the final compact feature table
output_path.parent.mkdir(parents=True, exist_ok=True)
terrain_features.to_parquet(output_path, index=False)
written_features = pd.read_parquet(output_path)
if list(written_features.columns) != ["cell_id", *FEATURE_COLUMNS]:
    raise ValueError("Written terrain Parquet has an unexpected schema.")
if len(written_features) != EXPECTED_RETAINED_CELL_COUNT:
    raise ValueError("Written terrain Parquet has an unexpected row count.")
if written_features["cell_id"].nunique() != EXPECTED_RETAINED_CELL_COUNT:
    raise ValueError("Written terrain Parquet has duplicate or missing cell IDs.")

run_elapsed_seconds = perf_counter() - run_started_at
process_memory = psutil.Process().memory_info()
process_peak_bytes = getattr(process_memory, "peak_wset", process_memory.rss)
print("\nSTATEWIDE WORKFLOW COMPLETE")
print("---------------------------")
print(f"Total batches: {len(all_batches):,}")
print(f"Newly computed checkpoints: {computed_count:,}")
print(f"Reused checkpoints: {reused_count:,}")
print(f"Last completed batch: {last_completed_batch}")
print(f"Total execution time: {run_elapsed_seconds / 60:.2f} minutes")
print(f"Process peak memory: {process_peak_bytes / 1024**3:.2f} GiB")
print(f"Final output: {output_path}")
print(f"Final file size: {output_path.stat().st_size / 1024**2:.2f} MiB")
print(f"Final rows / unique cell IDs: {len(written_features):,} / {written_features['cell_id'].nunique():,}")

# %%
