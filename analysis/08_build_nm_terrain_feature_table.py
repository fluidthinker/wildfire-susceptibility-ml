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

# These central batches test one horizontal and one vertical shared edge. This
# task intentionally stops after them rather than processing all 1,024 batches.
TEST_BATCH_STARTS = ((300, 300), (300, 320), (320, 300))
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
            print(f"Skipped valid checkpoint: {checkpoint_path}")
            _print_batch_qa(checkpoint, batch)
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
    print(f"Completed batch: {batch_id}")
    print(f"Interior / expanded terrain dimensions: {interior_shape} / {expanded_shape}")
    print(f"Elapsed time: {perf_counter() - started_at:.1f} seconds")
    print(f"Process peak memory: {peak_bytes / 1024**3:.2f} GiB")
    print(f"Saved checkpoint: {checkpoint_path}")
    _print_batch_qa(checkpoint, batch)

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


# %% Run only the approved three-batch test
test_batches = [
    _batch_spec(row, column, rectangular_rows, rectangular_columns, grid_bounds)
    for row, column in TEST_BATCH_STARTS
]
print("SPATIAL-BATCH TEST")
print("------------------")
print(f"Global qualifying Items: {len(selected_items):,}")
print(f"Test batches: {[batch['batch_id'] for batch in test_batches]}")
print(f"Complete batch size: {BATCH_SIZE_BLOCKS} x {BATCH_SIZE_BLOCKS} blocks")
print(f"Mosaic split_every: {MOSAIC_SPLIT_EVERY}")

test_checkpoints = []
skipped_batches = []
for batch in test_batches:
    checkpoint, skipped = _process_batch(batch, selected_items, grid_bounds)
    test_checkpoints.append(checkpoint)
    if skipped:
        skipped_batches.append(batch["batch_id"])


# %% Validate shared boundaries and combined coverage
combined_test = pd.concat(test_checkpoints, ignore_index=True)
expected_test_count = sum(int(batch["block_rows"]) * int(batch["block_columns"]) for batch in test_batches)
duplicate_test_blocks = int(combined_test.duplicated(["row_index", "column_index"]).sum())
if len(combined_test) != expected_test_count or duplicate_test_blocks:
    raise ValueError("Test checkpoints contain duplicate or missing blocks.")
west, east, south = test_batches
if west["column_stop"] != east["column_start"] or west["row_stop"] != south["row_start"]:
    raise ValueError("Test batch index ranges are not contiguous.")
west_x = combined_test.loc[combined_test["batch_id"] == west["batch_id"], "x_center"].max()
east_x = combined_test.loc[combined_test["batch_id"] == east["batch_id"], "x_center"].min()
north_y = combined_test.loc[combined_test["batch_id"] == west["batch_id"], "y_center"].min()
south_y = combined_test.loc[combined_test["batch_id"] == south["batch_id"], "y_center"].max()
if not (np.isclose(east_x - west_x, 1_000) and np.isclose(north_y - south_y, 1_000)):
    raise ValueError("Test block centers are discontinuous across a batch edge.")

print("\nADJACENT-BATCH QA")
print("-----------------")
print(f"Combined expected / actual blocks: {expected_test_count:,} / {len(combined_test):,}")
print(f"Duplicate tested block keys: {duplicate_test_blocks:,}")
print("Missing tested block keys: 0")
print("Horizontal / vertical center continuity: 1,000 m / 1,000 m")
print("Every batch used and removed its own one-pixel terrain-support halo.")
print(f"Valid checkpoints skipped during this run: {skipped_batches}")
print("The script intentionally stops after the three-batch validation set.")

# %%
