"""Build the retained-cell New Mexico 1-km terrain feature table."""

# %% Imports
import inspect
import math
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rioxarray  # noqa: F401  # Register the xarray ``.rio`` CRS accessor.
import stackstac
import xarray as xr
from pystac_client import Client
from rasterio.enums import Resampling
from xrspatial import aspect as xrspatial_aspect
from xrspatial import slope as xrspatial_slope


# %% Case-study parameters
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

# Two thousand pixels equal twenty complete 1-km blocks. Aligning chunk and
# aggregation boundaries keeps the statewide reduction graph straightforward.
CHUNK_SIZE_PIXELS = 2_000
TERRAIN_HALO_M = TARGET_RESOLUTION_M
ALIGNMENT_TOLERANCE_M = 1e-6
COMPONENT_TOLERANCE = 1e-6

repo_root = Path(__file__).resolve().parents[1]
boundary_path = (
    repo_root / "data" / "processed" / "boundaries" / "nm_boundary.gpkg"
)
grid_path = (
    repo_root / "data" / "processed" / "grids" / "nm_analysis_grid_1km.gpkg"
)
output_path = (
    repo_root
    / "data"
    / "processed"
    / "features"
    / "nm_terrain_features_1km.parquet"
)


# %% Read and validate the prepared spatial inputs
for prepared_path in (boundary_path, grid_path):
    if not prepared_path.exists():
        raise FileNotFoundError(f"Required prepared dataset not found: {prepared_path}")

boundary = gpd.read_file(boundary_path, layer="state_boundary")
analysis_grid = gpd.read_file(grid_path, layer="analysis_grid")

if len(boundary) != 1:
    raise ValueError(f"Expected one New Mexico boundary; found {len(boundary)}.")
if boundary.crs is None or boundary.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Prepared boundary must use {TARGET_CRS}.")
if boundary.geometry.isna().any() or boundary.geometry.is_empty.any():
    raise ValueError("Prepared boundary contains missing or empty geometry.")
if not boundary.geometry.is_valid.all():
    raise ValueError("Prepared boundary contains invalid geometry.")

if len(analysis_grid) != EXPECTED_RETAINED_CELL_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_RETAINED_CELL_COUNT:,} retained cells; "
        f"found {len(analysis_grid):,}."
    )
if analysis_grid.crs is None or analysis_grid.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Prepared analysis grid must use {TARGET_CRS}.")
if not {"cell_id", "geometry"}.issubset(analysis_grid.columns):
    raise ValueError("Prepared analysis grid requires cell_id and geometry columns.")
if not analysis_grid["cell_id"].is_unique:
    raise ValueError("Prepared analysis-grid cell IDs must be unique.")
if analysis_grid.geometry.isna().any() or analysis_grid.geometry.is_empty.any():
    raise ValueError("Prepared analysis grid contains missing or empty geometry.")
if not analysis_grid.geometry.is_valid.all():
    raise ValueError("Prepared analysis grid contains invalid geometry.")
if not analysis_grid.geom_type.eq("Polygon").all():
    raise ValueError("Prepared analysis grid must contain only Polygon geometries.")

cell_bounds = analysis_grid.geometry.bounds
cell_widths = cell_bounds["maxx"] - cell_bounds["minx"]
cell_heights = cell_bounds["maxy"] - cell_bounds["miny"]
if not (
    np.allclose(cell_widths, ANALYSIS_CELL_SIZE_M)
    and np.allclose(cell_heights, ANALYSIS_CELL_SIZE_M)
):
    raise ValueError("Every retained analysis cell must be a complete 1-km square.")

grid_bounds = tuple(float(value) for value in analysis_grid.total_bounds)
if not np.allclose(grid_bounds, EXPECTED_GRID_BOUNDS, atol=ALIGNMENT_TOLERANCE_M):
    raise ValueError(
        f"Retained-grid bounds differ from {EXPECTED_GRID_BOUNDS}: {grid_bounds}"
    )

query_geometry = boundary.to_crs(STAC_CRS).geometry.iloc[0].__geo_interface__


# %% Construct the validated lazy statewide elevation mosaic
catalog = Client.open(PC_STAC_URL)
items = list(
    catalog.search(collections=[COLLECTION], intersects=query_geometry).items()
)
items_10m = [item for item in items if item.properties.get("gsd") == 10]
if len(items_10m) != EXPECTED_10M_ITEM_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_10M_ITEM_COUNT} qualifying 10 m Items, "
        f"but found {len(items_10m)}."
    )
source_crs_values = sorted(
    {item.properties.get("proj:code") for item in items_10m}, key=str
)
if source_crs_values != ["EPSG:5498"]:
    raise ValueError(
        "Expected qualifying Items to report source CRS EPSG:5498, "
        f"but found {source_crs_values}."
    )

# Stable Item-ID order preserves the validated first-valid source precedence.
selected_items = sorted(items_10m, key=lambda item: item.id)
if len({item.id for item in selected_items}) != len(selected_items):
    raise ValueError("Qualifying 10 m Items contain duplicate Item IDs.")

target_bounds = (
    math.floor((grid_bounds[0] - TERRAIN_HALO_M) / TARGET_RESOLUTION_M)
    * TARGET_RESOLUTION_M,
    math.floor((grid_bounds[1] - TERRAIN_HALO_M) / TARGET_RESOLUTION_M)
    * TARGET_RESOLUTION_M,
    math.ceil((grid_bounds[2] + TERRAIN_HALO_M) / TARGET_RESOLUTION_M)
    * TARGET_RESOLUTION_M,
    math.ceil((grid_bounds[3] + TERRAIN_HALO_M) / TARGET_RESOLUTION_M)
    * TARGET_RESOLUTION_M,
)

signed_items = [pc.sign(item) for item in selected_items]

# stackstac 0.5.0 passes a no-op keyword removed by pandas 3. Limit this
# compatibility adjustment to stack construction and immediately restore it.
original_to_datetime = stackstac.prepare.pd.to_datetime
if "infer_datetime_format" not in inspect.signature(original_to_datetime).parameters:

    def stackstac_to_datetime(*args, **kwargs):
        kwargs.pop("infer_datetime_format", None)
        return original_to_datetime(*args, **kwargs)

    stackstac.prepare.pd.to_datetime = stackstac_to_datetime

try:
    elevation_stack = stackstac.stack(
        signed_items,
        assets=["data"],
        epsg=TARGET_EPSG,
        resolution=TARGET_RESOLUTION_M,
        bounds=target_bounds,
        snap_bounds=False,
        resampling=Resampling.bilinear,
        chunksize=(1, 1, CHUNK_SIZE_PIXELS, CHUNK_SIZE_PIXELS),
        dtype="float32",
        fill_value=np.float32(np.nan),
        rescale=False,
        sortby_date=False,
    )
finally:
    stackstac.prepare.pd.to_datetime = original_to_datetime

if elevation_stack.sizes.get("time") != len(selected_items):
    raise ValueError("The raster stack does not contain every selected Item.")
if elevation_stack.sizes.get("band") != 1:
    raise ValueError("Expected exactly one elevation band in the raster stack.")

elevation_mosaic = stackstac.mosaic(
    elevation_stack.squeeze("band", drop=True),
    dim="time",
    reverse=True,
    nodata=np.nan,
)


# %% Derive terrain before aggregation, then remove computational support
# Slope and aspect must be derived at 10 m: deriving them after aggregation
# would erase the local terrain variation that the cell summaries represent.
slope = xrspatial_slope(elevation_mosaic, name="slope", method="planar")
raw_aspect = xrspatial_aspect(
    elevation_mosaic,
    name="raw_aspect",
    method="planar",
).where(lambda values: values >= 0)

# Raw degrees are circular and cannot be averaged safely. Sine and cosine
# retain circular direction while keeping undefined flat aspects as NaN.
aspect_radians = np.deg2rad(raw_aspect)
aspect_sin = np.sin(aspect_radians).rename("aspect_sin")
aspect_cos = np.cos(aspect_radians).rename("aspect_cos")

# The outer halo exists only to supply the 3x3 terrain kernel at retained-cell
# edges. Remove exactly that support before aggregation so it contributes to no
# 1-km feature value.
trimmed_arrays = {
    "elevation": elevation_mosaic.isel(y=slice(1, -1), x=slice(1, -1)),
    "slope": slope.isel(y=slice(1, -1), x=slice(1, -1)),
    "aspect_sin": aspect_sin.isel(y=slice(1, -1), x=slice(1, -1)),
    "aspect_cos": aspect_cos.isel(y=slice(1, -1), x=slice(1, -1)),
}


# %% Verify exact raster/grid alignment before relying on block aggregation
pixels_per_cell = ANALYSIS_CELL_SIZE_M / TARGET_RESOLUTION_M
if not pixels_per_cell.is_integer():
    raise ValueError("The 1-km cell size is not an integer multiple of 10 m.")
pixels_per_cell = int(pixels_per_cell)
pixels_per_full_cell = pixels_per_cell**2

trimmed_elevation = trimmed_arrays["elevation"]
x_spacing = float(abs(trimmed_elevation.x[1] - trimmed_elevation.x[0]))
y_spacing = float(abs(trimmed_elevation.y[1] - trimmed_elevation.y[0]))
if not np.allclose(
    (x_spacing, y_spacing),
    TARGET_RESOLUTION_M,
    atol=ALIGNMENT_TOLERANCE_M,
):
    raise ValueError("Terrain pixel spacing is not exactly 10 m.")
if trimmed_elevation.shape != EXPECTED_TRIMMED_SHAPE:
    raise ValueError(
        f"Expected trimmed terrain shape {EXPECTED_TRIMMED_SHAPE}; "
        f"found {trimmed_elevation.shape}."
    )
if any(size % pixels_per_cell for size in trimmed_elevation.shape):
    raise ValueError("Trimmed raster dimensions do not divide into 100-pixel blocks.")

expected_x_centers = grid_bounds[0] + TARGET_RESOLUTION_M / 2 + np.arange(
    trimmed_elevation.sizes["x"]
) * TARGET_RESOLUTION_M
expected_y_centers = grid_bounds[3] - TARGET_RESOLUTION_M / 2 - np.arange(
    trimmed_elevation.sizes["y"]
) * TARGET_RESOLUTION_M
if not (
    np.allclose(trimmed_elevation.x.to_numpy(), expected_x_centers)
    and np.allclose(trimmed_elevation.y.to_numpy(), expected_y_centers)
):
    raise ValueError("Trimmed terrain pixel boundaries do not align with the grid.")

rectangular_rows = trimmed_elevation.sizes["y"] // pixels_per_cell
rectangular_columns = trimmed_elevation.sizes["x"] // pixels_per_cell
rectangular_block_count = rectangular_rows * rectangular_columns


# %% Aggregate lazily to aligned 1-km blocks
# Exact alignment makes 100x100 block reduction both faster and less ambiguous
# than hundreds of thousands of independent polygon overlays. Means capture
# typical conditions, while population standard deviations capture within-cell
# terrain variability because the pixels constitute the complete cell.
# Coarsen data without auxiliary raster coordinates because this xarray version
# incorrectly forwards ``ddof`` to its coordinate-mean function during std().
# Validated 1-km block-center coordinates are assigned explicitly afterward.
aggregation_inputs = {
    name: xr.DataArray(array.data, dims=("y", "x"), name=name)
    for name, array in trimmed_arrays.items()
}
elevation_coarsener = aggregation_inputs["elevation"].coarsen(
    y=pixels_per_cell, x=pixels_per_cell, boundary="exact"
)
slope_coarsener = aggregation_inputs["slope"].coarsen(
    y=pixels_per_cell, x=pixels_per_cell, boundary="exact"
)
aspect_sin_coarsener = aggregation_inputs["aspect_sin"].coarsen(
    y=pixels_per_cell, x=pixels_per_cell, boundary="exact"
)
aspect_cos_coarsener = aggregation_inputs["aspect_cos"].coarsen(
    y=pixels_per_cell, x=pixels_per_cell, boundary="exact"
)

aggregated = xr.Dataset(
    {
        "elevation_mean": elevation_coarsener.mean(skipna=True),
        "elevation_std": elevation_coarsener.std(skipna=True, ddof=0),
        "slope_mean": slope_coarsener.mean(skipna=True),
        "slope_std": slope_coarsener.std(skipna=True, ddof=0),
        "aspect_sin_mean": aspect_sin_coarsener.mean(skipna=True),
        "aspect_cos_mean": aspect_cos_coarsener.mean(skipna=True),
    }
)
aggregate_x_centers = grid_bounds[0] + ANALYSIS_CELL_SIZE_M / 2 + np.arange(
    rectangular_columns
) * ANALYSIS_CELL_SIZE_M
aggregate_y_centers = grid_bounds[3] - ANALYSIS_CELL_SIZE_M / 2 - np.arange(
    rectangular_rows
) * ANALYSIS_CELL_SIZE_M
aggregated = aggregated.assign_coords(
    x=aggregate_x_centers,
    y=aggregate_y_centers,
)

terrain_dask_backed = {
    name: dask.is_dask_collection(array.data)
    for name, array in trimmed_arrays.items()
}
aggregate_dask_backed = {
    name: dask.is_dask_collection(array.data)
    for name, array in aggregated.data_vars.items()
}
if not all(terrain_dask_backed.values()):
    raise ValueError("A 10 m terrain array became eager before aggregation.")
if not all(aggregate_dask_backed.values()):
    raise ValueError("A 1-km aggregate array became eager before final compute.")

print("STRUCTURAL VALIDATION")
print("---------------------")
print(f"Retained grid cells: {len(analysis_grid):,}")
print(f"Retained grid bounds: {grid_bounds}")
print(f"Trimmed terrain dimensions (y, x): {trimmed_elevation.shape}")
print(f"Pixel spacing (x, y): {x_spacing:g} m, {y_spacing:g} m")
print(f"10 m pixels per 1-km side: {pixels_per_cell}")
print(f"10 m pixels per complete 1-km block: {pixels_per_full_cell:,}")
print(
    "Rectangular aggregate dimensions (y, x): "
    f"{rectangular_rows:,}, {rectangular_columns:,}"
)
print(f"Rectangular aggregate blocks: {rectangular_block_count:,}")
print(f"10 m terrain arrays Dask-backed: {terrain_dask_backed}")
print(f"1-km aggregate arrays Dask-backed: {aggregate_dask_backed}")


# %% Materialize only the compact rectangular 1-km summaries
# Computing after aggregation avoids creating any multi-billion-pixel local
# intermediate. Dask can also share common upstream tasks among these outputs.
aggregated_loaded = aggregated.compute()


# %% Map each retained cell deterministically to one aggregated block
aggregate_x = aggregated_loaded.x.to_numpy()
aggregate_y = aggregated_loaded.y.to_numpy()
centroids = analysis_grid.geometry.centroid
column_indices = np.rint(
    (centroids.x.to_numpy() - (grid_bounds[0] + ANALYSIS_CELL_SIZE_M / 2))
    / ANALYSIS_CELL_SIZE_M
).astype(np.int64)
row_indices = np.rint(
    ((grid_bounds[3] - ANALYSIS_CELL_SIZE_M / 2) - centroids.y.to_numpy())
    / ANALYSIS_CELL_SIZE_M
).astype(np.int64)

if not (
    ((0 <= row_indices) & (row_indices < rectangular_rows)).all()
    and ((0 <= column_indices) & (column_indices < rectangular_columns)).all()
):
    raise ValueError("A retained cell maps outside the rectangular aggregates.")
if not (
    np.allclose(aggregate_x[column_indices], centroids.x.to_numpy())
    and np.allclose(aggregate_y[row_indices], centroids.y.to_numpy())
):
    raise ValueError("A retained-cell centroid does not match its aggregate center.")

block_keys = row_indices * rectangular_columns + column_indices
if np.unique(block_keys).size != len(analysis_grid):
    raise ValueError("Retained cells do not map one-to-one onto aggregate blocks.")

feature_columns = [
    "elevation_mean",
    "elevation_std",
    "slope_mean",
    "slope_std",
    "aspect_sin_mean",
    "aspect_cos_mean",
]
terrain_features = pd.DataFrame({"cell_id": analysis_grid["cell_id"].to_numpy()})
for feature_name in feature_columns:
    terrain_features[feature_name] = aggregated_loaded[feature_name].to_numpy()[
        row_indices, column_indices
    ]

# Component-vector length expresses within-cell directional consistency. When
# all aspects are undefined, both means and therefore strength remain NaN.
terrain_features["aspect_strength"] = np.hypot(
    terrain_features["aspect_sin_mean"],
    terrain_features["aspect_cos_mean"],
)
terrain_features = terrain_features.sort_values("cell_id").reset_index(drop=True)


# %% Validate the final retained-cell feature table
if len(terrain_features) != EXPECTED_RETAINED_CELL_COUNT:
    raise ValueError("Final terrain feature table has an unexpected row count.")
if not terrain_features["cell_id"].is_unique:
    raise ValueError("Final terrain feature table contains duplicate cell IDs.")
if set(terrain_features["cell_id"]) != set(analysis_grid["cell_id"]):
    raise ValueError("Final terrain feature table lost or added a cell ID.")

if terrain_features["elevation_mean"].notna().any() and not np.isfinite(
    terrain_features.loc[
        terrain_features["elevation_mean"].notna(), "elevation_mean"
    ]
).all():
    raise ValueError("Valid mean elevations include a non-finite value.")
if (terrain_features["elevation_std"].dropna() < 0).any():
    raise ValueError("elevation_std contains a negative value.")
if not terrain_features["slope_mean"].dropna().between(0, 90).all():
    raise ValueError("slope_mean falls outside [0, 90] degrees.")
if (terrain_features["slope_std"].dropna() < 0).any():
    raise ValueError("slope_std contains a negative value.")
if not terrain_features["aspect_sin_mean"].dropna().between(
    -1 - COMPONENT_TOLERANCE, 1 + COMPONENT_TOLERANCE
).all():
    raise ValueError("aspect_sin_mean falls outside approximately [-1, 1].")
if not terrain_features["aspect_cos_mean"].dropna().between(
    -1 - COMPONENT_TOLERANCE, 1 + COMPONENT_TOLERANCE
).all():
    raise ValueError("aspect_cos_mean falls outside approximately [-1, 1].")
if not terrain_features["aspect_strength"].dropna().between(
    0, 1 + COMPONENT_TOLERANCE
).all():
    raise ValueError("aspect_strength falls outside approximately [0, 1].")

duplicate_cell_count = int(terrain_features["cell_id"].duplicated().sum())
cells_with_missing = int(terrain_features[feature_columns + ["aspect_strength"]].isna().any(axis=1).sum())
all_aspect_undefined = int(
    terrain_features[["aspect_sin_mean", "aspect_cos_mean"]].isna().all(axis=1).sum()
)

print("\nFINAL TERRAIN FEATURE QA/QC")
print("---------------------------")
for feature_name in feature_columns + ["aspect_strength"]:
    values = terrain_features[feature_name]
    valid_values = values.dropna()
    print(f"{feature_name}:")
    print(f"  valid count: {valid_values.size:,}")
    print(f"  missing count: {values.isna().sum():,}")
    print(f"  minimum: {valid_values.min():.6f}")
    print(f"  maximum: {valid_values.max():.6f}")
    print(f"  mean: {valid_values.mean():.6f}")
    print(f"  standard deviation: {valid_values.std(ddof=0):.6f}")

print(f"Total final rows: {len(terrain_features):,}")
print(f"Unique cell_id count: {terrain_features['cell_id'].nunique():,}")
print(f"Duplicate cell_id count: {duplicate_cell_count:,}")
print(f"Cells with any missing terrain feature: {cells_with_missing:,}")
print(f"Cells with all aspect values undefined: {all_aspect_undefined:,}")


# %% Write only the compact, deterministic retained-cell table
output_path.parent.mkdir(parents=True, exist_ok=True)
terrain_features.to_parquet(output_path, index=False)
print(f"Saved terrain features to: {output_path}")
print(f"Output file size: {output_path.stat().st_size / 1024**2:.2f} MiB")

# %%
