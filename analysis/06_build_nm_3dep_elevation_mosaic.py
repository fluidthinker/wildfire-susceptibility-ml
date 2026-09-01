"""Construct and lightly validate a lazy statewide 3DEP elevation mosaic."""

# %% Imports
import math
from pathlib import Path

import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import box
import stackstac


# %% Case-study parameters
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"
STAC_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:5070"
TARGET_EPSG = 5070
TARGET_RESOLUTION_M = 10
EXPECTED_10M_ITEM_COUNT = 46
CHUNK_SIZE_PIXELS = 2_048
QA_WINDOW_SIZE_PIXELS = 512

# A 3x3 terrain neighborhood reaches one pixel beyond its center. One 10 m
# pixel is therefore the minimal halo needed to calculate future slope/aspect
# values at the outer edge of every retained 1-km cell.
TERRAIN_HALO_M = TARGET_RESOLUTION_M

repo_root = Path(__file__).resolve().parents[1]
boundary_path = (
    repo_root / "data" / "processed" / "boundaries" / "nm_boundary.gpkg"
)
grid_path = (
    repo_root / "data" / "processed" / "grids" / "nm_analysis_grid_1km.gpkg"
)


# %% Read and validate the prepared boundary and retained analysis grid
for prepared_path in (boundary_path, grid_path):
    if not prepared_path.exists():
        raise FileNotFoundError(f"Required prepared dataset not found: {prepared_path}")

boundary = gpd.read_file(boundary_path, layer="state_boundary")
analysis_grid = gpd.read_file(grid_path, layer="analysis_grid")

if len(boundary) != 1:
    raise ValueError(
        "Expected one prepared New Mexico boundary feature, "
        f"but found {len(boundary)}."
    )
if boundary.crs is None or boundary.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Prepared boundary must use {TARGET_CRS}.")
if boundary.geometry.isna().any() or boundary.geometry.is_empty.any():
    raise ValueError("Prepared boundary contains missing or empty geometry.")
if not boundary.geometry.is_valid.all():
    raise ValueError("Prepared boundary contains invalid geometry.")

required_grid_columns = {"cell_id", "geometry"}
missing_grid_columns = required_grid_columns.difference(analysis_grid.columns)
if missing_grid_columns:
    raise ValueError(
        f"Prepared analysis grid is missing columns: {sorted(missing_grid_columns)}"
    )
if analysis_grid.empty:
    raise ValueError("Prepared analysis grid is empty.")
if analysis_grid.crs is None or analysis_grid.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Prepared analysis grid must use {TARGET_CRS}.")
if not analysis_grid["cell_id"].is_unique:
    raise ValueError("Prepared analysis grid cell IDs must be unique.")
if analysis_grid.geometry.isna().any() or analysis_grid.geometry.is_empty.any():
    raise ValueError("Prepared analysis grid contains missing or empty geometry.")
if not analysis_grid.geometry.is_valid.all():
    raise ValueError("Prepared analysis grid contains invalid geometry.")
if not analysis_grid.geom_type.eq("Polygon").all():
    raise ValueError("Prepared analysis grid must contain only Polygon geometries.")

boundary_wgs84 = boundary.to_crs(STAC_CRS)
query_geometry = boundary_wgs84.geometry.iloc[0].__geo_interface__
grid_bounds = tuple(float(value) for value in analysis_grid.total_bounds)

print("PREPARED INPUTS")
print("---------------")
print(f"Boundary: {boundary_path}")
print(f"Analysis grid: {grid_path}")
print("Analysis grid layer: analysis_grid")
print(f"Retained analysis cells: {len(analysis_grid):,}")
print(f"Analysis grid CRS: {analysis_grid.crs}")
print(f"Analysis grid columns: {list(analysis_grid.columns)}")
print(f"Full retained-grid bounds: {grid_bounds}")


# %% Search and validate statewide 3DEP source metadata
catalog = Client.open(PC_STAC_URL)
search = catalog.search(
    collections=[COLLECTION],
    intersects=query_geometry,
)
items = list(search.items())
if not items:
    raise ValueError(f"No Items from {COLLECTION!r} intersect New Mexico.")

gsd_values = sorted(
    {item.properties.get("gsd") for item in items},
    key=lambda value: (value is None, str(value)),
)
items_10m = [item for item in items if item.properties.get("gsd") == 10]
source_crs_values = sorted(
    {item.properties.get("proj:code") for item in items_10m},
    key=lambda value: (value is None, str(value)),
)

if len(items_10m) != EXPECTED_10M_ITEM_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_10M_ITEM_COUNT} qualifying 10 m Items, "
        f"but found {len(items_10m)}."
    )
if source_crs_values != ["EPSG:5498"]:
    raise ValueError(
        "Expected qualifying Items to report source CRS EPSG:5498, "
        f"but found {source_crs_values}."
    )

# Item IDs provide a stable source order independent of STAC response order.
# This order also defines precedence in the first-valid mosaic below.
selected_items = sorted(items_10m, key=lambda item: item.id)
if len({item.id for item in selected_items}) != len(selected_items):
    raise ValueError("Qualifying 10 m Items contain duplicate Item IDs.")

print("\n3DEP NEW MEXICO SOURCE QA/QC")
print("------------------------------")
print(f"Total Items returned: {len(items):,}")
print(f"10 m Items: {len(items_10m):,}")
print(f"Distinct GSD values: {gsd_values}")
print(f"Source proj:code values: {source_crs_values}")
print(f"Participating Item count: {len(selected_items):,}")
print(f"First Item by deterministic ID order: {selected_items[0].id}")
print(f"Last Item by deterministic ID order: {selected_items[-1].id}")


# %% Define the deterministic statewide target grid
# Full retained cells can extend outside the state polygon, so the DEM extent
# follows the complete analysis-grid bounds rather than clipping to New Mexico.
# Add the one-pixel terrain halo before snapping outward to the shared 10 m grid.
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


# %% Sign and construct the lazy statewide mosaic
# Signing is delayed until immediately before raster access so short-lived COG
# URL credentials remain fresh during construction and the sampled read.
signed_items = [pc.sign(item) for item in selected_items]

# Bilinear resampling is explicit because elevation is continuous. Large spatial
# chunks limit scheduler overhead while Dask keeps statewide reads deferred.
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

if elevation_stack.sizes.get("time") != len(selected_items):
    raise ValueError("The raster stack does not contain every selected Item.")
if elevation_stack.sizes.get("band") != 1:
    raise ValueError("Expected exactly one elevation band in the raster stack.")

elevation_by_item = elevation_stack.squeeze("band", drop=True)

# Match the validated prototype: the first sorted Item supplies valid pixels,
# and later Items fill only its NoData areas.
elevation_mosaic = stackstac.mosaic(
    elevation_by_item,
    dim="time",
    reverse=True,
    nodata=np.nan,
)

raster_height = elevation_mosaic.sizes["y"]
raster_width = elevation_mosaic.sizes["x"]
total_pixel_count = raster_height * raster_width
estimated_float32_size_gib = total_pixel_count * np.dtype("float32").itemsize / 1024**3

print("\nLAZY STATEWIDE ELEVATION MOSAIC")
print("--------------------------------")
print(f"Target CRS: {TARGET_CRS}")
print(f"Target resolution: {TARGET_RESOLUTION_M} m")
print(f"Terrain halo: {TERRAIN_HALO_M} m (one 10 m pixel)")
print(f"Target bounds: {target_bounds}")
print(f"Raster dimensions (y, x): {raster_height:,}, {raster_width:,}")
print(f"Approximate total pixel count: {total_pixel_count:,}")
print(f"Dask chunks: {elevation_mosaic.data.chunks}")
print(f"Lazy Dask-backed array: {dask.is_dask_collection(elevation_mosaic.data)}")
print(
    "Estimated uncompressed float32 raster size: "
    f"{estimated_float32_size_gib:.2f} GiB "
    "(size estimate, not actual RAM use)"
)


# %% Compute only a representative central QA window
# A small center window confirms remote raster access and plausible values while
# intentionally avoiding the multi-billion-pixel statewide computation.
center_y = raster_height // 2
center_x = raster_width // 2
half_qa_window = QA_WINDOW_SIZE_PIXELS // 2
qa_window = elevation_mosaic.isel(
    y=slice(center_y - half_qa_window, center_y + half_qa_window),
    x=slice(center_x - half_qa_window, center_x + half_qa_window),
)
qa_loaded = qa_window.compute()
qa_values = qa_loaded.to_numpy()
qa_valid_mask = np.isfinite(qa_values)
if not qa_valid_mask.any():
    raise ValueError("The representative QA window contains no valid elevations.")

qa_valid_elevations = qa_values[qa_valid_mask]
qa_nodata_proportion = 1 - qa_valid_mask.mean()

print("\nCENTRAL QA WINDOW")
print("-----------------")
print(f"Window dimensions (y, x): {qa_values.shape}")
print(f"Minimum elevation: {qa_valid_elevations.min():.2f} m")
print(f"Maximum elevation: {qa_valid_elevations.max():.2f} m")
print(f"Mean elevation: {qa_valid_elevations.mean():.2f} m")
print(f"NoData / NaN proportion: {qa_nodata_proportion:.6%}")


# %% Plot the lightweight extent relationships without rendering the 10 m DEM
figure, axis = plt.subplots(figsize=(9, 9))
dem_bounds = gpd.GeoSeries([box(*target_bounds)], crs=TARGET_CRS)
grid_extent = gpd.GeoSeries([box(*grid_bounds)], crs=TARGET_CRS)

dem_bounds.boundary.plot(
    ax=axis,
    color="tab:red",
    linewidth=1.5,
    linestyle="--",
    label="DEM processing bounds (10 m halo)",
)
grid_extent.boundary.plot(
    ax=axis,
    color="tab:blue",
    linewidth=1.5,
    label="Retained 1-km grid extent",
)
boundary.boundary.plot(
    ax=axis,
    color="black",
    linewidth=1.0,
    label="New Mexico boundary",
)

axis.set_title("New Mexico 3DEP Statewide Processing Extent QA")
axis.set_xlabel("Easting (m, EPSG:5070)")
axis.set_ylabel("Northing (m, EPSG:5070)")
axis.set_aspect("equal")
axis.legend(loc="upper right")
figure.tight_layout()
plt.show()

# %%
