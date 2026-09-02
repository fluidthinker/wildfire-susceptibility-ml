"""Derive and lightly validate lazy 10 m terrain features for New Mexico."""

# %% Imports
import math
from pathlib import Path

import dask
import dask.array as da
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import rioxarray  # noqa: F401  # Register the xarray ``.rio`` CRS accessor.
import xarray as xr
from pystac_client import Client
from rasterio.enums import Resampling
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
CHUNK_BOUNDARY_QA_HALF_WIDTH_PIXELS = 256

# This OUTER halo supplies the 3x3 neighborhood at the edge of every retained
# 1-km cell. It is part of the statewide processing extent and is distinct from
# the temporary INTERNAL overlap exchanged between Dask chunks below.
TERRAIN_HALO_M = TARGET_RESOLUTION_M
INTERNAL_OVERLAP_PIXELS = 1

repo_root = Path(__file__).resolve().parents[1]
boundary_path = (
    repo_root / "data" / "processed" / "boundaries" / "nm_boundary.gpkg"
)
grid_path = (
    repo_root / "data" / "processed" / "grids" / "nm_analysis_grid_1km.gpkg"
)


# %% Terrain finite-difference functions
def horn_terrain_block(
    elevation: np.ndarray,
    cell_size: float,
    derivative_name: str,
) -> np.ndarray:
    """Calculate one Horn terrain derivative for an overlapped elevation block.

    The standard Horn (1981) 3x3 weighted finite difference estimates the
    eastward and northward elevation gradients. Aspect is the downslope azimuth
    measured clockwise from north. A missing neighbor invalidates the center
    result, and flat pixels receive NaN aspect because direction is undefined.

    Args:
        elevation: Two-dimensional elevation block, including any Dask overlap.
        cell_size: Square raster-cell size in projected coordinate units.
        derivative_name: Terrain derivative to return: ``"slope"`` or
            ``"aspect"``.

    Returns:
        Array with the requested terrain derivative in degrees and the same
        shape as ``elevation``. Its outermost row and column are NaN.

    Raises:
        ValueError: If the input or requested output is invalid.
    """
    if elevation.ndim != 2:
        raise ValueError("Horn terrain calculation requires a 2D elevation array.")
    if cell_size <= 0:
        raise ValueError("Cell size must be positive.")
    if derivative_name not in {"slope", "aspect"}:
        raise ValueError("Derivative name must be 'slope' or 'aspect'.")

    result = np.full(elevation.shape, np.nan, dtype=np.float32)
    if min(elevation.shape) < 3:
        return result

    northwest = elevation[:-2, :-2]
    north = elevation[:-2, 1:-1]
    northeast = elevation[:-2, 2:]
    west = elevation[1:-1, :-2]
    east = elevation[1:-1, 2:]
    southwest = elevation[2:, :-2]
    south = elevation[2:, 1:-1]
    southeast = elevation[2:, 2:]

    dz_dx = (
        (northeast + 2 * east + southeast)
        - (northwest + 2 * west + southwest)
    ) / (8 * cell_size)
    # Raster rows increase southward, so reversing the row difference makes
    # this derivative positive toward geographic north in EPSG:5070.
    dz_dy = (
        (northwest + 2 * north + northeast)
        - (southwest + 2 * south + southeast)
    ) / (8 * cell_size)

    if derivative_name == "slope":
        interior = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    else:
        gradient_magnitude = np.hypot(dz_dx, dz_dy)
        # Compass azimuth uses atan2(east, north) for the downslope vector.
        interior = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360
        interior = np.where(gradient_magnitude == 0, np.nan, interior)

    result[1:-1, 1:-1] = interior.astype(np.float32, copy=False)
    return result


def lazy_horn_derivative(
    elevation: xr.DataArray,
    cell_size: float,
    output: str,
) -> xr.DataArray:
    """Create a lazy overlap-aware Horn terrain derivative.

    Args:
        elevation: Two-dimensional, Dask-backed elevation DataArray.
        cell_size: Square raster-cell size in projected coordinate units.
        output: Terrain derivative to return: ``"slope"`` or ``"aspect"``.

    Returns:
        Dask-backed DataArray on the same coordinates and chunks as elevation.

    Raises:
        ValueError: If elevation is not two-dimensional and Dask-backed.
    """
    if elevation.dims != ("y", "x") or not dask.is_dask_collection(elevation.data):
        raise ValueError("Elevation must be a 2D Dask-backed (y, x) DataArray.")

    # A 3x3 calculation needs one neighboring elevation pixel on every side.
    # map_overlap borrows those pixels across INTERNAL 2048-pixel chunk edges,
    # then trims the temporary overlap so the statewide shape is unchanged.
    derivative_data = da.map_overlap(
        horn_terrain_block,
        elevation.data,
        depth={0: INTERNAL_OVERLAP_PIXELS, 1: INTERNAL_OVERLAP_PIXELS},
        boundary=np.nan,
        trim=True,
        dtype=np.float32,
        cell_size=cell_size,
        derivative_name=output,
    )
    return elevation.copy(data=derivative_data).rename(output)


# %% Read and validate the prepared boundary and retained analysis grid
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

if analysis_grid.empty:
    raise ValueError("Prepared analysis grid is empty.")
if analysis_grid.crs is None or analysis_grid.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Prepared analysis grid must use {TARGET_CRS}.")
if not {"cell_id", "geometry"}.issubset(analysis_grid.columns):
    raise ValueError("Prepared analysis grid requires cell_id and geometry columns.")
if not analysis_grid["cell_id"].is_unique:
    raise ValueError("Prepared analysis grid cell IDs must be unique.")
if analysis_grid.geometry.isna().any() or analysis_grid.geometry.is_empty.any():
    raise ValueError("Prepared analysis grid contains missing or empty geometry.")
if not analysis_grid.geometry.is_valid.all():
    raise ValueError("Prepared analysis grid contains invalid geometry.")
if not analysis_grid.geom_type.eq("Polygon").all():
    raise ValueError("Prepared analysis grid must contain only Polygon geometries.")

query_geometry = boundary.to_crs(STAC_CRS).geometry.iloc[0].__geo_interface__
grid_bounds = tuple(float(value) for value in analysis_grid.total_bounds)


# %% Reproduce the validated deterministic statewide 3DEP mosaic from script 06
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

# Stable Item-ID order defines the same first-valid source precedence as the
# seam prototype and statewide elevation workflow.
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
elevation_mosaic = stackstac.mosaic(
    elevation_by_item, dim="time", reverse=True, nodata=np.nan
)


# %% Construct lazy slope, raw aspect, and circular aspect components
# Elevation must be mosaicked first: differentiating separate source Items would
# treat tile edges as terrain edges and could create false slopes at overlaps.
slope = lazy_horn_derivative(elevation_mosaic, TARGET_RESOLUTION_M, "slope")
aspect = lazy_horn_derivative(elevation_mosaic, TARGET_RESOLUTION_M, "aspect")

# Degrees wrap at north (359 degrees is close to 1 degree), so their arithmetic
# mean is unsuitable for later 1-km aggregation. Creating components now lets a
# later workflow aggregate circular direction safely. Undefined flat aspects
# remain NaN through radians, sine, and cosine.
aspect_radians = np.deg2rad(aspect)
aspect_sin = np.sin(aspect_radians).rename("aspect_sin")
aspect_cos = np.cos(aspect_radians).rename("aspect_cos")
terrain_arrays = {
    "slope": slope,
    "aspect": aspect,
    "aspect_sin": aspect_sin,
    "aspect_cos": aspect_cos,
}

# These checks inspect metadata and task graphs only; none materializes the
# approximately statewide arrays.
expected_shape = elevation_mosaic.shape
expected_chunks = elevation_mosaic.data.chunks
x_spacing = float(abs(elevation_mosaic.x[1] - elevation_mosaic.x[0]))
y_spacing = float(abs(elevation_mosaic.y[1] - elevation_mosaic.y[0]))
if elevation_mosaic.rio.crs is None or elevation_mosaic.rio.crs.to_epsg() != TARGET_EPSG:
    raise ValueError(f"Elevation mosaic must use {TARGET_CRS}.")
if not np.isclose([x_spacing, y_spacing], TARGET_RESOLUTION_M).all():
    raise ValueError("Elevation mosaic does not have the expected 10 m spacing.")
for name, terrain_array in terrain_arrays.items():
    if not dask.is_dask_collection(terrain_array.data):
        raise ValueError(f"{name} is not Dask-backed.")
    if terrain_array.shape != expected_shape:
        raise ValueError(f"{name} does not preserve the elevation mosaic shape.")
    if terrain_array.data.chunks != expected_chunks:
        raise ValueError(f"{name} does not preserve the elevation mosaic chunks.")
    if terrain_array.rio.crs is None or terrain_array.rio.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"{name} must use {TARGET_CRS}.")

print("LAZY NEW MEXICO TERRAIN ARRAYS")
print("--------------------------------")
print("Terrain method: Horn (1981) weighted 3x3 finite difference")
print(f"Target CRS: {TARGET_CRS}")
print(f"Pixel spacing (x, y): {x_spacing:g} m, {y_spacing:g} m")
print(f"Outer terrain halo: {TERRAIN_HALO_M} m")
print(f"Internal Dask overlap: {INTERNAL_OVERLAP_PIXELS} pixel")
print(f"Raster dimensions (y, x): {expected_shape[0]:,}, {expected_shape[1]:,}")
print(f"Dask chunks: {expected_chunks}")
for name, terrain_array in terrain_arrays.items():
    print(f"{name} lazy / Dask-backed: {dask.is_dask_collection(terrain_array.data)}")


# %% Compute only a representative central QA window
raster_height, raster_width = expected_shape
center_y = raster_height // 2
center_x = raster_width // 2
half_qa_window = QA_WINDOW_SIZE_PIXELS // 2
qa_indexers = {
    "y": slice(center_y - half_qa_window, center_y + half_qa_window),
    "x": slice(center_x - half_qa_window, center_x + half_qa_window),
}
qa_loaded = dask.compute(
    elevation_mosaic.isel(**qa_indexers),
    slope.isel(**qa_indexers),
    aspect.isel(**qa_indexers),
    aspect_sin.isel(**qa_indexers),
    aspect_cos.isel(**qa_indexers),
)
elevation_qa, slope_qa, aspect_qa, aspect_sin_qa, aspect_cos_qa = (
    array.to_numpy() for array in qa_loaded
)

elevation_valid = elevation_qa[np.isfinite(elevation_qa)]
slope_valid = slope_qa[np.isfinite(slope_qa)]
aspect_valid = aspect_qa[np.isfinite(aspect_qa)]
aspect_sin_valid = aspect_sin_qa[np.isfinite(aspect_sin_qa)]
aspect_cos_valid = aspect_cos_qa[np.isfinite(aspect_cos_qa)]
if any(values.size == 0 for values in (elevation_valid, slope_valid, aspect_valid)):
    raise ValueError("The representative QA window lacks valid terrain values.")
if slope_valid.min() < 0 or slope_valid.max() > 90:
    raise ValueError("Slope values fall outside the physically valid [0, 90] range.")
if aspect_valid.min() < 0 or aspect_valid.max() >= 360:
    raise ValueError("Valid aspect values fall outside the documented [0, 360) range.")
component_tolerance = 1e-6
if aspect_sin_valid.min() < -1 - component_tolerance or aspect_sin_valid.max() > 1 + component_tolerance:
    raise ValueError("aspect_sin values fall outside approximately [-1, 1].")
if aspect_cos_valid.min() < -1 - component_tolerance or aspect_cos_valid.max() > 1 + component_tolerance:
    raise ValueError("aspect_cos values fall outside approximately [-1, 1].")

print("\nCENTRAL TERRAIN QA WINDOW")
print("-------------------------")
print(f"Window dimensions (y, x): {elevation_qa.shape}")
print(f"Elevation min/max/mean: {elevation_valid.min():.2f} / {elevation_valid.max():.2f} / {elevation_valid.mean():.2f} m")
print(f"Slope min/max/mean: {slope_valid.min():.4f} / {slope_valid.max():.4f} / {slope_valid.mean():.4f} degrees")
print(f"Slope NoData proportion: {1 - np.isfinite(slope_qa).mean():.6%}")
print(f"Aspect valid-pixel count: {aspect_valid.size:,}")
print(f"Aspect min/max (valid): {aspect_valid.min():.4f} / {aspect_valid.max():.4f} degrees")
print(f"Aspect NoData / undefined proportion: {1 - np.isfinite(aspect_qa).mean():.6%}")
print(f"aspect_sin min/max: {aspect_sin_valid.min():.6f} / {aspect_sin_valid.max():.6f}")
print(f"aspect_cos min/max: {aspect_cos_valid.min():.6f} / {aspect_cos_valid.max():.6f}")


# %% Check a small window crossing an internal 2048-pixel chunk boundary
# Adjacent-column slope jumps at the tested chunk edge are compared with all
# other column transitions in the local window. An extreme outlier would flag a
# possible artificial straight seam; this remains a diagnostic, not a terrain
# smoothness assumption.
internal_x_boundaries = np.cumsum(expected_chunks[1])[:-1]
if internal_x_boundaries.size == 0:
    raise ValueError("No internal x chunk boundary is available for seam QA.")
boundary_x_index = int(internal_x_boundaries[len(internal_x_boundaries) // 2])
boundary_half_height = QA_WINDOW_SIZE_PIXELS // 2
boundary_indexers = {
    "y": slice(center_y - boundary_half_height, center_y + boundary_half_height),
    "x": slice(
        boundary_x_index - CHUNK_BOUNDARY_QA_HALF_WIDTH_PIXELS,
        boundary_x_index + CHUNK_BOUNDARY_QA_HALF_WIDTH_PIXELS,
    ),
}
boundary_slope = slope.isel(**boundary_indexers).compute()
boundary_slope_values = boundary_slope.to_numpy()
column_jumps = np.nanmean(np.abs(np.diff(boundary_slope_values, axis=1)), axis=0)
local_boundary_column = CHUNK_BOUNDARY_QA_HALF_WIDTH_PIXELS - 1
boundary_jump = float(column_jumps[local_boundary_column])
comparison_jumps = np.delete(column_jumps, local_boundary_column)
comparison_jumps = comparison_jumps[np.isfinite(comparison_jumps)]
if not np.isfinite(boundary_jump) or comparison_jumps.size == 0:
    raise ValueError("Chunk-boundary QA window lacks sufficient valid slopes.")
comparison_p99 = float(np.percentile(comparison_jumps, 99))
seam_flagged = boundary_jump > comparison_p99

print("\nINTERNAL CHUNK-BOUNDARY QA")
print("--------------------------")
print(f"Tested global x-index boundary: {boundary_x_index:,}")
print(f"Mean absolute slope jump at boundary: {boundary_jump:.6f} degrees")
print(f"99th percentile of other local column jumps: {comparison_p99:.6f} degrees")
print(f"Artificial processing seam flagged: {seam_flagged}")

figure, axis = plt.subplots(figsize=(10, 7))
boundary_slope.plot.imshow(
    ax=axis,
    cmap="viridis",
    robust=True,
    cbar_kwargs={"label": "Slope (degrees)"},
)
boundary_x_coordinate = float(elevation_mosaic.x[boundary_x_index])
axis.axvline(
    boundary_x_coordinate,
    color="magenta",
    linestyle="--",
    linewidth=1.5,
    label="Internal Dask chunk boundary",
)
axis.set_title("Slope QA Across an Internal Dask Chunk Boundary")
axis.set_xlabel("Easting (m, EPSG:5070)")
axis.set_ylabel("Northing (m, EPSG:5070)")
axis.legend(loc="upper right")
figure.tight_layout()
plt.show()

# Statewide elevation and terrain arrays intentionally remain lazy. This script
# neither trims/aggregates to 1 km nor writes any statewide raster product.

# %%
