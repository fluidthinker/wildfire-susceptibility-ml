"""Validate a lazy 3DEP elevation mosaic across an internal tile boundary."""

# %% Imports
import math
from pathlib import Path

import dask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
from pyproj import Transformer, network
from pystac_client import Client
from rasterio.enums import Resampling
from shapely.geometry import LineString
import stackstac


# %% Prototype parameters
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"
STAC_CRS = "EPSG:4326"
TARGET_CRS = "EPSG:5070"
TARGET_EPSG = 5070
TARGET_RESOLUTION_M = 10
CHUNK_SIZE_PIXELS = 256

# These adjacent one-degree Items meet at longitude -106 degrees. Keeping the
# IDs explicit makes the prototype reproducible if STAC result ordering changes.
TEST_ITEM_IDS = ("n35w106-13", "n35w107-13")
SOURCE_SEAM_LONGITUDE = -106.0
TEST_CENTER_LATITUDE = 34.5
TEST_HALF_WIDTH_M = 3_000

repo_root = Path(__file__).resolve().parents[1]
boundary_path = (
    repo_root / "data" / "processed" / "boundaries" / "nm_boundary.gpkg"
)


# %% Read and validate the prepared New Mexico boundary
# Reuse the authoritative project boundary rather than introducing a second
# study-area definition for raster prototyping.
if not boundary_path.exists():
    raise FileNotFoundError(
        f"Prepared New Mexico boundary not found at {boundary_path}. "
        "Run analysis/01_fetch_state_boundary.py first."
    )

boundary = gpd.read_file(boundary_path, layer="state_boundary")

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

boundary_wgs84 = boundary.to_crs(STAC_CRS)
query_geometry = boundary_wgs84.geometry.iloc[0].__geo_interface__


# %% Search and select the deterministic adjacent 10 m Items
# Search with the state geometry as in the completed metadata QA/QC, then keep
# only explicit 10 m Items before selecting the small prototype pair.
catalog = Client.open(PC_STAC_URL)
search = catalog.search(
    collections=[COLLECTION],
    intersects=query_geometry,
)
items = list(search.items())
items_10m = [item for item in items if item.properties.get("gsd") == 10]

items_by_id = {item.id: item for item in items_10m}
missing_test_item_ids = [
    item_id for item_id in TEST_ITEM_IDS if item_id not in items_by_id
]
if missing_test_item_ids:
    raise ValueError(
        "Required adjacent 10 m prototype Items were not returned: "
        f"{missing_test_item_ids}"
    )

# Sorting establishes the source precedence used by the first-valid mosaic.
selected_items = [items_by_id[item_id] for item_id in sorted(TEST_ITEM_IDS)]
source_crs_values = {
    item.properties.get("proj:code") for item in selected_items
}
if source_crs_values != {"EPSG:5498"}:
    raise ValueError(
        "Expected both prototype Items to report source CRS EPSG:5498, "
        f"but found {sorted(source_crs_values, key=str)}."
    )

print("3DEP MULTI-TILE PROTOTYPE")
print("-------------------------")
print(f"Participating Item IDs: {', '.join(item.id for item in selected_items)}")
print(f"Source CRS: {next(iter(source_crs_values))}")
print(f"Target CRS: {TARGET_CRS}")


# %% Define a deterministic EPSG:5070 target grid across the source seam
# The output bounds are snapped to exact 10 m coordinates so repeated runs use
# identical pixel alignment. PROJ network access is disabled only while choosing
# local coordinate operations; no datum-grid downloads are needed for this QA.
proj_network_was_enabled = network.is_network_enabled()
try:
    network.set_network_enabled(False)
    to_target = Transformer.from_crs(STAC_CRS, TARGET_CRS, always_xy=True)
    seam_center_x, seam_center_y = to_target.transform(
        SOURCE_SEAM_LONGITUDE,
        TEST_CENTER_LATITUDE,
    )

    target_bounds = (
        math.floor(
            (seam_center_x - TEST_HALF_WIDTH_M) / TARGET_RESOLUTION_M
        )
        * TARGET_RESOLUTION_M,
        math.floor(
            (seam_center_y - TEST_HALF_WIDTH_M) / TARGET_RESOLUTION_M
        )
        * TARGET_RESOLUTION_M,
        math.ceil(
            (seam_center_x + TEST_HALF_WIDTH_M) / TARGET_RESOLUTION_M
        )
        * TARGET_RESOLUTION_M,
        math.ceil(
            (seam_center_y + TEST_HALF_WIDTH_M) / TARGET_RESOLUTION_M
        )
        * TARGET_RESOLUTION_M,
    )

    # A projected seam line identifies the original source-tile boundary on
    # the diagnostic map without assuming it is vertical in EPSG:5070.
    seam_wgs84 = gpd.GeoSeries(
        [
            LineString(
                [
                    (SOURCE_SEAM_LONGITUDE, TEST_CENTER_LATITUDE - 0.1),
                    (SOURCE_SEAM_LONGITUDE, TEST_CENTER_LATITUDE + 0.1),
                ]
            )
        ],
        crs=STAC_CRS,
    )
    seam_5070 = seam_wgs84.to_crs(TARGET_CRS)
finally:
    network.set_network_enabled(proj_network_was_enabled)

print(f"Output resolution: {TARGET_RESOLUTION_M} m")
print(f"Target bounds: {target_bounds}")


# %% Sign immediately before constructing the remote raster stack
# Signing adds short-lived query parameters to COG URLs. Delaying it until this
# point minimizes credential expiry during metadata inspection.
signed_items = [pc.sign(item) for item in selected_items]

# Stackstac creates one common target grid and leaves pixel reads as Dask tasks.
# Bilinear resampling is explicit because elevation is continuous; this changes
# values only as required by reprojection, not as an overlap-mosaic statistic.
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
    raise ValueError("The raster stack does not contain both prototype Items.")
if elevation_stack.sizes.get("band") != 1:
    raise ValueError("Expected exactly one elevation band in the raster stack.")

elevation_by_item = elevation_stack.squeeze("band", drop=True)

# ``reverse=True`` gives the first, deterministically sorted Item precedence;
# later Items fill only pixels where the earlier source has NoData.
elevation_mosaic = stackstac.mosaic(
    elevation_by_item,
    dim="time",
    reverse=True,
    nodata=np.nan,
)

print("\nLAZY MOSAIC")
print("-----------")
print(f"Output dimensions (y, x): {elevation_mosaic.sizes['y']:,}, "
      f"{elevation_mosaic.sizes['x']:,}")
print(f"Dask chunks: {elevation_mosaic.data.chunks}")
print(f"Lazy Dask-backed array: {dask.is_dask_collection(elevation_mosaic.data)}")


# %% Load only the small prototype window and compute QA/QC statistics
# Computing the mosaic and two-source stack together allows Dask to share their
# read graph while limiting access to this 6 km test window rather than full COGs.
elevation_loaded, sources_loaded = dask.compute(
    elevation_mosaic,
    elevation_by_item,
)

elevation_values = elevation_loaded.to_numpy()
valid_pixel_mask = np.isfinite(elevation_values)
if not valid_pixel_mask.any():
    raise ValueError("The prototype mosaic contains no valid elevation pixels.")

valid_elevations = elevation_values[valid_pixel_mask]
nodata_proportion = 1 - valid_pixel_mask.mean()

# Compare the two sources only where both contain data. Small differences can
# reflect source vintages or reprojection, whereas a large systematic offset
# would warn that first-valid mosaicking could expose an artificial seam.
source_values = sources_loaded.to_numpy()
overlap_mask = np.isfinite(source_values).all(axis=0)
overlap_pixel_count = int(overlap_mask.sum())
if overlap_pixel_count == 0:
    raise ValueError(
        "The selected Items have no valid overlap pixels in the prototype window."
    )

overlap_absolute_difference = np.abs(
    source_values[0][overlap_mask] - source_values[1][overlap_mask]
)

print("\nPROTOTYPE ELEVATION QA/QC")
print("--------------------------")
print(f"Minimum elevation: {valid_elevations.min():.2f} m")
print(f"Maximum elevation: {valid_elevations.max():.2f} m")
print(f"Mean elevation: {valid_elevations.mean():.2f} m")
print(f"NoData / NaN proportion: {nodata_proportion:.6%}")
print(f"Two-source overlap pixels: {overlap_pixel_count:,}")
print(
    "Mean absolute overlap difference: "
    f"{overlap_absolute_difference.mean():.3f} m"
)
print(
    "95th-percentile absolute overlap difference: "
    f"{np.percentile(overlap_absolute_difference, 95):.3f} m"
)
print(
    "Maximum absolute overlap difference: "
    f"{overlap_absolute_difference.max():.3f} m"
)


# %% Plot elevation and the original source-tile boundary
figure, axis = plt.subplots(figsize=(10, 8))
elevation_loaded.plot.imshow(
    ax=axis,
    cmap="terrain",
    robust=True,
    cbar_kwargs={"label": "Elevation (m)"},
)
seam_5070.plot(
    ax=axis,
    color="magenta",
    linewidth=1.5,
    linestyle="--",
    label="Original source-tile boundary",
)

axis.set_xlim(target_bounds[0], target_bounds[2])
axis.set_ylim(target_bounds[1], target_bounds[3])
axis.set_title("3DEP 10 m Elevation Mosaic Across an Internal Tile Boundary")
axis.set_xlabel("Easting (m, EPSG:5070)")
axis.set_ylabel("Northing (m, EPSG:5070)")
axis.set_aspect("equal")
axis.legend(loc="upper right")
figure.tight_layout()
plt.show()
