"""Validate annual PRISM precipitation normals on a small New Mexico grid subset."""

# %% Imports
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds


# %% Product and prototype parameters
TARGET_CRS = "EPSG:5070"
TARGET_EPSG = 5070
ANALYSIS_CELL_SIZE_M = 1_000
INTEGRATION_RESOLUTION_M = 100
TEST_GRID_SIDE_CELLS = 8

# PRISM's official data directory provides the current Norm91m annual
# precipitation normal without credentials or a third-party data platform.
PRISM_PRODUCT = "PRISM Norm91m 1991-2020 annual precipitation normal, M4"
PRISM_URL = (
    "https://data.prism.oregonstate.edu/normals/us/800m/ppt/monthly/"
    "prism_ppt_us_30s_2020_avg_30y.zip"
)
PRISM_ARCHIVE_NAME = "prism_ppt_us_30s_2020_avg_30y.zip"
PRISM_VARIABLE = "ppt"
PRISM_UNITS = "mm"
EXPECTED_SOURCE_RESOLUTION_DEGREES = 1 / 120
EXPECTED_SOURCE_NODATA = -9999.0

repo_root = Path(__file__).resolve().parents[1]
grid_path = repo_root / "data" / "processed" / "grids" / "nm_analysis_grid_1km.gpkg"
prism_dir = repo_root / "data" / "raw" / "prism"
prism_archive_path = prism_dir / PRISM_ARCHIVE_NAME


# %% Acquire the authoritative annual-normal archive
def download_source_if_missing(url: str, destination: Path) -> None:
    """Download a source archive safely when it is not already cached.

    The response is streamed in bounded chunks so the national archive is never
    held fully in memory. Bytes first go to a ``.part`` file and are renamed only
    after a complete transfer, preventing an interrupted download from looking
    like a reusable source archive.

    Args:
        url: Authoritative HTTPS source URL.
        destination: Local path for the completed archive.

    Raises:
        OSError: If local directories or files cannot be written.
        urllib.error.URLError: If the source cannot be reached.
    """
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "wildfire-susceptibility-ml/0.1"})
    try:
        with urlopen(request) as response, partial_path.open("wb") as output:
            # Chunking bounds memory use independently of the download size.
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        # The final filename signals that every response byte was received.
        partial_path.replace(destination)
    finally:
        # Never allow a failed transfer to become the next run's cached input.
        if partial_path.exists():
            partial_path.unlink()


def validate_prism_archive(archive_path: Path) -> str:
    """Validate the ZIP and exact PRISM product, then return its raster path.

    ZIP checks establish transport integrity; the embedded metadata checks the
    scientific identity of the Norm91m 1991--2020 annual M4 product. Rasterio's
    ZIP virtual filesystem then avoids extracting a second national raster.

    Args:
        archive_path: Downloaded authoritative PRISM archive.

    Returns:
        Rasterio ``zip://`` path to the archive's single GeoTIFF.

    Raises:
        ValueError: If the ZIP, member layout, normal period, or version differs
            from the validated source product.
    """
    try:
        with ZipFile(archive_path) as archive:
            # Opening a ZIP is insufficient: testzip() verifies member checksums.
            bad_member = archive.testzip()
            member_names = archive.namelist()
            tif_members = [name for name in member_names if name.lower().endswith(".tif")]
            info_members = [
                name for name in member_names if name.lower().endswith(".info.txt")
            ]
    except BadZipFile as error:
        raise ValueError(f"PRISM source is not a valid ZIP archive: {archive_path}") from error
    if bad_member is not None:
        raise ValueError(f"PRISM archive contains a corrupt member: {bad_member}")
    if len(tif_members) != 1:
        raise ValueError(f"Expected one PRISM COG in the archive; found {len(tif_members)}.")
    if len(info_members) != 1:
        raise ValueError(f"Expected one PRISM info file in the archive; found {len(info_members)}.")

    with ZipFile(archive_path) as archive:
        source_info = archive.read(info_members[0]).decode("utf-8")
    if "PRISM_DATASET_TYPE: an91/r2207d, normals/9120.a" not in source_info:
        raise ValueError("PRISM source does not identify the expected 1991-2020 annual normal.")
    if "PRISM_DATASET_VERSION: M4" not in source_info:
        raise ValueError("PRISM precipitation normal is not the expected M4 version.")

    return f"zip://{archive_path.as_posix()}!{tif_members[0]}"


# -----------------------------------------------------------------------------
# STEP 1 — Acquire and validate the authoritative PRISM source product
# -----------------------------------------------------------------------------
if not grid_path.exists():
    raise FileNotFoundError(f"Required prepared analysis grid not found: {grid_path}")
download_source_if_missing(PRISM_URL, prism_archive_path)
prism_raster_path = validate_prism_archive(prism_archive_path)


# %% Select a compact, contiguous block from the authoritative analysis grid
def load_and_select_prototype_grid(
    path: Path,
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float]]:
    """Load the authoritative grid and select its central 8-by-8 block.

    The nearest cell to the state's geometric center anchors bounds snapped to
    the existing 1-km EPSG:5070 lattice. Complete authoritative geometries and
    their ``cell_id`` values are retained; no cells are clipped or reconstructed.

    Args:
        path: GeoPackage containing the prepared ``analysis_grid`` layer.

    Returns:
        The selected 64 cells and ``left, bottom, right, top`` bounds.

    Raises:
        ValueError: If CRS, identifiers, cell count, or extent is unexpected.
    """
    analysis_grid = gpd.read_file(path, layer="analysis_grid")
    if analysis_grid.crs is None or analysis_grid.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"Prepared analysis grid must use {TARGET_CRS}.")
    if "cell_id" not in analysis_grid or not analysis_grid["cell_id"].is_unique:
        raise ValueError("Prepared analysis-grid cell_id values must exist and be unique.")

    centroids = analysis_grid.geometry.centroid
    state_center = analysis_grid.geometry.union_all().centroid
    squared_distance = (
        (centroids.x - state_center.x) ** 2
        + (centroids.y - state_center.y) ** 2
    )
    central_centroid = centroids.loc[squared_distance.idxmin()]

    # A full rectangle on the existing lattice makes raster-to-cell alignment
    # explicit and testable while preserving the validated prototype location.
    selection_left = (
        central_centroid.x
        - (TEST_GRID_SIDE_CELLS // 2) * ANALYSIS_CELL_SIZE_M
    )
    selection_bottom = (
        central_centroid.y
        - (TEST_GRID_SIDE_CELLS // 2) * ANALYSIS_CELL_SIZE_M
    )
    selection_right = selection_left + TEST_GRID_SIDE_CELLS * ANALYSIS_CELL_SIZE_M
    selection_top = selection_bottom + TEST_GRID_SIDE_CELLS * ANALYSIS_CELL_SIZE_M
    selection_mask = (
        centroids.x.ge(selection_left)
        & centroids.x.lt(selection_right)
        & centroids.y.ge(selection_bottom)
        & centroids.y.lt(selection_top)
    )
    test_grid = analysis_grid.loc[selection_mask, ["cell_id", "geometry"]].copy()

    if len(test_grid) != TEST_GRID_SIDE_CELLS**2:
        raise ValueError("The central prototype selection is not a complete 8 x 8 cell block.")
    if not 25 <= len(test_grid) <= 100:
        raise ValueError("Prototype test-cell count must remain between 25 and 100.")
    prototype_bounds = tuple(float(value) for value in test_grid.total_bounds)
    left, bottom, right, top = prototype_bounds
    expected_side_length = TEST_GRID_SIDE_CELLS * ANALYSIS_CELL_SIZE_M
    if not np.allclose((right - left, top - bottom), expected_side_length):
        raise ValueError("Prototype cells do not form the expected square extent.")
    return test_grid, prototype_bounds


# -----------------------------------------------------------------------------
# STEP 2 — Load the authoritative grid and select a complete central block
# -----------------------------------------------------------------------------
test_grid, prototype_bounds = load_and_select_prototype_grid(grid_path)
left, bottom, right, top = prototype_bounds


# %% Inspect only the native PRISM pixels supporting the test extent
# -----------------------------------------------------------------------------
# STEP 3 — Inspect native support pixels and create the integration surface
# -----------------------------------------------------------------------------
with rasterio.open(prism_raster_path) as source:
    if source.crs is None:
        raise ValueError("PRISM source raster does not declare a CRS.")
    if not np.allclose(source.res, EXPECTED_SOURCE_RESOLUTION_DEGREES):
        raise ValueError(f"Unexpected PRISM source resolution: {source.res}")
    if source.nodata != EXPECTED_SOURCE_NODATA:
        raise ValueError(f"Unexpected PRISM NoData value: {source.nodata}")

    source_crs = source.crs
    source_resolution = source.res
    source_bounds = source.bounds
    source_dimensions = (source.height, source.width)
    source_nodata = source.nodata
    # The target extent must be expressed in PRISM's native CRS before Rasterio
    # can translate geographic bounds into source pixel rows and columns.
    # Densifying the edges represents curvature introduced by the CRS transform.
    native_test_bounds = transform_bounds(
        TARGET_CRS, source.crs, left, bottom, right, top, densify_pts=21
    )
    # A raster window is a bounded rectangular read. It lets this prototype read
    # only source pixels supporting the 8-km test area, not the national raster.
    fractional_window = from_bounds(*native_test_bounds, transform=source.transform)
    # Bilinear interpolation consults neighboring pixels around each destination
    # location. The one-pixel halo ensures QA extrema include every possible
    # native contributor, including pixels just outside the strict test bounds.
    column_start = np.floor(fractional_window.col_off).astype(int) - 1
    row_start = np.floor(fractional_window.row_off).astype(int) - 1
    column_stop = np.ceil(
        fractional_window.col_off + fractional_window.width
    ).astype(int) + 1
    row_stop = np.ceil(
        fractional_window.row_off + fractional_window.height
    ).astype(int) + 1
    native_window = Window(
        column_start,
        row_start,
        column_stop - column_start,
        row_stop - row_start,
    )
    native_window = native_window.intersection(Window(0, 0, source.width, source.height))
    native_values = source.read(1, window=native_window, masked=True)
    if native_values.count() == 0:
        raise ValueError("The selected test area has no valid PRISM source pixels.")
    source_test_min = float(native_values.min())
    source_test_max = float(native_values.max())

    output_width = int((right - left) / INTEGRATION_RESOLUTION_M)
    output_height = int((top - bottom) / INTEGRATION_RESOLUTION_M)
    # Anchoring the 100-m raster at the 1-km grid bounds creates exactly 10 rows
    # and 10 columns per analysis cell. This is an integration grid only: it does
    # not create observations or climate information at 100-m resolution.
    output_transform = from_origin(
        left, top, INTEGRATION_RESOLUTION_M, INTEGRATION_RESOLUTION_M
    )
    precipitation_100m = np.full((output_height, output_width), np.nan, dtype=np.float32)

    # reproject() maps PRISM pixel locations into EPSG:5070, estimates values at
    # destination locations, and writes them onto the aligned 100-m grid. Bilinear
    # interpolation is used because precipitation is continuous; sampling that
    # surface at 100 m and averaging 10 x 10 samples approximates a cell area mean.
    reproject(
        source=rasterio.band(source, 1),
        destination=precipitation_100m,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=output_transform,
        dst_crs=TARGET_CRS,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        init_dest_nodata=True,
    )


# %% Aggregate the processed surface to one mean per authoritative 1-km cell
# -----------------------------------------------------------------------------
# STEP 4 — Aggregate aligned samples and attach means to authoritative cell IDs
# -----------------------------------------------------------------------------
def aggregate_to_analysis_cells(
    values: np.ndarray,
    grid: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Average aligned 100-m samples and attach them to authoritative cells.

    Each 1-km cell contains an aligned 10-by-10 block. Means are mapped by the
    authoritative cell centroids' raster positions, not GeoDataFrame row order.

    Args:
        values: Reprojected 100-m precipitation integration surface.
        grid: Selected authoritative analysis cells.
        bounds: Shared grid and raster bounds in EPSG:5070.

    Returns:
        Grid with raster indices and precipitation, and a sorted feature table.

    Raises:
        ValueError: If surface shape, completeness, or value domain is invalid.
    """
    samples_per_cell = ANALYSIS_CELL_SIZE_M // INTEGRATION_RESOLUTION_M
    expected_side = TEST_GRID_SIDE_CELLS * samples_per_cell
    if values.shape != (expected_side, expected_side):
        raise ValueError(f"Unexpected processed raster shape: {values.shape}")
    if np.isnan(values).any():
        raise ValueError("Unexpected missing precipitation within the prototype area.")
    if (values < 0).any():
        raise ValueError("Processed precipitation contains negative values.")

    # Reorganize into analysis rows x fine rows x analysis columns x fine columns.
    grouped_values = values.reshape(
        TEST_GRID_SIDE_CELLS,
        samples_per_cell,
        TEST_GRID_SIDE_CELLS,
        samples_per_cell,
    )
    # Average only the fine axes, leaving one mean for every 1-km cell.
    cell_means = grouped_values.mean(axis=(1, 3))

    left, _, _, top = bounds
    grid = grid.copy()
    test_centroids = grid.geometry.centroid
    # Rows increase south from the top; columns increase east from the left. The
    # half-cell offset converts centroid coordinates into zero-based positions.
    row_positions = (top - test_centroids.y) / ANALYSIS_CELL_SIZE_M - 0.5
    column_positions = (test_centroids.x - left) / ANALYSIS_CELL_SIZE_M - 0.5
    grid["row_index"] = np.rint(row_positions).astype(int)
    grid["column_index"] = np.rint(column_positions).astype(int)
    grid["annual_precip_mean"] = cell_means[
        grid["row_index"], grid["column_index"]
    ]
    table = (
        grid[["cell_id", "annual_precip_mean"]]
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    return grid, table


test_grid, prototype_table = aggregate_to_analysis_cells(
    precipitation_100m,
    test_grid,
    prototype_bounds,
)

missing_count = int(prototype_table["annual_precip_mean"].isna().sum())
duplicate_count = int(prototype_table["cell_id"].duplicated().sum())
if len(prototype_table) != len(test_grid) or duplicate_count:
    raise ValueError("Prototype output is not one row per selected analysis cell.")
if missing_count:
    raise ValueError("Prototype output contains unexpected missing precipitation.")
if not np.isfinite(prototype_table["annual_precip_mean"]).all():
    raise ValueError("Prototype output contains non-finite precipitation.")
if (prototype_table["annual_precip_mean"] < 0).any():
    raise ValueError("Prototype output contains negative precipitation.")


# %% Report source metadata and prototype QA/QC
# -----------------------------------------------------------------------------
# STEP 5 — Report source identity, coverage, and keyed-table QA/QC
# -----------------------------------------------------------------------------
print("PRISM SOURCE METADATA")
print("---------------------")
print(f"Access method: direct HTTPS from the authoritative PRISM data directory")
print(f"Product: {PRISM_PRODUCT}")
print("Normal period: 1991-2020")
print(f"Variable / units: annual precipitation / {PRISM_UNITS}")
print("Annual interpretation: authoritative cumulative annual normal")
print(f"Source CRS: {source_crs}")
print(f"Source resolution: {source_resolution[0]:.12f} x {source_resolution[1]:.12f} degrees (~800 m)")
print(f"Source raster dimensions (rows, columns): {source_dimensions}")
print(f"Source raster bounds: {tuple(source_bounds)}")
print(f"Source NoData value: {source_nodata}")
print(f"Source min/max over test window: {source_test_min:.6f} / {source_test_max:.6f} {PRISM_UNITS}")

print("\nPROTOTYPE QA/QC")
print("----------------")
print(f"Selected 1-km cells: {len(test_grid):,}")
print(f"Integration grid: {INTEGRATION_RESOLUTION_M} m bilinear surface in {TARGET_CRS}")
print(f"Output precipitation min: {prototype_table['annual_precip_mean'].min():.6f} {PRISM_UNITS}")
print(f"Output precipitation max: {prototype_table['annual_precip_mean'].max():.6f} {PRISM_UNITS}")
print(f"Output precipitation mean: {prototype_table['annual_precip_mean'].mean():.6f} {PRISM_UNITS}")
print(f"Missing output values: {missing_count:,}")
print(f"Duplicate cell_id count: {duplicate_count:,}")
print(f"Unique cell_id count: {prototype_table['cell_id'].nunique():,}")
print("\nFirst five prototype rows:")
print(prototype_table.head().to_string(index=False))


# %% Lightweight spatial QA plot
# -----------------------------------------------------------------------------
# STEP 6 — Visually check extent, orientation, and grid alignment
# -----------------------------------------------------------------------------
fig, axis = plt.subplots(figsize=(8, 7))
image = axis.imshow(
    precipitation_100m,
    extent=(left, right, bottom, top),
    origin="upper",
    cmap="Blues",
)
test_grid.boundary.plot(ax=axis, color="black", linewidth=0.7)
test_grid.geometry.centroid.plot(ax=axis, color="darkorange", markersize=7)
fig.colorbar(image, ax=axis, label="Annual precipitation normal (mm)")
axis.set_title("PRISM 1991-2020 annual precipitation prototype")
axis.set_xlabel("EPSG:5070 x (m)")
axis.set_ylabel("EPSG:5070 y (m)")
axis.set_aspect("equal")
fig.tight_layout()
plt.show()

# %%
