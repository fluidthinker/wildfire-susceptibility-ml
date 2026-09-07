"""Build statewide New Mexico PRISM annual precipitation features."""

# %% Imports
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds


# %% Parameters and paths
TARGET_CRS = "EPSG:5070"
TARGET_EPSG = 5070
ANALYSIS_CELL_SIZE_M = 1_000
INTEGRATION_RESOLUTION_M = 100
EXPECTED_RETAINED_CELL_COUNT = 314_920
EXPECTED_SOURCE_CRS = "EPSG:4269"
EXPECTED_SOURCE_RESOLUTION_DEGREES = 1 / 120
EXPECTED_SOURCE_NODATA = -9999.0

PRISM_PRODUCT = "PRISM Norm91m 1991-2020 annual precipitation normal, M4"
PRISM_URL = (
    "https://data.prism.oregonstate.edu/normals/us/800m/ppt/monthly/"
    "prism_ppt_us_30s_2020_avg_30y.zip"
)
PRISM_ARCHIVE_NAME = "prism_ppt_us_30s_2020_avg_30y.zip"
FEATURE_COLUMN = "annual_precip_mean"
FINAL_COLUMNS = ["cell_id", FEATURE_COLUMN]

repo_root = Path(__file__).resolve().parents[1]
grid_path = repo_root / "data" / "processed" / "grids" / "nm_analysis_grid_1km.gpkg"
prism_archive_path = (
    repo_root / "data" / "raw" / "prism" / PRISM_ARCHIVE_NAME
)
output_path = (
    repo_root
    / "data"
    / "processed"
    / "features"
    / "nm_prism_features_1km.parquet"
)


# %% Helper functions

def download_binary_file_if_missing(url: str, destination: Path) -> None:
    """Download the PRISM authoritative archive safely when it is not cached.

    The response is streamed in bounded chunks so the archive is never held
    fully in memory. Bytes first go to a .part file and are renamed only after
    a complete transfer, preventing an interrupted download from looking like
    a reusable source archive.

    Args:
        url: Direct authoritative HTTPS source URL.
        destination: Local cache path for the completed archive.

    Raises:
        OSError: If local directories or files cannot be written.
        urllib.error.URLError: If the authoritative source cannot be reached.
    """
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "wildfire-susceptibility-ml/0.1"})
    try:
        with urlopen(request) as response, partial_path.open("wb") as output:
            # Chunked copying bounds memory use independently of download size.
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        # Only the complete archive receives the stable reusable filename.
        partial_path.replace(destination)
    finally:
        # A failed transfer must not become the next run's cached input.
        if partial_path.exists():
            partial_path.unlink()


def validate_prism_archive(archive_path: Path) -> str:
    """Validate ZIP integrity and the exact approved PRISM product.

    The information member must identify the Norm91m 1991-2020 annual normal
    and M4 version. The returned virtual path lets Rasterio access the GeoTIFF
    without extracting and duplicating the national raster.

    Args:
        archive_path: Downloaded authoritative PRISM ZIP archive.

    Returns:
        Rasterio ZIP virtual-filesystem path to the single GeoTIFF.

    Raises:
        ValueError: If ZIP integrity, member layout, period, or version differs
            from the approved product.
    """
    try:
        with ZipFile(archive_path) as archive:
            # Opening a ZIP alone does not verify its compressed members.
            bad_member = archive.testzip()
            member_names = archive.namelist()
            tif_members = [
                name for name in member_names if name.lower().endswith(".tif")
            ]
            info_members = [
                name for name in member_names if name.lower().endswith(".info.txt")
            ]
    except BadZipFile as error:
        raise ValueError(
            f"PRISM source is not a valid ZIP archive: {archive_path}"
        ) from error

    if bad_member is not None:
        raise ValueError(f"PRISM archive contains a corrupt member: {bad_member}")
    if len(tif_members) != 1:
        raise ValueError(
            f"Expected one PRISM GeoTIFF; found {len(tif_members)}."
        )
    if len(info_members) != 1:
        raise ValueError(
            f"Expected one PRISM info file; found {len(info_members)}."
        )

    with ZipFile(archive_path) as archive:
        source_info = archive.read(info_members[0]).decode("utf-8")
    if "PRISM_DATASET_TYPE: an91/r2207d, normals/9120.a" not in source_info:
        raise ValueError(
            "PRISM source does not identify the expected 1991-2020 annual normal."
        )
    if "PRISM_DATASET_VERSION: M4" not in source_info:
        raise ValueError("PRISM precipitation normal is not the expected M4 version.")

    return f"zip://{archive_path.as_posix()}!{tif_members[0]}"


def load_analysis_grid(path: Path) -> gpd.GeoDataFrame:
    """Load and validate the authoritative retained 1-km analysis cells.

    The prepared EPSG:5070 geometries and cell_id values are authoritative.
    This workflow neither rebuilds cells nor invents identifiers.

    Args:
        path: GeoPackage containing the analysis_grid layer.

    Returns:
        Validated statewide grid containing cell_id and geometry.

    Raises:
        FileNotFoundError: If the prepared grid does not exist.
        ValueError: If schema, CRS, count, IDs, or geometry is unexpected.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required analysis grid not found: {path}")

    grid = gpd.read_file(path, layer="analysis_grid")
    if not {"cell_id", "geometry"}.issubset(grid.columns):
        raise ValueError("Prepared analysis grid requires cell_id and geometry.")
    if grid.crs is None or grid.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"Prepared analysis grid must use {TARGET_CRS}.")
    if len(grid) != EXPECTED_RETAINED_CELL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RETAINED_CELL_COUNT:,} retained cells; "
            f"found {len(grid):,}."
        )
    if grid["cell_id"].nunique() != EXPECTED_RETAINED_CELL_COUNT:
        raise ValueError("Prepared analysis-grid cell_id values must be unique.")
    if grid.geometry.isna().any() or grid.geometry.is_empty.any():
        raise ValueError("Prepared analysis grid has missing or empty geometry.")
    return grid[["cell_id", "geometry"]].copy()


def determine_aligned_processing_extent(
    grid: gpd.GeoDataFrame,
) -> tuple[tuple[float, float, float, float], int, int, float]:
    """Determine the minimum aligned statewide 100-m processing raster.

    Complete 1-km cells share the authoritative lattice. Total bounds must be
    divisible by both 1,000 m and 100 m, yielding exactly ten integration
    pixels along each dimension of every analysis cell.

    Args:
        grid: Authoritative statewide EPSG:5070 analysis grid.

    Returns:
        Bounds, raster height, raster width, and float32 array size in GiB.

    Raises:
        ValueError: If bounds are unaligned or dimensions are invalid.
    """
    bounds = tuple(float(value) for value in grid.total_bounds)
    left, bottom, right, top = bounds
    coordinates = np.asarray(bounds)
    lattice_positions = coordinates / ANALYSIS_CELL_SIZE_M
    if not np.allclose(lattice_positions, np.rint(lattice_positions)):
        raise ValueError("Statewide bounds are not aligned to the 1-km lattice.")

    extent_width_m = right - left
    extent_height_m = top - bottom
    if (
        extent_width_m % INTEGRATION_RESOLUTION_M
        or extent_height_m % INTEGRATION_RESOLUTION_M
    ):
        raise ValueError("Statewide extent is not divisible by 100 m.")

    width = int(extent_width_m / INTEGRATION_RESOLUTION_M)
    height = int(extent_height_m / INTEGRATION_RESOLUTION_M)
    if width <= 0 or height <= 0:
        raise ValueError("Statewide processing dimensions must be positive.")
    destination_bytes = height * width * np.dtype(np.float32).itemsize
    return bounds, height, width, destination_bytes / 1024**3


def validate_prism_raster(
    raster_path: str,
    target_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Validate native metadata and inspect statewide source support.

    EPSG:5070 statewide bounds are transformed into the native CRS before a
    Rasterio window is calculated. A one-pixel halo is included because
    bilinear interpolation can consult neighbors just outside strict bounds.

    Args:
        raster_path: Rasterio path to the archived PRISM GeoTIFF.
        target_bounds: Statewide processing bounds in EPSG:5070.

    Returns:
        Source metadata and valid-value extrema for the support window.

    Raises:
        ValueError: If CRS, resolution, NoData, or source support is unexpected.
    """
    left, bottom, right, top = target_bounds
    with rasterio.open(raster_path) as source:
        if source.crs is None or source.crs.to_string() != EXPECTED_SOURCE_CRS:
            raise ValueError(
                f"PRISM source must use {EXPECTED_SOURCE_CRS}; found {source.crs}."
            )
        if not np.allclose(source.res, EXPECTED_SOURCE_RESOLUTION_DEGREES):
            raise ValueError(f"Unexpected PRISM source resolution: {source.res}")
        if source.nodata != EXPECTED_SOURCE_NODATA:
            raise ValueError(f"Unexpected PRISM NoData value: {source.nodata}")

        # Transform bounds before translating them into native raster rows and
        # columns. Densified edges represent curvature between the two CRSs.
        native_bounds = transform_bounds(
            TARGET_CRS,
            source.crs,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
        fractional_window = from_bounds(*native_bounds, transform=source.transform)

        # The source window limits QA reading to pixels supporting New Mexico.
        # Its halo includes neighbors potentially used by bilinear interpolation.
        column_start = int(np.floor(fractional_window.col_off)) - 1
        row_start = int(np.floor(fractional_window.row_off)) - 1
        column_stop = int(
            np.ceil(fractional_window.col_off + fractional_window.width)
        ) + 1
        row_stop = int(
            np.ceil(fractional_window.row_off + fractional_window.height)
        ) + 1
        support_window = Window(
            column_start,
            row_start,
            column_stop - column_start,
            row_stop - row_start,
        )
        support_window = support_window.intersection(
            Window(0, 0, source.width, source.height)
        )
        support_values = source.read(1, window=support_window, masked=True)
        if support_values.count() == 0:
            raise ValueError("New Mexico has no valid PRISM source support.")

        return {
            "crs": source.crs,
            "resolution": source.res,
            "bounds": source.bounds,
            "dimensions": (source.height, source.width),
            "nodata": source.nodata,
            "support_min": float(support_values.min()),
            "support_max": float(support_values.max()),
        }


def reproject_precipitation_surface(
    raster_path: str,
    target_bounds: tuple[float, float, float, float],
    height: int,
    width: int,
) -> np.ndarray:
    """Reproject PRISM to the aligned statewide 100-m integration grid.

    Precipitation is continuous, so bilinear interpolation is retained from the
    validated prototype. The 100-m grid is only an integration aid for
    approximating 1-km means; upsampling creates no new climate information.

    Args:
        raster_path: Rasterio path to the archived PRISM GeoTIFF.
        target_bounds: Statewide EPSG:5070 destination bounds.
        height: Destination raster row count.
        width: Destination raster column count.

    Returns:
        Float32 statewide precipitation integration surface.

    Raises:
        ValueError: If dimensions disagree with the requested bounds.
    """
    left, bottom, right, top = target_bounds
    expected_width = int((right - left) / INTEGRATION_RESOLUTION_M)
    expected_height = int((top - bottom) / INTEGRATION_RESOLUTION_M)
    if (height, width) != (expected_height, expected_width):
        raise ValueError("Destination dimensions do not match statewide bounds.")

    # Anchoring at the authoritative upper-left boundary makes every 1-km cell
    # span exactly ten aligned rows and ten aligned columns.
    destination_transform = from_origin(
        left,
        top,
        INTEGRATION_RESOLUTION_M,
        INTEGRATION_RESOLUTION_M,
    )
    precipitation_100m = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(raster_path) as source:
        # reproject() does three conceptual jobs:
        # 1. maps native PRISM locations into EPSG:5070,
        # 2. estimates destination values using bilinear interpolation,
        # 3. writes those estimates onto the aligned 100-m destination grid.
        # Bilinear is appropriate because precipitation is continuous.
        reproject(
            source=rasterio.band(source, 1),
            destination=precipitation_100m,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=destination_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )
    return precipitation_100m


def aggregate_to_analysis_cells(
    precipitation_100m: np.ndarray,
    grid: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Aggregate 100-m samples and attach means to authoritative cell IDs.

    The rectangular raster is reorganized into analysis rows, ten fine rows,
    analysis columns, and ten fine columns. Means are attached using centroid-
    derived indices, independent of GeoDataFrame order.

    Args:
        precipitation_100m: Aligned statewide integration surface.
        grid: Authoritative retained 1-km cells.
        bounds: Shared raster and lattice bounds in EPSG:5070.

    Returns:
        Deterministically ordered two-column PRISM feature table.

    Raises:
        ValueError: If raster dimensions, alignment, or indices are invalid.
    """
    samples_per_cell = ANALYSIS_CELL_SIZE_M // INTEGRATION_RESOLUTION_M
    height, width = precipitation_100m.shape
    if height % samples_per_cell or width % samples_per_cell:
        raise ValueError("Integration surface cannot form 1-km blocks.")

    analysis_rows = height // samples_per_cell
    analysis_columns = width // samples_per_cell

    # Reorganize into analysis rows x fine rows x analysis columns x fine columns.
    grouped_values = precipitation_100m.reshape(
        analysis_rows,
        samples_per_cell,
        analysis_columns,
        samples_per_cell,
    )
    # Average only fine axes: each result summarizes one aligned 10 x 10 block.
    cell_means = grouped_values.mean(axis=(1, 3))

    left, _, _, top = bounds
    centroids = grid.geometry.centroid
    row_positions = (top - centroids.y) / ANALYSIS_CELL_SIZE_M - 0.5
    column_positions = (centroids.x - left) / ANALYSIS_CELL_SIZE_M - 0.5
    row_indices = np.rint(row_positions).astype(int)
    column_indices = np.rint(column_positions).astype(int)
    if not (
        np.allclose(row_positions, row_indices)
        and np.allclose(column_positions, column_indices)
    ):
        raise ValueError("Analysis-cell centroids do not align with raster blocks.")
    if not (
        row_indices.between(0, analysis_rows - 1).all()
        and column_indices.between(0, analysis_columns - 1).all()
    ):
        raise ValueError("An authoritative cell maps outside the raster.")

    # Centroid indices map rectangular means to prepared cell_id values without
    # inventing keys or retaining rectangular cells outside New Mexico.
    feature_table = pd.DataFrame(
        {
            "cell_id": grid["cell_id"].to_numpy(),
            FEATURE_COLUMN: cell_means[
                row_indices.to_numpy(),
                column_indices.to_numpy(),
            ],
        }
    )
    return feature_table.sort_values("cell_id").reset_index(drop=True)


def validate_feature_table(feature_table: pd.DataFrame) -> dict[str, float | int]:
    """Validate statewide schema, keys, completeness, and numeric values.

    Missing, non-finite, and negative precipitation are counted and rejected.
    No source or reprojection gaps are silently filled.

    Args:
        feature_table: Statewide keyed PRISM feature table.

    Returns:
        QA/QC counts and population descriptive statistics.

    Raises:
        ValueError: If schema, cardinality, keys, or values are invalid.
    """
    if list(feature_table.columns) != FINAL_COLUMNS:
        raise ValueError(f"PRISM feature schema must be exactly {FINAL_COLUMNS}.")

    values = feature_table[FEATURE_COLUMN]
    qa = {
        "rows": len(feature_table),
        "unique_ids": feature_table["cell_id"].nunique(),
        "duplicate_ids": int(feature_table["cell_id"].duplicated().sum()),
        "missing": int(values.isna().sum()),
        "non_finite": int((~np.isfinite(values)).sum()),
        "negative": int((values < 0).sum()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=0)),
    }
    if qa["rows"] != EXPECTED_RETAINED_CELL_COUNT:
        raise ValueError(f"Unexpected statewide row count: {qa['rows']:,}.")
    if qa["unique_ids"] != EXPECTED_RETAINED_CELL_COUNT or qa["duplicate_ids"]:
        raise ValueError("Feature table has missing or duplicate cell IDs.")
    if qa["missing"] or qa["non_finite"]:
        raise ValueError(
            "Precipitation has missing/non-finite values. Inspect PRISM NoData, "
            "reprojection coverage, and alignment before writing. "
            f"Missing: {qa['missing']:,}; non-finite: {qa['non_finite']:,}."
        )
    if qa["negative"]:
        raise ValueError(f"Precipitation has {qa['negative']:,} negative values.")
    return qa


def write_and_verify_parquet(
    feature_table: pd.DataFrame,
    destination: Path,
) -> tuple[pd.DataFrame, int]:
    """Write safely and independently verify the production Parquet.

    A temporary .part Parquet is read and validated before replacing the final
    path, so failed writing or verification cannot leave a partial final output.

    Args:
        feature_table: Validated deterministic PRISM features.
        destination: Final production Parquet path.

    Returns:
        Re-read feature table and final file size in bytes.

    Raises:
        ValueError: If read-back schema, keys, values, or equality fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    try:
        feature_table.to_parquet(partial_path, index=False)
        written_features = pd.read_parquet(partial_path)
        validate_feature_table(written_features)
        if not written_features.equals(feature_table):
            raise ValueError("Read-back features differ from the in-memory table.")
        partial_path.replace(destination)
    finally:
        if partial_path.exists():
            partial_path.unlink()

    return pd.read_parquet(destination), destination.stat().st_size


def report_qa_qc(
    source_metadata: dict[str, Any],
    qa: dict[str, float | int],
    height: int,
    width: int,
    destination_gib: float,
    elapsed_seconds: float,
    peak_memory_bytes: int,
    file_size_bytes: int,
) -> None:
    """Report source, processing-resource, and final output QA/QC.

    Args:
        source_metadata: Validated PRISM source metadata.
        qa: Validated feature counts and descriptive statistics.
        height: Integration-grid row count.
        width: Integration-grid column count.
        destination_gib: Float32 destination-array size in GiB.
        elapsed_seconds: Total workflow runtime.
        peak_memory_bytes: Observed operating-system process peak memory.
        file_size_bytes: Final Parquet size.
    """
    print("PRISM SOURCE")
    print("------------")
    print("Access: direct HTTPS from the authoritative PRISM data directory")
    print(f"Product: {PRISM_PRODUCT}")
    print("Variable / units: annual precipitation / mm")
    print(f"Source CRS: {source_metadata['crs']}")
    print(f"Source resolution: {source_metadata['resolution']}")
    print(f"Source NoData: {source_metadata['nodata']}")
    print(
        "Support-window min / max: "
        f"{source_metadata['support_min']:.6f} / "
        f"{source_metadata['support_max']:.6f} mm"
    )

    print("\nSTATEWIDE PROCESSING")
    print("--------------------")
    print("Strategy: one direct aligned float32 reprojection")
    print(f"Integration grid: {height:,} rows x {width:,} columns")
    print(f"Destination-array memory: {destination_gib:.3f} GiB")
    print(f"Observed process peak memory: {peak_memory_bytes / 1024**3:.3f} GiB")
    print(f"Execution time: {elapsed_seconds:.2f} seconds")

    print("\nFINAL PRISM FEATURE QA/QC")
    print("-------------------------")
    print(f"Total rows: {qa['rows']:,}")
    print(f"Unique cell_id count: {qa['unique_ids']:,}")
    print(f"Duplicate cell_id count: {qa['duplicate_ids']:,}")
    print(f"Missing precipitation count: {qa['missing']:,}")
    print(f"Non-finite precipitation count: {qa['non_finite']:,}")
    print(f"Negative precipitation count: {qa['negative']:,}")
    print(f"Minimum: {qa['minimum']:.6f} mm")
    print(f"Maximum: {qa['maximum']:.6f} mm")
    print(f"Mean: {qa['mean']:.6f} mm")
    print(f"Standard deviation: {qa['standard_deviation']:.6f} mm")
    print(f"Final output: {output_path}")
    print(f"Final file size: {file_size_bytes / 1024**2:.2f} MiB")


# %% Main workflow
def main() -> None:
    """Build and validate statewide New Mexico PRISM precipitation features."""
    started_at = perf_counter()

    # STEP 1 — Acquire and validate the authoritative PRISM source
    download_source_if_missing(PRISM_URL, prism_archive_path)
    prism_raster_path = validate_prism_archive(prism_archive_path)

    # STEP 2 — Load and validate the authoritative 1-km analysis grid
    analysis_grid = load_analysis_grid(grid_path)

    # STEP 3 — Determine the minimum statewide aligned processing extent
    bounds, height, width, destination_gib = determine_aligned_processing_extent(
        analysis_grid
    )

    # STEP 4 — Validate support and reproject to the 100-m integration grid
    source_metadata = validate_prism_raster(prism_raster_path, bounds)
    precipitation_100m = reproject_precipitation_surface(
        prism_raster_path,
        bounds,
        height,
        width,
    )

    # STEP 5 — Aggregate aligned 10 x 10 blocks to authoritative 1-km cells
    feature_table = aggregate_to_analysis_cells(
        precipitation_100m,
        analysis_grid,
        bounds,
    )

    # STEP 6 — Validate the statewide table without filling missing values
    qa = validate_feature_table(feature_table)

    # STEP 7 — Write the final deterministic Parquet safely
    written_features, file_size_bytes = write_and_verify_parquet(
        feature_table,
        output_path,
    )

    # STEP 8 — Read back and independently verify the final output
    read_back_qa = validate_feature_table(written_features)
    if read_back_qa != qa:
        raise ValueError("Read-back QA/QC differs from pre-write QA/QC.")

    # STEP 9 — Report source, resource, and final feature QA/QC
    elapsed_seconds = perf_counter() - started_at
    process_memory = psutil.Process().memory_info()
    peak_memory_bytes = getattr(process_memory, "peak_wset", process_memory.rss)
    report_qa_qc(
        source_metadata,
        qa,
        height,
        width,
        destination_gib,
        elapsed_seconds,
        peak_memory_bytes,
        file_size_bytes,
    )


# %% Run script
if __name__ == "__main__":
    main()
