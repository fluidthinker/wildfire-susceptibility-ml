"""Build statewide LF2016 Remap EVT features on the authoritative NM grid.

Circa-2016 vegetation precedes the 2017-2022 MTBS target period. EVT codes
are categorical identifiers. The aligned 25-m integration representation
adds no ecological resolution to the native 30-m observations.

Run first with --test-only, then without arguments to process the state.
Every run repeats the small production-path checks before scaling. Validated
native exports and batch Parquets survive restarts; final publication is atomic.
"""

# %% Imports
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject


# %% Constants and paths
ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = ROOT / "data/processed/grids/nm_analysis_grid_1km.gpkg"
RAW_DIR = ROOT / "data/raw/landfire"
EXPORT_DIR = RAW_DIR / "nm_statewide"
CHECKPOINT_DIR = ROOT / "data/interim/landfire_nm"
REPORT_DIR = ROOT / "outputs/landfire_nm"
FINAL_PATH = ROOT / "data/processed/features/nm_landfire_features_1km.parquet"
PROTOTYPE_PATH = ROOT / "data/interim/landfire_evt_prototype.parquet"
SERVICE_NAME = "Landfire_LF2016/LF2016_EVT_CONUS"
SERVICE_URL = f"https://lfps.usgs.gov/arcgis/rest/services/{SERVICE_NAME}/ImageServer"
PRODUCT = "LANDFIRE LF2016 Remap EVT CONUS; circa-2016; legend v200"
CRS = "EPSG:5070"
NODATA = -9999
EXPECTED_CELLS = 314_920
CELL_M = 1_000
RESOLUTION_M = 25
SAMPLES_SIDE = CELL_M // RESOLUTION_M
POSITIONS = SAMPLES_SIDE**2
BATCH_SIDE = 50  # At most 2,500 complete cells / 2,000 x 2,000 int16 samples.
SCHEMA = ["cell_id", "evt_dominant_class", "evt_dominant_fraction"]
# Conservative review gates, not vegetation filtering rules. All observations
# remain in QA. Any zero-coverage cell blocks the finite final-feature schema.
MAX_PARTIAL_FRACTION = 0.001
MAX_TIE_FRACTION = 0.01
METHOD_VERSION = "lf2016-nm-25m-plurality-v1"
REQUEST_ATTEMPTS = 5  # Wait 2, 4, 8, then 16 seconds between temporary failures.


# %% Helper functions
def write_json(value: dict, path: Path) -> None:
    """Atomically replace a JSON artifact, cleaning incomplete writes.

    Args:
        value: JSON-compatible metadata or QA summary.
        path: Destination artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        partial.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def digest_file(path: Path) -> str:
    """Hash an input or checkpoint without loading the file into memory.

    Args:
        path: Existing file whose content identifies safe reuse.

    Returns:
        SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def request_source(url: str, statistics: dict, destination: Path | None = None) -> dict | None:
    """Read USGS JSON or stream a raster, retrying transient acquisition failures.

    Args:
        url: Exact pinned source URL, including REST parameters when needed.
        statistics: Mutable run counters for requests and retries.
        destination: Optional file path; caller validates it before publication.

    Returns:
        Parsed JSON for metadata requests, otherwise None.

    Raises:
        OSError: After five unsuccessful network attempts.
        ValueError: For an ArcGIS application error, without product fallback.
    """
    for attempt in range(REQUEST_ATTEMPTS):
        try:
            statistics["requests"] += 1
            request = Request(url, headers={"User-Agent": "wildfire-susceptibility-ml/0.1"})
            with urlopen(request, timeout=120) as response:
                if destination is None:
                    result = json.load(response)
                    if "error" in result:
                        raise ValueError(f"USGS service error: {result['error']}")
                    return result
                with destination.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            return None
        except (URLError, TimeoutError, ConnectionError) as error:
            retryable_statuses = {408, 429, 500, 502, 503, 504}
            # A freshly generated ArcGIS download briefly returned HTTP 400
            # during the statewide run, then served a valid TIFF on retry.
            # Retry temporary artifact availability only; a bad metadata/export
            # query still fails immediately rather than hiding invalid requests.
            if destination is not None:
                retryable_statuses.update({400, 404})
            if isinstance(error, HTTPError) and error.code not in retryable_statuses:
                raise
            if attempt == REQUEST_ATTEMPTS - 1:
                raise
            statistics["retries"] += 1
            print(f"Retry {attempt + 1}: {error}", flush=True)
            sleep(2 ** (attempt + 1))
    raise RuntimeError("Unreachable request state")


def validate_inputs(statistics: dict) -> tuple[gpd.GeoDataFrame, dict, set[int], dict]:
    """Inspect authoritative geometry, live service identity, and cached legend.

    Args:
        statistics: Run acquisition counters.

    Returns:
        Grid, live service metadata, valid class codes, and input fingerprint.

    Raises:
        ValueError: For changed scientific inputs, geometry, or source metadata.
    """
    grid = gpd.read_file(GRID_PATH, layer="analysis_grid")[["cell_id", "geometry"]]
    if (len(grid) != EXPECTED_CELLS or grid.crs.to_epsg() != 5070
            or grid.cell_id.isna().any() or not grid.cell_id.is_unique):
        raise ValueError("Authoritative grid count, CRS, or keys failed validation.")
    geometry = grid.geometry
    bounds = geometry.bounds
    if (geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all()
            or not geometry.geom_type.eq("Polygon").all()
            or not np.allclose(geometry.area, CELL_M**2, rtol=0, atol=1e-3)
            or not np.allclose(bounds.maxx - bounds.minx, CELL_M, rtol=0, atol=1e-6)
            or not np.allclose(bounds.maxy - bounds.miny, CELL_M, rtol=0, atol=1e-6)):
        raise ValueError("Expected complete axis-aligned 1-km squares; no repairs applied.")
    metadata = request_source(SERVICE_URL + "?f=json", statistics)
    expected = {"name": SERVICE_NAME, "bandCount": 1, "pixelType": "S16",
                "pixelSizeX": 30, "pixelSizeY": 30, "noDataValue": NODATA}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Changed source {key}: {metadata.get(key)}")
    cached_metadata = json.loads((RAW_DIR / "lf2016_evt_service.json").read_text())
    if (metadata["extent"] != cached_metadata["extent"]
            or metadata["extent"]["spatialReference"]["wkid"] != 5070):
        raise ValueError("Native source lattice differs from the validated prototype.")
    legend_path = RAW_DIR / "LF16_EVT_200.csv"
    legend = pd.read_csv(legend_path)
    if (not {"VALUE", "EVT_NAME"}.issubset(legend.columns)
            or not legend.VALUE.is_unique or legend[["VALUE", "EVT_NAME"]].isna().any().any()
            or not pd.api.types.is_integer_dtype(legend.VALUE)):
        raise ValueError("Invalid version-specific EVT legend.")
    fill_names = legend.EVT_NAME.str.contains("NoData|Background|Unknown|Fill", case=False)
    if set(legend.loc[fill_names | legend.VALUE.le(0), "VALUE"]) != {NODATA}:
        raise ValueError("Documented background/NoData rules require review.")
    # Legitimate water, developed, barren, and agricultural classes stay valid.
    codes = set(legend.VALUE.astype(int)) - {NODATA}
    fingerprint = {"method": METHOD_VERSION, "grid_sha256": digest_file(GRID_PATH),
                   "legend_sha256": digest_file(legend_path), "service": SERVICE_URL,
                   "source_properties": expected, "source_extent": metadata["extent"],
                   "resolution_m": RESOLUTION_M, "batch_side_cells": BATCH_SIDE,
                   "numpy": np.__version__, "rasterio": rasterio.__version__,
                   "gdal": rasterio.__gdal_version__}
    write_json(metadata, EXPORT_DIR / "service.json")
    return grid, metadata, codes, fingerprint


def prepare_batches(grid: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list, dict]:
    """Map existing squares to integer positions and bounded rectangular windows.

    Args:
        grid: Validated authoritative geometry; never clipped or regenerated.

    Returns:
        Grid with processing indices, occupied batch groups, and memory estimate.

    Raises:
        ValueError: If cells do not occupy a unique common 1-km lattice.
    """
    grid = grid.copy()
    left, bottom, right, top = grid.total_bounds
    bounds = grid.geometry.bounds
    # Raster row indices increase downward from the top-left corner. Column
    # indices increase eastward. These offsets come from geometry, not cell IDs.
    row_positions = (top - bounds.maxy) / CELL_M
    column_positions = (bounds.minx - left) / CELL_M
    for name, positions in [("row", row_positions), ("column", column_positions)]:
        if not np.allclose(positions, np.rint(positions), rtol=0, atol=1e-9):
            raise ValueError("Authoritative cells do not share a 1-km lattice.")
        grid[name] = np.rint(positions).astype(int)
    if grid.duplicated(["row", "column"]).any():
        raise ValueError("Multiple cell IDs occupy the same spatial position.")
    grid["batch_row"] = grid.row // BATCH_SIDE
    grid["batch_column"] = grid.column // BATCH_SIDE
    batches = list(grid.groupby(["batch_row", "batch_column"], sort=True))
    shape = [int((top - bottom) / RESOLUTION_M), int((right - left) / RESOLUTION_M)]
    design = {"statewide_destination_shape_rows_columns": shape,
              "statewide_single_int16_gib": int(np.prod(shape)) * 2 / 1024**3,
              "maximum_batch_destination_shape": [BATCH_SIDE * SAMPLES_SIDE] * 2,
              "maximum_batch_destination_mib": (BATCH_SIDE * SAMPLES_SIDE)**2 * 2 / 1024**2,
              "batch_count": len(batches), "authoritative_cells": len(grid)}
    print("PROCESSING DESIGN\n" + json.dumps(design, indent=2), flush=True)
    return grid, batches, design


def native_parameters(metadata: dict, bounds: np.ndarray) -> dict:
    """Snap target bounds outward to native pixels and add one support pixel.

    Args:
        metadata: Validated native origin, resolution, and export limits.
        bounds: Complete-cell target rectangle in EPSG:5070.

    Returns:
        Raw categorical export parameters, identical in method to script 13.

    Raises:
        ValueError: If an export exceeds declared service dimensions.
    """
    left, bottom, right, top = bounds
    origin_x = metadata["extent"]["xmin"]
    origin_y = metadata["extent"]["ymax"]
    resolution = metadata["pixelSizeX"]
    # The halo provides source support beyond processing edges; it does not
    # create analysis cells or fill genuine source NoData. Both CRSs are 5070,
    # so one native pixel beyond outward snapping supports nearest-neighbor.
    left = origin_x + (np.floor((left - origin_x) / resolution) - 1) * resolution
    right = origin_x + (np.ceil((right - origin_x) / resolution) + 1) * resolution
    top = origin_y - (np.floor((origin_y - top) / resolution) - 1) * resolution
    bottom = origin_y - (np.ceil((origin_y - bottom) / resolution) + 1) * resolution
    width, height = int((right - left) / resolution), int((top - bottom) / resolution)
    if width > metadata["maxImageWidth"] or height > metadata["maxImageHeight"]:
        raise ValueError("Batch exceeds USGS export limits.")
    return {"f": "json", "bbox": f"{left},{bottom},{right},{top}",
            "bboxSR": 5070, "imageSR": 5070, "size": f"{width},{height}",
            "format": "tiff", "pixelType": "S16", "noData": NODATA,
            "interpolation": "RSP_NearestNeighbor", "adjustAspectRatio": "false",
            "renderingRule": json.dumps({"rasterFunction": "None"})}


def read_native(path: Path, parameters: dict, codes: set[int]) -> tuple[np.ndarray, Affine]:
    """Validate a raw export before reading its bounded categorical array.

    Args:
        path: Cached or temporary GeoTIFF.
        parameters: Expected source lattice and rectangle.
        codes: Documented valid EVT identities.

    Returns:
        Int16 source samples and their affine transform.

    Raises:
        ValueError: For changed raster properties or undocumented class codes.
    """
    width, height = map(int, parameters["size"].split(","))
    bounds = tuple(map(float, parameters["bbox"].split(",")))
    with rasterio.open(path) as source:
        if (source.driver != "GTiff" or source.crs.to_epsg() != 5070
                or source.res != (30, 30) or source.count != 1
                or source.dtypes != ("int16",) or source.nodata != NODATA
                or source.shape != (height, width) or tuple(source.bounds) != bounds
                or source.transform.b != 0 or source.transform.d != 0):
            raise ValueError(f"Native export properties failed: {path}")
        native = source.read(1, masked=True).filled(NODATA)
        transform = source.transform
    unknown = set(np.unique(native)) - codes - {NODATA}
    if unknown:
        raise ValueError(f"Undocumented native EVT codes: {unknown}")
    return native, transform


def acquire_native(name: str, parameters: dict, codes: set[int],
                   statistics: dict) -> tuple[np.ndarray, Affine]:
    """Reuse verified native exports or stream, validate, and publish new ones.

    Args:
        name: Deterministic spatial batch name.
        parameters: Exact native export request including renderer suppression.
        codes: Valid legend codes.
        statistics: Mutable acquisition/reuse counters.

    Returns:
        Native samples and affine transform.

    Raises:
        ValueError: If an existing complete cache has inconsistent provenance.
    """
    path = EXPORT_DIR / f"{name}.tif"
    record_path = EXPORT_DIR / f"{name}.json"
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if (record["parameters"] != parameters or record["service"] != SERVICE_URL
                or record["sha256"] != digest_file(path)):
            raise ValueError(f"Native cache provenance mismatch: {name}")
        statistics["native_reused"] += 1
        return read_native(path, parameters, codes)
    partial = path.with_suffix(".tif.part")
    try:
        response = request_source(SERVICE_URL + "/exportImage?" + urlencode(parameters), statistics)
        request_source(response["href"], statistics, partial)
        values, transform = read_native(partial, parameters, codes)
        record = {"service": SERVICE_URL, "parameters": parameters, "response": response,
                  "sha256": digest_file(partial)}
        partial.replace(path)
        # Provenance is the completion marker; an interrupted export without
        # its marker is never trusted as completed work.
        write_json(record, record_path)
        statistics["native_downloaded"] += 1
        return values, transform
    finally:
        partial.unlink(missing_ok=True)


def align_categories(native: np.ndarray, transform: Affine, bounds: np.ndarray) -> np.ndarray:
    """Copy native EVT identities onto the authoritative aligned 25-m rectangle.

    Args:
        native: Validated 30-m source samples including surrounding support.
        transform: Native affine transform.
        bounds: Complete-cell destination bounds.

    Returns:
        Bounded destination array, with 40 samples per analysis-cell side.

    Raises:
        ValueError: If resampling introduces any new class code.
    """
    left, bottom, right, top = bounds
    shape = (int((top - bottom) / RESOLUTION_M), int((right - left) / RESOLUTION_M))
    destination = np.full(shape, NODATA, dtype=np.int16)
    # from_origin anchors pixel edges at the actual analysis-cell bounds.
    # Source/destination resolution and alignment differ despite identical CRS.
    # Nearest-neighbor copies an existing class at each destination location;
    # bilinear or averaging would invent codes with no ecological meaning.
    reproject(source=native, destination=destination, src_transform=transform,
              src_crs=CRS, src_nodata=NODATA,
              dst_transform=from_origin(left, top, RESOLUTION_M, RESOLUTION_M),
              dst_crs=CRS, dst_nodata=NODATA, resampling=Resampling.nearest,
              init_dest_nodata=True, num_threads=1, warp_mem_limit=64)
    if not set(np.unique(destination)).issubset(set(np.unique(native)) | {NODATA}):
        raise ValueError("Reprojection introduced a new EVT identity.")
    return destination


def summarize_samples(samples: np.ndarray) -> tuple[dict, set[int]]:
    """Measure coverage and plurality without imputing missing vegetation.

    Args:
        samples: Exactly 1,600 destination samples for a complete analysis cell.

    Returns:
        Feature/QA fields and the observed valid class set.

    Raises:
        ValueError: If the cell does not have exactly 1,600 integer positions.
    """
    if samples.size != POSITIONS or not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("Expected exactly 1,600 categorical destination positions.")
    # Boolean indexing removes only documented fill; zero valid observations
    # produce missing features and a publication stop, never an invented class.
    valid = samples[samples != NODATA]
    # np.unique returns sorted class identities and parallel occurrence counts.
    # Masking classes by maximum count explicitly retains every tied winner.
    classes, counts = np.unique(valid, return_counts=True)
    winning_count = int(counts.max()) if len(counts) else 0
    winners = classes[counts == winning_count]
    # The first sorted tied code is only a deterministic fallback. Numerical
    # class magnitude has no ecological preference. Plurality need not exceed 50%.
    record = {"evt_dominant_class": int(winners[0]) if len(winners) else pd.NA,
              "evt_dominant_fraction": winning_count / len(valid) if len(valid) else np.nan,
              "total_positions": samples.size, "valid_sample_count": len(valid),
              "nodata_sample_count": samples.size - len(valid),
              "valid_fraction": len(valid) / samples.size,
              "winning_sample_count": winning_count, "number_of_classes": len(classes),
              "plurality_tie": len(winners) > 1,
              "tied_classes": json.dumps(winners.tolist()) if len(winners) > 1 else "[]"}
    return record, set(map(int, classes))


def aggregate_cells(values: np.ndarray, cells: gpd.GeoDataFrame) -> tuple[pd.DataFrame, set[int]]:
    """Group aligned raster samples by spatial position of retained cells only.

    Args:
        values: Complete-cell rectangular destination array.
        cells: Authoritative subset; rectangular gaps are not analysis cells.

    Returns:
        Sorted features plus internal coverage/tie QA and sampled class union.
    """
    left, _, _, top = cells.total_bounds
    # Reshape groups axes as cell rows, sample rows, cell columns, sample columns.
    # Indexing [row, :, column, :] extracts a whole 40-by-40 cell even when
    # the boundary batch has holes. Those holes never become fabricated rows.
    grouped = values.reshape(values.shape[0] // SAMPLES_SIDE, SAMPLES_SIDE,
                             values.shape[1] // SAMPLES_SIDE, SAMPLES_SIDE)
    records, observed = [], set()
    for cell in cells.itertuples():
        cell_left, _, _, cell_top = cell.geometry.bounds
        row = int(round((top - cell_top) / CELL_M))
        column = int(round((cell_left - left) / CELL_M))
        record, classes = summarize_samples(grouped[row, :, column, :].ravel())
        records.append({"cell_id": cell.cell_id, **record})
        observed.update(classes)
    table = pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)
    table["evt_dominant_class"] = table.evt_dominant_class.astype("Int64")
    return table, observed


def validate_table(table: pd.DataFrame, expected_ids: pd.Series, codes: set[int], final: bool = False) -> None:
    """Check exact authoritative keys, categorical domain, and count arithmetic.

    Args:
        table: Batch QA table or production feature table.
        expected_ids: Authoritative IDs required for this artifact.
        codes: Documented valid EVT codes.
        final: Require exact final schema and entirely finite features.

    Raises:
        ValueError: For any schema, key, count, or fraction violation.
    """
    expected = expected_ids.sort_values().reset_index(drop=True)
    if (not table.cell_id.reset_index(drop=True).equals(expected)
            or not table.cell_id.is_unique or table.cell_id.isna().any()):
        raise ValueError("Output keys/order do not exactly match authoritative cells.")
    if not pd.api.types.is_integer_dtype(table.evt_dominant_class):
        raise ValueError("EVT classes must have an integer dtype.")
    if not set(table.evt_dominant_class.dropna().astype(int)).issubset(codes):
        raise ValueError("Undocumented dominant class.")
    valid = pd.Series(True, index=table.index) if final else table.valid_sample_count.gt(0)
    fractions = table.loc[valid, "evt_dominant_fraction"]
    if (not np.isfinite(fractions).all() or not fractions.gt(0).all()
            or not fractions.le(1).all() or table.loc[valid, "evt_dominant_class"].isna().any()):
        raise ValueError("Nonempty cells require valid classes and finite fractions in (0, 1].")
    if final:
        if list(table.columns) != SCHEMA:
            raise ValueError("Final feature schema differs from specification.")
        return
    if (not table.total_positions.eq(POSITIONS).all()
            or not table.valid_sample_count.between(0, POSITIONS).all()
            or not (table.valid_sample_count + table.nodata_sample_count).eq(POSITIONS).all()
            or not table.valid_fraction.eq(table.valid_sample_count / POSITIONS).all()
            or not table.loc[valid, "winning_sample_count"].between(1, POSITIONS).all()
            or not fractions.eq(table.loc[valid, "winning_sample_count"] /
                                table.loc[valid, "valid_sample_count"]).all()
            or table.loc[~valid, SCHEMA[1:]].notna().any().any()
            or not table.loc[~valid, "winning_sample_count"].eq(0).all()):
        raise ValueError("Coverage and plurality arithmetic failed.")
    for record in table.itertuples():
        tied = json.loads(record.tied_classes)
        if (record.plurality_tie != (len(tied) > 1)
                or (tied and (tied != sorted(set(tied)) or not set(tied).issubset(codes)
                              or record.evt_dominant_class != tied[0]))):
            raise ValueError("Tie QA and deterministic winner disagree.")


def write_table(table: pd.DataFrame, path: Path, expected_ids: pd.Series,
                codes: set[int], final: bool = False) -> None:
    """Write a temporary Parquet and independently validate its readback.

    Args:
        table: Validated features or QA rows.
        path: Destination published only after successful readback.
        expected_ids: Required authoritative keys.
        codes: Documented categorical domain.
        final: Enforce the final three-column schema.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        table.to_parquet(partial, index=False)
        readback = pd.read_parquet(partial)
        validate_table(readback, expected_ids, codes, final)
        pd.testing.assert_frame_equal(table, readback)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def process_batch(key: tuple, cells: gpd.GeoDataFrame, metadata: dict,
                  codes: set[int], fingerprint: dict, statistics: dict) -> tuple[pd.DataFrame, set[int]]:
    """Run the shared test/statewide path, validating checkpoints before reuse.

    Args:
        key: Spatial batch row and column.
        cells: Complete authoritative cells assigned to this batch.
        metadata: Validated source description.
        codes: Legend domain.
        fingerprint: Scientific-input and method identity.
        statistics: Mutable run performance counters.

    Returns:
        Validated cell QA table and observed sample classes.

    Raises:
        ValueError: If a completion marker no longer matches its inputs/artifact.
    """
    name = f"r{key[0]:03d}_c{key[1]:03d}"
    path = CHECKPOINT_DIR / f"{name}.parquet"
    marker = CHECKPOINT_DIR / f"{name}.json"
    parameters = native_parameters(metadata, cells.total_bounds)
    identity = {"inputs": fingerprint, "parameters": parameters}
    if path.exists() and marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity or saved["sha256"] != digest_file(path):
            raise ValueError(f"Stale/corrupt checkpoint requires review: {name}")
        table = pd.read_parquet(path)
        validate_table(table, cells.cell_id, codes)
        observed = set(saved["observed_classes"])
        if not observed.issubset(codes):
            raise ValueError("Checkpoint observed classes are undocumented.")
        statistics["batches_reused"] += 1
        return table, observed
    native, transform = acquire_native(name, parameters, codes, statistics)
    values = align_categories(native, transform, cells.total_bounds)
    # STEP 4 is performed immediately within each batch to bound raster memory.
    table, observed = aggregate_cells(values, cells)
    validate_table(table, cells.cell_id, codes)
    write_table(table, path, cells.cell_id, codes)
    write_json({"identity": identity, "sha256": digest_file(path),
                "observed_classes": sorted(observed)}, marker)
    statistics["batches_computed"] += 1
    return table, observed


def compare_prototype(table: pd.DataFrame, require_all: bool = True) -> dict:
    """Require exact feature equality with the accepted prototype overlap.

    Args:
        table: Production results including prototype cells.
        require_all: Require all 64 prototype keys to be represented.

    Returns:
        Overlap and mismatch counts; differences are preserved before raising.

    Raises:
        ValueError: If prototype keys are absent or feature values differ.
    """
    prototype = pd.read_parquet(PROTOTYPE_PATH)[SCHEMA]
    joined = prototype.merge(table[SCHEMA], on="cell_id", suffixes=("_prototype", "_production"),
                             validate="one_to_one")
    if require_all and len(joined) != len(prototype):
        raise ValueError("Production test must cover every prototype cell.")
    mismatch = ((joined.evt_dominant_class_prototype != joined.evt_dominant_class_production)
                | (joined.evt_dominant_fraction_prototype != joined.evt_dominant_fraction_production))
    if mismatch.any():
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        joined.loc[mismatch].to_csv(REPORT_DIR / "prototype_mismatches.csv", index=False)
        raise ValueError(f"{mismatch.sum()} prototype regression mismatches; inspect QA CSV.")
    return {"overlap_cells": len(joined), "mismatches": int(mismatch.sum())}


def small_production_test(batches: list, metadata: dict, codes: set[int],
                          fingerprint: dict, statistics: dict) -> dict:
    """Exercise actual production batches, restart reads, and edge-case counting.

    Args:
        batches: Statewide spatial groups.
        metadata: Source metadata.
        codes: Valid EVT categories.
        fingerprint: Checkpoint input identity.
        statistics: Run counters.

    Returns:
        Successful test evidence saved separately from the final ML table.

    Raises:
        AssertionError: If counting, schema, regression, or restart behavior fails.
    """
    # Synthetic cases enter the very same 1,600-position counting function.
    low, high, third = sorted(codes)[:3]
    cases = [(np.array([low] * 700 + [high] * 700 + [third] * 200), low, 700 / 1600, True, 1600),
             (np.array([low] * 2 + [high] + [NODATA] * 1597), low, 2 / 3, False, 3),
             (np.full(POSITIONS, NODATA), None, None, False, 0),
             (np.full(POSITIONS, high), high, 1.0, False, 1600)]
    for samples, winner, fraction, tie, valid in cases:
        record, _ = summarize_samples(samples)
        assert record["valid_sample_count"] == valid
        assert record["nodata_sample_count"] == POSITIONS - valid
        assert record["plurality_tie"] == tie
        if valid:
            assert record["evt_dominant_class"] == winner
            assert record["evt_dominant_fraction"] == fraction
        else:
            assert pd.isna(record["evt_dominant_class"])
            assert np.isnan(record["evt_dominant_fraction"])
    prototype_ids = set(pd.read_parquet(PROTOTYPE_PATH).cell_id)
    selected = [(key, cells) for key, cells in batches if prototype_ids.intersection(cells.cell_id)]
    if batches[0][0] not in [key for key, _ in selected]:
        selected.append(batches[0])
    tables = []
    for key, cells in selected:
        print(f"SMALL PRODUCTION TEST: batch {key}, {len(cells):,} cells", flush=True)
        first, observed = process_batch(key, cells, metadata, codes, fingerprint, statistics)
        reused_before = statistics["batches_reused"]
        second, repeated_classes = process_batch(key, cells, metadata, codes, fingerprint, statistics)
        pd.testing.assert_frame_equal(first, second)
        assert observed == repeated_classes
        assert statistics["batches_reused"] == reused_before + 1
        tables.append(first)
    table = pd.concat(tables, ignore_index=True).sort_values("cell_id").reset_index(drop=True)
    regression = compare_prototype(table)
    features = table[SCHEMA].copy()
    validate_table(features, table.cell_id, codes, final=True)
    write_table(features, CHECKPOINT_DIR / "small_test_features.parquet", table.cell_id, codes, final=True)
    report = {"passed": True, "batches": len(selected), "cells": len(table),
              "synthetic_cases": len(cases), "restart_equality": True,
              "partial_cells": int(table.valid_sample_count.between(1, POSITIONS - 1).sum()),
              "zero_cells": int(table.valid_sample_count.eq(0).sum()), "prototype": regression}
    write_json(report, REPORT_DIR / "small_test.json")
    print("SMALL TEST PASSED\n" + json.dumps(report, indent=2), flush=True)
    return report


def verify_partial_source_support(table: pd.DataFrame, grid: gpd.GeoDataFrame,
                                  metadata: dict, codes: set[int]) -> dict:
    """Distinguish actual source NoData from insufficient batch-edge support.

    Args:
        table: Statewide cell QA, including measured partial coverage.
        grid: Authoritative cells with spatial batch indices.
        metadata: Validated native source lattice.
        codes: Documented class domain.

    Returns:
        Spatial extent of partial cells and independently verified NoData counts.

    Raises:
        ValueError: If any sample lacks source support or source counts disagree.
    """
    partial = grid.merge(table.loc[table.valid_sample_count.lt(POSITIONS)],
                         on="cell_id", validate="one_to_one")
    if partial.empty:
        return {"partial_cells_verified": 0, "source_nodata_positions": 0,
                "unsupported_destination_positions": 0}
    source_nodata_count = 0
    for key, cells in partial.groupby(["batch_row", "batch_column"], sort=True):
        batch = grid.loc[grid.batch_row.eq(key[0]) & grid.batch_column.eq(key[1])]
        name = f"r{key[0]:03d}_c{key[1]:03d}"
        parameters = native_parameters(metadata, batch.total_bounds)
        native, transform = read_native(EXPORT_DIR / f"{name}.tif", parameters, codes)
        for cell in cells.itertuples():
            left, _, _, top = cell.geometry.bounds
            # This independent check bypasses GDAL reprojection: both lattices
            # are EPSG:5070, so destination centers map directly into native
            # pixel coordinates. floor selects the containing nearest pixel.
            # np.ix_ takes all row/column combinations, giving 40 x 40 samples.
            offsets = (np.arange(SAMPLES_SIDE) + 0.5) * RESOLUTION_M
            columns = np.floor((left + offsets - transform.c) / transform.a).astype(int)
            rows = np.floor((top - offsets - transform.f) / transform.e).astype(int)
            if (rows.min() < 0 or rows.max() >= native.shape[0]
                    or columns.min() < 0 or columns.max() >= native.shape[1]):
                raise ValueError(f"Insufficient native source support: {cell.cell_id}")
            samples = native[np.ix_(rows, columns)]
            missing = int(np.count_nonzero(samples == NODATA))
            if missing != cell.nodata_sample_count:
                raise ValueError(f"Source and destination NoData disagree: {cell.cell_id}")
            source_nodata_count += missing
    # Geographic coordinates are QA only. Transform projected centroids
    # explicitly to NAD83 longitude/latitude; do not alter analysis geometry.
    centers = partial.geometry.centroid.to_crs(4269)
    export = pd.DataFrame(partial.drop(columns="geometry"))
    export["centroid_longitude_nad83"] = centers.x.to_numpy()
    export["centroid_latitude_nad83"] = centers.y.to_numpy()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    export.sort_values("cell_id").to_csv(REPORT_DIR / "partial_coverage_cells.csv", index=False)
    return {"partial_cells_verified": len(partial),
            "source_nodata_positions": source_nodata_count,
            "unsupported_destination_positions": 0,
            "partial_cell_bounds_epsg5070": partial.total_bounds.tolist(),
            "partial_centroid_bounds_nad83": centers.total_bounds.tolist()}


def coverage_report(table: pd.DataFrame, observed: set[int]) -> dict:
    """Measure statewide coverage, ties, and dominance before publication gates.

    Args:
        table: All sorted cell QA results.
        observed: Union of classes sampled within authoritative cells only.

    Returns:
        JSON-compatible metrics; zero-valid cells remain explicitly visible.
    """
    valid = table.valid_sample_count
    fractions = table.evt_dominant_fraction
    distribution = fractions.describe(percentiles=[0.25, 0.5, 0.75])
    return {"processed_cells": len(table), "unique_cell_ids": int(table.cell_id.nunique()),
            "full_coverage_cells": int(valid.eq(POSITIONS).sum()),
            "partial_coverage_cells": int(valid.between(1, POSITIONS - 1).sum()),
            "zero_valid_cells": int(valid.eq(0).sum()),
            "valid_samples_min_median_max": [int(valid.min()), float(valid.median()), int(valid.max())],
            "nodata_positions": int(table.nodata_sample_count.sum()),
            "unique_winner_cells": int((valid.gt(0) & ~table.plurality_tie).sum()),
            "exact_tie_cells": int(table.plurality_tie.sum()),
            "exact_tie_percent": float(table.plurality_tie.mean() * 100),
            "distinct_sample_classes": len(observed), "observed_classes": sorted(observed),
            "distinct_dominant_classes": int(table.evt_dominant_class.nunique()),
            "most_frequent_dominant_classes": {str(k): int(v) for k, v in
                                               table.evt_dominant_class.value_counts().head(10).items()},
            "dominant_fraction": {key: float(distribution[key]) if pd.notna(distribution[key]) else None
                                  for key in ["min", "25%", "50%", "mean", "75%", "max"]},
            "duplicate_ids": int(table.cell_id.duplicated().sum()),
            "missing_values": {key: int(table[key].isna().sum()) for key in SCHEMA},
            "invalid_fractions": int((~np.isfinite(fractions) | fractions.le(0) | fractions.gt(1)).sum())}


def peak_memory_mib() -> float | None:
    """Read the process peak resident working set when the OS exposes it.

    Returns:
        Peak MiB including grid geometry and GDAL, or None if unavailable.
    """
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in
                ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                 "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                 "PagefileUsage", "PeakPagefileUsage"]]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        if get_memory(current_process(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 1024**2
        return None
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024**2 if sys.platform == "darwin" else 1024)
    except ImportError:
        return None


# %% Main workflow
def main() -> None:
    """Validate small, scale with checkpoints, and publish only after statewide QA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true", help="Run production-path checks without scaling.")
    arguments = parser.parse_args()
    started = perf_counter()
    statistics = dict.fromkeys(["requests", "retries", "native_reused", "native_downloaded",
                               "batches_reused", "batches_computed"], 0)

    # STEP 1 — Validate authoritative inputs and LANDFIRE source.
    grid, metadata, codes, fingerprint = validate_inputs(statistics)

    # STEP 2 — Prepare statewide aligned processing grid (indices, not new cells).
    grid, batches, design = prepare_batches(grid)
    small_test = small_production_test(batches, metadata, codes, fingerprint, statistics)
    if arguments.test_only:
        print(f"Test runtime: {perf_counter() - started:.1f} seconds; {statistics}", flush=True)
        return

    # STEP 3 — Process LANDFIRE EVT in bounded-memory batches.
    # STEP 4 — Aggregate categorical samples by 1-km cell inside each batch.
    tables, observed = [], set()
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024):
        for index, (key, cells) in enumerate(batches, start=1):
            table, classes = process_batch(key, cells, metadata, codes, fingerprint, statistics)
            tables.append(table)
            observed.update(classes)
            print(f"Batch {index}/{len(batches)} {key}: {len(cells):,} cells; "
                  f"elapsed {perf_counter() - started:.1f}s", flush=True)
    table = pd.concat(tables, ignore_index=True).sort_values("cell_id").reset_index(drop=True)
    validate_table(table, grid.cell_id, codes)

    # STEP 5 — Run statewide coverage and tie QA without filtering or filling.
    summary = {"source": {"product": PRODUCT, "url": SERVICE_URL, "crs": CRS,
                          "native_resolution_m": 30, "destination_resolution_m": RESOLUTION_M,
                          "resampling": "nearest", "nodata": NODATA},
               "design": design, "qa": coverage_report(table, observed), "small_test": small_test,
               "review_gates": {"maximum_partial_cell_fraction": MAX_PARTIAL_FRACTION,
                                "maximum_exact_tie_fraction": MAX_TIE_FRACTION,
                                "zero_valid_cells_allowed": 0}, "published": False}
    write_table(table, CHECKPOINT_DIR / "statewide_qa.parquet", grid.cell_id, codes)
    write_json(summary, REPORT_DIR / "statewide_qa.json")
    # Keep the full code union in JSON; printing 175 individual identifiers
    # obscures the statewide measurements the console report should emphasize.
    console_qa = {key: value for key, value in summary["qa"].items() if key != "observed_classes"}
    print("STATEWIDE QA\n" + json.dumps(console_qa, indent=2), flush=True)
    summary["source_support_qa"] = verify_partial_source_support(table, grid, metadata, codes)
    write_json(summary, REPORT_DIR / "statewide_qa.json")
    if (table.valid_sample_count.eq(0).any()
            or table.valid_sample_count.between(1, POSITIONS - 1).mean() > MAX_PARTIAL_FRACTION
            or table.plurality_tie.mean() > MAX_TIE_FRACTION):
        raise ValueError("Coverage/tie review gate triggered; QA saved, final publication stopped.")

    # STEP 6 — Validate prototype regression.
    summary["prototype_regression"] = compare_prototype(table)

    # STEP 7 — Assemble final feature table, using only the three approved fields.
    features = table[SCHEMA].copy()
    validate_table(features, grid.cell_id, codes, final=True)

    # STEP 8 — Write, read back, and validate final Parquet before atomic rename.
    write_table(features, FINAL_PATH, grid.cell_id, codes, final=True)
    validate_table(pd.read_parquet(FINAL_PATH), grid.cell_id, codes, final=True)

    # STEP 9 — Report QA and performance; preserve enough provenance for reruns.
    summary.update({"published": True, "output": str(FINAL_PATH.relative_to(ROOT)),
                    "schema": SCHEMA, "file_bytes": FINAL_PATH.stat().st_size,
                    "output_sha256": digest_file(FINAL_PATH), "input_identity": fingerprint,
                    "runtime_seconds": perf_counter() - started,
                    "peak_working_set_mib": peak_memory_mib(), "acquisition": statistics})
    write_json(summary, REPORT_DIR / "statewide_qa.json")
    console_summary = {key: value for key, value in summary.items()
                       if key not in {"qa", "input_identity", "small_test"}}
    print("SUCCESS\n" + json.dumps(console_summary, indent=2), flush=True)


# %% Run script
if __name__ == "__main__":
    main()
