"""Prototype LANDFIRE 2016 Remap Existing Vegetation Type (EVT).

LANDFIRE = Landscape Fire and Resource Management Planning Tools. EVT describes
categorical vegetation/ecological classes: integer values are identifiers, not
quantities. Nearest-neighbor reprojection preserves class identity on an aligned
sampling grid. Each 1-km cell receives its plurality class and that class's
fraction of valid samples. Circa-2016 vegetation precedes the 2017-2022 MTBS
target period. This small prototype does not produce statewide features.
"""

# %% Imports
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject


# %% Constants and paths
PRODUCT = "LANDFIRE 2016 Remap EVT CONUS (LF2016_EVT_CONUS; 2016 legend v200)"
SERVICE_NAME = "Landfire_LF2016/LF2016_EVT_CONUS"
SERVICE_URL = f"https://lfps.usgs.gov/arcgis/rest/services/{SERVICE_NAME}/ImageServer"
LEGEND_URL = "https://www.landfire.gov/sites/default/files/CSV/LF2016/LF16_EVT_200.csv"
PRODUCT_URL = "https://www.landfire.gov/data/lf2016"
TARGET_CRS = "EPSG:5070"
CELL_SIZE_M = 1_000
EXPECTED_GRID_COUNT = 314_920
SIDE_CELLS = 8
NODATA = -9999
EXPECTED_NATIVE_RESOLUTION_M = 30
# A fixed NAD83 geographic reference near the Sandia Mountains selects a mixed
# vegetation setting independently of EVT values or subsequent wildfire labels.
ANCHOR_LON_LAT_NAD83 = (-106.4, 35.2)
REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = REPO_ROOT / "data/processed/grids/nm_analysis_grid_1km.gpkg"
RAW_DIR = REPO_ROOT / "data/raw/landfire"
SERVICE_PATH = RAW_DIR / "lf2016_evt_service.json"
LEGEND_PATH = RAW_DIR / "LF16_EVT_200.csv"
NATIVE_PATH = RAW_DIR / "lf2016_evt_sandia_native.tif"
EXPORT_PATH = RAW_DIR / "lf2016_evt_export.json"
TABLE_PATH = REPO_ROOT / "data/interim/landfire_evt_prototype.parquet"
PLOT_DIR = REPO_ROOT / "outputs/landfire_evt_prototype"


# %% Helper functions
def request_json(url: str, parameters: dict | None = None) -> dict:
    """Read official service JSON, rejecting application-level errors.

    Args:
        url: Official USGS service endpoint.
        parameters: Optional REST query parameters.

    Returns:
        Decoded service response.

    Raises:
        ValueError: If ArcGIS returns an error instead of the requested data.
        OSError: If the endpoint is unavailable; no version fallback is used.
    """
    query_url = url + "?" + urlencode(parameters or {"f": "json"})
    with urlopen(query_url, timeout=60) as response:
        result = json.load(response)
    if "error" in result:
        raise ValueError(f"LANDFIRE service error: {result['error']}")
    return result


def write_json(data: dict, destination: Path) -> None:
    """Write a small JSON artifact through a temporary file.

    Args:
        data: Serializable metadata or diagnostics.
        destination: Final JSON path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_binary(url: str, destination: Path) -> None:
    """Stream a small official source artifact without retaining partial work.

    Args:
        url: Source URL; the caller validates the acquired content before use.
        destination: Local cache path, reused if already present.

    Raises:
        OSError: If acquisition fails; no alternative product is substituted.
    """
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": "wildfire-susceptibility-ml/0.1"})
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def inspect_source_and_legend() -> tuple[dict, pd.DataFrame]:
    """Validate the pinned 2016 service and version-specific categorical legend.

    Official full-extent downloads and image services were inspected. A small
    raw-value GeoTIFF export avoids downloading the multi-GB national archive.
    The default image-service renderer returns vegetation names/symbols, so the
    raster request below explicitly disables it with rasterFunction='None'.

    Returns:
        Service metadata and the 2016 code-to-description/color table.

    Raises:
        ValueError: If product identity, raster properties, or legend change.
    """
    if not SERVICE_PATH.exists():
        write_json(request_json(SERVICE_URL), SERVICE_PATH)
    metadata = json.loads(SERVICE_PATH.read_text(encoding="utf-8"))
    expected = {"name": SERVICE_NAME, "bandCount": 1, "pixelType": "S16",
                "pixelSizeX": 30, "pixelSizeY": 30, "noDataValue": NODATA}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unexpected source {key}: {metadata.get(key)}")
    extent = metadata["extent"]
    if extent["spatialReference"]["wkid"] != 5070:
        raise ValueError("Expected the observed native EPSG:5070 service.")
    download_binary(LEGEND_URL, LEGEND_PATH)
    legend = pd.read_csv(LEGEND_PATH)
    required = {"VALUE", "EVT_NAME", "R", "G", "B"}
    if not required.issubset(legend.columns) or not legend.VALUE.is_unique:
        raise ValueError("Unexpected 2016 EVT legend schema or duplicate class codes.")
    if legend[list(required)].isna().any().any():
        raise ValueError("Required legend values are missing.")
    # Only documented fill is excluded. Water, developed, sparse, agricultural,
    # and ruderal codes are valid classes, not missing vegetation. A code such
    # as 7292 (Open Water) is a label, never a magnitude or an averaging input.
    special = legend.EVT_NAME.str.contains("NoData|Background|Unknown|Fill", case=False)
    excluded = set(legend.loc[special | (legend.VALUE <= 0), "VALUE"])
    if excluded != {NODATA}:
        raise ValueError(f"Special class codes require review: {excluded}")
    width = (extent["xmax"] - extent["xmin"]) / metadata["pixelSizeX"]
    height = (extent["ymax"] - extent["ymin"]) / metadata["pixelSizeY"]
    if width != int(width) or height != int(height):
        raise ValueError("Service extent does not fit its declared native lattice.")
    print(f"SOURCE: {PRODUCT}\n{SERVICE_URL}\nLegend: {LEGEND_URL}")
    print(f"Native mosaic: EPSG:5070; {int(height)} rows x {int(width)} columns; "
          f"30 m; int16; NoData={NODATA}; format={metadata.get('datasetFormat')}")
    print(f"Native extent: {extent}; legend classes including fill: {len(legend)}")
    print("Exclude only -9999 Fill-NoData; unknown observed codes cause a stop.")
    return metadata, legend.set_index("VALUE", drop=False)


def load_analysis_grid() -> gpd.GeoDataFrame:
    """Validate authoritative complete cells without rebuilding or clipping them.

    Returns:
        Existing grid with its original identifiers and geometry.

    Raises:
        ValueError: If count, CRS, IDs, geometry, or complete square areas fail QA.
    """
    grid = gpd.read_file(GRID_PATH, layer="analysis_grid")
    if grid.crs is None or grid.crs.to_epsg() != 5070:
        raise ValueError("Authoritative grid must already use EPSG:5070.")
    if len(grid) != EXPECTED_GRID_COUNT or "cell_id" not in grid:
        raise ValueError("Unexpected authoritative grid count/schema.")
    if grid.cell_id.isna().any() or not grid.cell_id.is_unique:
        raise ValueError("Authoritative IDs must be present and unique.")
    geometry = grid.geometry
    if (geometry.isna().any() or geometry.is_empty.any()
            or not geometry.is_valid.all() or not geometry.geom_type.eq("Polygon").all()):
        raise ValueError("Unexpected grid geometry; no repair is applied.")
    bounds = geometry.bounds
    if (not np.allclose(geometry.area, CELL_SIZE_M**2, rtol=0, atol=1e-3)
            or not np.allclose(bounds.maxx - bounds.minx, CELL_SIZE_M, rtol=0, atol=1e-6)
            or not np.allclose(bounds.maxy - bounds.miny, CELL_SIZE_M, rtol=0, atol=1e-6)):
        raise ValueError("Grid must contain complete 1-km squares.")
    return grid[["cell_id", "geometry"]]


def select_prototype(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select a deterministic 8-by-8 block near a fixed Sandia reference point.

    Args:
        grid: Authoritative statewide grid; only 64 cells enter raster processing.

    Returns:
        Complete selected cells sorted by their original IDs.

    Raises:
        ValueError: If the selection is not a complete contiguous 8-km square.
    """
    # The reference point is explicitly NAD83, matching the datum underlying
    # EPSG:5070. Transform errors must raise, not silently select a wrong cell.
    transformer = Transformer.from_crs(4269, 5070, always_xy=True)
    x, y = transformer.transform(*ANCHOR_LON_LAT_NAD83, errcheck=True)
    centers = grid.geometry.centroid
    distances = (centers.x - x)**2 + (centers.y - y)**2
    anchor_id = grid.loc[distances.eq(distances.min()), "cell_id"].sort_values().iloc[0]
    anchor = grid.loc[grid.cell_id.eq(anchor_id)].iloc[0]
    left = anchor.geometry.bounds[0] - (SIDE_CELLS // 2) * CELL_SIZE_M
    bottom = anchor.geometry.bounds[1] - (SIDE_CELLS // 2) * CELL_SIZE_M
    right, top = left + SIDE_CELLS * CELL_SIZE_M, bottom + SIDE_CELLS * CELL_SIZE_M
    mask = (centers.x.ge(left) & centers.x.lt(right)
            & centers.y.ge(bottom) & centers.y.lt(top))
    prototype = grid.loc[mask].sort_values("cell_id").reset_index(drop=True)
    if len(prototype) != SIDE_CELLS**2 or not np.array_equal(
        prototype.total_bounds, [left, bottom, right, top]
    ):
        raise ValueError("Prototype is not the expected complete 8-by-8 block.")
    print(f"Prototype: anchor {anchor_id}, NAD83 point {ANCHOR_LON_LAT_NAD83}, "
          f"{len(prototype)} cells, bounds={prototype.total_bounds.tolist()}")
    return prototype


def native_export_parameters(metadata: dict, grid: gpd.GeoDataFrame) -> dict:
    """Snap the small extent outward to source pixels, plus a one-pixel halo.

    Args:
        metadata: Validated native service origin and resolution.
        grid: The selected 64 authoritative cells.

    Returns:
        Raw-value native-resolution GeoTIFF export parameters.
    """
    left, bottom, right, top = grid.total_bounds
    extent = metadata["extent"]
    resolution = metadata["pixelSizeX"]
    origin_x, origin_y = extent["xmin"], extent["ymax"]
    # Native pixel boundaries are offset from the 1-km analysis lattice. Snap
    # to that native lattice so acquisition itself does not shift source pixels.
    left = origin_x + (np.floor((left - origin_x) / resolution) - 1) * resolution
    right = origin_x + (np.ceil((right - origin_x) / resolution) + 1) * resolution
    top = origin_y - (np.floor((origin_y - top) / resolution) - 1) * resolution
    bottom = origin_y - (np.ceil((origin_y - bottom) / resolution) + 1) * resolution
    width, height = int((right-left)/resolution), int((top-bottom)/resolution)
    return {"f": "json", "bbox": f"{left},{bottom},{right},{top}",
            "bboxSR": 5070, "imageSR": 5070, "size": f"{width},{height}",
            "format": "tiff", "pixelType": "S16", "noData": NODATA,
            "interpolation": "RSP_NearestNeighbor", "adjustAspectRatio": "false",
            "renderingRule": json.dumps({"rasterFunction": "None"})}


def acquire_native_raster(metadata: dict, grid: gpd.GeoDataFrame) -> Path:
    """Acquire/reuse only a small native-grid export with recorded provenance.

    Args:
        metadata: Validated source metadata.
        grid: Selected prototype cells.

    Returns:
        Local native-resolution GeoTIFF path.

    Raises:
        ValueError: If cached provenance or exported raster properties disagree.
    """
    parameters = native_export_parameters(metadata, grid)
    if NATIVE_PATH.exists():
        if not EXPORT_PATH.exists():
            raise ValueError("Cached native export has no provenance record.")
        provenance = json.loads(EXPORT_PATH.read_text(encoding="utf-8"))
        if provenance["service"] != SERVICE_URL or provenance["parameters"] != parameters:
            raise ValueError("Cached native export does not match this request.")
    else:
        response = request_json(SERVICE_URL + "/exportImage", parameters)
        provenance = {"service": SERVICE_URL, "parameters": parameters, "response": response}
        download_binary(response["href"], NATIVE_PATH)
        write_json(provenance, EXPORT_PATH)
    expected_bounds = tuple(float(value) for value in parameters["bbox"].split(","))
    expected_width, expected_height = map(int, parameters["size"].split(","))
    with rasterio.open(NATIVE_PATH) as source:
        if (source.driver != "GTiff" or source.crs.to_epsg() != 5070
                or source.res != (30, 30) or source.count != 1
                or source.dtypes != ("int16",) or source.nodata != NODATA
                or source.shape != (expected_height, expected_width)
                or tuple(source.bounds) != expected_bounds
                or source.transform.b != 0 or source.transform.d != 0):
            raise ValueError("Export differs from the requested native categorical raster.")
        print(f"Native GeoTIFF: {source.shape}; bounds={tuple(source.bounds)}; "
              f"transform={tuple(source.transform)}; tags={source.tags()}")
    return NATIVE_PATH


def design_destination(
    grid: gpd.GeoDataFrame, native_resolution: float
) -> tuple[int, tuple[int, int], Affine]:
    """Choose the coarsest whole-meter cell divisor no coarser than the source.

    Args:
        grid: Prototype cells whose bounds anchor the destination raster.
        native_resolution: Inspected native pixel size in meters.

    Returns:
        Resolution, destination shape, and affine transform.

    Raises:
        ValueError: If the native resolution or sample alignment is unexpected.
    """
    if native_resolution != EXPECTED_NATIVE_RESOLUTION_M:
        raise ValueError("Review intermediate resolution for changed source pixels.")
    candidates = [size for size in range(1, int(native_resolution) + 1)
                  if CELL_SIZE_M % size == 0]
    resolution = max(candidates)
    left, bottom, right, top = grid.total_bounds
    shape = (int((top-bottom)/resolution), int((right-left)/resolution))
    transform = from_origin(left, top, resolution, resolution)
    print(f"Destination: {resolution} m, shape={shape}; "
          f"{CELL_SIZE_M // resolution} x {CELL_SIZE_M // resolution} samples/cell.")
    print("30 m does not divide 1,000 m; 25 m preserves more detail than a coarser "
          "divisor and uses fewer samples than 20 m. It adds no new observations.")
    return resolution, shape, transform


def reproject_categorical(
    path: Path, shape: tuple[int, int], transform: Affine, legend: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Realign categorical pixels with nearest-neighbor, never interpolation.

    Args:
        path: Validated native-resolution GeoTIFF.
        shape: Small destination array shape.
        transform: Destination transform aligned with complete analysis cells.
        legend: Official version-specific class dictionary.

    Returns:
        Destination values, native values, and native transform for QA plotting.

    Raises:
        ValueError: If source or destination codes are absent from the legend,
            or destination contains categorical values absent from the source.
    """
    with rasterio.open(path) as source:
        native = source.read(1, masked=True).filled(NODATA)
        native_transform = source.transform
        unknown = set(np.unique(native)).difference(legend.index)
        if unknown:
            raise ValueError(f"Source codes absent from 2016 legend: {unknown}")
        destination = np.full(shape, NODATA, dtype=np.int16)
        # Both CRSs are EPSG:5070, but origins and pixel spacing differ.
        # reproject maps each destination sample to an existing nearest source
        # class. Bilinear would invent numeric codes between unrelated labels.
        reproject(source=native, destination=destination,
                  src_transform=source.transform, src_crs=source.crs,
                  src_nodata=NODATA, dst_transform=transform, dst_crs=TARGET_CRS,
                  dst_nodata=NODATA, resampling=Resampling.nearest,
                  init_dest_nodata=True)
    if not set(np.unique(destination)).issubset(set(np.unique(native)) | {NODATA}):
        raise ValueError("Nearest-neighbor introduced a new categorical value.")
    return destination, native, native_transform


def aggregate_categories(
    values: np.ndarray, grid: gpd.GeoDataFrame, resolution: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count aligned samples per class and preserve exact plurality ties.

    The mode is the most frequent category. 'Plurality' emphasizes that its
    share need not exceed 50%: 42% forest beats 35% shrub and 23% grass.
    'Dominant class' names this feature; its fraction uses VALID samples only.
    Exact ties provisionally use the smallest code solely for determinism,
    not because codes have ecological rank. All counts and tied codes survive.

    Args:
        values: Nearest-neighbor categorical destination array.
        grid: Authoritative prototype cells in any row order.
        resolution: Destination spacing dividing 1,000 m exactly.

    Returns:
        Cell table with QA-only coverage/tie fields and a long class-count table.

    Raises:
        ValueError: If sample dimensions or cell-to-array alignment disagree.
    """
    samples_per_side = CELL_SIZE_M // resolution
    if CELL_SIZE_M % resolution or values.shape != (SIDE_CELLS*samples_per_side,)*2:
        raise ValueError("Destination samples do not fit the prototype cell lattice.")
    # Reshape into cell rows x sample rows x cell columns x sample columns.
    # Each selected cell maps by spatial position, never by GeoDataFrame order.
    grouped = values.reshape(SIDE_CELLS, samples_per_side, SIDE_CELLS, samples_per_side)
    left, _, _, top = grid.total_bounds
    records, count_records = [], []
    for cell in grid.itertuples():
        center = cell.geometry.centroid
        row_position = (top-center.y)/CELL_SIZE_M - 0.5
        column_position = (center.x-left)/CELL_SIZE_M - 0.5
        row, column = int(round(row_position)), int(round(column_position))
        if (not np.allclose([row_position, column_position], [row, column], rtol=0, atol=1e-9)
                or not 0 <= row < SIDE_CELLS or not 0 <= column < SIDE_CELLS):
            raise ValueError(f"Misaligned authoritative cell: {cell.cell_id}")
        samples = grouped[row, :, column, :].ravel()
        valid = samples[samples != NODATA]
        classes, counts = np.unique(valid, return_counts=True)
        maximum = int(counts.max()) if len(counts) else 0
        winners = classes[counts == maximum]
        record = {"cell_id": cell.cell_id,
                  "evt_dominant_class": int(winners[0]) if len(winners) else pd.NA,
                  "evt_dominant_fraction": maximum/len(valid) if len(valid) else np.nan,
                  "valid_sample_count": len(valid), "winning_sample_count": maximum,
                  "number_of_classes": len(classes),
                  "nodata_sample_count": len(samples)-len(valid),
                  "plurality_tie": len(winners) > 1,
                  "tied_classes": json.dumps(winners.tolist()) if len(winners) > 1 else "[]"}
        records.append(record)
        for code, count in zip(classes, counts):
            count_records.append({"cell_id": cell.cell_id, "evt_class": int(code),
                                  "sample_count": int(count)})
    table = pd.DataFrame(records).sort_values("cell_id").reset_index(drop=True)
    table["evt_dominant_class"] = table.evt_dominant_class.astype("Int64")
    class_counts = pd.DataFrame(count_records, columns=["cell_id", "evt_class", "sample_count"])
    return table, class_counts.sort_values(["cell_id", "evt_class"]).reset_index(drop=True)


def validate_and_report(
    table: pd.DataFrame, counts: pd.DataFrame, grid: gpd.GeoDataFrame, resolution: int
) -> dict:
    """Validate keyed categorical outputs and report coverage without filling it.

    Args:
        table: Prototype cell features and QA-only fields.
        counts: Preserved per-cell class counts.
        grid: Selected authoritative cells.
        resolution: Aligned sample spacing.

    Returns:
        JSON-compatible summary including missingness, ties, and fractions.

    Raises:
        ValueError: If keys, counts, fractions, or missing-value semantics fail.
    """
    expected_ids = grid.cell_id.sort_values().reset_index(drop=True)
    if not table.cell_id.equals(expected_ids) or not table.cell_id.is_unique:
        raise ValueError("Prototype keys do not match the selected authoritative cells.")
    total_samples = (CELL_SIZE_M // resolution)**2
    if not (table.valid_sample_count + table.nodata_sample_count).eq(total_samples).all():
        raise ValueError("Sample counts do not sum to complete cell support.")
    valid = table.valid_sample_count > 0
    if (not table.loc[valid, "evt_dominant_fraction"].between(0, 1).all()
            or table.loc[valid, "evt_dominant_class"].isna().any()
            or table.loc[~valid, "evt_dominant_class"].notna().any()
            or table.loc[~valid, "evt_dominant_fraction"].notna().any()):
        raise ValueError("Unexpected fraction domain or missing-class behavior.")
    grouped = counts.groupby("cell_id").sample_count
    indexed = table.set_index("cell_id")
    if (not grouped.sum().reindex(indexed.index, fill_value=0).eq(indexed.valid_sample_count).all()
            or not grouped.max().reindex(indexed.index, fill_value=0).eq(indexed.winning_sample_count).all()
            or not np.allclose(table.loc[valid, "evt_dominant_fraction"],
                               table.loc[valid, "winning_sample_count"] /
                               table.loc[valid, "valid_sample_count"], rtol=0, atol=1e-12)):
        raise ValueError("Preserved class counts disagree with reported features.")
    summary = {"cells": len(table), "unique_cell_id": table.cell_id.nunique(),
               "duplicate_ids": int(table.cell_id.duplicated().sum()),
               "missing_dominant_class": int(table.evt_dominant_class.isna().sum()),
               "missing_dominant_fraction": int(table.evt_dominant_fraction.isna().sum()),
               "distinct_sample_classes": int(counts.evt_class.nunique()),
               "distinct_dominant_classes": int(table.evt_dominant_class.nunique()),
               "plurality_ties": int(table.plurality_tie.sum()),
               "valid_sample_min": int(table.valid_sample_count.min()),
               "valid_sample_max": int(table.valid_sample_count.max()),
               "nodata_samples": int(table.nodata_sample_count.sum()),
               "cells_with_nodata": int((table.nodata_sample_count > 0).sum())}
    for statistic in ["min", "median", "mean", "max"]:
        value = getattr(table.evt_dominant_fraction, statistic)()
        summary[f"dominant_fraction_{statistic}"] = float(value) if pd.notna(value) else None
    print("\nPROTOTYPE QA/QC\n" + json.dumps(summary, indent=2))
    print("Tie rule is PROVISIONAL: smallest class code; see tied_classes and class counts.")
    if table.plurality_tie.any():
        print(table.loc[table.plurality_tie].to_string(index=False))
    if not valid.all():
        print("WARNING: all-NoData cells retain missing features; no vegetation was filled.")
    return summary


def write_prototype(table: pd.DataFrame, counts: pd.DataFrame, legend: pd.DataFrame) -> None:
    """Persist a small verified interim table and inspectable class counts.

    Args:
        table: Prototype features plus explicitly QA-only extra columns.
        counts: Per-cell categorical counts for inspecting plurality and ties.
        legend: Official 2016 descriptions attached to the QA count CSV.

    Raises:
        ValueError: If the temporary Parquet does not read back exactly.
    """
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = TABLE_PATH.with_suffix(".parquet.part")
    try:
        table.to_parquet(partial, index=False)
        if not pd.read_parquet(partial).equals(table):
            raise ValueError("Prototype Parquet read-back differs.")
        partial.replace(TABLE_PATH)
    finally:
        partial.unlink(missing_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    annotated = counts.copy()
    annotated["evt_name"] = annotated.evt_class.map(legend.EVT_NAME)
    annotated.to_csv(PLOT_DIR / "class_counts.csv", index=False)
    represented = legend.loc[sorted(counts.evt_class.unique()), ["VALUE", "EVT_NAME"]]
    represented.to_csv(PLOT_DIR / "represented_classes.csv", index=False)
    print("\nObserved categorical classes (codes are labels):")
    print(represented.to_string(index=False))


def plot_spatial_qa(
    native: np.ndarray, native_transform: Affine, destination: np.ndarray,
    destination_transform: Affine, grid: gpd.GeoDataFrame,
    table: pd.DataFrame, legend: pd.DataFrame,
) -> None:
    """Plot native/aligned categories, cell fractions, and pixel-alignment detail.

    Args:
        native: Native 30-m categorical values around the prototype.
        native_transform: Native GeoTIFF affine transform.
        destination: Aligned categorical values.
        destination_transform: Aligned sampling transform.
        grid: Prototype authoritative cell polygons.
        table: Cell features and QA-only fields.
        legend: Official class names and RGB values for categorical colors.
    """
    codes = np.union1d(np.unique(native), np.unique(destination))
    codes = codes[codes != NODATA]
    colors = legend.loc[codes, ["R", "G", "B"]].to_numpy()/255
    cmap = ListedColormap(colors)
    cmap.set_bad("lightgray")
    mapped_grid = grid.merge(table, on="cell_id", validate="one_to_one")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for axis, array, transform, title in [
        (axes[0], native, native_transform, "Native EVT: 30 m"),
        (axes[1], destination, destination_transform, "Aligned EVT: 25 m nearest-neighbor"),
    ]:
        # Map arbitrary codes to color indices only for display; numerical code
        # spacing has no meaning. Official RGB colors are preserved.
        indices = np.ma.masked_where(array == NODATA, np.searchsorted(codes, array))
        height, width = array.shape
        extent = (transform.c, transform.c+width*transform.a,
                  transform.f+height*transform.e, transform.f)
        axis.imshow(indices, extent=extent, origin="upper", interpolation="nearest",
                    cmap=cmap, vmin=-0.5, vmax=len(codes)-0.5)
        grid.boundary.plot(ax=axis, color="black", linewidth=0.5)
        axis.set_title(title)
    mapped_grid.plot(column="evt_dominant_fraction", ax=axes[2], cmap="viridis",
                     vmin=0, vmax=1, legend=True, edgecolor="black", linewidth=0.5,
                     missing_kwds={"color": "lightgray"})
    axes[2].set_title("1-km plurality fraction (valid samples)")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("EPSG:5070 x (m)")
        axis.set_ylabel("EPSG:5070 y (m)")
        axis.ticklabel_format(style="plain", useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle("LANDFIRE 2016 Remap EVT: Sandia 8 x 8 prototype\n"
                 "Categorical names and codes: represented_classes.csv")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "spatial_qa.png", dpi=150)
    plt.close(fig)

    # A 200-m zoom inside the cell with the most classes exposes both origins.
    mixed = mapped_grid.sort_values(["number_of_classes", "cell_id"],
                                   ascending=[False, True]).iloc[0]
    x, y = mixed.geometry.centroid.coords[0]
    fig, axis = plt.subplots(figsize=(8, 7))
    indices = np.ma.masked_where(native == NODATA, np.searchsorted(codes, native))
    height, width = native.shape
    extent = (native_transform.c, native_transform.c+width*native_transform.a,
              native_transform.f+height*native_transform.e, native_transform.f)
    axis.imshow(indices, extent=extent, origin="upper", interpolation="nearest",
                cmap=cmap, vmin=-0.5, vmax=len(codes)-0.5)
    for transform, color, style, label in [
        (native_transform, "black", "-", "Native 30-m pixel edges"),
        (destination_transform, "cyan", "--", "Aligned 25-m sample edges"),
    ]:
        for coordinate in np.arange(transform.c, extent[1]+30, transform.a):
            if x-100 <= coordinate <= x+100:
                axis.axvline(coordinate, color=color, linestyle=style, linewidth=0.8)
        for coordinate in np.arange(transform.f, extent[2]-30, transform.e):
            if y-100 <= coordinate <= y+100:
                axis.axhline(coordinate, color=color, linestyle=style, linewidth=0.8)
        axis.plot([], [], color=color, linestyle=style, label=label)
    axis.set(xlim=(x-100, x+100), ylim=(y-100, y+100), aspect="equal",
             title=f"Different raster lattices inside {mixed.cell_id}",
             xlabel="EPSG:5070 x (m)", ylabel="EPSG:5070 y (m)")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "alignment_zoom.png", dpi=150)
    plt.close(fig)


# %% Main workflow
def main() -> None:
    """Run the small source-inspection and categorical-aggregation prototype."""
    started = perf_counter()
    # STEP 1 - Locate/acquire and validate LANDFIRE 2016 Remap EVT
    metadata, legend = inspect_source_and_legend()

    # STEP 2 - Inspect raster metadata and categorical class structure
    print("2016 legend example: 7292 =", legend.loc[7292, "EVT_NAME"])
    print("The service's symbol renderer is disabled for raw categorical export.")

    # STEP 3 - Load authoritative 1-km analysis grid
    analysis_grid = load_analysis_grid()

    # STEP 4 - Select a small deterministic prototype area
    prototype = select_prototype(analysis_grid)
    native_path = acquire_native_raster(metadata, prototype)

    # STEP 5 - Design an aligned categorical destination grid
    resolution, shape, transform = design_destination(prototype, metadata["pixelSizeX"])

    # STEP 6 - Reproject EVT with nearest-neighbor resampling
    aligned, native, native_transform = reproject_categorical(native_path, shape, transform, legend)

    # STEP 7 - Aggregate categorical samples to 1-km cells
    table, counts = aggregate_categories(aligned, prototype, resolution)

    # STEP 8 - Inspect plurality and dominant-fraction results
    summary = validate_and_report(table, counts, prototype, resolution)
    write_prototype(table, counts, legend)

    # STEP 9 - Plot spatial QA
    plot_spatial_qa(native, native_transform, aligned, transform, prototype, table, legend)

    # STEP 10 - Report findings
    summary.update({"product": PRODUCT, "service": SERVICE_URL, "legend": LEGEND_URL,
                    "prototype_bounds": prototype.total_bounds.tolist(),
                    "destination_resolution_m": resolution,
                    "native_window_class_count": int(len(set(np.unique(native)) - {NODATA})),
                    "elapsed_seconds": perf_counter()-started,
                    "tie_rule": "PROVISIONAL smallest code; tied classes and counts retained",
                    "coverage_rule": "Exclude -9999; denominator is valid samples; no imputation"})
    write_json(summary, PLOT_DIR / "qa_summary.json")
    print(f"\nInterim table: {TABLE_PATH}\nQA artifacts: {PLOT_DIR}")
    print(f"Execution time: {summary['elapsed_seconds']:.2f} seconds")
    print("QA-only extra columns are not approved final ML features. "
          "Review tie handling and coverage requirements before statewide scaling.")


# %% Run script
if __name__ == "__main__":
    main()
