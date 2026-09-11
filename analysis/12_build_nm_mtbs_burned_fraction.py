"""Build continuous statewide MTBS burned fractions for New Mexico.

MTBS = Monitoring Trends in Burn Severity. The national dataset contains mapped
large-fire burned-area boundaries; this project retains only Wildfire records
with ignition dates in 2017-2022. Fire polygons are reprojected to EPSG:5070 and
selected against the complete analysis-grid footprint, retaining cross-border
fires. Selected wildfire footprints are unioned to avoid double-counting repeat
or overlapping fires. Each complete 1-km cell receives burned_area_m2 and
burned_fraction. Binary target assignment is deliberately postponed.
"""

# %% Imports
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
from shapely import union_all
from shapely.geometry.base import BaseGeometry


# %% Constants and paths
TARGET_CRS = "EPSG:5070"
EXPECTED_CELL_COUNT = 314_920
EXPECTED_CELL_AREA_M2 = 1_000_000
EXPECTED_FIRE_COUNT = 129  # QA baseline, not a selection constraint.
START_YEAR = 2017
END_YEAR = 2022
AREA_TOLERANCE_M2 = 1e-3
FRACTION_TOLERANCE = 1e-9
OUTPUT_COLUMNS = ["cell_id", "burned_area_m2", "burned_fraction"]
REQUIRED_FIELDS = {"event_id", "incid_name", "incid_type", "ig_date", "burnbndac"}
MTBS_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
    "composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
)
EXPECTED_SHAPEFILE_STEM = "mtbs_perims_DD"
EXPECTED_COMPONENT_SUFFIXES = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_PATH = REPO_ROOT / "data/processed/grids/nm_analysis_grid_1km.gpkg"
ARCHIVE_PATH = REPO_ROOT / "data/raw/mtbs/mtbs_perimeter_data.zip"
OUTPUT_PATH = REPO_ROOT / "data/processed/targets/nm_mtbs_burned_fraction_1km.parquet"


# %% Helper functions
def download_source_if_missing(url: str, destination: Path) -> None:
    """Download the authoritative MTBS archive safely when absent.

    The response is streamed in bounded chunks rather than held in memory.
    Bytes are written to a .part path and renamed only after success, so an
    interrupted transfer cannot look like a reusable complete archive.

    Args:
        url: Authoritative direct-download HTTPS URL.
        destination: Local cache path for the complete ZIP.

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
            # Chunking bounds download memory independently of archive size.
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        validate_archive_and_get_vector_path(partial_path)
        partial_path.replace(destination)
    finally:
        if partial_path.exists():
            partial_path.unlink()


def validate_archive_and_get_vector_path(archive_path: Path) -> str:
    """Validate the MTBS ZIP and return its archived shapefile path.

    ZIP checksum testing catches corrupt members. Required sidecar components
    are checked because an ESRI shapefile is a coordinated set of files, not
    only a .shp geometry file. GeoPandas can read the archive directly, avoiding
    an unnecessary extracted copy of the national dataset.

    Args:
        archive_path: Cached national MTBS ZIP.

    Returns:
        GeoPandas ZIP virtual-filesystem path to the shapefile.

    Raises:
        ValueError: If the ZIP is malformed or components are missing.
    """
    try:
        with ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            member_names = archive.namelist()
    except BadZipFile as error:
        raise ValueError(f"MTBS source is not a valid ZIP: {archive_path}") from error

    if bad_member is not None:
        raise ValueError(f"MTBS ZIP contains a corrupt member: {bad_member}")

    expected_names = {
        f"{EXPECTED_SHAPEFILE_STEM}{suffix}"
        for suffix in EXPECTED_COMPONENT_SUFFIXES
    }
    missing_components = expected_names.difference(member_names)
    if missing_components:
        raise ValueError(
            f"MTBS archive is missing shapefile components: {missing_components}"
        )
    shapefiles = [
        name for name in member_names if Path(name).suffix.lower() == ".shp"
    ]
    if shapefiles != [f"{EXPECTED_SHAPEFILE_STEM}.shp"]:
        raise ValueError(f"Unexpected MTBS shapefile members: {shapefiles}")

    return f"zip://{archive_path.as_posix()}!{shapefiles[0]}"


def validate_polygon_geometry(
    frame: gpd.GeoDataFrame, label: str, *, require_valid: bool = True
) -> None:
    """Report geometry issues and stop without repairing source data.

    Args:
        frame: Polygon records to inspect before spatial operations.
        label: Dataset description for diagnostics.
        require_valid: Stop on invalid topology. False is used only before
            spatial selection; selected fires must pass the strict check.

    Raises:
        ValueError: If missing, empty, invalid, or nonpolygon geometry occurs.
    """
    geometry = frame.geometry
    issues = {
        "missing": int(geometry.isna().sum()),
        "empty": int(geometry.is_empty.sum()),
        "invalid": int((~geometry.is_valid & geometry.notna()).sum()),
        "incompatible": int(
            (~geometry.geom_type.isin(["Polygon", "MultiPolygon"])).sum()
        ),
    }
    print(f"{label} geometry QA: {issues}", flush=True)
    if (issues["missing"] or issues["empty"] or issues["incompatible"]
            or (require_valid and issues["invalid"])):
        raise ValueError(f"{label} geometry requires investigation: {issues}")


def load_analysis_grid(path: Path) -> gpd.GeoDataFrame:
    """Load complete authoritative cells without reprojection or clipping.

    Args:
        path: Existing analysis-grid GeoPackage.

    Returns:
        Validated cells in deterministic authoritative-ID order.

    Raises:
        ValueError: If CRS, count, IDs, geometry, or cell areas are unexpected.
    """
    grid = gpd.read_file(path, layer="analysis_grid")
    if not {"cell_id", "geometry"}.issubset(grid.columns):
        raise ValueError("Grid requires cell_id and geometry.")
    if grid.crs is None or grid.crs.to_epsg() != 5070:
        raise ValueError(f"Grid must already use {TARGET_CRS}.")
    if len(grid) != EXPECTED_CELL_COUNT:
        raise ValueError(f"Expected {EXPECTED_CELL_COUNT:,} cells; got {len(grid):,}.")
    if grid.cell_id.isna().any() or not grid.cell_id.is_unique:
        raise ValueError("Grid IDs must be present and unique.")
    validate_polygon_geometry(grid, "Analysis grid")
    cell_areas = grid.geometry.area
    if not np.allclose(
        cell_areas, EXPECTED_CELL_AREA_M2, rtol=0, atol=AREA_TOLERANCE_M2
    ):
        raise ValueError("Grid contains unexpected cell areas.")
    print(f"Grid: {len(grid):,} complete cells; area range {cell_areas.min()}"
          f" to {cell_areas.max()} m2", flush=True)
    return grid[["cell_id", "geometry"]].sort_values("cell_id").reset_index(drop=True)


def load_and_validate_schema(vector_path: str) -> gpd.GeoDataFrame:
    """Inspect the actual national source before any filtering.

    Args:
        vector_path: Validated archived shapefile path.

    Returns:
        National MTBS records with their original schema and CRS.

    Raises:
        ValueError: If required fields or the declared CRS are absent.
    """
    mtbs = gpd.read_file(vector_path)
    print(f"National MTBS rows: {len(mtbs):,}; CRS: {mtbs.crs}")
    print(mtbs.dtypes.to_string())
    missing = REQUIRED_FIELDS.difference(mtbs.columns)
    if missing or mtbs.crs is None:
        raise ValueError(f"Missing MTBS fields: {missing}; CRS: {mtbs.crs}")
    print(f"Actual fire types: {mtbs.incid_type.value_counts(dropna=False).to_dict()}")
    return mtbs


def select_period_wildfires(mtbs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Select exact Wildfire records using parsed ignition years, inclusively.

    Args:
        mtbs: National records with validated actual fields.

    Returns:
        National 2017-2022 Wildfire records in the source CRS.

    Raises:
        ValueError: If any ignition date cannot be parsed, matching script 11.
    """
    dates = pd.to_datetime(mtbs.ig_date, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Unparseable ignition dates: {dates.isna().sum():,}")
    fire_year = dates.dt.year
    mask = mtbs.incid_type.eq("Wildfire") & fire_year.between(START_YEAR, END_YEAR)
    selected = mtbs.loc[mask].copy()
    selected["fire_year"] = fire_year.loc[mask]
    print(f"National period Wildfires: {len(selected):,}", flush=True)
    return selected


def reproject_and_select_relevant_fires(
    period_fires: gpd.GeoDataFrame, grid: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Reproject and select fires by intersection with retained complete cells.

    No state-code or event-ID-prefix filter is used: cross-border fires can
    overlap cells retained by the centroid-based New Mexico grid design.

    Args:
        period_fires: National Wildfires in the specified ignition period.
        grid: Validated complete EPSG:5070 analysis cells.

    Returns:
        Validated grid-relevant perimeters in EPSG:5070.

    Raises:
        ValueError: If geometry or selected event identifiers require review.
    """
    # Check before spatial filtering so missing geometry cannot silently vanish.
    # National topology issues outside the study footprint do not affect this
    # target. Report them, then enforce validity on the selected perimeters.
    validate_polygon_geometry(
        period_fires, "National period Wildfires, source CRS", require_valid=False
    )
    projected = period_fires.to_crs(TARGET_CRS)
    validate_polygon_geometry(
        projected, "National period Wildfires, EPSG:5070", require_valid=False
    )
    footprint = union_all(grid.geometry.to_numpy())
    selected = projected.loc[projected.geometry.intersects(footprint)].copy()
    validate_polygon_geometry(selected, "Selected Wildfires, EPSG:5070")
    duplicate_ids = int(selected.event_id.duplicated().sum())
    missing_ids = int(selected.event_id.isna().sum())
    print(f"Selected event IDs: duplicates={duplicate_ids}, missing={missing_ids}")
    if duplicate_ids or missing_ids or selected.empty:
        raise ValueError("Selected event identifiers/count require investigation.")
    print(f"Selected fires: {len(selected)}; QA expectation: {EXPECTED_FIRE_COUNT}")
    if len(selected) != EXPECTED_FIRE_COUNT:
        print("WARNING: selected-fire count differs from the validated prototype; "
              "the source may have changed. No count was forced.")
    print("Fire counts by ignition year:")
    print(selected.fire_year.value_counts().reindex(
        range(START_YEAR, END_YEAR + 1), fill_value=0
    ).sort_index().to_string(), flush=True)
    return selected.sort_values("event_id").reset_index(drop=True)


def union_wildfire_footprints(selected: gpd.GeoDataFrame) -> BaseGeometry:
    """Union fires so repeated burning represents unique geography burned once.

    Args:
        selected: Validated selected fire polygons in EPSG:5070.

    Returns:
        Unique burned geography, including portions beyond the analysis grid.

    Raises:
        ValueError: If the union is empty, invalid, or nonpolygonal.
    """
    wildfire_union = union_all(selected.geometry.to_numpy())
    if (wildfire_union.is_empty or not wildfire_union.is_valid
            or wildfire_union.geom_type not in {"Polygon", "MultiPolygon"}):
        raise ValueError("Unexpected wildfire union geometry.")
    print(f"Wildfire union type: {wildfire_union.geom_type}")
    print(f"Unique unioned area, including outside grid: {wildfire_union.area:.6f} m2",
          flush=True)
    return wildfire_union


def calculate_burned_fractions(
    grid: gpd.GeoDataFrame, wildfire_union: BaseGeometry
) -> pd.DataFrame:
    """Intersect every complete cell with the union using vectorized geometry.

    Args:
        grid: Validated EPSG:5070 cells in deterministic ID order.
        wildfire_union: Unique selected wildfire geography in EPSG:5070.

    Returns:
        Geometry-free continuous target-preparation table, without binary labels.

    Raises:
        ValueError: If raw areas/fractions exceed physical bounds beyond noise.
    """
    print("Intersecting all statewide cells with the wildfire union...", flush=True)
    # Union first, then intersect: summing individual fire overlaps would count
    # repeat burning twice. The full cell area remains the denominator.
    burned_area = grid.geometry.intersection(wildfire_union).area
    cell_area = grid.geometry.area
    fraction = burned_area / cell_area
    if (not np.isfinite(burned_area).all() or not np.isfinite(fraction).all()
            or (burned_area < -AREA_TOLERANCE_M2).any()
            or (burned_area > cell_area + AREA_TOLERANCE_M2).any()
            or (fraction < -FRACTION_TOLERANCE).any()
            or (fraction > 1 + FRACTION_TOLERANCE).any()):
        raise ValueError("Raw intersection areas/fractions violate physical bounds.")
    noise_count = int(((fraction < 0) | (fraction > 1)).sum())
    print(f"Cells requiring floating-point boundary clipping: {noise_count}")
    # Only validated floating-point overshoot is clipped; retain tiny positive
    # overlaps because these are continuous information, not threshold labels.
    burned_area = burned_area.clip(lower=0, upper=cell_area)
    return pd.DataFrame({
        "cell_id": grid.cell_id,
        "burned_area_m2": burned_area,
        "burned_fraction": burned_area / cell_area,
    }).sort_values("cell_id").reset_index(drop=True)


def validate_target_table(table: pd.DataFrame, grid: gpd.GeoDataFrame) -> dict:
    """Validate exact output schema, authoritative keys, and physical values.

    Args:
        table: In-memory or read-back target-preparation table.
        grid: Validated authoritative cells, used to verify keys and denominators.

    Returns:
        Final row, key, and value QA counts and extrema.

    Raises:
        ValueError: If schema, keys, ordering, or numeric values are unexpected.
    """
    if table.columns.tolist() != OUTPUT_COLUMNS:
        raise ValueError(f"Unexpected output columns: {table.columns.tolist()}")
    if len(table) != EXPECTED_CELL_COUNT or not table.cell_id.equals(grid.cell_id):
        raise ValueError("Output rows/ordered IDs do not match the authoritative grid.")
    qa = {
        "rows": len(table), "unique_cell_id": table.cell_id.nunique(),
        "duplicate_cell_id": int(table.cell_id.duplicated().sum()),
        "missing_cell_id": int(table.cell_id.isna().sum()),
    }
    for column in OUTPUT_COLUMNS[1:]:
        values = table[column]
        qa.update({
            f"missing_{column}": int(values.isna().sum()),
            f"nonfinite_{column}": int((~np.isfinite(values)).sum()),
            f"negative_{column}": int((values < 0).sum()),
            f"minimum_{column}": float(values.min()),
            f"maximum_{column}": float(values.max()),
        })
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {column}: {qa}")
    qa["fractions_above_one"] = int((table.burned_fraction > 1).sum())
    if qa["fractions_above_one"] or (
        table.burned_area_m2 > grid.geometry.area
    ).any():
        raise ValueError(f"Values exceed cell support: {qa}")
    if not np.allclose(table.burned_fraction,
                       table.burned_area_m2 / grid.geometry.area,
                       rtol=0, atol=FRACTION_TOLERANCE):
        raise ValueError("Area and fraction are inconsistent.")
    qa["zero_overlap"] = int(table.burned_fraction.eq(0).sum())
    qa["nonzero_overlap"] = int((table.burned_fraction > 0).sum())
    return qa


def write_and_verify_parquet(
    table: pd.DataFrame, grid: gpd.GeoDataFrame, destination: Path
) -> int:
    """Publish the final filename only after temporary Parquet validation.

    Args:
        table: Validated continuous output table.
        grid: Authoritative cells for independent read-back validation.
        destination: Final Parquet path.

    Returns:
        Final file size in bytes.

    Raises:
        ValueError: If read-back validation or exact equality fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    try:
        table.to_parquet(partial_path, index=False)
        read_back = pd.read_parquet(partial_path)
        validate_target_table(read_back, grid)
        if not read_back.equals(table):
            raise ValueError("Read-back table differs from the in-memory output.")
        partial_path.replace(destination)
    finally:
        if partial_path.exists():
            partial_path.unlink()
    return destination.stat().st_size


def report_distribution(table: pd.DataFrame, qa: dict) -> None:
    """Report continuous overlap diagnostics without creating target labels.

    Args:
        table: Validated statewide burned fractions.
        qa: Validated final table diagnostics.

    Raises:
        ValueError: If mutually exclusive threshold bins do not cover all rows.
    """
    fraction = table.burned_fraction
    bins = {"exactly 0": int(fraction.eq(0).sum()),
            ">0 to <0.10": int(((fraction > 0) & (fraction < 0.10)).sum())}
    for lower, upper in [(0.10, 0.25), (0.25, 0.50), (0.50, 0.75)]:
        bins[f"{lower:.2f} to <{upper:.2f}"] = int(
            ((fraction >= lower) & (fraction < upper)).sum()
        )
    bins["0.75 to 1.00"] = int(fraction.between(0.75, 1).sum())
    if sum(bins.values()) != len(table):
        raise ValueError("Threshold bins do not cover the statewide table.")
    print("\nFINAL STATEWIDE QA/QC")
    for name, value in qa.items():
        print(f"{name}: {value}")
    print("\nTHRESHOLD BINS (diagnostics only)")
    for label, count in bins.items():
        print(f"{label}: {count:,}")
    for threshold in [0.10, 0.25, 0.50]:
        print(f"Hypothetical >= {threshold:.0%}: {(fraction >= threshold).sum():,}")
    print("\nNONZERO BURNED FRACTION SUMMARY")
    nonzero = fraction.loc[fraction > 0]
    print(nonzero.describe(percentiles=[0.25, 0.50, 0.75]).to_string())
    print(f"Unique burned area within retained cells: {table.burned_area_m2.sum():.6f} m2")


# %% Main workflow
def main() -> None:
    """Build, validate, and safely write continuous statewide MTBS overlap."""
    started_at = perf_counter()
    # STEP 1 - Acquire and validate authoritative MTBS source
    print(f"Source: {MTBS_URL}\nArchive: {ARCHIVE_PATH}", flush=True)
    download_source_if_missing(MTBS_URL, ARCHIVE_PATH)
    vector_path = validate_archive_and_get_vector_path(ARCHIVE_PATH)

    # STEP 2 - Load and validate authoritative 1-km analysis grid
    grid = load_analysis_grid(GRID_PATH)

    # STEP 3 - Inspect/validate actual MTBS schema
    mtbs = load_and_validate_schema(vector_path)

    # STEP 4 - Select 2017-2022 Wildfire records; spatial relevance follows below
    period_fires = select_period_wildfires(mtbs)

    # STEP 5 - Validate/reproject geometries and select grid-relevant fires
    selected = reproject_and_select_relevant_fires(period_fires, grid)

    # STEP 6 - Union selected wildfire footprints into unique burned geography
    wildfire_union = union_wildfire_footprints(selected)

    # STEP 7 - Calculate statewide burned_area_m2 and burned_fraction
    table = calculate_burned_fractions(grid, wildfire_union)

    # STEP 8 - Validate statewide target-preparation table
    qa = validate_target_table(table, grid)

    # STEP 9 - Write and read back final Parquet before assigning final filename
    file_size = write_and_verify_parquet(table, grid, OUTPUT_PATH)

    # STEP 10 - Report statewide threshold distribution and QA/QC
    report_distribution(table, qa)
    memory = psutil.Process().memory_info()
    peak_bytes = getattr(memory, "peak_wset", None)
    print(f"Final Parquet: {OUTPUT_PATH}")
    print(f"Final file size: {file_size:,} bytes ({file_size / 1024**2:.3f} MiB)")
    print(f"Execution time: {perf_counter() - started_at:.2f} seconds")
    if peak_bytes is not None:
        print(f"Peak process working set: {peak_bytes / 1024**3:.3f} GiB")
    else:
        print(f"Peak memory unavailable; current RSS: {memory.rss / 1024**3:.3f} GiB")


# %% Run script
if __name__ == "__main__":
    main()


