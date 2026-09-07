"""Prototype an MTBS wildfire-perimeter target for New Mexico.

Monitoring Trends in Burn Severity (MTBS) maps large-fire burned-area
boundaries using satellite imagery. Its national polygon product includes fire
attributes; this project retains only Wildfire records from 2017-2022 and will
eventually convert their unique overlap with each 1-km cell to burned fractions.
"""

# %% Imports
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import union_all


# %% Constants and paths
TARGET_CRS = "EPSG:5070"
TARGET_EPSG = 5070
ANALYSIS_CELL_SIZE_M = 1_000
EXPECTED_CELL_AREA_M2 = 1_000_000
START_YEAR = 2017
END_YEAR = 2022
PROTOTYPE_SIDE_CELLS = 12
PROVISIONAL_POSITIVE_THRESHOLD = 0.25
AREA_TOLERANCE_M2 = 1e-3
FRACTION_TOLERANCE = 1e-9

MTBS_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
    "composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
)
MTBS_ARCHIVE_NAME = "mtbs_perimeter_data.zip"
EXPECTED_SHAPEFILE_STEM = "mtbs_perims_DD"
EXPECTED_COMPONENT_SUFFIXES = {".shp", ".shx", ".dbf", ".prj", ".cpg"}

repo_root = Path(__file__).resolve().parents[1]
grid_path = repo_root / "data" / "processed" / "grids" / "nm_analysis_grid_1km.gpkg"
mtbs_archive_path = repo_root / "data" / "raw" / "mtbs" / MTBS_ARCHIVE_NAME
prototype_output_path = (
    repo_root / "data" / "interim" / "mtbs_wildfire_target_prototype.parquet"
)
plot_output_path = repo_root / "outputs" / "mtbs_wildfire_target_prototype.png"


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


def load_and_inspect_mtbs(vector_path: str) -> gpd.GeoDataFrame:
    """Load the actual MTBS dataset and report its observed schema.

    Filtering field names are not assumed here. This inspection deliberately
    precedes schema mapping so changes to the rolling national product remain
    visible and cause explicit validation failures.

    Args:
        vector_path: Archived national MTBS shapefile path.

    Returns:
        National MTBS GeoDataFrame with polygon-compatible geometry.

    Raises:
        ValueError: If CRS is absent or geometry types are incompatible.
    """
    mtbs = gpd.read_file(vector_path)
    geometry_types = mtbs.geometry.geom_type.value_counts(dropna=False)
    incompatible_types = set(geometry_types.index).difference(
        {"Polygon", "MultiPolygon"}
    )
    if mtbs.crs is None:
        raise ValueError("MTBS perimeter data does not declare a CRS.")
    if incompatible_types:
        raise ValueError(f"MTBS has incompatible geometry types: {incompatible_types}")

    print("ACTUAL NATIONAL MTBS SCHEMA")
    print("---------------------------")
    print(f"Rows: {len(mtbs):,}")
    print(f"Columns: {mtbs.columns.tolist()}")
    print("Dtypes:")
    print(mtbs.dtypes.to_string())
    print(f"CRS: {mtbs.crs}")
    print(f"Geometry types: {geometry_types.to_dict()}")
    print("\nFirst five non-geometry records:")
    print(mtbs.drop(columns="geometry").head().to_string(index=False))
    return mtbs


def identify_actual_schema(mtbs: gpd.GeoDataFrame) -> dict[str, str]:
    """Validate and document the field mapping observed in the current file.

    The rolling MTBS shapefile currently supplies ignition date rather than a
    separate year field, so fire year is derived from parsed ignition dates.

    Args:
        mtbs: Inspected national MTBS dataset.

    Returns:
        Semantic-to-physical field-name mapping.

    Raises:
        ValueError: If any required observed field is absent.
    """
    field_mapping = {
        "fire_id": "event_id",
        "fire_name": "incid_name",
        "fire_type": "incid_type",
        "ignition_date": "ig_date",
        "mapped_acres": "burnbndac",
    }
    missing_fields = set(field_mapping.values()).difference(mtbs.columns)
    if missing_fields:
        raise ValueError(f"Required actual MTBS fields are missing: {missing_fields}")

    ignition_dates = pd.to_datetime(
        mtbs[field_mapping["ignition_date"]],
        errors="coerce",
    )
    print("\nOBSERVED FIELD MAPPING")
    print("----------------------")
    for meaning, field_name in field_mapping.items():
        print(f"{meaning}: {field_name}")
    print(
        "Unique fire types: "
        f"{sorted(mtbs[field_mapping['fire_type']].dropna().unique().tolist())}"
    )
    print(
        f"Year range: {ignition_dates.dt.year.min()}-"
        f"{ignition_dates.dt.year.max()}"
    )
    return field_mapping


def load_analysis_grid(path: Path) -> gpd.GeoDataFrame:
    """Load and validate the authoritative New Mexico 1-km grid.

    Args:
        path: Prepared analysis-grid GeoPackage.

    Returns:
        GeoDataFrame containing authoritative cell IDs and complete polygons.

    Raises:
        FileNotFoundError: If the prepared grid is missing.
        ValueError: If schema, CRS, IDs, or 1-km cell areas are unexpected.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required analysis grid not found: {path}")
    grid = gpd.read_file(path, layer="analysis_grid")
    if not {"cell_id", "geometry"}.issubset(grid.columns):
        raise ValueError("Analysis grid requires cell_id and geometry.")
    if grid.crs is None or grid.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"Analysis grid must use {TARGET_CRS}.")
    if not grid["cell_id"].is_unique:
        raise ValueError("Authoritative cell_id values must be unique.")

    # EPSG:5070 uses meters, so polygon area is directly measured in square meters.
    cell_areas = grid.geometry.area
    if not np.allclose(cell_areas, EXPECTED_CELL_AREA_M2, atol=AREA_TOLERANCE_M2):
        raise ValueError("Analysis geometries are not complete 1-km square cells.")
    return grid[["cell_id", "geometry"]].copy()


def filter_selected_wildfires(
    mtbs: gpd.GeoDataFrame,
    fields: dict[str, str],
    analysis_grid: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Select 2017-2022 wildfires that spatially intersect New Mexico.

    Only the actual Wildfire value is eligible because prescribed and other
    fire types represent different processes from the historical wildfire
    target. Spatial intersection with the authoritative analysis area retains
    cross-border fires for the portion that can overlap retained cells.

    Args:
        mtbs: National MTBS perimeter dataset.
        fields: Validated actual field mapping.
        analysis_grid: Authoritative New Mexico retained cells.

    Returns:
        Selected perimeters in EPSG:5070 and record-count diagnostics.

    Raises:
        ValueError: If ignition dates cannot be parsed.
    """
    ignition_dates = pd.to_datetime(mtbs[fields["ignition_date"]], errors="coerce")
    if ignition_dates.isna().any():
        raise ValueError(
            f"MTBS has {ignition_dates.isna().sum():,} unparseable ignition dates."
        )
    fire_year = ignition_dates.dt.year
    wildfire_mask = mtbs[fields["fire_type"]].eq("Wildfire")
    period_mask = fire_year.between(START_YEAR, END_YEAR)

    period_wildfires = mtbs.loc[wildfire_mask & period_mask].copy()
    period_wildfires["fire_year"] = fire_year.loc[period_wildfires.index]

    # Reprojection occurs before intersection and all area calculations. EPSG:5070
    # provides a common equal-area, meter-based coordinate system for both layers.
    period_wildfires = period_wildfires.to_crs(TARGET_CRS)
    analysis_boundary = analysis_grid.geometry.union_all()
    selected = period_wildfires.loc[
        period_wildfires.geometry.intersects(analysis_boundary)
    ].copy()

    counts = {
        "national": len(mtbs),
        "national_wildfire": int(wildfire_mask.sum()),
        "national_period_all_types": int(period_mask.sum()),
        "national_period_wildfire": len(period_wildfires),
        "selected_nm_wildfire": len(selected),
    }
    return selected, counts


def validate_selected_geometries(
    selected: gpd.GeoDataFrame,
    fire_id_field: str,
) -> dict[str, int | float]:
    """Validate selected fire geometries without silently repairing them.

    Args:
        selected: New Mexico-intersecting wildfire perimeters in EPSG:5070.
        fire_id_field: Actual unique fire-identifier field.

    Returns:
        Geometry issue counts and total mapped polygon area.

    Raises:
        ValueError: If missing, empty, or invalid geometry is found.
    """
    qa = {
        "missing": int(selected.geometry.isna().sum()),
        "empty": int(selected.geometry.is_empty.sum()),
        "invalid": int((~selected.geometry.is_valid).sum()),
        "duplicate_fire_ids": int(selected[fire_id_field].duplicated().sum()),
        "total_area_m2": float(selected.geometry.area.sum()),
    }
    if qa["missing"] or qa["empty"] or qa["invalid"]:
        raise ValueError(
            "Selected MTBS geometries require investigation before overlap: "
            f"{qa}"
        )
    return qa


def select_prototype_cells(
    analysis_grid: gpd.GeoDataFrame,
    selected_fires: gpd.GeoDataFrame,
    fields: dict[str, str],
) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """Select a deterministic 12-by-12 lattice around a representative fire.

    The fire nearest the selected subset's median mapped acreage is used rather
    than an extreme event. Ties resolve by fire ID. The surrounding 144-cell
    square is large enough to include burned interior, perimeter edge, and
    nearby unburned cells while remaining a small prototype.

    Args:
        analysis_grid: Authoritative statewide 1-km grid.
        selected_fires: Validated 2017-2022 New Mexico wildfires.
        fields: Validated actual MTBS field mapping.

    Returns:
        Spatially coherent prototype cells and the selected anchor-fire record.

    Raises:
        ValueError: If a complete 12-by-12 retained-cell sample is unavailable.
    """
    acreage_field = fields["mapped_acres"]
    median_acres = selected_fires[acreage_field].median()
    candidates = selected_fires.assign(
        acreage_distance=(selected_fires[acreage_field] - median_acres).abs()
    )
    anchor_fire = candidates.sort_values(
        ["acreage_distance", fields["fire_id"]]
    ).iloc[0]
    anchor_centroid = anchor_fire.geometry.centroid

    grid_centroids = analysis_grid.geometry.centroid
    relative_columns = np.rint(
        (grid_centroids.x - anchor_centroid.x) / ANALYSIS_CELL_SIZE_M
    )
    relative_rows = np.rint(
        (grid_centroids.y - anchor_centroid.y) / ANALYSIS_CELL_SIZE_M
    )
    lower_offset = -(PROTOTYPE_SIDE_CELLS // 2 - 1)
    upper_offset = PROTOTYPE_SIDE_CELLS // 2
    sample_mask = relative_columns.between(lower_offset, upper_offset) & (
        relative_rows.between(lower_offset, upper_offset)
    )
    prototype_grid = analysis_grid.loc[sample_mask].copy()
    expected_count = PROTOTYPE_SIDE_CELLS**2
    if len(prototype_grid) != expected_count:
        raise ValueError(
            f"Expected {expected_count} prototype cells; found {len(prototype_grid)}."
        )
    return prototype_grid, anchor_fire


def calculate_burned_fractions(
    prototype_grid: gpd.GeoDataFrame,
    selected_fires: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, object]:
    """Calculate unique wildfire overlap within each prototype cell.

    Unioning all selected fire polygons first ensures overlapping or repeated
    footprints are counted only once. Polygon intersection, rather than any-
    overlap or centroid membership, retains the amount of burned area and makes
    edge-cell ambiguity visible.

    Args:
        prototype_grid: Authoritative 1-km prototype cells.
        selected_fires: Selected wildfire perimeters in EPSG:5070.

    Returns:
        Prototype cells with unique burned area and fraction, plus the unioned
        selected-wildfire geometry used in the calculation.

    Raises:
        ValueError: If fractions fall outside zero to one beyond tolerance.
    """
    wildfire_union = union_all(selected_fires.geometry.to_numpy())
    prototype = prototype_grid.copy()

    # Vectorized intersection returns the unique portion of the union within
    # each cell. EPSG:5070 areas are square meters.
    overlap_geometry = prototype.geometry.intersection(wildfire_union)
    prototype["burned_area_m2"] = overlap_geometry.area
    cell_area_m2 = prototype.geometry.area
    prototype["burned_fraction"] = prototype["burned_area_m2"] / cell_area_m2

    fraction = prototype["burned_fraction"]
    outside_domain = (fraction < -FRACTION_TOLERANCE) | (
        fraction > 1 + FRACTION_TOLERANCE
    )
    if outside_domain.any():
        raise ValueError("Burned fractions fall outside the expected [0, 1] domain.")
    # Clip only floating-point noise at mathematical bounds, not substantive data.
    prototype["burned_fraction"] = fraction.clip(0.0, 1.0)
    return prototype, wildfire_union


def summarize_threshold_distribution(
    prototype: gpd.GeoDataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Summarize fractions without permanently assigning an ML target.

    The 25 percent split is shown only as a provisional diagnostic. More
    detailed bins expose the actual edge-cell distribution for later review.

    Args:
        prototype: Prototype cells with burned_fraction.

    Returns:
        Detailed bin counts, provisional-category counts, and nonzero summary.
    """
    fraction = prototype["burned_fraction"]
    bins = {
        "exactly 0": int(fraction.eq(0).sum()),
        ">0 to <0.10": int(((fraction > 0) & (fraction < 0.10)).sum()),
        "0.10 to <0.25": int(
            ((fraction >= 0.10) & (fraction < 0.25)).sum()
        ),
        "0.25 to <0.50": int(
            ((fraction >= 0.25) & (fraction < 0.50)).sum()
        ),
        "0.50 to <0.75": int(
            ((fraction >= 0.50) & (fraction < 0.75)).sum()
        ),
        "0.75 to 1.00": int(
            ((fraction >= 0.75) & (fraction <= 1.00)).sum()
        ),
    }
    categories = {
        "clear_negative": int(fraction.eq(0).sum()),
        "ambiguous_edge_qa_only": int(
            ((fraction > 0) & (fraction < PROVISIONAL_POSITIVE_THRESHOLD)).sum()
        ),
        "positive_qa_only": int(
            (fraction >= PROVISIONAL_POSITIVE_THRESHOLD).sum()
        ),
    }
    nonzero = fraction.loc[fraction > 0]
    summary = {
        "minimum_nonzero": float(nonzero.min()),
        "median_nonzero": float(nonzero.median()),
        "mean_nonzero": float(nonzero.mean()),
        "maximum": float(fraction.max()),
    }
    return bins, categories, summary


def write_prototype_table(prototype: gpd.GeoDataFrame, path: Path) -> None:
    """Write a small interim QA table, never a statewide target.

    Args:
        prototype: Prototype cells with calculated overlap values.
        path: Interim Parquet destination.

    Raises:
        ValueError: If the read-back table differs from the written data.
    """
    table = (
        prototype[["cell_id", "burned_area_m2", "burned_fraction"]]
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    if not pd.read_parquet(path).equals(table):
        raise ValueError("Interim prototype Parquet failed read-back verification.")


def plot_spatial_qa(
    prototype: gpd.GeoDataFrame,
    selected_fires: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """Plot relevant perimeters and cells colored by burned fraction.

    Args:
        prototype: Prototype grid with burned fractions.
        selected_fires: Selected EPSG:5070 wildfire perimeters.
        path: PNG destination for the lightweight QA figure.
    """
    prototype_extent = prototype.geometry.union_all()
    nearby_fires = selected_fires.loc[
        selected_fires.geometry.intersects(prototype_extent)
    ]

    fig, axis = plt.subplots(figsize=(8, 8))
    prototype.plot(
        column="burned_fraction",
        ax=axis,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        edgecolor="gray",
        linewidth=0.5,
        legend=True,
        legend_kwds={"label": "Unique wildfire burned fraction"},
    )
    nearby_fires.boundary.plot(ax=axis, color="royalblue", linewidth=1.2)
    axis.set_title("MTBS 2017-2022 wildfire target prototype")
    axis.set_xlabel("EPSG:5070 x (m)")
    axis.set_ylabel("EPSG:5070 y (m)")
    axis.set_aspect("equal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_findings(
    mtbs: gpd.GeoDataFrame,
    fields: dict[str, str],
    selected: gpd.GeoDataFrame,
    counts: dict[str, int],
    geometry_qa: dict[str, int | float],
    prototype: gpd.GeoDataFrame,
    anchor_fire: pd.Series,
    bins: dict[str, int],
    categories: dict[str, int],
    summary: dict[str, float],
) -> None:
    """Print selection, geometry, prototype, and threshold QA findings.

    Args:
        mtbs: National MTBS dataset.
        fields: Actual schema mapping.
        selected: Selected New Mexico wildfire records.
        counts: National and filtered record counts.
        geometry_qa: Selected-perimeter geometry diagnostics.
        prototype: Prototype target cells.
        anchor_fire: Representative fire used to center the sample.
        bins: Detailed burned-fraction bin counts.
        categories: Provisional QA-only category counts.
        summary: Nonzero burned-fraction statistics.
    """
    ignition_dates = pd.to_datetime(mtbs[fields["ignition_date"]])
    selected_year_counts = selected["fire_year"].value_counts().sort_index()
    selected_columns = [
        fields["fire_id"],
        fields["fire_name"],
        "fire_year",
        fields["mapped_acres"],
    ]

    print("\nSOURCE AND FILTER QA/QC")
    print("------------------------")
    print(f"National records: {counts['national']:,}")
    print(f"National Wildfire records: {counts['national_wildfire']:,}")
    print(
        f"All fire types in {START_YEAR}-{END_YEAR}: "
        f"{counts['national_period_all_types']:,}"
    )
    print(
        f"National Wildfire records in period: "
        f"{counts['national_period_wildfire']:,}"
    )
    print(f"Selected New Mexico Wildfires: {counts['selected_nm_wildfire']:,}")
    print(
        f"Actual ignition-date range: {ignition_dates.min().date()} to "
        f"{ignition_dates.max().date()}"
    )
    print(f"Selected counts by year: {selected_year_counts.to_dict()}")
    print("\nSelected fire IDs, names, years, and mapped acres:")
    print(
        selected[selected_columns]
        .sort_values(["fire_year", fields["fire_id"]])
        .to_string(index=False)
    )

    print("\nSELECTED FIRE GEOMETRY QA/QC")
    print("-----------------------------")
    print(f"Missing geometries: {geometry_qa['missing']:,}")
    print(f"Empty geometries: {geometry_qa['empty']:,}")
    print(f"Invalid geometries: {geometry_qa['invalid']:,}")
    print(f"Duplicate fire IDs: {geometry_qa['duplicate_fire_ids']:,}")
    print(f"Geometries repaired: 0")
    print(f"Total selected mapped area: {geometry_qa['total_area_m2']:,.2f} m²")

    print("\nPROTOTYPE AND THRESHOLD QA")
    print("--------------------------")
    print(
        "Selection: 12 x 12 cells centered on the fire nearest median "
        "selected mapped acreage"
    )
    print(
        f"Anchor fire: {anchor_fire[fields['fire_id']]} / "
        f"{anchor_fire[fields['fire_name']]} / "
        f"{anchor_fire['fire_year']} / "
        f"{anchor_fire[fields['mapped_acres']]:,} acres"
    )
    print(f"Prototype cells: {len(prototype):,}")
    print(f"Burned-fraction bins: {bins}")
    print(f"Nonzero fraction summary: {summary}")
    print(
        "Provisional QA-only 0 / ambiguous / >=25% counts: "
        f"{categories}"
    )
    print("The 25% threshold remains tentative and is not a final target rule.")
    print(f"Interim prototype table: {prototype_output_path}")
    print(f"Spatial QA plot: {plot_output_path}")


# %% Main workflow
def main() -> None:
    """Run the small MTBS wildfire-target prototype and its QA/QC."""
    # STEP 1 — Acquire and validate MTBS source
    download_source_if_missing(MTBS_URL, mtbs_archive_path)
    vector_path = validate_archive_and_get_vector_path(mtbs_archive_path)
    analysis_grid = load_analysis_grid(grid_path)

    # STEP 2 — Inspect actual MTBS schema and fire-type values
    mtbs = load_and_inspect_mtbs(vector_path)
    fields = identify_actual_schema(mtbs)

    # STEP 3 — Filter to New Mexico and 2017-2022 Wildfire records
    selected_fires, counts = filter_selected_wildfires(mtbs, fields, analysis_grid)

    # STEP 4 — QA/QC selected fire perimeters
    geometry_qa = validate_selected_geometries(
        selected_fires,
        fields["fire_id"],
    )

    # STEP 5 — Select a small deterministic set of authoritative 1-km cells
    prototype_grid, anchor_fire = select_prototype_cells(
        analysis_grid,
        selected_fires,
        fields,
    )

    # STEP 6 — Calculate unique wildfire overlap and burned_fraction
    prototype, _ = calculate_burned_fractions(prototype_grid, selected_fires)
    write_prototype_table(prototype, prototype_output_path)

    # STEP 7 — Inspect the tentative threshold distribution
    bins, categories, summary = summarize_threshold_distribution(prototype)

    # STEP 8 — Plot spatial QA
    plot_spatial_qa(prototype, selected_fires, plot_output_path)

    # STEP 9 — Report findings
    report_findings(
        mtbs,
        fields,
        selected_fires,
        counts,
        geometry_qa,
        prototype,
        anchor_fire,
        bins,
        categories,
        summary,
    )


# %% Run script
if __name__ == "__main__":
    main()
