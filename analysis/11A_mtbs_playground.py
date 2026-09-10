"""Playground for exploring MTBS wildfire geometry and burned fractions.

This script reuses ideas from analysis/11_validate_mtbs_wildfire_target.py
but is intentionally designed for interactive learning.

Use the # %% cells in VS Code to run one section at a time and inspect:

- the national MTBS table
- selected 2017-2022 New Mexico wildfires
- individual wildfire polygons
- the unioned wildfire geometry
- prototype grid cells
- polygon intersections
- burned_area_m2
- burned_fraction
- Pandas boolean masks such as .eq(), >, <, and .sum()
"""

# %% Imports
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import union_all


# %% Constants and paths
TARGET_CRS = "EPSG:5070"
START_YEAR = 2017
END_YEAR = 2022
ANALYSIS_CELL_SIZE_M = 1_000

repo_root = Path(__file__).resolve().parents[1]

grid_path = (
    repo_root
    / "data"
    / "processed"
    / "grids"
    / "nm_analysis_grid_1km.gpkg"
)

mtbs_archive_path = (
    repo_root
    / "data"
    / "raw"
    / "mtbs"
    / "mtbs_perimeter_data.zip"
)

playground_plot_dir = repo_root / "outputs" / "mtbs_playground"
playground_plot_dir.mkdir(parents=True, exist_ok=True)


# %% 1 — Inspect the MTBS ZIP contents
with ZipFile(mtbs_archive_path) as archive:
    member_names = archive.namelist()

member_names[:20]


# %% 2 — Build the archived shapefile path
shapefile_name = [
    name
    for name in member_names
    if name.lower().endswith(".shp")
][0]
# %% 
vector_path = (
    f"zip://{mtbs_archive_path.as_posix()}!"
    f"{shapefile_name}"
)

vector_path


# %% 3 — Load the national MTBS dataset
mtbs = gpd.read_file(vector_path)

mtbs.head()


# %% 4 — Inspect useful columns
mtbs[
    [
        "event_id",
        "incid_name",
        "incid_type",
        "ig_date",
        "burnbndac",
        "geometry",
    ]
].head(20)


# %% 5 — Inspect basic MTBS structure
print("Rows:", len(mtbs))
print("CRS:", mtbs.crs)

print("\nColumns:")
print(mtbs.columns.tolist())

print("\nGeometry types:")
print(mtbs.geometry.geom_type.value_counts())

# %% 
mtbs.dtypes
# %%
mtbs.info()
# %%
mtbs.describe(include="all")
# %% 6 — Inspect fire types
mtbs["incid_type"].value_counts()


# %% 7 — Parse ignition dates and derive fire year
mtbs["ig_date_parsed"] = pd.to_datetime(
    mtbs["ig_date"],
    errors="coerce",
)

mtbs["fire_year"] = mtbs["ig_date_parsed"].dt.year

mtbs[
    [
        "incid_name",
        "incid_type",
        "ig_date",
        "fire_year",
    ]
].head(20)


# %% 8 — Filter to Wildfire and 2017-2022
wildfire_mask = mtbs["incid_type"].eq("Wildfire")

period_mask = mtbs["fire_year"].between(
    START_YEAR,
    END_YEAR,
)

period_wildfires = mtbs.loc[
    wildfire_mask & period_mask
].copy()

print("National MTBS records:", len(mtbs))
print("National Wildfire records:", wildfire_mask.sum())
print(
    "National 2017-2022 Wildfires:",
    len(period_wildfires),
)


# %% 9 — Load authoritative New Mexico analysis grid
analysis_grid = gpd.read_file(
    grid_path,
    layer="analysis_grid",
)

print("Grid rows:", len(analysis_grid))
print("Grid CRS:", analysis_grid.crs)

analysis_grid.head()


# %% 10 — Reproject filtered MTBS fires to EPSG:5070
period_wildfires_5070 = period_wildfires.to_crs(
    TARGET_CRS
)

print("Fire CRS:", period_wildfires_5070.crs)


# %% 11 — Build the analysis-grid footprint
analysis_footprint = analysis_grid.geometry.union_all()

print("Geometry type:", analysis_footprint.geom_type)
print(
    "Approximate grid-footprint area km²:",
    analysis_footprint.area / 1_000_000,
)


# %% 12 — Plot the New Mexico analysis-grid footprint
footprint_gs = gpd.GeoSeries(
    [analysis_footprint],
    crs=TARGET_CRS,
)

fig, ax = plt.subplots(figsize=(8, 8))

footprint_gs.boundary.plot(
    ax=ax,
    linewidth=1.0,
)

ax.set_title(
    "Authoritative 1-km Analysis Grid Footprint"
)
ax.set_aspect("equal")

plt.show()


# %% 13 — Select 2017-2022 wildfires that intersect the analysis grid
nm_fire_mask = (
    period_wildfires_5070.geometry
    .intersects(analysis_footprint)
)

selected_fires = (
    period_wildfires_5070.loc[nm_fire_mask]
    .copy()
)

print(
    "Selected NM-relevant wildfires:",
    len(selected_fires),
)

selected_fires[
    [
        "event_id",
        "incid_name",
        "fire_year",
        "burnbndac",
    ]
].head(20)


# %% 14 — Inspect selected fire counts by year
selected_fires["fire_year"].value_counts().sort_index()


# %% 15 — Plot the 129 individual selected wildfire polygons
fig, ax = plt.subplots(figsize=(8, 8))

selected_fires.plot(
    ax=ax,
    edgecolor="black",
)

ax.set_title(
    "Selected MTBS Wildfires, 2017-2022"
)
ax.set_aspect("equal")

plt.show()


# %% 16 — Save selected-fire plot
selected_fires_plot_path = (
    playground_plot_dir
    / "selected_mtbs_wildfires_2017_2022.png"
)

fig, ax = plt.subplots(figsize=(8, 8))

selected_fires.plot(
    ax=ax,
    edgecolor="black",
)

ax.set_title(
    "Selected MTBS Wildfires, 2017-2022"
)
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(
    selected_fires_plot_path,
    dpi=150,
)

plt.close(fig)

selected_fires_plot_path


# %% 17 — Explore .to_numpy()
geometry_array = (
    selected_fires.geometry.to_numpy()
)

print(type(selected_fires.geometry))
print(type(geometry_array))
print(type(geometry_array[0]))

geometry_array[:3]


# %% 18 — Union all selected wildfire geometries
wildfire_union = union_all(
    selected_fires.geometry.to_numpy()
)

# %% 
# %% Explore wildfire_union
print("Geometry type:", wildfire_union.geom_type)
print("Polygon parts:", len(wildfire_union.geoms))
print("Total area m²:", wildfire_union.area)
print("Bounds:", wildfire_union.bounds)

first_polygon = wildfire_union.geoms[0]

print("\nFirst polygon type:", first_polygon.geom_type)
print("First polygon area m²:", first_polygon.area)

coords = list(first_polygon.exterior.coords)

print("Number of exterior coordinates:", len(coords))
print("First 10 coordinates:")
print(coords[:10])
# %%
print("Union geometry type:", wildfire_union.geom_type)
print(
    "Unique burned area km²:",
    wildfire_union.area / 1_000_000,
)


# %% 19 — Plot the unioned wildfire footprint
wildfire_union_gs = gpd.GeoSeries(
    [wildfire_union],
    crs=TARGET_CRS,
)

fig, ax = plt.subplots(figsize=(8, 8))

wildfire_union_gs.plot(
    ax=ax,
)

ax.set_title(
    "Unioned MTBS Wildfire Footprint, 2017-2022"
)
ax.set_aspect("equal")

plt.show()


# %% 20 — Save unioned wildfire footprint
wildfire_union_plot_path = (
    playground_plot_dir
    / "wildfire_union_2017_2022.png"
)

fig, ax = plt.subplots(figsize=(8, 8))

wildfire_union_gs.plot(
    ax=ax,
)

ax.set_title(
    "Unioned MTBS Wildfire Footprint, 2017-2022"
)
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(
    wildfire_union_plot_path,
    dpi=150,
)

plt.close(fig)

wildfire_union_plot_path


# %% 21 — Pick the same representative anchor-fire idea as script 11
median_acres = selected_fires["burnbndac"].median()

candidates = selected_fires.assign(
    acreage_distance=(
        selected_fires["burnbndac"]
        - median_acres
    ).abs()
)

anchor_fire = candidates.sort_values(
    [
        "acreage_distance",
        "event_id",
    ]
).iloc[0]

anchor_fire[
    [
        "event_id",
        "incid_name",
        "fire_year",
        "burnbndac",
    ]
]


# %% 22 — Find the anchor-fire centroid
anchor_centroid = anchor_fire.geometry.centroid

print(anchor_centroid)


# %% 23 — Select a 12 x 12 grid around the anchor fire
grid_centroids = analysis_grid.geometry.centroid

relative_columns = np.rint(
    (
        grid_centroids.x
        - anchor_centroid.x
    )
    / ANALYSIS_CELL_SIZE_M
)

relative_rows = np.rint(
    (
        grid_centroids.y
        - anchor_centroid.y
    )
    / ANALYSIS_CELL_SIZE_M
)

lower_offset = -5
upper_offset = 6

sample_mask = (
    relative_columns.between(
        lower_offset,
        upper_offset,
    )
    &
    relative_rows.between(
        lower_offset,
        upper_offset,
    )
)

prototype_grid = (
    analysis_grid.loc[sample_mask]
    .copy()
)

print("Prototype cells:", len(prototype_grid))


# %% 24 — Find wildfire polygons near the prototype grid
prototype_extent = (
    prototype_grid.geometry.union_all()
)

nearby_fires = selected_fires.loc[
    selected_fires.geometry.intersects(
        prototype_extent
    )
].copy()

print("Nearby fires:", len(nearby_fires))

nearby_fires[
    [
        "event_id",
        "incid_name",
        "fire_year",
        "burnbndac",
    ]
]


# %% 25 — Plot prototype cells and nearby wildfire polygons
fig, ax = plt.subplots(figsize=(8, 8))

prototype_grid.boundary.plot(
    ax=ax,
    linewidth=0.7,
)

nearby_fires.boundary.plot(
    ax=ax,
    linewidth=1.5,
)

ax.set_title(
    "Prototype Grid and Nearby MTBS Wildfires"
)
ax.set_aspect("equal")

plt.show()


# %% 26 — THE IMPORTANT INTERSECTION
overlap_geometry = (
    prototype_grid.geometry
    .intersection(wildfire_union)
)

overlap_geometry.head()


# %% 27 — Inspect overlap geometry types
overlap_geometry.geom_type.value_counts()


# %% 28 — Inspect overlap geometry areas
burned_area_m2 = overlap_geometry.area

burned_area_m2.head(20)


# %% 29 — Calculate burned fraction
prototype = prototype_grid.copy()

prototype["burned_area_m2"] = (
    overlap_geometry.area
)

prototype["cell_area_m2"] = (
    prototype.geometry.area
)

prototype["burned_fraction"] = (
    prototype["burned_area_m2"]
    / prototype["cell_area_m2"]
)

prototype[
    [
        "cell_id",
        "burned_area_m2",
        "cell_area_m2",
        "burned_fraction",
    ]
].head(20)


# %% 30 — Sort by highest burned fraction
prototype[
    [
        "cell_id",
        "burned_area_m2",
        "burned_fraction",
    ]
].sort_values(
    "burned_fraction",
    ascending=False,
).head(20)


# %% 31 — Plot prototype cells by burned fraction
fig, ax = plt.subplots(figsize=(8, 8))

prototype.plot(
    column="burned_fraction",
    ax=ax,
    legend=True,
    edgecolor="black",
    linewidth=0.5,
)

nearby_fires.boundary.plot(
    ax=ax,
    linewidth=1.5,
)

ax.set_title(
    "Prototype Burned Fraction"
)
ax.set_aspect("equal")

plt.show()


# %% 32 — Save prototype burned-fraction plot
prototype_fraction_plot_path = (
    playground_plot_dir
    / "prototype_burned_fraction.png"
)

fig, ax = plt.subplots(figsize=(8, 8))

prototype.plot(
    column="burned_fraction",
    ax=ax,
    legend=True,
    edgecolor="black",
    linewidth=0.5,
)

nearby_fires.boundary.plot(
    ax=ax,
    linewidth=1.5,
)

ax.set_title(
    "Prototype Burned Fraction"
)
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(
    prototype_fraction_plot_path,
    dpi=150,
)

plt.close(fig)

prototype_fraction_plot_path


# %% 33 — PLAY with fraction.eq(0)
fraction = prototype["burned_fraction"]

fraction.head(20)


# %% 34 — What does .eq(0) produce?
is_zero = fraction.eq(0)

is_zero.head(20)


# %% 35 — Count the True values
zero_count = is_zero.sum()

print(
    "Exactly zero:",
    zero_count,
)


# %% 36 — Compare .eq(0) with == 0
print(
    "Using .eq(0):",
    fraction.eq(0).sum(),
)

print(
    "Using == 0:",
    (fraction == 0).sum(),
)


# %% 37 — Explore greater-than and less-than masks
greater_than_zero = (
    fraction > 0
)

less_than_ten_percent = (
    fraction < 0.10
)

greater_than_zero.head()


# %% 38 — Combine two boolean conditions
tiny_overlap_mask = (
    (fraction > 0)
    &
    (fraction < 0.10)
)

tiny_overlap_mask.head(20)


# %% 39 — Count tiny-overlap cells
tiny_overlap_count = (
    tiny_overlap_mask.sum()
)

print(
    "Cells with >0 and <10% burned:",
    tiny_overlap_count,
)


# %% 40 — Inspect the actual tiny-overlap cells
prototype.loc[
    tiny_overlap_mask,
    [
        "cell_id",
        "burned_area_m2",
        "burned_fraction",
    ],
].sort_values(
    "burned_fraction"
)


# %% 41 — Explore provisional 25% threshold
positive_25_mask = (
    fraction >= 0.25
)

ambiguous_mask = (
    (fraction > 0)
    &
    (fraction < 0.25)
)

print(
    "Zero:",
    fraction.eq(0).sum(),
)

print(
    "Ambiguous 0-25%:",
    ambiguous_mask.sum(),
)

print(
    ">=25%:",
    positive_25_mask.sum(),
)


# %% 42 — Compare several possible thresholds
for threshold in [
    0.10,
    0.25,
    0.50,
]:
    count = (
        fraction >= threshold
    ).sum()

    print(
        f">= {threshold:.0%}: "
        f"{count} cells"
    )


# %% 43 — Simple summary of burned fractions
prototype["burned_fraction"].describe()


# %% 44 — Inspect only cells with some wildfire overlap
nonzero_cells = prototype.loc[
    prototype["burned_fraction"] > 0,
    [
        "cell_id",
        "burned_area_m2",
        "burned_fraction",
        "geometry",
    ],
].copy()

nonzero_cells.sort_values(
    "burned_fraction"
)


# %% 45 — Inspect the smallest nonzero overlap
smallest_overlap_cell = (
    nonzero_cells
    .sort_values("burned_fraction")
    .iloc[0]
)

smallest_overlap_cell[
    [
        "cell_id",
        "burned_area_m2",
        "burned_fraction",
    ]
]


# %% 46 — Plot the smallest-overlap cell and wildfire geometry
smallest_cell_geometry = (
    smallest_overlap_cell.geometry
)

smallest_cell_gs = gpd.GeoSeries(
    [smallest_cell_geometry],
    crs=TARGET_CRS,
)

smallest_overlap_geometry = (
    smallest_cell_geometry
    .intersection(wildfire_union)
)

smallest_overlap_gs = gpd.GeoSeries(
    [smallest_overlap_geometry],
    crs=TARGET_CRS,
)

fig, ax = plt.subplots(figsize=(7, 7))

smallest_cell_gs.boundary.plot(
    ax=ax,
    linewidth=2,
)

smallest_overlap_gs.plot(
    ax=ax,
)

ax.set_title(
    "Smallest Nonzero Wildfire Overlap"
)
ax.set_aspect("equal")

plt.show()


# %% 47 — Inspect the largest burned-fraction cell
largest_overlap_cell = (
    prototype
    .sort_values(
        "burned_fraction",
        ascending=False,
    )
    .iloc[0]
)

largest_overlap_cell[
    [
        "cell_id",
        "burned_area_m2",
        "burned_fraction",
    ]
]


# %% 48 — Final playground summary
print("PLAYGROUND SUMMARY")
print("------------------")

print(
    "National MTBS records:",
    len(mtbs),
)

print(
    "National 2017-2022 Wildfires:",
    len(period_wildfires),
)

print(
    "NM-relevant Wildfires:",
    len(selected_fires),
)

print(
    "Prototype cells:",
    len(prototype),
)

print(
    "Cells with zero overlap:",
    fraction.eq(0).sum(),
)

print(
    "Cells with >0 and <10% overlap:",
    ((fraction > 0) & (fraction < 0.10)).sum(),
)

print(
    "Cells with >0 and <25% overlap:",
    ((fraction > 0) & (fraction < 0.25)).sum(),
)

print(
    "Cells with >=25% overlap:",
    (fraction >= 0.25).sum(),
)

print(
    "Maximum burned fraction:",
    fraction.max(),
)