"""Playground for exploring LANDFIRE EVT categorical raster aggregation.

This script is intentionally interactive and learning-oriented.

It explores the real LANDFIRE 2016 Remap EVT prototype produced by
analysis/13_validate_landfire_evt.py.

The main ideas are:

1. LANDFIRE EVT pixels contain categorical class codes, not quantities.
2. The native LANDFIRE raster has approximately 30-m pixels.
3. The project's analysis units are complete 1-km cells.
4. Because 30 m does not divide evenly into 1,000 m, the categorical raster
   is reprojected to an aligned 25-m grid.
5. Nearest-neighbor resampling preserves existing EVT class codes.
6. Each 1-km cell then contains exactly 40 x 40 = 1,600 destination samples.
7. Class counts determine the plurality / dominant EVT class.
8. Dominant fraction describes how strongly the winning class dominates.
9. QA also checks:
   - whether all 1,600 positions contain valid vegetation data
   - whether more than one class ties for the highest count

Use the # %% cells in VS Code one at a time.

ANDFIRE LF2016 EVT CONUS
huge national raster
        ↓
script 13 requests a small area
near the Sandia Mountains
        ↓
small local GeoTIFF
~270 × 270 pixels
at 30 m
        ↓
used by your playground


"""

# %% Imports
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Rectangle
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


# %% Constants and paths
TARGET_CRS = "EPSG:5070"

SOURCE_RESOLUTION_M = 30
DESTINATION_RESOLUTION_M = 25
ANALYSIS_CELL_SIZE_M = 1_000

SAMPLES_PER_CELL_SIDE = (
    ANALYSIS_CELL_SIZE_M // DESTINATION_RESOLUTION_M
)

EXPECTED_SAMPLES_PER_CELL = (
    SAMPLES_PER_CELL_SIDE**2
)

SOURCE_NODATA = -9999

repo_root = Path(__file__).resolve().parents[1]

grid_path = (
    repo_root
    / "data"
    / "processed"
    / "grids"
    / "nm_analysis_grid_1km.gpkg"
)

prototype_table_path = (
    repo_root
    / "data"
    / "interim"
    / "landfire_evt_prototype.parquet"
)

playground_output_dir = (
    repo_root
    / "outputs"
    / "landfire_evt_playground"
)

playground_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# %% 1 — Find the LANDFIRE prototype GeoTIFF created by script 13
def find_landfire_prototype_tif(repo: Path) -> Path:
    """Find the small LANDFIRE prototype raster by its known metadata.

    Script 13 produced an approximately 270 x 270 pixel GeoTIFF in EPSG:5070
    with approximately 30-m source pixels. This search avoids hard-coding a
    filename that may differ from one implementation to another.
    """
    candidates = []

    for tif_path in repo.rglob("*.tif"):
        try:
            with rasterio.open(tif_path) as src:
                is_epsg_5070 = (
                    src.crs is not None
                    and src.crs.to_epsg() == 5070
                )

                is_30m = np.allclose(
                    src.res,
                    (30.0, 30.0),
                    atol=0.01,
                )

                is_small_prototype = (
                    200 <= src.width <= 400
                    and 200 <= src.height <= 400
                )

                if (
                    is_epsg_5070
                    and is_30m
                    and is_small_prototype
                ):
                    candidates.append(tif_path)

        except rasterio.errors.RasterioIOError:
            pass

    if not candidates:
        raise FileNotFoundError(
            "Could not find the LANDFIRE prototype GeoTIFF. "
            "Run analysis/13_validate_landfire_evt.py first."
        )

    print("Candidate prototype GeoTIFFs:")

    for candidate in candidates:
        print(" ", candidate)

    return candidates[0]


landfire_tif_path = find_landfire_prototype_tif(
    repo_root
)

print(
    "\nUsing:",
    landfire_tif_path,
)


# %% 2 — Inspect the native LANDFIRE raster metadata
with rasterio.open(landfire_tif_path) as src:
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("Dtype:", src.dtypes[0])
    print("NoData:", src.nodata)
    print("Transform:")
    print(src.transform)


# %% 3 — Read the native 30-m categorical array
with rasterio.open(landfire_tif_path) as src:
    native_evt = src.read(1)

    native_transform = src.transform
    native_bounds = src.bounds
    native_crs = src.crs
    native_nodata = src.nodata


print("Array shape:", native_evt.shape)
print("Array dtype:", native_evt.dtype)

native_evt


# %% 4 — Inspect the actual class codes in the native raster
native_valid_mask = (
    native_evt != native_nodata
)

native_valid_values = (
    native_evt[native_valid_mask]
)

native_classes, native_counts = np.unique(
    native_valid_values,
    return_counts=True,
)

native_class_table = pd.DataFrame(
    {
        "evt_class": native_classes,
        "count": native_counts,
    }
).sort_values(
    "count",
    ascending=False,
)

native_class_table.head(20)


# %% 5 — Basic statistics about native categorical values
print(
    "Native raster positions:",
    native_evt.size,
)

print(
    "Valid categorical samples:",
    native_valid_mask.sum(),
)

print(
    "NoData samples:",
    (~native_valid_mask).sum(),
)

print(
    "Distinct EVT class codes:",
    len(native_classes),
)


# %% 6 — Plot the native 30-m categorical raster
fig, ax = plt.subplots(
    figsize=(9, 9)
)

image = ax.imshow(
    native_evt,
    extent=(
        native_bounds.left,
        native_bounds.right,
        native_bounds.bottom,
        native_bounds.top,
    ),
    origin="upper",
    interpolation="nearest",
)

ax.set_title(
    "Native LANDFIRE EVT Prototype — 30 m"
)

ax.set_xlabel(
    "EPSG:5070 x (m)"
)

ax.set_ylabel(
    "EPSG:5070 y (m)"
)

ax.set_aspect(
    "equal"
)

plt.show()


# %% 7 — Load the authoritative 1-km analysis grid
analysis_grid = gpd.read_file(
    grid_path,
    layer="analysis_grid",
)

print(
    "Analysis cells:",
    len(analysis_grid),
)

print(
    "Analysis CRS:",
    analysis_grid.crs,
)


# %% 8 — Select analysis cells overlapping the prototype raster
prototype_bbox = gpd.GeoSeries.from_bbox(
    native_bounds,
    crs=TARGET_CRS,
).iloc[0]

prototype_grid = analysis_grid.loc[
    analysis_grid.geometry.intersects(
        prototype_bbox
    )
].copy()

print(
    "Analysis cells touching prototype raster:",
    len(prototype_grid),
)


# %% 9 — Plot native raster with 1-km analysis-cell boundaries
fig, ax = plt.subplots(
    figsize=(10, 10)
)

ax.imshow(
    native_evt,
    extent=(
        native_bounds.left,
        native_bounds.right,
        native_bounds.bottom,
        native_bounds.top,
    ),
    origin="upper",
    interpolation="nearest",
)

prototype_grid.boundary.plot(
    ax=ax,
    linewidth=1.3,
)

ax.set_title(
    "Native 30-m LANDFIRE Raster + 1-km Analysis Grid"
)

ax.set_aspect(
    "equal"
)

plt.show()


# %% 10 — Understand why 30 m does not nest cleanly into 1 km
print(
    "1000 / 30 =",
    ANALYSIS_CELL_SIZE_M / SOURCE_RESOLUTION_M,
)

print(
    "1000 / 25 =",
    ANALYSIS_CELL_SIZE_M / DESTINATION_RESOLUTION_M,
)

print(
    "25-m samples per 1-km side:",
    SAMPLES_PER_CELL_SIDE,
)

print(
    "25-m samples per 1-km cell:",
    EXPECTED_SAMPLES_PER_CELL,
)


# %% 11 — Build an aligned 25-m destination extent

# Use complete 1-km cells that overlap the prototype raster.
grid_bounds = prototype_grid.total_bounds

left = grid_bounds[0]
bottom = grid_bounds[1]
right = grid_bounds[2]
top = grid_bounds[3]

destination_width = int(
    (right - left)
    / DESTINATION_RESOLUTION_M
)

destination_height = int(
    (top - bottom)
    / DESTINATION_RESOLUTION_M
)

destination_transform = from_origin(
    left,
    top,
    DESTINATION_RESOLUTION_M,
    DESTINATION_RESOLUTION_M,
)

print(
    "Destination rows:",
    destination_height,
)

print(
    "Destination columns:",
    destination_width,
)

print(
    "Destination transform:"
)

print(
    destination_transform
)


# %% 12 — Create empty 25-m categorical destination array
aligned_evt = np.full(
    (
        destination_height,
        destination_width,
    ),
    SOURCE_NODATA,
    dtype=np.int16,
)

print(
    "Aligned array shape:",
    aligned_evt.shape,
)


# %% 13 — Reproject 30 m → aligned 25 m using nearest neighbor
with rasterio.open(landfire_tif_path) as src:

    reproject(
        source=rasterio.band(
            src,
            1,
        ),
        destination=aligned_evt,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=destination_transform,
        dst_crs=TARGET_CRS,
        dst_nodata=SOURCE_NODATA,
        resampling=Resampling.nearest,
    )


print(
    "Finished nearest-neighbor reprojection."
)


# %% 14 — Confirm nearest-neighbor did not invent class codes
aligned_valid = (
    aligned_evt != SOURCE_NODATA
)

aligned_classes = np.unique(
    aligned_evt[aligned_valid]
)

invented_classes = np.setdiff1d(
    aligned_classes,
    native_classes,
)

print(
    "Native classes:",
    len(native_classes),
)

print(
    "Aligned classes:",
    len(aligned_classes),
)

print(
    "Invented classes:",
    invented_classes,
)


# %% 15 — Plot the aligned 25-m categorical raster
fig, ax = plt.subplots(
    figsize=(10, 10)
)

ax.imshow(
    aligned_evt,
    extent=(
        left,
        right,
        bottom,
        top,
    ),
    origin="upper",
    interpolation="nearest",
)

prototype_grid.boundary.plot(
    ax=ax,
    linewidth=1.3,
)

ax.set_title(
    "Aligned 25-m LANDFIRE Raster + 1-km Grid"
)

ax.set_aspect(
    "equal"
)

plt.show()


# %% 16 — Zoom into one 1-km cell
example_cell = (
    prototype_grid
    .sort_values("cell_id")
    .iloc[
        len(prototype_grid) // 2
    ]
)

example_cell_id = (
    example_cell["cell_id"]
)

example_geometry = (
    example_cell.geometry
)

example_bounds = (
    example_geometry.bounds
)

print(
    "Example cell:",
    example_cell_id,
)

print(
    "Bounds:",
    example_bounds,
)


# %% 17 — Calculate row/column location of example cell
cell_left = (
    example_bounds[0]
)

cell_bottom = (
    example_bounds[1]
)

cell_right = (
    example_bounds[2]
)

cell_top = (
    example_bounds[3]
)

column_start = int(
    round(
        (
            cell_left
            - left
        )
        / DESTINATION_RESOLUTION_M
    )
)

column_stop = (
    column_start
    + SAMPLES_PER_CELL_SIDE
)

row_start = int(
    round(
        (
            top
            - cell_top
        )
        / DESTINATION_RESOLUTION_M
    )
)

row_stop = (
    row_start
    + SAMPLES_PER_CELL_SIDE
)

print(
    "Rows:",
    row_start,
    row_stop,
)

print(
    "Columns:",
    column_start,
    column_stop,
)


# %% 18 — Extract the 40 x 40 categorical samples for one cell
cell_samples = aligned_evt[
    row_start:row_stop,
    column_start:column_stop,
]

print(
    "Cell-sample shape:",
    cell_samples.shape,
)

print(
    "Total positions:",
    cell_samples.size,
)

cell_samples


# %% 19 — Coverage QA: valid samples versus NoData
cell_valid_mask = (
    cell_samples
    != SOURCE_NODATA
)

valid_sample_count = int(
    cell_valid_mask.sum()
)

nodata_sample_count = int(
    (~cell_valid_mask).sum()
)

print(
    "Expected positions:",
    EXPECTED_SAMPLES_PER_CELL,
)

print(
    "Valid EVT samples:",
    valid_sample_count,
)

print(
    "NoData positions:",
    nodata_sample_count,
)

print(
    "Valid fraction:",
    valid_sample_count
    / EXPECTED_SAMPLES_PER_CELL,
)


# %% 20 — Count the EVT classes in this 1-km cell
valid_values = cell_samples[
    cell_valid_mask
]

classes, counts = np.unique(
    valid_values,
    return_counts=True,
)

class_count_table = pd.DataFrame(
    {
        "evt_class": classes,
        "sample_count": counts,
    }
).sort_values(
    "sample_count",
    ascending=False,
)

class_count_table


# %% 21 — Find the highest class count
highest_count = (
    class_count_table[
        "sample_count"
    ].max()
)

winning_rows = class_count_table.loc[
    class_count_table[
        "sample_count"
    ]
    == highest_count
]

winning_rows


# %% 22 — Detect whether there is a tie
tie_exists = (
    len(winning_rows)
    > 1
)

print(
    "Tie exists:",
    tie_exists,
)

print(
    "Number of tied winners:",
    len(winning_rows),
)


# %% 23 — Choose dominant class reproducibly

# If there is one winner, use it.
#
# If there is an exact tie, choose the numerically smallest EVT code.
# This does NOT mean that smaller codes are ecologically preferable.
# It merely guarantees that repeated runs produce the same answer.
dominant_class = int(
    winning_rows[
        "evt_class"
    ].min()
)

winning_sample_count = int(
    highest_count
)

dominant_fraction = (
    winning_sample_count
    / valid_sample_count
)

print(
    "Dominant EVT class:",
    dominant_class,
)

print(
    "Winning sample count:",
    winning_sample_count,
)

print(
    "Valid sample count:",
    valid_sample_count,
)

print(
    "Dominant fraction:",
    dominant_fraction,
)


# %% 24 — Plain-language summary of this cell
print(
    f"Cell {example_cell_id}"
)

print(
    f"has {valid_sample_count:,} "
    "valid vegetation samples "
    f"out of {EXPECTED_SAMPLES_PER_CELL:,} "
    "possible positions."
)

print(
    f"The most common EVT class is "
    f"{dominant_class}."
)

print(
    f"It occupies "
    f"{dominant_fraction:.1%} "
    "of valid samples."
)

if tie_exists:
    print(
        "More than one class had the "
        "same highest count."
    )

else:
    print(
        "The plurality winner is unique."
    )


# %% 25 — Plot the 40 x 40 samples inside the example cell
fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.imshow(
    cell_samples,
    origin="upper",
    interpolation="nearest",
)

ax.set_title(
    f"25-m EVT Samples Inside 1-km Cell\n"
    f"{example_cell_id}"
)

ax.set_xlabel(
    "25-m sample column"
)

ax.set_ylabel(
    "25-m sample row"
)

plt.show()


# %% 26 — Make the 40 x 40 sample matrix easier to inspect
cell_samples_dataframe = pd.DataFrame(
    cell_samples
)

cell_samples_dataframe


# %% 27 — Show the most common classes in the cell
class_count_table.head(
    15
)


# %% 28 — Show class proportions
class_count_table[
    "fraction_of_valid_samples"
] = (
    class_count_table[
        "sample_count"
    ]
    / valid_sample_count
)

class_count_table.head(
    15
)


# %% 29 — Demonstrate the meaning of plurality

# Example:
#
# Class A = 42%
# Class B = 35%
# Class C = 23%
#
# Class A is the plurality winner even though
# it does NOT have a majority (>50%).

print(
    "Dominant fraction:",
    dominant_fraction,
)

print(
    "Majority?",
    dominant_fraction > 0.5,
)

print(
    "Plurality winner:",
    dominant_class,
)


# %% 30 — Demonstrate what an exact tie looks like
synthetic_tie = np.array(
    [
        7054,
        7054,
        7292,
        7292,
        8001,
    ]
)

tie_classes, tie_counts = np.unique(
    synthetic_tie,
    return_counts=True,
)

synthetic_tie_table = pd.DataFrame(
    {
        "evt_class": tie_classes,
        "count": tie_counts,
    }
)

synthetic_tie_table


# %% 31 — Detect the synthetic tie
synthetic_max_count = (
    synthetic_tie_table[
        "count"
    ].max()
)

synthetic_winners = (
    synthetic_tie_table.loc[
        synthetic_tie_table[
            "count"
        ]
        == synthetic_max_count
    ]
)

print(
    "Synthetic tied winners:"
)

print(
    synthetic_winners
)

print(
    "Tie exists:",
    len(synthetic_winners) > 1,
)


# %% 32 — Apply deterministic tie rule to synthetic example
synthetic_chosen_class = int(
    synthetic_winners[
        "evt_class"
    ].min()
)

print(
    "Chosen class:",
    synthetic_chosen_class,
)

print(
    "Important: this class was chosen only "
    "to make the result reproducible."
)

print(
    "It does NOT mean the smaller class code "
    "has greater ecological importance."
)


# %% 33 — Demonstrate partial coverage / NoData
synthetic_coverage = np.array(
    [
        7054,
        7054,
        7292,
        SOURCE_NODATA,
        SOURCE_NODATA,
    ]
)

synthetic_valid_mask = (
    synthetic_coverage
    != SOURCE_NODATA
)

print(
    "Total destination positions:",
    synthetic_coverage.size,
)

print(
    "Valid samples:",
    synthetic_valid_mask.sum(),
)

print(
    "NoData positions:",
    (~synthetic_valid_mask).sum(),
)


# %% 34 — Why extra raster support cannot fix genuine NoData
print(
    """
Two different problems:

EDGE SUPPORT PROBLEM
--------------------
The source contains valid values nearby,
but we did not read enough surrounding raster.

Solution:
read a slightly larger source window / halo.


GENUINE NODATA PROBLEM
----------------------
The source itself contains -9999 at that location.

Adding more border pixels cannot manufacture
a vegetation class that the source does not contain.
"""
)


# %% 35 — Load script 13 prototype table
prototype_table = pd.read_parquet(
    prototype_table_path
)

prototype_table.head()


# %% 36 — Inspect prototype table structure
prototype_table.info()


# %% 37 — Inspect prototype table statistics
prototype_table.describe(
    include="all"
)


# %% 38 — Inspect dominant class frequencies
prototype_table[
    "evt_dominant_class"
].value_counts()


# %% 39 — Inspect dominant fractions
prototype_table[
    "evt_dominant_fraction"
].sort_values()


# %% 40 — Plot dominant-fraction histogram
fig, ax = plt.subplots(
    figsize=(8, 6)
)

prototype_table[
    "evt_dominant_fraction"
].hist(
    bins=10,
    ax=ax,
)

ax.set_title(
    "Prototype Dominant EVT Fraction"
)

ax.set_xlabel(
    "Dominant fraction"
)

ax.set_ylabel(
    "Number of 1-km cells"
)

plt.show()


# %% 41 — Look for QA coverage columns produced by script 13
coverage_columns = [
    column
    for column
    in prototype_table.columns
    if (
        "sample"
        in column.lower()
        or "nodata"
        in column.lower()
        or "valid"
        in column.lower()
    )
]

print(
    "Possible coverage QA columns:"
)

print(
    coverage_columns
)


# %% 42 — Look for tie-related columns
tie_columns = [
    column
    for column
    in prototype_table.columns
    if "tie" in column.lower()
]

print(
    "Possible tie QA columns:"
)

print(
    tie_columns
)


# %% 43 — Optional: display those QA columns
qa_columns = (
    ["cell_id"]
    + coverage_columns
    + tie_columns
)

qa_columns = list(
    dict.fromkeys(
        qa_columns
    )
)

prototype_table[
    qa_columns
].head(
    20
)


# %% 44 — Save a zoomed alignment illustration
fig, ax = plt.subplots(
    figsize=(9, 9)
)

example_xmin = cell_left
example_xmax = cell_right
example_ymin = cell_bottom
example_ymax = cell_top


# Draw the 1-km analysis cell.
analysis_rectangle = Rectangle(
    (
        example_xmin,
        example_ymin,
    ),
    ANALYSIS_CELL_SIZE_M,
    ANALYSIS_CELL_SIZE_M,
    fill=False,
    linewidth=3,
)

ax.add_patch(
    analysis_rectangle
)


# Draw aligned 25-m destination grid lines.
for x in np.arange(
    example_xmin,
    example_xmax
    + DESTINATION_RESOLUTION_M,
    DESTINATION_RESOLUTION_M,
):

    ax.axvline(
        x,
        linewidth=0.25,
        alpha=0.5,
    )


for y in np.arange(
    example_ymin,
    example_ymax
    + DESTINATION_RESOLUTION_M,
    DESTINATION_RESOLUTION_M,
):

    ax.axhline(
        y,
        linewidth=0.25,
        alpha=0.5,
    )


# Draw native 30-m source-grid lines using the
# actual raster transform / origin.
source_origin_x = (
    native_transform.c
)

source_origin_y = (
    native_transform.f
)


first_source_x = (
    source_origin_x
    + np.floor(
        (
            example_xmin
            - source_origin_x
        )
        / SOURCE_RESOLUTION_M
    )
    * SOURCE_RESOLUTION_M
)


first_source_y = (
    source_origin_y
    - np.floor(
        (
            source_origin_y
            - example_ymax
        )
        / SOURCE_RESOLUTION_M
    )
    * SOURCE_RESOLUTION_M
)


for x in np.arange(
    first_source_x,
    example_xmax
    + SOURCE_RESOLUTION_M,
    SOURCE_RESOLUTION_M,
):

    ax.axvline(
        x,
        linestyle="--",
        linewidth=0.8,
    )


for y in np.arange(
    first_source_y,
    example_ymin
    - SOURCE_RESOLUTION_M,
    -SOURCE_RESOLUTION_M,
):

    ax.axhline(
        y,
        linestyle="--",
        linewidth=0.8,
    )


ax.set_xlim(
    example_xmin,
    example_xmax,
)

ax.set_ylim(
    example_ymin,
    example_ymax,
)

ax.set_aspect(
    "equal"
)

ax.set_title(
    "One 1-km Cell\n"
    "Dashed = native 30-m grid | "
    "Fine solid = aligned 25-m grid"
)

alignment_plot_path = (
    playground_output_dir
    / "one_cell_30m_vs_25m_alignment.png"
)

fig.tight_layout()

fig.savefig(
    alignment_plot_path,
    dpi=180,
)

plt.show()

print(
    "Saved:",
    alignment_plot_path,
)


# %% 45 — Final mental-model summary
print(
    """
LANDFIRE PLAYGROUND SUMMARY
===========================

SOURCE
------
LANDFIRE EVT starts as categorical 30-m pixels.

ALIGNMENT
---------
30 m does not divide evenly into a 1-km cell.

1000 / 30 = 33.333...

So we re-express the categorical raster on
an aligned 25-m destination grid.

1000 / 25 = 40

Each 1-km cell therefore has exactly:

40 x 40 = 1600 destination positions.


NEAREST NEIGHBOR
----------------
Each 25-m destination position copies an
existing nearby LANDFIRE class.

It does NOT average class codes.


COVERAGE
--------
Every cell always has 1600 destination positions.

But some positions could theoretically contain
NoData if the source itself has no valid class there.

So:

number of positions = always 1600

number of valid vegetation samples
= something we verify


PLURALITY
---------
Count valid EVT class codes.

The class with the highest count is the
dominant / plurality class.


DOMINANT FRACTION
-----------------
winning class count
-------------------
valid sample count


TIES
----
If exactly one class has the highest count:

normal winner

If two or more classes share the highest count:

flag the cell as a tie

For reproducibility, the implementation may
temporarily choose the smallest EVT code,
but the QA report must still tell us that a
tie occurred.

The numerical size of the EVT code has no
ecological meaning.
"""
)   