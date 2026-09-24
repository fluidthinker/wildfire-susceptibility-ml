"""Explore directional holdouts using a pragmatic candidate 50-km block size.

This is exploratory QA for human review before script 16. It neither assigns
final splits nor estimates a spatial independence distance. Run from any folder.
"""

# %% Imports
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from shapely import box


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
GRID_PATH = ROOT / "data/processed/grids/nm_analysis_grid_1km.gpkg"
OUTPUT_DIR = ROOT / "outputs/modeling/spatial_holdout_exploration"
BLOCK_SIZE_M = 50_000
HOLDOUT_FRACTION = 0.20
EXPECTED_MODELING_ROWS = 313_129
EXPECTED_GRID_ROWS = 314_920
CRS = "EPSG:5070"
DIRECTIONS = ("west", "east", "north", "south")


# %% Helper functions
def validate_cell_keys(table: pd.DataFrame, expected_rows: int, label: str) -> None:
    """Check that an input contains the expected number of uniquely keyed cells.

    Args:
        table: Input rows with a cell_id column.
        expected_rows: Approved row count for this input.
        label: Input name used in failure messages.

    Raises:
        ValueError: If row count or cell keys differ from the known contract.
    """
    wrong_count = len(table) != expected_rows
    missing_ids = table.cell_id.isna().any()
    duplicate_ids = table.cell_id.duplicated().any()
    if wrong_count or missing_ids or duplicate_ids:
        raise ValueError(f"{label}: rows={len(table)}, missing IDs={missing_ids}, "
                         f"duplicate IDs={duplicate_ids}")


def load_eligible_modeling_cells() -> pd.DataFrame:
    """Read only eligible cell IDs and labels needed to explore geography.

    Returns:
        Validated modeling keys and binary labels, without predictor columns.

    Raises:
        ValueError: If the approved population or binary target contract fails.
    """
    cells = pd.read_parquet(MODELING_PATH, columns=["cell_id", "target"])
    validate_cell_keys(cells, EXPECTED_MODELING_ROWS, "Modeling dataset")
    if not cells.target.isin([0, 1]).all():
        raise ValueError("Modeling targets must be nonmissing binary 0/1 values")
    return cells


def load_authoritative_grid() -> gpd.GeoDataFrame:
    """Read the original complete 1-km cells without changing their geometry.

    Returns:
        Validated EPSG:5070 grid with only keys and geometry.

    Raises:
        ValueError: If keys, CRS, or complete square geometry are unexpected.
    """
    grid = gpd.read_file(GRID_PATH, layer="analysis_grid", columns=["cell_id"])
    validate_cell_keys(grid, EXPECTED_GRID_ROWS, "Authoritative grid")
    if grid.crs != CRS:
        raise ValueError(f"Expected {CRS}; found {grid.crs}. No reprojection performed.")
    invalid_geometry = grid.geometry.isna() | grid.geometry.is_empty | ~grid.geometry.is_valid
    if invalid_geometry.any():
        raise ValueError(f"Invalid or missing grid geometry: {invalid_geometry.sum()} cells")

    # Complete axis-aligned squares make centroid assignment unambiguous.
    # Validate existing geometry; do not repair or clip boundary cells.
    bounds = grid.geometry.bounds
    square_width = np.isclose(bounds.maxx - bounds.minx, 1000, rtol=0, atol=0.001)
    square_height = np.isclose(bounds.maxy - bounds.miny, 1000, rtol=0, atol=0.001)
    square_area = np.isclose(grid.area, 1_000_000, rtol=0, atol=0.01)
    if not (square_width & square_height & square_area).all():
        raise ValueError("Grid contains geometry other than complete 1-km squares")
    return grid


def join_modeling_geometry(cells: pd.DataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach authoritative geometry to eligible cells by exact cell ID.

    Args:
        cells: Eligible modeling keys and labels.
        grid: Complete authoritative grid, including excluded MTBS cells.

    Returns:
        One geometry per eligible cell, sorted by cell ID.

    Raises:
        ValueError: If any modeling key lacks authoritative geometry.
    """
    missing = ~cells.cell_id.isin(grid.cell_id)
    if missing.any():
        raise ValueError(f"{missing.sum()} modeling IDs are absent from the grid")
    joined = grid.merge(cells, on="cell_id", how="inner", validate="one_to_one")
    validate_cell_keys(joined, EXPECTED_MODELING_ROWS, "Eligible geometry join")
    return joined.sort_values("cell_id").reset_index(drop=True)


def assign_spatial_blocks(
    cells: gpd.GeoDataFrame, grid_bounds: np.ndarray
) -> tuple[gpd.GeoDataFrame, tuple[float, float]]:
    """Place each eligible cell centroid in a deterministic regular 50-km block.

    Args:
        cells: Eligible cells in EPSG:5070.
        grid_bounds: Bounds of the full authoritative grid, not an eligible subset.

    Returns:
        Cells with exploratory block positions and the snapped southwest anchor.
    """
    # Snapping the full grid bounds downward avoids moving the anchor when
    # eligibility changes. For example, x=anchor_x+75,000 belongs to column 1.
    anchor_x, anchor_y = np.floor(grid_bounds[:2] / BLOCK_SIZE_M) * BLOCK_SIZE_M
    centroids = cells.geometry.centroid
    assigned = cells.copy()
    assigned["centroid_x"] = centroids.x
    assigned["centroid_y"] = centroids.y
    assigned["block_col"] = np.floor((centroids.x - anchor_x) / BLOCK_SIZE_M).astype("int64")
    assigned["block_row"] = np.floor((centroids.y - anchor_y) / BLOCK_SIZE_M).astype("int64")
    assigned["block_id"] = "r" + assigned.block_row.astype(str) + "_c" + assigned.block_col.astype(str)
    return assigned, (float(anchor_x), float(anchor_y))


def summarize_blocks(cells: gpd.GeoDataFrame, anchor: tuple[float, float]) -> gpd.GeoDataFrame:
    """Count eligible cells and positives within each occupied regular block.

    Args:
        cells: Eligible cells with exploratory block positions.
        anchor: Southwest coordinate origin in meters.

    Returns:
        One summary row and full square geometry per occupied block.
    """
    # Grouping collapses cell rows into block rows; target sum counts positives.
    blocks = cells.groupby(["block_id", "block_row", "block_col"], as_index=False).agg(
        cells=("cell_id", "size"), positives=("target", "sum")
    ).sort_values(["block_row", "block_col"]).reset_index(drop=True)
    blocks["negatives"] = blocks.cells - blocks.positives
    blocks["positive_percent"] = 100 * blocks.positives / blocks.cells
    blocks["min_x_m"] = anchor[0] + blocks.block_col * BLOCK_SIZE_M
    blocks["min_y_m"] = anchor[1] + blocks.block_row * BLOCK_SIZE_M
    blocks["max_x_m"] = blocks.min_x_m + BLOCK_SIZE_M
    blocks["max_y_m"] = blocks.min_y_m + BLOCK_SIZE_M
    geometry = box(blocks.min_x_m, blocks.min_y_m, blocks.max_x_m, blocks.max_y_m)
    return gpd.GeoDataFrame(blocks, geometry=geometry, crs=CRS)


def select_directional_holdout(blocks: gpd.GeoDataFrame, direction: str) -> gpd.GeoDataFrame:
    """Select a directional band of whole blocks closest to 20% of cells.

    Complete columns or rows avoid arbitrary tie-breaking within a strip.
    Labels do not influence selection. Equal-distance cutoffs prefer fewer cells.

    Args:
        blocks: Occupied block counts and positions.
        direction: West, east, north, or south in projected coordinate space.

    Returns:
        Selected whole occupied blocks; strip granularity can miss 20%.

    Raises:
        ValueError: If the direction is unsupported.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction}")
    axis = "block_col" if direction in ("west", "east") else "block_row"
    ascending = direction in ("west", "south")
    strips = blocks.groupby(axis).cells.sum().sort_index(ascending=ascending)
    cumulative_cells = strips.cumsum()
    distance_from_target = (cumulative_cells - HOLDOUT_FRACTION * blocks.cells.sum()).abs()
    # idxmin returns the first minimum in directional order. Every strip adds
    # cells, so a tie chooses the smaller population, independently of labels.
    cutoff = distance_from_target.idxmin()
    selected = blocks[axis] <= cutoff if ascending else blocks[axis] >= cutoff
    return blocks.loc[selected].copy()


def count_block_components(blocks: gpd.GeoDataFrame) -> int:
    """Count connected groups using shared block edges rather than corner contact.

    Args:
        blocks: Selected occupied blocks with integer row/column positions.

    Returns:
        Number of four-neighbor connected components; one means contiguous.
    """
    remaining = set(zip(blocks.block_row, blocks.block_col))
    components = 0
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        while frontier:
            row, col = frontier.pop()
            # Only north/south/east/west adjacency qualifies. An unoccupied
            # block does not bridge two groups of selected occupied blocks.
            neighbors = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    return components


def summarize_holdout_candidate(
    selected: gpd.GeoDataFrame, direction: str, statewide_cells: int
) -> dict:
    """Describe a candidate's population, full-block extents, and connectivity.

    Args:
        selected: Whole occupied blocks in this candidate.
        direction: Candidate name.
        statewide_cells: Eligible population used as the percentage denominator.

    Returns:
        Summary suitable for CSV and human review; no ranking is applied.
    """
    cells = int(selected.cells.sum())
    positives = int(selected.positives.sum())
    components = count_block_components(selected)
    return {
        "direction": direction, "blocks": len(selected), "cells": cells,
        "statewide_percent": 100 * cells / statewide_cells,
        "positives": positives, "negatives": cells - positives,
        "positive_percent": 100 * positives / cells,
        "min_x_m": selected.min_x_m.min(), "max_x_m": selected.max_x_m.max(),
        "min_y_m": selected.min_y_m.min(), "max_y_m": selected.max_y_m.max(),
        "components": components, "contiguous": components == 1,
    }


def plot_spatial_blocks(blocks: gpd.GeoDataFrame, cells: gpd.GeoDataFrame) -> None:
    """Map occupied blocks and their positive counts for exploratory inspection.

    Args:
        blocks: Full occupied block squares and summary counts.
        cells: Eligible cell centroids used to show the actual analysis footprint.
    """
    fig, ax = plt.subplots(figsize=(8, 9))
    blocks.plot(column="positives", cmap="YlOrRd", edgecolor="0.5", linewidth=0.4,
                legend=True, legend_kwds={"label": "Positive eligible cells per block", "shrink": 0.6}, ax=ax)
    ax.scatter(cells.centroid_x, cells.centroid_y, s=0.03, c="0.25", alpha=0.12, linewidths=0)
    ax.set(title="New Mexico: candidate 50-km blocks\nFull squares; subtle dots show eligible cell footprint",
           xlabel="EPSG:5070 easting (m)", ylabel="EPSG:5070 northing (m)", aspect="equal")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spatial_blocks.png", dpi=160)
    plt.close(fig)


def plot_holdout_candidate(
    blocks: gpd.GeoDataFrame, cells: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame, summary: dict,
) -> None:
    """Show selected whole blocks against the remaining eligible cell footprint.

    Args:
        blocks: All occupied blocks, used as common map context.
        cells: Eligible cell centroids.
        selected: Candidate blocks highlighted in blue.
        summary: Candidate counts used in the title.
    """
    fig, ax = plt.subplots(figsize=(8, 9))
    blocks.plot(ax=ax, color="0.94", edgecolor="0.75", linewidth=0.4)
    selected.plot(ax=ax, color="#74add1", edgecolor="#2166ac", linewidth=0.6)
    ax.scatter(cells.centroid_x, cells.centroid_y, s=0.03, c="0.25", alpha=0.15, linewidths=0)
    title = (f"{summary['direction'].title()} exploratory spatial holdout\n"
             f"{summary['statewide_percent']:.2f}% of cells; {summary['positives']:,} positives; "
             f"{summary['positive_percent']:.2f}% prevalence")
    ax.set(title=title, xlabel="EPSG:5070 easting (m)", ylabel="EPSG:5070 northing (m)", aspect="equal")
    ax.legend(handles=[Patch(facecolor="#74add1", label="Candidate whole blocks"),
                       Patch(facecolor="0.94", label="Remaining occupied blocks")], loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"holdout_{summary['direction']}.png", dpi=160)
    plt.close(fig)


def save_exploration_report(
    cells: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, candidates: pd.DataFrame,
    anchor: tuple[float, float], runtime: float,
) -> None:
    """Save compact block/candidate tables and explain the exploratory decisions.

    Args:
        cells: Eligible modeled population with exploratory block positions.
        blocks: Occupied block summaries.
        candidates: Directional population summaries, without a preferred choice.
        anchor: Stable southwest origin in EPSG:5070 meters.
        runtime: Elapsed computation and map runtime, before report writing.
    """
    blocks.drop(columns="geometry").to_csv(OUTPUT_DIR / "block_summary.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "holdout_candidates.csv", index=False)
    statistics = blocks[["cells", "positives"]].agg(["min", "median", "mean", "max"])
    positives = int(cells.target.sum())
    zero_blocks = int((blocks.positives == 0).sum())
    fragmented = candidates.loc[~candidates.contiguous, "direction"].tolist()
    report = f"""# Exploratory spatial holdout QA

Inputs: `{MODELING_PATH.relative_to(ROOT).as_posix()}` and
`{GRID_PATH.relative_to(ROOT).as_posix()}` (layer `analysis_grid`).

Eligible rows: {len(cells):,}; unique cell IDs: {cells.cell_id.nunique():,}.
Negatives: {len(cells) - positives:,}; positives: {positives:,};
positive prevalence: {100 * positives / len(cells):.4f}%.
Occupied candidate 50-km blocks: {len(blocks)}.
Zero-positive blocks: {zero_blocks}; blocks with positives: {len(blocks) - zero_blocks}.

Block statistics:

```text
{statistics.to_string(float_format=lambda value: f'{value:.3f}')}
```

Directional candidates (extents are full block edges in EPSG:5070 meters):

```text
{candidates.to_string(index=False, float_format=lambda value: f'{value:.3f}')}
```

The anchor is ({anchor[0]:.0f}, {anchor[1]:.0f}) meters, obtained by snapping the
full authoritative grid's southwest bounds down to multiples of {BLOCK_SIZE_M:,} m.
Centroid coordinates determine block columns/rows using floor division.
Only eligible cells contribute to block counts. No geometry is repaired,
reprojected, or clipped; original complete 1-km cells are preserved.

Candidates use complete directional columns (west/east) or rows (north/south),
choosing the cumulative population closest to 20%. Ties prefer fewer cells.
This produces simple bands; whole-strip granularity can cause noticeable
departures from 20%. Selection uses cell counts and coordinates, never labels.
Directions refer to projected x/y axes. No candidate is ranked or chosen.

Contiguity means shared-edge connectivity of occupied 50-km blocks, not
continuous coverage by eligible 1-km cells. Corner contact does not count.
Fragmented candidates: {fragmented or 'none'}. Input geometry checks passed.

Maps: `spatial_blocks.png`, `holdout_west.png`, `holdout_east.png`,
`holdout_north.png`, and `holdout_south.png`. Full block squares can extend
beyond the state footprint; subtle centroid dots show eligible coverage.

50 km is a pragmatic candidate size, not an optimal size or a measured
autocorrelation distance. Adjacent training and holdout blocks have no buffer;
this exploration does not establish statistical independence. Counts describe
candidate populations but do not establish adequacy for later model evaluation.
Human review should approve the design before script 16 freezes any splits.

Engineering: narrow input reads, validated one-to-one key join, vectorized
centroid assignment, deterministic label-independent selection, and separate
helpers for loading, geometry assignment, summaries, connectivity, and maps.
No final per-cell split or CV artifact is written.

Runtime through map generation: {runtime:.3f} seconds.

80/20 takeaway: inspect where positive cells fall across whole blocks and how
directional bands change the held-out population before choosing a design.
"""
    (OUTPUT_DIR / "README.md").write_text(report, encoding="utf-8")
    print(report)


# %% Main workflow
def main() -> None:
    """Explore four spatial holdout candidates and save QA for human review."""
    started = perf_counter()
    # STEP 1 - Attach authoritative geometry to eligible modeling rows.
    # Only cell IDs and labels are needed; predictors remain in the source file.
    modeling = load_eligible_modeling_cells()
    grid = load_authoritative_grid()
    cells = join_modeling_geometry(modeling, grid)

    # STEP 2 - Assign deterministic candidate 50-km blocks from cell centroids.
    # Anchor to the full grid so exclusions cannot shift the block system.
    cells, anchor = assign_spatial_blocks(cells, grid.total_bounds)

    # STEP 3 - Summarize occupied blocks and their eligible positive populations.
    # Block counts reveal how much support exists for later spatial evaluation.
    blocks = summarize_blocks(cells, anchor)

    # STEP 4 - Form four directional bands using whole columns or rows.
    # Close-to-20% cell counts are preferred over cutting individual blocks.
    selections = {direction: select_directional_holdout(blocks, direction) for direction in DIRECTIONS}

    # STEP 5 - Describe each population and check occupied-block connectivity.
    # Report evidence without choosing a design or declaring adequacy.
    summaries = [summarize_holdout_candidate(selections[d], d, len(cells)) for d in DIRECTIONS]
    candidates = pd.DataFrame(summaries)

    # STEP 6 - Map block support and candidate geography on common map extents.
    # Visual QA exposes awkward boundaries that population totals can conceal.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_spatial_blocks(blocks, cells)
    for summary in summaries:
        plot_holdout_candidate(blocks, cells, selections[summary["direction"]], summary)

    # STEP 7 - Save exploratory evidence for the human decision before script 16.
    # Only aggregate tables and graphics are persisted, never final assignments.
    save_exploration_report(cells, blocks, candidates, anchor, perf_counter() - started)


# %% Run script
if __name__ == "__main__":
    main()
