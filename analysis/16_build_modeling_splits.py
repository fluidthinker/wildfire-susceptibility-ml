"""Freeze the approved eastern holdout and two training-only CV assignments.

The approved 15B design uses 50-km blocks and eastern columns 10 through 12.
This script validates that contract, without exploring alternative regions.
Later models must join this artifact by cell_id and reuse its saved folds.
"""

# %% Imports
import hashlib
import json
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

from wildfire_susceptibility.spatial_blocks import assign_spatial_blocks, count_block_components


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
GRID_PATH = ROOT / "data/processed/grids/nm_analysis_grid_1km.gpkg"
OUTPUT_PATH = ROOT / "data/processed/modeling/nm_modeling_splits.parquet"
QA_PATH = ROOT / "outputs/modeling/modeling_splits_qa.json"
BLOCK_SIZE_M = 50_000
APPROVED_ANCHOR = (-1_250_000.0, 950_000.0)
EAST_MIN_COLUMN = 10  # Approved west edge: -750,000 m in EPSG:5070.
EXPECTED_ROWS = 313_129
EXPECTED_GRID_ROWS = 314_920
EXPECTED_POSITIVES = 7_406
EXPECTED_TEST_ROWS = 60_560
EXPECTED_TEST_POSITIVES = 821
EXPECTED_BLOCKS = 153
EXPECTED_TEST_BLOCKS = 33
N_FOLDS = 5
RANDOM_SEED = 42
OUTPUT_COLUMNS = ["cell_id", "split", "spatial_block_id", "random_cv_fold", "spatial_cv_fold"]
FOLD_COLUMNS = ["random_cv_fold", "spatial_cv_fold"]


# %% Helper functions
def validate_cell_keys(table: pd.DataFrame, expected_rows: int, label: str) -> None:
    """Require the known population size and one nonmissing key per cell.

    Args:
        table: Input or output table with cell_id.
        expected_rows: Approved number of rows.
        label: Name used in a contract failure message.

    Raises:
        ValueError: If keys or population size are unexpected.
    """
    wrong_count = len(table) != expected_rows
    missing_keys = table.cell_id.isna().any()
    duplicate_keys = table.cell_id.duplicated().any()
    if wrong_count or missing_keys or duplicate_keys:
        raise ValueError(f"{label}: rows={len(table)}, missing keys={missing_keys}, "
                         f"duplicate keys={duplicate_keys}")


def load_modeling_population() -> pd.DataFrame:
    """Read eligible IDs and labels in stable order for reproducible folds.

    Returns:
        Approved binary modeling population sorted by cell_id.

    Raises:
        ValueError: If the population or target contract has changed.
    """
    population = pd.read_parquet(MODELING_PATH, columns=["cell_id", "target"])
    validate_cell_keys(population, EXPECTED_ROWS, "Modeling population")
    if not population.target.isin([0, 1]).all():
        raise ValueError("Modeling target must contain only nonmissing 0/1 labels")
    if int(population.target.sum()) != EXPECTED_POSITIVES:
        raise ValueError("Modeling positive count differs from the approved population")
    return population.sort_values("cell_id").reset_index(drop=True)


def load_authoritative_grid() -> gpd.GeoDataFrame:
    """Load authoritative geometry and enforce its approved 1-km grid contract.

    Returns:
        Original complete EPSG:5070 cells; geometry is never repaired or clipped.

    Raises:
        ValueError: If keys, CRS, or complete square geometry are invalid.
    """
    grid = gpd.read_file(GRID_PATH, layer="analysis_grid", columns=["cell_id"])
    validate_cell_keys(grid, EXPECTED_GRID_ROWS, "Authoritative grid")
    if grid.crs is None or grid.crs.to_epsg() != 5070:
        raise ValueError("Authoritative grid must already use EPSG:5070")
    invalid = grid.geometry.isna() | grid.geometry.is_empty | ~grid.geometry.is_valid
    if invalid.any():
        raise ValueError(f"Invalid or missing grid geometry: {invalid.sum()} cells")
    bounds = grid.bounds
    correct_width = np.isclose(bounds.maxx - bounds.minx, 1000, rtol=0, atol=0.001)
    correct_height = np.isclose(bounds.maxy - bounds.miny, 1000, rtol=0, atol=0.001)
    correct_area = np.isclose(grid.area, 1_000_000, rtol=0, atol=0.01)
    if not (correct_width & correct_height & correct_area).all():
        raise ValueError("Authoritative geometry must preserve complete 1-km squares")
    return grid


def join_eligible_geometry(population: pd.DataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach geometry only to eligible modeling keys before block assignment.

    Args:
        population: Validated modeling IDs and targets.
        grid: Validated full authoritative grid.

    Returns:
        Eligible geometries and labels in stable cell_id order.

    Raises:
        ValueError: If any eligible key lacks geometry.
    """
    missing = ~population.cell_id.isin(grid.cell_id)
    if missing.any():
        raise ValueError(f"{missing.sum()} eligible IDs lack authoritative geometry")
    cells = grid.merge(population, on="cell_id", how="inner", validate="one_to_one")
    return cells.sort_values("cell_id").reset_index(drop=True)


def assign_final_test_region(cells: gpd.GeoDataFrame, anchor: tuple[float, float]) -> pd.DataFrame:
    """Apply the approved eastern block cutoff and check the 15B regression counts.

    The exploratory closest-to-20% east rule selected column 10 onward. Freeze
    that boundary here rather than moving it if source populations change.

    Args:
        cells: Eligible cells with the shared 15B block positions and labels.
        anchor: Snapped origin returned by the shared block helper.

    Returns:
        Nonspatial working table containing labels, block IDs, and train/test.

    Raises:
        ValueError: If anchor, counts, extent, or contiguity differ from approval.
    """
    if anchor != APPROVED_ANCHOR:
        raise ValueError(f"Approved anchor {APPROVED_ANCHOR} changed to {anchor}")
    is_test = cells.block_col >= EAST_MIN_COLUMN
    test = cells.loc[is_test]
    blocks = test[["block_id", "block_row", "block_col"]].drop_duplicates()
    actual = (len(test), int(test.target.sum()), len(blocks), cells.block_id.nunique())
    expected = (EXPECTED_TEST_ROWS, EXPECTED_TEST_POSITIVES, EXPECTED_TEST_BLOCKS, EXPECTED_BLOCKS)
    if actual != expected:
        raise ValueError(f"Eastern design regression failed (rows, positives, test/all blocks): "
                         f"actual={actual}; expected={expected}")
    # Approved occupied block extents: x [-750000, -600000], y [1000000, 1600000].
    positions = (blocks.block_col.min(), blocks.block_col.max(),
                 blocks.block_row.min(), blocks.block_row.max())
    if positions != (10, 12, 1, 12) or count_block_components(blocks) != 1:
        raise ValueError("Eastern block extent or shared-edge contiguity changed")
    working = pd.DataFrame(cells[["cell_id", "target", "block_id"]]).rename(
        columns={"block_id": "spatial_block_id"}
    )
    working["split"] = np.where(is_test, "final_test", "train")
    return working


def assign_random_cv_folds(working: pd.DataFrame) -> pd.Series:
    """Assign stratified random folds within training rows and leave test rows null.

    Args:
        working: Stable cell_id-ordered table with target and train/test labels.

    Returns:
        Nullable integer folds aligned to the working table's index.
    """
    train = working.loc[working.split == "train"].sort_values("cell_id")
    folds = pd.Series(pd.NA, index=working.index, dtype="Int64")
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    # sklearn returns positional indices within train. Translate them back to
    # working-table indices so interleaved final-test rows never receive folds.
    for fold, (_, validation_positions) in enumerate(splitter.split(train.cell_id, train.target)):
        folds.loc[train.index[validation_positions]] = fold
    return folds


def assign_spatial_cv_folds(working: pd.DataFrame) -> pd.Series:
    """Assign whole training blocks greedily to balance cells and positives.

    Blocks with most positives go first, then most cells, then lexical block ID.
    Each block goes to the fold producing the smallest increase in squared
    normalized cell and positive loads, with equal weight for both objectives.
    Exact score ties prefer the lowest fold number. No search or tuning occurs.
    Folds may contain separated blocks; contiguity is required only for test.

    Args:
        working: Eligible rows with targets, blocks, and train/test labels.

    Returns:
        Nullable integer folds aligned to all rows; test rows remain null.

    Raises:
        ValueError: If training has too few blocks or no positive observations.
    """
    train = working.loc[working.split == "train"]
    blocks = train.groupby("spatial_block_id", as_index=False).agg(
        cells=("cell_id", "size"), positives=("target", "sum")
    ).sort_values(["positives", "cells", "spatial_block_id"], ascending=[False, False, True])
    if len(blocks) < N_FOLDS or blocks.positives.sum() == 0:
        raise ValueError("Training needs at least five blocks and positive cases")

    # Normalize by ideal per-fold counts so abundant negatives do not overwhelm
    # the positive-count objective. A block is indivisible throughout allocation.
    ideal_load = np.array([blocks.cells.sum(), blocks.positives.sum()], dtype=float) / N_FOLDS
    loads = np.zeros((N_FOLDS, 2), dtype=float)
    assignments = {}
    for block in blocks.itertuples(index=False):
        added = np.array([block.cells, block.positives], dtype=float)
        score_increase = (((loads + added) / ideal_load) ** 2 - (loads / ideal_load) ** 2).sum(axis=1)
        fold = int(np.argmin(score_increase))
        assignments[block.spatial_block_id] = fold
        loads[fold] += added
    folds = pd.Series(pd.NA, index=working.index, dtype="Int64")
    folds.loc[train.index] = train.spatial_block_id.map(assignments).astype("Int64")
    return folds


def validate_split_assignments(splits: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Check full coverage, approved holdout membership, and CV isolation.

    Args:
        splits: Proposed or read-back final split artifact.
        reference: Geometry-derived working table with approved membership/labels.

    Raises:
        ValueError: If schema, keys, membership, fold coverage, or grouping fails.
    """
    if list(splits.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Unexpected output columns: {list(splits.columns)}")
    validate_cell_keys(splits, EXPECTED_ROWS, "Split artifact")
    if set(splits.cell_id) != set(reference.cell_id):
        raise ValueError("Split artifact keys do not match eligible modeling keys")
    if not splits.split.isin(["train", "final_test"]).all() or splits.spatial_block_id.isna().any():
        raise ValueError("Every row needs an approved split label and spatial block")

    # Compare membership by key, not row position; this also catches a table with
    # plausible counts but the wrong eastern cells or incorrect block IDs.
    expected_membership = reference.set_index("cell_id")[["split", "spatial_block_id"]].sort_index()
    actual_membership = splits.set_index("cell_id")[["split", "spatial_block_id"]].sort_index()
    if not actual_membership.equals(expected_membership):
        raise ValueError("Output membership differs from the approved geometry-derived design")
    checked = splits.merge(reference[["cell_id", "target"]], on="cell_id", validate="one_to_one")
    train = checked.loc[checked.split == "train"]
    test = checked.loc[checked.split == "final_test"]
    wrong_population = len(train) != EXPECTED_ROWS - EXPECTED_TEST_ROWS or len(test) != EXPECTED_TEST_ROWS
    wrong_positives = int(test.target.sum()) != EXPECTED_TEST_POSITIVES
    if wrong_population or wrong_positives:
        raise ValueError("Approved training/test population counts changed")
    if set(train.spatial_block_id) & set(test.spatial_block_id):
        raise ValueError("A spatial block crosses training and final test")
    if test[FOLD_COLUMNS].notna().any().any():
        raise ValueError("Final-test rows must have NULL in both CV columns")

    for column in FOLD_COLUMNS:
        if not pd.api.types.is_integer_dtype(splits[column].dtype):
            raise ValueError(f"{column} must use a nullable integer dtype")
        missing_folds = train[column].isna().any()
        wrong_labels = set(train[column].dropna()) != set(range(N_FOLDS))
        if missing_folds or wrong_labels:
            raise ValueError(f"Training {column} must assign every row to exactly one fold 0-4")
        if not train.groupby(column).target.nunique().eq(2).all():
            raise ValueError(f"Every {column} validation fold must contain both target classes")
    if not train.groupby("spatial_block_id").spatial_cv_fold.nunique().eq(1).all():
        raise ValueError("A training block appears in multiple spatial CV folds")


def summarize_population(table: pd.DataFrame, group: str) -> list[dict]:
    """Count cells, blocks, and positive prevalence for split or CV groups.

    Args:
        table: Working rows containing target and the requested group field.
        group: Split or fold column to summarize.

    Returns:
        JSON-compatible group summaries.
    """
    summary = table.groupby(group).agg(
        rows=("cell_id", "size"), blocks=("spatial_block_id", "nunique"),
        positives=("target", "sum"),
    ).reset_index()
    summary["negatives"] = summary.rows - summary.positives
    summary["positive_percent"] = 100 * summary.positives / summary.rows
    return summary.to_dict(orient="records")


def write_split_parquet(splits: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Publish the assignments only after a temporary Parquet passes read-back QA.

    Args:
        splits: Validated final columns, without geometry or target.
        reference: Approved membership used to validate the stored artifact.

    Raises:
        ValueError: If read-back violates the split contract.
        AssertionError: If any stored value or dtype differs from the original.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT_PATH.with_suffix(".parquet.part")
    try:
        splits.to_parquet(partial, index=False, compression="zstd")
        written = pd.read_parquet(partial)
        validate_split_assignments(written, reference)
        pd.testing.assert_frame_equal(written, splits)
        partial.replace(OUTPUT_PATH)
    finally:
        partial.unlink(missing_ok=True)


def report_split_design(working: pd.DataFrame, runtime: float) -> None:
    """Save the frozen experiment's method and population summaries for review.

    Args:
        working: Labels and validated split/fold assignments.
        runtime: Elapsed seconds through successful Parquet publication.
    """
    train = working.loc[working.split == "train"]
    report = {
        "design": "nm-east-50km-five-fold-v1", "rows": len(working),
        "unique_cell_ids": working.cell_id.nunique(), "missing_cell_ids": 0, "duplicate_cell_ids": 0,
        "occupied_blocks": working.spatial_block_id.nunique(),
        "block_size_m": BLOCK_SIZE_M, "crs": "EPSG:5070", "anchor_m": APPROVED_ANCHOR,
        "east_min_column": EAST_MIN_COLUMN, "east_contiguous": True,
        "random_seed": RANDOM_SEED, "random_method": "StratifiedKFold, shuffle=True, sorted cell_id",
        "spatial_method": "Greedy whole-block assignment minimizing squared normalized cell/positive loads; equal weights; deterministic ties",
        "spatial_fold_contiguity_required": False,
        "spatial_limitation": "50 km is the approved pragmatic block size, not an estimated independence distance; no buffer is applied",
        "split_summary": summarize_population(working, "split"),
        "random_cv": summarize_population(train, "random_cv_fold"),
        "spatial_cv": summarize_population(train, "spatial_cv_fold"),
        "no_blocks_cross_spatial_folds": True, "no_blocks_cross_train_test": True,
        "final_test_cv_all_null": True, "unexpected_issues": [],
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "inputs": [MODELING_PATH.relative_to(ROOT).as_posix(), GRID_PATH.relative_to(ROOT).as_posix()],
        "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "output_columns": OUTPUT_COLUMNS, "file_size_bytes": OUTPUT_PATH.stat().st_size,
        "output_sha256": hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest(),
        "runtime_seconds": round(runtime, 3),
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = QA_PATH.with_suffix(".json.part")
    try:
        partial.write_text(text + "\n", encoding="utf-8")
        partial.replace(QA_PATH)
    finally:
        partial.unlink(missing_ok=True)
    print(text)


# %% Main workflow
def main() -> None:
    """Build and safely publish the approved experimental split assignments."""
    started = perf_counter()
    # STEP 1 - Load eligible IDs, labels, and authoritative geometry.
    # Only geometry needed to reproduce the approved spatial design is joined.
    population = load_modeling_population()
    grid = load_authoritative_grid()
    cells = join_eligible_geometry(population, grid)

    # STEP 2 - Reuse the exact block implementation shared with script 15B.
    # Full-grid bounds keep the snapped origin independent of eligibility.
    cells, anchor = assign_spatial_blocks(cells, grid.total_bounds, BLOCK_SIZE_M)

    # STEP 3 - Freeze the approved eastern cutoff and regression-check its population.
    # No alternative candidate or new geographic threshold is considered.
    working = assign_final_test_region(cells, anchor)

    # STEP 4 - Assign stratified random CV using only the training population.
    # Saved folds give later models the same conventional validation comparison.
    working["random_cv_fold"] = assign_random_cv_folds(working)

    # STEP 5 - Assign spatial CV by whole training blocks.
    # Balance cell and positive counts without breaking geographic groups.
    working["spatial_cv_fold"] = assign_spatial_cv_folds(working)

    # STEP 6 - Enforce coverage, holdout isolation, and one spatial fold per block.
    # Remove working labels and geometry from the reusable artifact.
    splits = working[OUTPUT_COLUMNS].copy()
    validate_split_assignments(splits, working)

    # STEP 7 - Read back and validate a temporary file before atomic publication.
    # All later model scripts must reuse the resulting assignments by cell_id.
    write_split_parquet(splits, working)

    # STEP 8 - Report frozen train/test populations and both CV schemes.
    # A compact QA record documents the decisions and implementation versions.
    report_split_design(working, perf_counter() - started)


# %% Run script
if __name__ == "__main__":
    main()
