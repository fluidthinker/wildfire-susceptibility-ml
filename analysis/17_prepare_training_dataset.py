"""Prepare raw training rows using the frozen assignments from script 16.

No features or folds are computed here. The saved eastern final test remains
outside this artifact, and target-derived burned_fraction is excluded.
"""

# %% Imports
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
SPLITS_PATH = ROOT / "data/processed/modeling/nm_modeling_splits.parquet"
OUTPUT_PATH = ROOT / "data/processed/modeling/nm_training_dataset.parquet"
QA_PATH = ROOT / "outputs/modeling/training_dataset_qa.json"
EXPECTED_INPUT_ROWS = 313_129
EXPECTED_TRAIN_ROWS = 252_569
PREDICTORS = [
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "aspect_sin_mean", "aspect_cos_mean", "aspect_strength",
    "annual_precip_mean", "evt_dominant_class", "evt_dominant_fraction",
]
ASPECT_COLUMNS = ["aspect_sin_mean", "aspect_cos_mean", "aspect_strength"]
# User-approved correction: one of the 39 statewide missing-aspect cells is
# final_test cell_018293, so only 38 belong to the raw training population.
EXPECTED_ASPECT_MISSING = 38
MODELING_COLUMNS = ["cell_id", *PREDICTORS, "burned_fraction", "target"]
FOLD_COLUMNS = ["random_cv_fold", "spatial_cv_fold"]
SPLIT_COLUMNS = ["cell_id", "split", "spatial_block_id", *FOLD_COLUMNS]
OUTPUT_COLUMNS = ["cell_id", "target", *PREDICTORS, "spatial_block_id", *FOLD_COLUMNS]
EXPECTED_FOLDS = {
    "random_cv_fold": [(50514, 1317), (50514, 1317), (50514, 1317), (50514, 1317), (50513, 1317)],
    "spatial_cv_fold": [(50435, 1376), (50605, 1321), (50454, 1307), (50408, 1280), (50667, 1301)],
}


# %% Helper functions
def validate_table_contract(
    table: pd.DataFrame, columns: list[str], expected_rows: int, label: str
) -> None:
    """Check the known schema and require exactly one row per eligible cell.

    Args:
        table: Loaded source or proposed training table.
        columns: Approved columns in their expected order.
        expected_rows: Approved population size.
        label: Table name for clear failure messages.

    Raises:
        ValueError: If schema, row count, or cell keys differ from the contract.
    """
    if list(table.columns) != columns:
        raise ValueError(f"{label}: expected columns {columns}; found {list(table.columns)}")
    wrong_count = len(table) != expected_rows
    missing_ids = int(table.cell_id.isna().sum())
    duplicate_ids = int(table.cell_id.duplicated().sum())
    if wrong_count or missing_ids or duplicate_ids:
        raise ValueError(f"{label}: rows={len(table)}, missing IDs={missing_ids}, duplicates={duplicate_ids}")


def validate_input_contracts(modeling: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Confirm compatible eligible populations and frozen train/test membership.

    Args:
        modeling: Completed raw predictors and target table.
        splits: Authoritative split assignments from script 16.

    Raises:
        ValueError: If source contracts, key sets, or split isolation fail.
    """
    validate_table_contract(modeling, MODELING_COLUMNS, EXPECTED_INPUT_ROWS, "Modeling input")
    validate_table_contract(splits, SPLIT_COLUMNS, EXPECTED_INPUT_ROWS, "Split input")
    modeling_ids = pd.Index(modeling.cell_id)
    split_ids = pd.Index(splits.cell_id)
    missing = modeling_ids.difference(split_ids)
    extra = split_ids.difference(modeling_ids)
    if len(missing) or len(extra):
        raise ValueError(f"Source key sets differ: {len(missing)} absent from splits, {len(extra)} extra")
    if not modeling.target.isin([0, 1]).all() or int(modeling.target.sum()) != 7406:
        raise ValueError("Modeling target contract differs from the approved binary population")

    # Validate existing membership; never reconstruct the geographic design.
    if not splits.split.isin(["train", "final_test"]).all():
        raise ValueError("Split labels must be train or final_test")
    if splits.spatial_block_id.isna().any():
        raise ValueError("Every split row must retain its saved spatial block")
    train = splits.loc[splits.split.eq("train")]
    test = splits.loc[splits.split.eq("final_test")]
    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != 60560:
        raise ValueError(f"Unexpected train/test counts: {len(train)}/{len(test)}")
    if test[FOLD_COLUMNS].notna().any().any():
        raise ValueError("Final-test rows have unexpected CV assignments")
    if set(train.spatial_block_id) & set(test.spatial_block_id):
        raise ValueError("A spatial block crosses train and final_test")


def join_modeling_and_splits(modeling: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Attach saved experiment membership by a validated one-to-one cell ID join.

    Args:
        modeling: Validated modeling population.
        splits: Validated saved assignments with exactly matching keys.

    Returns:
        All eligible modeling rows with their original split and fold values.

    Raises:
        ValueError: If the join loses rows or is not one-to-one.
    """
    joined = modeling.merge(splits, on="cell_id", how="inner", validate="one_to_one")
    if len(joined) != EXPECTED_INPUT_ROWS:
        raise ValueError(f"Pre-filter join returned {len(joined)} rows")
    return joined


def summarize_folds(training: pd.DataFrame, column: str) -> pd.DataFrame:
    """Describe the saved fold populations without assigning any new folds.

    Args:
        training: Raw training table containing target and saved assignments.
        column: Random or spatial fold column.

    Returns:
        Counts and positive prevalence ordered by fold label.
    """
    summary = training.groupby(column, sort=True).agg(
        rows=("cell_id", "size"), positives=("target", "sum"),
        blocks=("spatial_block_id", "nunique"),
    )
    summary["positive_percent"] = 100 * summary.positives / summary.rows
    return summary


def validate_training_dataset(training: pd.DataFrame) -> dict:
    """Require the approved raw training schema, labels, missingness, and folds.

    Args:
        training: Proposed or read-back training-only table.

    Returns:
        Per-predictor missing counts and saved-fold population summaries.

    Raises:
        ValueError: If any approved training contract differs.
    """
    validate_table_contract(training, OUTPUT_COLUMNS, EXPECTED_TRAIN_ROWS, "Training dataset")
    target_counts = training.target.value_counts().to_dict()
    if target_counts != {0: 245984, 1: 6585}:
        raise ValueError(f"Unexpected training target counts: {target_counts}")

    # Missingness is measured, not repaired. The corrected aspect contract is
    # 38 per column; all other raw predictors must remain fully observed.
    missingness = training[PREDICTORS].isna().sum().to_dict()
    for column in PREDICTORS:
        expected = EXPECTED_ASPECT_MISSING if column in ASPECT_COLUMNS else 0
        if missingness[column] != expected:
            raise ValueError(f"{column}: {missingness[column]} missing; expected {expected}")
        if not pd.api.types.is_numeric_dtype(training[column]):
            raise ValueError(f"{column}: expected the original numeric storage type")
        if np.isinf(training[column].dropna().to_numpy()).any():
            raise ValueError(f"{column}: infinite values are unexpected")
    if not pd.api.types.is_integer_dtype(training.evt_dominant_class):
        raise ValueError("EVT must retain its integer categorical identifier")

    # Script 16 owns the assignments. Validate labels and known counts without
    # invoking a splitter or changing even a single fold membership.
    if training.spatial_block_id.isna().any() or training.spatial_block_id.nunique() != 120:
        raise ValueError("Training must retain 120 nonmissing spatial blocks")
    fold_reports = {}
    for column in FOLD_COLUMNS:
        invalid_type = not pd.api.types.is_integer_dtype(training[column])
        missing_folds = training[column].isna().any()
        wrong_labels = set(training[column].dropna()) != set(range(5))
        if invalid_type or missing_folds or wrong_labels:
            raise ValueError(f"{column}: expected nonmissing integer labels 0-4")
        summary = summarize_folds(training, column)
        actual = list(summary[["rows", "positives"]].itertuples(index=False, name=None))
        if actual != EXPECTED_FOLDS[column]:
            raise ValueError(f"{column}: current fold counts {actual} differ from script 16")
        fold_reports[column] = summary.reset_index().to_dict(orient="records")
    if not training.groupby("spatial_block_id").spatial_cv_fold.nunique().eq(1).all():
        raise ValueError("A spatial block crosses spatial CV folds")
    return {"missingness": missingness, "folds": fold_reports}


def validate_source_preservation(
    training: pd.DataFrame, modeling: pd.DataFrame, splits: pd.DataFrame
) -> None:
    """Verify training keys and every raw value against the authoritative sources.

    Args:
        training: Selected training artifact to check.
        modeling: Original predictors and targets.
        splits: Original saved split and fold assignments.

    Raises:
        ValueError: If training membership differs or any final-test row appears.
        AssertionError: If a feature, target, block, fold, or dtype changed.
    """
    expected_ids = pd.Index(splits.loc[splits.split.eq("train"), "cell_id"]).sort_values()
    actual = training.set_index("cell_id").sort_index()
    if not actual.index.equals(expected_ids):
        raise ValueError("Training keys do not exactly match saved train membership")
    test_ids = splits.loc[splits.split.eq("final_test"), "cell_id"]
    if actual.index.isin(test_ids).any():
        raise ValueError("Final-test rows must be absent from the training artifact")
    # Exact comparison includes NaN positions and dtypes. This protects raw
    # feature precision and categorical codes as well as saved fold membership.
    raw_columns = ["target", *PREDICTORS]
    saved_columns = ["spatial_block_id", *FOLD_COLUMNS]
    expected_raw = modeling.set_index("cell_id").loc[actual.index, raw_columns]
    expected_saved = splits.set_index("cell_id").loc[actual.index, saved_columns]
    pd.testing.assert_frame_equal(actual[raw_columns], expected_raw, check_exact=True)
    pd.testing.assert_frame_equal(actual[saved_columns], expected_saved, check_exact=True)


def write_training_parquet(training: pd.DataFrame) -> None:
    """Publish only after the temporary Parquet passes validation and exact comparison.

    Args:
        training: Validated raw training table in deterministic cell_id order.

    Raises:
        ValueError: If read-back fails the training contract.
        AssertionError: If stored values or dtypes differ from the validated table.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT_PATH.with_suffix(".parquet.part")
    try:
        training.to_parquet(partial, index=False, compression="zstd")
        written = pd.read_parquet(partial)
        validate_training_dataset(written)
        pd.testing.assert_frame_equal(written, training, check_exact=True)
        partial.replace(OUTPUT_PATH)
    finally:
        # A failed write must not leave an incomplete product under a final name.
        partial.unlink(missing_ok=True)


def report_training_dataset(training: pd.DataFrame, qa: dict, runtime: float) -> None:
    """Print and save a compact record of the successfully published population.

    Args:
        training: Validated and published raw training data.
        qa: Missingness and fold summaries from validation.
        runtime: Elapsed seconds through publication, excluding report writing.
    """
    report = {
        "source_paths": [MODELING_PATH.relative_to(ROOT).as_posix(), SPLITS_PATH.relative_to(ROOT).as_posix()],
        "modeling_input_rows": EXPECTED_INPUT_ROWS, "split_input_rows": EXPECTED_INPUT_ROWS,
        "joined_rows": EXPECTED_INPUT_ROWS, "shared_key_sets_match": True,
        "training_rows": len(training), "unique_cell_ids": training.cell_id.nunique(),
        "missing_cell_ids": 0, "duplicate_cell_ids": 0,
        "positives": int(training.target.sum()), "negatives": int(training.target.eq(0).sum()),
        "positive_percent": 100 * float(training.target.mean()), **qa,
        "predictors": PREDICTORS, "categorical_predictors": ["evt_dominant_class"],
        "metadata_columns": ["cell_id", "spatial_block_id", *FOLD_COLUMNS],
        "schema": {column: str(dtype) for column, dtype in training.dtypes.items()},
        "final_test_rows_absent": True, "burned_fraction_absent": True,
        "raw_values_and_dtypes_preserved": True, "saved_folds_preserved_not_recomputed": True,
        "aspect_contract_note": "38 training missing values per aspect column; one of 39 statewide missing-aspect cells is final_test cell_018293",
        "output_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "file_size_bytes": OUTPUT_PATH.stat().st_size,
        "runtime_seconds": round(runtime, 3), "unexpected_issues": [],
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
    """Prepare and publish the raw training population for later ML experiments."""
    started = perf_counter()
    # STEP 1 - Load completed predictors/target and the frozen split artifact.
    # No geometry, source processing, or fold-generation code is needed here.
    modeling = pd.read_parquet(MODELING_PATH)
    splits = pd.read_parquet(SPLITS_PATH)

    # STEP 2 - Validate source schemas, keys, and approved split membership.
    # Exact matching keys prevent an inner join from silently losing cells.
    validate_input_contracts(modeling, splits)

    # STEP 3 - Attach saved assignments by a validated one-to-one join.
    # Check the full eligible population before filtering training rows.
    joined = join_modeling_and_splits(modeling, splits)

    # STEP 4 - Keep only saved training membership.
    # The final eastern test remains outside downstream model development.
    training_rows = joined.loc[joined.split.eq("train")]

    # STEP 5 - Select the explicit raw columns and stable cell order.
    # burned_fraction is target-derived QA; folds and IDs are metadata, not predictors.
    training = training_rows[OUTPUT_COLUMNS].sort_values("cell_id").reset_index(drop=True)

    # STEP 6 - Check the corrected missingness contract and unchanged source values.
    # Preserve 38 missing values per aspect column; do not impute or encode features.
    qa = validate_training_dataset(training)
    validate_source_preservation(training, modeling, splits)

    # STEP 7 - Validate a temporary Parquet before publishing the final filename.
    # Read-back equality protects every raw value and the saved CV assignments.
    write_training_parquet(training)

    # STEP 8 - Record the exact population and both saved CV compositions.
    # This documents the reusable input boundary for later experiments.
    report_training_dataset(training, qa, perf_counter() - started)
    print("Reporting training dataset...")
    report_function(training)

# %% 
def report_function(training_rows):
    print("Training rows shape:", training_rows.shape)
  

    print("Training rows head:\n", training_rows.head())

    print("First training row:\n", training_rows.iloc[0])

    print("Target value counts:\n", training_rows["target"].value_counts())

    print("Crosstab of random CV fold and target:\n", pd.crosstab(
         training_rows["random_cv_fold"],
         training_rows["target"],
     ))

    


# %% Run script
if __name__ == "__main__":
    main()


# %% Report training rows

# %%
