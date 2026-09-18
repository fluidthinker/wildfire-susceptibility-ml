"""Join completed NM features into the eligible binary modeling population.

Run with --validate-only for the prepublication checks, then without arguments
to publish. No geometry or geospatial processing is needed: the completed
tables already summarize the authoritative full 1-km cells. cell_id and
burned_fraction are QA fields, not predictors. EVT codes remain categorical.
"""

# %% Imports
import argparse
import json
from pathlib import Path
from time import perf_counter

import duckdb


# %% Constants and paths
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
QA_PATH = ROOT / "outputs/modeling/modeling_dataset_qa.json"
SOURCES = {
    "terrain": ROOT / "data/processed/features/nm_terrain_features_1km.parquet",
    "prism": ROOT / "data/processed/features/nm_prism_features_1km.parquet",
    "landfire": ROOT / "data/processed/features/nm_landfire_features_1km.parquet",
    "mtbs": ROOT / "data/processed/targets/nm_mtbs_burned_fraction_1km.parquet",
}
TERRAIN = [
    "elevation_mean", "elevation_std", "slope_mean", "slope_std",
    "aspect_sin_mean", "aspect_cos_mean", "aspect_strength",
]
PREDICTORS = TERRAIN + [
    "annual_precip_mean", "evt_dominant_class", "evt_dominant_fraction",
]
COLUMNS = ["cell_id", *PREDICTORS, "burned_fraction", "target"]
SOURCE_COLUMNS = {
    "terrain": ["cell_id", *TERRAIN],
    "prism": ["cell_id", "annual_precip_mean"],
    "landfire": ["cell_id", "evt_dominant_class", "evt_dominant_fraction"],
    "mtbs": ["cell_id", "burned_area_m2", "burned_fraction"],
}
EXPECTED_ROWS = 314_920
EXPECTED_COUNTS = {"negative": 305_723, "ambiguous": 1_791, "positive": 7_406}
ELIGIBLE_ROWS = EXPECTED_COUNTS["negative"] + EXPECTED_COUNTS["positive"]
# Current completed terrain product has 39 undefined values per aspect column.
# This is a regression gate, not a scientific rule for imputing flat terrain.
ASPECT_MISSING_LIMIT = 39
ASPECT_COLUMNS = ["aspect_sin_mean", "aspect_cos_mean", "aspect_strength"]
INTEGER_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                 "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT"}
NUMERIC_TYPES = INTEGER_TYPES | {"FLOAT", "DOUBLE"}


# %% Helper functions
def validate_table_contract(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    expected_rows: int,
    exact_schema: bool = False,
) -> dict:
    """Confirm a table has the approved columns and one row per cell.

    Args:
        connection: In-memory DuckDB connection containing the named relation.
        table: Internal relation name, never supplied by external input.
        columns: Required columns in final output order.
        expected_rows: Approved population size.
        exact_schema: Whether extra or reordered columns must fail.

    Returns:
        Schema and key counts for the QA report.

    Raises:
        ValueError: If columns, types, row count, or cell keys violate the contract.
    """
    described_columns = connection.execute(f"DESCRIBE {table}").fetchall()
    schema = {row[0]: row[1] for row in described_columns}
    missing_columns = sorted(set(columns) - set(schema))
    if missing_columns or (exact_schema and list(schema) != columns):
        raise ValueError(f"{table}: unexpected schema {schema}; missing {missing_columns}")
    if schema["cell_id"] != "VARCHAR":
        raise ValueError(f"{table}: expected VARCHAR cell_id, got {schema['cell_id']}")
    for column in columns[1:]:
        allowed_types = INTEGER_TYPES if column in {"evt_dominant_class", "target"} else NUMERIC_TYPES
        if schema[column] not in allowed_types:
            raise ValueError(f"{table}.{column}: unexpected type {schema[column]}")

    rows, unique_ids, missing_ids, duplicates = connection.execute(f"""
        SELECT count(*), count(DISTINCT cell_id),
               count(*) FILTER (WHERE cell_id IS NULL),
               count(cell_id) - count(DISTINCT cell_id)
        FROM {table}
    """).fetchone()
    report = dict(rows=rows, unique_ids=unique_ids, missing_ids=missing_ids,
                  duplicate_ids=duplicates, schema=schema, schema_matches=True)
    wrong_rows = rows != expected_rows
    invalid_keys = missing_ids != 0 or duplicates != 0
    if wrong_rows or invalid_keys:
        raise ValueError(f"{table}: table contract failed: {report}")
    return report


def validate_feature_values(
    connection: duckdb.DuckDBPyConnection, table: str, columns: list[str]
) -> dict[str, int]:
    """Check numeric domains and preserve the known small aspect missingness.

    SQL NULL and floating NaN both count as missing. Infinities always fail;
    missing non-aspect values fail even in rows that will later be excluded.

    Args:
        connection: Active DuckDB connection.
        table: Internal relation to check.
        columns: Numeric fields to validate.

    Returns:
        Missing count for each checked field.

    Raises:
        ValueError: If missingness or numeric domains are unexpected.
    """
    missingness = {}
    for column in columns:
        missing, infinite = connection.execute(f"""
            SELECT count(*) FILTER (WHERE {column} IS NULL OR isnan({column})),
                   count(*) FILTER (WHERE isinf({column}))
            FROM {table}
        """).fetchone()
        missingness[column] = missing
        limit = ASPECT_MISSING_LIMIT if column in ASPECT_COLUMNS else 0
        if missing > limit or infinite:
            raise ValueError(f"{table}.{column}: missing={missing} (limit {limit}), infinite={infinite}")

    # These bounds enforce the approved products, without altering any values.
    domains = {
        "burned_fraction": "burned_fraction BETWEEN 0 AND 1",
        "burned_area_m2": "burned_area_m2 BETWEEN 0 AND 1000000",
        "evt_dominant_fraction": "evt_dominant_fraction > 0 AND evt_dominant_fraction <= 1",
        "annual_precip_mean": "annual_precip_mean >= 0",
        "elevation_std": "elevation_std >= 0",
        "slope_mean": "slope_mean BETWEEN 0 AND 90",
        "slope_std": "slope_std >= 0",
        "aspect_sin_mean": "aspect_sin_mean BETWEEN -1 AND 1",
        "aspect_cos_mean": "aspect_cos_mean BETWEEN -1 AND 1",
        "aspect_strength": "aspect_strength BETWEEN 0 AND 1.00000001",
        "target": "target IN (0, 1)",
    }
    for column in columns:
        if column not in domains:
            continue
        invalid = connection.execute(f"""
            SELECT count(*) FROM {table}
            WHERE {column} IS NOT NULL AND NOT isnan({column})
              AND NOT ({domains[column]})
        """).fetchone()[0]
        if invalid:
            raise ValueError(f"{table}.{column}: {invalid} values outside approved domain")
    return missingness


def validate_sources(connection: duckdb.DuckDBPyConnection) -> dict:
    """Validate the four completed Parquets against their known contracts.

    Args:
        connection: In-memory DuckDB connection.

    Returns:
        Per-source path, schema, key counts, and missingness.

    Raises:
        FileNotFoundError: If a completed source is absent.
        ValueError: If a source contract fails.
    """
    reports = {}
    for name, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        # The relation API safely handles paths and scans Parquet directly.
        connection.read_parquet(str(path)).create_view(name)
        report = validate_table_contract(connection, name, SOURCE_COLUMNS[name], EXPECTED_ROWS)
        report["missingness"] = validate_feature_values(connection, name, SOURCE_COLUMNS[name][1:])
        report["path"] = path.relative_to(ROOT).as_posix()
        reports[name] = report
    return reports


def validate_shared_cell_ids(connection: duckdb.DuckDBPyConnection) -> dict:
    """Require exactly matching cell sets before any inner join can hide keys.

    Args:
        connection: Connection with all validated source views.

    Returns:
        Missing and extra key counts relative to the completed terrain table.

    Raises:
        ValueError: If a source has any different cell IDs.
    """
    report = {}
    for name in ("prism", "landfire", "mtbs"):
        missing = connection.execute(f"""
            SELECT count(*) FROM (
                SELECT cell_id FROM terrain EXCEPT SELECT cell_id FROM {name}
            )
        """).fetchone()[0]
        extra = connection.execute(f"""
            SELECT count(*) FROM (
                SELECT cell_id FROM {name} EXCEPT SELECT cell_id FROM terrain
            )
        """).fetchone()[0]
        report[name] = dict(missing=missing, extra=extra)
        if missing or extra:
            raise ValueError(f"{name}: mismatched cell IDs: {report[name]}")
    return report


def validate_target_counts(connection: duckdb.DuckDBPyConnection) -> dict:
    """Check the three approved MTBS target groups against current inputs.

    Args:
        connection: Connection containing the validated MTBS source view.

    Returns:
        Actual negative, ambiguous, positive, and eligible counts.

    Raises:
        ValueError: If any count differs from the reviewed MTBS population.
    """
    values = connection.execute("""
        SELECT count(*) FILTER (WHERE burned_fraction = 0),
               count(*) FILTER (WHERE burned_fraction > 0 AND burned_fraction < 0.25),
               count(*) FILTER (WHERE burned_fraction >= 0.25)
        FROM mtbs
    """).fetchone()
    counts = dict(zip(EXPECTED_COUNTS, values))
    if counts != EXPECTED_COUNTS:
        differences = {name: counts[name] - expected for name, expected in EXPECTED_COUNTS.items()}
        raise ValueError(f"MTBS counts changed: actual={counts}; differences={differences}")
    counts["eligible"] = counts["negative"] + counts["positive"]
    counts["positive_percent"] = 100 * counts["positive"] / counts["eligible"]
    return counts


def build_join_query() -> str:
    """Describe the approved feature join without loading sources into Pandas.

    Returns:
        Declarative SQL selecting only the requested source fields.
    """
    # SQL describes the desired result. JOIN combines rows sharing cell_id;
    # the source views query Parquet directly without a persistent database.
    return """
        SELECT t.cell_id,
               t.elevation_mean, t.elevation_std,
               t.slope_mean, t.slope_std,
               t.aspect_sin_mean, t.aspect_cos_mean, t.aspect_strength,
               p.annual_precip_mean,
               l.evt_dominant_class, l.evt_dominant_fraction,
               m.burned_fraction
        FROM terrain AS t
        INNER JOIN prism AS p USING (cell_id)
        INNER JOIN landfire AS l USING (cell_id)
        INNER JOIN mtbs AS m USING (cell_id)
    """


def validate_modeling_dataset(connection: duckdb.DuckDBPyConnection, table: str) -> dict:
    """Confirm the eligible population retains the exact schema and target rule.

    Args:
        connection: Active DuckDB connection.
        table: Internal modeling relation, including a read-back Parquet view.

    Returns:
        Final schema, keys, target counts, and per-field missingness.

    Raises:
        ValueError: If modeling rows, values, or target assignments are invalid.
    """
    report = validate_table_contract(connection, table, COLUMNS, ELIGIBLE_ROWS, exact_schema=True)
    report["missingness"] = validate_feature_values(connection, table, COLUMNS[1:])
    target_counts = dict(connection.execute(
        f"SELECT target, count(*) FROM {table} GROUP BY target ORDER BY target"
    ).fetchall())
    expected = {0: EXPECTED_COUNTS["negative"], 1: EXPECTED_COUNTS["positive"]}
    if target_counts != expected:
        raise ValueError(f"{table}: unexpected target counts {target_counts}")
    invalid_targets = connection.execute(f"""
        SELECT count(*) FROM {table}
        WHERE NOT ((burned_fraction = 0 AND target = 0)
                   OR (burned_fraction >= 0.25 AND target = 1))
    """).fetchone()[0]
    if invalid_targets:
        raise ValueError(f"{table}: {invalid_targets} target assignments violate the rule")
    report["target_counts"] = target_counts
    return report


def write_modeling_parquet(connection: duckdb.DuckDBPyConnection, expected: dict) -> dict:
    """Publish the Parquet only after its temporary file passes read-back checks.

    Args:
        connection: Connection containing the validated modeling relation.
        expected: Pre-write validation report to compare with the stored file.

    Returns:
        Validation report for the written file.

    Raises:
        ValueError: If the stored file differs from the validated result.
    """
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    try:
        # Stable row order makes repeated builds straightforward to compare.
        connection.sql("SELECT * FROM modeling ORDER BY cell_id").write_parquet(
            str(partial), compression="zstd"
        )
        connection.read_parquet(str(partial)).create_view("written")
        report = validate_modeling_dataset(connection, "written")
        if report != expected:
            raise ValueError("Read-back QA differs from the pre-write modeling result")
        # Compare actual values and keys as well as aggregate QA counts.
        difference = connection.execute("""
            SELECT count(*) FROM (
                (SELECT * FROM modeling EXCEPT ALL SELECT * FROM written)
                UNION ALL
                (SELECT * FROM written EXCEPT ALL SELECT * FROM modeling)
            )
        """).fetchone()[0]
        if difference:
            raise ValueError(f"Read-back values differ in {difference} rows")
        connection.execute("DROP VIEW written")
        partial.replace(OUTPUT)
        return report
    finally:
        # A failed build never leaves a partial file masquerading as final output.
        partial.unlink(missing_ok=True)


def report_modeling_dataset(report: dict, publish: bool) -> None:
    """Print QA and optionally preserve a compact record beside modeling outputs.

    Args:
        report: JSON-compatible run results.
        publish: Whether a final dataset was successfully published.
    """
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if publish:
        QA_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial = QA_PATH.with_suffix(".json.part")
        try:
            partial.write_text(text + "\n", encoding="utf-8")
            partial.replace(QA_PATH)
        finally:
            partial.unlink(missing_ok=True)


# %% Main workflow
def main(validate_only: bool = False) -> None:
    """Build the statewide modeling population from completed feature tables.

    Args:
        validate_only: Run all query checks without writing either artifact.
    """
    started = perf_counter()
    with duckdb.connect(":memory:") as connection:
        # STEP 1 - Validate completed Parquet contracts.
        # Earlier scripts explored the data; enforce those known contracts here.
        sources = validate_sources(connection)

        # STEP 2 - Compare all source key sets before joining.
        # An inner join must not silently discard missing cells.
        shared_keys = validate_shared_cell_ids(connection)

        # STEP 3 - Join features by authoritative cell_id directly in DuckDB.
        # Verify the complete population before the target filter is applied.
        connection.sql(build_join_query()).create_view("joined")
        joined = validate_table_contract(connection, "joined", COLUMNS[:-1], EXPECTED_ROWS, True)

        # STEP 4 - Derive binary target and exclude ambiguous MTBS edge cells.
        # CASE assigns labels; WHERE removes only cells with 0 < fraction < 0.25.
        counts = validate_target_counts(connection)
        connection.sql("""
            SELECT *, CASE WHEN burned_fraction = 0 THEN 0
                           WHEN burned_fraction >= 0.25 THEN 1 END AS target
            FROM joined
            WHERE burned_fraction = 0 OR burned_fraction >= 0.25
        """).create_view("modeling")

        # STEP 5 - Validate schema, labels, and missingness before writing.
        # Preserve all eligible cells, original class codes, and missing aspect.
        final = validate_modeling_dataset(connection, "modeling")

        # STEP 6 - Read back a temporary Parquet before atomic publication.
        # The validation-only run stops short of writing any artifacts.
        if not validate_only:
            final = write_modeling_parquet(connection, final)

        # STEP 7 - Report population composition and reproducible QA.
        # Explicit feature roles help prevent target leakage in later modeling.
        report = {
            "duckdb_version": duckdb.__version__, "validation_only": validate_only,
            "sources": sources, "shared_keys": shared_keys,
            "all_cell_id_sets_match": True, "joined": joined, "target": counts,
            "final": final, "predictors": PREDICTORS,
            "categorical_predictors": ["evt_dominant_class"],
            "qa_only_columns": ["cell_id", "burned_fraction"],
            "response": "target", "unexpected_issues": [],
            "output": OUTPUT.relative_to(ROOT).as_posix(),
            "file_size_bytes": None if validate_only else OUTPUT.stat().st_size,
            "runtime_seconds": round(perf_counter() - started, 3),
        }
        report_modeling_dataset(report, publish=not validate_only)


# %% Run script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Check inputs and queries without publication.")
    main(validate_only=parser.parse_args().validate_only)
