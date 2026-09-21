# %% Imports

from pathlib import Path

import duckdb


# %% Paths

ROOT = Path(__file__).resolve().parents[1]

MODELING = (
    ROOT
    / "data/processed/modeling/nm_wildfire_modeling_dataset.parquet"
)


# %% Find cells with missing aspect

missing_aspect = duckdb.sql(f"""
    SELECT
        cell_id,
        elevation_mean,
        elevation_std,
        slope_mean,
        slope_std,
        aspect_sin_mean,
        aspect_cos_mean,
        aspect_strength,
        annual_precip_mean,
        evt_dominant_class,
        burned_fraction,
        target
    FROM read_parquet('{MODELING.as_posix()}')
    WHERE
        aspect_sin_mean IS NULL
        OR aspect_cos_mean IS NULL
        OR aspect_strength IS NULL
    ORDER BY slope_mean
""").df()

print(missing_aspect)
# %%
# %% Summarize the missing-aspect cells

print("\nMISSING-ASPECT CELLS")
print("--------------------")

print(f"Rows: {len(missing_aspect)}")

print("\nSlope mean:")
print(missing_aspect["slope_mean"].describe())

print("\nSlope std:")
print(missing_aspect["slope_std"].describe())

print("\nElevation:")
print(missing_aspect["elevation_mean"].describe())

print("\nTargets:")
print(missing_aspect["target"].value_counts().sort_index())
# %%
# %% Compare missing-aspect cells with all modeling cells

comparison = duckdb.sql(f"""
    SELECT
        CASE
            WHEN aspect_sin_mean IS NULL
              OR aspect_cos_mean IS NULL
              OR aspect_strength IS NULL
            THEN 'missing aspect'
            ELSE 'aspect present'
        END AS aspect_group,

        COUNT(*) AS cells,
        AVG(slope_mean) AS mean_slope,
        MEDIAN(slope_mean) AS median_slope,
        MIN(slope_mean) AS min_slope,
        MAX(slope_mean) AS max_slope

    FROM read_parquet('{MODELING.as_posix()}')

    GROUP BY aspect_group
""").df()

print(comparison)
# %%
