# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt

from wildfire_susceptibility.boundary import fetch_state_boundary
from wildfire_susceptibility.grid import create_analysis_grid

# %% Case-study parameters
STATE_FIPS = "35"
TARGET_CRS = "EPSG:5070"
TIGER_YEAR = 2025
CELL_SIZE_M = 1000

# %% Fetch prepared state boundary
state = fetch_state_boundary(
    state_fips=STATE_FIPS,
    year=TIGER_YEAR,
    target_crs=TARGET_CRS,
)

# %% Build 1-km analysis grid
grid = create_analysis_grid(state, cell_size=CELL_SIZE_M)

# %% Inspect grid
print(f"Candidate cells: {grid.attrs['candidate_cell_count']:,}")
print(f"Retained cells: {len(grid):,}")
print(f"Grid CRS: {grid.crs}")
print(f"Unique cell IDs: {grid['cell_id'].is_unique}")
print(grid.head())

# %% Visual inspection
state_name = state.iloc[0]["NAME"]
fig, ax = plt.subplots(figsize=(8, 8))
grid.plot(ax=ax, color="lightsteelblue", edgecolor="none", rasterized=True)
state.boundary.plot(ax=ax, color="black", linewidth=0.8)
ax.set_title(f"{state_name} 1-km analysis grid")
ax.set_axis_off()
plt.show()

# %% Save analysis grid
repo_root = Path(__file__).resolve().parents[1]
grids_dir = repo_root / "data" / "processed" / "grids"
grids_dir.mkdir(parents=True, exist_ok=True)

state_abbr = state.iloc[0]["STUSPS"].lower()
grid_path = grids_dir / f"{state_abbr}_analysis_grid_1km.gpkg"
grid.to_file(grid_path, layer="analysis_grid", driver="GPKG")
print(f"Saved analysis grid to: {grid_path}")

# %%
