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

# Statewide coverage overview
fig, overview_ax = plt.subplots(figsize=(8, 8))
grid.plot(
    ax=overview_ax,
    facecolor="lightsteelblue",
    edgecolor="slategray",
    linewidth=0.05,
    rasterized=True,
)
state.boundary.plot(ax=overview_ax, color="black", linewidth=1.0)
overview_ax.set_title(f"{state_name} 1-km analysis grid — statewide overview")
overview_ax.set_axis_off()
plt.show()

# Spatially contiguous detail near the center of the state
state_center = state.geometry.iloc[0].centroid
detail_half_width = 10 * CELL_SIZE_M
detail_grid = grid.cx[
    state_center.x - detail_half_width : state_center.x + detail_half_width,
    state_center.y - detail_half_width : state_center.y + detail_half_width,
]

fig, detail_ax = plt.subplots(figsize=(8, 8))
detail_grid.plot(
    ax=detail_ax,
    facecolor="none",
    edgecolor="black",
    linewidth=0.6,
)
detail_ax.set_title(f"{state_name} 1-km grid — central detail")
detail_ax.set_aspect("equal")
detail_ax.set_axis_off()
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
