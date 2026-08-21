# %% Imports
import matplotlib.pyplot as plt

from wildfire_susceptibility.boundary import fetch_state_boundary

# %% Case-study parameters
STATE_FIPS = "35"
TARGET_CRS = "EPSG:5070"
TIGER_YEAR = 2025

# %% Fetch and prepare state boundary
state = fetch_state_boundary(
    state_fips=STATE_FIPS,
    year=TIGER_YEAR,
    target_crs=TARGET_CRS,
)

# %% Inspect result
print(state[["STATEFP", "STUSPS", "NAME"]])
print(f"Prepared CRS: {state.crs}")
print(f"Geometry is valid: {state.geometry.is_valid.all()}")
print(f"Geometry is non-empty: {not state.geometry.is_empty.any()}")

# %% Visual inspection
state_name = state.iloc[0]["NAME"]
fig, ax = plt.subplots(figsize=(8, 8))
state.plot(ax=ax, color="lightsteelblue", edgecolor="black")
ax.set_title(f"{state_name} boundary ({TARGET_CRS})")
ax.set_axis_off()
plt.show()

# %%
