# %% Imports
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import matplotlib.pyplot as plt

# %% Define project-relative paths
repo_root = Path(__file__).resolve().parents[1]
raw_root = repo_root / "data" / "raw"
tiger_dir = raw_root / "tiger"
processed_root = repo_root / "data" / "processed"
boundaries_dir = processed_root / "boundaries"

source_zip_path = tiger_dir / "tl_2025_us_state.zip"
prepared_boundary_path = boundaries_dir / "new_mexico_boundary.gpkg"

# Create directories when needed so the script is portable and repeatable.
tiger_dir.mkdir(parents=True, exist_ok=True)
boundaries_dir.mkdir(parents=True, exist_ok=True)

# %% Download the TIGER/Line archive
source_url = "https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip"

if not source_zip_path.exists():
    print(f"Downloading TIGER/Line states archive to: {source_zip_path}")
    urlretrieve(source_url, source_zip_path)
else:
    print(f"Using existing TIGER/Line archive: {source_zip_path}")

# %% Read the state boundary data
# GeoPandas can read the zipped shapefile directly, which keeps the workflow simple.
state_gdf = gpd.read_file(source_zip_path)

print(f"Loaded {len(state_gdf)} state/equivalent records.")
print(f"Columns: {list(state_gdf.columns[:12])}")
print(f"STATEFP column present: {'STATEFP' in state_gdf.columns}")
print(f"Source CRS: {state_gdf.crs}")

# %% Select New Mexico
if "STATEFP" not in state_gdf.columns:
    raise ValueError("STATEFP column is missing from the TIGER state dataset.")

new_mexico = state_gdf[state_gdf["STATEFP"] == "35"].copy()

if len(new_mexico) == 0:
    raise ValueError("No records were returned for STATEFP == '35'.")
if len(new_mexico) > 1:
    raise ValueError(f"Expected exactly one New Mexico record, but got {len(new_mexico)}.")

new_mexico = new_mexico[["STATEFP", "STUSPS", "NAME", "geometry"]]
print(f"Selected New Mexico rows: {len(new_mexico)}")
print(new_mexico[["STATEFP", "STUSPS", "NAME"]].head())

# %% Inspect the source CRS
print(f"Source CRS before reprojection: {new_mexico.crs}")

# %% Reproject New Mexico
# EPSG:5070 is a meter-based CONUS Albers equal-area projection appropriate for the later 1-km analysis grid.
new_mexico_projected = new_mexico.to_crs("EPSG:5070")
print(f"Projected CRS: {new_mexico_projected.crs}")

# %% Validate the prepared boundary
if len(new_mexico_projected) != 1:
    raise ValueError(f"Expected exactly one prepared feature, but got {len(new_mexico_projected)}.")
if new_mexico_projected.crs.to_epsg() != 5070:
    raise ValueError(f"Prepared boundary is not in EPSG:5070; actual CRS is {new_mexico_projected.crs}.")
if new_mexico_projected.geometry.is_empty.any():
    raise ValueError("Prepared New Mexico geometry contains empty geometries.")
if not new_mexico_projected.is_valid.all():
    invalid_rows = new_mexico_projected[~new_mexico_projected.is_valid]
    print(invalid_rows[["STATEFP", "NAME"]])
    raise ValueError("Prepared New Mexico geometry is invalid.")

print(f"Prepared geometry is valid and non-empty: {not new_mexico_projected.geometry.is_empty.all()}")

# %% Visual inspection
fig, ax = plt.subplots(figsize=(8, 8))
new_mexico_projected.plot(ax=ax, color="lightsteelblue", edgecolor="black")
ax.set_title("New Mexico boundary (EPSG:5070)")
ax.set_axis_off()
plt.show()

# %% Save processed boundary
new_mexico_projected.to_file(prepared_boundary_path, layer="new_mexico_boundary", driver="GPKG")
print(f"Saved prepared boundary to: {prepared_boundary_path}")
print(f"Saved CRS: {new_mexico_projected.crs}")
