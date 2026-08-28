"""Search 3DEP STAC metadata intersecting the New Mexico boundary."""

# %% Imports
from pathlib import Path

import geopandas as gpd
from pystac_client import Client


# %% Case-study parameters
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"
STAC_CRS = "EPSG:4326"
QA_ITEM_COUNT = 5

repo_root = Path(__file__).resolve().parents[1]
boundary_path = (
    repo_root / "data" / "processed" / "boundaries" / "nm_boundary.gpkg"
)


# %% Read and validate the existing New Mexico boundary
# Reuse the prepared boundary so this discovery step remains downstream of,
# and consistent with, the project's authoritative boundary workflow.
if not boundary_path.exists():
    raise FileNotFoundError(
        f"Prepared New Mexico boundary not found at {boundary_path}. "
        "Run analysis/01_fetch_state_boundary.py first."
    )

boundary = gpd.read_file(boundary_path, layer="state_boundary")

if boundary.empty:
    raise ValueError(f"Prepared boundary is empty: {boundary_path}")
if boundary.crs is None:
    raise ValueError(f"Prepared boundary has no CRS: {boundary_path}")
if boundary.geometry.isna().any() or boundary.geometry.is_empty.any():
    raise ValueError("Prepared boundary contains missing or empty geometry.")
if not boundary.geometry.is_valid.all():
    raise ValueError("Prepared boundary contains invalid geometry.")

print(f"Boundary: {boundary_path}")
print(f"Boundary CRS: {boundary.crs}")


# %% Prepare the spatial query geometry
# STAC spatial filters use longitude/latitude coordinates, while the prepared
# project boundary is projected for meter-based analysis.
if boundary.crs.to_epsg() == 4326:
    boundary_wgs84 = boundary.copy()
else:
    boundary_wgs84 = boundary.to_crs(STAC_CRS)

# The prepared state boundary should contain one feature. Enforcing that
# expectation avoids silently querying only part of the study area.
if len(boundary_wgs84) != 1:
    raise ValueError(
        "Expected one prepared New Mexico boundary feature, "
        f"but found {len(boundary_wgs84)}."
    )

query_geometry = boundary_wgs84.geometry.iloc[0].__geo_interface__
print(f"STAC query CRS: {boundary_wgs84.crs}")


# %% Search 3DEP metadata intersecting New Mexico
# Use the state geometry rather than its bounding box so the server applies
# the closest practical spatial filter to the actual study area.
catalog = Client.open(PC_STAC_URL)
search = catalog.search(
    collections=[COLLECTION],
    intersects=query_geometry,
)
items = list(search.items())

if not items:
    raise ValueError(
        f"No Items from {COLLECTION!r} intersect the New Mexico boundary."
    )


# %% Concise metadata QA/QC
print("\n3DEP NEW MEXICO SEARCH")
print("----------------------")
print(f"Total Items found: {len(items):,}")
print(f"Showing first {min(QA_ITEM_COUNT, len(items))} Items:")

for item in items[:QA_ITEM_COUNT]:
    print(f"\nItem ID: {item.id}")
    print(f"  BBOX: {item.bbox}")
    print(f"  Datetime: {item.datetime}")
    print(f"  GSD: {item.properties.get('gsd')}")
    print(f"  proj:code: {item.properties.get('proj:code')}")
    print(f"  Assets: {', '.join(item.assets)}")


# %%
