"""Search 3DEP STAC metadata intersecting the New Mexico boundary."""

# %% Imports
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from pystac_client import Client
from shapely.errors import ShapelyError
from shapely.geometry import box, shape


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
gsd_values = sorted(
    {item.properties.get("gsd") for item in items},
    key=lambda value: (value is None, str(value)),
)
items_10m = [item for item in items if item.properties.get("gsd") == 10]

print("\n3DEP NEW MEXICO SEARCH")
print("----------------------")
print(f"Total Items found: {len(items):,}")
print(f"Distinct GSD values: {gsd_values}")
print(f"10 m Items found: {len(items_10m):,}")
print(f"Showing first {min(QA_ITEM_COUNT, len(items_10m))} 10 m Items:")

for item in items_10m[:QA_ITEM_COUNT]:
    print(f"\nItem ID: {item.id}")
    print(f"  BBOX: {item.bbox}")
    print(f"  Datetime: {item.datetime}")
    print(f"  GSD: {item.properties.get('gsd')}")
    print(f"  proj:code: {item.properties.get('proj:code')}")
    print(f"  Assets: {', '.join(item.assets)}")


# %% Build 10 m Item footprints for spatial coverage QA/QC
footprint_records = []

for item in items_10m:
    item_geometry = None

    # Prefer the STAC geometry because it can describe a more precise footprint
    # than the rectangular bbox supplied for spatial indexing.
    if item.geometry is not None:
        try:
            candidate_geometry = shape(item.geometry)
        except (AttributeError, KeyError, TypeError, ValueError, ShapelyError):
            candidate_geometry = None

        if (
            candidate_geometry is not None
            and not candidate_geometry.is_empty
            and candidate_geometry.is_valid
        ):
            item_geometry = candidate_geometry

    if item_geometry is None:
        if item.bbox is None or len(item.bbox) != 4:
            raise ValueError(
                f"10 m Item {item.id!r} has no usable geometry or bbox."
            )
        item_geometry = box(*item.bbox)

    footprint_records.append(
        {
            "item_id": item.id,
            "gsd": item.properties.get("gsd"),
            "datetime": item.datetime,
            "geometry": item_geometry,
        }
    )

item_footprints_10m = gpd.GeoDataFrame(
    footprint_records,
    geometry="geometry",
    crs=STAC_CRS,
)

if item_footprints_10m.empty:
    raise ValueError("The 10 m Item footprint GeoDataFrame is empty.")
if (
    item_footprints_10m.geometry.isna().any()
    or item_footprints_10m.geometry.is_empty.any()
):
    raise ValueError("10 m Item footprints contain missing or empty geometry.")
if item_footprints_10m.crs is None or item_footprints_10m.crs.to_epsg() != 4326:
    raise ValueError("10 m Item footprints must have CRS EPSG:4326.")

print(f"\n10 m footprint features: {len(item_footprints_10m):,}")
print(f"10 m footprint CRS: {item_footprints_10m.crs}")


# %% Plot 10 m Item coverage over New Mexico
figure, axis = plt.subplots(figsize=(10, 10))

# Draw every tile boundary without fill so gaps, overlap, and edge tiles remain
# visible against the state outline.
item_footprints_10m.boundary.plot(
    ax=axis,
    color="tab:blue",
    linewidth=0.8,
    alpha=0.8,
)
boundary_wgs84.boundary.plot(
    ax=axis,
    color="black",
    linewidth=2.0,
)

axis.set_title("3DEP 10 m STAC Item Coverage — New Mexico")
axis.set_xlabel("Longitude")
axis.set_ylabel("Latitude")
axis.set_aspect("equal")
figure.tight_layout()
plt.show()
