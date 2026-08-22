# %%
from pystac_client import Client
import planetary_computer as pc
import stackstac


PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"


# %%
# Open the Planetary Computer STAC catalog.

catalog = Client.open(PC_STAC_URL)

print("CATALOG")
print("-------")
print(catalog)


# %%
# Search for only ONE 3DEP Item.
# This is metadata-only; no raster pixels are loaded here.

search = catalog.search(
    collections=[COLLECTION],
    max_items=1,
)

items = list(search.items())

print("\nSEARCH RESULT")
print("-------------")
print(f"Items found: {len(items)}")

if not items:
    raise ValueError("No 3DEP Items were returned.")

item = items[0]


# %%
# Inspect the returned STAC Item.

print("\nITEM METADATA")
print("-------------")
print(f"ID: {item.id}")
print(f"BBOX: {item.bbox}")
print(f"Datetime: {item.datetime}")

print("\nItem properties:")
for key, value in item.properties.items():
    print(f"{key}: {value}")

print("\nAvailable assets:")
for asset_name in item.assets:
    print(asset_name)


# %%
# Inspect the raster Asset.

data_asset = item.assets["data"]

print("\nDATA ASSET")
print("----------")
print(f"Href: {data_asset.href}")
print(f"Media type: {data_asset.media_type}")
print(f"Roles: {data_asset.roles}")
print(f"Title: {data_asset.title}")
print(f"Description: {data_asset.description}")

print("\nAsset extra fields:")
for key, value in data_asset.extra_fields.items():
    print(f"{key}: {value}")


# %%
# Sign the Item immediately before attempting raster access.
#
# Planetary Computer signing adds temporary access credentials
# to the Asset URLs. This does not download or modify the data.

signed_item = pc.sign(item)

print("\nSIGNING")
print("-------")
print(f"Signed Item ID: {signed_item.id}")
print("Signed Asset URL contains temporary query parameters:")
print("?" in signed_item.assets["data"].href)


# %%
# Define a deliberately small test window inside this Item.
#
# This prevents us from reading the full ~1-degree DEM tile while
# validating that remote raster access works correctly.

test_bbox = (
    -115.55,  # min longitude
    33.45,    # min latitude
    -115.50,  # max longitude
    33.50,    # max latitude
)


# %%
# Build a lazy Stackstac/Xarray representation of the small DEM window.
#
# stackstac does not read all pixel values yet.
# The actual raster read occurs when .compute() is called.

dem_test = stackstac.stack(
    [signed_item],
    assets=["data"],
    bounds_latlon=test_bbox,
    epsg=5498,
)

print("\nLAZY TEST WINDOW")
print("----------------")
print(dem_test)


# %%
# Trigger the actual remote COG pixel read.

dem_test_loaded = dem_test.compute()

print("\nLOADED TEST WINDOW")
print("------------------")

print(f"Shape: {dem_test_loaded.shape}")
print(f"CRS: {dem_test_loaded.attrs.get('crs')}")

print("\nElevation statistics:")
print(f"Min:  {float(dem_test_loaded.min().values):.2f} m")
print(f"Max:  {float(dem_test_loaded.max().values):.2f} m")
print(f"Mean: {float(dem_test_loaded.mean().values):.2f} m")


# %%
# Visual QA/QC of the small elevation window.

dem_test_loaded.squeeze().plot()
# %%
