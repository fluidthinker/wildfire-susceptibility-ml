# %%
from pystac_client import Client
import planetary_computer as pc


PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"


# %%
catalog = Client.open(PC_STAC_URL)

print(catalog)


# %%
search = catalog.search(
    collections=[COLLECTION],
    max_items=1,
)

items = list(search.items())

print(f"Items found: {len(items)}")


# %%
item = items[0]

print("Item ID:")
print(item.id)

print("\nAvailable assets:")
for asset_name in item.assets:
    print(asset_name)


# %%
signed_item = pc.sign(item)

print("\nSigned successfully:")
print(signed_item.id)
# %%


# %%
# Inspect the STAC Item metadata.

print("ITEM METADATA")
print("-------------")
print(f"ID: {item.id}")
print(f"BBOX: {item.bbox}")
print(f"Datetime: {item.datetime}")

print("\nItem properties:")
for key, value in item.properties.items():
    print(f"{key}: {value}")


# %%
# Inspect the actual raster asset associated with this Item.

data_asset = item.assets["data"]

print("DATA ASSET")
print("----------")
print(f"Href: {data_asset.href}")
print(f"Media type: {data_asset.media_type}")
print(f"Roles: {data_asset.roles}")
print(f"Title: {data_asset.title}")
print(f"Description: {data_asset.description}")


# %%
# Inspect any additional STAC metadata attached to the raster asset.

print("ASSET EXTRA FIELDS")
print("------------------")

for key, value in data_asset.extra_fields.items():
    print(f"{key}: {value}")


# %%
# Compare the asset URL before and after Planetary Computer signing.

signed_data_asset = signed_item.assets["data"]

print("UNSIGNED HREF:")
print(data_asset.href)

print("\nSIGNED HREF:")
print(signed_data_asset.href)
# %%
