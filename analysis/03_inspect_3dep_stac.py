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