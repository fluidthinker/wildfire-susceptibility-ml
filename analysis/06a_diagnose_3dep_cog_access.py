"""Diagnose direct Rasterio access to one signed Planetary Computer 3DEP COG."""

# %% Imports
from urllib.parse import urlsplit

import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.errors import RasterioError
from rasterio.windows import Window


# %% Diagnostic parameters
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "3dep-seamless"
ITEM_ID = "n35w106-13"
ASSET_NAME = "data"
WINDOW_SIZE_PIXELS = 64


# %% Retrieve one deterministic 10 m Item using metadata only
# Querying by exact ID isolates raster access from state-scale STAC searching
# and ensures repeated diagnostics test the same remote object.
catalog = Client.open(PC_STAC_URL)
search = catalog.search(
    collections=[COLLECTION],
    ids=[ITEM_ID],
)
items = list(search.items())

if len(items) != 1:
    raise ValueError(
        f"Expected exactly one STAC Item for {ITEM_ID!r}, but found {len(items)}."
    )

item = items[0]
if item.properties.get("gsd") != 10:
    raise ValueError(
        f"Expected {ITEM_ID!r} to have gsd == 10, "
        f"but found {item.properties.get('gsd')!r}."
    )
if ASSET_NAME not in item.assets:
    raise ValueError(f"Item {ITEM_ID!r} has no {ASSET_NAME!r} asset.")

data_asset = item.assets[ASSET_NAME]
asset_hostname = urlsplit(data_asset.href).hostname

print("3DEP SIGNED COG ACCESS DIAGNOSTIC")
print("---------------------------------")
print(f"Item ID: {item.id}")
print(f"GSD: {item.properties.get('gsd')} m")
print(f"Asset name: {ASSET_NAME}")
print(f"Asset host: {asset_hostname}")
print(f"Media type: {data_asset.media_type}")
print(f"Roles: {data_asset.roles}")
print(f"Rasterio version: {rasterio.__version__}")
print(f"GDAL version: {rasterio.__gdal_version__}")


# %% Sign immediately before direct Rasterio access
# Signing adds temporary credentials to the URL. Report only whether a query
# string exists so the SAS token is never exposed in console output.
signed_item = pc.sign(item)
signed_href = signed_item.assets[ASSET_NAME].href

print("\nSIGNING")
print("-------")
print(f"Signed Item ID: {signed_item.id}")
print(f"Signed asset has query parameters: {bool(urlsplit(signed_href).query)}")


# %% Open the COG metadata, then read one tiny centered window
# This baseline deliberately uses Rasterio's default GDAL configuration. It
# separates basic HTTPS/credential access from stackstac, Dask, and reprojection.
try:
    with rasterio.open(signed_href) as dataset:
        print("\nSIGNED COG OPEN")
        print("---------------")
        print("Open succeeded: True")
        print(f"Driver: {dataset.driver}")
        print(f"Width / height: {dataset.width:,} / {dataset.height:,}")
        print(f"CRS: {dataset.crs}")
        print(f"Band count: {dataset.count}")
        print(f"Dtype: {dataset.dtypes[0]}")
        print(f"Bounds: {dataset.bounds}")

        # A centered window is more likely than an outer corner to contain
        # valid terrain while remaining negligible relative to the full tile.
        window_width = min(WINDOW_SIZE_PIXELS, dataset.width)
        window_height = min(WINDOW_SIZE_PIXELS, dataset.height)
        read_window = Window(
            col_off=(dataset.width - window_width) // 2,
            row_off=(dataset.height - window_height) // 2,
            width=window_width,
            height=window_height,
        )
        elevation_window = dataset.read(1, window=read_window, masked=True)

        valid_pixels = elevation_window.compressed()
        print("\nTINY WINDOW READ")
        print("----------------")
        print("Read succeeded: True")
        print(f"Array shape: {elevation_window.shape}")
        print(f"Valid pixels: {valid_pixels.size:,}")
        if valid_pixels.size:
            print(f"Minimum valid value: {np.min(valid_pixels):.2f}")
            print(f"Maximum valid value: {np.max(valid_pixels):.2f}")
        else:
            print("Minimum / maximum: unavailable; the window is entirely NoData.")

except RasterioError as error:
    # Some Rasterio/GDAL messages include the complete source URL. Classify the
    # message without printing it so temporary signing credentials remain secret.
    error_text = str(error).lower()
    error_type = f"{type(error).__module__}.{type(error).__name__}"

    print("\nRASTER ACCESS ERROR")
    print("-------------------")
    print("Open or read succeeded: False")
    print(f"Error type: {error_type}")
    if "schannel" in error_text or "credentials" in error_text:
        print(
            "Error category: Windows Schannel/GDAL credential acquisition "
            "during HTTPS access."
        )
    elif "http" in error_text or "curl" in error_text:
        print("Error category: GDAL HTTP/CURL access.")
    else:
        print("Error category: Rasterio/GDAL raster access.")
    print("Error details omitted because they may contain the signed asset URL.")

except Exception as error:
    # Retain a final diagnostic boundary while applying the same token-safety
    # rule to exceptions raised below Rasterio's documented error hierarchy.
    print("\nUNEXPECTED ACCESS ERROR")
    print("-----------------------")
    print("Open or read succeeded: False")
    print(f"Error type: {type(error).__module__}.{type(error).__name__}")
    print("Error details omitted because they may contain the signed asset URL.")

# %%
