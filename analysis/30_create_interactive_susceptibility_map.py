"""Create a lightweight presentation map of the completed susceptibility surface.

Only the embedded display raster is reprojected, color-binned and masked to
the state boundary. The authoritative full-cell GeoParquet remains unchanged.
No modeling, evaluation, feature construction or data downloads occur here.
"""

# %% Imports
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

# Presentation must work from installed CRS resources without fetching optional
# datum grids. Set this before importing the GDAL/PROJ-backed libraries.
os.environ["PROJ_NETWORK"] = "OFF"

import folium
from branca.element import Element
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import array_bounds, from_origin, rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds


# %% Parameters and paths
ROOT = Path(__file__).resolve().parents[1]
SURFACE_PATH = ROOT / "data/processed/modeling/nm_statewide_susceptibility.geoparquet"
SUMMARY_PATH = ROOT / "outputs/modeling/statewide_susceptibility/summary.json"
SPLIT_PATH = ROOT / "data/processed/modeling/nm_modeling_splits.parquet"
BOUNDARY_PATH = ROOT / "data/processed/boundaries/nm_boundary.gpkg"
OUTPUT_PATH = ROOT / "outputs/maps/nm_wildfire_susceptibility_interactive.html"
EXPECTED_ROWS = 314_920
NODATA = -1.0
BINS = np.array([0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.67])
# Restrained sequential yellow-green-blue colors, darker for larger scores.
COLORS = ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#225ea8", "#0c2c84"]
LAYER_NAME = "Model-estimated wildfire susceptibility"
CAVEAT = ("Exploratory model output. Geographic transfer to the eastern holdout was weak; "
          "this map should not be interpreted as validated wildfire risk.")


# %% Helper functions
def fingerprint_inputs() -> dict[str, str]:
    """Hash the four source artifacts without loading them fully into memory.

    Returns:
        Repository-relative paths and SHA-256 digests.
    """
    result = {}
    for path in (SURFACE_PATH, SUMMARY_PATH, SPLIT_PATH, BOUNDARY_PATH):
        with path.open("rb") as source:
            result[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    return result


def validate_surface(surface: gpd.GeoDataFrame, summary: dict) -> None:
    """Require the completed full-cell surface and its published row contract.

    Args:
        surface: Script 29's authoritative susceptibility polygons.
        summary: Its published mapping metadata, not evaluation metrics.

    Raises:
        ValueError: If IDs, probabilities, geometry or metadata are unexpected.
    """
    if list(surface.columns) != ["cell_id", "susceptibility_probability", "geometry"]:
        raise ValueError("Unexpected statewide surface schema")
    if len(surface) != EXPECTED_ROWS or surface.cell_id.isna().any() or not surface.cell_id.is_unique:
        raise ValueError("Expected 314,920 unique statewide cell IDs")
    if surface.crs is None or surface.crs.to_epsg() != 5070:
        raise ValueError("Source surface must be EPSG:5070")
    geometry = surface.geometry
    if geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all():
        raise ValueError("Surface geometry must be valid and nonempty")
    if not geometry.geom_type.eq("Polygon").all() or not np.allclose(geometry.area, 1_000_000):
        raise ValueError("Expected complete 1-km polygon cells")
    bounds = geometry.bounds
    if not (np.allclose(bounds.maxx - bounds.minx, 1000) and np.allclose(bounds.maxy - bounds.miny, 1000)):
        raise ValueError("Expected regular 1-km square geometry")
    probability = surface.susceptibility_probability.to_numpy()
    if not np.isfinite(probability).all() or ((probability < BINS[0]) | (probability > BINS[-1])).any():
        raise ValueError("Probabilities must be finite and covered by the fixed display bins")
    if summary["predicted_rows"] != len(surface) or summary["crs"] != "EPSG:5070":
        raise ValueError("Script 29 summary contract differs")
    if float(probability.min()) != summary["probability_min"] or float(probability.max()) != summary["probability_max"]:
        raise ValueError("Surface range differs from Script 29 metadata")


def rasterize_surface(surface: gpd.GeoDataFrame) -> tuple[np.ndarray, object]:
    """Burn one original score into each aligned 1-km pixel without interpolation.

    Args:
        surface: Validated EPSG:5070 square cells.

    Returns:
        Float64 score raster and affine transform; -1 is explicit NoData.

    Raises:
        ValueError: If alignment, coverage or rasterized values differ.
    """
    west, south, east, north = surface.total_bounds
    if not np.allclose(np.array([west, south, east, north]) % 1000, 0):
        raise ValueError("Grid extent is not aligned to 1000 m")
    width, height = int((east - west) / 1000), int((north - south) / 1000)
    transform = from_origin(west, north, 1000, 1000)
    scores = surface.susceptibility_probability.to_numpy()
    raster = rasterize(zip(surface.geometry, scores), out_shape=(height, width),
                       transform=transform, fill=NODATA, dtype="float64", all_touched=False)
    centers = surface.geometry.centroid
    rows, columns = rowcol(transform, centers.x.to_numpy(), centers.y.to_numpy())
    if np.count_nonzero(raster != NODATA) != len(surface) or not np.array_equal(raster[rows, columns], scores):
        raise ValueError("Rasterization did not preserve one exact value per cell")
    return raster, transform


def reproject_display(source: np.ndarray, transform: object, source_crs: str,
                      destination_crs: str) -> tuple[np.ndarray, object]:
    """Reproject a presentation raster with nearest-neighbor value preservation.

    Args:
        source: Score raster with -1 NoData.
        transform: Source affine transform.
        source_crs: Explicit source CRS identifier.
        destination_crs: Explicit display CRS identifier.

    Returns:
        Reprojected raster and affine transform; no smoothed scores are created.

    Raises:
        ValueError: If reprojection produces new scores or an empty display.
    """
    height, width = source.shape
    bounds = array_bounds(height, width, transform)
    target_transform, target_width, target_height = calculate_default_transform(
        source_crs, destination_crs, width, height, *bounds,
    )
    destination = np.full((target_height, target_width), NODATA, dtype="float64")
    reproject(source, destination, src_transform=transform, src_crs=source_crs,
              dst_transform=target_transform, dst_crs=destination_crs,
              src_nodata=NODATA, dst_nodata=NODATA, resampling=Resampling.nearest)
    valid = destination != NODATA
    if not valid.any() or not np.isin(np.unique(destination[valid]), np.unique(source)).all():
        raise ValueError("Nearest-neighbor display must contain only original scores")
    return destination, target_transform


def build_display_image(raster: np.ndarray, transform: object, state: gpd.GeoDataFrame) -> np.ndarray:
    """Color-bin scores and make outside-state presentation pixels transparent.

    Args:
        raster: Final EPSG:3857 display scores.
        transform: Display affine transform.
        state: Existing state boundary, explicitly projected here for masking.

    Returns:
        RGBA uint8 image. Bins are left-inclusive; the final bin includes 0.67.
        Pixel-center masking is display-only and does not clip original cells.
    """
    inside = geometry_mask(state.to_crs(3857).geometry, out_shape=raster.shape,
                           transform=transform, invert=True, all_touched=False)
    valid = (raster != NODATA) & inside
    rgba = np.zeros((*raster.shape, 4), dtype="uint8")
    palette = np.array([[int(color[index:index + 2], 16) for index in (1, 3, 5)]
                        for color in COLORS], dtype="uint8")
    # Only the presentation color is classified; probabilities stay unchanged.
    bins = np.searchsorted(BINS[1:-1], raster[valid], side="right")
    rgba[valid, :3] = palette[bins]
    rgba[valid, 3] = 255
    if not valid.any() or rgba[~inside, 3].any():
        raise ValueError("Expected visible in-state pixels and transparent outside-state pixels")
    return rgba


def build_holdout_outline(surface: gpd.GeoDataFrame, splits: pd.DataFrame) -> gpd.GeoDataFrame:
    """Union the saved holdout cells and simplify only their display outline.

    Args:
        surface: Authoritative statewide cell geometries.
        splits: Existing Script 16 split assignments.

    Returns:
        One geographic feature (possibly multipart), simplified by 500 m.

    Raises:
        ValueError: If frozen membership, block isolation or joined keys fail.
    """
    if len(splits) != 313_129 or splits.cell_id.isna().any() or not splits.cell_id.is_unique:
        raise ValueError("Unexpected frozen split population")
    if splits.split.value_counts().to_dict() != {"train": 252_569, "final_test": 60_560}:
        raise ValueError("Frozen split counts differ")
    if splits.spatial_block_id.isna().any() or not splits.groupby("spatial_block_id").split.nunique().eq(1).all():
        raise ValueError("Saved spatial blocks must be isolated between populations")
    holdout = splits.loc[splits.split.eq("final_test")]
    if holdout.spatial_block_id.nunique() != 33 or not splits.cell_id.isin(surface.cell_id).all():
        raise ValueError("Holdout blocks or cell membership differ")
    selected = surface[["cell_id", "geometry"]].merge(holdout[["cell_id"]], on="cell_id", validate="one_to_one")
    if len(selected) != 60_560:
        raise ValueError("Holdout join lost cells")
    # Grid cells do not overlap: coverage union is suited to these polygons.
    # Keep holes/multipart structure; simplify in meters, never rebuild the split.
    outline = selected.geometry.union_all(method="coverage").simplify(500, preserve_topology=True)
    if outline.is_empty or not outline.is_valid:
        raise ValueError("Invalid dissolved display outline")
    return gpd.GeoDataFrame({"name": ["Eastern geographic holdout"]}, geometry=[outline], crs=5070).to_crs(4326)


def add_map_information(map_object: folium.Map) -> None:
    """Add compact interpretation text and exact numeric display-bin labels.

    Args:
        map_object: Map receiving fixed presentation panels.
    """
    items = []
    for index, color in enumerate(COLORS):
        closing = "]" if index == len(COLORS) - 1 else ")"
        items.append(f'<div><span style="background:{color}"></span>'
                     f'[{BINS[index]:g}, {BINS[index+1]:g}{closing}</div>')
    map_object.get_root().header.add_child(Element("""
        <style>
        .map-info {position:fixed;z-index:1000;background:rgba(255,255,255,.95);
          padding:12px 14px;border-radius:5px;box-shadow:0 1px 8px #0003;
          font:12px/1.4 Arial,sans-serif;color:#263238;max-width:calc(100vw - 100px)}
        #map-title {top:12px;left:60px;width:300px}
        #map-legend {bottom:28px;left:12px;width:225px}
        #map-legend span {display:inline-block;width:25px;height:12px;margin-right:8px;
          border:1px solid #999;vertical-align:middle}
        .map-info h3 {font-size:15px;margin:0 0 5px}
        .map-info p {margin:5px 0 0}
        @media(max-width:600px) {.map-info {font-size:10px;padding:8px}
          #map-title {width:235px} #map-legend {width:195px}}
        </style>
    """))
    map_object.get_root().html.add_child(Element(
        '<section id="map-title" class="map-info"><h3>New Mexico Wildfire Susceptibility</h3>'
        '<div>Model-estimated susceptibility from terrain, climate, vegetation, '
        f'and historical wildfire occurrence</div><p>{CAVEAT}</p></section>'
        '<section id="map-legend" class="map-info"><b>Model-estimated susceptibility</b>'
        + "".join(items) + '<p>Numeric model-score intervals; presentation only.</p></section>'
    ))
    folium.LayerControl(collapsed=True).add_to(map_object)


def publish_map(map_object: folium.Map, fingerprints: dict) -> int:
    """Read back the staged HTML, check its layers and sources, then publish.

    Args:
        map_object: Completed Folium map with an embedded PNG image.
        fingerprints: Original source hashes.

    Returns:
        Published HTML file size in bytes.

    Raises:
        ValueError: If required map components or input integrity checks fail.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".html.part")
    try:
        map_object.save(str(temporary))
        html = temporary.read_text(encoding="utf-8")
        required = ["data:image/png;base64,", "L.imageOverlay(", "New Mexico boundary",
                    "Eastern geographic holdout", "map-legend", "map-title", CAVEAT,
                    "L.control.layers(", "openstreetmap.org", "fitBounds("]
        if any(token not in html for token in required):
            raise ValueError("Missing required HTML map component")
        if html.count("L.geoJson(") != 2 or "cell_000001" in html:
            raise ValueError("Expected only two outline GeoJSON layers, no cell polygons")
        if fingerprint_inputs() != fingerprints:
            raise ValueError("Source artifacts changed while building the presentation")
        temporary.replace(OUTPUT_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return OUTPUT_PATH.stat().st_size


# %% Main workflow
def main() -> None:
    """Render the existing scores as a portable lightweight map and stop."""
    started = perf_counter()
    fingerprints = fingerprint_inputs()
    # STEP 1 - Read and validate the completed statewide output, never refit it.
    surface = gpd.read_parquet(SURFACE_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    validate_surface(surface, summary)
    state = gpd.read_file(BOUNDARY_PATH)
    if (len(state) != 1 or state.crs is None or not state.STATEFP.eq("35").all()
            or state.geometry.isna().any() or state.geometry.is_empty.any()
            or not state.geometry.is_valid.all()):
        raise ValueError("Expected an existing valid New Mexico boundary")

    # STEP 2 - Rasterize the 1-km cells at their exact native resolution.
    source, source_transform = rasterize_surface(surface)

    # STEP 3 - Reproject display scores to geographic coordinates with nearest neighbor.
    geographic, geographic_transform = reproject_display(source, source_transform, "EPSG:5070", "EPSG:4326")
    # Leaflet stretches ImageOverlay in its map CRS (Web Mercator). Convert
    # the geographic image to that pixel layout explicitly, still nearest
    # neighbor, to avoid both latitude distortion and Folium's interpolator.
    display, display_transform = reproject_display(geographic, geographic_transform, "EPSG:4326", "EPSG:3857")
    rgba = build_display_image(display, display_transform, state)
    west, south, east, north = transform_bounds(
        "EPSG:3857", "EPSG:4326", *array_bounds(*display.shape, display_transform),
    )

    # STEP 4 - Embed the PNG directly; only the browser requests basemap tiles.
    map_object = folium.Map(tiles="OpenStreetMap", control_scale=True)
    folium.raster_layers.ImageOverlay(rgba, bounds=[[south, west], [north, east]],
                                     name=LAYER_NAME, opacity=0.65, origin="upper",
                                     mercator_project=False, pixelated=True).add_to(map_object)

    # STEP 5 - Add only two small outline features, not 314,920 interactive cells.
    state_geographic = state[["geometry"]].to_crs(4326)
    holdout = build_holdout_outline(surface, pd.read_parquet(SPLIT_PATH))
    folium.GeoJson(state_geographic, name="New Mexico boundary",
                   style_function=lambda _: {"color": "#35434b", "weight": 1.2, "fillOpacity": 0},
                   tooltip="New Mexico boundary").add_to(map_object)
    folium.GeoJson(holdout, name="Eastern geographic holdout",
                   style_function=lambda _: {"color": "#a23a24", "weight": 2.2,
                                              "dashArray": "7 5", "fillOpacity": 0},
                   tooltip="Eastern geographic holdout").add_to(map_object)
    west, south, east, north = state_geographic.total_bounds
    map_object.fit_bounds([[south, west], [north, east]])

    # STEP 6 - Include numeric bins, the evaluation caveat and toggle controls.
    add_map_information(map_object)

    # STEP 7 - Validate staged HTML and unchanged sources before publication.
    size = publish_map(map_object, fingerprints)

    # STEP 8 - Report presentation QA and STOP before any README work.
    print(json.dumps({
        "statewide_rows": len(surface), "source_crs": "EPSG:5070",
        "native_raster_height_width": list(source.shape),
        "geographic_raster_height_width": list(geographic.shape), "geographic_crs": "EPSG:4326",
        "embedded_raster_height_width": list(display.shape), "embedded_pixel_layout": "EPSG:3857",
        "overlay_bounds_crs": "EPSG:4326", "resampling": "nearest neighbor for both display warps",
        "proj_network": "OFF; use locally available coordinate transformations only",
        "display_bins": BINS.tolist(), "state_boundary_source": BOUNDARY_PATH.relative_to(ROOT).as_posix(),
        "holdout_cells": 60_560, "holdout_display_features": len(holdout),
        "holdout_geometry_type": holdout.geometry.iloc[0].geom_type, "outline_simplification_m": 500,
        "basemap": "OpenStreetMap", "individual_cell_geojson": False,
        "outside_state": "transparent at display-pixel centers; source geometry unchanged",
        "output": OUTPUT_PATH.relative_to(ROOT).as_posix(), "html_bytes": size,
        "runtime_seconds": round(perf_counter() - started, 3),
        "modeling_or_performance_analysis": False,
        "viewing_note": "Embedded raster and outlines; internet needed for basemap tiles and CDN JavaScript/CSS",
    }, indent=2), flush=True)


# %% Run script
if __name__ == "__main__":
    main()

# %%
