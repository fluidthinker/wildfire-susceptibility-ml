"""Acquire and prepare U.S. state boundaries from TIGER/Line."""

from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
from pyproj import CRS


def fetch_state_boundary(
    state_fips: str,
    year: int = 2025,
    target_crs: str = "EPSG:5070",
) -> gpd.GeoDataFrame:
    """Download, prepare, save, and return one TIGER/Line state boundary."""
    repo_root = Path(__file__).resolve().parents[2]
    tiger_dir = repo_root / "data" / "raw" / "tiger"
    boundaries_dir = repo_root / "data" / "processed" / "boundaries"
    archive_path = tiger_dir / f"tl_{year}_us_state.zip"
    archive_url = (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/STATE/"
        f"tl_{year}_us_state.zip"
    )

    tiger_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        urlretrieve(archive_url, archive_path)

    states = gpd.read_file(archive_path)
    if "STATEFP" not in states.columns:
        raise ValueError("STATEFP column is missing from the TIGER state dataset.")

    state = states.loc[states["STATEFP"] == state_fips].copy()
    if state.empty:
        raise ValueError(f"No state matches STATEFP {state_fips!r}.")
    if len(state) > 1:
        raise ValueError(
            f"Expected exactly one state for STATEFP {state_fips!r}, "
            f"but found {len(state)}."
        )

    state = state[["STATEFP", "STUSPS", "NAME", "geometry"]]
    state_projected = state.to_crs(target_crs)

    if len(state_projected) != 1:
        raise ValueError(
            f"Expected exactly one prepared feature, but found {len(state_projected)}."
        )
    if state_projected.geometry.isna().any() or state_projected.geometry.is_empty.any():
        raise ValueError("Prepared state geometry is missing or empty.")
    if not state_projected.geometry.is_valid.all():
        raise ValueError("Prepared state geometry is invalid.")
    if state_projected.crs is None or state_projected.crs != CRS.from_user_input(
        target_crs
    ):
        raise ValueError(
            f"Prepared boundary CRS {state_projected.crs!s} does not match "
            f"requested CRS {target_crs!r}."
        )

    state_abbr = state_projected.iloc[0]["STUSPS"].lower()
    boundary_path = boundaries_dir / f"{state_abbr}_boundary.gpkg"
    state_projected.to_file(boundary_path, layer="state_boundary", driver="GPKG")

    return state_projected
