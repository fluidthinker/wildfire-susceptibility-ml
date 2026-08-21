"""Create regular analysis grids for projected study-area boundaries."""

import geopandas as gpd
import numpy as np
import shapely


def create_analysis_grid(
    boundary: gpd.GeoDataFrame,
    cell_size: int = 1000,
) -> gpd.GeoDataFrame:
    """Create a regular square grid within a study-area boundary.

    Candidate cells are created in the boundary's projected, meter-based CRS
    and aligned to exact multiples of ``cell_size``. Full cells are retained
    when their centroids fall strictly within the boundary; cells are not
    clipped, and centroids on the boundary are excluded.

    Args:
        boundary: GeoDataFrame containing exactly one valid, non-empty
            study-area geometry in a projected CRS with meter units.
        cell_size: Width and height of each square grid cell in meters.

    Returns:
        GeoDataFrame containing full retained grid cells and deterministic,
        unique cell IDs. The candidate-cell count is stored in
        ``GeoDataFrame.attrs["candidate_cell_count"]``.

    Raises:
        ValueError: If the boundary, CRS, geometry, cell size, generated grid,
            cell dimensions, cell IDs, or centroid membership is invalid.
    """
    if len(boundary) != 1:
        raise ValueError(
            f"Expected exactly one boundary feature, but received {len(boundary)}."
        )
    if boundary.crs is None:
        raise ValueError("Boundary must have a CRS.")
    if not boundary.crs.is_projected:
        raise ValueError("Boundary CRS must be projected, not geographic.")
    if not boundary.crs.axis_info or any(
        axis.unit_name.lower() not in {"metre", "meter"}
        for axis in boundary.crs.axis_info
    ):
        raise ValueError("Boundary CRS coordinate units must be meters.")
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than zero.")
    if boundary.geometry.isna().any() or boundary.geometry.is_empty.any():
        raise ValueError("Boundary geometry is missing or empty.")
    if not boundary.geometry.is_valid.all():
        raise ValueError("Boundary geometry is invalid.")

    # Snap the extent to cell-size multiples so alignment is deterministic
    # and compatible across study areas in the same projected CRS.
    min_x, min_y, max_x, max_y = boundary.total_bounds
    x_origins = np.arange(
        np.floor(min_x / cell_size) * cell_size,
        np.ceil(max_x / cell_size) * cell_size,
        cell_size,
    )
    y_origins = np.arange(
        np.floor(min_y / cell_size) * cell_size,
        np.ceil(max_y / cell_size) * cell_size,
        cell_size,
    )
    # Build all bounding-box candidates with vectorized coordinate arrays to
    # avoid a Python loop over hundreds of thousands of polygons.
    x_grid, y_grid = np.meshgrid(x_origins, y_origins)
    candidate_geometry = shapely.box(
        x_grid.ravel(),
        y_grid.ravel(),
        x_grid.ravel() + cell_size,
        y_grid.ravel() + cell_size,
    )
    candidate_count = len(candidate_geometry)

    # Preserve complete, equal-area cells for analysis; the strict ``within``
    # predicate excludes the rare centroid that lies exactly on the boundary.
    boundary_geometry = boundary.geometry.iloc[0]
    centroid_mask = shapely.within(
        shapely.centroid(candidate_geometry), boundary_geometry
    )
    retained_geometry = candidate_geometry[centroid_mask]

    grid = gpd.GeoDataFrame(
        {
            "cell_id": [
                f"cell_{number:06d}"
                for number in range(1, len(retained_geometry) + 1)
            ]
        },
        geometry=retained_geometry,
        crs=boundary.crs,
    )
    grid.attrs["candidate_cell_count"] = candidate_count

    # Verify the methodological guarantees after filtering: full valid
    # squares, stable IDs, and centroids that satisfy the inclusion rule.
    if grid.empty:
        raise ValueError("No grid cells have centroids within the boundary.")
    if grid.crs != boundary.crs:
        raise ValueError("Grid CRS does not match the boundary CRS.")
    if grid.geometry.isna().any() or grid.geometry.is_empty.any():
        raise ValueError("Grid contains missing or empty cell geometry.")
    if not grid.geometry.is_valid.all():
        raise ValueError("Grid contains invalid cell geometry.")
    if not np.allclose(grid.geometry.area.to_numpy(), cell_size**2):
        raise ValueError("Grid cell areas do not match the requested square size.")
    grid_bounds = grid.geometry.bounds
    if not (
        np.allclose(grid_bounds["maxx"] - grid_bounds["minx"], cell_size)
        and np.allclose(grid_bounds["maxy"] - grid_bounds["miny"], cell_size)
    ):
        raise ValueError("Grid cell dimensions do not match the requested size.")
    if not grid["cell_id"].is_unique:
        raise ValueError("Grid cell IDs are not unique.")
    if not shapely.within(grid.geometry.centroid.array, boundary_geometry).all():
        raise ValueError("A retained cell centroid falls outside the boundary.")

    return grid
