"""Regular spatial blocks shared by exploration and experiment construction."""

import geopandas as gpd
import numpy as np


def assign_spatial_blocks(
    cells: gpd.GeoDataFrame, grid_bounds: np.ndarray, block_size_m: int = 50_000
) -> tuple[gpd.GeoDataFrame, tuple[float, float]]:
    """Place each eligible cell centroid in a deterministic regular spatial block.

    Args:
        cells: Eligible cells in EPSG:5070.
        grid_bounds: Bounds of the full authoritative grid, not an eligible subset.
        block_size_m: Positive block width and height in meters.

    Returns:
        Cells with block positions and the snapped southwest anchor.

    Raises:
        ValueError: If CRS, block size, or bounds are invalid.
    """
    if cells.crs is None or cells.crs.to_epsg() != 5070:
        raise ValueError("Block assignment requires EPSG:5070; no reprojection is performed")
    if not np.isfinite(block_size_m) or block_size_m <= 0:
        raise ValueError("Block size must be positive and finite")
    if len(grid_bounds) != 4 or not np.isfinite(grid_bounds).all():
        raise ValueError("Full grid bounds must contain four finite coordinates")
    # Snapping the full grid bounds downward avoids moving the anchor when
    # eligibility changes. For example, x=anchor_x+75,000 belongs to column 1.
    anchor_x, anchor_y = np.floor(grid_bounds[:2] / block_size_m) * block_size_m
    centroids = cells.geometry.centroid
    assigned = cells.copy()
    assigned["centroid_x"] = centroids.x
    assigned["centroid_y"] = centroids.y
    assigned["block_col"] = np.floor((centroids.x - anchor_x) / block_size_m).astype("int64")
    assigned["block_row"] = np.floor((centroids.y - anchor_y) / block_size_m).astype("int64")
    assigned["block_id"] = "r" + assigned.block_row.astype(str) + "_c" + assigned.block_col.astype(str)
    return assigned, (float(anchor_x), float(anchor_y))


def count_block_components(blocks: gpd.GeoDataFrame) -> int:
    """Count connected groups using shared block edges rather than corner contact.

    Args:
        blocks: Selected occupied blocks with integer row/column positions.

    Returns:
        Number of four-neighbor connected components; one means contiguous.
    """
    remaining = set(zip(blocks.block_row, blocks.block_col))
    components = 0
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        while frontier:
            row, col = frontier.pop()
            # Only north/south/east/west adjacency qualifies. An unoccupied
            # block does not bridge two groups of selected occupied blocks.
            neighbors = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    return components
