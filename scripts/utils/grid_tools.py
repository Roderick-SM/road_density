"""Utility functions for constructing geodesic-aware latitude–longitude grids."""

from __future__ import annotations

from typing import Tuple

import geopandas as gpd  # type: ignore[import]
import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon  # type: ignore[import]
from shapely.geometry.base import BaseGeometry  # type: ignore[import]


def snap_bounds(bounds: Tuple[float, float, float, float], res: float) -> Tuple[float, float, float, float]:
    """Snap bounds to the nearest multiple of the resolution to avoid rounding mismatches."""
    from math import ceil, floor

    minx, miny, maxx, maxy = bounds
    minx = floor(minx / res) * res
    miny = floor(miny / res) * res
    maxx = ceil(maxx / res) * res
    maxy = ceil(maxy / res) * res
    return (minx, miny, maxx, maxy)


def generate_latlon_grid(bounds: Tuple[float, float, float, float], res_deg: float) -> gpd.GeoDataFrame:
    """Generate a regular latitude–longitude grid covering the provided bounds.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box as (min_lon, min_lat, max_lon, max_lat).
    res_deg : float
        Grid resolution in degrees.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing grid cell geometries and a sequential `cell_id`.

    Raises
    ------
    ValueError
        If the resolution is not positive or the bounds are invalid.
    """
    minx, miny, maxx, maxy = bounds
    if res_deg <= 0:
        raise ValueError("Resolution must be strictly positive.")
    if not (minx < maxx and miny < maxy):
        raise ValueError("Bounds must define a valid area.")

    # Ensure coverage includes upper bound edges.
    lon_edges = np.arange(minx, maxx + res_deg, res_deg)
    lat_edges = np.arange(miny, maxy + res_deg, res_deg)

    cells = []
    cell_id = 0
    for lon_start in lon_edges[:-1]:
        lon_end = lon_start + res_deg
        for lat_start in lat_edges[:-1]:
            lat_end = lat_start + res_deg
            polygon = Polygon(
                [
                    (lon_start, lat_start),
                    (lon_end, lat_start),
                    (lon_end, lat_end),
                    (lon_start, lat_end),
                ]
            )
            cells.append({"cell_id": cell_id, "geometry": polygon})
            cell_id += 1

    grid = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:4326")
    return grid


def area_geodesic_km2(geometry: BaseGeometry, geod: Geod) -> float:
    """Compute the geodesic area of the provided polygon geometry in square kilometers.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Polygon or MultiPolygon geometry in EPSG:4326.
    geod : pyproj.Geod
        Geodesic calculator configured for WGS84.

    Returns
    -------
    float
        Geodesic area in square kilometers.
    """
    if geometry.is_empty:
        return 0.0

    if geometry.geom_type == "Polygon":
        polygons = [geometry]
    else:
        polygons = list(geometry.geoms)

    area_m2 = 0.0
    for polygon in polygons:
        lon, lat = polygon.exterior.coords.xy
        poly_area, _ = geod.polygon_area_perimeter(lon, lat)
        area_m2 += abs(poly_area)
        for interior in polygon.interiors:
            lon_h, lat_h = interior.coords.xy
            hole_area, _ = geod.polygon_area_perimeter(lon_h, lat_h)
            area_m2 -= abs(hole_area)

    return area_m2 / 1_000_000.0


def lonlat_centers_from_grid(grid: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return longitude and latitude center coordinates for each grid cell.

    Parameters
    ----------
    grid : geopandas.GeoDataFrame
        Grid GeoDataFrame produced by :func:`generate_latlon_grid`.

    Returns
    -------
    tuple of numpy.ndarray
        Arrays of longitudes and latitudes corresponding to cell centroids.
    """
    centroids = grid.geometry.centroid
    return np.array(centroids.x), np.array(centroids.y)
