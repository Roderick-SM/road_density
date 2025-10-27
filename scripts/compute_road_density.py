"""Core module for computing road density on a regular latitude–longitude grid."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import geopandas as gpd  # type: ignore[import]
import numpy as np
import xarray as xr  # type: ignore[import]
from pyproj import Geod
from tqdm import tqdm  # type: ignore[import]

from scripts.utils.grid_tools import area_geodesic_km2, generate_latlon_grid, snap_bounds

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


def _prepare_geodataframe(path: Path) -> gpd.GeoDataFrame:
    """Load a vector dataset and reproject it to EPSG:4326 if necessary.

    Parameters
    ----------
    path : pathlib.Path
        File path to a geospatial vector dataset.

    Returns
    -------
    geopandas.GeoDataFrame
        Dataset in EPSG:4326.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the dataset is empty or lacks CRS information.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Input file contains no features: {path}")

    if gdf.crs is None:
        raise ValueError(f"Input file lacks CRS information: {path}")

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _format_resolution_label(res_deg: float) -> str:
    """Return a human-readable label for the grid resolution.

    Parameters
    ----------
    res_deg : float
        Grid resolution in decimal degrees.

    Returns
    -------
    str
        Resolution label with decimal point replaced by `p` (e.g., ``0p01``).
    """
    res_str = f"{res_deg:.4f}".rstrip("0").rstrip(".")
    return res_str.replace(".", "p")


def compute_road_density(
    path_roads: str,
    path_region: str,
    res_deg: float,
    out_prefix: str,
    output_dir: str = "./data/outputs",
    snap_to_resolution: bool = True,
) -> Tuple[Path, Path]:
    """Compute road network density on a regular latitude–longitude grid.

    Parameters
    ----------
    path_roads : str
        Path to a line-based road vector dataset.
    path_region : str
        Path to a polygon dataset delineating the study area.
    res_deg : float
        Grid resolution in decimal degrees.
    out_prefix : str
        Prefix for all output filenames.
    output_dir : str, optional
        Directory where outputs will be written, by default "./data/outputs".
    snap_to_resolution : bool, optional
        Whether to align bounds to multiples of the resolution to prevent floating-point
        mismatches, by default ``True``.

    Returns
    -------
    tuple of pathlib.Path
        Paths to the generated GeoPackage and NetCDF files respectively.
    """
    if res_deg <= 0:
        raise ValueError("Resolution must be strictly positive.")

    roads_path = Path(path_roads)
    region_path = Path(path_region)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_dir = output_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{out_prefix}_road_density_{timestamp}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    LOGGER.addHandler(file_handler)

    try:
        LOGGER.info("Loading input datasets.")
        roads = _prepare_geodataframe(roads_path)
        region = _prepare_geodataframe(region_path)

        # Dissolve region to a single geometry for efficient spatial operations.
        region_union = region.unary_union
        if region_union.is_empty:
            raise ValueError("Region geometry is empty after unary union.")

        LOGGER.info("Clipping road network to study region.")
        roads_clipped = gpd.clip(roads, region_union)
        roads_clipped = roads_clipped.explode(index_parts=False).reset_index(drop=True)

        minx, miny, maxx, maxy = region_union.bounds
        raw_bounds = (minx, miny, maxx, maxy)
        if snap_to_resolution:
            bounds = snap_bounds(raw_bounds, res_deg)
        else:
            bounds = raw_bounds
        min_lon, min_lat, max_lon, max_lat = bounds

        LOGGER.info("Generating base grid covering %.4f° to %.4f°, %.4f° to %.4f°.", *bounds)
        grid = generate_latlon_grid(bounds, res_deg)
        grid = grid.reset_index(drop=True)

        lon_edges = np.arange(min_lon, max_lon + res_deg, res_deg)
        lat_edges = np.arange(min_lat, max_lat + res_deg, res_deg)
        n_lon = len(lon_edges) - 1
        n_lat = len(lat_edges) - 1

        grid["lon_idx"] = grid["cell_id"] // n_lat
        grid["lat_idx"] = grid["cell_id"] % n_lat

        lon_centers = lon_edges[:-1] + res_deg / 2.0
        lat_centers = lat_edges[:-1] + res_deg / 2.0

        geod = Geod(ellps="WGS84")

        areas = np.zeros(len(grid), dtype=float)
        lengths = np.zeros(len(grid), dtype=float)
        densities = np.full(len(grid), np.nan, dtype=float)
        clipped_geometries = [None] * len(grid)
        valid_mask = np.zeros(len(grid), dtype=bool)

        roads_sindex = roads_clipped.sindex if not roads_clipped.empty else None

        LOGGER.info("Computing geodesic areas and road lengths.")
        for idx in tqdm(range(len(grid)), desc="Processing grid cells", unit="cell"):
            cell_geom = grid.geometry.iloc[idx]
            cell_intersection = cell_geom.intersection(region_union)
            if cell_intersection.is_empty:
                continue

            clipped_geometries[idx] = cell_intersection
            area_km2 = area_geodesic_km2(cell_intersection, geod)
            if area_km2 <= 0:
                continue

            areas[idx] = area_km2
            valid_mask[idx] = True

            if roads_sindex is None:
                lengths[idx] = 0.0
                densities[idx] = 0.0
                continue

            candidate_idx = roads_sindex.query(cell_geom, predicate="intersects")
            if len(candidate_idx) == 0:
                lengths[idx] = 0.0
                densities[idx] = 0.0
                continue

            subset = roads_clipped.iloc[candidate_idx]
            intersections = subset.intersection(cell_intersection)
            total_length_m = 0.0
            for geom in intersections:
                if geom.is_empty:
                    continue
                total_length_m += geod.geometry_length(geom)

            length_km = total_length_m / 1000.0
            lengths[idx] = length_km
            densities[idx] = length_km / area_km2 if area_km2 > 0 else 0.0

        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            raise ValueError("No grid cells overlap the study region; check inputs or resolution.")

        vector_grid = gpd.GeoDataFrame(
            {
                "cell_id": grid.loc[valid_mask, "cell_id"].to_numpy(),
                "area_km2": areas[valid_mask],
                "length_km": lengths[valid_mask],
                "density_km_per_km2": densities[valid_mask],
            },
            geometry=[clipped_geometries[i] for i in valid_indices],
            crs="EPSG:4326",
        )

        res_label = _format_resolution_label(res_deg)
        gpkg_path = output_path / f"{out_prefix}_road_density_{res_label}deg.gpkg"
        LOGGER.info("Writing GeoPackage output to %s", gpkg_path)
        vector_grid.to_file(gpkg_path, layer="road_density", driver="GPKG")

        density_grid = np.full((n_lat, n_lon), np.nan, dtype=float)
        area_grid = np.full((n_lat, n_lon), np.nan, dtype=float)
        length_grid = np.full((n_lat, n_lon), np.nan, dtype=float)

        for idx in range(len(grid)):
            lon_idx = int(grid.at[idx, "lon_idx"])
            lat_idx = int(grid.at[idx, "lat_idx"])
            density_grid[lat_idx, lon_idx] = densities[idx]
            area_grid[lat_idx, lon_idx] = areas[idx]
            length_grid[lat_idx, lon_idx] = lengths[idx]

        dataset = xr.Dataset(
            {
                "road_density": (("lat", "lon"), density_grid),
                "road_length": (("lat", "lon"), length_grid),
                "cell_area": (("lat", "lon"), area_grid),
            },
            coords={"lat": lat_centers, "lon": lon_centers},
            attrs={
                "description": "Road density computed on a regular latitude–longitude grid.",
                "source_files": f"roads={roads_path.name}; region={region_path.name}",
                "resolution_deg": res_deg,
                "processing_date": timestamp,
            },
        )

        dataset["road_density"].attrs["units"] = "km/km^2"
        dataset["road_length"].attrs["units"] = "km"
        dataset["cell_area"].attrs["units"] = "km^2"

        netcdf_path = output_path / f"{out_prefix}_road_density_{res_label}deg.nc"
        LOGGER.info("Writing NetCDF output to %s", netcdf_path)
        dataset.to_netcdf(netcdf_path)

        LOGGER.info("Computation finished successfully.")
        return gpkg_path, netcdf_path
    finally:
        LOGGER.removeHandler(file_handler)
        file_handler.close()
