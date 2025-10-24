# Road Density Datasets

## Description
This repository computes road density (km/km²) on regular latitude–longitude grids using any road dataset.

## Inputs
- `roads_input.gpkg` or `.geojson`: line geometries of road networks (e.g., OSM, Microsoft AI).
- `region_mask.shp`: polygon(s) defining the study area.

## Workflow
1. Reproject all inputs to EPSG:4326.
2. Clip roads to the region polygon.
3. Build lat/lon grid (0.005°, 0.01°, 0.05°, or 0.1°).
4. Compute geodesic area (km²) for each grid cell.
5. Measure total road length (km) within each cell.
6. Compute density = length_km / area_km2.

## Outputs
- `road_density_<res>deg.gpkg` — vector format with all fields.
- `road_density_<res>deg.nc` — NetCDF raster.

## Units
Road density in **km/km²**.

## CRS
All outputs in EPSG:4326 (WGS 84 latitude–longitude).

## Example
```bash
python scripts/run_density.py --roads data/inputs/roads_input.gpkg --region data/inputs/region_mask.shp --res 0.01 --prefix MyRegion_Roads
```
