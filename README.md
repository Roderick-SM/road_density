# Road Density

This repository computes road network density (km/km²) on regular latitude–longitude grids for any vector road dataset and study region polygon. The workflow clips roads to a region, builds a geodesic-aware grid, and exports density surfaces in GeoPackage and NetCDF formats.

## Features
- Automatic CRS harmonization to EPSG:4326 for both roads and region inputs.
- Regular latitude–longitude grid generation at configurable resolutions (0.005°, 0.01°, 0.05°, 0.1°).
- Geodesic cell area and road length calculations for accurate density estimates.
- Deterministic outputs in GeoPackage and NetCDF formats with metadata.
- CLI and Python API entry points with logging and progress bars.
- Optional GeoPackage comparison utility for density deltas and ratios.

## Quick Start
1. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate road_density
   ```
2. Populate `data/inputs/` with your road vector dataset (GeoPackage, GeoJSON, or Shapefile) and region polygon shapefile.
3. Run the density computation:
   ```bash
   python scripts/run_density.py \
       --roads data/inputs/roads_input.gpkg \
       --region data/inputs/region_mask.shp \
       --res 0.01 \
        --prefix MyRegion_Roads
   ```
   Add `--no-snap` if the supplied bounds are already aligned and you prefer exact extents.
4. Review outputs in `data/outputs/` (GeoPackage, NetCDF, and logs).

## Python API Example
```python
from scripts.compute_road_density import compute_road_density

compute_road_density(
    path_roads="data/inputs/roads_input.gpkg",
    path_region="data/inputs/region_mask.shp",
    res_deg=0.01,
    out_prefix="MyRegion_Roads",
)
```

## Project Structure
```
road_density/
├── README.md
├── environment.yml
├── data/
│   ├── inputs/
│   │   ├── roads_input.gpkg
│   │   └── region_mask.shp
│   └── outputs/
│       ├── road_density_<res>deg.gpkg
│       ├── road_density_<res>deg.nc
│       └── logs/
├── scripts/
│   ├── compute_road_density.py
│   ├── run_density.py
│   └── utils/
│       └── grid_tools.py
└── docs/
    └── Example_README.md
```

## Testing
- Provide sample inputs in `data/inputs/`.
- Run the CLI script and inspect summary logs.

## License
Specify the desired license terms for your distribution.
