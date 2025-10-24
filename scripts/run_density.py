"""Command-line interface for the road density workflow."""

from __future__ import annotations

import argparse
import logging
import sys

from scripts.compute_road_density import compute_road_density


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the density workflow.

    Returns
    -------
    argparse.Namespace
        Parsed arguments populated with user selections.
    """
    parser = argparse.ArgumentParser(
        description="Compute road network density on a latitude–longitude grid.",
    )
    parser.add_argument(
        "--roads",
        required=True,
        help="Path to the road network dataset (GeoPackage, GeoJSON, or Shapefile).",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Path to the polygon dataset defining the study region.",
    )
    parser.add_argument(
        "--res",
        required=True,
        type=float,
        help="Grid resolution in decimal degrees (e.g., 0.01).",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix to use for generated output file names.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/outputs",
        help="Directory where outputs will be written (default: ./data/outputs).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the command-line workflow for road density computation.

    Returns
    -------
    None
        This function is invoked for its side effects.
    """
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        gpkg_path, netcdf_path = compute_road_density(
            path_roads=args.roads,
            path_region=args.region,
            res_deg=args.res,
            out_prefix=args.prefix,
            output_dir=args.output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Road density computation failed: %s", exc)
        raise SystemExit(1) from exc

    logging.info("GeoPackage written to %s", gpkg_path)
    logging.info("NetCDF written to %s", netcdf_path)


if __name__ == "__main__":
    main()
