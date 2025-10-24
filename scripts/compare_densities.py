"""Utility for comparing two road density GeoPackages."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd  # type: ignore[import]
import numpy as np

LOGGER = logging.getLogger(__name__)


def _load_density(path: Path) -> gpd.GeoDataFrame:
    """Load a road density GeoPackage and validate required columns.

    Parameters
    ----------
    path : pathlib.Path
        Location of the GeoPackage to load.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the density layer.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Density file not found: {path}")

    gdf = gpd.read_file(path)
    required_cols = {"cell_id", "density_km_per_km2"}
    missing = required_cols - set(gdf.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return gdf


def compare_densities(first: str, second: str) -> gpd.GeoDataFrame:
    """Merge two road density layers and compute difference and ratio.

    Parameters
    ----------
    first : str
        Path to the reference road density GeoPackage.
    second : str
        Path to the comparison road density GeoPackage.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing joined density values and comparison metrics.
    """
    first_path = Path(first)
    second_path = Path(second)

    gdf_first = _load_density(first_path)
    gdf_second = _load_density(second_path)

    merged = gdf_first.merge(
        gdf_second[["cell_id", "density_km_per_km2"]],
        on="cell_id",
        how="inner",
        suffixes=("_first", "_second"),
    )

    if merged.empty:
        raise ValueError("Merged density grid is empty; ensure both files share cell_id values.")

    merged["density_diff"] = (
        merged["density_km_per_km2_second"] - merged["density_km_per_km2_first"]
    )
    merged["density_ratio"] = np.nan
    nonzero = merged["density_km_per_km2_first"] != 0
    merged.loc[nonzero, "density_ratio"] = (
        merged.loc[nonzero, "density_km_per_km2_second"]
        / merged.loc[nonzero, "density_km_per_km2_first"]
    )

    return merged


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the comparison tool.

    Returns
    -------
    argparse.Namespace
        Parsed arguments populated with user selections.
    """
    parser = argparse.ArgumentParser(
        description="Compare two road density GeoPackages and compute differences.",
    )
    parser.add_argument("--first", required=True, help="Reference GeoPackage path.")
    parser.add_argument("--second", required=True, help="Comparison GeoPackage path.")
    parser.add_argument(
        "--output",
        help="Optional path to write the merged GeoPackage. If omitted, only summary stats are printed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line interface for road density comparison.

    Returns
    -------
    None
        This function is executed for its side effects.
    """
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        merged = compare_densities(args.first, args.second)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Density comparison failed: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info(
        "Difference statistics -- mean: %.6f, min: %.6f, max: %.6f",
        merged["density_diff"].mean(),
        merged["density_diff"].min(),
        merged["density_diff"].max(),
    )

    if args.output:
        output_path = Path(args.output)
        LOGGER.info("Writing comparison GeoPackage to %s", output_path)
        merged.to_file(output_path, layer="density_comparison", driver="GPKG")


if __name__ == "__main__":
    main()
