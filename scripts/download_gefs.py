#!/usr/bin/env python
"""Download GEFS data for the study domain.

Usage:
    python scripts/download_gefs.py --start 2022-01-01 --end 2022-01-31
    python scripts/download_gefs.py --start 2022-01-01 --end 2022-12-31 --init-hour 00
"""

import click
import xarray as xr
from datetime import datetime
from pathlib import Path
from loguru import logger


@click.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--init-hour", default="00", help="Initialization hour (00/06/12/18)")
@click.option("--output-dir", default="data/raw/gefs", help="Output directory")
@click.option("--variable", default="temperature_2m", help="Variable to download")
@click.option("--format", "fmt", type=click.Choice(["zarr", "grib"]), default="zarr",
              help="Download format: zarr (dynamical.org) or grib (AWS S3)")
def main(start: str, end: str, init_hour: str, output_dir: str, variable: str, fmt: str):
    """Download GEFS forecast data for NW Russia study domain."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "zarr":
        download_zarr(start, end, init_hour, variable, output_dir)
    else:
        download_grib(start, end, init_hour, output_dir)


def download_zarr(start, end, init_hour, variable, output_dir):
    """Download via dynamical.org Zarr (recommended)."""
    from src.data.gefs import load_gefs_zarr

    logger.info(f"Loading GEFS Zarr: {start} to {end}, {init_hour}Z")

    # For Zarr, we load and save locally as NetCDF
    import pandas as pd
    dates = pd.date_range(start, end, freq="1D")

    for date in dates:
        init_str = f"{date.strftime('%Y-%m-%d')}T{init_hour}"
        out_file = output_dir / f"gefs_{date.strftime('%Y%m%d')}_{init_hour}z.nc"

        if out_file.exists():
            logger.info(f"Skipping {out_file} (exists)")
            continue

        try:
            ds = load_gefs_zarr(init_time=init_str, variable=variable)
            ds = ds.compute()
            ds.to_netcdf(out_file)
            logger.info(f"Saved: {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            logger.warning(f"Failed for {init_str}: {e}")


def download_grib(start, end, init_hour, output_dir):
    """Download via AWS S3 GRIB2 files."""
    from src.data.gefs import download_gefs_grib

    import pandas as pd
    dates = pd.date_range(start, end, freq="1D")

    for date in dates:
        date_str = date.strftime("%Y%m%d")
        try:
            download_gefs_grib(date_str, init_hour, output_dir=output_dir)
        except Exception as e:
            logger.warning(f"Failed for {date_str}: {e}")


if __name__ == "__main__":
    main()
