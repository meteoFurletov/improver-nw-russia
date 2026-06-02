#!/usr/bin/env python
"""Download AIFS forecast data from ECMWF Open Data.

IMPORTANT: ECMWF open data only retains ~12 recent forecast runs.
Set this up as a daily cron job to build your training archive.

Usage:
    python scripts/download_aifs.py --model aifs-single --init-hour 00
    python scripts/download_aifs.py --model aifs-ens --init-hour 00
    python scripts/download_aifs.py --setup-cron  # Print cron instructions
"""

import click
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger


@click.command()
@click.option(
    "--model",
    type=click.Choice(["aifs-single", "aifs-ens"]),
    default="aifs-single",
    help="AIFS model variant",
)
@click.option("--init-hour", default="00", help="Init hour (00/06/12/18)")
@click.option("--date", default=None, help="Date (YYYYMMDD). Default: yesterday")
@click.option("--output-dir", default="data/raw/aifs", help="Output directory")
@click.option("--setup-cron", is_flag=True, help="Print cron job setup instructions")
def main(model: str, init_hour: str, date: str, output_dir: str, setup_cron: bool):
    """Download AIFS forecast data."""
    if setup_cron:
        from src.data.aifs import setup_daily_archive

        setup_daily_archive(output_dir)
        return

    if date is None:
        date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")

    from src.data.aifs import download_aifs_opendata

    try:
        files = download_aifs_opendata(
            init_date=date,
            init_hour=init_hour,
            model=model,
            output_dir=output_dir,
        )
        logger.info(f"Downloaded {len(files)} files")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


if __name__ == "__main__":
    main()
