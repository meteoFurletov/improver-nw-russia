"""GEFS data access from dynamical.org (Zarr) and AWS S3 (GRIB2).

dynamical.org provides GEFS in cloud-optimised Zarr format, which allows
lazy loading of regional subsets without downloading full global files.
"""

from datetime import datetime
from pathlib import Path

import xarray as xr
from loguru import logger

from src.utils.config import get_domain_bounds

# Cloud-native Zarr source (preferred)
GEFS_ZARR_URL = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"

# AWS S3 for historical GRIB2 files
GEFS_S3_BUCKET = "s3://noaa-gefs-pds"
GEFS_REFORECAST_S3 = "s3://noaa-gefs-retrospective"


def load_gefs_zarr(
    init_time: str | datetime,
    variable: str = "temperature_2m",
    members: list[int] | None = None,
    lead_hours: list[int] | None = None,
) -> xr.Dataset:
    """Load GEFS forecast data from dynamical.org Zarr store.

    This is the recommended access method — lazy chunked loading,
    no GRIB2 conversion needed, subset before download.

    Args:
        init_time: Forecast initialization time, e.g. "2024-01-01T00".
        variable: Variable name in dynamical.org schema.
        members: List of ensemble member indices (0=control, 1-30=perturbed).
            None loads all 31 members.
        lead_hours: List of lead times in hours. None loads all.

    Returns:
        xr.Dataset with the requested data, spatially subset to the study domain.
    """
    bounds = get_domain_bounds()
    logger.info(f"Loading GEFS Zarr for init={init_time}, var={variable}")

    ds = xr.open_zarr(GEFS_ZARR_URL, chunks="auto")

    # Spatial subset to NW Russia domain
    ds = ds.sel(
        latitude=slice(bounds["lat_max"], bounds["lat_min"]),
        longitude=slice(bounds["lon_min"], bounds["lon_max"]),
    )

    # Select variable
    if variable in ds:
        ds = ds[[variable]]
    else:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{variable}' not found. Available: {available}")

    # Filter by init time
    if init_time is not None:
        ds = ds.sel(init_time=init_time, method="nearest")

    # Filter by ensemble member
    if members is not None:
        ds = ds.sel(member=members)

    # Filter by lead time
    if lead_hours is not None:
        ds = ds.sel(lead_time=[f"{h}h" for h in lead_hours])

    logger.info(f"Loaded GEFS: {ds.dims}")
    return ds


def download_gefs_grib(
    init_date: str,
    init_hour: str = "00",
    members: list[int] | None = None,
    output_dir: str | Path = "data/raw/gefs",
) -> list[Path]:
    """Download GEFS GRIB2 files from AWS S3 for a specific init time.

    Use this when you need the raw GRIB2 files (e.g., for IMPROVER's
    standardise step or when dynamical.org doesn't have the date).

    Args:
        init_date: Date string, e.g. "20240101".
        init_hour: Init hour, e.g. "00", "06", "12", "18".
        members: Member indices. None downloads all 31.
        output_dir: Local directory for downloaded files.

    Returns:
        List of downloaded file paths.
    """
    import s3fs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = s3fs.S3FileSystem(anon=True)

    if members is None:
        members = list(range(31))

    downloaded = []
    for m in members:
        member_str = f"gec00" if m == 0 else f"gep{m:02d}"
        s3_path = f"noaa-gefs-pds/gefs.{init_date}/{init_hour}/atmos/pgrb2sp25/{member_str}.t{init_hour}z.pgrb2s.0p25.f*"

        matching = fs.glob(s3_path)
        for remote in matching:
            local = output_dir / Path(remote).name
            if not local.exists():
                logger.info(f"Downloading {remote}")
                fs.get(remote, str(local))
                downloaded.append(local)
            else:
                downloaded.append(local)

    logger.info(f"Downloaded {len(downloaded)} GEFS GRIB2 files")
    return downloaded


def load_gefs_reforecast(
    date: str,
    variable: str = "tmp_2m",
    output_dir: str | Path = "data/raw/gefs_reforecast",
) -> xr.Dataset:
    """Load GEFSv12 reforecast data from AWS S3.

    The reforecast archive (2000-2019) has 5 members daily and 11 members
    weekly. Essential for training EMOS coefficients over a long history.

    Args:
        date: Date string, e.g. "20100101".
        variable: Variable name in reforecast schema.
        output_dir: Local cache directory.

    Returns:
        xr.Dataset with reforecast data.
    """
    # TODO: Implement reforecast loading
    # S3 path pattern: s3://noaa-gefs-retrospective/GEFSv12/reforecast/{YYYY}/{YYYY}{MM}{DD}00/
    raise NotImplementedError(
        "Reforecast loading not yet implemented. "
        "See: https://noaa-gefs-retrospective.s3.amazonaws.com/Description_of_reforecast_data.pdf"
    )
