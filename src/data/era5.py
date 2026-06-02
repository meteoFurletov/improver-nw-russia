"""ERA5 data access for gridded verification truth.

ERA5 serves as the primary gridded verification reference. Data is stored in
Zarr format, enabling lazy chunked access; we subset the NW Russia box on read.

SOURCE NOTE: the WeatherBench-2 ERA5 stores end in 2023, so they CANNOT verify
the AIFS ENS period (2025-07 -> present). We use Google's ARCO-ERA5 instead,
which is updated to near-present (ERA5T, ~1 week lag). As of 2026-06 it covers
through 2026-05-26 — comfortably spanning the tune/test windows.
"""

from datetime import datetime

import numpy as np
import xarray as xr
from loguru import logger

from src.utils.config import get_domain_bounds

# ARCO-ERA5 (Google Analysis-Ready Cloud-Optimized ERA5), hourly 0.25deg, Kelvin.
# Updated to near-present (ERA5T). Variables are CF-named; 2m_temperature in K.
ARCO_ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Legacy WeatherBench-2 stores (end 2023 — kept only for pre-2024 work).
ERA5_FULL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
ERA5_WB13 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# Module-level cache: opening ARCO-ERA5 metadata is slow (~1-2 min), do it once.
_ARCO_CACHE: dict[str, "xr.Dataset"] = {}


def open_arco_era5() -> xr.Dataset:
    """Open the ARCO-ERA5 analysis-ready Zarr store (lazy, cached)."""
    if ARCO_ERA5 not in _ARCO_CACHE:
        logger.info("Opening ARCO-ERA5 (first call reads metadata, ~1-2 min)…")
        _ARCO_CACHE[ARCO_ERA5] = xr.open_zarr(
            ARCO_ERA5, chunks=None, storage_options={"token": "anon"}
        )
    return _ARCO_CACHE[ARCO_ERA5]


def load_era5_truth(
    valid_times,
    variable: str = "2m_temperature",
    bounds: dict | None = None,
    like: "xr.DataArray | None" = None,
) -> xr.DataArray:
    """Load ERA5 truth for the given forecast valid-times over NW Russia.

    Args:
        valid_times: One or more valid times (init + lead), anything
            ``np.datetime64`` accepts. ARCO-ERA5 is hourly, so 00/06/12/18Z
            valid times match exactly.
        variable: ARCO-ERA5 variable name (default "2m_temperature", Kelvin).
        bounds: {lat_min, lat_max, lon_min, lon_max}. None -> domain config.
        like: optional AIFS DataArray; if given, ERA5 is reindexed onto its
            exact lat/lon grid (nearest) so the two co-grid for verification.

    Returns:
        xr.DataArray of truth on the NW Russia box, indexed by ``time``.
    """
    bounds = bounds or get_domain_bounds()
    ds = open_arco_era5()
    da = ds[variable]

    lat = da.latitude.values
    lat_slice = (
        slice(bounds["lat_max"], bounds["lat_min"])
        if lat[0] > lat[-1]
        else slice(bounds["lat_min"], bounds["lat_max"])
    )
    da = da.sel(latitude=lat_slice, longitude=slice(bounds["lon_min"], bounds["lon_max"]))

    times = np.atleast_1d(np.array(valid_times, dtype="datetime64[ns]"))
    da = da.sel(time=times, method="nearest").load()

    if like is not None:
        # co-grid onto the AIFS subset (same 0.25deg lattice -> exact match)
        da = da.reindex(latitude=like.latitude, longitude=like.longitude, method="nearest")

    da.attrs.setdefault("units", "K")
    da.attrs["standard_name"] = "air_temperature"
    da.name = variable
    logger.info(
        f"Loaded ERA5 truth {variable}: dims={dict(da.sizes)} "
        f"range=[{float(da.min()):.1f},{float(da.max()):.1f}]"
    )
    return da


def load_era5(
    start: str | datetime,
    end: str | datetime,
    variable: str = "2m_temperature",
    source: str = "wb13",
) -> xr.DataArray:
    """Load ERA5 data from WeatherBench 2 for the study domain.

    Args:
        start: Start date, e.g. "2022-01-01".
        end: End date, e.g. "2022-12-31".
        variable: ERA5 variable name. Common options:
            - "2m_temperature"
            - "10m_u_component_of_wind"
            - "10m_v_component_of_wind"
            - "mean_sea_level_pressure"
            - "total_precipitation_6hr" (derived)
        source: "wb13" (6-hourly, 13 levels) or "full" (hourly, 37 levels).

    Returns:
        xr.DataArray subset to domain and time range.
    """
    bounds = get_domain_bounds()
    url = ERA5_WB13 if source == "wb13" else ERA5_FULL

    logger.info(f"Opening ERA5 ({source}) from GCS")
    ds = xr.open_zarr(url, chunks="auto")

    # Subset spatially
    da = ds[variable].sel(
        latitude=slice(bounds["lat_max"], bounds["lat_min"]),
        longitude=slice(bounds["lon_min"], bounds["lon_max"]),
    )

    # Subset temporally
    da = da.sel(time=slice(start, end))

    logger.info(
        f"Loaded ERA5 {variable}: "
        f"time={da.time.size}, lat={da.latitude.size}, lon={da.longitude.size}"
    )
    return da


def load_era5_climatology(
    variable: str = "2m_temperature",
    clim_years: tuple[int, int] = (1991, 2020),
) -> xr.DataArray:
    """Compute ERA5 climatology for skill score reference.

    Computes day-of-year mean and standard deviation over the
    climatological period, smoothed with a 15-day running mean.

    Args:
        variable: ERA5 variable name.
        clim_years: Start and end year for climatology.

    Returns:
        xr.Dataset with 'mean' and 'std' fields indexed by dayofyear.
    """
    bounds = get_domain_bounds()

    logger.info(f"Computing ERA5 climatology ({clim_years[0]}-{clim_years[1]})")
    ds = xr.open_zarr(ERA5_WB13, chunks="auto")

    da = ds[variable].sel(
        latitude=slice(bounds["lat_max"], bounds["lat_min"]),
        longitude=slice(bounds["lon_min"], bounds["lon_max"]),
        time=slice(str(clim_years[0]), str(clim_years[1])),
    )

    # Group by day of year and compute statistics
    clim_mean = da.groupby("time.dayofyear").mean("time")
    clim_std = da.groupby("time.dayofyear").std("time")

    # Smooth with 15-day rolling mean to remove noise
    clim_mean = clim_mean.rolling(dayofyear=15, center=True, min_periods=1).mean()
    clim_std = clim_std.rolling(dayofyear=15, center=True, min_periods=1).mean()

    result = xr.Dataset({"mean": clim_mean, "std": clim_std})
    logger.info("Climatology computed")
    return result
