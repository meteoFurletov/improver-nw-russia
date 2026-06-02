"""AIFS data access.

Two complementary routes:

  1. dynamical.org Icechunk Zarr  (PRIMARY, for history)
     The full AIFS ENS archive (2025-07-02 -> present, 51 members, 6-hourly
     inits) served as cloud-optimised Zarr. We open it lazily and subset the
     NW Russia box on read, so the kept volume stays single-digit GB. This is
     the source for backfilling history.

  2. ecmwf-opendata / Herbie  (for the forward daily archive)
     The open-data portal and its AWS/Azure mirrors retain only ~12 runs
     (~2-3 days), so they CANNOT backfill. They are only useful for collecting
     "today" before dynamical.org ingests it (kept here for completeness).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from src.utils.config import get_domain_bounds

# dynamical.org catalog name for the AIFS ENS forecast archive
DYNAMICAL_AIFS_ENS = "ecmwf-aifs-ens-forecast"

# Map our short variable names -> (dynamical var name, CF standard_name, CF units)
AIFS_VARIABLES = {
    "t2m": ("temperature_2m", "air_temperature", "K"),
    "2t": ("temperature_2m", "air_temperature", "K"),
    "air_temperature": ("temperature_2m", "air_temperature", "K"),
}

# Module-level cache: opening the Icechunk store reads metadata (~30s), do it once.
_DYNAMICAL_CACHE: dict[str, "xr.Dataset"] = {}


def open_aifs_ens_dynamical(chunks=None) -> xr.Dataset:
    """Open the dynamical.org AIFS ENS Icechunk Zarr store (lazy, cached).

    Returns the full global dataset; subset before calling ``.load()``.
    Dimensions: (init_time, lead_time, ensemble_member, latitude, longitude).
    """
    if DYNAMICAL_AIFS_ENS not in _DYNAMICAL_CACHE:
        import dynamical_catalog

        logger.info("Opening dynamical.org AIFS ENS store (first call reads metadata)…")
        _DYNAMICAL_CACHE[DYNAMICAL_AIFS_ENS] = dynamical_catalog.open(
            DYNAMICAL_AIFS_ENS, chunks=chunks
        )
    return _DYNAMICAL_CACHE[DYNAMICAL_AIFS_ENS]


def load_aifs_ens(
    init_times: str | datetime | list,
    lead_hours: list[int] | None = None,
    variable: str = "t2m",
    bounds: dict | None = None,
    member_dim: str = "realization",
) -> xr.DataArray:
    """Load AIFS ENS forecasts from dynamical.org, subset to the NW Russia box.

    Args:
        init_times: One or more forecast initialisation times (anything
            ``np.datetime64`` accepts, e.g. "2026-03-15" or a list).
        lead_hours: Forecast lead times in hours (e.g. [24, 48, 72, 96, 120]).
            None keeps all available lead times.
        variable: Short variable key (see ``AIFS_VARIABLES``). Default "t2m".
        bounds: {lat_min, lat_max, lon_min, lon_max}. None -> domain config.
        member_dim: Name to give the ensemble dimension on output. IMPROVER's
            convention is "realization"; the raw store calls it
            "ensemble_member".

    Returns:
        xr.DataArray with dims (init_time?, lead_time, <member_dim>, latitude,
        longitude). Temperature is converted to Kelvin and units set to "K".
        A singleton ``init_time`` is squeezed to a scalar coordinate.
    """
    if variable not in AIFS_VARIABLES:
        raise KeyError(f"Unknown variable '{variable}'. Known: {list(AIFS_VARIABLES)}")
    dvar, std_name, out_units = AIFS_VARIABLES[variable]

    bounds = bounds or get_domain_bounds()
    ds = open_aifs_ens_dynamical()
    da = ds[dvar]

    # --- spatial subset (latitude may be ascending or descending) ---
    lat = da.latitude.values
    lat_slice = (
        slice(bounds["lat_max"], bounds["lat_min"])
        if lat[0] > lat[-1]
        else slice(bounds["lat_min"], bounds["lat_max"])
    )
    da = da.sel(
        latitude=lat_slice,
        longitude=slice(bounds["lon_min"], bounds["lon_max"]),
    )

    # --- init time(s): nearest match so we tolerate "2026-03-15" -> 00Z ---
    inits = np.atleast_1d(np.array(init_times, dtype="datetime64[ns]"))
    da = da.sel(init_time=inits, method="nearest")

    # --- lead times ---
    if lead_hours is not None:
        leads = [np.timedelta64(int(h), "h") for h in lead_hours]
        da = da.sel(lead_time=leads)

    # --- materialise ---
    da = da.load()

    # --- unit conversion: dynamical serves t2m in degree_Celsius ---
    src_units = str(da.attrs.get("units", "")).lower()
    if std_name == "air_temperature" and out_units == "K" and "celsius" in src_units:
        da = da + 273.15
    da.attrs["units"] = out_units
    da.attrs["standard_name"] = std_name
    da.name = variable

    # --- normalise the ensemble dim name to IMPROVER's "realization" ---
    if "ensemble_member" in da.dims and member_dim != "ensemble_member":
        da = da.rename({"ensemble_member": member_dim})

    # squeeze a singleton init_time to a scalar coord (keeps it as metadata)
    if "init_time" in da.dims and da.sizes["init_time"] == 1:
        da = da.squeeze("init_time")

    logger.info(
        f"Loaded AIFS ENS {variable}: dims={dict(da.sizes)} "
        f"units={da.attrs['units']} range=[{float(da.min()):.1f},{float(da.max()):.1f}]"
    )
    return da


def download_aifs_opendata(
    init_date: str | datetime,
    init_hour: str = "00",
    model: str = "aifs-single",
    lead_hours: list[int] | None = None,
    output_dir: str | Path = "data/raw/aifs",
) -> list[Path]:
    """Download AIFS forecast from ECMWF Open Data.

    Args:
        init_date: Date string "YYYYMMDD" or datetime.
        init_hour: "00", "06", "12", or "18".
        model: "aifs-single" (deterministic) or "aifs-ens" (51-member ensemble).
        lead_hours: Specific lead times. None downloads all available.
        output_dir: Local directory for downloaded GRIB2 files.

    Returns:
        List of downloaded file paths.
    """
    from ecmwf.opendata import Client

    output_dir = Path(output_dir) / model
    output_dir.mkdir(parents=True, exist_ok=True)

    client = Client(model=model)

    if isinstance(init_date, datetime):
        init_date = init_date.strftime("%Y%m%d")

    # Build request
    request = {
        "date": init_date,
        "time": int(init_hour),
        "type": "pf" if "ens" in model else "fc",
        "param": ["2t", "10u", "10v", "msl", "tp"],  # 2m temp, 10m winds, MSLP, precip
    }

    if lead_hours is not None:
        request["step"] = lead_hours

    # For ensemble, request all members
    if "ens" in model:
        request["number"] = list(range(1, 51))  # 50 perturbed members

    # Download
    target = output_dir / f"aifs_{init_date}_{init_hour}z.grib2"
    logger.info(f"Downloading {model} for {init_date} {init_hour}Z → {target}")

    try:
        client.retrieve(
            target=str(target),
            **request,
        )
        logger.info(f"Downloaded: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return [target]
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def load_aifs_herbie(
    init_date: str,
    init_hour: str = "00",
    model: str = "aifs",
    product: str = "oper",
    lead_hour: int = 24,
) -> xr.Dataset:
    """Load AIFS forecast using Herbie (convenient for exploration).

    Herbie provides a clean interface to ECMWF open data with automatic
    caching and xarray integration.

    Args:
        init_date: Date string "YYYY-MM-DD".
        init_hour: Init hour.
        model: Herbie model name ("aifs").
        product: "oper" for deterministic, "enfo" for ensemble.
        lead_hour: Forecast lead time in hours.

    Returns:
        xr.Dataset subset to study domain.
    """
    from herbie import Herbie

    bounds = get_domain_bounds()

    H = Herbie(
        f"{init_date} {init_hour}:00",
        model=model,
        product=product,
        fxx=lead_hour,
    )

    # Download and open as xarray
    ds = H.xarray(":2t:", remove_grib=False)

    # Subset to domain
    ds = ds.sel(
        latitude=slice(bounds["lat_max"], bounds["lat_min"]),
        longitude=slice(bounds["lon_min"], bounds["lon_max"]),
    )

    logger.info(f"Loaded AIFS via Herbie: {ds.dims}")
    return ds


def setup_daily_archive(
    output_dir: str | Path = "data/raw/aifs",
    models: list[str] | None = None,
) -> None:
    """Print cron job setup instructions for daily AIFS archiving.

    The ECMWF open data portal only retains ~12 recent runs.
    You MUST set up daily collection to build a training archive.
    """
    if models is None:
        models = ["aifs-single", "aifs-ens"]

    print("=" * 60)
    print("AIFS DAILY ARCHIVE SETUP")
    print("=" * 60)
    print()
    print("The ECMWF open data portal retains only ~2-3 days of data.")
    print("Set up a cron job to collect AIFS forecasts daily:")
    print()
    print("# Add to crontab (crontab -e):")
    print("# Run at 08:00 UTC daily to collect 00Z run")
    for model in models:
        print(
            f"0 8 * * * cd /path/to/project && "
            f"python scripts/download_aifs.py --model {model} --init-hour 00 "
            f"--output-dir {output_dir}"
        )
    print()
    print("Start archiving NOW to maximise your training window.")
    print("=" * 60)
