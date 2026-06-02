"""Convert model data to IMPROVER-compatible Iris cubes.

This is the MOST CRITICAL module in the project. IMPROVER operates
entirely on Iris cubes with specific metadata conventions. Getting
this conversion right is the key to making the whole pipeline work.

Key metadata requirements for IMPROVER:
  - Correct CF standard_name (e.g., "air_temperature")
  - Proper coordinate system (GeogCS with WGS84 ellipsoid)
  - Time coordinates: forecast_reference_time, forecast_period, time
  - Ensemble members as "realization" dimension
  - Model identification via "model_id" attribute (for blending)
  - Units must be CF-compliant (K for temperature, m/s for wind)

Reference: examine the IMPROVER acceptance test data at
  github.com/metoppv/improver_test_data
to understand the exact expected format.
"""

from datetime import datetime, timedelta

import iris
import iris.coords
import iris.cube
import numpy as np
import xarray as xr
from iris.coord_systems import GeogCS
from loguru import logger


# WGS84 ellipsoid parameters
WGS84 = GeogCS(semi_major_axis=6378137.0, semi_minor_axis=6356752.314245)


def xarray_to_iris_cube(
    da: xr.DataArray,
    standard_name: str = "air_temperature",
    units: str = "K",
    model_id: str = "gefs",
    forecast_ref_time: datetime | None = None,
    forecast_period_hours: int | None = None,
    realization: int | None = None,
) -> iris.cube.Cube:
    """Convert a single xarray DataArray field to an IMPROVER-compatible Iris cube.

    This handles a 2D lat-lon field for a single time/member/lead time.
    For ensemble data, call this per-member and merge with iris.cube.CubeList.

    Args:
        da: 2D xr.DataArray with latitude and longitude coordinates.
        standard_name: CF standard name for the variable.
        units: CF units string.
        model_id: Model identifier (used by IMPROVER for blending).
        forecast_ref_time: Initialization time of the forecast.
        forecast_period_hours: Lead time in hours.
        realization: Ensemble member number (0-indexed).

    Returns:
        iris.cube.Cube with IMPROVER-compatible metadata.
    """
    data = da.values.astype(np.float32)

    # --- Build coordinates ---
    lat_points = da.latitude.values.astype(np.float64)
    lon_points = da.longitude.values.astype(np.float64)

    lat_coord = iris.coords.DimCoord(
        lat_points,
        standard_name="latitude",
        units="degrees",
        coord_system=WGS84,
    )
    lon_coord = iris.coords.DimCoord(
        lon_points,
        standard_name="longitude",
        units="degrees",
        coord_system=WGS84,
    )

    # Ensure latitude is monotonically increasing for Iris
    if lat_points[0] > lat_points[-1]:
        lat_coord = lat_coord[::-1]
        data = data[::-1, :]

    # Set bounds for lat/lon
    lat_coord.guess_bounds()
    lon_coord.guess_bounds()

    # --- Create the cube ---
    cube = iris.cube.Cube(
        data,
        standard_name=standard_name,
        units=units,
        dim_coords_and_dims=[(lat_coord, 0), (lon_coord, 1)],
    )

    # --- Time coordinates ---
    time_unit = iris.coords.CellMethod  # placeholder
    epoch = "hours since 1970-01-01 00:00:00"

    if forecast_ref_time is not None:
        # Reference time (init time)
        ref_hours = (forecast_ref_time - datetime(1970, 1, 1)).total_seconds() / 3600
        frt_coord = iris.coords.AuxCoord(
            np.float64(ref_hours),
            standard_name="forecast_reference_time",
            units=epoch,
        )
        cube.add_aux_coord(frt_coord)

    if forecast_period_hours is not None:
        # Lead time
        fp_coord = iris.coords.AuxCoord(
            np.int32(forecast_period_hours),
            standard_name="forecast_period",
            units="hours",
        )
        cube.add_aux_coord(fp_coord)

    if forecast_ref_time is not None and forecast_period_hours is not None:
        # Valid time = ref time + lead time
        valid_time = forecast_ref_time + timedelta(hours=forecast_period_hours)
        valid_hours = (valid_time - datetime(1970, 1, 1)).total_seconds() / 3600
        time_coord = iris.coords.AuxCoord(
            np.float64(valid_hours),
            standard_name="time",
            units=epoch,
        )
        cube.add_aux_coord(time_coord)

    # --- Realization (ensemble member) ---
    if realization is not None:
        real_coord = iris.coords.AuxCoord(
            np.int32(realization),
            standard_name="realization",
            units="1",
        )
        cube.add_aux_coord(real_coord)

    # --- Model identification (critical for multi-model blending) ---
    cube.attributes["model_id"] = model_id
    cube.attributes["model_configuration"] = model_id

    # --- Height coordinate for screen-level variables ---
    if standard_name == "air_temperature":
        height_coord = iris.coords.AuxCoord(
            np.float32(2.0),
            standard_name="height",
            units="m",
        )
        cube.add_aux_coord(height_coord)

    logger.debug(
        f"Created Iris cube: {standard_name}, model={model_id}, "
        f"real={realization}, fp={forecast_period_hours}h, "
        f"shape={cube.shape}"
    )
    return cube


def gefs_xr_to_iris(
    ds: xr.Dataset,
    variable: str = "temperature_2m",
    model_id: str = "gefs",
    init_time: datetime | None = None,
) -> iris.cube.CubeList:
    """Convert a multi-member GEFS xarray Dataset to Iris CubeList.

    Produces one cube per (member, lead_time) combination.
    These can then be merged/concatenated for IMPROVER processing.

    Args:
        ds: xr.Dataset from dynamical.org with dimensions
            (member, lead_time, latitude, longitude).
        variable: Variable name in the dataset.
        model_id: Model identifier for IMPROVER.
        init_time: Initialization time. If None, extracted from dataset.

    Returns:
        iris.cube.CubeList ready for IMPROVER processing.
    """
    cubes = iris.cube.CubeList()

    da = ds[variable]

    # Extract init time from dataset if not provided
    if init_time is None and "init_time" in ds.coords:
        init_time = da.init_time.values.astype("datetime64[s]").astype(datetime)

    members = da.member.values if "member" in da.dims else [0]
    lead_times = da.lead_time.values if "lead_time" in da.dims else [0]

    for member in members:
        for lt in lead_times:
            # Extract 2D field
            field = da.sel(member=member, lead_time=lt)

            # Convert lead_time to hours
            if hasattr(lt, "astype"):
                lt_hours = int(lt.astype("timedelta64[h]").astype(int))
            else:
                lt_hours = int(lt)

            cube = xarray_to_iris_cube(
                da=field,
                standard_name="air_temperature",
                units="K",
                model_id=model_id,
                forecast_ref_time=init_time,
                forecast_period_hours=lt_hours,
                realization=int(member),
            )
            cubes.append(cube)

    logger.info(f"Converted {len(cubes)} cubes from GEFS xarray dataset")
    return cubes


def aifs_ens_to_iris_cube(
    da: xr.DataArray,
    init_time: datetime,
    lead_hours: int,
    model_id: str = "aifs-ens",
    standard_name: str = "air_temperature",
    units: str = "K",
    realization_dim: str = "realization",
) -> iris.cube.Cube:
    """Build a single IMPROVER-valid realization cube for one AIFS ENS forecast.

    Takes a DataArray for ONE (init_time, lead_time) with dims
    ``(realization, latitude, longitude)`` — as returned by
    ``src.data.aifs.load_aifs_ens`` after selecting a single lead — and returns
    one cube whose leading dimension is ``realization`` (IMPROVER's ensemble
    convention), with scalar time/height coords and a ``model_id`` attribute.

    Args:
        da: DataArray with dims (realization, latitude, longitude).
        init_time: forecast_reference_time (forecast init).
        lead_hours: forecast_period in hours.
        model_id: model identifier (for later blending).
        standard_name / units: CF metadata for the variable.
        realization_dim: name of the ensemble dim on ``da``.

    Returns:
        iris.cube.Cube with dim_coords (realization, latitude, longitude).
    """
    if realization_dim not in da.dims:
        raise ValueError(f"Expected '{realization_dim}' dim, got {da.dims}")
    # canonical dim order
    da = da.transpose(realization_dim, "latitude", "longitude")
    data = da.values.astype(np.float32)

    # --- spatial coords (ensure ascending latitude for Iris) ---
    # IMPROVER mandates float32 lat/lon points & bounds.
    lat_pts = da.latitude.values.astype(np.float32)
    lon_pts = da.longitude.values.astype(np.float32)
    lat_coord = iris.coords.DimCoord(
        lat_pts, standard_name="latitude", units="degrees", coord_system=WGS84
    )
    lon_coord = iris.coords.DimCoord(
        lon_pts, standard_name="longitude", units="degrees", coord_system=WGS84
    )
    if lat_pts[0] > lat_pts[-1]:
        lat_coord = lat_coord[::-1]
        data = data[:, ::-1, :]
    lat_coord.guess_bounds()
    lon_coord.guess_bounds()

    # --- realization dim coord ---
    real_pts = da[realization_dim].values.astype(np.int32)
    real_coord = iris.coords.DimCoord(real_pts, standard_name="realization", units="1")

    cube = iris.cube.Cube(
        data,
        standard_name=standard_name,
        units=units,
        dim_coords_and_dims=[(real_coord, 0), (lat_coord, 1), (lon_coord, 2)],
    )

    # --- scalar time coords ---
    # IMPROVER mandates int64 SECONDS-since-epoch for time/frt and int32 seconds
    # for forecast_period (improver.metadata.check_datatypes). Hours/float fail.
    epoch = "seconds since 1970-01-01 00:00:00"
    ref_secs = int((init_time - datetime(1970, 1, 1)).total_seconds())
    valid_time = init_time + timedelta(hours=int(lead_hours))
    valid_secs = int((valid_time - datetime(1970, 1, 1)).total_seconds())
    cube.add_aux_coord(
        iris.coords.AuxCoord(np.int64(ref_secs), standard_name="forecast_reference_time", units=epoch)
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord(np.int32(int(lead_hours) * 3600), standard_name="forecast_period", units="seconds")
    )
    cube.add_aux_coord(
        iris.coords.AuxCoord(np.int64(valid_secs), standard_name="time", units=epoch)
    )
    if standard_name == "air_temperature":
        cube.add_aux_coord(iris.coords.AuxCoord(np.float32(2.0), standard_name="height", units="m"))

    # IMPROVER expects a title attribute and uses model id for blending
    cube.attributes["model_id"] = model_id
    cube.attributes["mosg__model_configuration"] = model_id
    cube.attributes["title"] = f"{model_id} forecast"
    cube.attributes["institution"] = "ECMWF"
    cube.attributes["source"] = "AIFS ENS via dynamical.org"

    logger.debug(
        f"AIFS cube: {standard_name} model={model_id} fp={lead_hours}h "
        f"shape={cube.shape} reals={real_pts.size}"
    )
    return cube


def save_iris_netcdf(cube: iris.cube.Cube, filepath: str) -> None:
    """Save Iris cube to NetCDF in IMPROVER-compatible format.

    Args:
        cube: Iris cube to save.
        filepath: Output path (should end in .nc).
    """
    iris.save(cube, filepath)
    logger.info(f"Saved: {filepath}")


def validate_improver_compatibility(cube: iris.cube.Cube) -> list[str]:
    """Check if an Iris cube meets IMPROVER metadata requirements.

    Returns a list of issues found. Empty list = compatible.
    """
    issues = []

    # Check standard_name
    if cube.standard_name is None:
        issues.append("Missing standard_name")

    # Check coordinate system
    for coord_name in ["latitude", "longitude"]:
        try:
            coord = cube.coord(coord_name)
            if coord.coord_system is None:
                issues.append(f"{coord_name} missing coord_system")
        except iris.exceptions.CoordinateNotFoundError:
            issues.append(f"Missing {coord_name} coordinate")

    # Check time coordinates
    for time_coord in ["forecast_reference_time", "forecast_period", "time"]:
        try:
            cube.coord(time_coord)
        except iris.exceptions.CoordinateNotFoundError:
            issues.append(f"Missing {time_coord} coordinate")

    # Check model_id attribute
    if "model_id" not in cube.attributes:
        issues.append("Missing 'model_id' attribute (needed for blending)")

    # Check units
    if cube.units is None or str(cube.units) == "unknown":
        issues.append("Missing or unknown units")

    if issues:
        logger.warning(f"IMPROVER compatibility issues: {issues}")
    else:
        logger.info("Cube passes IMPROVER compatibility checks")

    return issues
