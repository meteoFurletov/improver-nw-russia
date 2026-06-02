"""Grid utilities for the NW Russia domain.

IMPROVER neighbourhood processing operates on an equal-area grid whose x/y
coordinates are in metres (a metre radius is meaningless on a lat/lon grid).
This module builds a Lambert Azimuthal Equal Area target grid covering the
domain, used as the ``target-grid`` argument to ``improver regrid``.
"""

from __future__ import annotations

import cartopy.crs as ccrs
import iris
import iris.coords
import iris.cube
import numpy as np
from iris.coord_systems import GeogCS, LambertAzimuthalEqualArea
from loguru import logger

from src.utils.config import get_domain_bounds

# WGS84 ellipsoid (matches the source lat/lon cubes)
WGS84 = GeogCS(semi_major_axis=6378137.0, semi_minor_axis=6356752.314245)


def make_equal_area_grid(
    bounds: dict | None = None,
    resolution_m: float = 20000.0,
    central_lat: float | None = None,
    central_lon: float | None = None,
    pad_cells: int = 1,
) -> iris.cube.Cube:
    """Build a Lambert Azimuthal Equal Area target grid covering the domain.

    Args:
        bounds: {lat_min, lat_max, lon_min, lon_max}. None -> domain config.
        resolution_m: grid spacing in metres (default 20 km ~ AIFS 0.25deg).
        central_lat / central_lon: projection origin. None -> domain centre.
        pad_cells: extra cells added around the projected bounding box.

    Returns:
        iris.cube.Cube with dims (projection_y_coordinate, projection_x_coordinate)
        in metres and a LambertAzimuthalEqualArea coord system. Data is zeros;
        only the coordinates matter for regridding.
    """
    bounds = bounds or get_domain_bounds()
    lat0 = central_lat if central_lat is not None else 0.5 * (bounds["lat_min"] + bounds["lat_max"])
    lon0 = central_lon if central_lon is not None else 0.5 * (bounds["lon_min"] + bounds["lon_max"])

    proj = ccrs.LambertAzimuthalEqualArea(
        central_latitude=lat0, central_longitude=lon0
    )
    geodetic = ccrs.Geodetic()

    # project a dense set of boundary points to find the x/y extent
    lons = np.linspace(bounds["lon_min"], bounds["lon_max"], 50)
    lats = np.linspace(bounds["lat_min"], bounds["lat_max"], 50)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    xyz = proj.transform_points(geodetic, grid_lon.ravel(), grid_lat.ravel())
    xs, ys = xyz[:, 0], xyz[:, 1]

    def _axis(lo, hi):
        lo = np.floor(lo / resolution_m - pad_cells) * resolution_m
        hi = np.ceil(hi / resolution_m + pad_cells) * resolution_m
        return np.arange(lo, hi + resolution_m, resolution_m, dtype=np.float32)

    x_pts = _axis(xs.min(), xs.max())
    y_pts = _axis(ys.min(), ys.max())

    cs = LambertAzimuthalEqualArea(
        latitude_of_projection_origin=lat0,
        longitude_of_projection_origin=lon0,
        false_easting=0.0,
        false_northing=0.0,
        ellipsoid=WGS84,
    )
    x_coord = iris.coords.DimCoord(
        x_pts, standard_name="projection_x_coordinate", units="m", coord_system=cs
    )
    y_coord = iris.coords.DimCoord(
        y_pts, standard_name="projection_y_coordinate", units="m", coord_system=cs
    )
    x_coord.guess_bounds()
    y_coord.guess_bounds()

    cube = iris.cube.Cube(
        np.zeros((y_pts.size, x_pts.size), dtype=np.float32),
        long_name="target_grid",
        units="1",
        dim_coords_and_dims=[(y_coord, 0), (x_coord, 1)],
    )
    logger.info(
        f"Equal-area target grid: {y_pts.size}x{x_pts.size} @ {resolution_m/1000:.0f}km "
        f"origin=({lat0:.1f},{lon0:.1f})"
    )
    return cube
