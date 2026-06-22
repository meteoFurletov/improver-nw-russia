"""Build IMPROVER ``land_binary_mask`` cubes for coastline-aware regridding.

IMPROVER's ``nearest-with-mask`` regrid (``improver regrid --regrid-mode
nearest-with-mask``) needs TWO land-sea masks, both named ``land_binary_mask``
with land=1, sea=0 (sea includes lakes):

  * a SOURCE-grid mask on the AIFS native 0.25deg lat/lon lattice (passed as the
    optional ``land_sea_mask`` argument), and
  * a TARGET-grid mask on the equal-area domain grid from
    :func:`src.utils.grid.make_equal_area_grid` (passed as the ``target_grid``
    argument, replacing the plain zeros grid).

We derive both from the SAME high-resolution coastline (Natural Earth 1:10m
``land`` polygons minus ``lakes`` polygons), classified at each grid-cell centre
by point-in-polygon. Using a sub-kilometre coastline to classify our 20 km /
0.25deg cells means the shorelines (Gulf of Finland, Lake Ladoga, Lake Onega,
White Sea) are resolved as accurately as the grid allows — the resolution limit
is the model/target grid, not the mask source.

Output cubes mirror IMPROVER's own regrid acceptance-test masks
(``glm_landmask.nc`` / ``ukvx_landmask.nc``): ``standard_name="land_binary_mask"``,
``int8``, ``units="1"``, with coordinate bounds.
"""

from __future__ import annotations

from functools import lru_cache

import cartopy.crs as ccrs
import iris
import iris.cube
import numpy as np
import shapely
from cartopy.io import shapereader
from iris.coord_systems import LambertAzimuthalEqualArea
from loguru import logger

# WGS84 ellipsoid (matches src.utils.grid / src.data.iris_convert).
WGS84 = ccrs.Globe(semimajor_axis=6378137.0, semiminor_axis=6356752.314245, ellipse=None)


@lru_cache(maxsize=8)
def _land_geometry(resolution: str, bbox: tuple[float, float, float, float]):
    """Natural Earth land MINUS lakes, clipped to ``bbox`` (lon/lat degrees).

    Args:
        resolution: Natural Earth scale, e.g. "10m" (finest) or "50m".
        bbox: (lon_min, lat_min, lon_max, lat_max) clip window.

    Returns:
        A shapely (multi)polygon of land (lakes carved out) within the window.
    """
    clip = shapely.box(*bbox)
    land_path = shapereader.natural_earth(resolution=resolution, category="physical", name="land")
    lakes_path = shapereader.natural_earth(resolution=resolution, category="physical", name="lakes")

    def _clip_union(path):
        geoms = [g for g in shapereader.Reader(path).geometries() if g.intersects(clip)]
        if not geoms:
            return None
        # buffer(0) repairs any self-intersecting NE polygons before union
        u = shapely.union_all([shapely.make_valid(g) for g in geoms])
        return u.intersection(clip)

    land = _clip_union(land_path)
    if land is None:
        # whole window is sea
        return shapely.Polygon()
    lakes = _clip_union(lakes_path)
    geom = land.difference(lakes) if lakes is not None else land
    logger.info(
        f"Land geometry @ {resolution} clipped to {bbox}: "
        f"land area frac of bbox = {geom.area / clip.area:.3f}"
    )
    return geom


def _cell_centres_lonlat(grid_cube: iris.cube.Cube) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon2d, lat2d) of cell centres for a 2D iris grid cube.

    Handles both a lat/lon (GeogCS) grid and a projection_x/y (LAEA) grid; for
    the latter the centres are inverse-projected to geographic lon/lat.
    """
    names = {c.name() for c in grid_cube.coords(dim_coords=True)}
    if {"latitude", "longitude"} <= names:
        lat = grid_cube.coord("latitude").points.astype(np.float64)
        lon = grid_cube.coord("longitude").points.astype(np.float64)
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lon2d, lat2d
    if {"projection_x_coordinate", "projection_y_coordinate"} <= names:
        xco = grid_cube.coord("projection_x_coordinate")
        yco = grid_cube.coord("projection_y_coordinate")
        cs: LambertAzimuthalEqualArea = xco.coord_system
        proj = ccrs.LambertAzimuthalEqualArea(
            central_latitude=cs.latitude_of_projection_origin,
            central_longitude=cs.longitude_of_projection_origin,
            false_easting=cs.false_easting,
            false_northing=cs.false_northing,
            globe=WGS84,
        )
        xx, yy = np.meshgrid(xco.points.astype(np.float64), yco.points.astype(np.float64))
        lonlat = ccrs.Geodetic(globe=WGS84).transform_points(proj, xx.ravel(), yy.ravel())
        return lonlat[:, 0].reshape(xx.shape), lonlat[:, 1].reshape(xx.shape)
    raise ValueError(f"Unsupported grid coords: {names}")


def land_binary_mask_like(
    grid_cube: iris.cube.Cube,
    resolution: str = "10m",
    pad_deg: float = 1.0,
) -> iris.cube.Cube:
    """Build a ``land_binary_mask`` cube on the same grid as ``grid_cube``.

    Each cell is classified land(1)/sea(0) by testing its centre against a
    high-resolution land geometry (Natural Earth ``resolution`` land minus
    lakes). Lakes and ocean are both sea(0).

    Args:
        grid_cube: 2D iris cube defining the target grid (lat/lon or LAEA x/y).
        resolution: Natural Earth coastline scale used to classify cells.
        pad_deg: degrees of margin added around the grid's lon/lat span when
            clipping the coastline geometry.

    Returns:
        iris.cube.Cube ``land_binary_mask`` (int8, units "1") on ``grid_cube``'s
        grid, with the same dim coords/bounds and coord system.
    """
    lon2d, lat2d = _cell_centres_lonlat(grid_cube)
    bbox = (
        float(lon2d.min()) - pad_deg, float(lat2d.min()) - pad_deg,
        float(lon2d.max()) + pad_deg, float(lat2d.max()) + pad_deg,
    )
    geom = _land_geometry(resolution, bbox)
    mask = shapely.contains_xy(geom, lon2d, lat2d).astype(np.int8)

    # copy() preserves dim coords (with bounds) + coord system; overwrite data.
    cube = grid_cube.copy(data=mask)
    cube.standard_name = "land_binary_mask"
    cube.long_name = None
    cube.var_name = None
    cube.units = "1"
    cube.attributes.clear()
    cube.attributes["title"] = "NW Russia land-sea mask (Natural Earth 10m, lakes=sea)"
    cube.attributes["source"] = f"Natural Earth {resolution} physical land minus lakes"
    cube.attributes["institution"] = "improver-nw-russia"
    # drop any scalar aux coords carried over from a forecast cube
    for ac in list(cube.aux_coords):
        cube.remove_coord(ac)

    land = int(mask.sum())
    total = int(mask.size)
    logger.info(
        f"land_binary_mask {cube.shape}: land={land}/{total} "
        f"({100 * land / total:.1f}% land, {100 * (total - land) / total:.1f}% sea)"
    )
    return cube
