#!/usr/bin/env python
"""Build + validate the land-sea masks for coastline-aware (nearest-with-mask) regrid.

Produces, for the tune-regrid step:
  * data/processed/skeleton/masks/landmask_target_equalarea.nc  (target_grid arg)
  * data/processed/skeleton/masks/landmask_source_aifs.nc        (land_sea_mask arg)
  * data/processed/skeleton/masks/masks_overview.png             (resolution check)

Prints overall land/sea fractions and a per-water-body sea-cell count on the
TARGET (20 km equal-area) grid, so we can confirm the Gulf of Finland, Lake
Ladoga, Lake Onega and the White Sea are actually resolved BEFORE running the
regrid sweep.

    python scripts/build_landmask.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import iris
import numpy as np
from loguru import logger

warnings.filterwarnings("ignore")

from src.data.iris_convert import save_iris_netcdf  # noqa: E402
from src.data.landmask import _cell_centres_lonlat, land_binary_mask_like  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.grid import make_equal_area_grid  # noqa: E402

# Named coastal water bodies (the headroom for nearest-with-mask) as lon/lat
# boxes within the domain: (lon_min, lon_max, lat_min, lat_max).
WATER_BODIES = {
    "Gulf of Finland (E)": (28.0, 30.3, 59.5, 60.5),
    "Lake Ladoga":         (30.0, 33.0, 60.0, 61.8),
    "Lake Onega":          (34.3, 36.6, 60.8, 62.9),
    "White Sea":           (34.0, 41.0, 63.5, 66.5),
}

MASK_DIR = Path("data/processed/skeleton/masks")
SOURCE_REALIZATION = Path("data/processed/skeleton/2026031500/lead024/01_realization.nc")


def _source_grid_cube() -> iris.cube.Cube:
    """A clean 2D AIFS-native lat/lon grid cube (the regrid source grid)."""
    cube = iris.load_cube(str(SOURCE_REALIZATION))
    twod = cube[0]  # drop realization -> (lat, lon)
    for ac in list(twod.aux_coords):
        twod.remove_coord(ac)
    return twod


def _box_report(name: str, mask: iris.cube.Cube) -> str:
    lon2d, lat2d = _cell_centres_lonlat(mask)
    data = np.asarray(mask.data)
    lines = [f"  per-water-body sea cells on the {name} grid:"]
    for wb, (lo, hi, la, lb) in WATER_BODIES.items():
        inbox = (lon2d >= lo) & (lon2d <= hi) & (lat2d >= la) & (lat2d <= lb)
        n = int(inbox.sum())
        sea = int((data[inbox] == 0).sum())
        frac = sea / n if n else float("nan")
        lines.append(f"    {wb:<22} cells={n:>4}  sea={sea:>4} ({100*frac:5.1f}% sea)")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config("aifs_skeleton")
    bounds = cfg["domain"]
    res_m = cfg["target_grid"]["resolution_m"]
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # --- target (equal-area 20 km) mask — the grid the pipeline regrids onto ---
    target_grid = make_equal_area_grid(bounds=bounds, resolution_m=res_m)
    target_mask = land_binary_mask_like(target_grid, resolution="10m")
    target_path = MASK_DIR / "landmask_target_equalarea.nc"
    save_iris_netcdf(target_mask, str(target_path))

    # --- source (AIFS native 0.25deg) mask — passed as land_sea_mask ---
    source_mask = land_binary_mask_like(_source_grid_cube(), resolution="10m")
    source_path = MASK_DIR / "landmask_source_aifs.nc"
    save_iris_netcdf(source_mask, str(source_path))

    # --- numeric resolution check ---
    def frac(m):
        d = np.asarray(m.data); return 100 * int(d.sum()) / d.size
    print("\n=== LAND-SEA MASK RESOLUTION CHECK ===")
    print(f"target equal-area grid: {target_mask.shape} @ {res_m/1000:.0f} km  "
          f"-> {frac(target_mask):.1f}% land / {100-frac(target_mask):.1f}% sea")
    print(f"source AIFS grid:       {source_mask.shape} @ 0.25deg           "
          f"-> {frac(source_mask):.1f}% land / {100-frac(source_mask):.1f}% sea")
    print(_box_report("TARGET equal-area", target_mask))
    print(_box_report("SOURCE AIFS", source_mask))

    # --- figure ---
    _plot(target_mask, source_mask, MASK_DIR / "masks_overview.png")
    print(f"\nSaved:\n  {target_path}\n  {source_path}\n  {MASK_DIR/'masks_overview.png'}")


def _plot(target_mask, source_mask, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    pc = ccrs.PlateCarree()
    cmap = ListedColormap(["#2c6fa8", "#d9c79a"])  # 0=sea (blue), 1=land (tan)
    lakes = cfeature.NaturalEarthFeature("physical", "lakes", "10m",
                                         edgecolor="navy", facecolor="none", linewidth=0.7)

    def _draw(ax, mask, title, extent=None, label_boxes=None, grid_edges=True):
        lon2d, lat2d = _cell_centres_lonlat(mask)
        ax.pcolormesh(lon2d, lat2d, np.asarray(mask.data), cmap=cmap, vmin=0, vmax=1,
                      transform=pc, shading="nearest",
                      edgecolors="white" if grid_edges else "none", linewidth=0.15)
        ax.coastlines("10m", color="k", linewidth=0.8)
        ax.add_feature(lakes)
        ax.set_extent(extent or [lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()], crs=pc)
        gl = ax.gridlines(draw_labels=True, alpha=0.25, linewidth=0.4)
        gl.top_labels = gl.right_labels = False
        for wb, (lo, hi, la, lb) in WATER_BODIES.items():
            ax.add_patch(Rectangle((lo, la), hi - lo, lb - la, transform=pc,
                                   fill=False, edgecolor="red", linewidth=1.2, zorder=5))
            if label_boxes and wb in label_boxes:
                ax.text(0.5 * (lo + hi), 0.5 * (la + lb), wb, transform=pc, fontsize=8,
                        ha="center", va="center", color="darkred", zorder=6, clip_on=True,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        ax.set_title(title, fontsize=10)

    fig = plt.figure(figsize=(12.5, 11))
    ax1 = fig.add_subplot(2, 2, 1, projection=pc)
    _draw(ax1, target_mask, f"TARGET: equal-area 20 km {target_mask.shape} (target_grid arg)",
          label_boxes=set(WATER_BODIES))
    ax2 = fig.add_subplot(2, 2, 2, projection=pc)
    _draw(ax2, source_mask, f"SOURCE: AIFS 0.25deg {source_mask.shape} (land_sea_mask arg)",
          label_boxes={"White Sea"})
    ax3 = fig.add_subplot(2, 2, 3, projection=pc)
    _draw(ax3, target_mask, "TARGET zoom: Gulf of Finland / Ladoga / Onega",
          extent=[28, 37.5, 59, 63.3], label_boxes={"Lake Ladoga", "Lake Onega"})
    ax4 = fig.add_subplot(2, 2, 4, projection=pc)
    _draw(ax4, target_mask, "TARGET zoom: White Sea",
          extent=[33, 42, 63, 67], label_boxes={"White Sea"})

    fig.suptitle("NW Russia land-sea masks (sea=blue incl. lakes, land=tan; red=named water bodies)\n"
                 "Natural Earth 10m coastline, classified at each grid-cell centre",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
