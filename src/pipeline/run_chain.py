"""End-to-end IMPROVER walking-skeleton chain for AIFS ENS over NW Russia.

Phase 0: every stage at its simplest setting. The chain runs all seven stages
for one (init_date, lead_time) and saves an intermediate NetCDF per stage so
each stage's contribution can be verified in isolation (Phase 1 tuning).

Stages
  1. Ensemble input      AIFS ENS 51 members -> IMPROVER-valid realization cube
  2. Standardise & regrid bilinear onto an equal-area (metre) target grid
  3. Probabilities       realizations -> percentiles; thresholds -> probability
  4. Calibration         pass-through (no EMOS in Phase 0)
  5. Neighbourhood       square neighbourhood (radius in metres)
  6. Blend               pass-through (single model -> inert)
  7. Products            (gridded) final probability product

The raw realization cube from stage 1 is the BASELINE scored against ERA5;
stages 2/5 are the first tuning levers.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.data.aifs import load_aifs_ens
from src.data.iris_convert import aifs_ens_to_iris_cube, save_iris_netcdf
from src.utils.config import load_config
from src.utils.grid import make_equal_area_grid

# Resolve the `improver` CLI next to the running interpreter (conda env bin),
# falling back to PATH. shutil.which alone fails when the env bin isn't on PATH.
_candidate = Path(sys.executable).parent / "improver"
IMPROVER = str(_candidate) if _candidate.exists() else (shutil.which("improver") or "improver")


def _improver(*args: str) -> None:
    """Run an `improver` CLI command, raising with stderr on failure."""
    cmd = [IMPROVER, *map(str, args)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"improver {args[0]} failed (exit {res.returncode}):\n{res.stderr[-1500:]}"
        )


def run_skeleton_for_date(
    init_date: str | datetime,
    config: str = "aifs_skeleton",
    profile: str = "skeleton",
    outdir: str | Path = "data/processed",
) -> dict:
    """Run the full skeleton chain for one init date across all lead times.

    Returns a dict {lead_hours: {stage_name: path}} of stage outputs.
    """
    cfg = load_config(config)
    prof = cfg["profiles"][profile]
    bounds = cfg["domain"]
    leads = cfg["lead_times_hours"]
    percentiles = cfg["percentiles"]
    thresholds_k = [c + 273.15 for c in cfg["thresholds_celsius"]]

    init = datetime.fromisoformat(str(init_date)) if not isinstance(init_date, datetime) else init_date
    tag = init.strftime("%Y%m%d%H")
    run_dir = Path(outdir) / profile / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Equal-area target grid (shared across leads) — needed for neighbourhood.
    grid_path = run_dir / "target_grid.nc"
    res_m = cfg["target_grid"]["resolution_m"]
    save_iris_netcdf(make_equal_area_grid(bounds=bounds, resolution_m=res_m), str(grid_path))

    results: dict = {}
    for lead in leads:
        ld = run_dir / f"lead{lead:03d}"
        ld.mkdir(exist_ok=True)
        out: dict = {}

        # --- Stage 1: ensemble input -> realization cube (BASELINE, native grid) ---
        da = load_aifs_ens(init, lead_hours=[lead], variable=cfg["variable"], bounds=bounds)
        if "lead_time" in da.dims:
            da = da.isel(lead_time=0)
        cube = aifs_ens_to_iris_cube(da, init_time=init, lead_hours=lead, model_id=cfg["model"])
        out["01_realization"] = ld / "01_realization.nc"
        save_iris_netcdf(cube, str(out["01_realization"]))

        # --- Stage 2: standardise & regrid -> equal-area grid ---
        out["02_regrid"] = ld / "02_regrid.nc"
        _improver("regrid", "--regrid-mode", prof["regrid_mode"],
                  out["01_realization"], grid_path, "--output", out["02_regrid"])

        # --- Stage 3: probabilities ---
        out["03_percentiles"] = ld / "03_percentiles.nc"
        _improver("generate-percentiles", "--coordinates", "realization",
                  "--percentiles", ",".join(map(str, percentiles)),
                  out["02_regrid"], "--output", out["03_percentiles"])

        prob_paths = {}
        for thr_k in thresholds_k:
            p = ld / f"03_prob_{thr_k:.2f}.nc"
            _improver("threshold", "--threshold-values", f"{thr_k}",
                      "--comparison-operator", "gt", "--collapse-coord", "realization",
                      out["02_regrid"], "--output", p)
            prob_paths[thr_k] = p
        out["03_prob"] = prob_paths

        # --- Stage 4: calibration (pass-through) ---
        out["04_calibrated"] = {k: ld / f"04_calib_{k:.2f}.nc" for k in prob_paths}
        for k, src in prob_paths.items():
            shutil.copy(src, out["04_calibrated"][k])

        # --- Stage 5: neighbourhood processing (metres) ---
        out["05_nbhood"] = {}
        for k, src in out["04_calibrated"].items():
            p = ld / f"05_nbhood_{k:.2f}.nc"
            _improver("nbhood", "--neighbourhood-output", "probabilities",
                      "--neighbourhood-shape", "square",
                      "--radii", str(prof["neighbourhood_radius_m"]),
                      src, "--output", p)
            out["05_nbhood"][k] = p

        # --- Stage 6: blend (pass-through; single model) ---
        out["06_blended"] = out["05_nbhood"]

        # --- Stage 7: products (gridded final probability) ---
        out["07_product"] = out["06_blended"]

        results[lead] = out
        logger.info(f"[{tag}] lead +{lead}h: 7 stages complete -> {ld}")

    return results
