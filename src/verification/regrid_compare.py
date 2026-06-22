"""Phase-1 tune-regrid: bilinear vs nearest-with-mask, isolated to the regrid stage.

Both arms take the SAME AIFS ENS realization cube (native 0.25deg) and regrid it
onto the SAME 20 km equal-area grid with IMPROVER's ``RegridLandSea`` operator —
one ``bilinear`` (the skeleton baseline), one ``nearest-with-mask`` (the tuned,
coastline-aware variant). Both are scored against ERA5 regridded (bilinear) onto
that same equal-area grid, on the cells valid in BOTH arms and the truth.

Because everything except the regrid mode is identical, the per-date difference
(tuned − baseline) isolates the regrid stage. The delta CI is a PAIRED bootstrap:
one resample of the date indices (seed 42) is applied to BOTH arms each iteration.

Note: the existing ``verification_sweep/scorecard.json`` is the RAW native-grid
baseline and is NOT grid-comparable to these equal-area arms — that is why the
bilinear-EA baseline is regenerated here. See memory ``regrid-tuning-baseline``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import iris
import numpy as np
import properscoring as ps
from loguru import logger

from improver.regrid.landsea import RegridLandSea

from src.data.aifs import load_aifs_ens
from src.data.era5 import load_era5_truth
from src.data.iris_convert import aifs_ens_to_iris_cube, xarray_to_iris_cube
from src.utils.config import load_config
from src.verification.scorecard import _season, bootstrap_ci, sample_init_dates

MASK_DIR = Path("data/processed/skeleton/masks")
METRICS = ["crps", "bias", "rmse", "spread_skill"]
# bilinear = skeleton baseline; nearest_with_mask = tuned (coastline-aware).
# "nearest" (plain) is a DIAGNOSTIC arm: it isolates the bilinear->nearest
# interpolation effect from the land-sea mask effect (nwm = nearest + mask).
ARMS = ["bilinear", "nearest", "nearest_with_mask"]


# --------------------------------------------------------------------------- #
# Regrid + scoring primitives
# --------------------------------------------------------------------------- #
def _filled(cube: iris.cube.Cube) -> np.ndarray:
    """Cube data as float64 ndarray, masked/extrapolated points -> NaN."""
    d = cube.data
    return np.ma.filled(d, np.nan).astype(np.float64) if np.ma.isMaskedArray(d) else np.asarray(d, np.float64)


def _scores(fc: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    """Scalar verification scores on the common-valid cells (one date, one lead).

    Mirrors the baseline metric definitions (properscoring CRPS; population spread).

    Args:
        fc: (realization, y, x) regridded ensemble.
        truth: (y, x) regridded ERA5.
        valid: (y, x) bool, cells finite in both arms and the truth.
    """
    f = fc[:, valid]          # (R, N)
    o = truth[valid]          # (N,)
    em = f.mean(axis=0)
    rmse = float(np.sqrt(((em - o) ** 2).mean()))
    spread = float(f.std(axis=0).mean())   # ddof=0, matches metrics.spread_skill_ratio
    return {
        "crps": float(ps.crps_ensemble(o, np.moveaxis(f, 0, -1)).mean()),
        "bias": float((em - o).mean()),
        "rmse": rmse,
        "spread_skill": float(spread / rmse) if rmse > 0 else float("nan"),
        "n_valid": int(valid.sum()),
        "rank": _rank_hist(f, o),
    }


def _rank_hist(f: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Rank histogram counts (obs rank among members), ranks 1..R+1."""
    R = f.shape[0]
    ranks = (f < o).sum(axis=0) + 1            # # members below obs + 1 (ties negligible for floats)
    hist, _ = np.histogram(ranks, bins=np.arange(0.5, R + 2.5, 1))
    return hist


# --------------------------------------------------------------------------- #
# Sweep: per (date, lead) scores for both arms on identical cells
# --------------------------------------------------------------------------- #
def run_sweep(config: str = "aifs_skeleton", limit: int = 0) -> dict:
    """Regrid both arms over the configured dates; return per-date scores.

    Returns a dict with ``records`` = {"bilinear": {lead: [rec,...]},
    "nearest_with_mask": {lead: [...]}} aligned by date, plus metadata.
    """
    cfg = load_config(config)
    bounds, leads = cfg["domain"], cfg["lead_times_hours"]
    winter_months = cfg["seasons"]["winter_months"]
    prof = cfg["profiles"]["tune-regrid"]
    vicinity = float(prof.get("landmask_vicinity_m", 25000))

    # Universal target grid = the saved target land-sea mask cube (defines the
    # cells AND carries land_binary_mask for nearest-with-mask). Bilinear ignores
    # the mask values; using one cube for every regrid guarantees identical cells.
    tmask = iris.load_cube(str(MASK_DIR / "landmask_target_equalarea.nc"))
    smask = iris.load_cube(str(MASK_DIR / "landmask_source_aifs.nc"))

    dates = sample_init_dates(cfg)
    if limit:
        dates = dates[:limit]

    # --- pass 1: load every date's ensemble ---
    fc_by_date, valid_by_date, failed = {}, {}, []
    for init in dates:
        try:
            da = load_aifs_ens(init, lead_hours=leads, variable=cfg["variable"], bounds=bounds)
            fc_by_date[init] = da
            valid_by_date[init] = [np.datetime64(init + timedelta(hours=h)) for h in leads]
        except Exception as e:  # noqa: BLE001 - one bad init must not kill the sweep
            failed.append({"date": init.isoformat(), "error": f"{type(e).__name__}: {str(e)[:120]}"})
            logger.warning(f"SKIP {init.date()}: {type(e).__name__}: {str(e)[:120]}")
    used = list(fc_by_date)
    if not used:
        raise RuntimeError("All dates failed to load; nothing to score.")

    # --- pass 2: ONE ERA5 read for all valid times, each regridded bilinear -> EA ---
    all_valid = sorted({t for ts in valid_by_date.values() for t in ts})
    truth_all = load_era5_truth(all_valid, bounds=bounds)
    truth_ea: dict[np.datetime64, np.ndarray] = {}
    for t in truth_all.time.values:
        tc = xarray_to_iris_cube(truth_all.sel(time=t), standard_name="air_temperature",
                                 units="K", model_id="era5")
        truth_ea[np.datetime64(t)] = _filled(RegridLandSea("bilinear")(tc, tmask))

    # --- pass 3: regrid all arms, score on cells valid in EVERY arm + truth ---
    records = {arm: {l: [] for l in leads} for arm in ARMS}
    for init in used:
        da = fc_by_date[init]
        season = _season(init, winter_months)
        for i, lead in enumerate(leads):
            real = aifs_ens_to_iris_cube(da.isel(lead_time=i), init_time=init,
                                         lead_hours=lead, model_id=cfg["model"])
            fields = {
                "bilinear": _filled(RegridLandSea("bilinear", extrapolation_mode="nanmask")(real, tmask)),
                "nearest": _filled(RegridLandSea("nearest", extrapolation_mode="nanmask")(real, tmask)),
                "nearest_with_mask": _filled(RegridLandSea(
                    "nearest-with-mask", extrapolation_mode="nanmask",
                    landmask=smask, landmask_vicinity=vicinity)(real, tmask)),
            }
            truth = truth_ea[valid_by_date[init][i]]
            valid = np.isfinite(truth)
            for fc in fields.values():
                valid &= np.isfinite(fc.mean(0))     # common-valid across ALL arms
            for arm, fc in fields.items():
                rec = _scores(fc, truth, valid)
                rec["season"] = season
                records[arm][lead].append(rec)
        logger.info(f"[{init.date()}] regridded + scored {len(ARMS)} arms ({len(leads)} leads)")

    return {
        "config": config, "leads": leads,
        "n_dates_attempted": len(dates), "n_dates_used": len(used),
        "failed_dates": failed, "dates_used": [d.isoformat() for d in used],
        "seasons": {d.isoformat(): _season(d, winter_months) for d in used},
        "landmask_vicinity_m": vicinity, "records": records,
        "bootstrap": cfg["bootstrap"],
    }


# --------------------------------------------------------------------------- #
# Aggregation: per-arm scorecard + paired delta
# --------------------------------------------------------------------------- #
def _stratum_recs(records_arm: dict, lead: int, stratum: str) -> list:
    return [r for r in records_arm[lead] if stratum == "all" or r["season"] == stratum]


def arm_scorecard(sweep: dict, arm: str) -> dict:
    """Baseline-format scorecard (point + bootstrap CI) for one arm."""
    leads, bs = sweep["leads"], sweep["bootstrap"]
    rec = sweep["records"][arm]
    out = {"arm": arm, "leads": leads, "grid": "equal-area 20km", "strata": {}}
    for stratum in ("all", "winter", "non_winter"):
        per_lead = {}
        for lead in leads:
            recs = _stratum_recs(rec, lead, stratum)
            if not recs:
                continue
            entry = {}
            for k in METRICS:
                ci = bootstrap_ci([r[k] for r in recs], bs["n_iterations"], bs["ci"], bs["seed"])
                entry[k] = ci["point"]
                entry[f"{k}_ci"] = [ci["lo"], ci["hi"]]
            entry["rank_hist"] = np.sum([r["rank"] for r in recs], axis=0).tolist()
            entry["n_dates"] = len(recs)
            per_lead[lead] = entry
        out["strata"][stratum] = {"per_lead": per_lead}
    out["per_lead"] = out["strata"]["all"]["per_lead"]
    return out


def paired_delta(sweep: dict, tuned_arm: str = "nearest_with_mask",
                 baseline_arm: str = "bilinear") -> dict:
    """Paired-bootstrap delta (tuned − baseline) per lead/stratum/metric.

    One resample of the date indices (seed 42) is applied to BOTH arms within each
    iteration; delta = mean(tuned[idx]) − mean(baseline[idx]). NaN-safe WITHOUT
    breaking pairing: a date is dropped only when NaN in EITHER arm (removes the
    same date from both), so the bootstrap indices stay aligned.
    """
    leads, bs = sweep["leads"], sweep["bootstrap"]
    n_iter, ci, seed = bs["n_iterations"], bs["ci"], bs["seed"]
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    rb, rn = sweep["records"][baseline_arm], sweep["records"][tuned_arm]

    out = {"tuned_arm": tuned_arm, "baseline_arm": baseline_arm}
    for stratum in ("all", "winter", "non_winter"):
        per_lead = {}
        for lead in leads:
            base = _stratum_recs(rb, lead, stratum)
            tune = _stratum_recs(rn, lead, stratum)
            if not base:
                continue
            entry = {"n_dates": len(base)}
            for k in METRICS:
                b = np.array([r[k] for r in base], dtype=float)
                t = np.array([r[k] for r in tune], dtype=float)
                good = np.isfinite(b) & np.isfinite(t)   # paired drop -> pairing preserved
                b, t = b[good], t[good]
                n = b.size
                delta = float((t - b).mean()) if n else float("nan")
                if n >= 2:
                    rng = np.random.default_rng(seed)
                    idx = rng.integers(0, n, size=(n_iter, n))
                    boot = t[idx].mean(axis=1) - b[idx].mean(axis=1)   # same idx -> paired
                    lo, hi = (float(x) for x in np.percentile(boot, [lo_pct, hi_pct]))
                else:
                    lo = hi = delta
                entry[k] = {"delta": delta, "ci": [lo, hi], "n_used": int(n),
                            "baseline": float(b.mean()) if n else float("nan"),
                            "tuned": float(t.mean()) if n else float("nan")}
            per_lead[lead] = entry
        out[stratum] = {"per_lead": per_lead}
    return out


def classify(ci: list) -> str:
    """Significance class of a CRPS delta CI (lower CRPS = better)."""
    lo, hi = ci
    if hi < 0:
        return "IMPROVES"
    if lo > 0:
        return "DEGRADES"
    return "null"


def crps_verdict(delta: dict, leads: list) -> list[str]:
    """Human-readable per-lead/stratum verdict on the CRPS delta."""
    lines = []
    for stratum in ("all", "winter", "non_winter"):
        if stratum not in delta:
            continue
        pl = delta[stratum]["per_lead"]
        for lead in leads:
            if lead not in pl:
                continue
            c = pl[lead]["crps"]
            d, (lo, hi), n = c["delta"], c["ci"], pl[lead]["n_dates"]
            tag = {"IMPROVES": "IMPROVES (CI<0)", "DEGRADES": "DEGRADES (CI>0)",
                   "null": "no sig. effect (CI straddles 0)"}[classify(c["ci"])]
            lines.append(f"  [{stratum:<10} +{lead:>3}h n={n:>2}] "
                         f"CRPS {c['baseline']:.3f}->{c['tuned']:.3f}  "
                         f"delta {d:+.4f} K [{lo:+.4f},{hi:+.4f}]  -> {tag}")
    return lines


def serialise_records(sweep: dict) -> dict:
    """JSON-safe per-date records (rank arrays -> lists) for reproducible re-aggregation."""
    rec = {arm: {str(l): [{**{k: v for k, v in r.items() if k != "rank"},
                            "rank": r["rank"].tolist()} for r in recs]
                 for l, recs in by_lead.items()}
           for arm, by_lead in sweep["records"].items()}
    return {k: sweep[k] for k in ("config", "leads", "dates_used", "n_dates_used",
                                  "n_dates_attempted", "failed_dates", "landmask_vicinity_m",
                                  "seasons", "bootstrap")} | {"records": rec}
