"""Baseline scorecard: raw AIFS ENS scored against ERA5 over NW Russia.

The walking-skeleton baseline is the raw 51-member ensemble (native 0.25 deg
grid) verified against ERA5 truth on the common grid.

This module scales the baseline from n=1 to a multi-date sample so the numbers
are statistically usable:
  * the unit of independence is the DATE (spatial points within one forecast
    are correlated), so per (date, lead) we reduce to one scalar score;
  * uncertainty is a bootstrap OVER DATES (resample dates with replacement),
    giving a CI per lead time -- this is what makes a later tuning delta
    interpretable;
  * scores are also stratified winter (DJF) vs non-winter.

Calibration stays pass-through; the 7-stage chain is not involved here (the
baseline is the raw ensemble).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from src.data.aifs import load_aifs_ens
from src.data.era5 import load_era5_truth
from src.utils.config import load_config
from src.verification import metrics as M


# --------------------------------------------------------------------------- #
# Date sampling
# --------------------------------------------------------------------------- #
def sample_init_dates(cfg: dict) -> list[datetime]:
    """Weekly (or cadence_days) 00Z init dates across the tune window."""
    s = cfg["sampling"]
    win = cfg[f"{s['window']}_period"]
    start = max(date.fromisoformat(str(win["start"])),
                date.fromisoformat(str(s["data_start"])))
    end = date.fromisoformat(str(win["end"]))
    out, d = [], start
    while d <= end:
        out.append(datetime(d.year, d.month, d.day, s["init_hour"]))
        d += timedelta(days=s["cadence_days"])
    return out


def _season(dt: datetime, winter_months: list[int]) -> str:
    return "winter" if dt.month in winter_months else "non_winter"


# --------------------------------------------------------------------------- #
# Bootstrap over dates
# --------------------------------------------------------------------------- #
def bootstrap_ci(per_date_values, n_iter: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> dict:
    """Resample DATES (not grid cells) with replacement -> CI on the mean.

    Args:
        per_date_values: one scalar score per date (the unit of independence).
    Returns dict {point, lo, hi, n}. With n<2 the CI collapses to the point.
    """
    vals = np.asarray(per_date_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = vals.size
    point = float(vals.mean()) if n else float("nan")
    if n < 2:
        return {"point": point, "lo": point, "hi": point, "n": int(n)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_iter, n))      # vectorized resample of dates
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n": int(n)}


# --------------------------------------------------------------------------- #
# Scorecard
# --------------------------------------------------------------------------- #
def baseline_scorecard(init_dates, config: str = "aifs_skeleton",
                       reliability_threshold_c: float = 0.0) -> dict:
    """Score raw AIFS ENS vs ERA5 over many dates, with bootstrap CIs + strata.

    Robust to per-date failures: a date that fails to load is logged and
    skipped (recorded in ``failed_dates``), it does not abort the sweep.
    """
    cfg = load_config(config)
    bounds, leads = cfg["domain"], cfg["lead_times_hours"]
    thr_k = reliability_threshold_c + 273.15
    winter_months = cfg["seasons"]["winter_months"]
    bs = cfg["bootstrap"]
    inits = [datetime.fromisoformat(str(d)) if not isinstance(d, datetime) else d
             for d in np.atleast_1d(init_dates)]

    # --- pass 1: load every date's ensemble (lazy store opened once) ---
    fc_by_date, valid_by_date, failed = {}, {}, []
    for init in inits:
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

    # --- pass 2: ONE ERA5 read for all unique valid times, co-gridded ---
    grid_like = fc_by_date[used[0]].isel(lead_time=0, realization=0)
    all_valid = sorted({t for ts in valid_by_date.values() for t in ts})
    truth_all = load_era5_truth(all_valid, like=grid_like).load()
    truth_lut = {np.datetime64(t): truth_all.sel(time=t) for t in truth_all.time.values}

    # --- pass 3: per (date, lead) reduce to scalar scores + keep grids for pooling ---
    # records[lead] = list of dicts (one per used date)
    records = {lead: [] for lead in leads}
    for init in used:
        da = fc_by_date[init]
        season = _season(init, winter_months)
        for i, lead in enumerate(leads):
            fc = da.isel(lead_time=i)                         # (realization, lat, lon), K
            ob = truth_lut[valid_by_date[init][i]].reset_coords(drop=True)
            ens_mean = fc.mean("realization")
            records[lead].append({
                "season": season,
                "crps": float(M.crps_ensemble(fc, ob, "realization").mean()),
                "bias": float((ens_mean - ob).mean()),
                "rmse": float(np.sqrt(((ens_mean - ob) ** 2).mean())),
                "spread_skill": M.spread_skill_ratio(fc, ob, "realization"),
                "ranks": M.rank_histogram(fc, ob, "realization"),
                "prob": (fc > thr_k).mean("realization"),       # grid, for reliability pooling
                "binobs": (ob > thr_k).astype(float),
            })

    # --- aggregate per stratum (all / winter / non_winter) ---
    def aggregate(stratum: str) -> dict:
        out = {"leads": leads, "per_lead": {}}
        for lead in leads:
            recs = [r for r in records[lead]
                    if stratum == "all" or r["season"] == stratum]
            if not recs:
                continue
            crps_ci = bootstrap_ci([r["crps"] for r in recs], bs["n_iterations"], bs["ci"], bs["seed"])
            bias_ci = bootstrap_ci([r["bias"] for r in recs], bs["n_iterations"], bs["ci"], bs["seed"])
            ss_ci = bootstrap_ci([r["spread_skill"] for r in recs], bs["n_iterations"], bs["ci"], bs["seed"])
            ranks = np.sum([r["ranks"] for r in recs], axis=0)
            rel = M.reliability_diagram_data(
                xr.concat([r["prob"] for r in recs], dim="case"),
                xr.concat([r["binobs"] for r in recs], dim="case"), n_bins=10,
            )
            out["per_lead"][lead] = {
                "crps": crps_ci["point"], "crps_ci": [crps_ci["lo"], crps_ci["hi"]],
                "bias": bias_ci["point"], "bias_ci": [bias_ci["lo"], bias_ci["hi"]],
                "rmse": float(np.mean([r["rmse"] for r in recs])),
                "spread_skill": ss_ci["point"], "spread_skill_ci": [ss_ci["lo"], ss_ci["hi"]],
                "rank_hist": ranks.tolist(),
                "reliability": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in rel.items()},
                "n_dates": crps_ci["n"],
            }
        return out

    scorecard = {
        "config": config,
        "leads": leads,
        "reliability_threshold_c": reliability_threshold_c,
        "n_dates_attempted": len(inits),
        "n_dates_used": len(used),
        "failed_dates": failed,
        "dates_used": [d.isoformat() for d in used],
        "strata": {s: aggregate(s) for s in ("all", "winter", "non_winter")},
    }
    # convenience: top-level per_lead mirrors the 'all' stratum
    scorecard["per_lead"] = scorecard["strata"]["all"]["per_lead"]

    for lead in leads:
        p = scorecard["per_lead"].get(lead)
        if p:
            logger.info(
                f"+{lead}h (n={p['n_dates']}): CRPS={p['crps']:.3f} "
                f"[{p['crps_ci'][0]:.3f},{p['crps_ci'][1]:.3f}]K  "
                f"bias={p['bias']:+.3f} [{p['bias_ci'][0]:+.3f},{p['bias_ci'][1]:+.3f}]K  "
                f"spread/skill={p['spread_skill']:.3f}"
            )
    return scorecard


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_scorecard(scorecard: dict, outdir: str | Path, title_suffix: str = "") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    leads = scorecard["leads"]
    paths = []

    def _xy_ci(stratum, field, ci_field):
        pl = scorecard["strata"][stratum]["per_lead"]
        ls = [l for l in leads if l in pl]
        y = np.array([pl[l][field] for l in ls])
        lo = np.array([pl[l][ci_field][0] for l in ls])
        hi = np.array([pl[l][ci_field][1] for l in ls])
        return ls, y, np.vstack([y - lo, hi - y])

    # 1. CRPS vs lead with bootstrap CI error bars + seasonal overlay
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for stratum, style in [("all", dict(fmt="o-", color="k", capsize=3)),
                           ("winter", dict(fmt="s--", color="tab:blue", capsize=2)),
                           ("non_winter", dict(fmt="^--", color="tab:orange", capsize=2))]:
        if scorecard["strata"][stratum]["per_lead"]:
            ls, y, err = _xy_ci(stratum, "crps", "crps_ci")
            n = scorecard["strata"][stratum]["per_lead"][ls[0]]["n_dates"]
            ax.errorbar(ls, y, yerr=err, label=f"{stratum} (n={n})", **style)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("CRPS (K)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"AIFS ENS baseline CRPS vs ERA5 {title_suffix}".strip())
    p = outdir / "crps_vs_lead.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 1b. Bias vs lead with CI (the headline robustness test)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for stratum, style in [("all", dict(fmt="o-", color="k", capsize=3)),
                           ("winter", dict(fmt="s--", color="tab:blue", capsize=2)),
                           ("non_winter", dict(fmt="^--", color="tab:orange", capsize=2))]:
        if scorecard["strata"][stratum]["per_lead"]:
            ls, y, err = _xy_ci(stratum, "bias", "bias_ci")
            ax.errorbar(ls, y, yerr=err, label=stratum, **style)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("ens-mean bias (K)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"Bias vs ERA5 (95% CI over dates) {title_suffix}".strip())
    p = outdir / "bias_vs_lead.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 2. Reliability (all-dates) per lead
    pl = scorecard["per_lead"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for l in leads:
        if l in pl:
            r = pl[l]["reliability"]
            ax.plot(r["forecast_freq"], r["observed_freq"], "o-", ms=4, label=f"+{l}h")
    ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency")
    ax.set_title(f"Reliability P(T>{scorecard['reliability_threshold_c']:.0f}C) {title_suffix}".strip())
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    p = outdir / "reliability.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 3. Rank histograms (all-dates) per lead
    n = len(leads)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, l in zip(axes, leads):
        if l not in pl:
            continue
        h = np.array(pl[l]["rank_hist"]); ax.bar(np.arange(1, h.size + 1), h, width=1.0)
        ax.axhline(h.sum() / h.size, color="r", ls="--", lw=1)
        ax.set_title(f"+{l}h"); ax.set_xlabel("rank")
    axes[0].set_ylabel("count")
    fig.suptitle(f"Rank histograms {title_suffix}".strip())
    p = outdir / "rank_histogram.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    logger.info(f"Saved {len(paths)} figures to {outdir}")
    return paths
