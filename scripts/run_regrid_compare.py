#!/usr/bin/env python
"""Phase-1 tune-regrid sweep: bilinear vs nearest-with-mask, paired delta vs lead.

Regrids three arms onto the 20 km equal-area grid over the SAME 39 dates as the
Phase-0 baseline, scores vs ERA5 on that grid, and reports PAIRED-bootstrap
deltas (seed 42) per lead, stratified all / winter / non-winter.

HEADLINE delta (the question asked): nearest-with-mask − bilinear.
ATTRIBUTION diagnostics (why): nearest − bilinear (interpolation effect) and
nearest-with-mask − nearest (pure land-sea mask effect).

    python scripts/run_regrid_compare.py            # full 39-date sweep
    python scripts/run_regrid_compare.py --limit 3  # smoke test
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import click
import numpy as np
from loguru import logger

warnings.filterwarnings("ignore")

OUTDIR = Path("data/processed/skeleton/verification_regrid")


@click.command()
@click.option("--config", default="aifs_skeleton", help="Config name (configs/<name>.yaml)")
@click.option("--limit", type=int, default=0, help="Use only the first N dates (smoke test)")
def main(config, limit):
    from src.verification.regrid_compare import (
        arm_scorecard, classify, crps_verdict, paired_delta, run_sweep, serialise_records,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== tune-regrid sweep: bilinear / nearest / nearest-with-mask on equal-area grid ===")
    sweep = run_sweep(config=config, limit=limit)
    leads = sweep["leads"]

    meta = {k: sweep[k] for k in ("config", "leads", "n_dates_attempted", "n_dates_used",
                                  "failed_dates", "dates_used", "landmask_vicinity_m")}
    meta.update(seed=sweep["bootstrap"]["seed"], n_iterations=sweep["bootstrap"]["n_iterations"],
                grid="equal-area 20km", truth="ERA5 bilinear->equal-area")

    # per-arm scorecards (baseline format)
    scards = {arm: arm_scorecard(sweep, arm) for arm in ("bilinear", "nearest", "nearest_with_mask")}
    (OUTDIR / "scorecard.json").write_text(json.dumps({**meta, **scards["nearest_with_mask"]}, indent=2))
    (OUTDIR / "baseline_bilinear_ea.json").write_text(json.dumps({**meta, **scards["bilinear"]}, indent=2))

    # headline + attribution deltas
    headline = paired_delta(sweep, "nearest_with_mask", "bilinear")
    diag_interp = paired_delta(sweep, "nearest", "bilinear")            # bilinear -> nearest
    diag_mask = paired_delta(sweep, "nearest_with_mask", "nearest")     # + land-sea mask
    report = {**meta,
              "experiment": "tune-regrid on equal-area grid (regrid stage isolated)",
              "headline": "nearest_with_mask - bilinear (negative CRPS delta = improvement)",
              "attribution": {"interpolation (nearest-bilinear)": "blockier nearest vs bilinear",
                              "mask (nwm-nearest)": "land-sea snap effect on top of nearest"},
              "deltas": {"headline_nwm_vs_bilinear": headline,
                         "diag_nearest_vs_bilinear": diag_interp,
                         "diag_nwm_vs_nearest": diag_mask}}
    (OUTDIR / "delta_report.json").write_text(json.dumps(report, indent=2))
    (OUTDIR / "records.json").write_text(json.dumps(serialise_records(sweep), indent=2))

    _plot_delta(headline, leads, OUTDIR / "delta_vs_lead.png",
                "Coastline-aware regrid effect on CRPS: nearest-with-mask − bilinear")
    _plot_attribution({"nearest − bilinear (interpolation)": diag_interp,
                       "nwm − nearest (land-sea mask)": diag_mask,
                       "nwm − bilinear (total)": headline}, leads,
                      OUTDIR / "delta_attribution.png")

    # ---- printed verdict ----
    print(f"\n=== TUNE-REGRID (n_used={sweep['n_dates_used']}/{sweep['n_dates_attempted']}, "
          f"failed={len(sweep['failed_dates'])}, vicinity={sweep['landmask_vicinity_m']:.0f}m) ===")
    print("HEADLINE CRPS delta = nearest-with-mask − bilinear (negative = improvement), 95% paired CI:\n")
    for line in crps_verdict(headline, leads):
        print(line)

    cells = [(s, l) for s in ("all", "winter", "non_winter") if s in headline
             for l in leads if l in headline[s]["per_lead"]]
    cls = {c: classify(headline[c[0]]["per_lead"][c[1]]["crps"]["ci"]) for c in cells}
    n_imp = sum(v == "IMPROVES" for v in cls.values())
    n_deg = sum(v == "DEGRADES" for v in cls.values())
    n_null = sum(v == "null" for v in cls.values())
    print(f"\nHEADLINE over {len(cells)} (stratum,lead) cells: "
          f"IMPROVES={n_imp}  DEGRADES={n_deg}  null={n_null}")
    if n_imp == 0 and n_deg == len(cells):
        print("VERDICT: nearest-with-mask does NOT improve t2m CRPS anywhere — it SIGNIFICANTLY "
              "DEGRADES it at every lead/stratum. Recommend keeping bilinear.")
    elif n_imp == 0:
        print("VERDICT: nearest-with-mask does NOT significantly improve CRPS anywhere.")
    else:
        print(f"VERDICT: nearest-with-mask significantly improves CRPS at {n_imp} cell(s): "
              f"{[c for c in cells if cls[c]=='IMPROVES']}")

    # attribution summary at +24h all (where the effect is largest)
    def d24(delta):
        return delta["all"]["per_lead"][leads[0]]["crps"]["delta"]
    print(f"\nATTRIBUTION (all, +{leads[0]}h CRPS delta): "
          f"total nwm−bilinear={d24(headline):+.4f}K = "
          f"interpolation(nearest−bilinear)={d24(diag_interp):+.4f}K "
          f"+ mask(nwm−nearest)={d24(diag_mask):+.4f}K")
    print(f"\nOutputs in {OUTDIR}/:\n  scorecard.json (tuned)  baseline_bilinear_ea.json  "
          f"delta_report.json  records.json  delta_vs_lead.png  delta_attribution.png")


def _series(delta, lead_list, stratum="all"):
    pl = delta.get(stratum, {}).get("per_lead", {})
    ls = [l for l in lead_list if l in pl]
    d = np.array([pl[l]["crps"]["delta"] for l in ls])
    lo = np.array([pl[l]["crps"]["ci"][0] for l in ls])
    hi = np.array([pl[l]["crps"]["ci"][1] for l in ls])
    return ls, d, np.vstack([d - lo, hi - d])


def _plot_delta(delta, leads, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = [("all", dict(fmt="o-", color="k", capsize=4, lw=1.8)),
              ("winter", dict(fmt="s--", color="tab:blue", capsize=3)),
              ("non_winter", dict(fmt="^--", color="tab:orange", capsize=3))]
    for j, (stratum, style) in enumerate(styles):
        if stratum not in delta or not delta[stratum]["per_lead"]:
            continue
        ls, d, err = _series(delta, leads, stratum)
        x = np.array(ls, float) + (j - 1) * 1.2
        n = delta[stratum]["per_lead"][ls[0]]["n_dates"]
        ax.errorbar(x, d, yerr=err, label=f"{stratum} (n={n})", **style)
    ax.axhline(0, color="grey", lw=1.2, zorder=0)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("CRPS delta (K):  nearest-with-mask − bilinear")
    ax.set_title(f"{title}\n(95% paired-bootstrap CI; below 0 = improves, CI crossing 0 = not significant)",
                 fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def _plot_attribution(deltas: dict, leads, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = [dict(fmt="o-", color="tab:green", capsize=3),
              dict(fmt="s-", color="tab:red", capsize=3),
              dict(fmt="d-", color="k", capsize=4, lw=1.8)]
    for (label, delta), style in zip(deltas.items(), styles):
        ls, d, err = _series(delta, leads, "all")
        x = np.array(ls, float)
        ax.errorbar(x, d, yerr=err, label=label, **style)
    ax.axhline(0, color="grey", lw=1.2, zorder=0)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("CRPS delta (K), all dates")
    ax.set_title("Attribution of the regrid-mode CRPS change (all dates, 95% paired CI)\n"
                 "total degradation = blockier nearest interpolation + land-sea mask snap", fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
