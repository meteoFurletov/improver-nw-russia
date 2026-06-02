#!/usr/bin/env python
"""Run the AIFS ENS walking skeleton and/or the multi-date baseline scorecard.

Single date (runs the 7-stage chain + scores that date):
    python scripts/run_skeleton.py --date 2026-03-15 --profile skeleton

Multi-date baseline sweep (baseline only; raw ENS vs ERA5; bootstrap CIs):
    python scripts/run_skeleton.py --sweep
    python scripts/run_skeleton.py --sweep --limit 3        # smoke test
"""

import json
import warnings
from pathlib import Path

import click
from loguru import logger

warnings.filterwarnings("ignore")


@click.command()
@click.option("--date", "dates", multiple=True, default=["2026-03-15"], help="Init date(s) YYYY-MM-DD")
@click.option("--sweep", is_flag=True, help="Sample dates from config (baseline only, skip chain)")
@click.option("--limit", type=int, default=0, help="With --sweep, use only the first N dates (smoke test)")
@click.option("--config", default="aifs_skeleton", help="Config name (configs/<name>.yaml)")
@click.option("--profile", default="skeleton", help="Profile: skeleton | tune-regrid | tune-nbhood")
@click.option("--outdir", default="data/processed", help="Output directory")
@click.option("--skip-chain", is_flag=True, help="Only run verification scorecard")
def main(dates, sweep, limit, config, profile, outdir, skip_chain):
    from src.pipeline.run_chain import run_skeleton_for_date
    from src.utils.config import load_config
    from src.verification.scorecard import baseline_scorecard, plot_scorecard, sample_init_dates

    if sweep:
        cfg = load_config(config)
        dates = sample_init_dates(cfg)
        if limit:
            dates = dates[:limit]
        skip_chain = True
        logger.info(f"Sweep: {len(dates)} dates {dates[0].date()} -> {dates[-1].date()}")
    else:
        dates = list(dates)

    if not skip_chain:
        for d in dates:
            logger.info(f"=== chain: {d} ({profile}) ===")
            run_skeleton_for_date(d, config=config, profile=profile, outdir=outdir)

    logger.info("=== baseline scorecard (raw AIFS ENS vs ERA5) ===")
    sc = baseline_scorecard(dates, config=config)

    vdir = Path(outdir) / profile / ("verification_sweep" if sweep else "verification")
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "scorecard.json").write_text(json.dumps(sc, indent=2))
    suffix = f"(n={sc['n_dates_used']})"
    figs = plot_scorecard(sc, vdir, title_suffix=suffix)

    logger.info(f"Scorecard JSON: {vdir/'scorecard.json'}  ({len(figs)} figures)")

    # ---- printed summary + verdict on the n=1 findings ----
    pl = sc["strata"]["all"]["per_lead"]
    print(f"\n=== BASELINE (n_used={sc['n_dates_used']}/{sc['n_dates_attempted']}, "
          f"failed={len(sc['failed_dates'])}) ===")
    print(f"{'lead':>5} {'CRPS [95% CI]':>26} {'bias [95% CI]':>28} {'spread/skill':>13}")
    for l in sc["leads"]:
        if l not in pl:
            continue
        p = pl[l]
        print(f"{l:>4}h {p['crps']:>7.3f} [{p['crps_ci'][0]:.3f},{p['crps_ci'][1]:.3f}]K"
              f"   {p['bias']:>+7.3f} [{p['bias_ci'][0]:+.3f},{p['bias_ci'][1]:+.3f}]K"
              f"   {p['spread_skill']:>6.3f}")

    def _verdict():
        far = max(sc["leads"])
        p = pl.get(far)
        if not p:
            return
        warm = p["bias_ci"][0] > 0
        underdisp = p["spread_skill_ci"][1] < 1.0
        w = sc["strata"]["winter"]["per_lead"].get(far, {})
        nw = sc["strata"]["non_winter"]["per_lead"].get(far, {})
        print(f"\n=== VERDICT (at +{far}h, vs the n=1 finding) ===")
        print(f"  warm bias survives: {warm}  "
              f"(bias {p['bias']:+.3f}K, CI {'excludes' if warm else 'includes'} 0)")
        print(f"  under-dispersion survives: {underdisp}  "
              f"(spread/skill {p['spread_skill']:.3f}, CI {p['spread_skill_ci']})")
        if w and nw:
            print(f"  seasonal: winter bias {w['bias']:+.3f}K vs non-winter {nw['bias']:+.3f}K "
                  f"-> {'winter-driven' if abs(w['bias'])>abs(nw['bias'])+0.1 else 'not strongly seasonal'}")
    _verdict()


if __name__ == "__main__":
    main()
