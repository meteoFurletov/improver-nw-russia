#!/usr/bin/env python
"""Run the verification suite.

Usage:
    python scripts/run_verification.py --config configs/verification.yaml
"""

import click
from loguru import logger

from src.utils.config import load_config


@click.command()
@click.option("--config", default="configs/verification.yaml", help="Verification config")
@click.option("--experiment", default=None, help="Specific experiment to verify")
def main(config: str, experiment: str):
    """Run forecast verification."""
    cfg = load_config("verification")

    logger.info(f"Ground truth: {cfg['ground_truth']['primary']}")
    logger.info(f"Lead times: {cfg['lead_times']}")

    # TODO: Implement verification runner
    raise NotImplementedError(
        "Verification runner not yet implemented. "
        "Start with src/verification/metrics.py for individual metrics."
    )


if __name__ == "__main__":
    main()
