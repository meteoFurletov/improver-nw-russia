#!/usr/bin/env python
"""Run the IMPROVER post-processing pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/pipeline.yaml --model gefs
    python scripts/run_pipeline.py --config configs/pipeline.yaml --model aifs-ens
"""

import click
from pathlib import Path
from loguru import logger

from src.utils.config import load_config


@click.command()
@click.option("--config", default="configs/pipeline.yaml", help="Pipeline config file")
@click.option("--model", required=True, help="Model to process: gefs, aifs-single, aifs-ens")
@click.option("--date", default=None, help="Process specific date (YYYYMMDD)")
@click.option("--dry-run", is_flag=True, help="Print pipeline steps without executing")
def main(config: str, model: str, date: str, dry_run: bool):
    """Run the IMPROVER post-processing pipeline for a given model."""
    cfg = load_config("pipeline")

    logger.info(f"Pipeline for model={model}")
    logger.info(f"Variable: {cfg['variable']} at {cfg['height']}m")

    steps = cfg["steps"]
    active_steps = [name for name, step in steps.items() if step.get("enabled", False)]
    logger.info(f"Active steps: {active_steps}")

    if dry_run:
        logger.info("DRY RUN — steps that would execute:")
        for i, step in enumerate(active_steps, 1):
            logger.info(f"  {i}. {step}")
        return

    # TODO: Implement pipeline execution
    # The pipeline will:
    # 1. Load raw forecast data (GEFS or AIFS)
    # 2. Convert to Iris cubes (src/data/iris_convert.py)
    # 3. Run each IMPROVER CLI step in sequence
    # 4. Save intermediate and final outputs
    #
    # Each step calls the corresponding IMPROVER CLI, e.g.:
    #   improver threshold --threshold-values 273.15 280.0 input.nc output.nc
    #   improver nbhood --radius 20000 input.nc output.nc
    #   improver apply-emos-coefficients input.nc coefficients.nc output.nc
    #
    # Or use the Python API directly:
    #   import improver.cli as imprcli
    #   output = imprcli.threshold.process(cube, threshold_values=[273.15])

    raise NotImplementedError(
        "Pipeline execution not yet implemented. "
        "Start with notebooks/04_gefs_pipeline.ipynb to develop step-by-step."
    )


if __name__ == "__main__":
    main()
