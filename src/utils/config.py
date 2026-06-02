"""Configuration loader for the project."""

from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file by name (without extension).

    Args:
        name: Config file name, e.g. 'domain', 'models', 'pipeline'.

    Returns:
        Parsed config dictionary.
    """
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_domain_bounds() -> dict[str, float]:
    """Return domain lat/lon bounds from config."""
    cfg = load_config("domain")
    d = cfg["domain"]
    return {
        "lat_min": d["lat_min"],
        "lat_max": d["lat_max"],
        "lon_min": d["lon_min"],
        "lon_max": d["lon_max"],
    }


def get_study_periods() -> dict[str, list[str]]:
    """Return training/validation/test date ranges."""
    cfg = load_config("domain")
    return cfg["periods"]
