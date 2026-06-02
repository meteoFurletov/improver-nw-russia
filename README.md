# Probabilistic Post-Processing and Blending of AI & NWP Forecasts over Northwest Russia

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

This repository contains the code for a PhD research project on probabilistic weather forecast post-processing using the [IMPROVER](https://github.com/metoppv/improver) framework. The project applies IMPROVER to non-Met Office models (GEFS, AIFS) over northwest Russia (~55–70°N, 28–55°E), with the ultimate goal of multi-model blending of AI and traditional NWP ensemble forecasts.

## Research Roadmap

| Paper | Title | Status | Target Journal |
|-------|-------|--------|----------------|
| 1 | Applying IMPROVER to GEFS over northwest Russia | 🔧 In progress | Weather and Forecasting |
| 2 | Probabilistic post-processing of AIFS | 📋 Planned | Meteorological Applications |
| 3 | Blending AIFS + GEFS through IMPROVER | 📋 Planned | AI for the Earth Systems |

## Study Domain

Northwest Russia: **55–70°N, 28–55°E**

Covers the Northwestern Federal District including St. Petersburg, Murmansk, Arkhangelsk, Komi Republic. Key forecasting challenges: surface-based temperature inversions, prolonged snow cover, maritime-continental climate gradient.

## Data Sources

| Dataset | Members | Resolution | Access |
|---------|---------|-----------|--------|
| GEFS (GEFSv12) | 31 | 0.25° | [dynamical.org](https://dynamical.org) (Zarr) / AWS S3 |
| GEFS Reforecast | 5–11 | 0.25° | AWS S3 (`noaa-gefs-retrospective`) |
| AIFS Single | 1 | 0.25° | ECMWF Open Data |
| AIFS ENS | 51 | 0.25° | ECMWF Open Data |
| ECMWF ENS | 51 | 0.25° | ECMWF Open Data / dynamical.org |
| ERA5 | — | 0.25° | WeatherBench 2 (GCS Zarr) |
| ISD Stations | — | point | NOAA / AWS S3 |

## Project Structure

```
├── configs/                  # YAML configuration files
│   ├── domain.yaml          # Domain bounds, grid specs
│   ├── models.yaml          # Model-specific settings
│   ├── pipeline.yaml        # IMPROVER processing chain config
│   └── verification.yaml   # Verification metrics & thresholds
├── src/
│   ├── data/                # Data access & conversion
│   │   ├── gefs.py         # GEFS download & subsetting
│   │   ├── aifs.py         # AIFS download via ecmwf-opendata
│   │   ├── era5.py         # ERA5 from WeatherBench 2
│   │   ├── stations.py     # ISD station data loading
│   │   └── iris_convert.py # Convert xarray → Iris cubes (IMPROVER-compatible)
│   ├── pipeline/            # IMPROVER processing wrappers
│   │   ├── standardise.py  # Metadata standardisation
│   │   ├── calibration.py  # EMOS & reliability calibration
│   │   ├── neighbourhood.py # Neighbourhood processing
│   │   ├── blending.py     # Time-lag & multi-model blending
│   │   └── run_chain.py    # End-to-end pipeline runner
│   ├── verification/        # Forecast verification
│   │   ├── metrics.py      # CRPS, Brier Score, reliability, rank histogram
│   │   ├── plots.py        # Verification visualisations
│   │   └── station_verify.py # Point-based station verification
│   └── utils/               # Shared utilities
│       ├── config.py       # Config loader
│       ├── grid.py         # Grid operations, regridding
│       └── io.py           # I/O helpers for NetCDF/Zarr
├── scripts/                  # CLI entry points
│   ├── download_gefs.py    # Download & archive GEFS data
│   ├── download_aifs.py    # Download & archive AIFS data
│   ├── run_pipeline.py     # Run full post-processing pipeline
│   └── run_verification.py # Run verification suite
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_iris_conversion_test.ipynb
│   ├── 03_improver_test_data.ipynb
│   ├── 04_gefs_pipeline.ipynb
│   └── 05_verification_analysis.ipynb
├── tests/                    # Unit tests
│   ├── test_iris_convert.py
│   ├── test_pipeline.py
│   └── test_metrics.py
├── docs/                     # Additional documentation
│   └── iris_metadata_guide.md
├── environment.yml           # Conda environment
├── pyproject.toml           # Project metadata & tool config
├── Makefile                 # Common commands
├── .gitignore
└── LICENSE
```

## Quick Start

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate improver-nw-russia
```

### 2. Configure your domain

Edit `configs/domain.yaml` with your domain bounds (defaults to northwest Russia).

### 3. Explore the data

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

### 4. Run the pipeline (Paper 1 — GEFS)

```bash
python scripts/download_gefs.py --start 2022-01-01 --end 2022-12-31
python scripts/run_pipeline.py --config configs/pipeline.yaml --model gefs
python scripts/run_verification.py --config configs/verification.yaml
```

## Installation

### Requirements

- macOS or Linux
- Conda (Miniforge/Mambaforge recommended)
- Python 3.12+
- ~500 GB disk for multi-year regional data archive

### From source

```bash
git clone https://github.com/<your-username>/improver-nw-russia.git
cd improver-nw-russia
conda env create -f environment.yml
conda activate improver-nw-russia
pip install -e .
```

## Key References

- Roberts, N. et al. (2023). IMPROVER: The New Probabilistic Postprocessing System at the Met Office. *BAMS*, 104(3). [doi:10.1175/BAMS-D-21-0273.1](https://doi.org/10.1175/BAMS-D-21-0273.1)
- Trotta, B. et al. (2025). Statistical post-processing yields accurate probabilistic forecasts from AI weather models. [arXiv:2504.12672](https://arxiv.org/abs/2504.12672)
- Lang, S. et al. (2025). An update to ECMWF's machine-learned weather forecast model AIFS. [arXiv:2509.18994](https://arxiv.org/abs/2509.18994)
- Hamill, T. M. et al. (2022). The Reanalysis and Reforecast for GEFSv12. [doi:10.1175/MWR-D-21-0245.1](https://doi.org/10.1175/MWR-D-21-0245.1)

## License

MIT License. See [LICENSE](LICENSE) for details.
