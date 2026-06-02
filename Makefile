.PHONY: install test lint notebooks clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Create conda environment and install project
	conda env create -f environment.yml
	conda run -n improver-nw-russia pip install -e .

test:  ## Run unit tests
	pytest tests/ -v --tb=short

test-slow:  ## Run all tests including slow ones
	pytest tests/ -v --tb=short -m ""

lint:  ## Run linter
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:  ## Auto-format code
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

notebooks:  ## Start Jupyter Lab
	jupyter lab notebooks/

# --- Data commands ---
download-gefs:  ## Download GEFS data for test period
	python scripts/download_gefs.py --start 2022-01-01 --end 2022-01-31

download-aifs:  ## Download latest AIFS forecast
	python scripts/download_aifs.py --model aifs-single --init-hour 00

# --- Pipeline commands ---
run-pipeline:  ## Run IMPROVER pipeline (Paper 1 - GEFS)
	python scripts/run_pipeline.py --config configs/pipeline.yaml --model gefs

run-verify:  ## Run verification suite
	python scripts/run_verification.py --config configs/verification.yaml

# --- IMPROVER exploration ---
improver-test:  ## Run IMPROVER unit tests to verify installation
	python -c "import improver; print(f'IMPROVER version: {improver.__version__}')"
	python -c "import iris; print(f'Iris version: {iris.__version__}')"

improver-test-data:  ## Clone IMPROVER acceptance test data
	git clone https://github.com/metoppv/improver_test_data.git
	@echo "Set IMPROVER_ACC_TEST_DIR=$$PWD/improver_test_data"

clean:  ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
