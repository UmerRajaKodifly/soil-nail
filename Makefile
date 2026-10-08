# Soil-nail detection — common developer entrypoints.
# Paths are relative to the repo root.

.PHONY: setup check lint format typecheck test security licenses \
	ingest pilot run ui report

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

setup:
	bash scripts/setup.sh
	$(PIP) install -e ".[dev]"
	$(PIP) freeze > requirements.lock

lint:
	$(PYTHON) -m ruff check src/soilnail tests
	$(PYTHON) -m ruff format --check src/soilnail tests

format:
	$(PYTHON) -m ruff format src/soilnail tests

typecheck:
	$(PYTHON) -m mypy --strict src/soilnail

test:
	$(PYTHON) -m pytest

security:
	$(PYTHON) -m bandit -r src/soilnail -c pyproject.toml -q
	# PYSEC-2026-3447 ignored: torch cu128 requires setuptools<82 (see configs/pip_audit_ignore.yaml)
	$(PYTHON) -m pip_audit --ignore-vuln PYSEC-2026-3447

licenses:
	$(PYTHON) -m piplicenses --format=markdown --output-file=docs/LICENSES.md
	$(PYTHON) scripts/prepend_licenses_banner.py
	@echo "Wrote docs/LICENSES.md — flag for legal review; not a legal conclusion."

check: lint typecheck test security licenses
	@echo "OK: make check passed (ruff + mypy --strict + pytest + bandit + pip-audit + licenses)"

ingest:
	$(PYTHON) -m soilnail.cli ingest --config configs/ingest.yaml

pilot:
	$(PYTHON) -m soilnail.cli pilot --config configs/experiments.yaml

run:
	$(PYTHON) -m soilnail.cli run --config configs/experiments.yaml

ui:
	$(PYTHON) -m soilnail.cli ui --config configs/ui.yaml

report:
	$(PYTHON) -m soilnail.cli report --config configs/report.yaml
