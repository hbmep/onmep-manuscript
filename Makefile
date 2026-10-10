SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/python -m pip
PIP_NO_CACHE := --no-cache-dir

.PHONY: base env dev docs

base:
	rm -rf $(VENV) build
	@echo "Creating virtual environment with $(PY)..."
	$(PY) -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Purging pip cache..."
	$(PIP) cache purge

env: base
	@echo "Installing package..."
	$(PIP) install $(PIP_NO_CACHE) -e .[cuda12]

dev: base
	@echo "Installing package..."
	$(PIP) install $(PIP_NO_CACHE) -e .[cuda12]
	$(PIP) install $(PIP_NO_CACHE) --no-deps -e ../hbmep
	$(PIP) install $(PIP_NO_CACHE) --no-deps -e ../onmep
