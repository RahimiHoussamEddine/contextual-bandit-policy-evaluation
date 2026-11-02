PYTHON ?= .venv/bin/python
PIP_CACHE_DIR := $(CURDIR)/.cache/pip
export OMP_NUM_THREADS := 2
export MKL_NUM_THREADS := 2

.PHONY: install-dev install-lock fetch validate-data train evaluate test quality smoke-app app api clean

install-dev:
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -e '.[app,dev]'

install-lock:
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -r requirements.lock.txt
	PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install --no-deps --no-build-isolation -e .

fetch:
	$(PYTHON) -m bandit_ope.data

validate-data:
	$(PYTHON) -m bandit_ope.data --validate-only

train:
	$(PYTHON) -m bandit_ope.train

evaluate:
	$(PYTHON) -m bandit_ope.evaluate

test:
	$(PYTHON) -m pytest

quality:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

smoke-app:
	$(PYTHON) scripts/smoke_app.py

app:
	$(PYTHON) -m streamlit run app.py --server.address=127.0.0.1 --server.headless=true

api:
	$(PYTHON) -m uvicorn bandit_ope.api:app --app-dir src --host 127.0.0.1 --port 8000

clean:
	rm -rf .cache data/processed models
