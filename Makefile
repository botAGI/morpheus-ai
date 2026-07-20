PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
UV ?= $(shell command -v uv 2>/dev/null)
HOST ?= 127.0.0.1
PORT ?= 8000
UI_PORT ?= 5173
IMAGE ?= morpheus-wake:local

.PHONY: install-dev lint test verify build serve docker-build docker-run clean

install-dev:
	@if [ -n "$(UV)" ]; then \
		$(UV) pip install --python "$(PYTHON)" -e ".[dev]"; \
	else \
		$(PYTHON) -m pip install --upgrade pip; \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests/ -q

verify: lint test

build:
	@if [ -n "$(UV)" ]; then \
		$(UV) pip install --python "$(PYTHON)" --upgrade build twine; \
	else \
		$(PYTHON) -m pip install --upgrade build twine; \
	fi
	rm -rf dist
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

serve:
	morpheus serve --ui --host $(HOST) --port $(PORT) --ui-port $(UI_PORT)

docker-build:
	docker build --pull -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8000 -p $(UI_PORT):5173 $(IMAGE)

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
