PYTHON ?= python3
HOST ?= 127.0.0.1
PORT ?= 8000
UI_PORT ?= 5173
IMAGE ?= morpheus-ai:local

.PHONY: install-dev lint test verify build serve docker-build docker-run clean

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest tests/ -q

verify: lint test

build:
	$(PYTHON) -m pip install --upgrade build twine
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
