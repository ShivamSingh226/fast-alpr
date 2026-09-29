# Directories
SRC_PATHS := fast_alpr/ db/ test/ demo_server.py
YAML_PATHS := .github/ mkdocs.yml
PYTHON ?= .venv/bin/python
CAMERAS ?=
CAMERA_START_INDEX ?=
CAMERA_END_INDEX ?=
DURATION ?= 60
FRAME_STRIDE ?= 10
PROTOCOL ?= rtsp
OUTPUT_CSV ?= live_alpr_detections.csv
HOST ?= 127.0.0.1
PORT ?= 8000
CAMERA_ARGS = $(foreach camera,$(CAMERAS),--camera-id $(camera))
CAMERA_RANGE_ARGS = $(if $(strip $(CAMERA_START_INDEX)),--start-index $(CAMERA_START_INDEX),) $(if $(strip $(CAMERA_END_INDEX)),--end-index $(CAMERA_END_INDEX),)

# Tasks
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  help             : Show this help message"
	@echo "  install          : Install project with all required dependencies"
	@echo "  redis-client     : Install the optional Python Redis client"
	@echo "  sync             : Alias for install"
	@echo "  format           : Format code using Ruff format"
	@echo "  check_format     : Check code formatting with Ruff format"
	@echo "  ruff             : Run Ruff linter"
	@echo "  yamllint         : Run yamllint linter"
	@echo "  pylint           : Run Pylint linter"
	@echo "  mypy             : Run MyPy static type checker"
	@echo "  lint             : Run linters (Ruff, Pylint and Mypy)"
	@echo "  test             : Run tests using pytest"
	@echo "  demo             : Ask for a camera range (or set CAMERA_START_INDEX and CAMERA_END_INDEX)"
	@echo "  demo-sim         : Run the dashboard with generated sample detections"
	@echo "  live             : Ask for a camera range and process feeds into CSV"
	@echo "  live-hls         : Run the batch processor using HLS"
	@echo "  csv              : Follow the live detection CSV"
	@echo "  checks           : Check format, lint, and test"
	@echo "  clean            : Clean up caches and build artifacts"

.PHONY: install
install:
	@echo "==> Installing project with all required dependencies..."
	uv sync --locked --all-groups --extra onnx --no-editable

.PHONY: sync
sync: install

.PHONY: redis-client
redis-client:
	@set -e; stage_dir=$$(mktemp -d); trap 'rm -rf "$$stage_dir"' EXIT; \
		site_packages=$$($(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
		uv --no-config pip install --target "$$stage_dir" "redis>=5.0"; \
		cp -R "$$stage_dir"/. "$$site_packages"/

.PHONY: format
format:
	@echo "==> Sorting imports..."
	@# Currently, the Ruff formatter does not sort imports, see https://docs.astral.sh/ruff/formatter/#sorting-imports
	@$(PYTHON) -m ruff check --select I --fix $(SRC_PATHS)
	@echo "=====> Formatting code..."
	@$(PYTHON) -m ruff format $(SRC_PATHS)

.PHONY: check_format
check_format:
	@echo "=====> Checking format..."
	@$(PYTHON) -m ruff format --check --diff $(SRC_PATHS)
	@echo "=====> Checking imports are sorted..."
	@$(PYTHON) -m ruff check --select I --exit-non-zero-on-fix $(SRC_PATHS)

.PHONY: ruff
ruff:
	@echo "=====> Running Ruff..."
	@$(PYTHON) -m ruff check $(SRC_PATHS)

.PHONY: yamllint
yamllint:
	@echo "=====> Running yamllint..."
	@$(PYTHON) -m yamllint $(YAML_PATHS)

.PHONY: pylint
pylint:
	@echo "=====> Running Pylint..."
	@$(PYTHON) -m pylint $(SRC_PATHS)

.PHONY: mypy
mypy:
	@echo "=====> Running Mypy..."
	@$(PYTHON) -m mypy $(SRC_PATHS)

.PHONY: lint
lint: ruff yamllint pylint mypy

.PHONY: test
test:
	@echo "=====> Running tests..."
	@$(PYTHON) -m pytest test/

.PHONY: demo
demo:
	@$(PYTHON) demo_server.py --protocol $(PROTOCOL) --frame-stride $(FRAME_STRIDE) --output-csv "$(OUTPUT_CSV)" --host "$(HOST)" --port $(PORT) $(CAMERA_ARGS) $(CAMERA_RANGE_ARGS)

.PHONY: demo-sim
demo-sim:
	@$(PYTHON) demo_server.py --simulate --output-csv "$(OUTPUT_CSV)" --host "$(HOST)" --port $(PORT)

.PHONY: live
live:
	@$(PYTHON) -m fast_alpr.live $(CAMERA_ARGS) $(CAMERA_RANGE_ARGS) --protocol $(PROTOCOL) --duration $(DURATION) --frame-stride $(FRAME_STRIDE) --output-csv "$(OUTPUT_CSV)"

.PHONY: live-hls
live-hls:
	@$(MAKE) live PROTOCOL=hls

.PHONY: csv
csv:
	@tail -f "$(OUTPUT_CSV)"

.PHONY: clean
clean:
	@echo "=====> Cleaning caches..."
	@$(PYTHON) -m ruff clean
	@rm -rf .cache .pytest_cache .mypy_cache build dist *.egg-info

checks: format lint test
