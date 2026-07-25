PYTHON ?= python
WORKSPACE ?=
STAGE ?=
ACTOR ?= model

.PHONY: help test test-v5 status freeze-raw submit checks gate invalidate paper

help:
	@echo "FMA developer facade"
	@echo "  make test-v5"
	@echo "  make test"
	@echo "  make status WORKSPACE=path"
	@echo "Authority commands require FMA_V5_AUTHORITY_KEY_FILE outside WORKSPACE."

test-v5:
	@$(PYTHON) -m pytest tests/test_v5_stage_workspace.py tests/test_v5_external_harness.py tests/test_v5_paper.py tests/test_v5_scaffold.py -q

test:
	@$(PYTHON) -m pytest

status:
	@$(PYTHON) -m fma.v5 status --workspace "$(WORKSPACE)"

freeze-raw:
	@$(PYTHON) -m fma.v5 freeze-raw --workspace "$(WORKSPACE)"

submit:
	@$(PYTHON) -m fma.v5 submit --workspace "$(WORKSPACE)" --stage "$(STAGE)" --actor "$(ACTOR)"

checks:
	@$(PYTHON) -m fma.v5 checks --workspace "$(WORKSPACE)" --stage "$(STAGE)"

gate:
	@$(PYTHON) -m fma.v5 gate --workspace "$(WORKSPACE)" --stage "$(STAGE)"

invalidate:
	@$(PYTHON) -m fma.v5 invalidate --workspace "$(WORKSPACE)" --stage "$(STAGE)" --reason "operator requested rework"

paper:
	@$(PYTHON) -m fma.v5 paper --workspace "$(WORKSPACE)"
