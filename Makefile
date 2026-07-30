PYTHON ?= python
WORKSPACE ?=
STAGE ?=
ACTOR ?= model

.PHONY: help test test-v5 test-v6 test-operator status freeze-raw submit checks gate invalidate paper

help:
	@echo "FMA developer facade"
	@echo "  make test-v5"
	@echo "  make test-v6"
	@echo "  make test-operator"
	@echo "  make test"
	@echo "  make status WORKSPACE=path"
	@echo "Authority commands require FMA_V5_AUTHORITY_KEY_FILE outside WORKSPACE."

test-v5:
	@$(PYTHON) -m pytest tests/test_v5_stage_workspace.py tests/test_v5_external_harness.py tests/test_v5_paper.py tests/test_v5_scaffold.py tests/test_single_writer_lock.py -q

test-v6:
	@$(PYTHON) -m pytest tests/test_v6_1_scientific_success.py tests/test_v6_3_external_qualification.py tests/test_v6_recovery_kernel.py tests/test_v6_8_capability_sdk.py tests/test_v6_9_portfolio_runtime.py -q

test-operator:
	@$(PYTHON) -m pytest tests/test_v7_operator_authority.py tests/test_v7_operator_cli.py tests/test_v7_operator_http.py tests/test_v7_operator_intake.py tests/test_v7_operator_ledger.py -q

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
