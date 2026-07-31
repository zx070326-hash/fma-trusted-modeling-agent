PYTHON ?= python

.PHONY: help test smoke

help:
	@echo "THIN Modeling Agent"
	@echo "  make test"
	@echo "  make smoke"

test:
	@$(PYTHON) -m pytest tests/test_thin_modeling_agent.py -q

smoke:
	@$(PYTHON) -m modeling_agent --version
	@$(PYTHON) -m modeling_agent --help
