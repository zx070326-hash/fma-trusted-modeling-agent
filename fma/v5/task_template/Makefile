PYTHON ?= python
FMA := $(PYTHON) -m fma.v5

.PHONY: help status freeze-raw checks paper clean gate-s0 gate-s1 gate-s2 gate-s3 gate-s4 gate-s5 gate-s6 invalidate-s0 invalidate-s1 invalidate-s2 invalidate-s3 invalidate-s4 invalidate-s5 invalidate-s6

help:
	@echo "FMA V5 facade: status, freeze-raw, checks, gate-s0..gate-s6, invalidate-s0..invalidate-s6, paper"
	@echo "Gate commands request an external-harness certificate; this Makefile cannot self-approve."

status:
	@$(FMA) status --workspace .

checks:
	@$(FMA) checks --workspace .

freeze-raw:
	@$(FMA) freeze-raw --workspace .

gate-s0:
	@$(FMA) gate --workspace . --stage S0

gate-s1:
	@$(FMA) gate --workspace . --stage S1

gate-s2:
	@$(FMA) gate --workspace . --stage S2

gate-s3:
	@$(FMA) gate --workspace . --stage S3

gate-s4:
	@$(FMA) gate --workspace . --stage S4

gate-s5:
	@$(FMA) gate --workspace . --stage S5

gate-s6:
	@$(FMA) gate --workspace . --stage S6

invalidate-s0:
	@$(FMA) invalidate --workspace . --stage S0

invalidate-s1:
	@$(FMA) invalidate --workspace . --stage S1

invalidate-s2:
	@$(FMA) invalidate --workspace . --stage S2

invalidate-s3:
	@$(FMA) invalidate --workspace . --stage S3

invalidate-s4:
	@$(FMA) invalidate --workspace . --stage S4

invalidate-s5:
	@$(FMA) invalidate --workspace . --stage S5

invalidate-s6:
	@$(FMA) invalidate --workspace . --stage S6

paper:
	@$(FMA) paper --workspace .

clean:
	@echo "Refusing to delete evidence. Create a new workspace for a clean run."
