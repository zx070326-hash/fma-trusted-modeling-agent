# FMA task constitution

Workspace: `{{WORKSPACE_ID}}`

Objective: {{OBJECTIVE}}

## Mission and authority

Build a falsifiable, reproducible mathematical model. Persuasive prose is not
evidence. Files are the durable state of the task; every claim must trace to a
source, executable check, or explicitly labelled uncertainty.

This scaffold contains no result and grants no scientific qualification or
real-world action authority. The model may draft artifacts and propose stage
transitions. It may not approve its own work, issue a gate certificate, expose
a hidden holdout, register a prediction, or authorize release. Only the
external harness may issue a gate certificate bound to the exact artifact
snapshot and independent review receipts. A file named `*.stamp` is not a
certificate.

## S0-S6 workflow

- S0 diagnoses data richness, stationarity, query type, and downstream decision.
- S1 creates three genuinely distinct candidates, then formalizes the selected
  model's assumptions, symbols, failure modes, and validation plan.
- S2 traces every datum and requirement through a ledger and reproducible
  transformation. Synthetic inputs must be declared and sensitivity-tested.
- S3 implements versioned models, solvers, and figure generators with L0-L2
  evidence. A successful run is not validation.
- S4 requires applicable L1-L4 checks, uncertainty analysis, and independent
  adversarial review. Missing domain checks are `NOT_RUN`, never implicit pass.
- S5 writes a decision dossier and may request immutable prediction
  registration. It does not authorize consequential action.
- S6 builds the paper from `docs/` and machine-readable `results/`, with
  consistency and provenance checks. Do not hand-copy or invent numbers.

Backward work is allowed and expected. Record why in `docs/decisions.log`,
invalidate the changed stage and all downstream certificates through the
harness, then re-run the checks.

## Evidence and data discipline

- Treat `problem/` and `data/raw/` as user- or harness-supplied inputs. Do not
  overwrite them.
- Put transformations in code and derived files in `data/processed/`.
- Record the source, access terms, transformation, uncertainty, and use of each
  input. If a needed value is unavailable, say so; declared synthetic data is
  permissible only with range-based sensitivity analysis.
- Keep numerical outputs in structured files under `results/`. Paper text and
  figures must consume those files.
- Append important reasoning to `docs/notebook.md`, decisions to
  `docs/decisions.log`, and abandoned-model diagnoses to
  `docs/model_genealogy.md`. These are agent-legible journals, not authority:
  only graph events and authenticated receipts prove history. Do not rewrite
  them, but never treat that instruction alone as an integrity guarantee.

## Verification discipline

Separate generation from evaluation. A reviewer gets only the committed
artifacts in its review scope. Domain claims need domain-specific verifier
adapters; generic file/schema checks cannot establish physical correctness,
identifiability, calibration, or generalization.

Stop on failed, stale, missing, or unverifiable evidence. Preserve `FAIL`,
`NOT_RUN`, and `HUMAN` outcomes. Never weaken a check merely to make a gate
pass. Record uncertainty and the smallest next discriminating experiment.

The `Makefile` is a convenience facade over the trusted runtime. Its commands
request evaluation; they do not themselves create authority.
