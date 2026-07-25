# FMA repository guide

This repository is a trustworthy mathematical-modelling kernel.  Preserve
historical schemas and content hashes.  New behaviour is additive by version;
never silently reinterpret a V1--V4 artifact as V5.

## Authority boundary

- Models may diagnose, generate candidates, write code, run permitted tools,
  and propose conclusions.
- Code owned by the harness freezes public inputs, snapshots artifacts,
  executes typed checks, verifies event/hash chains, and controls graph
  transitions.
- A model may not review itself, sign a gate, read private acceptance data,
  register a prediction, grant scientific qualification, or authorize an
  external/real-world action.
- S0--S6 gate certificates establish workflow progress only.  Scientific
  qualification remains a separate V4 private-promotion decision, and
  consequential action remains human-owned.
- Missing domain evidence is `NOT_RUN` or `HUMAN`; file presence and persuasive
  prose never count as L0--L4 scientific evidence.

## V5 task workflow

The V5 task workspace is a projection over the V4 event-sourced modelling
graph:

`stage work -> independent verifier gate -> next stage work`

Human-readable files live in `problem/`, `docs/`, `data/`, `src/`, `checks/`,
`results/`, `predictions/`, `gates/`, and `paper/`.  They are evidence inputs
or projections, not authority.  A raw `*.stamp` file has no effect.  Every
accepted certificate binds the exact file manifest, predecessor certificate,
check receipts, independent-review receipts, evaluator epoch, graph node, and
external HMAC authority.

Backward work is normal.  Revoke the changed stage through the graph; do not
delete history.  The V4 revocation closure invalidates downstream work, after
which V5 creates a new acyclic attempt lineage.

## Development discipline

- Keep changes small and versioned.
- Use strict typed schemas, safe relative paths, content-addressed artifacts,
  deterministic JSON, and fail-closed transitions.
- Keep generators and evaluators in separate contexts.  Record exact input
  hashes, transport evidence, runtime identity, and output hashes.
- Generic checks may establish structure, provenance, replay integrity, and
  paper consistency.  Markov sufficiency, conservation, convergence, Sobol
  sensitivity, calibration, extrapolation, and similar scientific checks
  require a task-specific adapter with computation evidence.
- Fixture/control runs must set capability and real-world claim flags to
  false.

## Verification

Run focused tests first:

```powershell
python -m pytest tests/test_v5_stage_workspace.py -q
python -m pytest tests/test_v5_external_harness.py -q
python -m pytest tests/test_v5_paper.py -q
python -m pytest tests/test_v5_scaffold.py -q
python -m pytest tests/test_single_writer_lock.py -q
```

Then run the complete suite before claiming compatibility:

```powershell
python -m pytest
```

Do not weaken a check to make a gate green.  Report passed, failed, skipped,
not-run, fixture-only, and unverified scopes separately.
