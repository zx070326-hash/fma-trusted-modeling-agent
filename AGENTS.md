# THIN Modeling Agent repository guide

This repository has one product: `modeling_agent`. Keep it small, open-ended,
and centered on solving real mathematical-modeling problems.

## Design boundary

- The model may decompose the problem, propose any model family, revise the
  Problem Graph, write task-local code, run permitted tools, and change
  direction after failure.
- The harness owns budgets, task-local filesystem boundaries, durable state,
  artifact hashes, mechanical checks, evidence admission, and stop conditions.
- Checked `working` evidence may guide research but cannot support a final
  claim. Decision-relevant `claim` evidence requires a fresh verifier context.
- The generator cannot approve its own evidence or final answer.
- Generic checks establish only the property they actually test. They do not
  establish mechanism, causality, extrapolation, or scientific qualification.
- External, destructive, financial, regulated, or real-world actions remain
  human-owned.

## Development discipline

- Prefer a better prompt, tool description, or evaluation contract over a new
  subsystem.
- Add code only for a failure repeated on frozen unseen tasks.
- Do not introduce fixed stages, model-family allowlists, certificate stacks,
  or duplicate state planes.
- Keep the source facts in the Problem Graph, execution observations, and
  Evidence Graph. Derive status and delivery projections from them.
- Preserve failed attempts and revocation lineage; do not present a stopped or
  unchecked run as successful modeling.
- Keep dependencies optional unless the core cannot work without them.

## Verification

Run the focused suite first:

```powershell
python -m pytest tests/test_thin_modeling_agent.py -q
```

Then run the complete suite and CLI smoke checks:

```powershell
python -m pytest
python -m modeling_agent --version
python -m modeling_agent --help
```

Report passed, failed, skipped, stopped, and unverified scopes separately.
