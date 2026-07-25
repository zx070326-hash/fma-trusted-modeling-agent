---
name: verification-manual
description: Apply the L0-L5 validation pyramid and preserve failed or unavailable checks.
---

# Verification manual (P0)

Generation and evaluation must use separate contexts. Evaluate the committed
snapshot, not an unrecorded chat claim.

- L0 reproducibility: environment, seeds, inputs, code version, deterministic
  fixtures, and replayable commands.
- L1 structural: units, conservation/balance, bounds, signs, symmetries, and limits.
- L2 numerical: toy cases, convergence, manufactured solutions where applicable,
  solver diagnostics, and tolerance sensitivity.
- L3 empirical: leakage-safe splits, residual structure, baseline duel,
  calibration, and cross-model checks.
- L4 robustness: parameter/global sensitivity, ensemble disagreement,
  extrapolation flags, perturbations, and uncertainty propagation.
- L5 traceability: data ledger, claim-to-result links, paper consistency, and
  immutable prediction-registration provenance.

Not every check applies to every model. The verifier must record `PASS`, `FAIL`,
`NOT_RUN`, or `HUMAN`, with inputs, method, threshold, output evidence, and
evaluator identity. `NOT_RUN` is not `PASS`; a stage whose required
domain-specific adapter is absent remains blocked.

Generic schema, hash, and file checks establish workflow integrity only. They
cannot prove physical validity, causal identification, out-of-distribution
performance, or scientific qualification.
