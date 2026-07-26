# Iteration 36 prospective adaptive positive-series protocol

## Question

On one newly selected complete, positive, annual World Bank scalar series, can the frozen V5.7 graph retain a passing autonomous-ODE branch or recover through a guarded stationary log-growth branch, pass public L0-L4, and register four future predictions without accessing their values?

## Freeze boundary

- Implementation commit: `b82417838d4b48b6f1fbda67305a8b8e6a85fd87`
- Frozen at: `2026-07-26T08:59:14+08:00`
- Adaptive protocol: `d1c0956121170a2ef1a24ca7acf2fa8f50efb0a1ee506e36a3aacb9c91e33b0f`
- Source exclusion registry: `443e82acba227c1e5d7604691b5ca0e3b9786d7d7f6577ce97b73983cef9aef4`
- I34 and I35 are excluded by source identity, response bytes, and provenance-record hash.
- The source-selection universe, V5.2 source thresholds, V5.6 primary thresholds, V5.7 adaptive thresholds, candidates, gate, and prediction rules are frozen.
- The selection seed and all custody keys must be generated only after this directory is committed.
- Source selection is a secret HMAC permutation followed only by frozen data-quality eligibility; no model score or source preference is used.

## Frozen graph

1. Evaluate the V5.6 four-family autonomous-ODE graph.
2. Retain that branch only if its public L1-L4 all pass.
3. Otherwise evaluate exactly `log_random_walk_drift` and `log_growth_ar1` on log increments.
4. Admit a recovery only if all frozen fit, improvement, stationarity, stability, break, outlier, interval, and growth-plausibility guards pass.
5. If no branch is admissible, emit diagnostic predictions but `ABSTAIN`; diagnostic output is never registered.

## Gate and stop rules

- Public `ELIGIBLE` requires a real non-fixture task and PASS at every L0-L4 level.
- `ELIGIBLE` registers exactly four predictions, but private evaluation remains `BLOCKED_EXTERNAL_HOST_NOT_RUN` on this same host.
- Any public failure yields `ABSTAIN`, provisional predictions, and `NOT_AUTHORIZED_NOT_RUN`.
- One source draw, one public modelling attempt, and at most one private evaluation are allowed.
- No threshold change, candidate addition, task replacement, or retry is allowed after seeing I36 public results.

## Claim limits

This run can establish same-host real-data public modelling evidence only. It cannot establish external-host independence, causal identification, general mathematical-modelling capability, scientific qualification, or real-world action authority.
