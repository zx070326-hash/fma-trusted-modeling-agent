# Iteration 35 prospective real hybrid-ODE protocol

## Question

On one newly selected complete, positive, annual World Bank scalar series, can
the frozen V5.6 graph select a registered autonomous trend, invoke the AR(1)
predictive residual recovery only when triggered, pass independently replayed
L0–L4 public checks, and register four future predictions without access to
their values?

## Freeze and unseen-task boundary

- All adapter code, candidate families, residual modes, thresholds, prediction
  rules, source-selection rules, and stopping rules are frozen in Git before a
  new selection seed is generated or any I35 source is requested.
- I34 is excluded in three namespaces: canonical source identity, exact API
  response bytes, and released source-provenance record.
- Source selection uses a secret HMAC permutation and data-quality eligibility
  only. No model fit, public score, target value, or source preference may
  affect which eligible series is selected.
- The selected source identity stays encrypted through the public model run.
  Hash-only probe receipts do not establish source-inference resistance.

## Public model graph

1. Evaluate `constant`, `exponential`, `gompertz`, and `logistic` trend-only
   candidates on the frozen 70/30 chronological split.
2. Trigger the complete four-family zero-intercept AR(1) residual branch only
   when the initially selected trend has residual lag-1 correlation above
   `0.60`.
3. Admit recovery candidates only when stationarity, innovation whiteness,
   coefficient stability, structural-break, outlier, and same-family
   validation-improvement guards all pass.
4. Select only from admissible candidates using the frozen validation score
   and complexity penalty. If none is admissible, preserve the initial choice
   as diagnostic output and fail L3.
5. Refit the selected structure on all public observations and forecast target
   horizons 1–4 as trend plus recursively decaying residual correction.

The AR(1) term is a predictive observation process, not a causal mechanism.

## Decision and stopping rules

- Public `ELIGIBLE` requires a non-fixture task and `PASS` at every level L0,
  L1, L2, L3, and L4.
- `ELIGIBLE` causes the code-owned harness to register the four predictions;
  without a separately administered physical host, private evaluation remains
  `BLOCKED_EXTERNAL_HOST_NOT_RUN`.
- Any public failure yields `ABSTAIN`, provisional predictions only, and
  `NOT_AUTHORIZED_NOT_RUN`.
- There is one private-evaluation budget. No retry, threshold change, candidate
  addition, or task replacement is allowed after observing I35 public results.
- Source provenance may be released after terminal public closeout. Private
  target values and keys remain unopened unless a valid external authorization
  path exists.

## Claim limits

This is a same-host real-data public experiment. It can establish local
protocol execution and public scientific evidence only. It cannot establish
external-host independence, causal identification, general mathematical
modelling capability, scientific qualification, or real-world action authority.
