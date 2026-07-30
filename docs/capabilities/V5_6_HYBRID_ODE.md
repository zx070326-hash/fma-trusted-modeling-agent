# FMA V5.6 hybrid ODE and residual-process adapter

Status: `FROZEN_BEFORE_SYNTHETIC_MECHANISM_RESULTS`

This additive adapter addresses the specific capability gap exposed by
Iteration 34: a deterministic ODE can forecast well while leaving strongly
serially structured residuals. V5.6 does not reinterpret or reopen V5.2,
V5.3, or Iteration 34.

## Scientific object

The adapter represents an observed positive scalar series as:

```text
y_t = x(t; family, theta) + e_t
```

where the deterministic trend `x` is one of the existing registered ODE
families:

```text
constant, exponential, gompertz, logistic
```

and the observation residual is either:

```text
trend_only:   forecast(e_t+h) = 0
ar1_residual: e_t = phi * e_t-1 + innovation_t
```

The residual process is a predictive observation model. It is not evidence
that the real system has an AR(1) causal mechanism, and it cannot upgrade an
observational series into a causal model.

## Layered candidate graph

The frozen graph is:

1. `initial_trend_branch`
   - evaluate all four `trend_only` candidates on the same chronological
     development split;
2. `residual_recovery_branch`
   - activate only when the selected initial candidate's absolute validation
     residual lag-1 correlation exceeds the frozen recovery trigger;
   - if activated, evaluate exactly one zero-intercept AR(1) residual process
     for each registered trend family;
3. `mechanism_guard_branch`
   - reject candidates with a near-unit-root residual, correlated innovations,
     unstable `phi`, innovation mean shift, a single excessive innovation,
     ill-conditioned dimensionless trend parameters, or insufficient public
     forecast improvement;
4. `frozen_selection`
   - choose among scientifically admissible activated candidates by lowest
     validation relative RMSE plus the frozen per-parameter complexity
     penalty, then by lower parameter count and stable candidate ID.

No new equation, residual order, transform, window, threshold, or candidate
may be introduced after a result is observed. If recovery candidates are
activated but none clears the guard, the best initial trend remains selected
and the scientific bundle fails rather than silently choosing a persuasive
recovery.

## Generated and prescribed components

Prescribed by the harness:

- the four trend equations;
- the two residual modes;
- the recovery trigger;
- the chronological split;
- candidate budget and selection order;
- L0-L4 checks, thresholds, stopping rules, and claim limits.

Generated from public data:

- dimensionless trend parameters;
- the zero-intercept AR(1) coefficient;
- candidate forecasts and innovations;
- graph transition observations;
- bootstrap and window-sensitivity evidence.

The model may propose an interpretation, but code owns candidate
materialization, graph transitions, evidence computation, and acceptance.

## Dimensionless fitting repair

Every trend fit uses:

```text
s = (t - t0) / (t_last - t0)
z = y / mean(y_train)
```

Optimization occurs in dimensionless `(s, z)` coordinates. Physical `r` and
`K` values are projections for interpretation; acceptance uses the
dimensionless Jacobian condition number. This prevents unit scale alone from
creating or hiding an identifiability failure.

## Frozen thresholds

The exact threshold object is
`V5_6_HYBRID_THRESHOLDS.json`.

- raw file SHA-256:
  `7ac2797aab17d89abfc7855a293249cae6ae3661d5f4276ac64139dcd6165866`;
- canonical semantic SHA-256:
  `56b022cfe6ee5ccc3b1f534c2e038b651c7862d323bf81fe60a03fbac057eb00`.

Important fixed values:

- recovery residual lag trigger: `0.60`;
- maximum innovation lag correlation: `0.35`;
- maximum absolute `phi`: `0.95`;
- minimum AR(1) validation improvement over its same-family trend:
  `5%`;
- maximum `phi` window range: `0.30`;
- maximum innovation mean shift: `1.5` training-innovation standard
  deviations;
- maximum single innovation: `5.0` standard deviations;
- maximum dimensionless condition number: `1e8`.

## L0-L4 evidence

### L0 reproducibility

- two fresh subprocesses;
- identical deterministic semantic output hashes;
- authenticated replay receipts;
- exact source, executable, Python, NumPy, SciPy, and platform binding.

### L1 data and graph contract

- sealed positive finite snapshot and thresholds;
- strictly increasing, effectively regular cadence;
- adequate chronological training and validation slices;
- exact candidate registry and graph branch rules.

### L2 limiting and implementation checks

- dimensionless and physical predictions agree;
- `phi=0` reduces the residual correction to zero;
- closed-form and recursive AR(1) forecasts agree;
- a stationary residual correction decays with horizon.

### L3 development validity

- optimizer convergence;
- validation error bound;
- persistence improvement;
- innovation whiteness;
- AR(1) stationarity margin;
- residual-parameter window stability;
- innovation mean-shift and single-shock guards;
- validation interval coverage;
- dimensionless identifiability condition.

### L4 robustness

- innovation bootstrap success and interval width;
- window forecast sensitivity;
- registered-family disagreement;
- recovery-versus-trend ablation;
- declared support and claim limits.

## Prospective mechanism and failure tests

Before any new real task is selected, fixtures must establish:

1. logistic trend plus stationary AR(1) residual activates recovery, selects
   the hybrid candidate, approximately recovers `phi`, and whitens
   innovations;
2. logistic trend plus uncorrelated residual does not activate unnecessary
   recovery;
3. a near-unit-root residual fails the stationarity guard;
4. an unregistered structural break cannot pass merely because AR(1)
   improves forecast loss;
5. removing the AR(1) recovery harms the correlated-residual fixture, while
   removing the mechanism guard creates a false pass on at least one failure
   fixture;
6. repeated authenticated replays agree.

Fixture/control evidence must keep scientific qualification and real-world
claim flags false.

## Real-task rule

Only after the adapter and the prospective fixtures are frozen and pass may a
new source-selection seed choose a new real task. Iteration 34 data may be used
only as a labeled retrospective regression control, never as an unseen or
private qualification case.

The real task must reuse the V5.5 split-custody and V5.4 public eligibility
boundaries. Without a separately administered external host and independent
management keys, private evaluation remains `NOT_RUN` even if the public gate
is eligible.
