# V5.7 adaptive positive-series mechanism suite

Status: `PASS`

| Case | Branch | Selected | L3 | L4 | Acceptance | Expected |
|---|---|---|---|---|---:|---:|
| valid-hybrid-ode-preserved | hybrid_ode | logistic.ar1_residual | PASS | PASS | True | True |
| stationary-log-drift-recovery | log_growth | log_random_walk_drift | PASS | PASS | True | True |
| stationary-log-growth-ar1-recovery | log_growth | log_growth_ar1 | PASS | PASS | True | True |
| scaled-log-growth-ar1-recovery | log_growth | log_growth_ar1 | PASS | PASS | True | True |
| growth-structural-break-reject | unresolved | log_growth_ar1 | FAIL | FAIL | False | True |
| i34-retrospective-reject | unresolved | log_growth_ar1 | FAIL | FAIL | False | True |
| i35-retrospective-development-recovery | log_growth | log_growth_ar1 | PASS | PASS | True | True |

## Mechanism evidence

- A valid V5.6 hybrid ODE remains on the primary branch.
- Constant-drift and stationary AR(1) log-growth fixtures route to their registered recovery models.
- Growth structural break and I34 fail closed.
- Positive level scaling invariance: `true`.
- I35 is recovered only as disclosed retrospective development evidence.

## Claim boundary

- Fixture scientific acceptance is mechanism evidence only.
- I34 and I35 are disclosed retrospective controls, not unseen tasks.
- No causal mechanism, private qualification, external host, or real-world action is established.
