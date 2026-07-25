# V5.6 hybrid ODE mechanism suite

Status: `PASS`

| Case | Recovery | Selected | L3 | Acceptance | Expected outcome |
|---|---:|---|---|---:|---:|
| stationary-ar1-recovery | True | logistic.ar1_residual | PASS | True | True |
| iid-no-recovery | False | logistic.trend_only | PASS | True | True |
| near-unit-root-reject | True | logistic.trend_only | FAIL | False | True |
| training-structural-break-reject | True | logistic.trend_only | FAIL | False | True |
| validation-structural-break-reject | True | gompertz.trend_only | FAIL | False | True |
| i34-retrospective-reject | True | logistic.trend_only | FAIL | False | True |

## Mechanism evidence

- Stationary AR(1) fixture recovery improvement: `0.313975`.
- Stationary AR(1) recovered phi: `0.716436` for frozen truth `0.90`.
- IID residual fixture did not activate recovery.
- Near-unit-root and both training/validation structural-break fixtures failed closed.
- I34 retrospective logistic AR(1) improvement exceeded 50%, but innovation lag remained above the frozen bound, so I34 remained `FAIL`.

## Claim boundary

- Synthetic cases are mechanism controls, not real-world capability claims.
- I34 is a disclosed retrospective control, not an unseen task.
- No private target, qualification, causal identification, or real-world action is authorized.
