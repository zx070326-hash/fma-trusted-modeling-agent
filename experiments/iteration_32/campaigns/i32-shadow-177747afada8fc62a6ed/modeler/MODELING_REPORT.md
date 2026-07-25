# I32 Public-Only Modeling Report

## Scope and integrity

- Campaign: `i32-shadow-177747afada8fc62a6ed`.
- Read scope: only the supplied public packet and this reproducibility script; no target values, private capsules, external data, network, environment variables, or process inspection were used.
- `hash_manifest.json`: all 8 public artifacts matched both SHA-256 and declared size.
- Custody attestation: the declared signing-payload SHA-256 matched and its Ed25519 signature verified with the pinned public key.
- This is same-host blinded-context shadow work only. It is not a gate, source identification, private evaluation, promotion, scientific qualification, or real-world authorization.

## First-principles assessment

The packet supplies 28 exactly daily scalar observations in a custody-only positive-affine transformed index. The supplied additive-error scale is explicitly not source-certified. The visible scalar series therefore does not identify the physical state, forcing, or a conservation law.

A one-dimensional autonomous ODE with unique solutions has an order-preserving positive-time flow. The affine ODE time-one map requires `p = exp(b) > 0`. In every 12–24 point expanding-window fit, the unconstrained affine slope was non-positive, so the positive-flow constraint collapsed the model to near-immediate equilibrium. The nearby public states from -600 to -430 also had visibly heterogeneous next-day increments. This makes an autonomous scalar ODE inadequate for the public reduced-state forecast task; it does not identify an alternative physical mechanism.

## Public rolling validation and recovery

Selection was fixed before ranking: use only 13 expanding origins (training sizes 12–24), forecast horizons 1–4 at every origin (52 forecasts/candidate), and rank finite candidates by equal-weight aggregate normalized MAE; break ties by normalized RMSE and then fewer fitted parameters. No target values were scored or requested.

| Candidate | h1 nMAE | h2 nMAE | h3 nMAE | h4 nMAE | aggregate nMAE | aggregate nRMSE | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| robust_local_level_median_w5_recovery | 0.536787 | 0.427103 | 0.450328 | 0.341831 | 0.439013 | 0.548958 | selected |
| robust_local_level_median_w7_recovery | 0.508392 | 0.410998 | 0.516868 | 0.520259 | 0.489129 | 0.575866 | not selected |
| local_level_mean_w3 | 0.503956 | 0.504040 | 0.473158 | 0.550067 | 0.507805 | 0.564538 | not selected |
| robust_local_level_median_w10_recovery | 0.577050 | 0.503391 | 0.484446 | 0.489744 | 0.513658 | 0.578403 | not selected |
| robust_local_level_median_w3_recovery | 0.635113 | 0.497457 | 0.468129 | 0.467705 | 0.517101 | 0.653321 | not selected |
| local_level_mean_w5 | 0.508612 | 0.539720 | 0.507595 | 0.516834 | 0.518190 | 0.550629 | not selected |
| local_level_mean_w7 | 0.496955 | 0.499401 | 0.568277 | 0.560225 | 0.531214 | 0.561988 | not selected |
| local_level_mean_w10 | 0.531621 | 0.515982 | 0.534249 | 0.554507 | 0.534090 | 0.564158 | not selected |
| persistence_last_value | 0.598114 | 0.540263 | 0.553867 | 0.453761 | 0.536501 | 0.706086 | not selected |
| two_state_ar2_ridge_recovery | 0.585948 | 0.553177 | 0.541236 | 0.564313 | 0.561168 | 0.601379 | not selected |
| discrete_affine_ar1_recovery | 0.558966 | 0.539295 | 0.591175 | 0.593333 | 0.570692 | 0.598296 | not selected |
| linear_autonomous_scalar_ode | 0.558966 | 0.539295 | 0.591175 | 0.593333 | 0.570692 | 0.598296 | not selected |
| quadratic_autonomous_scalar_ode | 0.574631 | 0.539778 | 0.580422 | 0.590866 | 0.571424 | 0.619644 | not selected |

The selected robust local-level median (fixed five-observation window) improves aggregate public rolling nMAE from `0.536501` for persistence to `0.439013`. Its horizon-specific nMAEs are h1 `0.536787`, h2 `0.427103`, h3 `0.450328`, h4 `0.341831`; persistence is h1 `0.598114`, h2 `0.540263`, h3 `0.553867`, h4 `0.453761`.

## Locked final refit and forecasts

After selection, the family and fixed window were locked and refit once on all 28 public observations. The latent level estimate is the median of the final five public observations; the model has no fitted continuous parameter and projects that level across h1–h4.

| Target | Time | Frozen public-transformed prediction |
|---|---:|---:|
| target-h1 | 60963 | -556.226279129016 |
| target-h2 | 60964 | -556.226279129016 |
| target-h3 | 60965 | -556.226279129016 |
| target-h4 | 60966 | -556.226279129016 |

## Stability, diagnostics, and limitations

- Selected h1 rolling residuals: mean `-44.303452`, RMSE `714.414775`, lag-1 autocorrelation `-0.314277`, Durbin-Watson `2.489980`. These are diagnostic summaries only, not calibrated uncertainty or a source-certified error model.
- Window sensitivity was evaluated from the public data only. The median-window final state is reported for windows 3, 5, 7, and 10 in `candidate_results.json`; window 5 is locked because it had the lowest aggregate rolling MAE.
- Failure reasons are retained for all scalar-ODE, discrete-recovery, mean-level, and alternative median-window candidates in `candidate_results.json`; none of the rejected candidates were silently discarded.
- The score contract defines a private comparison against persistence, but no private score is available here. No claim that the frozen submission passes that comparison is made.

## Freeze and provenance

- `public_scientific_acceptance=false`
- `shadow_submission_frozen=true`
- `scientific_qualification=false`
- `external_qualification=false`; `real_world_action_authorized=false`.
- Candidate artifact SHA-256: `c3f59a8f0ec2aa782eb2693368bb3e32c8a8112a399e9fe13b406042a071eb01`.
- Submission SHA-256: `7eb762310c1af3609d49ae2d2920135c01db19d1be5f3ed94f9d1bf76867419b`.
