# I33 Public-Only Modeling Report

## Integrity and scope

- All 12 public manifest entries matched declared SHA-256 and size; the custody Ed25519 signature, V5.2 snapshot and threshold seals, V5.3 plan compatibility, V5.4 contract seal, protocol binding, and candidate-policy file hash all verified.
- The work uses only the sealed monthly public scalar state. No source identity, target value, custody content, private evaluation, network, browser, environment variable, process inspection, or other experiment was accessed.
- This remains same-host blinded-context shadow evidence. V5.4 public eligibility has **not run**; no final submission or prediction registration exists.

## First-principles assessment

The 28-point scalar rises from a low initial level toward a mid-series plateau, then declines sharply in the late public segment. A scalar positive-affine transformed observation series does not by itself identify the latent state, exogenous forcing, causal mechanism, or a source-certified noise model. The registered autonomous ODE families are therefore tested as constrained forecast skeletons, not treated as identified labor-market mechanisms.

## Initial candidates and graph recovery

The first round compared all frozen families (constant, exponential, Gompertz, logistic) against the required persistence baseline on 13 expanding origins × four horizons. None improved aggregate normalized MAE over persistence, so the frozen recovery requirement fired. The recovery retained the registered exponential/Gompertz/logistic equations and optimizer, changing only the public training window to the latest min(18, origin) observations. No unregistered family was introduced; search count is seven, within the budget of 16.

| Candidate | Family | Window | Aggregate nMAE | Aggregate nRMSE | h1 nMAE | h2 nMAE | h3 nMAE | h4 nMAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| persistence-last-value | persistence_last_value | all_available_public_prefix | 0.257765 | 0.385296 | 0.137219 | 0.210671 | 0.295450 | 0.387722 |
| recovery_logistic_trailing18 | logistic | last_min(18, origin) public observations | 0.445251 | 0.606105 | 0.294618 | 0.397772 | 0.492012 | 0.596600 |
| initial_logistic | logistic | all_available_public_prefix | 0.447426 | 0.610559 | 0.297299 | 0.400087 | 0.494005 | 0.598313 |
| recovery_gompertz_trailing18 | gompertz | last_min(18, origin) public observations | 0.481548 | 0.618053 | 0.315340 | 0.429857 | 0.533624 | 0.647371 |
| initial_gompertz | gompertz | all_available_public_prefix | 0.482765 | 0.620760 | 0.316236 | 0.430998 | 0.534962 | 0.648863 |
| initial_constant | constant | all_available_public_prefix | 1.177828 | 1.255315 | 1.310433 | 1.220537 | 1.137950 | 1.042390 |
| recovery_exponential_trailing18 | exponential | last_min(18, origin) public observations | 1.342038 | 1.412291 | 0.839143 | 1.160347 | 1.496967 | 1.871695 |
| initial_exponential | exponential | all_available_public_prefix | 1.594670 | 1.683248 | 1.017303 | 1.385033 | 1.772764 | 2.203581 |

The recovery executed but did not create a public MAE improvement over persistence. The typed V5.2 development selection inside the real V5.3 bundle selected the registered `logistic` family on its frozen chronological development split; this is the locked family used for the single all-28-point final refit. The recovery does not change its family, and no post-hoc re-ranking overwrote the typed bundle selection.

## Real V5.3 evidence status

The saved `forecast_bundle_v53.json` contains a real FMA V5.3 bundle with development L0–L4 and h1–h4 all-time final-refit evidence. L1, L2, and L4 pass; L3 fails the frozen validation error, baseline-improvement, and interval-coverage checks. L0 is `NOT_RUN`: two authenticated fresh-process replay receipts are unavailable under this public-only no-process/no-environment scope. Therefore the bundle field is `scientific_acceptance=false`. This is a code-produced evidence value, not a model verification claim.

## Provisional predictions

These are V5.3 final-refit outputs only; they are not submitted or privately scored.

| Target | Time | Provisional prediction |
|---|---:|---:|
| target-h1 | 36340 | 350.996961801074 |
| target-h2 | 36341 | 351.052144152066 |
| target-h3 | 36342 | 351.093143608383 |
| target-h4 | 36343 | 351.123603292825 |

## Claim boundary and artifact hashes

- `scientific_qualification=false`; `external_qualification=false`; `real_world_action_authorized=false`.
- `v5_4_public_eligibility_gate_status=NOT_RUN`; `private_evaluation_performed=false`; `final_modeler_submission_created=false`.
- candidate results SHA-256: `1bd0523f2919106d94e0ee0ccf3470a56995fac0c60a319a20992b5103480e01`
- paired public losses SHA-256: `fc89db02583626d68d145a131d94c43e3d688c213922384fe7ff1471a787dcd4`
- V5.3 bundle SHA-256: `9507d00e71ff0e66e8e3d2085b85870f2ffa1f0add33c7658a2f935034751876`
- provisional predictions SHA-256: `ffe66bc096fa0f4e1278a0db1eb65aa0e0a4f0107de597496e8eef6485d85d44`
