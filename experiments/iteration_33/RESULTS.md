# Iteration 33 result: public ABSTAIN before private evaluation

## Outcome

Iteration 33 exercised the new V5.4 public-eligibility layer on a blinded,
same-host scalar-ODE shadow task. The public modeling process ran, tested four
registered families, executed the required graph-recovery branch, and produced
a sealed V5.3 forecast bundle. The independent public gate then returned
`ABSTAIN`.

No eligibility receipt or private-evaluation authorization was issued. The
private target capsule was not decrypted, no target value was accessed, and
the private evaluation budget remains `0 / 1`.

This is a workflow-control success and a modeling result failure. It shows that
the agent can stop an unsupported candidate before a private score is spent; it
does not show that the selected model solved the forecasting task. A strict
identifier mismatch described below also means I33 is not a clean prospective
protocol-qualification run.

## Frozen chain

| Stage | Evidence | Result |
|---|---|---|
| Prospective protocol | `PROTOCOL.json`, SHA-256 `bdc701...011b` | Frozen before task selection |
| Public custody packet | public manifest SHA-256 `b43e70...a189` | Integrity and Ed25519 custody signature verified |
| Public modeling | modeler manifest SHA-256 `824a9c...caf3` | Seven searched candidates plus persistence baseline |
| V5.3 adapter | forecast bundle SHA-256 `9507d0...1876` | L1/L2/L4 pass, L3 fail, L0 `NOT_RUN`; scientific acceptance false |
| V5.4 public gate | input `24778f...2129`; assessment `c1c7e3...dd2` | `ABSTAIN` |
| Private evaluation | closeout record | `NOT_RUN`, budget consumed `0` |

## Protocol conformance finding

The frozen top-level protocol names the baseline
`persistence_last_value`. The released candidate policy and V5.4 contract name
it `persistence-last-value`. The implementation in both cases is the same
last-observation persistence estimator, and every paired loss was recomputed
against that estimator. Therefore the numerical ABSTAIN conclusion is
unchanged.

The identifiers are nevertheless not equal. Under the repository's strict
binding rules, this is a protocol-conformance `FAIL`; it must not be silently
normalized or repaired after results. Future campaign scaffolding must derive
the typed baseline identifier directly from the prospective protocol and test
the equality before task release.

## Public modeling and recovery

The initial round evaluated constant, exponential, Gompertz, and logistic
families on 13 expanding origins and horizons h1-h4. All four lost to the
frozen persistence baseline on aggregate normalized MAE. The registered
recovery node therefore reran exponential, Gompertz, and logistic with a
trailing-18 public window. Recovery did not introduce a new family and did not
reverse the result.

| Candidate | Aggregate normalized MAE |
|---|---:|
| Persistence baseline | 0.257765 |
| Initial logistic | 0.447426 |
| Recovery logistic, trailing 18 | 0.445251 |
| Initial Gompertz | 0.482765 |
| Recovery Gompertz, trailing 18 | 0.481548 |
| Initial constant | 1.177828 |
| Recovery exponential, trailing 18 | 1.342038 |
| Initial exponential | 1.594670 |

The frozen V5.3 development procedure selected the logistic family. For V5.4,
the corresponding pre-registered expanding-history candidate was compared
pairwise with persistence. No post-private or post-gate model selection
occurred.

## Why V5.4 abstained

The gate verified 52 paired losses over 13 origins and four horizons. Candidate
advantage is defined as baseline normalized absolute loss minus candidate
normalized absolute loss, so negative values favor persistence.

- Mean paired advantage: `-0.189660579843882`
- Selection-adjusted moving-block-bootstrap lower bound: `-0.364444250673318`
- Origin win fraction: `0.076923076923077`
- Contiguous time-block means: `[-0.044866, -0.156527, -0.403788]`
- Recent four-origin mean: `-0.403788`
- Horizon means h1-h4: `[-0.160080, -0.189416, -0.198555, -0.210592]`

The non-fixture, candidate-budget, origin-count, and complete-grid checks
passed. Public scientific acceptance, overall advantage, adjusted-bootstrap
bound, origin win rate, every time-stability check, and every horizon-stability
check failed. Because the frozen contract requires all checks, `ABSTAIN` is
mandatory.

## Closeout limitation discovered

The protocol promised source-identity disclosure after campaign close, but the
custodian had co-sealed exact source, series, and date-range metadata inside
the same encrypted capsule as private targets. There is no separately
releasable provenance envelope. Under the explicit no-decryption closeout
boundary, provenance disclosure is therefore `BLOCKED_NOT_RUN`.

The signed custody attestation supports the claims that a blinded sequence was
committed before public release and was asserted not to reuse earlier
campaigns. It does not make the exact source independently reproducible. A
future campaign must use separate ciphertext and key domains for provenance
metadata and private target values.

## Verification scope

- V5.4 focused tests: `8 passed`
- Directed V5.1-V5.4 regression: `81 passed`
- Full repository suite after V5.4 source closure: `350 passed`
- I33 gate runner: Ruff check and format check passed; frozen artifacts match
  the deterministic dry run.

This was a same-host, separately contextualized shadow experiment with one
recorded protocol deviation and blocked exact-source disclosure. It is not a
clean protocol qualification, external-host qualification, scientific
qualification, or authorization for a real-world action.
