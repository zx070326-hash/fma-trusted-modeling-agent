# Iteration 34 private-chain preflight results

Status: `LOCAL_CONTROL_PLANE_PREFLIGHT_PASS`

## Frozen inputs

- Protocol file: `experiments/iteration_34/PREFLIGHT_PROTOCOL.md`
- Protocol SHA-256:
  `755220f3bb21de80dfb41a222c975b33755d2afdeb8ed08dac9b03124b8aefd9`
- Protocol freeze commit: `66e9080`
- Initial implementation commit: `8bec192`

The protocol was committed before the implementation and before any preflight
test result.

## Primary results

| Frozen hypothesis | Result | Mechanical evidence |
|---|---|---|
| Invalid or `ABSTAIN` public evidence cannot cause a target-key read | PASS | The CLI was given nonexistent target and worker key paths. It rejected the public authorization first and created neither a ledger nor output. |
| V5.3 and V5.5 custody must bind the same private capsule | PASS | The compatibility bridge verifies both custodian signatures and exact protocol, score, forecast-plan, capsule, ciphertext, provenance, key, and host bindings. Commitment drift and envelope swaps were rejected. |
| Private evaluation budget is create-once | PASS | The request-hash claim is atomically created before key access. A repeat with another output path was rejected. |
| The durable claim is exact | PASS | Rewritten claim bytes were rejected before target decryption. |
| Worker output binds the full public/private chain | PASS | The public verifier checked both worker signatures, V5.4 replay and authorization, both custody generations, the exact claim bytes, V5.3 receipt, request, prediction hashes, and worker identity. |
| Public outputs contain no private target or point feedback | PASS | Target values and the secrecy canary were absent; both receipts retain no-private-value and no-per-target-feedback flags. |

## Adversarial results

The frozen matrix rejected:

- fixture evidence paired with a forged signed `ELIGIBLE` assessment;
- a wrong eligibility authority key;
- baseline/public-launch identifier drift;
- a wrong custody signing key;
- V5.3/V5.5 capsule commitment drift;
- target/provenance envelope swaps;
- a wrong AES target key;
- ciphertext changed and rehashed without a new custody signature;
- prediction snapshot drift;
- a changed durable claim;
- a duplicate request using another output path;
- an already existing output;
- an unpinned worker key.

The public eligibility assessment is deterministically recomputed from the
frozen contract and evidence. A signed assessment alone is not trusted.

## Verification

- New authorized-private focused suite: `11 passed`
- V5.3 through V5.5 focused compatibility: `36 passed`
- All V5 tests: `96 passed`
- Full repository: `370 passed in 2697.69s (0:44:57)`
- Ruff on the changed Python files: PASS
- Failed: `0`
- Skipped/xfail: `0` reported

## Claim limits

This result proves local control-plane mechanics only.

| Claim | Status |
|---|---|
| Local V5.4-to-V5.5 authorization/decryption ordering | PASS |
| Local single-ledger one-use budget enforcement | PASS |
| Historical V1-V5 compatibility under the repository suite | PASS |
| Separately administered physical custodian/worker host | NOT_RUN |
| Independent management-key control | NOT_RUN |
| Real unseen ODE task | NOT_RUN |
| External private qualification | NOT_RUN |
| General mathematical-modeling capability | NOT_ESTABLISHED |
| Real-world action authorization | FALSE |

The positive path used synthetic control data and emitted
`fixture_only=true`, `scientific_qualification_granted=false`, and
`real_world_action_authorized=false`. Its aggregate quality value is not a
model-performance result.

## Next gate

Freeze a separate prospective scientific I34 protocol before task selection,
have a custodian choose and encrypt one source/holdout without generator
access, release only the public task packet, and run public modeling. Private
evaluation remains `NOT_RUN` unless the public gate is `ELIGIBLE` and a
separately administered worker with pinned independent keys is available.
