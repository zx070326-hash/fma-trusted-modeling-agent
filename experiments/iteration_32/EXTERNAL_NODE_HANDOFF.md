# External node handoff

This handoff deliberately separates public files from private files and keys.
Do not run the custodian or private-worker commands on the generator or
coordinator machine.

## 1. Coordinator freezes public contracts

Create and seal:

- one new public `ODETimeSeriesSnapshotV52`;
- one `ODEThresholdsV52`;
- one `ODEForecastPlanV53` listing every private target ID and time;
- one `PrivateScoreContractV53` containing target IDs and scoring policy, but
  no target values.

Transfer only the score contract and pinned public-key configuration to the
custodian administrator.

## 2. Custodian commits private data before generator release

On the independently administered host:

```powershell
python -m fma.v5_3.custodian_worker `
  --score-contract C:\custody\public\score_contract.json `
  --private-targets C:\custody\private\targets.json `
  --private-source-manifest-hash <sha256> `
  --external-anchor-receipt-hash <sha256> `
  --custodian-host-id <external-host-id> `
  --coordinator-host-id <coordinator-host-id> `
  --generator-host-id <generator-host-id> `
  --attestation-id <attestation-id> `
  --attester-key-id <custody-key-id> `
  --private-key C:\custody\keys\custody-ed25519.pem `
  --private-capsule-output C:\custody\private\capsule.json `
  --public-attestation-output C:\custody\public\custody_attestation.json
```

Return only `custody_attestation.json`. Keep `targets.json`, `capsule.json`,
the canary, and all private keys on the external host.

## 3. Generator completes the public campaign

The coordinator verifies the pinned custody signature, then allows the
generator to build the public V5.3 forecast bundle. Open S0--S4 through the
normal V5 stage workflow and independent reviews. Register the exact
`final_refit.predictions` through `PublicPredictionRegistryV53`, bind the
registration through `bind_i32_public_campaign_to_v5_v53`, and create one
`PrivateEvaluationRequestV53`.

Transfer to the external host:

- the sealed private evaluation request;
- the public score contract;
- the content-addressed registered prediction snapshot.

Do not transfer a mutable working prediction.

## 4. External host performs the single private evaluation

```powershell
python -m fma.v5_3.external_private_worker `
  --request C:\custody\public\private_request.json `
  --score-contract C:\custody\public\score_contract.json `
  --prediction C:\custody\public\registered_prediction.json `
  --private-capsule C:\custody\private\capsule.json `
  --worker-id <worker-id> `
  --worker-host-id <external-host-id> `
  --worker-key-id <worker-key-id> `
  --worker-private-key C:\custody\keys\worker-ed25519.pem `
  --output C:\custody\public\worker_receipt.json
```

The output contains one aggregate score and no per-target values or errors.

## 5. Independent host management attests the runtime

Using a key distinct from both custody and worker keys:

```powershell
python -m fma.v5_3.host_attester `
  --worker-receipt C:\custody\public\worker_receipt.json `
  --worker-public-key C:\custody\keys\worker-ed25519.pub.pem `
  --coordinator-host-id <coordinator-host-id> `
  --generator-host-id <generator-host-id> `
  --attestation-id <host-attestation-id> `
  --host-attester-key-id <host-key-id> `
  --host-attester-private-key C:\custody\keys\host-ed25519.pem `
  --output C:\custody\public\host_attestation.json
```

Return the worker receipt and host attestation. The coordinator verifies both
against pinned public keys. A valid private result is still not qualification.

## 6. Independent promotion

An independently administered promotion service must verify the external
anchor records, current V5 graph binding, immutable registry, public evidence,
private verification, key separation, and absence of integrity incidents. It
then signs `ExternalPromotionDecisionV53`. Without that signed decision, the
final status is `NOT_RUN`; a local process, a new chat, or a claimed hostname
cannot substitute for this step.

