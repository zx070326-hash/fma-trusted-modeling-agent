# FMA V5.5 authorized encrypted private chain

V5.5 now provides one additive private-worker path that joins the previously
separate V5.3, V5.4, and V5.5 controls. Historical artifacts and their hashes
are unchanged.

## Enforced order

```text
frozen V5.5 protocol
  -> exact public launch binding
  -> deterministic replay of V5.4 public evidence
  -> authenticated ELIGIBLE receipt and request authorization
  -> V5.3/V5.5 custody bridge and both custody signatures
  -> prediction snapshot and semantic-hash checks
  -> atomic create-once private-evaluation claim
  -> verify exact durable claim bytes
  -> read target AES key
  -> decrypt target capsule in the private worker
  -> emit signed V5.3 aggregate receipt
  -> emit signed V5.5 authorization/custody/budget wrapper
```

`ABSTAIN`, forged assessment, baseline drift, prediction drift, custody drift,
or a duplicate request stops before the private-target key read.

## Additive artifacts

- `LegacyCustodyBridgeV55` binds the V5.3 external custody attestation and V5.5
  split-custody attestation to the same protocol, score contract, forecast
  plan, capsule commitment, encrypted target envelope, source-provenance
  commitment, custody key, and host identities.
- `PrivateEvaluationBudgetClaimV55` is written with create-once filesystem
  semantics under a host-controlled ledger root. A crash after this claim
  consumes the only private evaluation; automatic retry is intentionally
  denied.
- `AuthorizedEncryptedPrivateWorkerReceiptV55` is signed by the worker and
  binds the public authorization chain, both custody generations, exact claim
  bytes, V5.3 aggregate receipt, worker identity, and claim limits.
- `AuthorizedEncryptedPrivateOutputV55` is the create-once transport bundle.

The public verifier needs no target key. It verifies both worker signatures and
every public binding, and it never grants scientific qualification.

## Worker boundary

Run the worker only on the separately administered private host. The
administrator must pin:

- the ledger root and `budget_ledger_id`;
- eligibility, custody, worker, host-attester, and promotion public keys;
- worker executable and runner source;
- coordinator, generator, custodian, and worker host identities;
- read permissions for the target key and write permissions for the ledger and
  output.

The CLI deliberately loads only public artifacts, public keys, the prediction,
and ciphertexts before public precondition verification. It checks the output
path and writes the one-use claim before opening the target-key file.

```powershell
python -m fma.v5_5.authorized_private_worker `
  --protocol D:\campaign\prospective_protocol_v55.json `
  --candidate-policy D:\campaign\candidate_selection_policy_v55.json `
  --public-launch-binding D:\campaign\public_launch_binding_v55.json `
  --eligibility-contract D:\campaign\public_eligibility_contract_v54.json `
  --eligibility-input D:\campaign\public_eligibility_input_v54.json `
  --eligibility-assessment D:\campaign\public_eligibility_assessment_v54.json `
  --eligibility-receipt D:\campaign\public_eligibility_receipt_v54.json `
  --private-authorization D:\campaign\private_authorization_v54.json `
  --eligibility-authority-public-key D:\keys\eligibility-public.pem `
  --request D:\campaign\private_request_v53.json `
  --score-contract D:\campaign\score_contract_v53.json `
  --prediction D:\campaign\prediction_v50.json `
  --v53-custody-attestation D:\campaign\custody_attestation_v53.json `
  --private-target-envelope D:\custody\private_target_envelope_v55.json `
  --source-provenance-envelope D:\custody\source_provenance_envelope_v55.json `
  --split-custody-attestation D:\custody\split_custody_attestation_v55.json `
  --legacy-custody-bridge D:\campaign\legacy_custody_bridge_v55.json `
  --custody-public-key D:\keys\custody-public.pem `
  --expected-coordinator-host-id coordinator-host `
  --expected-generator-host-id generator-host `
  --private-target-key D:\secure\private-target.key `
  --budget-ledger-id i34-private-ledger `
  --budget-ledger-root D:\private-worker\budget-ledger `
  --worker-id private-worker `
  --worker-host-id custodian-host `
  --worker-key-id i34-worker-key `
  --worker-private-key D:\secure\worker-private.pem `
  --output D:\private-worker\receipts\authorized-private-output-v55.json
```

Do not use `--fixture-only` in a real campaign. The preflight suite uses that
flag so its positive controls cannot be promoted.

## Evidence boundary

The Iteration 34 preflight tests control-plane mechanics only. A local green
run does not prove:

- that the custodian or worker is a separately administered physical host;
- that a real task was unseen;
- that the source provenance is truthful;
- that the selected mathematical model is scientifically adequate;
- that private qualification or any real-world action is authorized.

Those claims still require prospective real-task evidence, independent host
and key attestations, V5.3 private-run verification, and a separate promotion
authority.
