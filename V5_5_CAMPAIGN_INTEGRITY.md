# FMA V5.5 campaign integrity

V5.5 fixes two prospective-campaign failures observed during Iteration 33:

1. the frozen protocol and executed public contract used different spellings
   for the persistence baseline identifier;
2. exact source provenance and private target values were co-sealed, so the
   source could not be disclosed after closeout without also opening the
   private-target capsule.

V5.5 is additive. It does not rewrite V5.3, V5.4, or any Iteration 33 artifact.

## Exact protocol materialization

`ProspectiveCampaignProtocolV55` is sealed before task selection. It contains
the only accepted `baseline_id`, the candidate registry and budget, and all
V5.4 eligibility parameters.

After task selection, `materialize_public_launch_v55` derives:

- `CandidateSelectionPolicyV55`;
- `PublicEligibilityContractV54`;
- `PublicLaunchBindingV55`.

The materializer has no baseline argument. The exact protocol value is copied
into both downstream artifacts, and the binding verifier checks exact string
identity, hashes, budgets, horizons, and every eligibility parameter. A
hyphen/underscore change therefore fails before modeling.

Coordinator CLI:

```powershell
python -m fma.v5_5.protocol_materializer `
  --protocol D:\campaign\prospective_protocol_v55.json `
  --task-id i34-case `
  --eligibility-contract-id i34-eligibility `
  --output-dir D:\campaign\public-launch
```

The output directory is create-once.

## Split custody

The external custodian encrypts two independent plaintexts:

```text
private targets
  -> AES-256-GCM target key/domain
  -> private_target_envelope_v55.json

source provenance only
  -> AES-256-GCM provenance key/domain
  -> source_provenance_envelope_v55.json
```

The code rejects equal encryption keys, equal key IDs, equal envelope IDs,
invalid key lengths, and envelope/hash/binding drift. Each plaintext includes a
high-entropy canary before its public commitment is computed. A
custodian-controlled Ed25519 key signs both ciphertext commitments and their
distinct key fingerprints.

The provenance record contains source authority, locator, table or series ID,
public/private date ranges, source artifact hash, and prior-campaign exclusion
hashes. It contains no target values.

Custodian CLI:

```powershell
python -m fma.v5_5.custodian_worker `
  --protocol D:\custody\prospective_protocol_v55.json `
  --score-contract D:\custody\score_contract_v53.json `
  --private-targets D:\custody\private_targets.json `
  --source-provenance D:\custody\source_provenance.json `
  --private-target-key-id i34-target-aes `
  --private-target-key D:\secure\i34-target.key `
  --source-provenance-key-id i34-provenance-aes `
  --source-provenance-key D:\secure\i34-provenance.key `
  --custodian-host-id custodian-host `
  --coordinator-host-id coordinator-host `
  --generator-host-id generator-host `
  --attestation-id i34-split-custody `
  --custody-key-id i34-custody-signing `
  --custody-private-key D:\secure\i34-custody-ed25519.pem `
  --output-dir D:\custody\sealed
```

Keys and plaintext inputs remain outside generator/coordinator contexts.

## Closeout-only provenance release

An independent closeout authority signs:

- terminal status;
- the exact terminal evidence file hash;
- protocol hash;
- split-custody attestation hash;
- permission to release provenance;
- an explicit denial of private-target release.

Its Ed25519 key must differ from the custody signing key.

```powershell
python -m fma.v5_5.closeout_authority `
  --protocol D:\campaign\prospective_protocol_v55.json `
  --split-custody-attestation D:\custody\sealed\split_custody_attestation_v55.json `
  --terminal-status ABSTAIN `
  --terminal-evidence D:\campaign\public_gate_assessment.json `
  --authorization-id i34-closeout `
  --closeout-authority-key-id i34-closeout-signing `
  --closeout-authority-private-key D:\closeout\i34-closeout-ed25519.pem `
  --output D:\campaign\closeout_authorization_v55.json
```

The release worker accepts the provenance envelope and provenance AES key. Its
CLI and library API do not accept a private-target envelope or target key.

```powershell
python -m fma.v5_5.provenance_release_worker `
  --protocol D:\campaign\prospective_protocol_v55.json `
  --source-provenance-envelope D:\custody\sealed\source_provenance_envelope_v55.json `
  --split-custody-attestation D:\custody\sealed\split_custody_attestation_v55.json `
  --closeout-authorization D:\campaign\closeout_authorization_v55.json `
  --terminal-evidence D:\campaign\public_gate_assessment.json `
  --source-provenance-key D:\secure\i34-provenance.key `
  --custody-public-key D:\campaign\custody-public.pem `
  --closeout-authority-public-key D:\campaign\closeout-public.pem `
  --output-dir D:\campaign\source-disclosure
```

The disclosure receipt permanently records
`private_target_envelope_accessed=false`,
`private_target_key_accessed=false`, and
`scientific_qualification_granted=false`. Public verification can replay the
record, ciphertext commitment, custody signature, terminal-evidence binding,
closeout signature, and disclosure receipt without any decryption key.

## Authorized encrypted private evaluation

The original V5.5 custody layer did not itself connect V5.4 authorization to
the V5.3 private-worker CLI. The additive authorized path now performs a
deterministic replay of public eligibility, verifies the exact launch and both
custody generations, atomically consumes the single private-evaluation budget,
and only then reads the target AES key. Its worker-signed wrapper binds the
historical V5.3 aggregate receipt without changing that receipt.

See [V5.5 authorized encrypted private chain](V5_5_AUTHORIZED_PRIVATE_CHAIN.md).

## Claim boundary

V5.5 closes protocol-identity and provenance-release mechanics. Local tests use
fixture keys and data; they establish implementation behavior only. They do
not establish an independent physical host, real-data provenance, scientific
qualification, or modeling performance. A new I34 protocol must be frozen and
run prospectively to test those properties.

## Implementation verification

- V5.5 focused tests: `9 passed`
- V5.3-V5.5 focused compatibility: `25 passed`
- Required V5 core plus V5.1-V5.5 regression: `90 passed`
- Full repository regression: `359 passed in 2301.89s`
