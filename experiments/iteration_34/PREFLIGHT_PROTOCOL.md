# Iteration 34 preflight protocol

Status: FROZEN BEFORE IMPLEMENTATION RESULTS

## Question

Can the V5.3 private evaluator be reached through one fail-closed V5.5 path
that binds the frozen public launch, V5.4 eligibility authorization, V5.3
custody contract, V5.5 split-custody ciphertext, and a one-use private
evaluation budget before private target material is read?

This is a control-plane experiment. It is not a mathematical-model quality
experiment, an external-host qualification, or evidence of scientific
qualification.

## Primary hypotheses

1. An invalid or `ABSTAIN` public gate cannot cause the private-target key file
   to be opened.
2. A private target can be evaluated only when the V5.3 custody attestation and
   V5.5 split-custody attestation bind the same protocol, score contract,
   capsule commitment, ciphertext, and custodian signature.
3. The private worker consumes a create-once request budget before target
   decryption. A retry, including one using another output path, is rejected.
4. The emitted V5.5 receipt binds the V5.3 worker receipt, V5.4 authorization,
   V5.5 launch/custody artifacts, worker identity, and exact one-use claim.
5. No private target values, per-target errors, or secrecy canaries appear in
   public outputs.

## Fixed implementation boundary

- Historical V5.3, V5.4, and existing V5.5 artifact semantics are unchanged.
- New behavior is additive under `fma.v5_5`.
- Public artifacts are loaded and verified before the target key is read.
- The encrypted target envelope may be public; its AES key and decrypted
  capsule remain private-worker inputs.
- A crash after the one-use claim but before a receipt consumes the private
  evaluation budget. Safety takes priority over automatic retry.
- The one-use ledger root is an external-host deployment control. This
  preflight can test its mechanics but cannot prove independent administration.

## Adversarial matrix

The focused suite must reject:

1. `ABSTAIN` or forged V5.4 authorization;
2. drifted V5.5 public-launch binding or baseline identity;
3. wrong eligibility authority key;
4. wrong custody signing key;
5. V5.3/V5.5 capsule commitment mismatch;
6. swapped target and provenance envelopes;
7. wrong AES target key;
8. tampered ciphertext or associated-data binding;
9. prediction snapshot or semantic-hash drift;
10. duplicate one-use claim, even with a different output path;
11. an existing output path;
12. a wrapper receipt signed by an unpinned worker key.

The `ABSTAIN` test must supply a nonexistent target-key path and fail for public
authorization before any file-not-found error. This is the mechanical probe
for read ordering.

## Success and stopping rules

Pass only if all primary hypotheses and adversarial cases pass, focused V5.3
through V5.5 tests pass, and the complete repository suite passes without
weakening existing checks. Any failure is retained as evidence and repaired
only within the fixed boundary above. A threshold, test expectation, or claim
limit will not be changed after observing a failure merely to obtain a pass.

## Claim limits

All generated fixtures and preflight receipts must state:

- `fixture_only=true`;
- `scientific_qualification_granted=false`;
- `real_world_action_authorized=false`.

Passing this protocol establishes only local control-plane behavior. A real
unseen ODE campaign still requires task provenance frozen before generator
access, independently administered custody and worker hosts, pinned independent
keys, and a separate promotion decision.
