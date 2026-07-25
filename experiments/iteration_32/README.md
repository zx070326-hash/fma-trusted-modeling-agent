# Iteration 32: external-ready ODE qualification closure

Iteration 32 is an implementation and readiness campaign, not a scientific
qualification result. It repairs the protocol defects exposed by Iteration 31
without reusing the leaked 1980--2020 population holdout.

Implemented scope:

- a sealed public forecast plan that enumerates every target horizon;
- horizon-by-horizon bootstrap, window, and candidate-family robustness;
- a final all-public-data refit that is separately recertified and bound to the
  exact registered prediction;
- Ed25519-signed external custody, external worker, host-management, and
  promotion artifacts, with only pinned public keys on the coordinator;
- immutable public prediction registration outside the task workspace;
- binding to an authenticated, current V5 S4 gate and the V5 graph event chain;
- fail-closed fixture, missing-promotion, stale-gate, wrong-key, and duplicate
  registration behavior.

Not run:

- no new real task has been selected or frozen;
- no private target value, private key, external anchor record, external worker
  receipt, or external promotion decision exists in this directory;
- no scientific qualification or real-world action is authorized.

Use `PROTOCOL_TEMPLATE.json` only to instantiate a new campaign after an
independently administered custodian and separate keys are available. The
generator must not receive the task's private source, capsule, canary, or any
private key.

