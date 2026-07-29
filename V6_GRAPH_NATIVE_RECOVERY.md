# FMA V6 graph-native recovery kernel

## Implemented boundary

V6 is an additive control layer over the V5 event-sourced stage graph. It does
not reinterpret V1--V5 artifacts and does not grant scientific qualification.

The implementation closes four concrete runtime gaps:

1. `ProblemSignatureV60` and `CapabilityRegistryV60` return an exact
   `ROUTABLE` or `CAPABILITY_GAP` decision. The default registry exposes only
   the executable scalar ODE V5.2 and adaptive positive-series V5.7 packs.
2. `FailureDiagnosisV60` normalizes a failure into a stable signature and a
   code-owned earliest affected stage.
3. `RecoveryPlanV60` selects `RETRY`, `PATCH`, `BRANCH`, `ACQUIRE_DATA`,
   `ABSTAIN`, or `HUMAN` under frozen attempt and repetition budgets.
4. `RecoveryKernelV60` calls the existing V5 `invalidate_from` authority,
   preserves graph history, creates a new attempt lineage when scientific
   content changes, and moves failed filesystem projections into an
   attempt-scoped quarantine.

The Studio bridge exposes:

```text
POST /api/v1/tasks/{task_id}/recover
```

Every task snapshot contains a `recovery` projection with the attempt budget,
same-attempt retry count, distinct failure signatures, stop state, last action,
rollback root, and hash-chain tip.

## Authority split

The caller may report a category and failure code. It cannot select an
arbitrary rollback node:

| Failure category | Code-owned recovery root | Action |
|---|---:|---|
| transient operation before authoritative submit | none | same-attempt retry |
| partial artifacts before authoritative submit | none | quarantine and retry |
| data contract or support failure | S2 | acquire data |
| model assumption or identifiability failure | S1 | branch |
| numerical implementation failure | S3 | patch |
| uncertainty/calibration implementation failure | S4 | patch |
| paper consistency failure | S6 | patch |
| capability gap | none | human decision |
| private holdout exposure | none | abstain |

If an operational or partial-artifact claim is made after the stage has an
authoritative outcome, the kernel upgrades it to a graph-revoking patch. This
prevents a caller from deleting accepted projections while leaving their gate
apparently current.

The Studio back-half automatically performs one safe partial-artifact recovery
and retries the stage. A closed gate is converted into a recovery transition
only when code observes a failed or errored check. Reviewer-only rejection,
missing evidence, and policy ambiguity remain human-owned.

For the one registered scientific direction change, a single `run_backhalf`
request now performs the bounded loop:

```text
ODE attempt
  -> failed scientific check
  -> revoke from S1 and quarantine projections
  -> fresh S1 role processes
  -> route the preserved public series to the V5.7 candidate graph
  -> continue S2--S6
```

Automatic resume is permitted only when the predecessor pack is V5.2 ODE, the
recovery receipt authorizes an S1 `BRANCH`, the series satisfies the V5.7
contract, and the registered V5.7 pack is available. If the complete V5.7
candidate graph remains unresolved, the outcome is `CAPABILITY_GAP / HUMAN`.
The harness preserves that failure instead of inventing a third family or
tuning frozen thresholds.

## Durable recovery state

Each task stores the non-authoritative recovery projection under:

```text
.fma/recovery_v60/
|-- state.json
|-- events.jsonl
`-- attempts/
    `-- a{n}/{failed-stage}/quarantine/{original-relative-path}
```

The recovery event log is append-only and hash chained. Diagnoses, plans, and
transition receipts are also committed into the V5 content-addressed evidence
store. Quarantined file bytes are checked before and after movement. Historical
V5 snapshots remain authoritative.

## Stop rules

The default policy allows at most three scientific attempts and two
occurrences of one normalized failure signature. Recovery stops or pauses when:

- a private holdout or private adaptive signal was exposed;
- the scientific-attempt budget is exhausted;
- the same normalized failure repeats beyond policy;
- expected information gain is below the frozen threshold;
- no registered compatible capability pack exists.

`ABSTAIN` closes adaptive recovery. `HUMAN` pauses the affected decision
without granting the model new authority. Neither means retrying until a gate
turns green.

## Current scientific boundary

The recovery control plane is implemented and locally regression tested. It
does not by itself establish that a replacement model is scientifically
correct. The executable capability surface remains:

- positive scalar autonomous ODE, at least 12 observations;
- adaptive positive scalar series, at least 26 observations under the frozen
  70/30 split and eight-point-per-slice requirement.

V5.7 can change from the bounded ODE family to registered stochastic
log-growth candidates. Arbitrary vector ODEs, PDEs, networks, control,
optimization, and automatic real-world data acquisition remain capability
gaps. Private/gold evidence is never available to the adaptive recovery loop.

## Verification

Focused tests cover:

- exact capability-gap reporting;
- same-attempt partial-artifact quarantine;
- S1 rollback and new-attempt creation after model failure;
- preservation of unchanged public raw data across a model branch;
- fail-closed private-holdout exposure;
- repeated-failure stopping;
- Studio recovery projection and event receipt;
- explicit V5.7 Studio execution with two authenticated fresh-process replays;
- one-command ODE failure, S1 revocation, adaptive branch, and S2--S6 success;
- an unresolved adaptive graph pausing at a capability gap.

The original registered ODE S2--S6 Studio path remains a separate regression
and must continue to pass without granting scientific qualification or
real-world action authority.

Scientific-success semantics are now evaluated separately by the additive
V6.1 gate described in `V6_1_SCIENTIFIC_SUCCESS_GATE.md`. Recovery transitions
do not themselves establish predictive, mechanistic, decision, or
generalization success.
