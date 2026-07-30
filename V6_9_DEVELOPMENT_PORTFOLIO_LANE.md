# V6.9 development portfolio side lane

V6.9 closes one concrete execution gap without changing V6.8 artifacts or
granting stage/scientific authority. It is an additive, development-only lane
between an open S0 gate and the untouched S1 frontier.

## Executable chain

```text
S0 OPEN
  -> freeze observation-free two-pack protocol
  -> bind one caller-declared public positive scalar series
  -> build one common rolling-origin plan
  -> execute ODE and positive-log-increment packs on identical prefixes
  -> recompute each pack's input-bound verifier evidence
  -> compare both branches with the V6.8 common RMSE selector
  -> require strict improvement over persistence
  -> SELECT or ABSTAIN
```

The exact pack set is frozen before the series is staged:

- registered scalar autonomous ODE;
- registered positive log increment.

Both branches receive the same ordered training prefixes and one-step targets.
A failed branch remains in the receipt set. A winner that does not beat the
common persistence baseline by the frozen margin is rejected.

Studio defaults to at most 12 rolling origins and the V6.9 runtime has an
absolute 32-origin cap. The frozen V6.8 resource envelopes are not yet an
OS-enforced sandbox: a production deployment still needs killable worker
processes with wall/CPU/memory/artifact enforcement.

## Transaction and recovery

The mutation sequence is:

```text
NOT_STARTED -> FROZEN -> DATA_STAGED -> COMPLETED
                               |
                               +-> RECOVERY_PENDING -> reconcile -> COMPLETED
```

The workspace stores authenticated, content-addressed artifacts for:

1. the observation-free freeze intent;
2. the data/run intent and common origin plan;
3. both branch execution receipts;
4. the complete portfolio run;
5. the completion binding.

`project()` is read-only. `reconcile()` recomputes the deterministic expected
run and commits only missing exact-match artifacts. It never overwrites a
different result. If S0 authority changes, the lane becomes `STALE_PENDING`;
data staging, execution, reconciliation, and S1 all fail closed.

Recovery covers authenticated logical commit boundaries. A physically torn
record in the repository's underlying V4/V5 event log remains an integrity
incident and fails closed; V6.9 does not silently truncate or reinterpret that
base log.

## Studio projection

Studio exposes:

- `POST /api/v1/tasks/{task_id}/portfolio-v69/prepare`
- `POST /api/v1/tasks/{task_id}/portfolio-v69/data`
- `POST /api/v1/tasks/{task_id}/portfolio-v69/run`
- `POST /api/v1/tasks/{task_id}/portfolio-v69/reconcile`

The task snapshot contains a read-only `portfolio_v69` projection. It exposes
hashes, branch statuses, selection, and baseline status, but never the
observation vector or the authenticated run intent.

The side lane is available only for `evidence_scope=development` and
`workflow_mode=legacy`. Once its freeze intent exists, both synchronous and
background S1 entry points are blocked. An OS-backed per-task file lock closes
same-host races across Studio service instances and processes. A multi-host
deployment still requires a distributed lease before claiming equivalent
concurrency protection.

The scalar/continuous/autonomous problem signature is selected by the caller
when choosing this deliberately narrow lane; it is not inferred from S0 prose.
The Studio projection therefore records
`derived_from_s0_typed_problem_signature=false`. A future stage-capable version
must obtain that signature from independently reviewed typed S0 evidence.

## Authority ceiling

Every V6.9 artifact and Studio projection keeps:

```text
scientific_evidence_status = NOT_RUN
claim_ceiling = development_protocol_only
scientific_qualification_granted = false
real_world_action_authorized = false
```

The lane writes no S1-S6 gate certificate. A completed `SELECT` is a
development comparison result, not scientific qualification, external
validation, mechanistic truth, or permission for real-world action.
