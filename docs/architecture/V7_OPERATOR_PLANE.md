# FMA V7.0 Thin Operator Plane

## Outcome

V7.0 adds a small operational sidecar around the existing FMA authority and
science kernel.  It improves intake, continuity, concurrency control, and
agent legibility without reinterpreting any V1--V6 artifact or changing the
meaning of an S0--S6 certificate.

```text
Browser / Codex / fma-ops
            |
            v
Operator Plane V7.0
immutable intake | next packet | SQLite jobs
lease + heartbeat + fencing | ownership | doctor
            |
            | typed request and operational receipt
            v
Existing FMA Studio and Authority Plane
role processes | adapters | manifests | reviews
authenticated graph transition | recovery | claim ceiling
            |
            v
External qualification remains separately administered
```

An operator row answers **who is doing what and whether the work can resume**.
It never answers whether a model is scientifically supported.

## Five absorbed capabilities

### 1. One agent-legible operating surface

`fma-ops` exposes intake, status, next-packet projection, reconciliation, and
doctor checks through stable JSON.  Studio snapshots expose the same
`operator_v70` and `next_packet_v70` projections.  There is one implementation,
not separate CLI, browser, and Codex state machines.

`next` is read-only.  It compiles the current authority graph into one bounded
packet containing:

- exact workspace, policy, graph snapshot, frontier, and gate bindings;
- declared read/write scope;
- declared tool profile;
- expected output, enforced attempt/lease limits, and a declared wall-time budget;
- explicit workflow-only claim scope.

### 2. A separate durable task ledger

`.fma-op-v70/operator.sqlite3` uses SQLite WAL, `synchronous=FULL`,
foreign keys, transactional state changes, and a hash-linked event stream.
The database lives outside every task workspace, so it is excluded from stage
manifests and scientific hashes.

The filesystem remains the portable system of record for scientific evidence.
SQLite remains the queryable and recoverable system of record for operations.

### 3. Lease, heartbeat, fencing, idempotency, and ownership

Every claim increments both `attempt_epoch` and `fencing_token`.  Heartbeat,
submit, and failure transitions must match the exact worker, attempt, and
fence, and an expired lease cannot submit late work.  This closes the classic
lease ABA failure in which an old worker returns after a new worker has taken
over.

Semantic idempotency binds:

```text
workspace + action + authority binding + operator policy
```

Reusing an idempotency key with another packet hash fails closed.  Path
ownership is normalized component-by-component; a parent and child path
conflict.  Current stage writers conservatively own the whole task because the
existing stage runtime has broad writes.  Narrower same-task parallelism is
allowed only after a worker has a provably isolated write set.

### 4. Transactional, content-addressed intake

Intake files are treated as untrusted data, never instructions:

```text
validate names and size
  -> hash exact bytes
  -> publish immutable content-addressed blobs
  -> write and verify an intake manifest in staging
  -> atomic rename to a committed intake
  -> update the current-intake projection
  -> optionally build and verify a complete staging workspace
  -> atomic rename to the formal task path
```

The task mission and evidence snapshot bind the intake manifest hash.  The
workspace receives content-verified copies under `problem/intake/`.  Their
exact manifest, file set, byte sizes, and hashes are rechecked whenever the
operator binds a graph packet.  An interrupted publication may leave recovery
evidence in the hidden staging area, but it does not create a visible task or
move the current-task projection.

### 5. Status and doctor as code-owned projections

Status combines three independent facts:

- operator task/lease state;
- authenticated graph and stage state;
- scientific success and claim-ceiling state.

Doctor enumerates the whole operator ledger, event chain, committed intakes,
and all task directories.  Corrupt tasks are reported instead of silently
disappearing from `list_tasks`.

## Design philosophy learned and retained

### Filesystem for science, database for operations

Scientific evidence must be portable, content-addressed, and reproducible.
Scheduling state needs transactions, leases, and efficient queries.  Putting
both into one graph creates accidental authority; keeping them separate makes
the trust direction explicit.

### Autonomy means recoverable continuity

Autonomy is not a prompt saying “continue until done.”  It is durable state,
bounded work, an observable lease, a deterministic next action, a stop
condition, and a recovery path that survives process or context loss.

### Reconcile exact submissions before retrying work

An expired lease means the outcome is unknown, not necessarily absent.  It is
quarantined as `RECOVERY_PENDING` and is not automatically replayed.
Automatic repair is limited to a persisted `SUBMITTED` receipt whose exact
output binding and workspace manifest still equal the authenticated graph.
V7.0 then repairs only the operator projection and does not rerun Codex,
repeat a review, or mint another certificate.

This does not promise exactly-once execution.  It promises that a surviving,
exactly bound effect is recognized once; every other crash window remains
explicitly unresolved.

### Completion is a verified postcondition

A worker submission is only `SUBMITTED`.  The service then reopens and verifies
the authority workspace and checks the action-specific postcondition.  Only
that existing authority fact is projected as `ACCEPTED` or `REJECTED` in the
operator ledger.  The projection itself has no gate power.

### Parallelism comes from isolation

Agent count does not create safe parallelism.  Non-overlapping, normalized
write ownership does.  Different task workspaces can run concurrently now.
Within one scientific graph, mutation remains serialized until branch scratch
spaces and integration boundaries are mechanically enforced.

### Progress is a projection, not a flag

The next packet is regenerated from the current graph.  It is not remembered
from a chat or a mutable `current_stage` flag.  A rollback or new attempt
changes the graph binding and makes the old packet stale.

### Compute optimistically; commit pessimistically

Long computation may happen outside a global authority lock.  Recognition is
separate: the result must carry exact input and output bindings, declared file
changes, a worker submission hash, and an action-specific postcondition.
Existing FMA gate code still rechecks predecessors, manifests, reviews, and
authority signatures at its own commit boundary.  Expensive computation is a
proposal; only a fresh authority proof makes it visible as progress.

### Independence is an information-flow property

A differently named reviewer is not automatically independent.  Generator
and evaluator contexts remain separated by role, visible inputs, write
permissions, and receipts.  The operator schedules those roles but never
collapses their contexts or lets a worker sign its own gate.

### Revoke history; do not rewrite it

Changed evidence invalidates downstream graph state through the existing
revocation closure.  Failed attempts and negative checks remain evidence for
diagnosis and model evolution.  Operational retry creates a new attempt epoch;
it does not edit an old failure into success.

### Stop conditions are part of correctness

Success, a stable evidence or permission blocker, an exhausted attempt budget,
and an explicit human boundary are distinct terminal reasons.  A persuasive
answer, one failed attempt, or an idle worker is not a valid stop condition.

### Narrative and UI are views, not truth sources

The Studio presents evidence, operator continuity, and claim ceilings, but it
cannot create scientific support.  A polished report, UI state, task row, raw
stamp, or worker assertion never raises the claim ceiling.

### Corrupt, missing, empty, failed, and not-run are different

V7.0 does not convert unreadable state into an empty/default project.  Doctor
reports corruption; unfinished publication is recovery-pending; absent
scientific evidence remains `NOT_RUN`.

## Current enforced scope and hard limits

The following boundaries are intentional and testable:

- V7 currently gives durable tracking and exact-submission recovery to the
  connected long Studio runs (`S0`, `S1`, back-half, and V6.9 portfolio).
  Short preparation, ingestion, and repair operations still use Studio's
  existing synchronous transaction and task lock rather than operator leases.
- Long workers are daemon threads in the Studio process.  Fencing rejects a
  stale heartbeat or submission, but cannot kill a callback or undo file
  writes it already made.  Hard cancellation requires an independent worker
  process, a staging workspace, and an atomic integration boundary.
- `allowed_tool_profile`, `max_wall_seconds`, and packet write paths are
  contracts plus audit evidence in this version.  Attempts and leases are
  enforced; process-level tool sandboxing, a hard wall-clock kill, and
  pre-write path confinement are not.
- Intake provides transport provenance and mutation detection.  Attachments
  are not yet semantically parsed, role-labelled, or injected into S0/S1
  reasoning.  This prevents accidental observation leakage but means intake
  alone does not make a task scientifically executable.
- The ledger is a local single-host SQLite control plane, not a distributed
  queue or remote worker scheduler.
- Only an exact `SUBMITTED` output can be auto-reconciled.  An expired
  pre-submission attempt remains `RECOVERY_PENDING` and needs a human or a
  future process-level recovery protocol.

## Design provenance

The implementation absorbed invariants from
[`Ephemeral6/modeling-harness` at `e6b4e53`](https://github.com/Ephemeral6/modeling-harness/tree/e6b4e53e5210f5a231993f57e72dd8c36217570e),
not its authority model:

- its [SQLite workflow ledger and normalized ownership](https://github.com/Ephemeral6/modeling-harness/blob/e6b4e53e5210f5a231993f57e72dd8c36217570e/modelharness/workflow.py);
- its [staged, atomic intake](https://github.com/Ephemeral6/modeling-harness/blob/e6b4e53e5210f5a231993f57e72dd8c36217570e/modelharness/intake.py);
- its [graph-derived next packet and stop policy](https://github.com/Ephemeral6/modeling-harness/blob/e6b4e53e5210f5a231993f57e72dd8c36217570e/modelharness/autopilot.py);
- its [optimistic computation followed by a locked second hash pass](https://github.com/Ephemeral6/modeling-harness/blob/e6b4e53e5210f5a231993f57e72dd8c36217570e/modelharness/stages.py#L157-L181);
- its [explicit list of guarantees that remain outside the kernel](https://github.com/Ephemeral6/modeling-harness/blob/e6b4e53e5210f5a231993f57e72dd8c36217570e/docs/FAILURE_MODEL.md#L19-L27).

FMA intentionally diverges where scientific side effects make a retry unsafe:
an expired lease is quarantined instead of returned directly to `PENDING`.
The existing authenticated FMA graph, independent role receipts, domain
checks, and external qualification boundary remain authoritative.

## CLI

Examples:

```powershell
fma-ops --task-root D:\fma-tasks intake `
  --idempotency-key case-001-v1 `
  --objective "Estimate and validate the dynamics described in the attached brief." `
  --attachment .\brief.pdf `
  --attachment .\series.csv

fma-ops --task-root D:\fma-tasks intake `
  --idempotency-key case-001-v1 `
  --objective "Estimate and validate the dynamics described in the attached brief." `
  --attachment .\brief.pdf `
  --attachment .\series.csv `
  --create-task `
  --authority-key-file .\studio.key

fma-ops --task-root D:\fma-tasks next `
  --task-id task-id `
  --authority-key-file .\studio.key

fma-ops --task-root D:\fma-tasks doctor

fma-ops --task-root D:\fma-tasks doctor `
  --authority-key-file .\studio.key

fma-ops --task-root D:\fma-tasks reconcile `
  --authority-key-file .\studio.key
```

Without an authority key, doctor can verify only the operational store and
must report the authority layer as `NOT_RUN`.

## Claim boundary

V7.0 can establish:

- durable operational continuity;
- immutable intake integrity;
- task idempotency and lease ownership;
- packet-to-graph binding;
- verified projection of an already authenticated stage outcome.

V7.0 cannot establish:

- scientific validity;
- external generalization;
- independent external qualification;
- causal or mechanistic truth;
- authorization for a real-world action.
