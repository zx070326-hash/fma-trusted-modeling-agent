# Modeling Harness 4.0.1 vs THIN 0.2.0

## Executive conclusion

The most important difference is not graph sophistication.

- The friend's harness is a **sidecar research operating system**. A native
  Codex, Claude Code, or similar coding agent works directly in the project
  with its normal file and terminal abilities. `modelharness` exposes the next
  work packet, tool policy, evidence operations, and status, but does not call
  the model itself.
- THIN is an **in-loop model conductor**. It launches a new, ephemeral,
  tool-disabled Codex process for every modeling or verification step and
  requires a structured JSON action. THIN then executes five narrow tools on
  the model's behalf.

The friend's design preserves more of the foundation model's native research
and coding capability. THIN has a smaller and clearer authority boundary, but
its current execution topology fragments research, adds review tax, and
prevents the model from using the native Codex environment that already solves
many modeling tasks well.

The redesign should therefore keep THIN's small trusted kernel while moving the
researcher back to a native project-local Codex session. Control the task
contract, external side effects, evidence promotion, and final claims - not
each internal research action.

## Audited versions

| System | Version/commit | Local path |
|---|---|---|
| Friend Modeling Harness | 4.0.1, `6992ce8af5c8c88a21f9752b0c2a8e09f0bbd667` | `research/friend-modeling-harness` |
| THIN Modeling Agent | 0.2.0, current dirty worktree | `modeling_agent` |

The friend implementation previously used in the ICM F and Silvermere runs was
3.1.0 at `f7fe298`. Version 4.0.1 adds about 3,200 lines and changes review
lineage, runtime state, proposals, scheduling, benchmarking, and recovery.
Those old task scores do not validate 4.0.1.

## Measured implementation surface

| Measure | Friend 4.0.1 | THIN 0.2.0 |
|---|---:|---:|
| Runtime Python files | 52 | 8 |
| Runtime Python lines | 7,772 | 2,579 |
| Template files | 52 | none |
| Template lines | 1,759 | none |
| Test files | 15 | 1 |
| Tests currently passing | 43 | 29 |
| Full test time on this host | 18.43 s | about 1.5 s |
| Fresh-project files | 59 | 2 |
| Fresh-project top-level runtime directories | 13 | 1 |
| Core dependencies | none | none |

Both repositories currently pass their own tests and CLI help smoke checks.
These tests establish implementation contracts, not scientific superiority.

## Architecture comparison

| Dimension | Friend 4.0.1 | THIN 0.2.0 | Assessment |
|---|---|---|---|
| Model driver | External native Agent reads `AGENTS.md` and repeatedly invokes CLI | Code launches Codex CLI itself | Friend preserves native Agent competence; THIN is easier to replay |
| Model context | Native coding-agent session and project filesystem | Fresh ephemeral process each step; recent observations reconstructed | THIN loses continuity and pays repeated context reconstruction |
| Model tools | Native project-local terminal/files plus registered toolchain | `read_text`, `read_files`, `write_text`, `write_files`, restricted Python | THIN is safer but too narrow for open modeling |
| Initial problem structure | Seven S0-S6-derived nodes already exist | One root node; model may revise graph | THIN is more genuinely open-ended |
| Research scheduler | Frontier scoring, durable tasks, leases, recovery, State Capsule | Model selects work directly from graph context | Friend is stronger for long multi-worker projects; excessive for ordinary tasks |
| Evidence | Typed graph, producer/reviewer identity, append-only review lineage, freshness | working/claim admission, artifact hashes, checks, fresh verifier | Both have the right principle; friend has much more machinery |
| Execution semantics | execution status, verdict, authority, freshness are orthogonal | compact tool result plus evidence status | Friend is more expressive for ambiguous recovery; THIN is easier to understand |
| Tools | Capability catalog, environment doctor, use/skip decisions, run/recover/verify | Five fixed local tools | Friend is discoverable and broad; its catalog creates setup and policy overhead |
| Delivery | Profiles, method packs, S0-S6 milestone projection, narrative audit | best-effort submission plus final verifier | THIN has the better default; friend better supports contest-specific packaging |
| Evaluation | Episode packages and rubric engine; only two example fixtures | Frozen raw-vs-THIN manifest; task-specific evaluator is external | Neither currently has a real multi-task capability benchmark |
| Claim boundary | Workflow/evidence qualification separate from deliverable | Verified final separate from best-effort submission | Both are conceptually sound |

## What the friend designed well

### 1. Native Agent freedom

The harness does not replace Codex's own planning, terminal, file editing, and
debugging loop. This was the main reason the 3.1 arm could produce a coherent
ICM F paper and executable model while THIN exhausted its step budget.

### 2. Agent-legible project state

`work next` returns a concrete frontier, task contract, inputs, acceptance
conditions, tool plan, failure state, and stop policy. A human or Agent can
inspect the same durable state.

### 3. Tool capability discovery

The doctor and tool registry distinguish available capabilities from missing
optional software. This is better than making the model guess which scientific
libraries and solvers exist.

### 4. Append-only review and recovery

Review revisions are preserved rather than overwritten. Execution outcome,
scientific verdict, reviewer authority, and artifact freshness are not
collapsed into one PASS/FAIL flag.

### 5. Evaluation episodes

The episode package is the right unit for external comparison: task, config,
environment, events, tools, evidence, reviews, result, paper, model, and
budget. It should be retained in a smaller form.

## Where the friend remains over-engineered

### 1. The default scaffold encodes a process before understanding the problem

A fresh project creates 59 files and reports seven S0-S6 nodes. Even if the
documentation calls stages projections, the first actual work packet is still
`s0.problem_definition`, with a role, method pack, tool decision, producer task
identity, evidence IDs, contract hash, and acceptance objects.

This is a large cognitive tax before any mathematical work occurs.

### 2. It is not a self-driving Agent

There is no OpenAI, Anthropic, or Codex model adapter in the runtime. The CLI
creates work packets but does not launch the worker or fresh reviewer. Protocol
correctness therefore depends on the external Agent following a long operating
manual. The 3.1 ICM run demonstrated this failure: review tasks and evidence
verification were invoked in the wrong order.

### 3. The state surface is larger than the scientific state

Problem Graph, workflow tasks, tool plans, tool decisions, tool runs, reviews,
evidence, profiles, method packs, stages, proposals, stamps, events, and State
Capsule projections all need reconciliation. The design tries to keep these
authorities separate, but the Agent still has to understand their interactions.

### 4. Method packs and profiles can become soft model-family constraints

They are advisory rather than hard allowlists, but they bias the Agent toward
predefined workflows and increase prompt/context volume. They should be
retrieved only when a task demonstrates a need.

### 5. Its Benchmark Lab is mainly infrastructure

The repository explicitly says that its two fixtures are format examples, not
the proposed 40-60 L1, 15-20 L2, and three-domain L3 suite. Current benchmark
code therefore does not establish research performance.

## Where THIN is better

1. One root problem is the only initial scientific commitment.
2. Research, execution, and evidence facts live in one compact durable state.
3. Status and delivery are derived rather than maintained as additional
   authorities.
4. The harness itself launches distinct modeler and verifier contexts.
5. Best-effort delivery survives failed qualification.
6. The system is small enough to audit and change surgically.

## Where THIN is currently worse

### 1. It disables the strongest part of Codex

Every model call is read-only, ephemeral, tool-disabled, history-free, and
schema-constrained. The model cannot directly inspect, edit, execute, debug,
and iterate as a native coding/research agent. It must express all work through
THIN's miniature JSON tool protocol.

### 2. It fragments the research trajectory

One modeling task becomes many independent calls. The state summary preserves
facts but not the full local problem-solving momentum that makes long native
Codex runs effective.

### 3. Verification is too frequent

The model spends turns packaging intermediate evidence and responding to
reviewer wording differences. On ICM F this prevented final synthesis. On
Silvermere it consumed 735 seconds and still ended on a filename assumption.

### 4. The tool surface is too restrictive

Restricted Python is useful for untrusted computation, but open mathematical
modeling often needs normal scientific packages, multiple scripts, plotting,
document rendering, solver discovery, and ordinary shell diagnostics.

### 5. It lacks a first-class task contract

The task is stored as objective text. Mandatory output fields and deliverable
paths are not compiled into a deterministic final conformance check. This
allowed avoidable delivery/check-path failures.

## Empirical evidence so far

The tested friend version was 3.1, not the downloaded 4.0.1.

| Frozen task | Raw Codex | THIN 0.2.0 | Friend 3.1 |
|---|---|---|---|
| 2026 ICM F | Complete, blind score 89 | Partial, score 84 | Complete analytical bundle, score 92; qualification incomplete |
| Silvermere hidden future | Valid, 2.078% nRMSE, zero decision regret | Valid, 9.119% nRMSE, 1.485% regret | Invalid task contract; 7.160% diagnostic nRMSE |

The evidence supports three limited claims:

1. native model autonomy can outperform THIN's fragmented loop;
2. the friend's sidecar approach can preserve solution closure, but may fail
   task grounding and protocol conformance;
3. neither harness has shown consistent gain over Raw Codex or a simple
   baseline.

## Recommended THIN redesign

### Keep

- one mutable Problem Graph starting from a single root;
- project-local side-effect boundary;
- artifact hashes and structured execution observations;
- working versus decision-relevant claim distinction;
- fresh independent verification for decision-critical claims;
- revocation when source artifacts or graph obligations change;
- best-effort delivery and explicit claim ceiling;
- frozen external ablations.

### Absorb from the friend

- native project-local Codex as the primary researcher;
- a lightweight `doctor` that reports available scientific capabilities;
- machine-readable task and deliverable contracts;
- append-only evaluation episodes;
- explicit recovery for unknown execution outcomes;
- declared generator/check paths rather than filename conventions.

### Do not absorb

- pre-created S0-S6 problem nodes;
- fixed role queues;
- method-pack routing in the default path;
- delivery profiles in the research kernel;
- task leases and multi-worker scheduling before parallelism is measured as
  necessary;
- milestone stamps;
- broad proposal/state ledgers for ordinary project-local research;
- a large static tool catalog as mandatory context.

## Target architecture

```text
user problem + attachments
        |
        v
small task-contract compiler
  - required outputs
  - hard constraints
  - evaluation hooks
        |
        v
native project-local Codex researcher
  - normal read/write/shell/scientific tools
  - model-owned decomposition and direction changes
  - one durable research session or resumable task
        |
        +----> lightweight Problem Graph / research notes
        |
        +----> artifacts + declared generators + checks
        |
        v
deterministic conformance and replay
        |
        v
fresh verifier only for decision-critical claims and final answer
        |
        +---- reject --> native researcher repairs
        |
        v
best artifact bundle + bounded claims + evaluation episode
```

The trusted kernel should own only:

1. task-local filesystem and external-action permissions;
2. budgets and stop conditions;
3. task contract and final conformance;
4. artifact/run receipts;
5. evidence promotion and revocation;
6. fresh verifier launch;
7. external evaluation packaging.

Everything else should remain in the model's research workspace unless a
repeated frozen-task failure proves that code-owned orchestration is necessary.

## Immediate next experiment

Before rewriting the package, implement this topology as one ablation arm:

`Raw Codex` vs `THIN 0.2.0` vs `Native Codex + Thin Sidecar`.

Use externally supplied frozen tasks, not a task generated by the same Agent.
Freeze the task contract, grader, simple baseline, model, budgets, and stop
rules. Measure scientific score, contract pass rate, replay, wall time, model
turns, verifier calls, and human intervention. Only replace THIN 0.2.0 if the
sidecar arm wins across multiple tasks rather than one anecdote.
