# V6.8 Capability Pack Factory

## Objective

Add mathematical-modeling capability without rewriting the trusted S0--S6
authority graph or copying an entire end-to-end workflow for every domain.

The fast path is not arbitrary code generation. It is:

`reference pack -> declarative manifest -> typed IR -> domain executor ->
independent verifier obligations -> frozen benchmark -> authority-owned
promotion`

V6.0 and V6.7 artifacts retain their original schemas and hashes.

## Implemented foundation

| Component | Current state | Authority ceiling |
|---|---|---|
| V6.8 capability SDK | Implemented | Development control plane only |
| Exact development registry | Implemented | Cannot admit a stage pack |
| Shared executable conformance kit | Implemented | Fixture contract evidence only |
| Two-to-eight branch portfolio protocol | Implemented | Pre-data control artifact |
| Code-owned common RMSE and outer selector | Implemented and replayable | Local selection only |
| Scalar autonomous ODE pack | Typed V6.8 wrapper over unchanged V5.2 kernel | Benchmark cases `NOT_RUN` |
| Pure positive log-increment pack | Typed execution and input-bound verifier | L0 external authority `NOT_RUN` |
| Real two-pack portfolio compilation | Implemented | Branch scheduling not connected |
| Studio S1--S4 scheduling | Not implemented | `NOT_RUN` |
| External/private promotion | Deliberately disabled in V6.8 | No stage-workflow admission |

## Pack construction lanes

Three lanes should be developed in parallel, with separate ownership.

### Lane A: mathematical kernel

The domain implementer supplies only:

- a typed, non-text model IR;
- the candidate grammar;
- code-owned compiler and executor;
- domain-specific numerical routines;
- declared applicability and claim ceiling;
- explicit recovery operators.

The implementer cannot approve its own verifier or promotion.

### Lane B: verifier and evidence obligations

An independent verifier author supplies:

- L0 runtime, replay, and provenance checks;
- L1 measurement, unit, and applicability checks;
- L2 toy-oracle, invariant, and metamorphic checks;
- L3 empirical comparison, identifiability, or optimality checks;
- L4 uncertainty, stress, drift, and support checks;
- failure fixtures and mutant implementations.

Missing computation remains `NOT_RUN`; prose and file presence do not pass.

### Lane C: benchmark and qualification

A benchmark owner freezes:

- canonical gold cases;
- boundary and metamorphic cases;
- baseline/no-signal cases;
- incompatible and non-identifiable cases;
- numerical, data, and provenance traps;
- public real retrospective cases;
- an externally held fresh task.

Private outcomes are not returned to the generator. A code-owned promotion
gate, not the pack or model, changes maturity.

## Shared factory services

Future packs should reuse these implemented services instead of
reimplementing them:

- sealed manifest and exact-hash registry;
- typed IR schema binding;
- implementation and module-source identity;
- resource and recovery envelopes;
- common baseline/loss declarations;
- skeleton duplicate and subsumption rejection;
- direct-callable runtime definitions with no artifact-driven dynamic import;
- executable compiler, executor, verifier, tamper, incompatibility, and
  benchmark cases;
- deterministic conformance reports and sealed case receipts;
- input-bound verifier recomputation;
- local replay diagnostics that cannot close L0 without an external trust root;
- ordered L0--L4 result shape;
- conservative portfolio wall budgets;
- code-owned common loss and executable select/parsimony/abstain logic;
- fail-closed stage and external-promotion boundary.

The positive log-increment pack is the reference statistical implementation.
The scalar ODE typed wrapper is the backward-compatibility reference.

## Promotion ladder

Each pack advances independently for each supported claim:

1. `PROPOSED_UNEXECUTABLE`
2. `DEVELOPMENT_SANDBOX`
3. `LOCAL_CONFORMANCE_PASS`
4. `PUBLIC_BENCHMARK_PASS`
5. `INTERNAL_HIDDEN_QUALIFIED`
6. `REAL_RETROSPECTIVE_QUALIFIED`
7. `EXTERNALLY_QUALIFIED`
8. `HUMAN_ACTION_REVIEW_ELIGIBLE`

The current V6.8 registry exposes only the development sandbox. A local
conformance report is explicitly unable to promote maturity. V6.8 rejects
construction of a stage manifest or stage registry; a signed, committed
external-promotion schema must be additive in a successor version.

## Fastest high-quality pack workflow

Do not clone an end-to-end agent for each domain. Add only four pack-owned
pieces:

1. measurement boundary plus typed IR;
2. mathematical compiler/executor;
3. input-bound L0--L4 recomputation verifier;
4. public gold, rejection, tamper, and `NOT_RUN` cases.

The shared definition map, exact registry, runtime kit, receipts, budget
checks, loss, selector, and claim ceilings are reused. Mathematical
implementation, verifier design, and benchmark ownership proceed in parallel,
then meet at the shared kit. Promotion remains a separate external operation.

This is faster because a new pack does not rebuild routing, replay shapes,
portfolio selection, or workflow authority. It is higher quality because the
same rejection and evidence semantics are applied to every pack.

## Fast implementation sequence for the next pack

The next pack should be a local linear Gaussian state-space model:

1. Freeze its measurement boundary and typed IR.
2. Implement local-level and local-linear-trend candidates.
3. Add persistence and deterministic-trend baselines.
4. Add Kalman likelihood and filtered/forecast replay oracles.
5. Add missing/irregular observation cases only in a successor version.
6. Run the shared conformance suite.
7. Add pack-specific gold, mutant, and metamorphic cases.
8. Compile an ODE/statistical/state-space development portfolio.
9. Run gold-stage injection and component ablations.
10. Freeze a fresh task before any stage-workflow promotion request.

Do not start vector ODE, PDE, network, ABM, control, and causal packs
simultaneously. One reference implementation should first prove that the SDK
reduces new-pack code while preserving rejection and replay quality.

## Verification commands

```powershell
python -m pytest tests/test_v6_8_capability_sdk.py -q
python -m pytest tests/test_v6_8_positive_log_increment.py -q
python -m pytest tests/test_v6_8_scalar_ode_pack.py -q
python -m pytest tests/test_v6_8_capability_catalog.py -q
python -m pytest tests/test_v6_8_capability_runtime.py -q
python -m pytest tests/test_v6_8_portfolio_selection.py -q
python -m pytest tests/test_v6_recovery_kernel.py tests/test_v6_7_predata_protocol.py tests/test_v6_7_predata_transaction.py -q
```

Passing these commands proves local structure, deterministic routing, and the
declared fixture behaviors only. It does not prove external generalization,
scientific qualification, or real-world decision safety.

Remaining infrastructure gaps are Studio/WAL integration, process and memory
enforcement, an external replay signer/trust root, frozen executable ODE
benchmark case IDs, hidden/private qualification, and a fresh unseen task.
