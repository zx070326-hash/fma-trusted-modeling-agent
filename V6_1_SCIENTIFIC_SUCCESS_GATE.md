# FMA V6.1 scientific-success gate

## Purpose

V6.1 separates five statements that were previously easy to conflate:

1. the S0--S6 workflow is internally valid;
2. the registered adapter passed its local L0--L4 checks;
3. the model-selection pipeline survived leakage-safe confirmation;
4. the data, mechanism, decision value, and generalization support the
   requested claim;
5. an independent authority granted scientific qualification.

Only the first three can currently be computed inside the Studio path. The
local gate cannot grant scientific qualification or authorize action.

## Frozen contract

S2 writes `docs/scientific_success_contract_v61.json` before S3 computation.
The contract binds:

- the workspace and selected capability pack;
- the claim kind (`predictive` for the current packs);
- the confirmation method;
- the required success dimensions;
- quantitative thresholds;
- the prohibition on private adaptive feedback and action authority.

The contract is included in the S2 manifest. Recovery therefore revokes and
quarantines it with the adapter binding instead of silently reusing an old
success definition after a model branch.

## Leakage-safe confirmation

The V5.2 adapter uses one 70/30 development split both for candidate selection
and local L3 diagnostics. V6.1 does not promote that slice into an independent
test. It evaluates the entire selection pipeline using one-step rolling
origins:

```text
for each final confirmation origin:
    prefix only
      -> fit/select the registered candidate graph
      -> freeze one-step prediction
      -> compare with the next observation
aggregate predictions
  -> compare against persistence
  -> check residual dependence
  -> check interval coverage
  -> require every inner selection to be admissible
```

The ODE contract requires at least 17 history points plus six confirmation
origins (23 total). The adaptive contract requires at least 26 history points
plus eight confirmation origins (34 total). A shorter series remains
executable when its adapter permits it, but confirmation is `NOT_RUN`.

## Dimensions

Every report contains all dimensions with `PASS`, `FAIL`, `NOT_RUN`, or
`HUMAN`:

| Dimension | Current evidence source |
|---|---|
| workflow integrity | current authenticated S0--S6 gates and graph replay |
| data provenance | fixture flag and source snapshot; independent source review is not yet implemented |
| local adapter checks | registered ODE or adaptive L0--L4 bundle |
| leakage-safe confirmation | V6.1 nested rolling-origin computation |
| decision value | `NOT_RUN`; no regret/utility evaluator exists |
| mechanism identification | `NOT_RUN`; current packs make no causal claim |
| external generalization | `NOT_RUN`; no independent environment result |
| scientific qualification | `NOT_RUN`; requires separate private promotion |

Fixture data force the scientific-success result to `NOT_RUN` even when the
local predictive gate passes. Non-fixture user-supplied data require `HUMAN`
provenance review until an authenticated source adapter exists.

## Outputs

After S6, the harness commits a content-addressed
`scientific_success_report_v61` and writes a non-authoritative projection at:

```text
.fma/scientific_success_v61/report.json
```

Task snapshots and the frontend expose:

- `local_predictive_gate_status`;
- `scientific_success_status`;
- `claim_ceiling`;
- every dimension and reason code;
- rolling-confirmation metrics and model lineage;
- fixed `scientific_qualification_granted=false`;
- fixed `real_world_action_authorized=false`.

The projection becomes unreadable as current evidence when any bound gate is
revoked. Historical content-addressed evidence remains preserved.

## Current claim boundary

The strongest current result is either:

- `fixture_protocol_only`, or
- `local_retrospective_adapter_evidence`.

The architecture contains a future
`local_leakage_safe_predictive_evidence` ceiling, but the present Studio input
path has no authenticated independent data-provenance verifier. Therefore a
locally supplied series cannot silently obtain full scientific success.

Decision success, causal mechanism success, external generalization, and
scientific qualification remain separate future adapters, not prose fields a
model can self-approve.
