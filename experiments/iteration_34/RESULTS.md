# Iteration 34 real World Bank ODE qualification experiment

Status: `PUBLIC_ABSTAIN_PRIVATE_NOT_RUN_CLOSED_OUT`

## Decision

The frozen public gate returned `ABSTAIN`. The single private evaluation was
not authorized and remains unused (`0/1`). This iteration grants no scientific
qualification and authorizes no real-world action.

The rolling-origin rule and the independent V5.3 adapter both selected
`logistic`. All four registered candidate grids completed and both
authenticated replay processes produced the same deterministic output hash.
The public scientific acceptance still failed because the selected model's
development residuals exceeded the frozen lag-correlation limit:

- absolute lag-1 residual correlation: `0.9591695185`;
- frozen maximum: `0.85`;
- V5.3 check: `l3_residual_lag_bounded=false`.

The underlying V5.2 evidence also reported a raw logistic parameter condition
number of `3.5946104960e10`, above the frozen `1e10` threshold. V5.3's
scale-aware condition check passed, but it did not remove the residual failure.
No threshold, family, window, transform, or stopping rule was changed after
seeing this result.

## Public predictive evidence

The failure was not caused by poor paired forecast loss. On the frozen 13
origins by four horizons:

| Rank | Family | Aggregate normalized MAE | Aggregate normalized RMSE |
|---:|---|---:|---:|
| 1 | logistic | 0.025703054 | 0.030064450 |
| 2 | gompertz | 0.028737087 | 0.033769188 |
| 3 | exponential | 0.106448096 | 0.123861936 |
| 4 | constant | 1.338850253 | 1.373156477 |

The required `persistence_last_value` baseline had aggregate normalized MAE
`0.191951986`. Logistic beat it at every aggregated public origin:

- mean paired advantage: `0.166248933`;
- multiplicity-adjusted bootstrap lower bound: `0.152053792`;
- origin win fraction: `1.0`;
- contiguous-block, recent-window, every-horizon, and every recent-horizon
  advantage checks: all pass.

V5.4 therefore failed only `public_scientific_acceptance`; all other public
stability checks passed.

## Source and closeout

After the signed `ABSTAIN` terminal evidence, the provenance-only closeout
worker released the source record:

- authority: World Bank Indicators API v2;
- series: `Urban population - Viet Nam` (`SP.URB.TOTL`, `VNM`);
- public period: 1990-2017;
- unrevealed-target period: 2018-2021;
- source artifact SHA-256:
  `45be724f7ae6019c97793e4add15d5cb09f332a717ad499e0ca507e122fea75d`.

The release path cannot accept the target envelope or target key. Its receipt
records:

- `private_target_envelope_accessed=false`;
- `private_target_key_accessed=false`;
- `private_evaluation_performed=false`;
- `scientific_qualification_granted=false`.

The source is now public after closeout, so this task can never be reused as a
private qualification task.

## Evidence bindings

- frozen public manifest:
  `10d25bc781abb258a0ae09f08dbec408c9c63eace1d39bf86fb95198e25496a6`;
- candidate evidence:
  `1a432c7fa50d8a067852e4af4ade75c6d28779632f45aab57886058623eef61e`;
- V5.3 forecast bundle:
  `661c93b5431903ed97f1ebaedc438c261d572bb29e01ee1129c1dbea5da43004`;
- V5.4 assessment:
  `c6018c574c2cd802801af9ca944525e83081d81a9b83893a3efd0b2e36d3afb8`;
- V5.4 signed receipt:
  `45a49da70a3a25b1c5d94c95197dbe30227864ae22103fb2c99c87799cce7d2b`;
- terminal public result:
  `9cb3863cebc529e5e2080758467015c07653091579a22485234cf77ed5c41f5d`;
- closeout authorization:
  `976ff92c27bdd385a61a35502fd12f3d14724a545ccaf1e8d6f107a01a3f17c1`;
- source disclosure receipt:
  `9a699dd1c721232cbe4ca4f40f6e6075d1f43ef1ac4beb432a9e83690c55a015`.

The public runner was frozen at commit `d002d0068ed75b38f3fe141ff9f408c6af8d4b0d`
before the real result was generated.

## Capability conclusion

This run establishes a real, public-only capability:

1. integrity-check a blinded real-data launch;
2. execute a frozen graph of four ODE candidates;
3. lock the family before final refit;
4. obtain two authenticated fresh-process replays;
5. independently recompute and sign the V5.4 public decision;
6. stop without consuming private data; and
7. release provenance through a target-incapable closeout path.

It does not establish that an autonomous scalar ODE is a credible causal model
of urban population, that the agent can solve arbitrary modeling problems, or
that the workflow is externally qualified.

## Prospective repair direction

The result identifies a narrow scientific gap rather than a reason to weaken
the gate: strong forecast advantage can coexist with serially structured
residuals and weak raw parameter identifiability. Any follow-up must be a new,
prospectively frozen task. Its adapter should compare the autonomous ODE
skeleton against a non-autonomous/state-space alternative with an explicit
residual process, reparameterized dimensionless fitting, and the same
public/private custody boundary. Iteration 34 itself remains closed.
