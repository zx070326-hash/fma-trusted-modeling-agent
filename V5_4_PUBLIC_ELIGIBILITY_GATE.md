# FMA V5.4 public eligibility gate

## Objective

V5.4 prevents a candidate from consuming a private evaluation merely because
its pooled public cross-validation mean beats a baseline. It adds a
prospectively frozen, code-owned eligibility decision between public model
selection and private evaluation:

`public candidate evidence -> ELIGIBLE or ABSTAIN -> private request`

It does not reinterpret V5.3 evidence, read private outcomes, grant scientific
qualification, or authorize real-world action.

## Paired evidence

Every loss row binds one candidate and the frozen baseline to the same public
forecast origin and horizon. The gate first averages horizons within each
origin so multiple horizons do not masquerade as independent samples.

The advantage convention is:

`advantage = baseline normalized absolute error - candidate normalized absolute error`

Positive values favor the candidate.

## Frozen checks

A V5.4 contract is sealed before candidate generation and specifies:

- the required forecast horizons and minimum number of public origins;
- the baseline and candidate-selection policy hash;
- the maximum candidate-search count used for multiplicity correction;
- the number of contiguous time blocks and recent origins;
- minimum overall, block, recent, horizon, and win-fraction advantages;
- moving-block bootstrap length, replicates, confidence, and seed;
- non-fixture and independently verified scientific-acceptance requirements.

The deterministic assessment requires all checks to pass:

1. complete paired origin-horizon coverage;
2. sufficient public origins;
3. positive pooled mean advantage;
4. positive selection-adjusted moving-block-bootstrap lower bound;
5. the frozen origin win fraction;
6. positive advantage in every contiguous time block;
7. positive advantage in the recent window;
8. positive mean advantage at every horizon;
9. positive recent advantage at every horizon;
10. non-fixture evidence and verified public scientific acceptance.

Any failed check produces `ABSTAIN`. Missing, malformed, duplicated, or
out-of-budget evidence is rejected before an assessment can be created.

## Authority boundary

The assessment is deterministic but not self-authorizing. A separate harness
authority signs the assessment with Ed25519. The public key can be pinned by a
remote worker; no shared signing secret crosses the host boundary.

`PrivateEvaluationAuthorizationV54` can be constructed only when:

- the contract, input, and assessment are sealed and mutually bound;
- the assessment is `ELIGIBLE`;
- the eligibility receipt has a valid Ed25519 signature;
- the V5.3 private request is sealed and bound into the authorization.

The authorization still grants no scientific qualification. It only proves
that the public evidence met the preregistered conditions for presenting one
private request.

## Iteration 32 counterfactual

The consumed I32 shadow candidate had better pooled public normalized MAE than
persistence, but it would have received `ABSTAIN` under V5.4:

- `public_scientific_acceptance` was false;
- only 46.15% of public origins favored the candidate;
- the final contiguous time block favored persistence;
- the recent aggregate advantage was negative;
- every recent horizon advantage was negative.

This is a prospective control learned from a preserved negative result. I32
must not be retuned or rescored.

## Verification

Focused tests cover stable eligibility, Ed25519 verification, wrong-key
rejection, late-regime collapse, bootstrap-only rejection, fixture and
scientific-assessment rejection, incomplete grids, frozen search budgets, and
the I32 counterfactual.

## V5.5 launch integrity

V5.4 assumes its baseline and candidate-policy bindings were frozen correctly.
V5.5 makes that assumption executable: it derives both artifacts from one
sealed prospective protocol and separates source-provenance encryption from
private-target encryption. See
[`V5_5_CAMPAIGN_INTEGRITY.md`](V5_5_CAMPAIGN_INTEGRITY.md).
