# Iteration 34 prospective scientific protocol

Status: `FROZEN_BEFORE_SOURCE_SELECTION`

## Question

On one newly selected, complete, positive annual scalar series from the frozen
World Bank Indicators API candidate pool, can the graph-governed scalar ODE
workflow:

1. select a candidate using public rolling-origin evidence only;
2. satisfy the frozen V5.2 L0-L4 scientific checks;
3. show stable public advantage over the exact
   `persistence_last_value` baseline under V5.4; and
4. preserve the private target budget unless all public requirements pass?

The primary endpoint is the V5.4 public decision: `ELIGIBLE` or `ABSTAIN`.
Private score and scientific qualification are not primary endpoints and are
not run without the separately administered external boundary below.

## Source selection

- API: World Bank Indicators API v2 over HTTPS.
- Candidate set: the exact country and indicator codes in
  `SOURCE_SELECTION_SPEC_V55.json`.
- Window: 1990-2017 public, 2018-2021 private.
- Eligibility: all 32 annual values must be present, positive, finite, and
  unique by year.
- Selection: HMAC the secret 32-byte selection seed with every
  country-indicator pair, sort by the digests, and choose the first pair that
  meets only the data-quality rule.
- Selection must not use trend shape, model fit, forecast loss, candidate
  identity, target values, or a desired result.
- The public series receives only a positive scaling and the time coordinate
  receives only an additive translation. These preserve the registered
  autonomous-ODE family forms.
- Exact country, indicator, URL, raw response hash, years, and transforms stay
  in the encrypted provenance/target domains until closeout.

The candidate pool is public, so source inference from the released trajectory
is possible. This campaign establishes logical blinding only; it does not
establish information-theoretic source secrecy or egress restriction.

## Candidate graph and selection

The registered nodes are:

1. `constant` as the minimal equilibrium skeleton;
2. `exponential` as the unrestricted proportional-growth evolution;
3. `gompertz` as an asymmetric saturation recovery;
4. `logistic` as a symmetric saturation recovery;
5. `persistence_last_value` as the non-ODE comparison baseline.

All four ODE families are evaluated on the same 13 expanding public origins
(origin sizes 12 through 24) and horizons 1 through 4. The graph records
constant/exponential as the initial simple branch and Gompertz/logistic as the
saturation recovery branch. No new family, trailing-window variant, transform,
hyperparameter, or threshold may be introduced after public results.

Selection is lexicographic:

1. lowest aggregate normalized MAE;
2. then lowest aggregate normalized RMSE;
3. then lower complexity order:
   `constant < exponential < gompertz < logistic`.

The final family is locked before fitting all 28 public observations. The
V5.3 typed bundle is computed independently. If its selected family differs
from the frozen rolling-origin selection, public scientific acceptance is
false and V5.4 must `ABSTAIN`.

## Public scientific and stability gates

- V5.2 thresholds are exactly `ODE_THRESHOLDS_V52.json`.
- L0 requires two authenticated fresh-process replays with identical
  deterministic output hashes.
- L1-L4 use the frozen V5.2/V5.3 adapter checks without weakening.
- V5.4 uses the exact settings sealed in
  `PROSPECTIVE_CAMPAIGN_PROTOCOL_V55.json`.
- `public_scientific_acceptance_verified` is true only if L0-L4 pass and the
  selected-family alignment above holds.
- V5.4 must recompute public evidence and require every check.
- `ABSTAIN` consumes zero of one private evaluations.

## Private and closeout rules

- Private targets and source provenance use separate AES-256-GCM keys.
- Public release includes only ciphertexts, commitments, signed split custody,
  transformed public observations, and typed public contracts.
- Same-host logical custody cannot produce external qualification.
- Private evaluation may run only after `ELIGIBLE`, through the V5.5 authorized
  encrypted worker, on a separately administered host with pinned independent
  custody, worker, host-attester, ledger, and promotion keys.
- If that external boundary is unavailable, private status is
  `BLOCKED_NOT_RUN`; the private budget remains 0/1.
- After `ABSTAIN` or terminal external blockage, an independent closeout key
  may release source provenance only. The target envelope and key remain
  unopened.

## Fixed claim limits

- A passing public gate is not scientific qualification.
- A same-host custodian process or fresh chat is not an external host.
- A World Bank observational series does not establish a causal autonomous ODE
  mechanism.
- This single task cannot establish general mathematical-modeling capability.
- `scientific_qualification_granted=false` until an independent promotion
  decision verifies all external evidence.
- `real_world_action_authorized=false`.

## Stopping rules

Stop and preserve evidence when:

- protocol/spec/threshold/hash binding fails;
- no candidate source satisfies the frozen data-quality rule;
- source or target plaintext reaches generator context before closeout;
- any public scientific or V5.4 check fails;
- the selected model/graph differs from the frozen rule;
- an external worker or independent-key claim is missing or unverifiable;
- the single private evaluation is claimed, succeeds, fails, crashes, or times
  out.

No rule, candidate, source pair, split, threshold, or private retry may be
changed after source selection.
