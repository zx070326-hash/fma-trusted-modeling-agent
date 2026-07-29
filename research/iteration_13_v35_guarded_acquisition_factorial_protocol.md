# Iteration 13 / V3.5: guarded acquisition factorial retest

Frozen on: 2026-07-22

## Prior evidence

- V3.3.2 acquisition failure report:
  `24d143f42011dc2d1e2c1b2efcf527c37831704cfe43d41e33a9f60da64ba830`.
- V3.4 single-exceedance Adapter failure report:
  `865bc6b481f884c45a79a67ab6c868a90e8056e65df43ace11c72fec4f703ecf`.
- V3.4.1 persistent Adapter report:
  `eeffd042c3197812d32e9d92e187bcbca29b8902b6d4a2ce708c3e0b5b6230fb`.

V3.4.1 passed only the interaction-layer gates and explicitly authorized an
acquisition retest. It did not qualify acquisition, routing, the whole agent, or
real-world action.

## First-principles causal design

A two-arm `random` versus `guarded acquisition` comparison would estimate the
package effect but could not distinguish whether benefit came from acquisition,
the Adapter, or their interaction. V3.5 therefore freezes three paired arms on
the same private cases:

1. `R — shared_random_baseline`: two shared random anchors followed by the next
   prefrozen random action, executed without online interruption.
2. `A — paired_advantage_unguarded`: the exact V3.3.2 paired-advantage trust
   decision selects the third action, which is executed in full.
3. `A+G — paired_advantage_persistent_guard`: the exact same selected action is
   executed through the exact V3.4.1 two-consecutive-exceedance Adapter.

All three arms share:

- problem clarification and data-quality gates;
- the first two action ids and observation hashes;
- acquisition receipts and paired trust decision;
- the case-local leave-one-anchor-out mismatch calibration;
- the common streaming observation-noise schedule;
- estimator, target probes, private adjudicator, exposure maxima, and statistical
  gates.

The three estimands are:

- acquisition main diagnostic: `loss(R) - loss(A)`;
- guard moderation diagnostic: `loss(A) - loss(A+G)`;
- primary deployable package effect: `loss(R) - loss(A+G)`.

Only the third estimand controls V3.5 readiness. The other two remain explicit so
the report cannot attribute a package effect to the wrong component.

## Action and abstention invariants

- `R` must execute the trust receipt's `fallback_action_hash`.
- `A` and `A+G` must execute the trust receipt's `selected_action_hash` as their
  proposal; their proposed action hashes must match exactly.
- If the trust decision falls back, all three arms execute the same action and
  no acquisition change is counted.
- If no admissible anchor/third action exists, all three arms abstain with the
  same typed reason. No replacement action or threshold relaxation is allowed.
- `A+G` may execute a different intervention only after two consecutive online
  mismatch exceedances. Its exposure must be component-wise no greater than `R`
  and `A` for duration, energy, peak, and switch count.

## Fresh formal evaluation

V3.5 freezes a new sixteen-seed set before private world-pack generation. The
primary `R` versus `A+G` package must pass every gate:

- target, anchor, trust, common-noise, and abstention parity across all three arms;
- `R` fallback binding and `A/A+G` selected-action binding;
- at least one actual acquisition action change (`selected != fallback`);
- complete calibration, segment, observation, intervention, and exposure receipts;
- candidate exposure component-wise dominated by both comparison arms;
- zero synthetic hidden-state envelope violations;
- nonnegative paired macro-improvement 95% bootstrap lower bound;
- no mechanism mean regression worse than `0.02`;
- material negative-transfer rate at most `0.10`;
- worst-case target-loss non-regression.

No minimum Adapter interruption is imposed in V3.5 because the Adapter was the
subject of the separate V3.4.1 experiment; requiring an interruption here would
condition package evaluation on a post-treatment event. The interruption count
and guard moderation effect remain mandatory diagnostics.

Passing would make only the exact `paired advantage + persistent Adapter` package
ready for a router experiment. It would not generate an acquisition qualification,
confirmation, formal safety claim, or real-world authorization.
