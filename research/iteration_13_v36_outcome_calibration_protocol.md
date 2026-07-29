# Iteration 13 / V3.6: outcome-calibrated acquisition protocol

Frozen on: 2026-07-22

## Prior failure and component gap

V3.5 report hash:
`f87ff351684617b2c3b1f8f525904c71b6f28fd2240c61849ccb26b8a16c0c01`.

The guarded acquisition package had positive mean improvement `0.01042887`, but
its 95% paired bootstrap interval `[-0.00158224, 0.02571202]` crossed zero and it
caused one material negative transfer. V3.5 was therefore rejected and did not
authorize a router experiment.

The remaining failure is not action execution. It is that a within-episode model
score has no learned relationship to realized outer outcome across episodes.

## Literature lookup and non-transfer boundary

Primary sources consulted on 2026-07-22:

- Angelopoulos et al., *Learn then Test: Calibrating Predictive Algorithms to
  Achieve Risk Control*, [arXiv:2110.01052](https://arxiv.org/abs/2110.01052).
- Angelopoulos et al., *Conformal Risk Control*,
  [arXiv:2208.02814](https://arxiv.org/abs/2208.02814), published at ICLR 2024.

Transferable workflow:

- learn or choose a post-processing parameter only on a calibration set;
- freeze it before a separate evaluation set;
- state the risk/loss being controlled and keep parameter search distinct from
  final testing.

Guarantees not transferred:

- the 30 V3.6 ledger rows are selected action-change cases from two heterogeneous
  protocols, not an exchangeable calibration sample;
- realized target improvement is not the monotone loss setup required by
  conformal risk control;
- the cutoff grid is selected adaptively and the sample is too small for a useful
  negative-transfer upper bound;
- therefore V3.6 claims no conformal coverage, finite-sample risk control,
  calibrated probability, or distribution-free guarantee.

The component is named `OutcomeCalibrationLedger` because it records score and
outcome history; `empirical_training_gate_not_probability_calibration` is frozen
into its schema.

## Frozen training ledger

Training sources:

- V3.3.2 active eligible cases: paired-advantage q20 and realized
  `random_loss - active_loss`;
- V3.5 eligible cases where the acquisition arm actually changed the third
  action: the same q20 and realized `random_loss - unguarded_loss`.

The Harness independently verifies both source runs, reloads their sealed bundles,
and creates 30 typed rows. Private target loss is permitted only in this offline
training artifact; it is never visible to the episode policy.

Frozen cutoff grid: `[0.03, 0.06, 0.09, 0.12, 0.15]`.

For each cutoff, the ledger recomputes the mean outcome gain, a 5,000-replicate
case bootstrap interval, and material negative-transfer count (`gain < -0.02`).
The selection rule is the lowest cutoff with positive training bootstrap lower
bound and zero observed material negative transfers:

| q20 cutoff | n | mean gain | 95% bootstrap CI | material negatives |
|---:|---:|---:|---:|---:|
| 0.03 | 30 | -0.263736 | [-0.857378, 0.048496] | 5 |
| 0.06 | 17 | 0.042295 | [-0.001009, 0.100493] | 2 |
| 0.09 | 8 | 0.087212 | [0.000117, 0.192464] | 1 |
| 0.12 | 4 | 0.087373 | [0.008903, 0.165842] | 0 |
| 0.15 | 2 | 0.103451 | [0.005482, 0.201420] | 0 |

Selected training cutoff: `q20 >= 0.12`.

The table is training evidence only. In particular, `n=4` above the selected
cutoff is an overfitting warning, not strength.

## Fresh three-arm evaluation

V3.6 uses another frozen sixteen-seed / 64-case world pack:

1. `R`: shared-random third action, unguarded;
2. `G`: original V3.5 paired-advantage trust decision plus persistent Adapter;
3. `C`: outcome-calibrated selector plus the same persistent Adapter. `C` uses
   the active proposal only when the original trust decision is active and
   `paired_advantage_q20 >= ledger.selected_cutoff`; otherwise it executes the
   prefrozen random fallback.

The main estimand is `loss(R) - loss(C)`. The original package effect
`loss(R)-loss(G)` and calibration moderation `loss(G)-loss(C)` are diagnostics.

Primary gates remain:

- shared target, anchors, trust, mismatch calibration, noise, and abstention;
- exact policy-to-ledger binding and at least one calibration-caused action change;
- complete segment/intervention/observation/exposure receipts;
- candidate exposure no greater than the random baseline envelope;
- zero synthetic state-envelope violations;
- nonnegative calibrated-package macro 95% bootstrap lower bound;
- mechanism non-regression within `0.02`;
- material negative-transfer rate at most `0.10`;
- worst-case target-loss non-regression.

Passing authorizes only a router experiment for the exact ledger-bound package.
It does not create a general calibration claim, acquisition qualification,
confirmation, real-world safety result, or whole-agent qualification.
