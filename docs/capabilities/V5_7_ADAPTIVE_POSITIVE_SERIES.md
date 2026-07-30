# V5.7 adaptive positive-series modelling protocol

## Purpose

V5.6 correctly rejects positive scalar series whose apparent residual process
is nonstationary, but it cannot change the representation. V5.7 adds one
bounded recovery representation for positive, regularly sampled scalar
series: stochastic dynamics of log increments.

I35 is now disclosed development evidence. It may diagnose and test V5.7, but
it can never regain unseen-task or qualification status. A future I36 must be
selected only after V5.7 freezes and must exclude both I34 and I35.

## Candidate graph

1. Run the complete V5.6 hybrid autonomous-ODE branch.
2. Treat V5.6 L1–L4 as the primary scientific branch. V5.7 supplies its own
   independently replayed L0.
3. If every primary level passes, keep the V5.6 selected candidate.
4. Otherwise generate exactly two recovery candidates:
   - `log_random_walk_drift`: stationary log increments with constant mean;
   - `log_growth_ar1`: stationary AR(1) log increments around a constant mean.
5. Select only among scientifically admissible recovery candidates using
   chronological one-step validation loss plus the frozen complexity penalty.
6. If no recovery candidate is admissible, retain the best diagnostic
   candidate and fail L3. No model family may be generated after results.

This graph is predictive. It does not infer a causal macroeconomic mechanism.

## Frozen guards

- chronological split: 70/30, at least eight observations per slice;
- validation relative RMSE at most 0.15;
- improvement over level persistence at least 0.10;
- absolute innovation lag-1 correlation at most 0.35;
- absolute growth AR coefficient at most 0.95;
- AR(1) growth must improve its same-family drift baseline by at least 0.05;
- growth-AR coefficient window range at most 0.30;
- standardized drift window range at most 1.0;
- validation/training innovation mean shift at most 1.5 standard deviations;
- largest innovation at most 5.0 standard deviations;
- validation interval coverage at least 0.50;
- absolute mean annual log growth at most 0.50;
- bootstrap success at least 0.80 over 40 frozen-seed replicates;
- relative forecast interval width at most 2.0;
- refit-window forecast sensitivity at most 1.0.

## Evidence levels

- L0: two authenticated fresh-process deterministic replays bound to source,
  executable, environment, input bytes, and semantic input.
- L1: sealed positive series, units, cadence, split sizes, and exact graph.
- L2: log/level forecast identities, zero-phi reduction, stationary
  mean-reversion, and positive-scale invariance.
- L3: chronological predictive validity and all candidate-specific guards.
- L4: residual bootstrap, refit-window sensitivity, branch ablation, and claim
  limits.

## Claim boundary

Passing V5.7 means one registered predictive representation survived the
frozen public checks. It does not establish causal identification, universal
mathematical-modelling competence, external-host independence, private
qualification, or real-world action authority.
