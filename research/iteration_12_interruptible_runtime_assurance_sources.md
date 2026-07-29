# Iteration 12: interruptible Reality Adapter evidence boundary

Access date: 2026-07-22

## Decision first

Iteration 12 will test one component only: whether the exact V3.3.2 action proposer
causes less negative transfer when every proposed experiment is executed through a
segment-authorized, monotonically de-escalating Reality Adapter.

This is an empirical synthetic runtime-assurance experiment. It is not a theorem,
not a calibrated probability claim, and not evidence that a real plant is safe.
The online gate may only reduce authority: continue the frozen action, switch
permanently to zero input, or terminate. It may never increase amplitude, energy,
duration, or switching authority after seeing an observation.

## Reproducible literature lookup

Databases queried:

- OpenAlex API, `https://api.openalex.org/works`, bibliographic searches:
  - `predictive safety filter learning based control constrained nonlinear dynamical systems`
  - `runtime assurance simplex architecture learning enabled control`
  - `safe exploration dynamical systems backup controller online model mismatch`
- Crossref API, `https://api.crossref.org/works`, bibliographic search:
  - `Learning-based model predictive control Safe exploration robustness guarantees`
- arXiv records and publisher DOI landing pages were used to verify the selected
  papers rather than treating search rank as evidence.

Observed retrieval limits:

- The third OpenAlex query was broad and noisy.
- A later Crossref request returned HTTP 429. This is a retrieval-rate failure,
  not evidence that a paper or concept is absent.
- Exact OpenAlex DOI resolution succeeded for Wabersich and Zeilinger. Exact
  OpenAlex resolution did not return usable records for two other identifiers,
  so their arXiv and publisher records were used instead.

## Sources, transferable ideas, and non-transferable guarantees

### Predictive safety filter

Wabersich, K. P., and Zeilinger, M. N. (2021), *A predictive safety filter for
learning-based control of constrained nonlinear dynamical systems*, Automatica
129, 109597. DOI: [10.1016/j.automatica.2021.109597](https://doi.org/10.1016/j.automatica.2021.109597).
Preprint: [arXiv:1812.05506](https://arxiv.org/abs/1812.05506).

Transferable architecture:

- inspect the primary controller's proposed input at the current state;
- place an independently implemented filter between proposal and actuation;
- replace an inadmissible proposal with a predeclared backup behavior.

Not transferable to FMA V3.4:

- probabilistic constraint-satisfaction guarantees;
- MPC recursive-feasibility conclusions;
- any guarantee requiring a calibrated uncertainty model, invariant terminal
  set, or verified plant assumptions.

### Safe exploration with model predictive control

Koller, T., Berkenkamp, F., Turchetta, M., and Krause, A. (2018),
*Learning-based Model Predictive Control for Safe Exploration*. DOI:
[10.1109/CDC.2018.8619572](https://doi.org/10.1109/CDC.2018.8619572).
Preprint: [arXiv:1803.08287](https://arxiv.org/abs/1803.08287).

Transferable architecture:

- uncertainty about dynamics must affect permission, not merely ranking;
- a backup/fallback must be fixed before the uncertain action is observed;
- safe-exploration claims depend on explicit assumptions and cannot be inferred
  from ensemble agreement alone.

Not transferable to FMA V3.4:

- the paper's safety result, because this benchmark has neither the required
  calibrated Gaussian-process confidence sets nor a verified terminal safe set.

### Runtime assurance

Hobbs, K. L. et al. (2023), *Runtime Assurance for Safety-Critical Systems: An
Introduction to Safety Filtering Approaches for Complex Control Systems*, IEEE
Control Systems 43(2). DOI:
[10.1109/MCS.2023.3234380](https://doi.org/10.1109/MCS.2023.3234380).
Preprint: [arXiv:2110.03506](https://arxiv.org/abs/2110.03506).

Transferable architecture:

- keep the unverified high-performance controller separate from a runtime
  authority that can filter its output;
- make switching and fallback decisions explicit and auditable;
- evaluate the assurance layer independently from the primary controller.

Not transferable to FMA V3.4:

- the review is design guidance, not validation evidence for this implementation;
- calling a component `runtime assurance` does not establish safety.

## Frozen first-principles protocol

### Single component delta

- Baseline: the V3.3.2 paired-advantage proposer executes its selected third
  experiment as one uninterrupted six-segment action.
- Candidate: the identical proposer, anchors, trust decision, proposed action,
  observation noise schedule, estimator, and target evaluator are retained. Only
  the third experiment is executed one segment at a time through the Reality
  Adapter.

### Public calibration and online rule

1. Fit two leave-one-anchor-out models from the public pilot plus one shared
   anchor.
2. Predict every segment of the held-out anchor from that segment's observed
   starting state.
3. Freeze the case-local threshold as the maximum of the twelve resulting
   segment trajectory NRMSE values. This is a conservative empirical threshold,
   not a calibrated confidence bound.
4. Fit the online prediction model from the public pilot plus both anchors.
5. After each active segment, compare its observation with the one-segment
   prediction. If mismatch exceeds the frozen threshold, all later authority is
   reduced to zero input. Re-escalation is forbidden.
6. If adding a zero-input segment would exceed the prefrozen switch-count cap,
   terminate instead. Any observed state-envelope violation also terminates.
7. If the frozen acquisition gate offers no admissible shared anchor or third
   action, both arms abstain with the same typed reason. The Harness does not
   substitute a hidden action or relax the gate, and the case is excluded from
   paired performance estimation.

### Exposure accounting

The Harness records and hashes:

- elapsed intervention time;
- input energy;
- peak amplitude;
- switch count;
- executed segment count;
- maximum hidden clean-state envelope ratio (synthetic adjudicator only);
- interruption/termination reason.

For every case, candidate exposure must be component-wise no greater than the
unguarded baseline. Equal experiment counts are not accepted as exposure parity.

### Fresh evaluation gates

The archived V3.3.2 failure pack is development evidence only. Formal V3.4
adjudication uses new seeds frozen before private world-pack generation and
requires all of the following:

- proposal, target, anchor, trust-decision, and noise-schedule parity;
- paired abstention parity for data-quality and no-admissible-action cases;
- complete segment and exposure receipts;
- at least one exercised interruption;
- candidate exposure dominated by baseline exposure in every case;
- zero synthetic hidden-state envelope violations;
- paired macro target-loss improvement with nonnegative 95% bootstrap lower bound;
- no mechanism mean regression worse than 0.02;
- material negative-transfer rate at most 0.10;
- no qualification, router evolution, confirmation, or real-world authorization.

Passing these gates would qualify only the interaction-layer candidate for a
later acquisition retest. Failing them refutes this particular online mismatch
rule; it does not refute interruptible reality interfaces in general.
