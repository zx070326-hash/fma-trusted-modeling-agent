# Iteration 12 / V3.4.1: persistent online mismatch protocol

Frozen on: 2026-07-22

## Prior result

V3.4 formal report hash:
`865bc6b481f884c45a79a67ab6c868a90e8056e65df43ace11c72fec4f703ecf`.

V3.4 passed proposal, target, anchor, trust, noise, abstention, receipt,
exposure, synthetic state-envelope, negative-transfer-rate, and worst-case-loss
gates. It failed the paired macro lower-bound gate because the only interruption
slightly worsened one logistic case:

- baseline target loss: `0.029645347907229003`;
- single-exceedance guard loss: `0.02976178039093795`;
- guard improvement: `-0.00011643248370894754`;
- formal macro improvement: `-0.0000032342356585818763`;
- 95% bootstrap interval: `[-0.000009702706975745629, 0.0]`.

The V3.4 candidate is therefore refuted. No V3.4 qualification or confirmation
exists.

## Development-only mechanism check

The interrupted case had segment mismatch values
`[0.287418, 0.479786, 0.044329, 0.165207, 0.094408, 0.060795]` against a
case-local frozen threshold `0.1444703449386849`. The single-exceedance rule
switched after segment one.

One predeclared structural alternative was evaluated on that archived failure
case only: require two consecutive threshold exceedances. It would switch after
segment two and produced diagnostic target loss `0.019059360947456392`, an
improvement of `0.01058598695977261` over the unguarded action.

This counterfactual is hypothesis-generation evidence. It is not included in
formal V3.4.1 adjudication and cannot establish general benefit.

## Frozen single-component delta

- Baseline: unchanged V3.3.2 paired-advantage proposer followed by an
  uninterrupted selected third action.
- V3.4.1 candidate: identical proposer, anchors, trust decision, selected
  action, estimator, common observation-noise schedule, exposure envelope, and
  evaluator.
- The only change from V3.4 is the online trigger:
  - V3.4: one segment with `NRMSE > case_threshold` reduces authority;
  - V3.4.1: two consecutive segments with `NRMSE > case_threshold` are required.
- A non-exceeding segment resets the consecutive count to zero.
- Once confirmed, the candidate still switches monotonically to zero input or
  terminates if the switch cap would be exceeded. Re-escalation is forbidden.

## Formal evidence boundary

V3.4.1 uses a new frozen sixteen-seed set, hence 64 new synthetic cases. It does
not reuse V3.4 private outcomes. The V3.4.1 method document, policies, statistical
gates, and seeds are content-bound before private world-pack generation.

The V3.4 gates remain unchanged:

- proposal/target/anchor parity;
- trust-decision, common-noise, and abstention parity;
- complete segment and exposure receipts;
- at least one exercised interruption;
- candidate exposure component-wise dominated by baseline;
- zero synthetic hidden-state envelope violations;
- nonnegative paired macro 95% bootstrap lower bound;
- no mechanism mean regression worse than `0.02`;
- material negative-transfer rate at most `0.10`;
- worst-case target-loss non-regression.

Passing qualifies only this interaction-layer candidate for a later acquisition
retest. Router evolution, whole-agent qualification, confirmation, real-world
execution, and formal safety claims remain forbidden.
