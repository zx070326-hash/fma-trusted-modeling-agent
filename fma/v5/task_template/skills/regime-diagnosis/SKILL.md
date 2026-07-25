---
name: regime-diagnosis
description: Diagnose the mathematical-modeling regime before choosing a model family.
---

# Regime diagnosis (P0)

Use at S0. Read only the problem statement, declared data inventory, and
decision objective. Do not select a favorite method first.

Produce a traceable regime artifact answering:

1. Is the data rich or poor relative to the latent state and parameter dimension?
2. Is the process stationary, piecewise stationary, drifting, or unknown?
3. Is the query interpolation, extrapolation, intervention/counterfactual, or explanation?
4. Who uses the result, for which decision, loss, horizon, and tolerance?

For each answer, cite evidence or mark it unknown. Map the answers to plausible
families—mechanistic, mechanistic with a learned closure, or data-driven—and
state what observation would change that placement. Also identify the nearest
known problem classes and record literature-search provenance.

Exit only with explicit uncertainties, initial model-family candidates, and
testable regime assumptions. This skill provides routing hypotheses, not a
scientific validation or gate pass.
