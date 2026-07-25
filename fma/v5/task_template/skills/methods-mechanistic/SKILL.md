---
name: methods-mechanistic
description: Construct and compare mechanistic model skeletons without treating templates as truth.
---

# Mechanistic methods (P0)

Use after S0 suggests a mechanistic skeleton is defensible. Start from states,
flows, constraints, symmetries, conservation or balance relations, causal
ordering, and the downstream observable—not from a named equation.

For every candidate:

- define states, parameters, controls, observables, units, initial/boundary conditions;
- derive the mathematical form and mark every closure or constitutive assumption;
- check dimensional consistency, admissible bounds, limiting cases, and
  identifiability in principle;
- state required data, numerical method, expected failure modes, and an
  abandonment criterion;
- propose the smallest experiment that distinguishes it from the other candidates.

Generate at least one simple baseline and preserve competing explanations until
evidence separates them. A familiar ODE, PDE, stochastic process, optimization,
network, or queueing form is only a candidate skeleton.

This minimal P0 skill is not the proposed 98-template HMML catalog and does not
claim to reproduce MM-Agent. Domain equations and closures still require
source-backed derivation and independent verification.
