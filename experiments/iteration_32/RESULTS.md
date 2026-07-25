# Iteration 32 results

Iteration 32 closes the locally implementable protocol gaps found in Iteration
31. It does not contain a new scientific experiment.

The public forecast object is now one coherent object: chronological
development selection, the locked selected family, the all-public-data final
refit, every requested forecast horizon, and the exact registered prediction
are content-addressed together. A failure at any target horizon rejects the
whole public bundle.

The private path is now executable on another host. Custody, worker,
host-management, and promotion roles use distinct Ed25519 signatures. The
coordinator needs only pinned public keys. The external worker returns one
aggregate score; it cannot grant qualification.

The public result is bound after a current authenticated V5 S4 gate. If an S4
artifact changes, the gate becomes stale and the I32 graph binding no longer
verifies.

Validation evidence:

- Ruff source and test lint: passed;
- focused V5.3 closure tests: 8 passed, 0 failed;
- directed V5/V5.1/V5.2/V5.3 regression: 73 passed, 0 failed;
- complete repository suite: 342 passed, 0 failed, 0 skipped in 2313.3 seconds;
- exact implementation hashes: `SOURCE_MANIFEST.json`.

Current conclusion:

- local implementation and fixture protocol mechanics: implemented;
- new real unseen ODE campaign: `NOT_RUN`;
- external private evaluation and promotion: `NOT_RUN`;
- scientific qualification: false;
- real-world action authorization: false.

## Blinded-context shadow campaign

A two-Chat, same-host, cryptographically committed shadow campaign was later
run under campaign ID `i32-shadow-177747afada8fc62a6ed`. The custodian released
28 public observations and retained four future targets in an encrypted
capsule outside the modeling workspace. The modeler compared thirteen
candidates, rejected autonomous scalar ODE sufficiency, performed recovery,
and froze one four-horizon submission.

The recovered five-point robust local-level median improved public rolling
normalized MAE from `0.536501` for persistence to `0.439013`. In the only
private aggregate evaluation it passed the absolute normalized RMSE and MAE
limits, but its normalized MAE of `0.555509641873` did not beat the frozen
persistence result of `0.334022038567`. The outcome is therefore
`SHADOW_REJECTED`.

The evaluation budget is consumed and this task must not be retuned or
rescored. It remains same-host shadow evidence, not external qualification.
The signed receipt, immutable submission, and final result are under
`campaigns/i32-shadow-177747afada8fc62a6ed/`.
