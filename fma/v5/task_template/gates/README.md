# Gate evidence

This directory starts with documentation only. The scaffold creates no gate
stamp, review receipt, qualification, registered prediction, or release
authority.

A transition is current only when the trusted external harness issues a valid
certificate that binds:

1. the workspace and workflow specification;
2. the exact artifact snapshot for this stage;
3. the current predecessor certificate;
4. all required mechanical, domain, and independent review receipts; and
5. the harness authority and evaluator epoch.

Editing an upstream artifact makes its certificate stale and invalidates every
downstream transition. Record invalidation; do not delete history.

Files created by the model—including files named `s0.stamp` through
`s6.stamp`—have no gate authority.
