from __future__ import annotations

from pathlib import Path

from .schemas import (
    ArtifactRef,
    OptimizationModelIR,
    ProblemContract,
    PromotionDecision,
    ReproductionReport,
    SolutionArtifact,
    ValidationVector,
)


def write_dossier(
    path: str | Path,
    *,
    run_id: str,
    contract: ProblemContract,
    ir: OptimizationModelIR,
    solution: SolutionArtifact,
    validation: ValidationVector,
    reproduction: ReproductionReport,
    decision: PromotionDecision,
    artifacts: dict[str, ArtifactRef],
    event_chain_verified: bool,
) -> Path:
    destination = Path(path).resolve()
    check_rows = "\n".join(
        f"| `{name}` | `{record.status}` | {record.detail} |"
        for name, record in sorted(validation.checks.items())
    )
    artifact_rows = "\n".join(
        f"| `{kind}` | `{reference.sha256}` | `{reference.relative_path}` |"
        for kind, reference in sorted(artifacts.items())
    )
    values = ", ".join(f"{name}={value:g}" for name, value in sorted(solution.values.items()))
    warnings = "\n".join(f"- {warning}" for warning in validation.warnings) or "- None"
    violations = "\n".join(f"- {item}" for item in validation.violations) or "- None"
    assumptions = [
        clause.statement for clause in contract.clauses if clause.kind.value == "assumption"
    ]
    assumption_lines = "\n".join(f"- {item}" for item in assumptions) or "- None declared"
    content = f"""# Model Dossier: {ir.candidate_id}

## Verdict

- Run: `{run_id}`
- Promotion: `{decision.status}`
- Validation scope: `{decision.validation_scope}`
- Candidate claim status at issue time: `{decision.status}`
- Event hash chain verified: `{str(event_chain_verified).lower()}`

This verdict is limited to the sealed optimization IR, the frozen executable microcase tests, and the exact bounded-integer oracle. Clause-ID coverage is not a proof that natural-language requirements were translated faithfully. It does **not** validate real-world semantics or establish autonomous frontier-problem solving.

## Frozen contract and model

- Contract: `{contract.contract_id}` version `{contract.version}`
- Contract hash: `{contract.frozen_hash}`
- Question: {contract.question}
- System boundary: {contract.system_boundary}
- Decision horizon: {contract.decision_horizon}
- IR hash: `{ir.ir_hash}`
- Skeleton: `{ir.skeleton_id}`
- Lineage root: `{ir.lineage.root_kind}`

## Solver result

- Status: `{solution.solver_status}`
- Values: {values or "None"}
- Objective: `{solution.objective_value}`
- Solver: `{solution.solver}`

## Validation vector

| Check | Status | Evidence |
|---|---|---|
{check_rows}

Fresh-process replay: `{reproduction.status}` — {reproduction.detail}

### Violations

{violations}

### Warnings and unresolved evidence

{warnings}

### Contract assumptions

{assumption_lines}

## Evidence artifacts

| Kind | SHA-256 | Run-relative path |
|---|---|---|
{artifact_rows}

## Responsibility boundary

`validated@synthetic_oracle` means that the executable result matched the sealed IR, passed the frozen executable microcases and independent source-level checks, matched an exact enumeration oracle, and replayed in a fresh process. The solve itself ran locally in the controller process, not in a security sandbox. Any natural-language semantic acceptance, real-world data acquisition, external experiment, deployment, or consequential decision still requires a separate authorization and validation policy.
"""
    destination.write_text(content, encoding="utf-8")
    return destination
