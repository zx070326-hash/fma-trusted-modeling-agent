from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .controller import ModelingAgent, StaticExplorer, revoke_run_evidence
from .examples import (
    dropped_constraint_certificate,
    resource_allocation_contract,
    resource_allocation_ir,
    submitted_solution,
)
from .optimization import compile_optimization, solve_compiled


def run_demo(output_root: str | Path) -> dict[str, object]:
    experiment_id = (
        datetime.now(timezone.utc).strftime("experiment-%Y%m%dT%H%M%SZ-")
        + uuid4().hex[:6]
    )
    experiment_directory = Path(output_root).resolve() / experiment_id
    experiment_directory.mkdir(parents=True, exist_ok=False)
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)

    good = ModelingAgent(experiment_directory / "good").run(
        contract, StaticExplorer([ir])
    )[0]

    tampered = submitted_solution(
        ir,
        values={"x": 10.0, "y": 10.0},
        objective_value=80.0,
        message="syntactically valid but infeasible claimed optimum",
    )
    tampered_outcome = ModelingAgent(experiment_directory / "tampered_solution").assess_candidate(
        contract,
        ir,
        submitted_solution=tampered,
    )

    dropped_certificate = dropped_constraint_certificate(ir)
    dropped_solution = submitted_solution(
        ir,
        values={"x": 0.0, "y": 8.0},
        objective_value=40.0,
        matrix_hash=dropped_certificate.matrix_hash,
        execution_hash=dropped_certificate.execution_array_hash,
        message="result from a compiler that omitted resource_b",
    )
    dropped_outcome = ModelingAgent(experiment_directory / "dropped_constraint").assess_candidate(
        contract,
        ir,
        submitted_solution=dropped_solution,
        submitted_certificate=dropped_certificate,
    )

    sign_flip_solution = submitted_solution(
        ir,
        values={"x": 0.0, "y": 0.0},
        objective_value=0.0,
        message="feasible result from an objective-direction sign flip",
    )
    sign_flip_outcome = ModelingAgent(experiment_directory / "objective_sign_flip").assess_candidate(
        contract,
        ir,
        submitted_solution=sign_flip_solution,
    )

    mutated_compilation = compile_optimization(ir)
    mutated_compilation.matrix[1, :] = 0.0
    try:
        solve_compiled(mutated_compilation)
        certified_array_mutation_blocked = False
    except RuntimeError as exc:
        certified_array_mutation_blocked = "numeric arrays changed" in str(exc)

    revocable = ModelingAgent(experiment_directory / "revocation").run(
        contract, StaticExplorer([ir])
    )[0]
    revocation_receipt = revoke_run_evidence(
        revocable,
        "reproduction",
        "fault injection: reproduction evidence withdrawn",
    )
    affected = revocation_receipt["affected_node_ids"]
    post_revocation_status = revocation_receipt["effective_claim_status"]

    checks = {
        "good_exact_solution": (
            good.decision.status == "validated"
            and abs(good.solution.values["x"] - 3.0) <= 1e-7
            and abs(good.solution.values["y"] - 2.0) <= 1e-7
            and abs((good.solution.objective_value or 0.0) - 19.0) <= 1e-7
        ),
        "good_all_hard_gates": all(good.decision.gate_results.values()),
        "tampered_solution_rejected": (
            tampered_outcome.decision.status == "run_invalid"
            and tampered_outcome.validation.checks["feasibility"].status == "fail"
        ),
        "dropped_constraint_rejected": (
            dropped_outcome.decision.status == "run_invalid"
            and dropped_outcome.validation.checks["compiler_fidelity"].status == "fail"
        ),
        "objective_sign_flip_rejected": (
            sign_flip_outcome.decision.status == "run_invalid"
            and sign_flip_outcome.validation.checks["optimality_oracle"].status == "fail"
        ),
        "certified_array_mutation_blocked": certified_array_mutation_blocked,
        "revocation_cascades": (
            revocable.decision.status == "validated"
            and post_revocation_status == "revoked"
            and revocable.claim_node_id in affected
        ),
    }
    summary: dict[str, object] = {
        "experiment_schema": "1.0",
        "experiment_id": experiment_id,
        "experiment_directory": str(experiment_directory),
        "validation_scope": "synthetic_oracle",
        "chain_established": all(checks.values()),
        "checks": checks,
        "runs": {
            "good": good.model_dump(mode="json"),
            "tampered_solution": tampered_outcome.model_dump(mode="json"),
            "dropped_constraint": dropped_outcome.model_dump(mode="json"),
            "objective_sign_flip": sign_flip_outcome.model_dump(mode="json"),
            "revocation": {
                "initial": revocable.model_dump(mode="json"),
                "post_revocation_claim_status": post_revocation_status,
                "affected_node_ids": affected,
                "revocation_receipt": revocation_receipt,
            },
        },
        "claim_boundary": (
            "This proves the trusted execution, verification, promotion, and revocation "
            "mechanism on one hand-authored bounded MILP. It does not prove autonomous "
            "problem discovery, real-world semantic validity, or frontier-modeling ability."
        ),
    }
    summary_path = experiment_directory / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = experiment_directory / "experiment_report.md"
    report_path.write_text(
        _markdown_report(summary, summary_path),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    return summary


def _markdown_report(summary: dict[str, object], summary_path: Path) -> str:
    checks = summary["checks"]
    assert isinstance(checks, dict)
    rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    )
    runs = summary["runs"]
    assert isinstance(runs, dict)
    good = runs["good"]
    tampered = runs["tampered_solution"]
    dropped = runs["dropped_constraint"]
    sign_flip = runs["objective_sign_flip"]
    revocation = runs["revocation"]
    return f"""# FMA V1 最小可信链实验

## 结论

`chain_established = {str(summary['chain_established']).lower()}`。

在一个手写、有界整数线性规划微例上，正常候选得到 `(x, y) = (3, 2)`、目标值 `19`，完成契约哈希绑定、冻结可执行微例、确定性编译、SciPy 求解、原始 IR 独立解释、穷举 oracle、全新进程重放和代码化 Promotion。解篡改、编译证据丢约束、认证后数组变更、目标方向反转均被拒绝；撤回复现证据后，已晋级结论沿证据 DAG 自动变为 `revoked` 并产生撤销回执。

## 机器可检验结果

| Assertion | Result |
|---|---|
{rows}

## 各实验状态

| Case | Promotion / final status | Intended fault |
|---|---|---|
| good | `{good['decision']['status']}@{good['decision']['validation_scope']}` | none |
| tampered_solution | `{tampered['decision']['status']}` | infeasible claimed solution |
| dropped_constraint | `{dropped['decision']['status']}` | compiler certificate omitted a hard constraint |
| certified_array_mutation | `blocked_before_solver` | certified numeric matrix changed in memory |
| objective_sign_flip | `{sign_flip['decision']['status']}` | feasible but non-optimal solution |
| revocation | `{revocation['post_revocation_claim_status']}` | reproduction evidence withdrawn after promotion |

完整机器输出：`{summary_path.name}`。每个 case 的目录内含内容寻址工件、哈希链事件、SQLite 证据图和 dossier。

## 严格边界

{summary['claim_boundary']}

因此，这次实验只证明“可信执行与晋级 harness 链路成立”，尚未证明“建模智能”或“能自主解决前沿真实问题”。主求解仍在控制器同一进程内；泄漏审计明确为 `not_run`，不能把本实验称为安全沙箱或隐藏集验证。
"""
