from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Protocol

import numpy
import scipy

from .dossier import write_dossier
from .evidence import EvidenceGraph
from .optimization import compile_optimization, solve_compiled
from .promotion import PromotionEngine
from .schemas import (
    ArtifactRef,
    CandidateRunOutcome,
    CompilerCertificate,
    OptimizationModelIR,
    ProblemContract,
    ReproductionReport,
    SolutionArtifact,
)
from .storage import RunStore
from .validation import (
    REQUIRED_HARD_CHECKS,
    TOLERANCE,
    attach_reproduction,
    preflight_candidate,
    validate_candidate,
)


class Explorer(Protocol):
    def propose(self, contract: ProblemContract) -> list[OptimizationModelIR]: ...


class StaticExplorer:
    """A deterministic stand-in that isolates the trusted harness from LLM quality."""

    def __init__(self, candidates: list[OptimizationModelIR]) -> None:
        self.candidates = candidates

    def propose(self, contract: ProblemContract) -> list[OptimizationModelIR]:
        return list(self.candidates)


class CandidatePreflightError(ValueError):
    pass


def _environment_manifest(contract: ProblemContract, ir: OptimizationModelIR) -> dict[str, object]:
    return {
        "manifest_schema": "1.0",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "compiler": "scipy_milp_v1",
        "verifier": "optimization_verifier_v1",
        "promotion_policy": "optimization_promotion_v1",
        "contract_hash": contract.frozen_hash,
        "ir_hash": ir.ir_hash,
        "execution_isolation": "none_same_process",
        "network_policy_enforced": False,
        "network_observation": "not_instrumented",
        "hidden_evaluation_channel_present": False,
    }


def _fresh_process_replay(
    store: RunStore,
    contract: ProblemContract,
    ir: OptimizationModelIR,
    certificate: CompilerCertificate,
    reference_solution: SolutionArtifact,
) -> tuple[ReproductionReport, dict[str, object]]:
    bundle = {
        "contract": contract.model_dump(mode="json"),
        "model_ir": ir.model_dump(mode="json"),
        "reference_certificate": certificate.model_dump(mode="json"),
        "reference_solution": reference_solution.model_dump(mode="json"),
    }
    input_path = store.run_directory / "replay_input.json"
    output_path = store.run_directory / "replay_output.json"
    input_path.write_text(
        json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "fma.replay", str(input_path), str(output_path)],
            cwd=package_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report = ReproductionReport(
            status="fail",
            source_ir_hash=ir.ir_hash,
            reference_objective=reference_solution.objective_value,
            checks={"fresh_process": False},
            detail=f"fresh-process replay could not run: {exc}",
        )
        return report, {"error": str(exc)}

    if completed.returncode != 0 or not output_path.exists():
        detail = (
            f"fresh-process replay exited {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
        report = ReproductionReport(
            status="fail",
            source_ir_hash=ir.ir_hash,
            reference_objective=reference_solution.objective_value,
            checks={"fresh_process": False},
            detail=detail,
        )
        return report, {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    replay = json.loads(output_path.read_text(encoding="utf-8"))
    replay_certificate = CompilerCertificate.model_validate(replay["compiler_certificate"])
    replay_solution = SolutionArtifact.model_validate(replay["solution"])
    same_keys = set(reference_solution.values) == set(replay_solution.values)
    if same_keys and reference_solution.values:
        value_delta = max(
            abs(reference_solution.values[name] - replay_solution.values[name])
            for name in reference_solution.values
        )
    elif same_keys:
        value_delta = 0.0
    else:
        value_delta = None
    objective_match = (
        reference_solution.objective_value is not None
        and replay_solution.objective_value is not None
        and abs(reference_solution.objective_value - replay_solution.objective_value) <= TOLERANCE
    )
    verifier_checks = replay.get("validation_checks", {})
    replay_required_checks = REQUIRED_HARD_CHECKS - {
        "reproducibility",
        "optimality_oracle",
    }
    replay_verifier_pass = all(
        verifier_checks.get(name) == "pass"
        for name in replay_required_checks
    )
    checks = {
        "fresh_process": True,
        "compiler_hash_match": replay_certificate.matrix_hash == certificate.matrix_hash,
        "execution_array_hash_match": (
            replay_certificate.execution_array_hash == certificate.execution_array_hash
        ),
        "objective_match": objective_match,
        "replay_verifier_pass": replay_verifier_pass,
    }
    passed = all(checks.values())
    detail = (
        "fresh Python process rebuilt, solved, and independently rechecked the same result"
        if passed
        else "fresh-process replay disagreed with one or more submitted artifacts"
    )
    report = ReproductionReport(
        status="pass" if passed else "fail",
        source_ir_hash=ir.ir_hash,
        replay_matrix_hash=replay_certificate.matrix_hash,
        replay_execution_hash=replay_certificate.execution_array_hash,
        reference_objective=reference_solution.objective_value,
        replay_objective=replay_solution.objective_value,
        value_delta_max=value_delta,
        checks=checks,
        detail=detail,
    )
    return report, replay


class ModelingAgent:
    """Single-explorer loop with a deterministic, non-LLM trusted control plane."""

    def __init__(self, output_root: str | Path, *, max_candidates: int = 8) -> None:
        self.output_root = Path(output_root).resolve()
        self.max_candidates = max_candidates
        self.promotion = PromotionEngine()

    def run(
        self,
        contract: ProblemContract,
        explorer: Explorer,
    ) -> list[CandidateRunOutcome]:
        contract.assert_frozen()
        candidates = explorer.propose(contract)
        if len(candidates) > self.max_candidates:
            candidates = candidates[: self.max_candidates]
        outcomes: list[CandidateRunOutcome] = []
        for candidate in candidates:
            try:
                outcomes.append(self.assess_candidate(contract, candidate))
            except CandidatePreflightError:
                continue
        return outcomes

    def assess_candidate(
        self,
        contract: ProblemContract,
        ir: OptimizationModelIR,
        *,
        submitted_solution: SolutionArtifact | None = None,
        submitted_certificate: CompilerCertificate | None = None,
        run_id: str | None = None,
    ) -> CandidateRunOutcome:
        contract.assert_frozen()
        ir.assert_sealed()
        preflight_failures = preflight_candidate(contract, ir)
        if preflight_failures:
            raise CandidatePreflightError(
                "candidate rejected before compilation: " + "; ".join(preflight_failures)
            )
        store = RunStore(self.output_root, run_id=run_id)
        graph = EvidenceGraph(store.run_directory / "evidence.sqlite3")
        try:
            store.emit("contract_frozen", {"contract_hash": contract.frozen_hash})
            artifacts: dict[str, ArtifactRef] = {}
            evidence_nodes: dict[str, str] = {}

            def commit_evidence(kind: str, payload: object) -> None:
                reference = store.put_artifact(kind, payload)
                artifacts[kind] = reference
                node_id = graph.add_node(
                    kind,
                    artifact_hash=reference.sha256,
                    metadata={"relative_path": reference.relative_path},
                )
                evidence_nodes[kind] = node_id

            commit_evidence("contract", contract)
            commit_evidence("model_ir", ir)
            claim_node_id = graph.add_node(
                "claim",
                status="proposed",
                metadata={
                    "candidate_id": ir.candidate_id,
                    "statement": (
                        "submitted solution is optimal for the sealed IR and passes the "
                        "frozen executable microcase tests"
                    ),
                    "scope": "synthetic_oracle",
                },
            )
            graph.add_edge(evidence_nodes["contract"], claim_node_id)
            graph.add_edge(evidence_nodes["model_ir"], claim_node_id)
            store.emit("candidate_proposed", {"candidate_id": ir.candidate_id})

            compiled = compile_optimization(ir)
            certificate = submitted_certificate or compiled.certificate
            commit_evidence("compiler_certificate", certificate)
            graph.add_edge(evidence_nodes["compiler_certificate"], claim_node_id)
            store.emit(
                "candidate_compiled",
                {
                    "candidate_id": ir.candidate_id,
                    "certificate_source": "submitted" if submitted_certificate else "harness",
                },
            )

            solution = submitted_solution or solve_compiled(compiled)
            commit_evidence("solution", solution)
            graph.add_edge(evidence_nodes["solution"], claim_node_id)
            store.emit(
                "candidate_executed",
                {
                    "candidate_id": ir.candidate_id,
                    "solution_source": "submitted" if submitted_solution else "harness",
                    "solver_status": solution.solver_status,
                },
            )

            base_validation = validate_candidate(contract, ir, certificate, solution)
            reproduction, replay_payload = _fresh_process_replay(
                store, contract, ir, certificate, solution
            )
            validation = attach_reproduction(
                base_validation,
                passed=reproduction.status == "pass",
                detail=reproduction.detail,
            )
            commit_evidence("validation", validation)
            graph.add_edge(evidence_nodes["validation"], claim_node_id)
            commit_evidence("reproduction", reproduction)
            graph.add_edge(evidence_nodes["reproduction"], claim_node_id)
            artifacts["replay_output"] = store.put_artifact("replay_output", replay_payload)
            environment = _environment_manifest(contract, ir)
            commit_evidence("environment", environment)
            graph.add_edge(evidence_nodes["environment"], claim_node_id)
            store.emit(
                "candidate_verified",
                {
                    "candidate_id": ir.candidate_id,
                    "hard_gate_statuses": {
                        name: validation.checks[name].status
                        for name in sorted(REQUIRED_HARD_CHECKS)
                    },
                },
            )

            decision = self.promotion.decide(
                graph,
                store,
                claim_node_id,
            )
            artifacts["promotion_input_snapshot"] = decision.evidence_snapshot_artifact
            artifacts["promotion"] = store.put_artifact("promotion", decision)
            decision_node = graph.add_node(
                "promotion",
                artifact_hash=artifacts["promotion"].sha256,
                metadata={"relative_path": artifacts["promotion"].relative_path},
            )
            graph.add_edge(claim_node_id, decision_node, relation="evaluated_by")
            store.emit("promotion_decided", decision)

            event_chain_verified = store.verify_event_chain()
            dossier_path = write_dossier(
                store.run_directory / "dossier.md",
                run_id=store.run_id,
                contract=contract,
                ir=ir,
                solution=solution,
                validation=validation,
                reproduction=reproduction,
                decision=decision,
                artifacts=artifacts,
                event_chain_verified=event_chain_verified,
            )
            artifacts["dossier"] = store.put_artifact(
                "dossier", {"markdown": dossier_path.read_text(encoding="utf-8")}
            )
            dossier_node = graph.add_node(
                "dossier",
                artifact_hash=artifacts["dossier"].sha256,
                metadata={"relative_path": artifacts["dossier"].relative_path},
            )
            graph.add_edge(claim_node_id, dossier_node, relation="reported_by")
            final_snapshot = graph.snapshot()
            store.put_artifact("evidence_snapshot", final_snapshot)
            store.emit(
                "run_completed",
                {
                    "candidate_id": ir.candidate_id,
                    "claim_status": graph.node_status(claim_node_id),
                },
            )
            if not store.verify_event_chain():
                raise RuntimeError("event hash chain failed after final commit")
            return CandidateRunOutcome(
                run_id=store.run_id,
                candidate_id=ir.candidate_id,
                run_directory=str(store.run_directory),
                claim_node_id=claim_node_id,
                claim_status=graph.node_status(claim_node_id),
                decision=decision,
                solution=solution,
                validation=validation,
                reproduction=reproduction,
                evidence_node_ids=evidence_nodes,
                dossier_path=str(dossier_path),
            )
        finally:
            graph.close()


def revoke_run_evidence(
    outcome: CandidateRunOutcome,
    evidence_kind: str,
    reason: str,
) -> dict[str, object]:
    """Append a revocation receipt and a superseding graph snapshot to an existing run."""
    if evidence_kind not in outcome.evidence_node_ids:
        raise KeyError(evidence_kind)
    store = RunStore.open_existing(outcome.run_directory)
    with EvidenceGraph(store.run_directory / "evidence.sqlite3") as graph:
        affected = graph.revoke_node(outcome.evidence_node_ids[evidence_kind], reason)
        effective_status = graph.node_status(outcome.claim_node_id)
        snapshot = graph.snapshot()
    snapshot_reference = store.put_artifact("evidence_snapshot", snapshot)
    receipt = {
        "receipt_schema": "1.0",
        "run_id": outcome.run_id,
        "candidate_id": outcome.candidate_id,
        "revoked_evidence_kind": evidence_kind,
        "revoked_node_id": outcome.evidence_node_ids[evidence_kind],
        "reason": reason,
        "affected_node_ids": affected,
        "effective_claim_status": effective_status,
        "superseding_snapshot": snapshot_reference.model_dump(mode="json"),
        "supersedes_dossier": outcome.dossier_path,
    }
    receipt_reference = store.put_artifact("revocation_receipt", receipt)
    store.emit(
        "evidence_revoked",
        {
            "receipt": receipt_reference.model_dump(mode="json"),
            "effective_claim_status": effective_status,
        },
    )
    if not store.verify_event_chain():
        raise RuntimeError("event hash chain failed after revocation")
    return {
        **receipt,
        "receipt": receipt_reference.model_dump(mode="json"),
    }
