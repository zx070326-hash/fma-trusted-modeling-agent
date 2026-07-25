from __future__ import annotations

import json
import sys

import numpy
import scipy

from .evidence import EvidenceGraph
from .schemas import (
    ArtifactRef,
    CompilerCertificate,
    OptimizationModelIR,
    ProblemContract,
    PromotionDecision,
    ReproductionReport,
    SolutionArtifact,
    ValidationVector,
)
from .storage import RunStore
from .validation import REQUIRED_HARD_CHECKS, TOLERANCE, attach_reproduction, validate_candidate


REQUIRED_EVIDENCE_KINDS = {
    "contract",
    "model_ir",
    "compiler_certificate",
    "solution",
    "validation",
    "reproduction",
    "environment",
}


class PromotionEngine:
    """Code-only gate that derives its verdict from stored evidence, not caller booleans."""

    policy_version = "optimization_promotion_v1"

    def decide(
        self,
        graph: EvidenceGraph,
        store: RunStore,
        claim_node_id: str,
    ) -> PromotionDecision:
        if graph.node_status(claim_node_id) == "revoked":
            raise RuntimeError("revoked claims require new lineage; they cannot be re-promoted")

        snapshot = graph.snapshot()
        snapshot_reference = store.put_artifact("promotion_input_snapshot", snapshot)
        support = graph.supporting_nodes(claim_node_id)
        grouped: dict[str, list[dict[str, object]]] = {}
        for node in support:
            grouped.setdefault(str(node["kind"]), []).append(node)

        loaded: dict[str, object] = {}
        evidence_results: dict[str, bool] = {}
        invalid_evidence: list[str] = []
        for kind in sorted(REQUIRED_EVIDENCE_KINDS):
            nodes = grouped.get(kind, [])
            if len(nodes) != 1 or nodes[0]["status"] != "current":
                evidence_results[f"evidence:{kind}"] = False
                if len(nodes) > 1:
                    invalid_evidence.append(f"ambiguous {kind} evidence")
                continue
            node = nodes[0]
            try:
                metadata = json.loads(str(node["metadata_json"]))
                reference = ArtifactRef(
                    kind=kind,
                    sha256=str(node["artifact_hash"]),
                    relative_path=str(metadata["relative_path"]),
                )
                loaded[kind] = store.load_artifact(reference)
                evidence_results[f"evidence:{kind}"] = True
            except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                evidence_results[f"evidence:{kind}"] = False
                invalid_evidence.append(f"invalid {kind} evidence: {exc}")

        candidate_id = "unknown_candidate"
        effective_validation: ValidationVector | None = None
        validation_artifact_match = False
        reproduction_binding = False
        environment_binding = False
        if all(evidence_results.values()):
            try:
                contract = ProblemContract.model_validate(loaded["contract"])
                ir = OptimizationModelIR.model_validate(loaded["model_ir"])
                certificate = CompilerCertificate.model_validate(
                    loaded["compiler_certificate"]
                )
                solution = SolutionArtifact.model_validate(loaded["solution"])
                stored_validation = ValidationVector.model_validate(loaded["validation"])
                reproduction = ReproductionReport.model_validate(loaded["reproduction"])
                environment = loaded["environment"]
                if not isinstance(environment, dict):
                    raise TypeError("environment evidence must be an object")
                candidate_id = ir.candidate_id

                reproduction_binding = self._reproduction_is_bound(
                    reproduction, certificate, solution
                )
                recomputed = validate_candidate(contract, ir, certificate, solution)
                effective_validation = attach_reproduction(
                    recomputed,
                    passed=reproduction_binding,
                    detail=(
                        reproduction.detail
                        if reproduction_binding
                        else "stored reproduction receipt failed cross-artifact binding"
                    ),
                )
                validation_artifact_match = stored_validation == effective_validation
                environment_binding = (
                    environment.get("contract_hash") == contract.frozen_hash
                    and environment.get("ir_hash") == ir.ir_hash
                    and environment.get("compiler") == certificate.compiler_version
                    and environment.get("verifier") == effective_validation.verifier_version
                    and environment.get("promotion_policy") == self.policy_version
                    and environment.get("python") == sys.version
                    and environment.get("numpy") == numpy.__version__
                    and environment.get("scipy") == scipy.__version__
                )
            except (KeyError, TypeError, ValueError) as exc:
                invalid_evidence.append(f"typed evidence reconstruction failed: {exc}")

        gate_results: dict[str, bool] = {
            **evidence_results,
            "binding:validation_artifact": validation_artifact_match,
            "binding:reproduction": reproduction_binding,
            "binding:environment": environment_binding,
        }
        failed_checks: list[str] = []
        unresolved_checks: list[str] = []
        for name in sorted(REQUIRED_HARD_CHECKS):
            record = effective_validation.checks.get(name) if effective_validation else None
            gate_results[f"check:{name}"] = record is not None and record.status == "pass"
            if record is None or record.status in {"not_run", "warning"}:
                unresolved_checks.append(name)
            elif record.status == "fail":
                failed_checks.append(name)

        missing_evidence = [
            key.removeprefix("evidence:")
            for key, passed in evidence_results.items()
            if not passed
        ]
        reasons: list[str] = []
        binding_failed = not (
            validation_artifact_match and reproduction_binding and environment_binding
        )
        if invalid_evidence or failed_checks or (not missing_evidence and binding_failed):
            status = "run_invalid"
            reasons.extend(invalid_evidence)
            if failed_checks:
                reasons.append(f"hard checks failed: {', '.join(failed_checks)}")
            if not missing_evidence and binding_failed:
                reasons.append("cross-artifact binding failed")
        elif missing_evidence or unresolved_checks:
            status = "needs_evidence"
            if missing_evidence:
                reasons.append(f"missing current evidence: {', '.join(missing_evidence)}")
            if unresolved_checks:
                reasons.append(
                    f"hard checks unresolved: {', '.join(unresolved_checks)}"
                )
        elif all(gate_results.values()):
            status = "validated"
            reasons.append("all pre-registered synthetic-oracle gates passed")
        else:
            status = "needs_evidence"
            reasons.append("one or more promotion gates are unresolved")

        decision = PromotionDecision(
            candidate_id=candidate_id,
            claim_node_id=claim_node_id,
            status=status,
            validation_scope="synthetic_oracle",
            gate_results=gate_results,
            reasons=reasons,
            evidence_snapshot_hash=str(snapshot["snapshot_hash"]),
            evidence_snapshot_artifact=snapshot_reference,
        )
        graph._set_claim_status_by_promotion(
            claim_node_id,
            status,
            policy_version=self.policy_version,
        )
        return decision

    @staticmethod
    def _reproduction_is_bound(
        reproduction: ReproductionReport,
        certificate: CompilerCertificate,
        solution: SolutionArtifact,
    ) -> bool:
        objective_match = (
            reproduction.reference_objective is not None
            and solution.objective_value is not None
            and abs(reproduction.reference_objective - solution.objective_value) <= TOLERANCE
        )
        replay_objective_match = (
            reproduction.replay_objective is not None
            and solution.objective_value is not None
            and abs(reproduction.replay_objective - solution.objective_value) <= TOLERANCE
        )
        return (
            reproduction.status == "pass"
            and reproduction.source_ir_hash == solution.source_ir_hash
            and reproduction.replay_matrix_hash == certificate.matrix_hash
            and reproduction.replay_execution_hash == certificate.execution_array_hash
            and objective_match
            and replay_objective_match
            and bool(reproduction.checks)
            and all(reproduction.checks.values())
        )
