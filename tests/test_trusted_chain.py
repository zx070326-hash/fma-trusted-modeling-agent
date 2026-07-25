from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fma.demo import run_demo
from fma.controller import (
    CandidatePreflightError,
    ModelingAgent,
    StaticExplorer,
    revoke_run_evidence,
)
from fma.evidence import EvidenceGraph
from fma.examples import (
    dropped_constraint_certificate,
    resource_allocation_contract,
    resource_allocation_ir,
    submitted_solution,
)
from fma.optimization import compile_optimization, solve_compiled
from fma.promotion import PromotionEngine, REQUIRED_EVIDENCE_KINDS
from fma.schemas import OptimizationModelIR, ProblemContract
from fma.storage import RunStore
from fma.validation import (
    REQUIRED_HARD_CHECKS,
    attach_reproduction,
    preflight_candidate,
    validate_candidate,
)


def test_demo_establishes_the_full_synthetic_trust_chain(tmp_path):
    summary = run_demo(tmp_path)

    assert summary["chain_established"] is True
    assert all(summary["checks"].values())
    good = summary["runs"]["good"]
    assert good["solution"]["values"] == {"x": 3.0, "y": 2.0}
    assert good["solution"]["objective_value"] == pytest.approx(19.0)
    assert good["decision"]["status"] == "validated"
    assert good["decision"]["validation_scope"] == "synthetic_oracle"
    assert good["validation"]["metrics"]["oracle_assignments"] == 121
    assert good["validation"]["metrics"]["oracle_feasible_assignments"] == 14
    assert good["validation"]["metrics"]["oracle_objective"] == pytest.approx(19.0)
    assert summary["runs"]["tampered_solution"]["decision"]["status"] == "run_invalid"
    assert summary["runs"]["dropped_constraint"]["decision"]["status"] == "run_invalid"
    assert summary["runs"]["objective_sign_flip"]["decision"]["status"] == "run_invalid"
    assert summary["runs"]["revocation"]["post_revocation_claim_status"] == "revoked"


def test_compilation_is_deterministic_and_complete():
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)

    first = compile_optimization(ir).certificate
    second = compile_optimization(ir).certificate

    assert first == second
    assert [row["constraint_id"] for row in first.constraint_rows] == [
        "resource_a",
        "resource_b",
    ]
    assert first.objective_vector == [-3.0, -5.0]

    first_compiled = compile_optimization(ir)
    first_compiled.matrix[1, :] = 0.0
    with pytest.raises(RuntimeError, match="numeric arrays changed"):
        solve_compiled(first_compiled)


def test_independent_checker_detects_result_and_compiler_tampering():
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)
    compiled = compile_optimization(ir)

    wrong_objective = submitted_solution(
        ir,
        values={"x": 3.0, "y": 2.0},
        objective_value=20.0,
    )
    objective_validation = validate_candidate(
        contract, ir, compiled.certificate, wrong_objective
    )
    assert objective_validation.checks["objective_consistency"].status == "fail"

    dropped = dropped_constraint_certificate(ir)
    invalid_result = submitted_solution(
        ir,
        values={"x": 0.0, "y": 8.0},
        objective_value=40.0,
        matrix_hash=dropped.matrix_hash,
        execution_hash=dropped.execution_array_hash,
    )
    dropped_validation = validate_candidate(contract, ir, dropped, invalid_result)
    assert dropped_validation.checks["compiler_fidelity"].status == "fail"
    assert dropped_validation.checks["feasibility"].status == "fail"

    sign_flip_result = submitted_solution(
        ir,
        values={"x": 0.0, "y": 0.0},
        objective_value=0.0,
    )
    sign_validation = validate_candidate(
        contract, ir, compiled.certificate, sign_flip_result
    )
    assert sign_validation.checks["feasibility"].status == "pass"
    assert sign_validation.checks["optimality_oracle"].status == "fail"


def test_unmapped_contract_clause_cannot_be_promoted(tmp_path):
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)
    data = ir.model_dump(exclude={"ir_hash"})
    data["constraints"][1]["contract_clause_ids"] = ["resource_a_limit"]
    bad_ir = OptimizationModelIR.seal(**data)
    failures = preflight_candidate(contract, bad_ir)
    assert any("unmapped hard-constraint clause" in failure for failure in failures)
    with pytest.raises(CandidatePreflightError, match="before compilation"):
        ModelingAgent(tmp_path).assess_candidate(contract, bad_ir)
    assert ModelingAgent(tmp_path).run(contract, StaticExplorer([bad_ir])) == []


def test_clause_id_spoofing_fails_frozen_executable_preflight(tmp_path):
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)
    data = ir.model_dump(exclude={"ir_hash"})
    data["constraints"][1]["coefficients"] = {"x": 0.0, "y": 0.0}
    spoofed_ir = OptimizationModelIR.seal(**data)

    failures = preflight_candidate(contract, spoofed_ir)
    assert any("resource_b_counterexample" in failure for failure in failures)
    with pytest.raises(CandidatePreflightError, match="before compilation"):
        ModelingAgent(tmp_path).assess_candidate(contract, spoofed_ir)

    extra_data = ir.model_dump(exclude={"ir_hash"})
    extra_data["constraints"].append(
        {
            "constraint_id": "spoofed_x_zero",
            "coefficients": {"x": 1.0},
            "sense": "<=",
            "rhs": 0.0,
            "lhs_unit": "resource_a_unit",
            "rhs_unit": "resource_a_unit",
            "contract_clause_ids": ["resource_a_limit"],
        }
    )
    extra_constraint_ir = OptimizationModelIR.seal(**extra_data)
    extra_failures = preflight_candidate(contract, extra_constraint_ir)
    assert any("known_feasible_optimum" in failure for failure in extra_failures)


def test_objective_direction_spoof_fails_known_optimum_acceptance(tmp_path):
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)
    data = ir.model_dump(exclude={"ir_hash"})
    data["objective"]["sense"] = "minimize"
    spoofed_ir = OptimizationModelIR.seal(**data)

    assert preflight_candidate(contract, spoofed_ir) == []
    outcome = ModelingAgent(tmp_path).assess_candidate(contract, spoofed_ir)
    assert outcome.validation.checks["contract_acceptance"].status == "fail"
    assert outcome.decision.status == "run_invalid"


def test_unavailable_exact_oracle_requests_evidence_instead_of_passing(tmp_path):
    integer_contract = resource_allocation_contract()
    contract_data = integer_contract.model_dump(exclude={"frozen_hash"})
    for decision in contract_data["decisions"]:
        decision["kind"] = "continuous"
    contract = ProblemContract.freeze(**contract_data)
    integer_ir = resource_allocation_ir(integer_contract)
    data = integer_ir.model_dump(exclude={"ir_hash"})
    for variable in data["variables"]:
        variable["kind"] = "continuous"
    data["contract_hash"] = contract.frozen_hash
    continuous_ir = OptimizationModelIR.seal(**data)
    outcome = ModelingAgent(tmp_path).assess_candidate(contract, continuous_ir)

    assert outcome.validation.checks["optimality_oracle"].status == "not_run"
    assert not outcome.validation.hard_gates_pass(REQUIRED_HARD_CHECKS)
    assert outcome.decision.status == "needs_evidence"


def test_contract_and_ir_content_hashes_detect_mutation():
    contract = resource_allocation_contract()
    contract_data = contract.model_dump()
    contract_data["question"] = "A changed question that invalidates the old content hash"
    with pytest.raises(ValidationError, match="frozen_hash"):
        ProblemContract.model_validate(contract_data)

    ir = resource_allocation_ir(contract)
    ir_data = ir.model_dump()
    ir_data["objective"]["constant"] = 1.0
    with pytest.raises(ValidationError, match="ir_hash"):
        OptimizationModelIR.model_validate(ir_data)


def test_v1_contract_hash_remains_compatible_when_legacy_artifact_omits_public_facts():
    contract = resource_allocation_contract()
    legacy = contract.model_dump(
        mode="json", exclude={"frozen_hash", "public_facts", "decisions"}
    )
    legacy["schema_version"] = "1.0"
    from fma.hashing import sha256_value

    legacy["frozen_hash"] = sha256_value(legacy)
    loaded = ProblemContract.model_validate(legacy)

    assert loaded.public_facts == []
    assert loaded.content_hash() == legacy["frozen_hash"]
    loaded.assert_frozen()


def test_new_contracts_use_the_public_facts_and_decisions_schema_version():
    contract = resource_allocation_contract()

    assert contract.schema_version == "1.1"
    assert contract.decisions
    assert contract.public_facts


def test_artifact_reads_and_event_log_detect_tampering(tmp_path):
    store = RunStore(tmp_path)
    reference = store.put_artifact("sample", {"answer": 19})
    assert store.load_artifact(reference) == {"answer": 19}
    assert store.verify_event_chain() is True

    artifact_path = store.run_directory / reference.relative_path
    envelope = json.loads(artifact_path.read_text(encoding="utf-8"))
    envelope["payload"]["answer"] = 20
    artifact_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity"):
        store.load_artifact(reference)

    lines = store.event_path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["payload"]["run_id"] = "tampered"
    lines[0] = json.dumps(event)
    store.event_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert store.verify_event_chain() is False


def test_revoked_claim_cannot_be_repromoted(tmp_path):
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)
    compiled = compile_optimization(ir)
    outcome = ModelingAgent(tmp_path).run(contract, StaticExplorer([ir]))[0]
    assert outcome.decision.status == "validated"
    receipt = revoke_run_evidence(outcome, "reproduction", "test evidence withdrawal")
    assert receipt["effective_claim_status"] == "revoked"
    store = RunStore.open_existing(outcome.run_directory)
    with EvidenceGraph(store.run_directory / "evidence.sqlite3") as graph:
        with pytest.raises(RuntimeError, match="cannot be re-promoted"):
            PromotionEngine().decide(graph, store, outcome.claim_node_id)


def test_promotion_reloads_artifacts_and_rejects_dummy_evidence(tmp_path):
    store = RunStore(tmp_path)
    with EvidenceGraph(store.run_directory / "evidence.sqlite3") as graph:
        claim = graph.add_node("claim", status="proposed")
        for kind in REQUIRED_EVIDENCE_KINDS:
            node = graph.add_node(
                kind,
                artifact_hash="a" * 64,
                metadata={"relative_path": f"artifacts/{'a' * 64}.json"},
            )
            graph.add_edge(node, claim)
        decision = PromotionEngine().decide(graph, store, claim)
        assert decision.status == "run_invalid"
        assert not all(decision.gate_results.values())


def test_promotion_snapshot_is_exactly_retrievable(tmp_path):
    contract = resource_allocation_contract()
    ir = resource_allocation_ir(contract)
    outcome = ModelingAgent(tmp_path).run(contract, StaticExplorer([ir]))[0]
    store = RunStore.open_existing(outcome.run_directory)
    snapshot = store.load_artifact(outcome.decision.evidence_snapshot_artifact)
    assert snapshot["snapshot_hash"] == outcome.decision.evidence_snapshot_hash


def test_run_and_artifact_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        RunStore(tmp_path, run_id="../escaped")
    store = RunStore(tmp_path)
    from fma.schemas import ArtifactRef

    outside = ArtifactRef(kind="sample", sha256="a" * 64, relative_path="../outside.json")
    with pytest.raises(RuntimeError, match="escapes"):
        store.load_artifact(outside)
