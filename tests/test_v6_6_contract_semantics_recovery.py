from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pytest

from fma.hashing import sha256_value
from fma.v6 import recovery_kernel as recovery_kernel_module
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import (
    StageWorkspaceError,
    StageWorkspaceV50,
)
from fma.v5.workspace_schemas import TaskWorkspaceSpecV50, WorkflowProfileV50
from fma.v6.recovery_kernel import (
    RecoveryKernelV60,
    RecoveryPolicyV60,
    RecoveryTransitionCompletionV66,
)


AUTHORITY_KEY = b"v6-6-contract-recovery-test-authority"


def _workspace(tmp_path: Path) -> StageWorkspaceV50:
    root = tmp_path / "task"
    scaffold_task_workspace(
        root,
        "v6-6-contract-recovery",
        "Repair a rejected S0 modelling contract without erasing history.",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="v6-6-contract-recovery",
        graph_id="v5-v6-6-contract-recovery",
        objective="Repair a rejected S0 modelling contract without erasing history.",
        mission_hash="1" * 64,
        evidence_snapshot_hash="2" * 64,
        evaluator_epoch="v6-6-contract-recovery-test",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="synthetic_fixture",
        max_nodes=96,
        max_outcomes=96,
    )
    return StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id="v6-6-contract-test-key",
    )


def test_contract_semantics_is_restricted_to_failed_stage_s0(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(workspace)

    with pytest.raises(
        ValueError,
        match="restricted to failed stage S0",
    ):
        kernel.diagnose(
            failed_stage="S1",
            category="contract_semantics",
            failure_code="normalized_finding_signature",
            evidence_refs=["a" * 64],
        )

    assert kernel.events() == []
    assert kernel.summary()["scientific_attempts_started"] == 1
    assert workspace._latest_attempt("S0") == 1
    assert workspace.verify()


def test_contract_semantics_creates_s0_patch_attempt_and_quarantines_profile(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    payloads = {
        "problem/contract.json": b'{"contract":"attempt-1"}\n',
        "problem/decision_function.json": b'{"decision":"attempt-1"}\n',
        "docs/regime.json": b'{"regime":"attempt-1"}\n',
        "docs/s0_evaluation_profile_v66.json": b'{"profile":"frozen-v6.6"}\n',
    }
    expected_hashes: dict[str, str] = {}
    for relative_path, payload in payloads.items():
        path = workspace.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected_hashes[relative_path] = hashlib.sha256(payload).hexdigest()

    old_bindings = {
        kind: str(workspace._binding("S0", kind).node_hash)
        for kind in ("work", "gate")
    }
    kernel = RecoveryKernelV60(workspace)

    diagnosis, plan, receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="s0_findings_0123456789abcdef",
        evidence_refs=kernel.evidence_refs_for_stage("S0"),
        expected_information_gain=0.5,
    )

    assert diagnosis.earliest_affected_stage == "S0"
    assert diagnosis.retryable is True
    assert diagnosis.candidate_change_required is False
    assert diagnosis.data_change_required is False
    assert plan.action == "PATCH"
    assert plan.revoke_from == "S0"
    assert plan.automatic_execution_permitted is True
    assert receipt.status == "ATTEMPT_CREATED"
    assert receipt.predecessor_attempt == 1
    assert receipt.successor_attempt == 2
    assert receipt.quarantined_file_hashes == dict(sorted(expected_hashes.items()))
    assert workspace._latest_attempt("S0") == 2

    quarantine_root = (
        workspace.root
        / ".fma"
        / "recovery_v60"
        / "attempts"
        / "a1"
        / "s0"
        / "quarantine"
    )
    for relative_path, digest in expected_hashes.items():
        assert not (workspace.root / relative_path).exists()
        quarantined = quarantine_root / relative_path
        assert quarantined.is_file()
        assert hashlib.sha256(quarantined.read_bytes()).hexdigest() == digest

    state = workspace.graph.project_state()
    current_node_hashes = {str(node.node_hash) for node in state.nodes}
    assert set(old_bindings.values()).issubset(current_node_hashes)
    for kind, old_hash in old_bindings.items():
        new_hash = str(workspace._binding("S0", kind).node_hash)
        assert any(
            edge.relation == "supersedes"
            and str(edge.source_node_hash) == old_hash
            and str(edge.target_node_hash) == new_hash
            for edge in state.edges
        )
    assert workspace.verify()


def test_same_contract_semantic_signature_is_patched_only_once(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(
        workspace,
        policy=RecoveryPolicyV60.seal(
            max_scientific_attempts=4,
            max_same_failure=4,
            minimum_information_gain=0.05,
        ),
    )
    evidence_refs = ["b" * 64]

    diagnosis, plan, first_receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="s0_findings_repeated",
        evidence_refs=evidence_refs,
    )
    assert first_receipt.status == "ATTEMPT_CREATED"
    assert workspace._latest_attempt("S0") == 2

    with pytest.raises(
        PermissionError,
        match="already patched",
    ):
        kernel.execute(diagnosis, plan)
    assert workspace._latest_attempt("S0") == 2

    _, second_plan, second_receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="s0_findings_repeated",
        evidence_refs=evidence_refs,
    )
    assert second_plan.action == "ABSTAIN"
    assert second_plan.revoke_from is None
    assert second_plan.automatic_execution_permitted is False
    assert second_receipt.status == "ABSTAINED"
    assert (
        second_receipt.before_graph_state_hash
        == second_receipt.after_graph_state_hash
    )
    assert workspace._latest_attempt("S0") == 2
    assert kernel.summary()["stopped"] is True
    assert workspace.verify()


def test_contract_semantics_respects_total_scientific_attempt_budget(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(
        workspace,
        policy=RecoveryPolicyV60.seal(
            max_scientific_attempts=2,
            max_same_failure=4,
            minimum_information_gain=0.05,
        ),
    )

    _, _, first_receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="s0_findings_first",
        evidence_refs=["c" * 64],
    )
    assert first_receipt.status == "ATTEMPT_CREATED"
    assert workspace._latest_attempt("S0") == 2

    _, second_plan, second_receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="s0_findings_distinct",
        evidence_refs=["d" * 64],
    )
    assert second_plan.action == "ABSTAIN"
    assert second_plan.attempt_budget_remaining == 0
    assert second_receipt.status == "ABSTAINED"
    assert (
        second_receipt.before_graph_state_hash
        == second_receipt.after_graph_state_hash
    )
    assert workspace._latest_attempt("S0") == 2
    assert workspace.verify()


def test_contract_semantics_with_private_evidence_never_mutates_graph(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(workspace)

    diagnosis, plan, receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="private_reviewer_context",
        evidence_refs=["e" * 64],
        private_evidence_used=True,
    )

    assert diagnosis.retryable is False
    assert plan.action == "ABSTAIN"
    assert plan.forbidden_evidence_refs == ["e" * 64]
    assert receipt.status == "ABSTAINED"
    assert receipt.before_graph_state_hash == receipt.after_graph_state_hash
    assert workspace._latest_attempt("S0") == 1
    assert workspace.verify()


def test_graph_mutation_crash_replays_intent_without_creating_attempt_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    contract = workspace.root / "problem" / "contract.json"
    contract.write_text('{"attempt":1}\n', encoding="utf-8", newline="\n")
    kernel = RecoveryKernelV60(workspace)
    original_quarantine = kernel._quarantine_expected
    interrupted = False

    def crash_after_graph_mutation(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt(
                "simulated loss after graph mutation before quarantine"
            )
        return original_quarantine(*args, **kwargs)

    monkeypatch.setattr(
        kernel,
        "_quarantine_expected",
        crash_after_graph_mutation,
    )
    with pytest.raises(KeyboardInterrupt):
        kernel.recover(
            failed_stage="S0",
            category="contract_semantics",
            failure_code="crash_after_invalidate",
            evidence_refs=kernel.evidence_refs_for_stage("S0"),
        )

    assert workspace._latest_attempt("S0") == 2
    assert contract.is_file()
    assert len(
        workspace._artifacts_of_kind("recovery_transition_intent_v66")
    ) == 1
    assert (
        workspace._artifacts_of_kind(
            "recovery_transition_completion_v66"
        )
        == []
    )

    monkeypatch.setattr(
        kernel,
        "_quarantine_expected",
        original_quarantine,
    )
    _, _, receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="crash_after_invalidate",
        evidence_refs=kernel.evidence_refs_for_stage("S0"),
    )

    assert receipt.status == "ATTEMPT_CREATED"
    assert receipt.predecessor_attempt == 1
    assert receipt.successor_attempt == 2
    assert workspace._latest_attempt("S0") == 2
    assert not contract.exists()
    assert len(
        workspace._artifacts_of_kind(
            "recovery_transition_completion_v66"
        )
    ) == 1
    state = kernel.load_state()
    assert state.scientific_attempts_started == 2
    assert list(state.failure_counts.values()) == [1]
    assert workspace.verify()


def test_receipt_commit_crash_rebuilds_budget_from_committed_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(
        workspace,
        policy=RecoveryPolicyV60.seal(
            max_scientific_attempts=4,
            max_same_failure=4,
            minimum_information_gain=0.05,
        ),
    )
    original_completion = kernel._commit_transition_completion
    interrupted = False

    def crash_after_receipt(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt(
                "simulated loss after transition receipt commit"
            )
        return original_completion(*args, **kwargs)

    monkeypatch.setattr(
        kernel,
        "_commit_transition_completion",
        crash_after_receipt,
    )
    with pytest.raises(KeyboardInterrupt):
        kernel.recover(
            failed_stage="S0",
            category="contract_semantics",
            failure_code="crash_after_receipt",
            evidence_refs=["f" * 64],
        )

    assert workspace._latest_attempt("S0") == 2
    assert len(
        workspace._artifacts_of_kind("recovery_transition_receipt_v60")
    ) == 1
    assert (
        workspace._artifacts_of_kind(
            "recovery_transition_completion_v66"
        )
        == []
    )
    assert len(kernel.committed_transition_records()) == 1
    assert kernel.completed_transition_records() == []

    intent = kernel._verified_transition_intents()[0]
    receipt_artifact_hash, committed_receipt, _, _ = (
        kernel.committed_transition_records()[0]
    )
    forged_unsigned = RecoveryTransitionCompletionV66(
        workspace_spec_hash=workspace.spec.spec_hash,
        policy_hash=kernel.policy.policy_hash,
        intent_hash=intent.intent_hash,
        transition_receipt_hash=committed_receipt.receipt_hash,
        transition_receipt_artifact_hash=receipt_artifact_hash,
        completed_at=committed_receipt.executed_at,
        authority_key_id=workspace.authority_key_id,
    )
    forged_payload = forged_unsigned.model_dump(mode="json")
    forged_payload["authority_auth_tag"] = "0" * 64
    forged_payload["completion_hash"] = sha256_value(
        {
            key: value
            for key, value in forged_payload.items()
            if key != "completion_hash"
        }
    )
    workspace.commit_evidence(
        "recovery_transition_completion_v66",
        RecoveryTransitionCompletionV66.model_validate(
            forged_payload
        ).model_dump(mode="json"),
    )
    assert kernel.completed_transition_records() == []

    monkeypatch.setattr(
        kernel,
        "_commit_transition_completion",
        original_completion,
    )
    _, _, replayed = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="crash_after_receipt",
        evidence_refs=["f" * 64],
    )
    assert replayed.status == "ATTEMPT_CREATED"
    assert workspace._latest_attempt("S0") == 2
    assert kernel.load_state().scientific_attempts_started == 2
    completed = kernel.completed_transition_records()
    assert len(completed) == 1
    assert completed[0][1].receipt_hash == replayed.receipt_hash

    _, stop_plan, stopped = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="crash_after_receipt",
        evidence_refs=["f" * 64],
    )
    assert stop_plan.action == "ABSTAIN"
    assert stopped.status == "ABSTAINED"
    assert workspace._latest_attempt("S0") == 2
    assert kernel.load_state().stopped is True
    assert workspace.verify()


def test_completed_transition_records_rejects_multiple_authenticated_completions(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(workspace)
    _, _, receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="duplicate_authenticated_completion",
        evidence_refs=["d" * 64],
    )
    assert receipt.status == "ATTEMPT_CREATED"
    assert len(kernel.completed_transition_records()) == 1

    original = kernel._verified_transition_completions()[0]
    duplicate_unsigned = RecoveryTransitionCompletionV66(
        workspace_spec_hash=original.workspace_spec_hash,
        policy_hash=original.policy_hash,
        intent_hash=original.intent_hash,
        transition_receipt_hash=original.transition_receipt_hash,
        transition_receipt_artifact_hash=(
            original.transition_receipt_artifact_hash
        ),
        completed_at=original.completed_at + timedelta(microseconds=1),
        authority_key_id=original.authority_key_id,
    )
    duplicate_payload = duplicate_unsigned.model_dump(mode="json")
    duplicate_payload["authority_auth_tag"] = workspace._mac(
        "recovery_transition_completion_v66",
        duplicate_unsigned.unsigned_hash(),
    )
    duplicate_payload["completion_hash"] = sha256_value(
        {
            key: value
            for key, value in duplicate_payload.items()
            if key != "completion_hash"
        }
    )
    duplicate = RecoveryTransitionCompletionV66.model_validate(
        duplicate_payload
    )
    workspace.commit_evidence(
        "recovery_transition_completion_v66",
        duplicate.model_dump(mode="json"),
    )

    with pytest.raises(
        StageWorkspaceError,
        match="multiple authenticated completions",
    ):
        kernel.completed_transition_records()


def test_mid_quarantine_crash_replays_dynamic_projection_from_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    contract = workspace.root / "problem" / "contract.json"
    contract.write_text('{"attempt":1}\n', encoding="utf-8", newline="\n")
    closure = (
        workspace.root
        / ".fma"
        / "scientific_closure_v62"
        / "closure.json"
    )
    closure.parent.mkdir(parents=True, exist_ok=True)
    closure.write_text('{"stale":true}\n', encoding="utf-8", newline="\n")
    kernel = RecoveryKernelV60(workspace)
    original_replace = recovery_kernel_module.os.replace
    interrupted = False

    def move_then_crash(source, destination):
        nonlocal interrupted
        if Path(source).resolve() == closure.resolve() and not interrupted:
            interrupted = True
            original_replace(source, destination)
            raise KeyboardInterrupt(
                "simulated loss after one dynamic projection move"
            )
        return original_replace(source, destination)

    monkeypatch.setattr(
        recovery_kernel_module.os,
        "replace",
        move_then_crash,
    )
    with pytest.raises(KeyboardInterrupt):
        kernel.recover(
            failed_stage="S0",
            category="contract_semantics",
            failure_code="crash_mid_dynamic_quarantine",
            evidence_refs=kernel.evidence_refs_for_stage("S0"),
        )

    assert workspace._latest_attempt("S0") == 2
    assert not closure.exists()
    assert contract.is_file()
    monkeypatch.setattr(
        recovery_kernel_module.os,
        "replace",
        original_replace,
    )

    _, _, receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="crash_mid_dynamic_quarantine",
        evidence_refs=kernel.evidence_refs_for_stage("S0"),
    )

    assert receipt.status == "ATTEMPT_CREATED"
    assert workspace._latest_attempt("S0") == 2
    assert not contract.exists()
    assert not closure.parent.exists()
    assert set(receipt.quarantined_file_hashes) == {
        ".fma/scientific_closure_v62/closure.json",
        "problem/contract.json",
    }
    assert len(
        workspace._artifacts_of_kind(
            "recovery_transition_completion_v66"
        )
    ) == 1
    assert workspace.verify()
