from __future__ import annotations

from pathlib import Path

from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import (
    TaskWorkspaceSpecV50,
    WorkflowProfileV50,
)
from fma.v6.recovery_kernel import RecoveryKernelV60


AUTHORITY_KEY = b"v6-7-review-recovery-test-authority"


def _workspace(tmp_path: Path) -> StageWorkspaceV50:
    root = tmp_path / "task"
    scaffold_task_workspace(
        root,
        "v6-7-review-recovery",
        "Recover a rejected pre-data modelling formalization.",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="v6-7-review-recovery",
        graph_id="v5-v6-7-review-recovery",
        objective="Recover a rejected pre-data modelling formalization.",
        mission_hash="1" * 64,
        evidence_snapshot_hash="2" * 64,
        evaluator_epoch="v6-7-review-recovery-epoch",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="synthetic_fixture",
        max_nodes=96,
        max_outcomes=96,
    )
    return StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id="v6-7-review-recovery-key",
    )


def test_s1_review_rejection_creates_patch_attempt_and_preserves_predata(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    measurement = (
        workspace.root / "docs" / "measurement_study_design_contract_v67.json"
    )
    protocol = workspace.root / "docs" / "predata_execution_protocol_v67.json"
    source = workspace.root / "docs" / "source_contract_v62.json"
    model_spec = workspace.root / "docs" / "model_spec.json"
    for path in (measurement, protocol, source, model_spec):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")

    kernel = RecoveryKernelV60(workspace)
    diagnosis, plan, receipt = kernel.recover(
        failed_stage="S1",
        category="review_rejection",
        failure_code="formalization_protocol_mismatch",
        evidence_refs=kernel.evidence_refs_for_stage("S1"),
    )

    assert diagnosis.earliest_affected_stage == "S1"
    assert diagnosis.candidate_change_required is False
    assert diagnosis.data_change_required is False
    assert plan.action == "PATCH"
    assert plan.revoke_from == "S1"
    assert receipt.status == "ATTEMPT_CREATED"
    assert receipt.successor_attempt == 2
    assert measurement.is_file()
    assert protocol.is_file()
    assert source.is_file()
    assert not model_spec.exists()
    assert "docs/model_spec.json" in receipt.quarantined_file_hashes
    assert (
        "docs/measurement_study_design_contract_v67.json"
        not in receipt.quarantined_file_hashes
    )
    assert (
        "docs/predata_execution_protocol_v67.json"
        not in receipt.quarantined_file_hashes
    )
    assert "docs/source_contract_v62.json" not in receipt.quarantined_file_hashes
    assert workspace.verify()


def test_s0_recovery_quarantines_predata_bound_to_revoked_s0(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    measurement = (
        workspace.root / "docs" / "measurement_study_design_contract_v67.json"
    )
    protocol = workspace.root / "docs" / "predata_execution_protocol_v67.json"
    source = workspace.root / "docs" / "source_contract_v62.json"
    for path in (measurement, protocol, source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")

    kernel = RecoveryKernelV60(workspace)
    _, plan, receipt = kernel.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="measurement_semantics_rejected",
        evidence_refs=kernel.evidence_refs_for_stage("S0"),
    )

    assert plan.action == "PATCH"
    assert receipt.status == "ATTEMPT_CREATED"
    assert not measurement.exists()
    assert not protocol.exists()
    assert not source.exists()
    assert (
        "docs/measurement_study_design_contract_v67.json"
        in receipt.quarantined_file_hashes
    )
    assert (
        "docs/predata_execution_protocol_v67.json"
        in receipt.quarantined_file_hashes
    )
    assert "docs/source_contract_v62.json" in receipt.quarantined_file_hashes
    assert workspace.verify()


def test_s2_data_recovery_preserves_frozen_v67_source_contract(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    paths = {
        relative: workspace.root / relative
        for relative in (
            "docs/measurement_study_design_contract_v67.json",
            "docs/predata_execution_protocol_v67.json",
            "docs/source_contract_v62.json",
            "data/ledger.json",
        )
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")

    kernel = RecoveryKernelV60(workspace)
    _, plan, receipt = kernel.recover(
        failed_stage="S2",
        category="data_contract",
        failure_code="source_response_incompatible_with_frozen_contract",
        evidence_refs=kernel.evidence_refs_for_stage("S2"),
    )

    assert plan.action == "ACQUIRE_DATA"
    assert receipt.status == "ATTEMPT_CREATED"
    assert paths["docs/source_contract_v62.json"].is_file()
    assert paths["docs/measurement_study_design_contract_v67.json"].is_file()
    assert paths["docs/predata_execution_protocol_v67.json"].is_file()
    assert not paths["data/ledger.json"].exists()
    assert "data/ledger.json" in receipt.quarantined_file_hashes
    assert "docs/source_contract_v62.json" not in receipt.quarantined_file_hashes
    assert workspace.verify()
