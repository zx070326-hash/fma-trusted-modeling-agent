from __future__ import annotations

from pathlib import Path

from fma.studio.service import StudioTaskService
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import TaskWorkspaceSpecV50, WorkflowProfileV50
from fma.v6.recovery_kernel import (
    ProblemSignatureV60,
    RecoveryKernelV60,
    RecoveryPolicyV60,
    default_capability_registry_v60,
)


AUTHORITY_KEY = b"v6-recovery-test-authority-key-material"


def _workspace(tmp_path: Path) -> StageWorkspaceV50:
    root = tmp_path / "task"
    scaffold_task_workspace(
        root,
        "v6-recovery",
        "Recover a failed public mathematical modelling attempt.",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="v6-recovery",
        graph_id="v5-v6-recovery",
        objective="Recover a failed public mathematical modelling attempt.",
        mission_hash="1" * 64,
        evidence_snapshot_hash="2" * 64,
        evaluator_epoch="v6-test-epoch",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="synthetic_fixture",
        max_nodes=96,
        max_outcomes=96,
    )
    return StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id="v6-test-key",
    )


def test_capability_registry_reports_exact_gap_and_two_runnable_packs() -> None:
    registry = default_capability_registry_v60()
    small = registry.route(
        ProblemSignatureV60(
            state_kind="scalar",
            time_kind="continuous",
            dynamics_kind="autonomous",
            observation_kind="complete",
            task_kind="prediction",
            observation_count=11,
            positive_observations=True,
            strictly_increasing_time=True,
        )
    )
    assert small.status == "CAPABILITY_GAP"
    assert small.compatible_pack_ids == []
    assert small.incompatibilities == {
        "adaptive_positive_series_v57": ["observation_count:11<26"],
        "scalar_autonomous_ode_v52": ["observation_count:11<12"],
    }

    broad = registry.route(
        ProblemSignatureV60(
            state_kind="scalar",
            time_kind="continuous",
            dynamics_kind="autonomous",
            observation_kind="complete",
            task_kind="prediction",
            observation_count=40,
            positive_observations=True,
            strictly_increasing_time=True,
        )
    )
    assert broad.status == "ROUTABLE"
    assert broad.compatible_pack_ids == [
        "adaptive_positive_series_v57",
        "scalar_autonomous_ode_v52",
    ]


def test_partial_frontier_artifact_is_quarantined_without_new_attempt(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    partial = workspace.root / "results" / "index.json"
    partial.write_text('{"partial":true}\n', encoding="utf-8", newline="\n")
    kernel = RecoveryKernelV60(workspace)
    before = workspace._latest_attempt("S3")

    diagnosis, plan, receipt = kernel.recover(
        failed_stage="S3",
        category="partial_artifact",
        failure_code="interrupted_before_manifest",
        evidence_refs=kernel.evidence_refs_for_stage("S3"),
    )

    assert diagnosis.earliest_affected_stage is None
    assert plan.action == "RETRY"
    assert receipt.status == "SAME_ATTEMPT_RETRY_READY"
    assert workspace._latest_attempt("S3") == before
    assert not partial.exists()
    quarantined = (
        workspace.root
        / ".fma"
        / "recovery_v60"
        / "attempts"
        / f"a{before}"
        / "s3"
        / "quarantine"
        / "results"
        / "index.json"
    )
    assert quarantined.is_file()
    assert workspace.verify()


def test_s2_partial_retry_preserves_user_raw_data(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    raw = workspace.root / "data" / "raw" / "ode_series.json"
    raw.write_text('{"frozen":"user"}\n', encoding="utf-8", newline="\n")
    ledger = workspace.root / "data" / "ledger.json"
    ledger.write_text('{"partial":true}\n', encoding="utf-8", newline="\n")
    kernel = RecoveryKernelV60(workspace)

    _, _, receipt = kernel.recover(
        failed_stage="S2",
        category="partial_artifact",
        failure_code="interrupted_after_raw_freeze",
        evidence_refs=kernel.evidence_refs_for_stage("S2"),
    )

    assert receipt.status == "SAME_ATTEMPT_RETRY_READY"
    assert raw.is_file()
    assert not ledger.exists()
    assert "data/ledger.json" in receipt.quarantined_file_hashes
    assert "data/raw/ode_series.json" not in receipt.quarantined_file_hashes


def test_model_failure_revokes_from_s1_and_preserves_raw_data(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    model_spec = workspace.root / "docs" / "model_spec.json"
    model_spec.write_text("{}\n", encoding="utf-8", newline="\n")
    executable_intent = (
        workspace.root / "docs" / "executable_candidate_intent_v62.json"
    )
    executable_ir = (
        workspace.root / "docs" / "executable_candidate_ir_v62.json"
    )
    executable_resolution = (
        workspace.root / "docs" / "executable_candidate_resolution_v62.json"
    )
    executable_receipt = (
        workspace.root / "results" / "executable_candidate_receipt_v62.json"
    )
    raw = workspace.root / "data" / "raw" / "ode_series.json"
    source_contract = workspace.root / "docs" / "source_contract_v62.json"
    measurement = workspace.root / "docs" / "measurement_schema_v62.json"
    source_raw = (
        workspace.root
        / "data"
        / "source_provenance_v62"
        / "raw_response.json"
    )
    source_receipt = source_raw.with_name("receipt.json")
    source_acquisition = source_raw.with_name(
        "acquisition_authority_receipt.json"
    )
    source_verification = source_raw.with_name("verification.json")
    provenance_binding = source_raw.with_name("binding.json")
    source_reverification = (
        workspace.root / "checks" / "s2_source_reverification_v62.json"
    )
    decision_contract = (
        workspace.root / "docs" / "decision_value_contract_v62.json"
    )
    rolling_confirmation = (
        workspace.root / "results" / "rolling_confirmation_v61.json"
    )
    decision_evidence = (
        workspace.root / "results" / "decision_value_evidence_v62.json"
    )
    closure_root = (
        workspace.root / ".fma" / "scientific_closure_v62"
    )
    closure_report = closure_root / "report.json"
    closure_verification = closure_root / "checks" / "verification.json"
    for source_path in (
        source_contract,
        measurement,
        source_raw,
        source_receipt,
        source_acquisition,
        source_verification,
        provenance_binding,
        source_reverification,
    ):
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("{}\n", encoding="utf-8", newline="\n")
    raw.write_text('{"frozen":"public"}\n', encoding="utf-8", newline="\n")
    result = workspace.root / "results" / "index.json"
    for derived_path in (
        executable_intent,
        executable_ir,
        executable_resolution,
        executable_receipt,
        decision_contract,
        rolling_confirmation,
        decision_evidence,
        closure_report,
        closure_verification,
        result,
    ):
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        derived_path.write_text("{}\n", encoding="utf-8", newline="\n")
    kernel = RecoveryKernelV60(workspace)

    diagnosis, plan, receipt = kernel.recover(
        failed_stage="S4",
        category="model_assumption",
        failure_code="holdout_residual_structure",
        evidence_refs=kernel.evidence_refs_for_stage("S4"),
        expected_information_gain=0.6,
    )

    assert diagnosis.earliest_affected_stage == "S1"
    assert plan.action == "BRANCH"
    assert receipt.status == "ATTEMPT_CREATED"
    assert receipt.revoke_from == "S1"
    assert receipt.predecessor_attempt == 1
    assert receipt.successor_attempt == 2
    assert raw.is_file()
    assert source_contract.is_file()
    assert measurement.is_file()
    assert source_raw.is_file()
    assert source_receipt.is_file()
    assert source_acquisition.is_file()
    assert not source_verification.exists()
    assert not source_reverification.exists()
    assert not provenance_binding.exists()
    assert not executable_intent.exists()
    assert not executable_ir.exists()
    assert not executable_resolution.exists()
    assert not executable_receipt.exists()
    assert not decision_contract.exists()
    assert not rolling_confirmation.exists()
    assert not decision_evidence.exists()
    assert not closure_root.exists()
    assert (
        "data/source_provenance_v62/binding.json"
        in receipt.quarantined_file_hashes
    )
    for expected_path in (
        "docs/executable_candidate_intent_v62.json",
        "docs/executable_candidate_ir_v62.json",
        "docs/executable_candidate_resolution_v62.json",
        "results/executable_candidate_receipt_v62.json",
        "data/source_provenance_v62/verification.json",
        "checks/s2_source_reverification_v62.json",
        "docs/decision_value_contract_v62.json",
        "results/rolling_confirmation_v61.json",
        "results/decision_value_evidence_v62.json",
        ".fma/scientific_closure_v62/report.json",
        ".fma/scientific_closure_v62/checks/verification.json",
    ):
        assert expected_path in receipt.quarantined_file_hashes
    assert not model_spec.exists()
    assert not result.exists()
    assert workspace._latest_attempt("S1") == 2
    assert workspace.status().stage_statuses["S1"] == "pending"
    assert workspace.verify()


def test_s4_recovery_quarantines_stage_evidence_and_stale_closure(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    executable_receipt = (
        workspace.root / "results" / "executable_candidate_receipt_v62.json"
    )
    decision_contract = (
        workspace.root / "docs" / "decision_value_contract_v62.json"
    )
    rolling_confirmation = (
        workspace.root / "results" / "rolling_confirmation_v61.json"
    )
    decision_evidence = (
        workspace.root / "results" / "decision_value_evidence_v62.json"
    )
    closure_report = (
        workspace.root
        / ".fma"
        / "scientific_closure_v62"
        / "report.json"
    )
    for path in (
        executable_receipt,
        decision_contract,
        rolling_confirmation,
        decision_evidence,
        closure_report,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")
    kernel = RecoveryKernelV60(workspace)

    _, plan, receipt = kernel.recover(
        failed_stage="S4",
        category="uncertainty_calibration",
        failure_code="rolling_coverage_below_contract",
        evidence_refs=kernel.evidence_refs_for_stage("S4"),
    )

    assert plan.action == "PATCH"
    assert receipt.revoke_from == "S4"
    assert executable_receipt.is_file()
    assert decision_contract.is_file()
    assert not rolling_confirmation.exists()
    assert not decision_evidence.exists()
    assert not closure_report.parent.exists()
    assert (
        "docs/decision_value_contract_v62.json"
        not in receipt.quarantined_file_hashes
    )
    assert (
        "results/executable_candidate_receipt_v62.json"
        not in receipt.quarantined_file_hashes
    )
    for expected_path in (
        "results/rolling_confirmation_v61.json",
        "results/decision_value_evidence_v62.json",
        ".fma/scientific_closure_v62/report.json",
    ):
        assert expected_path in receipt.quarantined_file_hashes
    assert workspace.verify()


def test_s6_recovery_invalidates_only_post_gate_closure_projection(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    rolling_confirmation = (
        workspace.root / "results" / "rolling_confirmation_v61.json"
    )
    decision_evidence = (
        workspace.root / "results" / "decision_value_evidence_v62.json"
    )
    closure_root = workspace.root / ".fma" / "scientific_closure_v62"
    closure_report = closure_root / "report.json"
    closure_verification = closure_root / "verification.json"
    for path in (
        rolling_confirmation,
        decision_evidence,
        closure_report,
        closure_verification,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")
    kernel = RecoveryKernelV60(workspace)

    _, plan, receipt = kernel.recover(
        failed_stage="S6",
        category="paper_consistency",
        failure_code="paper_claim_binding_stale",
        evidence_refs=kernel.evidence_refs_for_stage("S6"),
    )

    assert plan.action == "PATCH"
    assert receipt.revoke_from == "S6"
    assert rolling_confirmation.is_file()
    assert decision_evidence.is_file()
    assert not closure_root.exists()
    assert (
        "results/rolling_confirmation_v61.json"
        not in receipt.quarantined_file_hashes
    )
    assert (
        "results/decision_value_evidence_v62.json"
        not in receipt.quarantined_file_hashes
    )
    assert sorted(
        path
        for path in receipt.quarantined_file_hashes
        if path.startswith(".fma/scientific_closure_v62/")
    ) == [
        ".fma/scientific_closure_v62/report.json",
        ".fma/scientific_closure_v62/verification.json",
    ]
    assert workspace.verify()


def test_private_holdout_exposure_abstains_without_graph_mutation(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(workspace)
    before = workspace.graph.project_state().snapshot.snapshot_hash

    diagnosis, plan, receipt = kernel.recover(
        failed_stage="S4",
        category="private_holdout_exposed",
        failure_code="private_score_visible",
        evidence_refs=["a" * 64],
        holdout_exposed=True,
        private_evidence_used=True,
    )

    assert diagnosis.retryable is False
    assert plan.action == "ABSTAIN"
    assert receipt.status == "ABSTAINED"
    assert receipt.before_graph_state_hash == receipt.after_graph_state_hash
    assert workspace.graph.project_state().snapshot.snapshot_hash == before
    assert kernel.summary()["stopped"] is True


def test_capability_gap_pauses_for_human_without_permanent_stop(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    kernel = RecoveryKernelV60(workspace)

    _, plan, receipt = kernel.recover(
        failed_stage="S1",
        category="capability_gap",
        failure_code="no_vector_pde_pack",
        evidence_refs=["c" * 64],
    )

    assert plan.action == "HUMAN"
    assert receipt.status == "HUMAN_REQUIRED"
    summary = kernel.summary()
    assert summary["stopped"] is False
    assert summary["human_required"] is True
    assert summary["human_reason"] == "no_vector_pde_pack"


def test_repeated_failure_signature_stops_instead_of_retrying_forever(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    policy = RecoveryPolicyV60.seal(
        max_scientific_attempts=3,
        max_same_failure=1,
        minimum_information_gain=0.05,
    )
    kernel = RecoveryKernelV60(workspace, policy=policy)
    evidence = ["b" * 64]

    _, _, first = kernel.recover(
        failed_stage="S3",
        category="operational_transient",
        failure_code="worker_timeout",
        evidence_refs=evidence,
    )
    assert first.status == "SAME_ATTEMPT_RETRY_READY"

    _, second_plan, second = kernel.recover(
        failed_stage="S3",
        category="operational_transient",
        failure_code="worker_timeout",
        evidence_refs=evidence,
    )
    assert second_plan.action == "ABSTAIN"
    assert second.status == "ABSTAINED"


def test_studio_recovery_entrypoint_projects_state_and_next_action(
    tmp_path: Path,
) -> None:
    service = StudioTaskService(
        tmp_path / "tasks",
        authority_key=AUTHORITY_KEY,
        authority_key_id="v6-studio-test-key",
    )
    created = service.create_task(
        {
            "objective": (
                "Recover an interrupted public numerical implementation "
                "without overwriting historical scientific evidence."
            ),
            "workspace_id": "studio-recovery",
        }
    )
    root = tmp_path / "tasks" / created["task_id"]
    partial = root / "results" / "index.json"
    partial.write_text('{"partial":true}\n', encoding="utf-8", newline="\n")

    recovered = service.recover(
        created["task_id"],
        {
            "failed_stage": "S3",
            "category": "partial_artifact",
            "failure_code": "interrupted_before_manifest",
        },
    )

    assert recovered["recovery"]["same_attempt_retries"] == 1
    assert recovered["recovery"]["stopped"] is False
    event = recovered["events"][-1]
    assert event["event_type"] == "recovery_transition_v60"
    assert event["details"]["transition_status"] == (
        "SAME_ATTEMPT_RETRY_READY"
    )
    assert not partial.exists()
