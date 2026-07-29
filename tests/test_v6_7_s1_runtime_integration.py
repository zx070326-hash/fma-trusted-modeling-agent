from __future__ import annotations

import json
from pathlib import Path

import pytest

from fma.v5_1.codex_stage_driver import StageRoleDriverV51
from fma.v6.measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    MeasurementStudyDesignContractV67,
)
from fma.v6.predata_protocol import (
    CANDIDATE_EXECUTION_BINDING_PATH_V67,
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    PreDataExecutionProtocolV67,
    compile_predata_execution_protocol_v67,
    registered_positive_series_capability_pack_v67,
)
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    WorldBankSourceContractV62,
)
from fma.studio.s1_runtime import (
    S1FormalizationRejectedV67,
    S1RuntimeError,
    StudioS1OrchestratorV58,
)
from tests.test_studio_bridge import (
    OBJECTIVE,
    _ode_backhalf_draft,
    _service,
)
from tests.test_v6_7_measurement_study_design import _contract_data


def _write_model(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(mode="json")  # type: ignore[attr-defined]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _predata_artifacts(
    tmp_path: Path,
    *,
    task_id: str,
    draft_factory=_ode_backhalf_draft,
):
    requests = []

    def recording_draft(request):
        requests.append(request)
        return draft_factory(request)

    service = _service(tmp_path, recording_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": task_id})
    service.run_s0(task_id)
    workspace = service._workspace(task_id)
    assert workspace.spec.spec_hash is not None
    assert workspace.current_gate("S0") is not None

    source = WorldBankSourceContractV62.seal(
        contract_id="world-bank-brazil-gdp-per-capita",
        country_code="BRA",
        indicator_id="NY.GDP.PCAP.KD",
        start_year=1980,
        end_year=2023,
        minimum_observations=40,
        time_unit="year",
        state_unit="constant_2015_USD_per_person",
        attribution=(
            "World Bank, World Development Indicators, NY.GDP.PCAP.KD."
        ),
    )
    data = _contract_data()
    data.update(
        {
            "workspace_spec_hash": workspace.spec.spec_hash,
            "s0_gate_hash": workspace.current_gate("S0"),
            "source_contract_id": source.contract_id,
            "source_contract_hash": source.contract_hash,
        }
    )
    contract = MeasurementStudyDesignContractV67.seal(**data)
    pack = registered_positive_series_capability_pack_v67(
        "scalar_autonomous_ode_v52"
    )
    protocol = compile_predata_execution_protocol_v67(
        measurement_contract=contract,
        capability_pack=pack,
    )
    _write_model(workspace.root / SOURCE_CONTRACT_PATH, source)
    _write_model(
        workspace.root / MEASUREMENT_STUDY_DESIGN_PATH_V67,
        contract,
    )
    _write_model(
        workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67,
        protocol,
    )
    return service, workspace, source, contract, protocol, requests


def _orchestrator(
    service,
    workspace,
    contract: MeasurementStudyDesignContractV67 | None,
    protocol: PreDataExecutionProtocolV67 | None,
) -> StudioS1OrchestratorV58:
    return StudioS1OrchestratorV58(
        workspace=workspace,
        task_id=workspace.spec.workspace_id,
        driver_factory=lambda: StageRoleDriverV51(
            service._transport(workspace.spec.workspace_id)
        ),
        event_callback=lambda *_: None,
        measurement_contract_v67=contract,
        predata_protocol_v67=protocol,
    )


def test_v67_s1_requires_an_exact_replay_verified_predata_pair(
    tmp_path: Path,
) -> None:
    service, workspace, _, contract, protocol, _ = _predata_artifacts(
        tmp_path,
        task_id="v67-pair",
    )

    with pytest.raises(S1RuntimeError, match="requires both"):
        _orchestrator(service, workspace, contract, None)

    foreign_payload = contract.model_dump(
        mode="json",
        exclude={"contract_hash"},
    )
    foreign_payload["workspace_spec_hash"] = "f" * 64
    foreign_contract = MeasurementStudyDesignContractV67.seal(
        **foreign_payload
    )
    with pytest.raises(S1RuntimeError, match="another workspace"):
        _orchestrator(service, workspace, foreign_contract, protocol)

    changed = protocol.model_dump(mode="json", exclude={"protocol_hash"})
    changed["fitting"]["optimizer_settings"]["max_nfev"] = 1
    replay_divergent = PreDataExecutionProtocolV67.seal(**changed)
    _write_model(
        workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67,
        replay_divergent,
    )
    with pytest.raises(S1RuntimeError, match="compiler replay"):
        _orchestrator(service, workspace, contract, replay_divergent)


def test_v67_s1_exposes_exact_protocol_and_binds_it_into_reviewed_manifest(
    tmp_path: Path,
) -> None:
    service, workspace, source, contract, protocol, requests = _predata_artifacts(
        tmp_path,
        task_id="v67-full-s1",
    )
    orchestrator = _orchestrator(
        service,
        workspace,
        contract,
        protocol,
    )

    result = orchestrator.run()

    assert result["gate_decision"] == "OPEN"
    manifest_paths = {
        item.relative_path for item in workspace._manifest_for_stage("S1").files
    }
    assert {
        SOURCE_CONTRACT_PATH,
        MEASUREMENT_STUDY_DESIGN_PATH_V67,
        PREDATA_EXECUTION_PROTOCOL_PATH_V67,
        CANDIDATE_EXECUTION_BINDING_PATH_V67,
    } <= manifest_paths

    synthesizer = next(
        request
        for request in requests
        if request.role_name == "s1_candidate_synthesizer"
    )
    assert synthesizer.public_inputs["predata_execution_protocol_v67"] == (
        protocol.model_dump(mode="json")
    )
    assert synthesizer.public_inputs[
        "measurement_study_design_contract_v67"
    ] == contract.model_dump(mode="json")
    assert synthesizer.public_inputs["source_contract_v62"] == source.model_dump(
        mode="json"
    )
    mechanical = synthesizer.public_inputs[
        "predata_protocol_mechanical_verification_v67"
    ]
    assert mechanical["replay_verified"] is True
    assert mechanical["ordered_level_rules_verified"] == [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    ]
    assert "Do not replace, broaden, or silently reinterpret" in (
        synthesizer.public_inputs["refinement_rule"]
    )

    auditor = next(
        request
        for request in requests
        if request.role_name == "s1_formalization_auditor"
    )
    assert auditor.public_inputs["predata_execution_protocol_v67"] == (
        protocol.model_dump(mode="json")
    )
    audit_rule = auditor.public_inputs["audit_rule"]
    assert "compatible with and do not loosen or contradict" in audit_rule
    assert "Do not ask the model to restate optimizer" in audit_rule
    assert "fully specified" not in audit_rule
    binding = auditor.public_inputs["candidate_execution_binding_v67"]
    assert binding["allowed_adapter_ids"] == ["scalar_autonomous_ode_v52"]
    assert binding["adapter_resolution_stage"] == "pre_data_compiler"
    assert binding["legacy_v62_resolution_is_authority"] is False

    final_reviewers = [
        request
        for request in requests
        if request.role_name in {"s1_referee", "s1_red_team"}
    ]
    assert len(final_reviewers) == 2
    for request in final_reviewers:
        packet = request.public_inputs["review_evidence_packet"]
        assert packet["source_contract_v62"] == source.model_dump(mode="json")
        assert packet["measurement_study_design_contract_v67"] == (
            contract.model_dump(mode="json")
        )
        assert packet["predata_execution_protocol_v67"] == protocol.model_dump(
            mode="json"
        )
        assert packet["candidate_execution_binding_v67"][
            "allowed_adapter_ids"
        ] == ["scalar_autonomous_ode_v52"]
        assert packet["candidate_execution_binding_v67"][
            "legacy_v62_resolution_is_authority"
        ] is False
        assert packet["predata_protocol_mechanical_verification_v67"][
            "replay_verified"
        ] is True
        assert "code-owned protocol, not model prose" in (
            request.public_inputs["review_rule"]
        )


def test_v67_second_preflight_rejection_emits_typed_recovery_handoff(
    tmp_path: Path,
) -> None:
    finding = (
        "The selected candidate contradicts the exact registered family "
        "semantics in the sealed pre-data protocol."
    )

    def rejecting_auditor(request):
        payload = _ode_backhalf_draft(request)
        if request.role_name == "s1_formalization_auditor":
            payload["verdict"] = "REJECT"
            payload["rationale"] = (
                "Candidate-protocol compatibility remains false after review."
            )
            payload["findings"] = [finding]
        return payload

    service, workspace, _, contract, protocol, _ = _predata_artifacts(
        tmp_path,
        task_id="v67-rejected-s1",
        draft_factory=rejecting_auditor,
    )
    orchestrator = _orchestrator(
        service,
        workspace,
        contract,
        protocol,
    )

    with pytest.raises(S1FormalizationRejectedV67) as captured:
        orchestrator.run()

    error = captured.value
    assert error.recovery_category == "review_rejection"
    assert error.recovery_failure_code == "s1_formalization_review_rejected"
    assert error.findings == (finding,)
    assert error.normalized_findings == (finding.casefold(),)
    assert len(error.normalized_finding_signature) == 64
    assert len(error.reviewer_receipt_hash) == 64
    assert error.protocol_hash == protocol.protocol_hash
    assert error.handoff_artifact_hash is not None
    handoffs = workspace._artifacts_of_kind(
        "s1_formalization_rejection_handoff_v67"
    )
    assert len(handoffs) == 1
    assert handoffs[0][0].sha256 == error.handoff_artifact_hash
    assert handoffs[0][1]["predecessor_attempt"] == 1
    assert handoffs[0][1]["recovery_disposition"] == "bounded_patch"
    with pytest.raises(AttributeError):
        error.findings = ("model-selected rollback",)  # type: ignore[misc]
    assert workspace.current_gate("S1") is None


def test_iteration_39_regression_uses_v67_binding_not_legacy_s2_envelope(
    tmp_path: Path,
) -> None:
    """A protocol-aware reviewer must not hit the former stage-order deadlock."""

    def protocol_aware_reviewer(request):
        payload = _ode_backhalf_draft(request)
        binding = None
        if request.role_name == "s1_formalization_auditor":
            binding = request.public_inputs.get(
                "candidate_execution_binding_v67"
            )
        elif request.role_name in {"s1_referee", "s1_red_team"}:
            binding = request.public_inputs["review_evidence_packet"].get(
                "candidate_execution_binding_v67"
            )
        if request.role_name in {
            "s1_formalization_auditor",
            "s1_referee",
            "s1_red_team",
        } and (
            not isinstance(binding, dict)
            or binding.get("allowed_adapter_ids")
            != ["scalar_autonomous_ode_v52"]
            or binding.get("adapter_resolution_stage")
            != "pre_data_compiler"
            or binding.get("legacy_v62_resolution_is_authority") is not False
        ):
            payload["verdict"] = "REJECT"
            payload["rationale"] = (
                "The legacy two-adapter S2 envelope remains authoritative."
            )
            payload["findings"] = [
                "Exact pre-data adapter resolution is absent."
            ]
        return payload

    service, workspace, _, contract, protocol, _ = _predata_artifacts(
        tmp_path,
        task_id="i39-v67-regression",
        draft_factory=protocol_aware_reviewer,
    )

    result = _orchestrator(
        service,
        workspace,
        contract,
        protocol,
    ).run()

    assert result["gate_decision"] == "OPEN"
    assert workspace.current_gate("S1") is not None
