from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fma.codex_driver import CodexCLIConfig
from fma.hashing import canonical_json
from fma.schemas import ArtifactRef
from fma.v5_1.codex_stage_driver import FixtureStageRoleTransportV51
from fma.studio.backhalf_runtime import RAW_RELATIVE_PATH
from fma.studio.service import (
    StudioConflictError,
    StudioTaskService,
    StudioValidationError,
    StudioWorldBankDataRequestV62,
    build_world_bank_predata_bundle_v67,
)
from fma.studio.s1_runtime import S1FormalizationRejectedV67
from fma.v6.measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    MeasurementStudyDesignContractV67,
)
from fma.v6.predata_protocol import (
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    PreDataExecutionProtocolV67,
)
from fma.v6.provenance import MEASUREMENT_SCHEMA_PATH
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    SOURCE_RAW_PATH,
    SOURCE_RECEIPT_PATH,
    SourceHTTPResponseV62,
    WorldBankSourceContractV62,
)
from fma.v6.recovery_kernel import RecoveryKernelV60
from fma.v6.s1_review_recovery import (
    S1FormalizationRejectionHandoffV67,
    build_s1_formalization_rejection_handoff_v67,
    s1_recovery_evidence_refs_v67,
)
from fma.v6 import real_local_campaign_cli
from fma.v6 import real_local_campaign as campaign_module
from fma.v6 import real_local_campaign_v67_cli
from fma.v6.real_local_campaign import (
    EVENTS_PATH_V65,
    SPEC_PATH_V65,
    CodexRuntimeBudgetsV65,
    RealLocalCampaignEventV65,
)
from fma.v6.real_local_campaign_v67 import (
    ACTION_ORDER_V67,
    EVENTS_PATH_V67,
    SPEC_PATH_V67,
    CodexRuntimeBudgetsV67,
    CodexRuntimeContractV67,
    RealLocalCampaignRunnerV67,
    RealLocalCampaignSpecV67,
    build_codex_runtime_contract_v67,
)
from fma.v6.source_auth import SOURCE_ACQUISITION_AUTH_PATH
from tests.test_studio_bridge import (
    OBJECTIVE,
    _ode_backhalf_draft,
    _service,
)
from tests.test_v6_5_real_local_campaign import (
    AUTHORITY_KEY,
    _FakeStudio,
    _factory_for,
    _frozen_runtime_contract,
    _spec,
)


def _request(
    *,
    semantic_name: str = "real GDP per capita",
) -> StudioWorldBankDataRequestV62:
    return StudioWorldBankDataRequestV62(
        adapter_id="scalar_autonomous_ode_v52",
        contract_id="wb-brazil-real-gdp-per-capita",
        country_code="BRA",
        indicator_id="NY.GDP.PCAP.KD",
        start_year=1980,
        end_year=2023,
        minimum_observations=23,
        state_unit="constant_2015_USD_per_person",
        attribution=("World Bank, World Development Indicators, NY.GDP.PCAP.KD."),
        semantic_name=semantic_name,
        operational_definition=(
            "Annual GDP per capita in constant 2015 US dollars under the "
            "registered official indicator definition."
        ),
        observation_time_basis="annual calendar-year observation",
        aggregation_level="national country aggregate per resident",
        fixture_only=True,
    )


def _campaign_spec_v67() -> RealLocalCampaignSpecV67:
    legacy = _spec()
    payload = legacy.model_dump(
        mode="json",
        exclude={"schema_version", "spec_hash"},
    )
    payload["world_bank_request"]["minimum_observations"] = 34
    return RealLocalCampaignSpecV67.seal(**payload)


def _runtime_contract_v67() -> CodexRuntimeContractV67:
    legacy = _frozen_runtime_contract()
    payload = legacy.model_dump(
        mode="json",
        exclude={"schema_version", "budgets", "contract_hash"},
    )
    return CodexRuntimeContractV67.seal(
        **payload,
        budgets=CodexRuntimeBudgetsV67.from_config(
            CodexCLIConfig(requested_model="gpt-test")
        ),
    )


def _real_campaign_spec_v67() -> RealLocalCampaignSpecV67:
    legacy = _spec(
        runtime_contract=_frozen_runtime_contract(),
        execution_mode="real",
    )
    payload = legacy.model_dump(
        mode="json",
        exclude={
            "schema_version",
            "codex_runtime_contract",
            "spec_hash",
        },
    )
    payload["world_bank_request"]["minimum_observations"] = 34
    return RealLocalCampaignSpecV67.seal(
        **payload,
        codex_runtime_contract=_runtime_contract_v67(),
    )


def _prepare_service(
    tmp_path: Path,
    *,
    requests: list[Any] | None = None,
    world_bank_fetcher: Any | None = None,
) -> tuple[StudioTaskService, str, StudioWorldBankDataRequestV62]:
    def draft(request: Any) -> dict[str, Any]:
        if requests is not None:
            requests.append(request)
        return _ode_backhalf_draft(request)

    service = _service(
        tmp_path,
        draft,
        world_bank_fetcher=world_bank_fetcher,
    )
    task_id = "v67-service-predata"
    service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": task_id,
            "evidence_scope": "development",
            "workflow_mode": "v67",
        }
    )
    service.run_s0(task_id)
    return service, task_id, _request()


def _predata_protocol(
    service: StudioTaskService,
    task_id: str,
) -> PreDataExecutionProtocolV67:
    workspace = service._workspace(task_id)
    protocol = PreDataExecutionProtocolV67.model_validate_json(
        (workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67).read_text(
            encoding="utf-8"
        )
    )
    assert protocol.protocol_hash is not None
    return protocol


def _commit_s1_rejection_handoff(
    service: StudioTaskService,
    task_id: str,
    *,
    existing_repair_context_hash: str | None = None,
) -> tuple[
    PreDataExecutionProtocolV67,
    ArtifactRef,
    S1FormalizationRejectionHandoffV67,
]:
    workspace = service._workspace(task_id)
    protocol = _predata_protocol(service, task_id)
    assert workspace.spec.spec_hash is not None
    assert workspace.current_gate("S0") is not None
    handoff = build_s1_formalization_rejection_handoff_v67(
        workspace_spec_hash=workspace.spec.spec_hash,
        s0_gate_hash=workspace.current_gate("S0"),
        predata_protocol_hash=protocol.protocol_hash,
        reviewer_receipt_hash="d" * 64,
        findings=["Candidate family semantics contradict the frozen protocol."],
        predecessor_attempt=workspace._latest_attempt("S1"),
        existing_repair_context_hash=existing_repair_context_hash,
    )
    reference = workspace.commit_evidence(
        "s1_formalization_rejection_handoff_v67",
        handoff.model_dump(mode="json"),
    )
    return protocol, reference, handoff


def test_prepare_predata_freezes_exact_bundle_without_data_access(
    tmp_path: Path,
) -> None:
    fetch_calls: list[object] = []

    def forbidden_fetch(*args: object, **kwargs: object) -> object:
        fetch_calls.append((args, kwargs))
        raise AssertionError("pre-data preparation opened the source")

    service, task_id, request = _prepare_service(
        tmp_path,
        world_bank_fetcher=forbidden_fetch,
    )
    workspace = service._workspace(task_id)
    assert workspace.spec.spec_hash is not None
    assert workspace.current_gate("S0") is not None

    result = service.prepare_predata_v67(task_id, request)

    source = WorldBankSourceContractV62.model_validate_json(
        (workspace.root / SOURCE_CONTRACT_PATH).read_text(encoding="utf-8")
    )
    measurement = MeasurementStudyDesignContractV67.model_validate_json(
        (workspace.root / MEASUREMENT_STUDY_DESIGN_PATH_V67).read_text(encoding="utf-8")
    )
    protocol = PreDataExecutionProtocolV67.model_validate_json(
        (workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67).read_text(
            encoding="utf-8"
        )
    )
    expected = build_world_bank_predata_bundle_v67(
        request=request,
        workspace_spec_hash=workspace.spec.spec_hash,
        s0_gate_hash=workspace.current_gate("S0"),
    )
    assert (source, measurement, protocol) == expected
    assert protocol.source_contract_id == source.contract_id
    assert protocol.source_contract_hash == source.contract_hash
    assert measurement.source_contract_id == source.contract_id
    assert measurement.source_contract_hash == source.contract_hash
    assert result["workflow"]["stage_statuses"]["S1"] == "frontier"
    assert fetch_calls == []
    assert not (workspace.root / SOURCE_RAW_PATH).exists()
    assert not (workspace.root / RAW_RELATIVE_PATH).exists()

    evidence = workspace._artifacts_of_kind("predata_preparation_v67")
    assert len(evidence) == 1
    assert evidence[0][1]["network_accessed"] is False
    assert evidence[0][1]["observation_values_accessed"] is False
    assert result["events"][-1]["event_type"] == ("predata_bundle_prepared_v67")
    assert result["events"][-1]["details"]["network_accessed"] is False

    replay = service.prepare_predata_v67(task_id, request)
    assert replay["predata_v67"] == result["predata_v67"]
    assert replay["events"] == result["events"]


def test_run_s1_receives_predata_and_ingest_mismatch_stops_before_fetch(
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    fetch_calls: list[object] = []

    def forbidden_fetch(*args: object, **kwargs: object) -> object:
        fetch_calls.append((args, kwargs))
        raise AssertionError("mismatched request reached the network")

    service, task_id, request = _prepare_service(
        tmp_path,
        requests=requests,
        world_bank_fetcher=forbidden_fetch,
    )
    service.prepare_predata_v67(task_id, request)

    result = service.run_s1(task_id)

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    synthesizer = next(
        item for item in requests if item.role_name == "s1_candidate_synthesizer"
    )
    assert "measurement_study_design_contract_v67" in (synthesizer.public_inputs)
    assert "predata_execution_protocol_v67" in synthesizer.public_inputs
    assert "source_contract_v62" in synthesizer.public_inputs

    mismatched = request.model_copy(
        update={"semantic_name": "a different registered construct"}
    )
    with pytest.raises(
        StudioConflictError,
        match="differs from the frozen V6.7",
    ):
        service.ingest_world_bank_data(task_id, mismatched)

    workspace = service._workspace(task_id)
    assert fetch_calls == []
    for relative_path in (
        SOURCE_RAW_PATH,
        SOURCE_RECEIPT_PATH,
        SOURCE_ACQUISITION_AUTH_PATH,
        RAW_RELATIVE_PATH,
    ):
        assert not (workspace.root / relative_path).exists()

    def fixture_fetcher(
        contract: WorldBankSourceContractV62,
    ) -> SourceHTTPResponseV62:
        records = [
            {
                "indicator": {
                    "id": contract.indicator_id,
                    "value": request.semantic_name,
                },
                "country": {
                    "id": "BR",
                    "value": "Brazil",
                },
                "countryiso3code": contract.country_code,
                "date": str(year),
                "value": float(1000 + year - contract.start_year),
                "unit": "",
                "obs_status": "",
                "decimal": 0,
            }
            for year in range(
                contract.end_year,
                contract.start_year - 1,
                -1,
            )
        ]
        body = json.dumps(
            [
                {
                    "page": 1,
                    "pages": 1,
                    "per_page": 1000,
                    "total": len(records),
                    "sourceid": "2",
                    "lastupdated": "2026-07-28",
                },
                records,
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        return SourceHTTPResponseV62(
            status=200,
            final_url=contract.exact_url,
            content_type="application/json",
            body=body,
        )

    service.world_bank_fetcher = fixture_fetcher
    intake = service.ingest_world_bank_data(task_id, request)

    assert intake["backhalf"]["data_received"] is True
    for relative_path in (
        SOURCE_RAW_PATH,
        SOURCE_RECEIPT_PATH,
        SOURCE_ACQUISITION_AUTH_PATH,
        MEASUREMENT_SCHEMA_PATH,
        RAW_RELATIVE_PATH,
    ):
        assert (workspace.root / relative_path).is_file()


def test_s1_typed_rejection_creates_successor_and_retries_with_bounded_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Any] = []
    service, task_id, request = _prepare_service(
        tmp_path,
        requests=requests,
    )
    service.prepare_predata_v67(task_id, request)
    workspace = service._workspace(task_id)
    protocol = PreDataExecutionProtocolV67.model_validate_json(
        (workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67).read_text(
            encoding="utf-8"
        )
    )
    assert protocol.protocol_hash is not None

    import fma.studio.service as service_module

    real_orchestrator = service_module.StudioS1OrchestratorV58
    construction_count = 0

    class RejectOnce:
        def run(self) -> None:
            raise S1FormalizationRejectedV67(
                findings=["Candidate family semantics contradict the frozen protocol."],
                reviewer_receipt_hash="d" * 64,
                protocol_hash=protocol.protocol_hash,
            )

    def orchestrator_factory(**kwargs: Any) -> Any:
        nonlocal construction_count
        construction_count += 1
        if construction_count == 1:
            return RejectOnce()
        return real_orchestrator(**kwargs)

    monkeypatch.setattr(
        service_module,
        "StudioS1OrchestratorV58",
        orchestrator_factory,
    )

    result = service.run_s1(task_id)

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert workspace._latest_attempt("S1") == 2
    assert construction_count == 2
    assert (
        len(workspace._artifacts_of_kind("s1_formalization_rejection_evidence_v67"))
        == 1
    )
    assert len(workspace._artifacts_of_kind("s1_bounded_repair_context_v67")) == 1
    repair_synthesizer = next(
        item
        for item in requests
        if item.role_name == "s1_candidate_synthesizer"
        and "s1_graph_repair_context_v67" in item.public_inputs
    )
    context = repair_synthesizer.public_inputs["s1_graph_repair_context_v67"]
    assert context["successor_attempt"] == 2
    assert context["protocol_change_permitted"] is False
    assert context["adapter_change_permitted"] is False
    assert context["threshold_change_permitted"] is False
    assert result["recovery"]["scientific_attempts_started"] == 2
    assert result["recovery"]["last_revoke_from"] == "S1"


def test_real_s1_review_rejection_rebuilds_attempt_and_opens_gate(
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    auditor_calls = 0
    graph_repair_seen = False

    def reject_first_graph_attempt(request: Any) -> dict[str, Any]:
        nonlocal auditor_calls, graph_repair_seen
        requests.append(request)
        payload = _ode_backhalf_draft(request)
        if (
            request.role_name == "s1_candidate_synthesizer"
            and "s1_graph_repair_context_v67" in request.public_inputs
        ):
            graph_repair_seen = True
        if request.role_name == "s1_formalization_auditor":
            auditor_calls += 1
            if not graph_repair_seen:
                payload["verdict"] = "REJECT"
                payload["rationale"] = (
                    "The candidate still contradicts the exact frozen "
                    "capability protocol."
                )
                payload["findings"] = [
                    "Candidate family semantics contradict the frozen protocol."
                ]
        return payload

    service = _service(tmp_path, reject_first_graph_attempt)
    task_id = "v67-real-s1-graph-repair"
    service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": task_id,
            "evidence_scope": "development",
            "workflow_mode": "v67",
        }
    )
    service.run_s0(task_id)
    service.prepare_predata_v67(task_id, _request())

    result = service.run_s1(task_id)
    workspace = service._workspace(task_id)

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert workspace._latest_attempt("S1") == 2
    assert auditor_calls == 3
    assert graph_repair_seen is True
    repair_requests = [
        item for item in requests if "s1_graph_repair_context_v67" in item.public_inputs
    ]
    assert len(repair_requests) == 1
    assert repair_requests[0].role_name == "s1_candidate_synthesizer"
    assert result["recovery"]["scientific_attempts_started"] == 2
    assert result["recovery"]["human_required"] is False
    assert result["recovery"]["stopped"] is False


def test_second_s1_typed_rejection_stops_at_human_without_attempt_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, task_id, request = _prepare_service(tmp_path)
    service.prepare_predata_v67(task_id, request)
    workspace = service._workspace(task_id)
    protocol = PreDataExecutionProtocolV67.model_validate_json(
        (workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67).read_text(
            encoding="utf-8"
        )
    )
    assert protocol.protocol_hash is not None

    import fma.studio.service as service_module

    construction_count = 0

    class AlwaysReject:
        def __init__(self, **_: Any) -> None:
            nonlocal construction_count
            construction_count += 1

        def run(self) -> None:
            raise S1FormalizationRejectedV67(
                findings=["Candidate family semantics contradict the frozen protocol."],
                reviewer_receipt_hash="e" * 64,
                protocol_hash=protocol.protocol_hash,
            )

    monkeypatch.setattr(
        service_module,
        "StudioS1OrchestratorV58",
        AlwaysReject,
    )

    with pytest.raises(
        StudioValidationError,
        match="HUMAN_REQUIRED",
    ):
        service.run_s1(task_id)

    snapshot = service.snapshot(task_id)
    assert workspace._latest_attempt("S1") == 2
    assert snapshot["recovery"]["human_required"] is True
    assert snapshot["recovery"]["stopped"] is False
    assert snapshot["next_valid_actions"] == ["inspect_s1"]
    assert construction_count == 2
    assert (
        len(workspace._artifacts_of_kind("s1_formalization_rejection_terminal_v67"))
        == 1
    )
    with pytest.raises(StudioValidationError, match="HUMAN_REQUIRED"):
        service.run_s1(task_id)
    assert construction_count == 2


def test_s1_crash_window_without_committed_repair_context_fails_closed(
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    service, task_id, request = _prepare_service(
        tmp_path,
        requests=requests,
    )
    service.prepare_predata_v67(task_id, request)
    workspace = service._workspace(task_id)
    before_calls = len(requests)

    _, _, receipt = RecoveryKernelV60(workspace).recover(
        failed_stage="S1",
        category="review_rejection",
        failure_code="s1_formalization_review_rejected",
        evidence_refs=RecoveryKernelV60(workspace).evidence_refs_for_stage("S1"),
    )
    assert receipt.status == "ATTEMPT_CREATED"
    assert workspace._latest_attempt("S1") == 2

    with pytest.raises(
        StudioConflictError,
        match="lacks its committed bounded repair context",
    ):
        service.run_s1(task_id)
    assert len(requests) == before_calls


def test_s1_restart_after_handoff_before_kernel_resumes_once(
    tmp_path: Path,
) -> None:
    requests: list[Any] = []
    service, task_id, request = _prepare_service(
        tmp_path,
        requests=requests,
    )
    service.prepare_predata_v67(task_id, request)
    workspace = service._workspace(task_id)
    _, handoff_ref, _ = _commit_s1_rejection_handoff(service, task_id)
    assert RecoveryKernelV60(workspace).committed_transition_records() == []

    result = service.run_s1(task_id)

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert workspace._latest_attempt("S1") == 2
    assert len(workspace._artifacts_of_kind("s1_bounded_repair_context_v67")) == 1
    completed = RecoveryKernelV60(workspace).completed_transition_records()
    assert len(completed) == 1
    assert handoff_ref.sha256 in completed[0][2].evidence_refs
    assert any(
        "s1_graph_repair_context_v67" in item.public_inputs
        for item in requests
        if item.stage == "S1"
    )
    assert workspace.verify()


def test_s1_restart_after_receipt_before_completion_closes_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, task_id, request = _prepare_service(tmp_path)
    service.prepare_predata_v67(task_id, request)
    workspace = service._workspace(task_id)
    _, handoff_ref, handoff = _commit_s1_rejection_handoff(service, task_id)
    kernel = RecoveryKernelV60(workspace)

    def crash_before_completion(*_: Any, **__: Any) -> None:
        raise KeyboardInterrupt("simulated loss before transition completion")

    monkeypatch.setattr(
        kernel,
        "_commit_transition_completion",
        crash_before_completion,
    )
    with pytest.raises(KeyboardInterrupt):
        kernel.recover(
            failed_stage="S1",
            category="review_rejection",
            failure_code="s1_formalization_review_rejected",
            evidence_refs=s1_recovery_evidence_refs_v67(
                handoff,
                handoff_artifact_hash=handoff_ref.sha256,
            ),
            expected_information_gain=0.5,
        )

    assert workspace._latest_attempt("S1") == 2
    assert len(kernel.committed_transition_records()) == 1
    assert kernel.completed_transition_records() == []
    result = service.run_s1(task_id)

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert workspace._latest_attempt("S1") == 2
    assert len(RecoveryKernelV60(workspace).completed_transition_records()) == 1
    assert len(workspace._artifacts_of_kind("s1_bounded_repair_context_v67")) == 1
    assert workspace.verify()


def test_s1_restart_after_completion_before_projection_rebuilds_context(
    tmp_path: Path,
) -> None:
    service, task_id, request = _prepare_service(tmp_path)
    service.prepare_predata_v67(task_id, request)
    workspace = service._workspace(task_id)
    _, handoff_ref, handoff = _commit_s1_rejection_handoff(service, task_id)
    kernel = RecoveryKernelV60(workspace)
    _, _, receipt = kernel.recover(
        failed_stage="S1",
        category="review_rejection",
        failure_code="s1_formalization_review_rejected",
        evidence_refs=s1_recovery_evidence_refs_v67(
            handoff,
            handoff_artifact_hash=handoff_ref.sha256,
        ),
        expected_information_gain=0.5,
    )
    assert receipt.status == "ATTEMPT_CREATED"
    assert len(kernel.completed_transition_records()) == 1
    assert workspace._artifacts_of_kind("s1_bounded_repair_context_v67") == []

    result = service.run_s1(task_id)

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert workspace._latest_attempt("S1") == 2
    assert (
        len(workspace._artifacts_of_kind("s1_formalization_rejection_evidence_v67"))
        == 1
    )
    assert len(workspace._artifacts_of_kind("s1_bounded_repair_context_v67")) == 1
    assert workspace.verify()


def test_s1_restart_after_terminal_receipt_materializes_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, task_id, request = _prepare_service(tmp_path)
    service.prepare_predata_v67(task_id, request)
    workspace = service._workspace(task_id)
    protocol = _predata_protocol(service, task_id)

    import fma.studio.service as service_module

    construction_count = 0

    class AlwaysReject:
        def __init__(self, **_: Any) -> None:
            nonlocal construction_count
            construction_count += 1

        def run(self) -> None:
            raise S1FormalizationRejectedV67(
                findings=["Candidate family semantics contradict the frozen protocol."],
                reviewer_receipt_hash="e" * 64,
                protocol_hash=protocol.protocol_hash,
            )

    monkeypatch.setattr(
        service_module,
        "StudioS1OrchestratorV58",
        AlwaysReject,
    )
    original_commit = service._commit_evidence_once

    def crash_before_terminal_projection(
        target_workspace,
        kind: str,
        payload: dict[str, Any],
    ):
        if kind == "s1_formalization_rejection_terminal_v67":
            raise KeyboardInterrupt("simulated loss before terminal projection")
        return original_commit(target_workspace, kind, payload)

    monkeypatch.setattr(
        service,
        "_commit_evidence_once",
        crash_before_terminal_projection,
    )
    with pytest.raises(KeyboardInterrupt):
        service.run_s1(task_id)

    assert workspace._latest_attempt("S1") == 2
    assert RecoveryKernelV60(workspace).load_state().human_required is True
    assert workspace._artifacts_of_kind("s1_formalization_rejection_terminal_v67") == []
    monkeypatch.setattr(
        service,
        "_commit_evidence_once",
        original_commit,
    )

    with pytest.raises(StudioValidationError, match="HUMAN_REQUIRED"):
        service.run_s1(task_id)

    assert workspace._latest_attempt("S1") == 2
    assert construction_count == 2
    assert (
        len(workspace._artifacts_of_kind("s1_formalization_rejection_terminal_v67"))
        == 1
    )
    terminal_events = [
        event
        for event in service.snapshot(task_id)["events"]
        if event["event_type"] == "s1_graph_recovery_stopped_v67"
    ]
    assert len(terminal_events) == 1
    assert workspace.verify()


class _V67FakeStudio(_FakeStudio):
    def __init__(self, task_root: Path) -> None:
        super().__init__(task_root)
        self.predata_v67 = False

    def _snapshot(self, task_id: str) -> dict[str, Any]:
        snapshot = super()._snapshot(task_id)
        snapshot["predata_v67_prepared"] = self.predata_v67
        return snapshot

    def prepare_predata_v67(
        self,
        task_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self._maybe_fail("prepare_predata_v67")
        assert request["contract_id"]
        assert self.s0 is True
        assert self.s1 is False
        self.predata_v67 = True
        return self._snapshot(task_id)


def test_campaign_uses_v67_order_when_service_exposes_predata_builder(
    tmp_path: Path,
) -> None:
    fake = _V67FakeStudio(tmp_path / "workspace")
    runner = RealLocalCampaignRunnerV67(
        tmp_path,
        authority_key=AUTHORITY_KEY,
        service_factory=_factory_for(fake),
    )
    runner.prepare(_campaign_spec_v67())

    receipt = runner.execute(execute_live=True)

    assert receipt.terminal_status == "COMPLETED_CONTROL"
    assert fake.calls == [
        "create_task",
        "run_s0",
        "prepare_predata_v67",
        "run_s1",
        "ingest_world_bank_data",
        "run_backhalf",
    ]
    assert receipt.completed_actions == fake.calls
    assert runner.spec_path.name == "campaign_spec_v67.json"
    assert runner.events_path.name == "campaign_events_v67.jsonl"
    assert receipt.schema_version == "6.7-real-local-campaign-terminal"
    assert runner.verify(require_real=False) is True


def test_campaign_predata_validation_requires_transaction_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fma.studio.service as service_module

    spec = _campaign_spec_v67()
    campaign_root = tmp_path / "campaign"
    runner = RealLocalCampaignRunnerV67(
        campaign_root,
        authority_key=AUTHORITY_KEY,
        authority_key_id="campaign-predata-test",
    )
    service = StudioTaskService(
        runner.workspace_root,
        authority_key=AUTHORITY_KEY,
        authority_key_id="campaign-predata-test",
        role_transport_factory=lambda _: FixtureStageRoleTransportV51(
            _ode_backhalf_draft
        ),
    )
    service.create_task(
        {
            "objective": spec.objective,
            "workspace_id": spec.task_id,
            "evidence_scope": "public_data",
            "workflow_mode": "v67",
        }
    )
    service.run_s0(spec.task_id)
    original_commit = service_module.StageWorkspaceV50.commit_evidence

    def interrupt_completion(self, kind, payload):
        if kind == "predata_preparation_completion_v67":
            raise KeyboardInterrupt("crash before transaction completion")
        return original_commit(self, kind, payload)

    monkeypatch.setattr(
        service_module.StageWorkspaceV50,
        "commit_evidence",
        interrupt_completion,
    )
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(
            spec.task_id,
            spec.world_bank_request,
        )
    monkeypatch.setattr(
        service_module.StageWorkspaceV50,
        "commit_evidence",
        original_commit,
    )

    pending = service.snapshot(spec.task_id)
    assert pending["predata_v67"]["transaction_status"] == "RECOVERY_PENDING"
    assert runner._validate_predata_workspace(spec, pending) == (
        "PARTIAL",
        None,
    )

    completed = service.reconcile_predata_v67(spec.task_id)
    status, binding_hash = runner._validate_predata_workspace(
        spec,
        completed,
    )
    assert status == "COMPLETE"
    assert isinstance(binding_hash, str) and len(binding_hash) == 64


def test_v67_runtime_contract_adds_six_action_budget_only() -> None:
    config = CodexCLIConfig(requested_model="gpt-test")
    legacy = _frozen_runtime_contract()
    contract = _runtime_contract_v67()

    assert CodexRuntimeBudgetsV65.from_config(config).campaign_action_limit == 5
    assert legacy.schema_version == "6.5-codex-runtime-contract"
    assert legacy.budgets.campaign_action_limit == 5
    assert contract.schema_version == "6.7-codex-runtime-contract"
    assert contract.budgets.campaign_action_limit == 6
    assert isinstance(contract.budgets, CodexRuntimeBudgetsV67)
    legacy_budgets = legacy.budgets.model_dump(mode="json")
    v67_budgets = contract.budgets.model_dump(mode="json")
    legacy_budgets.pop("campaign_action_limit")
    v67_budgets.pop("campaign_action_limit")
    assert v67_budgets == legacy_budgets

    common_exclusions = {"schema_version", "budgets", "contract_hash"}
    assert contract.model_dump(
        mode="json",
        exclude=common_exclusions,
    ) == legacy.model_dump(
        mode="json",
        exclude=common_exclusions,
    )

    real_v65 = _spec(
        runtime_contract=legacy,
        execution_mode="real",
    )
    payload = real_v65.model_dump(
        mode="json",
        exclude={
            "schema_version",
            "codex_runtime_contract",
            "spec_hash",
        },
    )
    with pytest.raises(ValueError):
        RealLocalCampaignSpecV67.seal(
            **payload,
            codex_runtime_contract=legacy,
        )


def test_v67_campaign_rejects_insufficient_predata_confirmation_budget() -> None:
    legacy = _spec()
    payload = legacy.model_dump(
        mode="json",
        exclude={"schema_version", "spec_hash"},
    )
    payload["world_bank_request"]["minimum_observations"] = 33

    with pytest.raises(
        ValueError,
        match="requires at least 34",
    ):
        RealLocalCampaignSpecV67.seal(**payload)


def test_v67_runner_verifies_v67_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _runtime_contract_v67()
    spec = _real_campaign_spec_v67()
    config = CodexCLIConfig(requested_model="gpt-test")
    executable = tmp_path / "codex.exe"
    original_file_hash = campaign_module._file_hash

    monkeypatch.setattr(
        campaign_module,
        "_trusted_runtime_adapter",
        lambda *_: (
            "openai_codex_cli",
            "native_codex_cli_v1",
            {
                "transport": "native_process",
                "process_runner": ("fma.codex_driver._default_process_runner"),
                "cli_locator": "fma.codex_driver.discover_codex_cli",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        campaign_module,
        "_locate_runtime_executable",
        lambda **_: executable,
    )
    monkeypatch.setattr(
        campaign_module,
        "_runtime_process_call",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"codex-cli {config.expected_cli_version}\n",
        ),
    )
    monkeypatch.setattr(
        campaign_module,
        "_file_hash",
        lambda path: (
            contract.executable_sha256
            if Path(path) == executable
            else original_file_hash(Path(path))
        ),
    )
    runner = RealLocalCampaignRunnerV67(
        tmp_path / "campaign",
        authority_key=AUTHORITY_KEY,
        codex_config=config,
    )

    built = build_codex_runtime_contract_v67(
        config=config,
        source_adapter_id=contract.source_adapter_id,
    )
    verified = runner._assert_runtime_contract(spec)

    assert built == contract
    assert verified == contract
    assert isinstance(verified, CodexRuntimeContractV67)
    assert verified.budgets.campaign_action_limit == 6


def test_runtime_contract_action_budget_must_match_runner_order(
    tmp_path: Path,
) -> None:
    class MismatchedRunner(RealLocalCampaignRunnerV67):
        @property
        def action_order(self) -> tuple[Any, ...]:
            return (*ACTION_ORDER_V67, "unexpected_action")

    runner = MismatchedRunner(
        tmp_path,
        authority_key=AUTHORITY_KEY,
        codex_config=CodexCLIConfig(requested_model="gpt-test"),
    )

    with pytest.raises(
        campaign_module.RealLocalCampaignError,
        match="action budget differs",
    ):
        runner._assert_runtime_contract(_real_campaign_spec_v67())


def test_v65_schema_rejects_v67_action_and_keeps_five_step_hash() -> None:
    legacy = _spec()

    assert legacy.spec_hash == (
        "c778518b91de8286d09064fe098ec306e1f3e9646f4501a8778eb30767b0369f"
    )
    with pytest.raises(ValueError):
        RealLocalCampaignEventV65.seal(
            campaign_id=legacy.campaign_id,
            sequence=1,
            event_type="ACTION_INTENT",
            status="INTENT_RECORDED",
            action_id="action-invalid-v67",
            action="prepare_predata_v67",
            request_hash="a" * 64,
            recorded_at="2026-07-28T00:00:00Z",
        )


def test_v65_cli_defaults_and_authority_paths_remain_unchanged() -> None:
    args = real_local_campaign_cli._parser().parse_args(
        [
            "status",
            "--campaign-root",
            "campaign",
            "--authority-key-file",
            "authority.key",
        ]
    )

    assert args.authority_key_id == "real-local-v65"
    assert SPEC_PATH_V65 == "campaign_spec_v65.json"
    assert EVENTS_PATH_V65 == "campaign_events_v65.jsonl"
    assert SPEC_PATH_V67 == "campaign_spec_v67.json"
    assert EVENTS_PATH_V67 == "campaign_events_v67.jsonl"


def test_v67_cli_freezes_v67_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key_path = tmp_path / "authority.key"
    key_path.write_bytes(AUTHORITY_KEY)
    spec_path = tmp_path / "unsealed-spec-v67.json"
    payload = _real_campaign_spec_v67().model_dump(
        mode="json",
        exclude={"codex_runtime_contract", "spec_hash"},
    )
    spec_path.write_text(
        canonical_json(payload) + "\n",
        encoding="utf-8",
    )
    campaign_root = tmp_path / "campaign"
    contract = _runtime_contract_v67()
    observed: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> CodexRuntimeContractV67:
        observed.update(kwargs)
        return contract

    monkeypatch.setattr(
        real_local_campaign_v67_cli,
        "build_codex_runtime_contract_v67",
        fake_build,
    )

    code = real_local_campaign_v67_cli.main(
        [
            "prepare",
            "--campaign-root",
            str(campaign_root),
            "--spec-file",
            str(spec_path),
            "--authority-key-file",
            str(key_path),
            "--freeze-codex-runtime",
            "--model",
            "gpt-test",
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["prepared"] is True
    assert (
        observed["source_adapter_id"] == (payload["world_bank_request"]["adapter_id"])
    )
    assert observed["config"].requested_model == "gpt-test"
    frozen = RealLocalCampaignSpecV67.model_validate_json(
        (campaign_root / SPEC_PATH_V67).read_text(encoding="utf-8")
    )
    assert isinstance(
        frozen.codex_runtime_contract,
        CodexRuntimeContractV67,
    )
    assert frozen.codex_runtime_contract.budgets.campaign_action_limit == 6


def test_v67_cli_prepare_status_and_control_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    key_path = tmp_path / "authority.key"
    key_path.write_bytes(AUTHORITY_KEY)
    spec_path = tmp_path / "spec-v67.json"
    spec_path.write_text(
        canonical_json(_campaign_spec_v67().model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    prepared_root = tmp_path / "prepared"

    prepare_code = real_local_campaign_v67_cli.main(
        [
            "prepare",
            "--campaign-root",
            str(prepared_root),
            "--spec-file",
            str(spec_path),
            "--authority-key-file",
            str(key_path),
        ]
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepare_code == 0
    assert prepared["schema_version"] == "6.7-real-local-campaign-status"
    assert prepared["action_order"][2] == "prepare_predata_v67"
    assert (prepared_root / SPEC_PATH_V67).is_file()
    assert not (prepared_root / SPEC_PATH_V65).exists()

    status_code = real_local_campaign_v67_cli.main(
        [
            "status",
            "--campaign-root",
            str(prepared_root),
            "--authority-key-file",
            str(key_path),
        ]
    )
    status = json.loads(capsys.readouterr().out)
    assert status_code == 0
    assert status["schema_version"] == "6.7-real-local-campaign-status"
    assert status["prepared"] is True

    control_root = tmp_path / "control"
    fake = _V67FakeStudio(control_root / "workspace")
    control_runner = RealLocalCampaignRunnerV67(
        control_root,
        authority_key=AUTHORITY_KEY,
        service_factory=_factory_for(fake),
    )
    control_runner.prepare(_campaign_spec_v67())
    control_runner.execute(execute_live=True)
    monkeypatch.setattr(
        real_local_campaign_v67_cli,
        "_runner_from_args",
        lambda _: control_runner,
    )

    verify_code = real_local_campaign_v67_cli.main(
        [
            "verify",
            "--campaign-root",
            str(control_root),
            "--authority-key-file",
            str(key_path),
            "--allow-control",
        ]
    )
    verification = json.loads(capsys.readouterr().out)
    assert verify_code == 0
    assert verification["schema_version"] == ("6.7-real-local-campaign-verification")
    assert verification["verification_scope"] == "control_protocol_only"
    assert verification["verified"] is True
