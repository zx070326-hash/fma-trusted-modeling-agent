from __future__ import annotations

import json
from pathlib import Path

import pytest

from fma.studio.backhalf_runtime import (
    ADAPTER_BINDING_PATH,
    EXECUTABLE_CANDIDATE_RESOLUTION_PATH_V67,
    RAW_RELATIVE_PATH,
    ExecutableCandidateResolutionV67,
    StudioODEDataRequestV59,
    V67S2CompatibilityError,
    _effective_adapter_id,
    ingest_ode_data_v59,
    load_v67_s2_contract_v67,
    validate_v67_data_compatibility_v67,
    validate_v67_pre_acquisition_v67,
)
from fma.v6.measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    MeasurementStudyDesignContractV67,
)
from fma.v6.predata_protocol import (
    CANDIDATE_EXECUTION_BINDING_PATH_V67,
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    compile_predata_execution_protocol_v67,
    registered_positive_series_capability_pack_v67,
)
from fma.v6.provenance import (
    MEASUREMENT_SCHEMA_PATH,
    MeasurementSchemaV62,
)
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    SOURCE_RAW_PATH,
    SOURCE_RECEIPT_PATH,
    SourceHTTPResponseV62,
    WorldBankSourceContractV62,
)
from fma.v6.source_auth import (
    SOURCE_ACQUISITION_AUTH_PATH,
    SourceTransportAuthorityV62,
)
from tests.test_v6_7_s1_runtime_integration import (
    _orchestrator,
    _predata_artifacts,
    _write_model,
)


def _fixture_v67_workspace(tmp_path: Path, *, task_id: str):
    service, workspace, source, contract, _, _ = _predata_artifacts(
        tmp_path,
        task_id=task_id,
    )
    source_payload = source.model_dump(
        mode="json",
        exclude={"contract_hash"},
    )
    source_payload["fixture_only"] = True
    fixture_source = WorldBankSourceContractV62.seal(**source_payload)
    measurement_payload = contract.model_dump(
        mode="json",
        exclude={"contract_hash"},
    )
    measurement_payload["source_contract_id"] = fixture_source.contract_id
    measurement_payload["source_contract_hash"] = fixture_source.contract_hash
    fixture_measurement = MeasurementStudyDesignContractV67.seal(
        **measurement_payload
    )
    protocol = compile_predata_execution_protocol_v67(
        measurement_contract=fixture_measurement,
        capability_pack=registered_positive_series_capability_pack_v67(
            "scalar_autonomous_ode_v52"
        ),
    )
    _write_model(workspace.root / SOURCE_CONTRACT_PATH, fixture_source)
    _write_model(
        workspace.root / MEASUREMENT_STUDY_DESIGN_PATH_V67,
        fixture_measurement,
    )
    _write_model(
        workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67,
        protocol,
    )
    result = _orchestrator(
        service,
        workspace,
        fixture_measurement,
        protocol,
    ).run()
    assert result["gate_decision"] == "OPEN"
    return (
        service,
        workspace,
        fixture_source,
        fixture_measurement,
        protocol,
    )


def _fixture_response(
    contract: WorldBankSourceContractV62,
) -> SourceHTTPResponseV62:
    records = [
        {
            "indicator": {
                "id": contract.indicator_id,
                "value": "GDP per capita (constant 2015 US$)",
            },
            "country": {"id": "BR", "value": "Brazil"},
            "countryiso3code": contract.country_code,
            "date": str(year),
            "value": 10000.0 + 175.0 * (year - contract.start_year),
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        }
        for year in range(contract.end_year, contract.start_year - 1, -1)
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


def _acquired_request(
    workspace,
    source: WorldBankSourceContractV62,
    measurement: MeasurementStudyDesignContractV67,
):
    authority = SourceTransportAuthorityV62.from_stage_workspace(workspace)
    acquisition = authority.acquire_world_bank_series(
        workspace_spec=workspace.spec,
        task_id=workspace.spec.workspace_id,
        contract=source,
        fetcher=_fixture_response,
    )
    fetched = acquisition.fetched
    schema = MeasurementSchemaV62.seal(
        measurement_id=measurement.measurement.measurement_id,
        source_contract_hash=source.contract_hash,
        indicator_id=source.indicator_id,
        semantic_name=measurement.measurement.measurement_id,
        operational_definition=(
            measurement.measurement.operational_definition
        ),
        observation_time_basis=measurement.measurement.time_basis,
        aggregation_level=measurement.measurement.aggregation_basis,
        time_unit=fetched.snapshot.time_unit,
        state_unit=fetched.snapshot.state_unit,
    )
    request = StudioODEDataRequestV59(
        adapter_id="scalar_autonomous_ode_v52",
        time_unit=fetched.snapshot.time_unit,
        state_unit=fetched.snapshot.state_unit,
        times=fetched.snapshot.times,
        observations=fetched.snapshot.observations,
        source_id=fetched.receipt.source_id,
        license_status="world-bank-fixture-control",
        fixture_only=True,
    )
    return acquisition, schema, request


def _materialize_source_evidence(
    workspace,
    acquisition,
    measurement_schema: MeasurementSchemaV62,
) -> None:
    raw_path = workspace.root / SOURCE_RAW_PATH
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(acquisition.fetched.raw_body)
    _write_model(
        workspace.root / SOURCE_RECEIPT_PATH,
        acquisition.fetched.receipt,
    )
    _write_model(
        workspace.root / SOURCE_ACQUISITION_AUTH_PATH,
        acquisition.authority_receipt,
    )
    _write_model(
        workspace.root / MEASUREMENT_SCHEMA_PATH,
        measurement_schema,
    )


def test_v67_pre_acquisition_mismatch_is_typed_and_precedes_raw_write(
    tmp_path: Path,
) -> None:
    _, workspace, source, measurement, protocol = _fixture_v67_workspace(
        tmp_path,
        task_id="v67-pre-acquisition-reject",
    )

    with pytest.raises(V67S2CompatibilityError) as captured:
        validate_v67_pre_acquisition_v67(
            workspace,
            adapter_id="adaptive_positive_series_v57",
            source_contract=source,
            measurement_unit=measurement.measurement.unit,
            time_basis=measurement.measurement.time_basis,
            missing_value_policy="reject_incomplete_series",
        )

    evidence = captured.value.evidence
    assert evidence.failure_owner == "capability"
    assert evidence.compatibility_phase == "pre_acquisition"
    assert evidence.reason_codes == ["adapter_matches_frozen_protocol"]
    assert evidence.predata_protocol_hash == protocol.protocol_hash
    assert evidence.raw_ode_data_written_by_backhalf is False
    assert not (workspace.root / RAW_RELATIVE_PATH).exists()


def test_v67_never_reads_recovery_state_or_silently_switches_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, workspace, source, measurement, _ = _fixture_v67_workspace(
        tmp_path,
        task_id="v67-no-silent-switch",
    )
    acquisition, schema, request = _acquired_request(
        workspace,
        source,
        measurement,
    )
    _materialize_source_evidence(workspace, acquisition, schema)

    def forbidden_recovery_read(*_args, **_kwargs):
        raise AssertionError("V6.7 adapter selection read recovery state")

    monkeypatch.setattr(
        "fma.studio.backhalf_runtime.RecoveryKernelV60.load_state",
        forbidden_recovery_read,
    )

    assert _effective_adapter_id(workspace, request) == (
        "scalar_autonomous_ode_v52"
    )
    changed = request.model_copy(
        update={"adapter_id": "adaptive_positive_series_v57"}
    )
    with pytest.raises(V67S2CompatibilityError) as captured:
        ingest_ode_data_v59(workspace, changed)
    assert captured.value.evidence.failure_owner == "capability"
    assert "adapter_matches_frozen_protocol" in (
        captured.value.evidence.reason_codes
    )
    assert not (workspace.root / RAW_RELATIVE_PATH).exists()


def test_v67_tampered_protocol_is_rejected_as_stale_current_lineage(
    tmp_path: Path,
) -> None:
    _, workspace, _, _, _ = _fixture_v67_workspace(
        tmp_path,
        task_id="v67-stale-protocol",
    )
    protocol_path = workspace.root / PREDATA_EXECUTION_PROTOCOL_PATH_V67
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["protocol_hash"] = "f" * 64
    protocol_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(V67S2CompatibilityError) as captured:
        load_v67_s2_contract_v67(workspace)

    assert captured.value.evidence.compatibility_phase == "artifact_replay"
    assert captured.value.evidence.failure_owner == "capability"
    assert workspace.current_gate("S1") is None
    assert not (workspace.root / RAW_RELATIVE_PATH).exists()


def test_v67_s2_binds_protocol_resolution_and_manifest_hashes(
    tmp_path: Path,
) -> None:
    service, workspace, source, measurement, protocol = (
        _fixture_v67_workspace(
            tmp_path,
            task_id="v67-s2-binding",
        )
    )
    acquisition, schema, request = _acquired_request(
        workspace,
        source,
        measurement,
    )

    context = validate_v67_data_compatibility_v67(
        workspace,
        request,
        source_contract=source,
        source_receipt=acquisition.fetched.receipt,
        source_acquisition_receipt=acquisition.authority_receipt,
        measurement_schema=schema,
        source_raw_body=acquisition.fetched.raw_body,
        require_source_evidence=True,
    )
    assert context is not None
    _materialize_source_evidence(workspace, acquisition, schema)
    ingest_ode_data_v59(workspace, request)

    decision = service._backhalf_orchestrator(
        workspace.spec.workspace_id,
        workspace,
    ).run_s2()

    assert decision == "OPEN"
    binding = json.loads(
        (workspace.root / ADAPTER_BINDING_PATH).read_text(encoding="utf-8")
    )
    resolution = ExecutableCandidateResolutionV67.model_validate_json(
        (
            workspace.root
            / EXECUTABLE_CANDIDATE_RESOLUTION_PATH_V67
        ).read_text(encoding="utf-8")
    )
    assert binding["schema_version"] == "6.7"
    assert binding["adapter_id"] == protocol.adapter_binding.adapter_id
    assert binding["predata_protocol_hash"] == protocol.protocol_hash
    assert binding["measurement_contract_hash"] == measurement.contract_hash
    assert binding["source_contract_hash"] == source.contract_hash
    assert binding["silent_adapter_substitution_permitted"] is False
    assert resolution.predata_protocol_hash == protocol.protocol_hash
    assert resolution.measurement_contract_hash == measurement.contract_hash
    assert resolution.source_contract_hash == source.contract_hash
    assert resolution.candidate_execution_binding_hash == (
        context.candidate_binding.binding_hash
    )
    assert resolution.adapter_id == protocol.adapter_binding.adapter_id
    manifest_paths = {
        item.relative_path
        for item in workspace._manifest_for_stage("S2").files
    }
    assert {
        SOURCE_CONTRACT_PATH,
        MEASUREMENT_STUDY_DESIGN_PATH_V67,
        PREDATA_EXECUTION_PROTOCOL_PATH_V67,
        CANDIDATE_EXECUTION_BINDING_PATH_V67,
        EXECUTABLE_CANDIDATE_RESOLUTION_PATH_V67,
        ADAPTER_BINDING_PATH,
    } <= manifest_paths
