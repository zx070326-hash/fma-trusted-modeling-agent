from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fma.hashing import sha256_value
from fma.v5.workspace_schemas import (
    DataLedgerEntryV50,
    DataLedgerV50,
    FileBindingV50,
    RawDataBaselineV50,
    TaskWorkspaceSpecV50,
    WorkflowProfileV50,
)
from fma.v6.provenance import (
    MeasurementSchemaV62,
    S2TransformReceiptV62,
    build_data_provenance_binding_v62,
)
from fma.v6.public_source import (
    SOURCE_RAW_PATH,
    SourceHTTPResponseV62,
    WorldBankSourceContractV62,
    materialize_world_bank_series_v62,
)
from fma.v6.source_auth import (
    SourceAcquisitionReceiptV62,
    SourceTransportAuthorityV62,
)


NOW = datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc)
SOURCE_AUTHORITY_SECRET = b"provenance-source-authority-" + b"k" * 32


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _body() -> bytes:
    records = [
        {
            "indicator": {
                "id": "SP.POP.TOTL",
                "value": "Population, total",
            },
            "country": {"id": "NZ", "value": "New Zealand"},
            "countryiso3code": "NZL",
            "date": str(year),
            "value": float(3_000_000 + (year - 2000) * 50_000),
            "unit": "",
            "obs_status": "",
            "decimal": 0,
        }
        for year in range(2023, 1999, -1)
    ]
    metadata = {
        "page": 1,
        "pages": 1,
        "per_page": 1000,
        "total": len(records),
        "sourceid": "2",
        "lastupdated": "2026-07-13",
    }
    return json.dumps([metadata, records], separators=(",", ":")).encode(
        "utf-8"
    )


def _workspace_spec() -> TaskWorkspaceSpecV50:
    return TaskWorkspaceSpecV50.seal(
        workspace_id="fixture-provenance",
        graph_id="v5-fixture-provenance",
        objective="Bind authenticated source evidence into current S2.",
        mission_hash=sha256_value({"mission": "provenance-v62"}),
        evidence_snapshot_hash=sha256_value({"evidence": "fixture-control"}),
        evaluator_epoch="provenance-v62",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="synthetic_fixture",
    )


def _fixture_source(
    tmp_path: Path,
    *,
    spec: TaskWorkspaceSpecV50,
    authority: SourceTransportAuthorityV62,
):
    contract = WorldBankSourceContractV62.seal(
        contract_id="fixture-nzl-population",
        country_code="NZL",
        indicator_id="SP.POP.TOTL",
        start_year=2000,
        end_year=2023,
        minimum_observations=23,
        state_unit="persons",
        fixture_only=True,
        attribution=(
            "World Bank, World Development Indicators, SP.POP.TOTL, "
            "used in a local fixture control."
        ),
    )
    acquisition = authority.acquire_world_bank_series(
        workspace_spec=spec,
        task_id=spec.workspace_id,
        contract=contract,
        fetcher=lambda item: SourceHTTPResponseV62(
            status=200,
            final_url=item.exact_url,
            content_type="application/json",
            body=_body(),
        ),
        retrieved_at=NOW,
        authenticated_at=NOW,
    )
    materialize_world_bank_series_v62(
        workspace_root=tmp_path,
        fetched=acquisition.fetched,
    )
    return acquisition


def _binding_inputs(tmp_path: Path):
    spec = _workspace_spec()
    source_authority = SourceTransportAuthorityV62(
        key_id="provenance-source-key",
        secret=SOURCE_AUTHORITY_SECRET,
    )
    acquisition = _fixture_source(
        tmp_path,
        spec=spec,
        authority=source_authority,
    )
    fetched = acquisition.fetched
    raw_path = tmp_path / "data" / "raw" / "ode_series.json"
    _write_json(
        raw_path,
        {
            "times": fetched.snapshot.times,
            "observations": fetched.snapshot.observations,
            "source_id": fetched.snapshot.source_id,
            "fixture_only": True,
        },
    )
    processed_path = tmp_path / "data" / "processed" / "ode_snapshot.json"
    _write_json(processed_path, fetched.snapshot.model_dump(mode="json"))
    transform_path = tmp_path / "src" / "models" / "prepare_ode_data.py"
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.write_text(
        "def transform(payload: dict) -> dict:\n    return payload\n",
        encoding="utf-8",
        newline="\n",
    )
    raw_tree_hash = sha256_value(
        {"data/raw/ode_series.json": _sha(raw_path)}
    )
    binding = FileBindingV50(
        relative_path="data/raw/ode_series.json",
        sha256=_sha(raw_path),
        size_bytes=raw_path.stat().st_size,
        snapshot_artifact_hash="d" * 64,
    )
    unsigned = RawDataBaselineV50(
        workspace_spec_hash=spec.spec_hash,
        s1_gate_hash="b" * 64,
        s2_attempt=1,
        raw_tree_hash=raw_tree_hash,
        files=[binding],
        frozen_at=NOW,
        authority_key_id="fixture-authority",
    )
    baseline_payload = unsigned.model_dump(mode="json")
    baseline_payload["authority_auth_tag"] = "c" * 64
    baseline_payload["baseline_hash"] = sha256_value(
        {
            key: value
            for key, value in baseline_payload.items()
            if key != "baseline_hash"
        }
    )
    baseline = RawDataBaselineV50.model_validate(baseline_payload)
    transform_params = {
        "adapter_id": "scalar_autonomous_ode_v52",
        "identity_transform": True,
        "drop_missing": False,
    }
    ledger = DataLedgerV50.seal(
        raw_baseline_tree_hash=raw_tree_hash,
        entries=[
            DataLedgerEntryV50(
                data_item_id="population_series",
                semantic_name="annual total population",
                units="persons",
                source_kind="official",
                source_ref=fetched.receipt.source_id,
                raw_relative_path="data/raw/ode_series.json",
                accessed_at=NOW + timedelta(seconds=1),
                license_status="recorded_not_independently_reviewed",
                raw_response_hash=_sha(raw_path),
                transform_script_relative_path=(
                    "src/models/prepare_ode_data.py"
                ),
                transform_script_hash=_sha(transform_path),
                transform_params=transform_params,
                transform_params_hash=sha256_value(transform_params),
                processed_artifact_hash=_sha(processed_path),
                quality_flags=["fixture_control"],
            )
        ],
    )
    measurement = MeasurementSchemaV62.seal(
        measurement_id="annual_total_population",
        source_contract_hash=fetched.contract.contract_hash,
        indicator_id=fetched.contract.indicator_id,
        semantic_name="Population, total",
        operational_definition=(
            "Annual total population value represented by the registered "
            "World Bank indicator in this fixture."
        ),
        observation_time_basis="calendar year",
        aggregation_level="country total",
        time_unit="year",
        state_unit="persons",
    )
    transform_receipt = S2TransformReceiptV62.seal(
        workspace_spec_hash=spec.spec_hash,
        raw_baseline_hash=baseline.baseline_hash,
        s2_attempt=1,
        task_id=fetched.snapshot.task_id,
        input_relative_path="data/raw/ode_series.json",
        input_hash=_sha(raw_path),
        transform_relative_path="src/models/prepare_ode_data.py",
        transform_hash=_sha(transform_path),
        output_relative_path="data/processed/ode_snapshot.json",
        output_hash=_sha(processed_path),
        command=["python", "transform.py", "input.json", "output.json"],
        runtime_identity="fixture python runtime",
        stdout_hash=hashlib.sha256(b"").hexdigest(),
        stderr_hash=hashlib.sha256(b"").hexdigest(),
        started_at=NOW,
        finished_at=NOW + timedelta(milliseconds=1),
    )

    class WorkspaceStub:
        def __init__(self) -> None:
            self.root = tmp_path
            self.spec = spec

        @staticmethod
        def current_gate(stage: str):
            return "b" * 64 if stage == "S1" else None

        @staticmethod
        def verify_raw_baseline(candidate: RawDataBaselineV50) -> bool:
            return candidate.baseline_hash == baseline.baseline_hash

        @staticmethod
        def _latest_attempt(stage: str) -> int:
            return 1

        @staticmethod
        def _raw_baseline_for_current_s2() -> RawDataBaselineV50:
            return baseline

    workspace = WorkspaceStub()
    reverified = source_authority.reverify_world_bank_source_at_s2(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        reverified_at=NOW + timedelta(seconds=2),
    )
    return SimpleNamespace(
        workspace=workspace,
        baseline=baseline,
        ledger=ledger,
        fetched=fetched,
        verification=reverified.verification,
        source_acquisition_receipt=acquisition.authority_receipt,
        s2_source_reverification_receipt=reverified.authority_receipt,
        source_authority=source_authority,
        measurement=measurement,
        transform_receipt=transform_receipt,
    )


def test_fixture_source_binds_mechanically_but_not_scientifically(
    tmp_path: Path,
) -> None:
    inputs = _binding_inputs(tmp_path)
    result = build_data_provenance_binding_v62(
        workspace=inputs.workspace,
        raw_baseline=inputs.baseline,
        ledger=inputs.ledger,
        snapshot=inputs.fetched.snapshot,
        source_contract=inputs.fetched.contract,
        source_receipt=inputs.fetched.receipt,
        source_verification=inputs.verification,
        source_acquisition_receipt=inputs.source_acquisition_receipt,
        s2_source_reverification_receipt=(
            inputs.s2_source_reverification_receipt
        ),
        source_authority=inputs.source_authority,
        measurement_schema=inputs.measurement,
        transform_receipt=inputs.transform_receipt,
    )

    assert result.status == "PASS"
    assert all(result.checks.values())
    assert result.scientific_provenance_status == "NOT_RUN"
    assert result.fixture_only is True
    assert result.source_acquisition_authority_receipt_hash == (
        inputs.source_acquisition_receipt.receipt_hash
    )
    assert result.s2_source_reverification_receipt_hash == (
        inputs.s2_source_reverification_receipt.receipt_hash
    )
    assert result.official_live_transport_authenticated is False
    assert result.independent_measurement_review is False
    assert result.scientific_qualification_granted is False
    assert result.real_world_action_authorized is False


def test_source_mutation_breaks_s2_provenance_binding(
    tmp_path: Path,
) -> None:
    inputs = _binding_inputs(tmp_path)
    source_path = tmp_path / SOURCE_RAW_PATH
    source_path.write_bytes(source_path.read_bytes() + b"\n")

    result = build_data_provenance_binding_v62(
        workspace=inputs.workspace,
        raw_baseline=inputs.baseline,
        ledger=inputs.ledger,
        snapshot=inputs.fetched.snapshot,
        source_contract=inputs.fetched.contract,
        source_receipt=inputs.fetched.receipt,
        source_verification=inputs.verification,
        source_acquisition_receipt=inputs.source_acquisition_receipt,
        s2_source_reverification_receipt=(
            inputs.s2_source_reverification_receipt
        ),
        source_authority=inputs.source_authority,
        measurement_schema=inputs.measurement,
        transform_receipt=inputs.transform_receipt,
    )

    assert result.status == "FAIL"
    assert (
        result.checks["source_acquisition_authority_authenticated"] is False
    )
    assert (
        result.checks["current_s2_source_reverification_authenticated"]
        is False
    )
    assert result.checks["source_integrity_replay_passed"] is False
    assert result.scientific_provenance_status == "FAIL"


def test_measurement_unit_mismatch_fails_binding(tmp_path: Path) -> None:
    inputs = _binding_inputs(tmp_path)
    wrong_measurement = MeasurementSchemaV62.seal(
        **inputs.measurement.model_dump(
            exclude={"schema_hash", "state_unit"}
        ),
        state_unit="thousands_of_persons",
    )

    result = build_data_provenance_binding_v62(
        workspace=inputs.workspace,
        raw_baseline=inputs.baseline,
        ledger=inputs.ledger,
        snapshot=inputs.fetched.snapshot,
        source_contract=inputs.fetched.contract,
        source_receipt=inputs.fetched.receipt,
        source_verification=inputs.verification,
        source_acquisition_receipt=inputs.source_acquisition_receipt,
        s2_source_reverification_receipt=(
            inputs.s2_source_reverification_receipt
        ),
        source_authority=inputs.source_authority,
        measurement_schema=wrong_measurement,
        transform_receipt=inputs.transform_receipt,
    )

    assert result.status == "FAIL"
    assert result.checks["measurement_schema_bound"] is False


def test_unsigned_acquisition_cannot_satisfy_provenance(
    tmp_path: Path,
) -> None:
    inputs = _binding_inputs(tmp_path)
    unsigned_acquisition = SourceAcquisitionReceiptV62.model_validate(
        inputs.source_acquisition_receipt.model_dump(
            mode="json",
            exclude={"authority_auth_tag", "receipt_hash"},
        )
    )

    result = build_data_provenance_binding_v62(
        workspace=inputs.workspace,
        raw_baseline=inputs.baseline,
        ledger=inputs.ledger,
        snapshot=inputs.fetched.snapshot,
        source_contract=inputs.fetched.contract,
        source_receipt=inputs.fetched.receipt,
        source_verification=inputs.verification,
        source_acquisition_receipt=unsigned_acquisition,
        s2_source_reverification_receipt=(
            inputs.s2_source_reverification_receipt
        ),
        source_authority=inputs.source_authority,
        measurement_schema=inputs.measurement,
        transform_receipt=inputs.transform_receipt,
    )

    assert result.status == "FAIL"
    assert (
        result.checks["source_acquisition_authority_authenticated"] is False
    )
    assert (
        result.checks["current_s2_source_reverification_authenticated"]
        is False
    )
    assert result.scientific_provenance_status == "FAIL"
