from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import fma.v6.public_source as public_source_module
from fma.hashing import sha256_value
from fma.v5.workspace_schemas import (
    RawDataBaselineV50,
    TaskWorkspaceSpecV50,
    WorkflowProfileV50,
)
from fma.v6.source_auth import (
    SourceAcquisitionReceiptV62,
    SourceTransportAuthorityV62,
)
from fma.v6.public_source import (
    SOURCE_RAW_PATH,
    SourceHTTPResponseV62,
    WorldBankSourceContractV62,
    fetch_world_bank_series_v62,
    materialize_world_bank_series_v62,
    verify_world_bank_source_v62,
)


NOW = datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc)
AUTHORITY_SECRET = b"source-authority-test-key-" + b"k" * 32


def _contract() -> WorldBankSourceContractV62:
    return WorldBankSourceContractV62.seal(
        contract_id="chn-population-2000-2023",
        country_code="CHN",
        indicator_id="SP.POP.TOTL",
        start_year=2000,
        end_year=2023,
        minimum_observations=23,
        state_unit="persons",
        fixture_only=True,
        attribution=(
            "World Bank, World Development Indicators, SP.POP.TOTL, "
            "accessed through the World Bank Indicators API."
        ),
    )


def _body(*, omit_year: int | None = None) -> bytes:
    records = []
    for year in range(2023, 1999, -1):
        if year == omit_year:
            continue
        records.append(
            {
                "indicator": {
                    "id": "SP.POP.TOTL",
                    "value": "Population, total",
                },
                "country": {"id": "CN", "value": "China"},
                "countryiso3code": "CHN",
                "date": str(year),
                "value": float(1_200_000_000 + (year - 2000) * 5_000_000),
                "unit": "",
                "obs_status": "",
                "decimal": 0,
            }
        )
    metadata = {
        "page": 1,
        "pages": 1,
        "per_page": 1000,
        "total": len(records),
        "sourceid": "2",
        "lastupdated": "2026-07-13",
    }
    return json.dumps([metadata, records], separators=(",", ":")).encode("utf-8")


def _response(
    contract: WorldBankSourceContractV62,
    *,
    body: bytes | None = None,
    final_url: str | None = None,
) -> SourceHTTPResponseV62:
    return SourceHTTPResponseV62(
        status=200,
        final_url=final_url or contract.exact_url,
        content_type="application/json",
        body=body or _body(),
    )


def _spec(
    workspace_id: str = "source-auth",
    *,
    evidence_scope: str = "synthetic_fixture",
) -> TaskWorkspaceSpecV50:
    return TaskWorkspaceSpecV50.seal(
        workspace_id=workspace_id,
        graph_id=f"v5-{workspace_id}",
        objective="Authenticate and reverify one bounded source series.",
        mission_hash=sha256_value(
            {"workspace_id": workspace_id, "mission": "source-auth-test"}
        ),
        evidence_snapshot_hash=sha256_value(
            {"workspace_id": workspace_id, "evidence": "public-only"}
        ),
        evaluator_epoch="source-auth-v62",
        profile=WorkflowProfileV50.seal(),
        evidence_scope=evidence_scope,
    )


def _baseline(
    spec: TaskWorkspaceSpecV50,
    *,
    attempt: int = 1,
    s1_gate_hash: str = "1" * 64,
) -> RawDataBaselineV50:
    draft = RawDataBaselineV50(
        workspace_spec_hash=spec.spec_hash,
        s1_gate_hash=s1_gate_hash,
        s2_attempt=attempt,
        raw_tree_hash=sha256_value({}),
        files=[],
        frozen_at=NOW + timedelta(minutes=1),
        authority_key_id="source-test-key",
    )
    payload = draft.model_dump(mode="json")
    payload["baseline_hash"] = sha256_value(
        {
            key: value
            for key, value in payload.items()
            if key != "baseline_hash"
        }
    )
    return RawDataBaselineV50.model_validate(payload)


class _WorkspaceStub:
    def __init__(
        self,
        root: Path,
        spec: TaskWorkspaceSpecV50,
        baseline: RawDataBaselineV50,
    ) -> None:
        self.root = root.resolve()
        self.spec = spec
        self.authority_key_id = "source-test-key"
        self._authority_key = AUTHORITY_SECRET
        self.baseline = baseline
        self.s1_gate_hash = baseline.s1_gate_hash
        self.attempt = baseline.s2_attempt

    def current_gate(self, stage: str) -> str | None:
        return self.s1_gate_hash if stage == "S1" else None

    def _latest_attempt(self, stage: str) -> int:
        assert stage == "S2"
        return self.attempt

    def _raw_baseline_for_current_s2(self) -> RawDataBaselineV50:
        return self.baseline

    def verify_raw_baseline(self, baseline: RawDataBaselineV50) -> bool:
        return bool(
            baseline.baseline_hash == self.baseline.baseline_hash
            and baseline.workspace_spec_hash == self.spec.spec_hash
        )


def _authenticated_fixture(
    tmp_path: Path,
) -> tuple[
    SourceTransportAuthorityV62,
    TaskWorkspaceSpecV50,
    object,
]:
    authority = SourceTransportAuthorityV62(
        key_id="source-test-key",
        secret=AUTHORITY_SECRET,
    )
    spec = _spec()
    acquisition = authority.acquire_world_bank_series(
        workspace_spec=spec,
        task_id=spec.workspace_id,
        contract=_contract(),
        fetcher=lambda item: _response(item),
        retrieved_at=NOW,
        authenticated_at=NOW,
    )
    materialize_world_bank_series_v62(
        workspace_root=tmp_path,
        fetched=acquisition.fetched,
    )
    return authority, spec, acquisition


def test_world_bank_source_fetch_materialize_and_replay(
    tmp_path: Path,
) -> None:
    contract = _contract()
    fetched = fetch_world_bank_series_v62(
        task_id="official-population",
        contract=contract,
        fetcher=lambda item: _response(item),
        retrieved_at=NOW,
    )
    materialize_world_bank_series_v62(
        workspace_root=tmp_path,
        fetched=fetched,
    )
    verification = verify_world_bank_source_v62(
        workspace_root=tmp_path,
        contract=fetched.contract,
        receipt=fetched.receipt,
        snapshot=fetched.snapshot,
    )

    assert verification.status == "PASS"
    assert all(verification.checks.values())
    assert fetched.snapshot.fixture_only is True
    assert fetched.receipt.fixture_only is True
    assert verification.evidence_scope == "fixture_control_integrity"
    assert verification.scientific_provenance_status == "NOT_RUN"
    assert fetched.receipt.observation_count == 24
    assert fetched.receipt.first_year == 2000
    assert fetched.receipt.last_year == 2023
    assert verification.scientific_qualification_granted is False
    assert verification.real_world_action_authorized is False


def test_injected_fetcher_cannot_masquerade_as_live_official_source() -> None:
    contract = WorldBankSourceContractV62.seal(
        **_contract().model_dump(
            exclude={"contract_hash", "fixture_only"}
        ),
        fixture_only=False,
    )
    with pytest.raises(ValueError, match="requires fixture_only=true"):
        fetch_world_bank_series_v62(
            task_id="false-official-source",
            contract=contract,
            fetcher=lambda item: _response(item),
            retrieved_at=NOW,
        )


def test_world_bank_source_rejects_redirect() -> None:
    contract = _contract()
    with pytest.raises(ValueError, match="redirected"):
        fetch_world_bank_series_v62(
            task_id="redirected-source",
            contract=contract,
            fetcher=lambda item: _response(
                item,
                final_url="https://example.invalid/copied.json",
            ),
            retrieved_at=NOW,
        )


def test_world_bank_source_rejects_missing_year() -> None:
    contract = _contract()
    with pytest.raises(ValueError, match="missing or unexpected years"):
        fetch_world_bank_series_v62(
            task_id="missing-year",
            contract=contract,
            fetcher=lambda item: _response(item, body=_body(omit_year=2013)),
            retrieved_at=NOW,
        )


def test_world_bank_source_replay_detects_raw_mutation(
    tmp_path: Path,
) -> None:
    contract = _contract()
    fetched = fetch_world_bank_series_v62(
        task_id="mutated-source",
        contract=contract,
        fetcher=lambda item: _response(item),
        retrieved_at=NOW,
    )
    materialize_world_bank_series_v62(
        workspace_root=tmp_path,
        fetched=fetched,
    )
    raw_path = tmp_path / SOURCE_RAW_PATH
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")

    verification = verify_world_bank_source_v62(
        workspace_root=tmp_path,
        contract=fetched.contract,
        receipt=fetched.receipt,
        snapshot=fetched.snapshot,
    )

    assert verification.status == "FAIL"
    assert verification.checks["raw_response_hash_bound"] is False
    assert "raw_response_hash_bound" in verification.reason_codes


def test_authenticated_acquisition_and_current_s2_reverification(
    tmp_path: Path,
) -> None:
    authority, spec, acquisition = _authenticated_fixture(tmp_path)
    baseline = _baseline(spec)
    workspace = _WorkspaceStub(tmp_path, spec, baseline)
    fetched = acquisition.fetched

    assert authority.verify_acquisition(
        workspace_spec=spec,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        raw_body=fetched.raw_body,
        receipt=acquisition.authority_receipt,
    )
    reverified = authority.reverify_world_bank_source_at_s2(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        reverified_at=NOW + timedelta(minutes=2),
    )

    assert reverified.verification.status == "PASS"
    assert reverified.verification.scientific_provenance_status == "NOT_RUN"
    assert reverified.authority_receipt.s2_attempt == 1
    assert reverified.authority_receipt.workspace_spec_hash == spec.spec_hash
    assert (
        reverified.authority_receipt.source_acquisition_receipt_hash
        == acquisition.authority_receipt.receipt_hash
    )
    assert (
        reverified.authority_receipt.official_live_transport_authenticated
        is False
    )
    assert reverified.authority_receipt.runtime_identity.runtime_hash
    assert authority.verify_s2_reverification(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        verification=reverified.verification,
        receipt=reverified.authority_receipt,
    )
    assert authority.is_s2_reverification_admissible(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        verification=reverified.verification,
        receipt=reverified.authority_receipt,
    )


def test_hand_resealed_acquisition_cannot_forge_transport_authority(
    tmp_path: Path,
) -> None:
    authority, spec, acquisition = _authenticated_fixture(tmp_path)
    fetched = acquisition.fetched
    payload = acquisition.authority_receipt.model_dump(
        mode="json", exclude={"receipt_hash"}
    )
    payload["raw_response_hash"] = "f" * 64
    payload["receipt_hash"] = sha256_value(payload)
    forged = SourceAcquisitionReceiptV62.model_validate(payload)

    assert forged.receipt_hash == forged.content_hash()
    assert not authority.verify_acquisition(
        workspace_spec=spec,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        raw_body=fetched.raw_body,
        receipt=forged,
    )


def test_unsigned_self_sealed_source_records_cannot_enter_s2(
    tmp_path: Path,
) -> None:
    authority, spec, acquisition = _authenticated_fixture(tmp_path)
    fetched = acquisition.fetched
    unsigned = SourceAcquisitionReceiptV62.model_validate(
        acquisition.authority_receipt.model_dump(
            mode="json",
            exclude={"authority_auth_tag", "receipt_hash"},
        )
    )
    baseline = _baseline(spec)
    workspace = _WorkspaceStub(tmp_path, spec, baseline)

    assert not authority.verify_acquisition(
        workspace_spec=spec,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        raw_body=fetched.raw_body,
        receipt=unsigned,
    )
    with pytest.raises(ValueError, match="lacks authenticated authority"):
        authority.reverify_world_bank_source_at_s2(
            workspace=workspace,
            raw_baseline=baseline,
            contract=fetched.contract,
            source_receipt=fetched.receipt,
            snapshot=fetched.snapshot,
            acquisition_receipt=unsigned,
            reverified_at=NOW + timedelta(minutes=2),
        )


def test_acquisition_is_bound_to_exact_workspace_spec(
    tmp_path: Path,
) -> None:
    authority, spec, acquisition = _authenticated_fixture(tmp_path)
    fetched = acquisition.fetched
    baseline = _baseline(spec)
    workspace = _WorkspaceStub(tmp_path, spec, baseline)
    reverified = authority.reverify_world_bank_source_at_s2(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        reverified_at=NOW + timedelta(minutes=2),
    )
    other_spec = _spec("other-source-auth")
    other_baseline = _baseline(other_spec)
    other_workspace = _WorkspaceStub(tmp_path, other_spec, other_baseline)

    assert spec.spec_hash != other_spec.spec_hash
    assert not authority.verify_acquisition(
        workspace_spec=other_spec,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        raw_body=fetched.raw_body,
        receipt=acquisition.authority_receipt,
    )
    assert not authority.verify_s2_reverification(
        workspace=other_workspace,
        raw_baseline=other_baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        verification=reverified.verification,
        receipt=reverified.authority_receipt,
    )


def test_s2_reverification_rejects_old_attempt(
    tmp_path: Path,
) -> None:
    authority, spec, acquisition = _authenticated_fixture(tmp_path)
    fetched = acquisition.fetched
    baseline_v1 = _baseline(spec, attempt=1)
    workspace = _WorkspaceStub(tmp_path, spec, baseline_v1)
    reverified = authority.reverify_world_bank_source_at_s2(
        workspace=workspace,
        raw_baseline=baseline_v1,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        reverified_at=NOW + timedelta(minutes=2),
    )

    baseline_v2 = _baseline(spec, attempt=2)
    workspace.baseline = baseline_v2
    workspace.attempt = 2

    assert not authority.verify_s2_reverification(
        workspace=workspace,
        raw_baseline=baseline_v2,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        verification=reverified.verification,
        receipt=reverified.authority_receipt,
    )


def test_s2_reverification_can_reuse_v5_workspace_authority(
    tmp_path: Path,
) -> None:
    acquisition_authority, spec, acquisition = _authenticated_fixture(
        tmp_path
    )
    fetched = acquisition.fetched
    baseline = _baseline(spec)
    workspace = _WorkspaceStub(tmp_path, spec, baseline)
    workspace.authority_key_id = "workspace-source-key"
    workspace._authority_key = b"workspace-source-authority-" + b"w" * 32
    s2_authority = SourceTransportAuthorityV62.from_stage_workspace(
        workspace
    )

    reverified = s2_authority.reverify_world_bank_source_at_s2(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        acquisition_authority=acquisition_authority,
        reverified_at=NOW + timedelta(minutes=2),
    )

    assert reverified.authority_receipt.authority_mode == "v5_workspace_hmac"
    assert (
        reverified.authority_receipt.authority_key_id
        == "workspace-source-key"
    )
    assert s2_authority.is_s2_reverification_admissible(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        verification=reverified.verification,
        receipt=reverified.authority_receipt,
        acquisition_authority=acquisition_authority,
    )


def test_raw_tamper_yields_authenticated_fail_but_not_s2_admission(
    tmp_path: Path,
) -> None:
    authority, spec, acquisition = _authenticated_fixture(tmp_path)
    fetched = acquisition.fetched
    raw_path = tmp_path / SOURCE_RAW_PATH
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    baseline = _baseline(spec)
    workspace = _WorkspaceStub(tmp_path, spec, baseline)

    reverified = authority.reverify_world_bank_source_at_s2(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        reverified_at=NOW + timedelta(minutes=2),
    )

    assert reverified.verification.status == "FAIL"
    assert reverified.authority_receipt.replay_status == "FAIL"
    assert reverified.authority_receipt.scientific_provenance_status == "FAIL"
    assert authority.verify_s2_reverification(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        verification=reverified.verification,
        receipt=reverified.authority_receipt,
    )
    assert not authority.is_s2_reverification_admissible(
        workspace=workspace,
        raw_baseline=baseline,
        contract=fetched.contract,
        source_receipt=fetched.receipt,
        snapshot=fetched.snapshot,
        acquisition_receipt=acquisition.authority_receipt,
        verification=reverified.verification,
        receipt=reverified.authority_receipt,
    )


def test_live_transport_schema_cannot_upgrade_beyond_human(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = SourceTransportAuthorityV62(
        key_id="source-test-key",
        secret=AUTHORITY_SECRET,
    )
    spec = _spec(evidence_scope="public_data")
    live_contract = WorldBankSourceContractV62.seal(
        **_contract().model_dump(
            exclude={"contract_hash", "fixture_only"}
        ),
        fixture_only=False,
    )
    monkeypatch.setattr(
        public_source_module,
        "_default_fetcher",
        lambda item: _response(item),
    )
    acquisition = authority.acquire_world_bank_series(
        workspace_spec=spec,
        task_id=spec.workspace_id,
        contract=live_contract,
        retrieved_at=NOW,
        authenticated_at=NOW,
    )

    assert acquisition.authority_receipt.transport_mode == (
        "live_https_no_redirect"
    )
    assert acquisition.authority_receipt.scientific_provenance_status == "HUMAN"
    assert acquisition.fetched.snapshot.fixture_only is False
