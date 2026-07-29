"""V6.2 provenance binding for one current S2 attempt.

The World Bank adapter verifies transport bytes and parsing.  This module
binds that evidence to the authenticated S2 raw baseline, data ledger,
identity transform, and processed snapshot.  The binding is deliberately
claim-limited: fixture evidence is ``NOT_RUN`` and an otherwise valid public
source remains ``HUMAN`` until an independent measurement review is supplied
by a future external authority adapter.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import DataLedgerV50, RawDataBaselineV50
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52

from .public_source import (
    SourceVerificationV62,
    WorldBankSourceContractV62,
    WorldBankSourceReceiptV62,
)
from .source_auth import (
    S2SourceReverificationReceiptV62,
    SourceAcquisitionReceiptV62,
    SourceTransportAuthorityV62,
)


MEASUREMENT_SCHEMA_PATH = "docs/measurement_schema_v62.json"
PROVENANCE_BINDING_PATH = "data/source_provenance_v62/binding.json"
PROCESSED_SNAPSHOT_PATH = "data/processed/ode_snapshot.json"
TRANSFORM_PATH = "src/models/prepare_ode_data.py"

ProvenanceStatusV62 = Literal["PASS", "FAIL"]
ScientificProvenanceStatusV62 = Literal["FAIL", "NOT_RUN", "HUMAN"]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_file(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("provenance path escapes workspace")
    return candidate


class MeasurementSchemaV62(StrictModel):
    """Unreviewed operational meaning bound to an exact source contract."""

    schema_version: Literal["6.2-measurement-schema"] = (
        "6.2-measurement-schema"
    )
    measurement_id: Identifier
    source_contract_hash: Sha256
    indicator_id: Identifier
    semantic_name: Annotated[str, Field(min_length=3, max_length=300)]
    operational_definition: Annotated[
        str, Field(min_length=10, max_length=2000)
    ]
    observation_time_basis: Annotated[
        str, Field(min_length=3, max_length=300)
    ]
    aggregation_level: Annotated[str, Field(min_length=3, max_length=300)]
    time_unit: Identifier
    state_unit: Identifier
    missing_value_policy: Literal["reject"] = "reject"
    transformation_kind: Literal["identity"] = "identity"
    semantic_review_status: Literal["UNREVIEWED"] = "UNREVIEWED"
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    schema_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_schema(self) -> "MeasurementSchemaV62":
        if self.schema_hash and self.schema_hash != self.content_hash():
            raise ValueError("measurement schema hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "schema_hash")

    def assert_sealed(self) -> None:
        if not self.schema_hash or self.schema_hash != self.content_hash():
            raise ValueError("measurement schema is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "MeasurementSchemaV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"schema_hash"})
        payload["schema_hash"] = draft.content_hash()
        return cls(**payload)


class DataProvenanceBindingV62(StrictModel):
    """Mechanical source-to-S2 binding; never a qualification certificate."""

    schema_version: Literal["6.2-data-provenance-binding"] = (
        "6.2-data-provenance-binding"
    )
    workspace_spec_hash: Sha256
    s1_gate_hash: Sha256
    s2_attempt: Annotated[int, Field(ge=1)]
    raw_baseline_hash: Sha256
    raw_tree_hash: Sha256
    ledger_hash: Sha256
    processed_snapshot_hash: Sha256
    processed_file_hash: Sha256
    transform_script_hash: Sha256
    transform_receipt_hash: Sha256
    transform_params_hashes: list[Sha256]
    source_contract_hash: Sha256
    source_receipt_hash: Sha256
    source_verification_hash: Sha256
    source_acquisition_authority_receipt_hash: Sha256
    s2_source_reverification_receipt_hash: Sha256
    source_acquisition_authority_key_id: Identifier
    source_reverification_authority_key_id: Identifier
    source_acquisition_authority_mode: Literal[
        "external_hmac", "v5_workspace_hmac"
    ]
    source_reverification_authority_mode: Literal[
        "external_hmac", "v5_workspace_hmac"
    ]
    source_transport_mode: Literal[
        "live_https_no_redirect", "fixture_injected"
    ]
    official_live_transport_authenticated: bool
    source_raw_hash: Sha256
    measurement_schema_hash: Sha256
    fixture_only: bool
    status: ProvenanceStatusV62
    scientific_provenance_status: ScientificProvenanceStatusV62
    checks: dict[Identifier, bool]
    reason_codes: list[Identifier]
    independent_measurement_review: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    binding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "DataProvenanceBindingV62":
        if self.transform_params_hashes != sorted(
            set(self.transform_params_hashes)
        ):
            raise ValueError(
                "transform parameter hashes must be sorted and unique"
            )
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("provenance reasons must be sorted and unique")
        expected_pass = bool(self.checks) and all(self.checks.values())
        if (self.status == "PASS") != expected_pass:
            raise ValueError("provenance status differs from checks")
        required_source_auth_checks = {
            "source_acquisition_authority_authenticated",
            "current_s2_source_reverification_authenticated",
        }
        if not required_source_auth_checks.issubset(self.checks):
            raise ValueError(
                "provenance binding lacks required source authority checks"
            )
        if self.status == "PASS" and (
            not all(
                self.checks[check_id]
                for check_id in required_source_auth_checks
            )
            or self.source_acquisition_authority_receipt_hash == "0" * 64
            or self.s2_source_reverification_receipt_hash == "0" * 64
        ):
            raise ValueError(
                "provenance PASS requires authenticated source receipts"
            )
        expected_transport = (
            "fixture_injected"
            if self.fixture_only
            else "live_https_no_redirect"
        )
        if self.source_transport_mode != expected_transport:
            raise ValueError("source transport mode differs from fixture scope")
        if self.official_live_transport_authenticated != (
            not self.fixture_only
        ):
            raise ValueError(
                "official live transport flag differs from fixture scope"
            )
        expected_scientific = (
            "FAIL"
            if self.status == "FAIL"
            else "NOT_RUN"
            if self.fixture_only
            else "HUMAN"
        )
        if self.scientific_provenance_status != expected_scientific:
            raise ValueError("provenance scientific status is overstated")
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("provenance binding hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    def assert_sealed(self) -> None:
        if not self.binding_hash or self.binding_hash != self.content_hash():
            raise ValueError("provenance binding is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DataProvenanceBindingV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"binding_hash"})
        payload["binding_hash"] = draft.content_hash()
        return cls(**payload)


class S2TransformReceiptV62(StrictModel):
    """Receipt proving the declared S2 transform was the executed transform."""

    schema_version: Literal["6.2-s2-transform-receipt"] = (
        "6.2-s2-transform-receipt"
    )
    workspace_spec_hash: Sha256
    raw_baseline_hash: Sha256
    s2_attempt: Annotated[int, Field(ge=1)]
    task_id: Identifier
    input_relative_path: str
    input_hash: Sha256
    transform_relative_path: str
    transform_hash: Sha256
    output_relative_path: str
    output_hash: Sha256
    command: list[str] = Field(min_length=4)
    runtime_identity: str = Field(min_length=3, max_length=500)
    exit_code: Literal[0] = 0
    stdout_hash: Sha256
    stderr_hash: Sha256
    started_at: datetime
    finished_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "S2TransformReceiptV62":
        if (
            self.started_at.utcoffset() is None
            or self.finished_at.utcoffset() is None
        ):
            raise ValueError("transform receipt times must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("transform receipt time range is reversed")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("transform receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("transform receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "S2TransformReceiptV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


def build_data_provenance_binding_v62(
    *,
    workspace: StageWorkspaceV50,
    raw_baseline: RawDataBaselineV50,
    ledger: DataLedgerV50,
    snapshot: ODETimeSeriesSnapshotV52,
    source_contract: WorldBankSourceContractV62,
    source_receipt: WorldBankSourceReceiptV62,
    source_verification: SourceVerificationV62,
    source_acquisition_receipt: SourceAcquisitionReceiptV62,
    s2_source_reverification_receipt: S2SourceReverificationReceiptV62,
    source_authority: SourceTransportAuthorityV62,
    acquisition_authority: SourceTransportAuthorityV62 | None = None,
    measurement_schema: MeasurementSchemaV62,
    transform_receipt: S2TransformReceiptV62,
    processed_snapshot_relative_path: str = PROCESSED_SNAPSHOT_PATH,
    transform_relative_path: str = TRANSFORM_PATH,
) -> DataProvenanceBindingV62:
    """Recompute one exact source-to-S2 evidence binding."""

    root = workspace.root.resolve()
    checks: dict[str, bool] = {}
    try:
        snapshot.assert_sealed()
        source_contract.assert_sealed()
        source_receipt.assert_sealed()
        source_verification.assert_sealed()
        source_acquisition_receipt.runtime_identity.assert_sealed()
        s2_source_reverification_receipt.runtime_identity.assert_sealed()
        measurement_schema.assert_sealed()
        transform_receipt.assert_sealed()
        if (
            not source_acquisition_receipt.receipt_hash
            or source_acquisition_receipt.receipt_hash
            != source_acquisition_receipt.content_hash()
            or not s2_source_reverification_receipt.receipt_hash
            or s2_source_reverification_receipt.receipt_hash
            != s2_source_reverification_receipt.content_hash()
        ):
            raise ValueError("source authority receipt envelope differs")
        typed_inputs_sealed = True
    except ValueError:
        typed_inputs_sealed = False

    source_raw_path = _safe_file(root, source_receipt.raw_relative_path)
    processed_path = _safe_file(root, processed_snapshot_relative_path)
    transform_path = _safe_file(root, transform_relative_path)
    source_raw_exists = source_raw_path.is_file()
    processed_exists = processed_path.is_file()
    transform_exists = transform_path.is_file()
    source_raw_hash = (
        _file_hash(source_raw_path) if source_raw_exists else "0" * 64
    )
    source_raw_body = source_raw_path.read_bytes() if source_raw_exists else b""
    processed_file_hash = (
        _file_hash(processed_path) if processed_exists else "0" * 64
    )
    processed_snapshot_matches = False
    if processed_exists:
        try:
            replayed_snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
                processed_path.read_text(encoding="utf-8")
            )
            replayed_snapshot.assert_sealed()
            processed_snapshot_matches = (
                replayed_snapshot.snapshot_hash == snapshot.snapshot_hash
                and replayed_snapshot.model_dump(mode="json")
                == snapshot.model_dump(mode="json")
            )
        except (OSError, UnicodeDecodeError, ValueError):
            processed_snapshot_matches = False
    transform_script_hash = (
        _file_hash(transform_path) if transform_exists else "0" * 64
    )

    raw_paths = {item.relative_path for item in raw_baseline.files}
    ledger_raw_paths = {
        item.raw_relative_path
        for item in ledger.entries
        if item.raw_relative_path is not None
    }
    ledger_raw_hashes_match = True
    for entry in ledger.entries:
        if entry.raw_relative_path is None or entry.raw_response_hash is None:
            ledger_raw_hashes_match = False
            continue
        try:
            ledger_raw_hashes_match = ledger_raw_hashes_match and (
                _file_hash(_safe_file(root, entry.raw_relative_path))
                == entry.raw_response_hash
            )
        except (OSError, ValueError):
            ledger_raw_hashes_match = False

    accessed_after_source = all(
        entry.accessed_at is not None
        and source_receipt.retrieved_at <= entry.accessed_at
        for entry in ledger.entries
    )
    transform_params_hashes = sorted(
        {str(item.transform_params_hash) for item in ledger.entries}
    )
    acquiring_authority = acquisition_authority or source_authority
    acquisition_runtime = source_acquisition_receipt.runtime_identity
    reverification_runtime = s2_source_reverification_receipt.runtime_identity
    try:
        source_acquisition_authenticated = bool(
            source_raw_exists
            and acquiring_authority.verify_acquisition(
                workspace_spec=workspace.spec,
                contract=source_contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                raw_body=source_raw_body,
                receipt=source_acquisition_receipt,
            )
        )
    except (OSError, TypeError, ValueError):
        source_acquisition_authenticated = False
    try:
        current_s2_reverification_authenticated = bool(
            source_authority.verify_s2_reverification(
                workspace=workspace,
                raw_baseline=raw_baseline,
                contract=source_contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                acquisition_receipt=source_acquisition_receipt,
                verification=source_verification,
                receipt=s2_source_reverification_receipt,
                acquisition_authority=acquiring_authority,
            )
        )
    except (OSError, TypeError, ValueError):
        current_s2_reverification_authenticated = False
    current_s2_reverification_admissible = bool(
        current_s2_reverification_authenticated
        and source_verification.status == "PASS"
        and s2_source_reverification_receipt.replay_status == "PASS"
    )
    checks.update(
        {
            "typed_inputs_sealed": typed_inputs_sealed,
            "workspace_and_attempt_bound": (
                raw_baseline.workspace_spec_hash == workspace.spec.spec_hash
                and raw_baseline.s1_gate_hash == workspace.current_gate("S1")
                and raw_baseline.s2_attempt
                == workspace._latest_attempt("S2")
                and (
                    current_baseline := (
                        workspace._raw_baseline_for_current_s2()
                    )
                )
                is not None
                and current_baseline.baseline_hash
                == raw_baseline.baseline_hash
            ),
            "authenticated_raw_baseline_current": (
                workspace.verify_raw_baseline(raw_baseline)
            ),
            "ledger_bound_to_raw_baseline": (
                ledger.ledger_hash is not None
                and ledger.ledger_hash == ledger.content_hash()
                and ledger.raw_baseline_tree_hash
                == raw_baseline.raw_tree_hash
                and raw_paths == ledger_raw_paths
                and ledger_raw_hashes_match
            ),
            "official_source_binding_exact": (
                source_contract.contract_hash
                == source_receipt.contract_hash
                == source_verification.contract_hash
                and source_receipt.receipt_hash
                == source_verification.receipt_hash
                and source_receipt.snapshot_hash == snapshot.snapshot_hash
                and source_verification.snapshot_hash
                == snapshot.snapshot_hash
                and source_receipt.source_id == snapshot.source_id
                and source_acquisition_receipt.source_receipt_hash
                == source_receipt.receipt_hash
                and source_acquisition_receipt.snapshot_hash
                == snapshot.snapshot_hash
                and s2_source_reverification_receipt.source_acquisition_receipt_hash
                == source_acquisition_receipt.receipt_hash
                and s2_source_reverification_receipt.source_receipt_hash
                == source_receipt.receipt_hash
                and s2_source_reverification_receipt.source_verification_hash
                == source_verification.verification_hash
                and all(
                    item.source_kind == "official"
                    and item.source_ref == source_receipt.source_id
                    for item in ledger.entries
                )
            ),
            "source_acquisition_authority_authenticated": (
                source_acquisition_authenticated
            ),
            "current_s2_source_reverification_authenticated": (
                current_s2_reverification_authenticated
            ),
            "source_authority_runtime_code_consistent": (
                acquisition_runtime.public_source_code_hash
                == reverification_runtime.public_source_code_hash
                and acquisition_runtime.source_auth_code_hash
                == reverification_runtime.source_auth_code_hash
            ),
            "source_integrity_replay_passed": (
                current_s2_reverification_admissible
                and source_verification.status == "PASS"
                and source_raw_exists
                and source_raw_hash == source_receipt.response_bytes_hash
            ),
            "measurement_schema_bound": (
                measurement_schema.source_contract_hash
                == source_contract.contract_hash
                and measurement_schema.indicator_id
                == source_contract.indicator_id
                and measurement_schema.time_unit == snapshot.time_unit
                and measurement_schema.state_unit == snapshot.state_unit
            ),
            "processed_snapshot_exact": (
                processed_exists
                and processed_snapshot_matches
                and snapshot.snapshot_hash is not None
                and all(
                    item.processed_artifact_hash == processed_file_hash
                    for item in ledger.entries
                )
            ),
            "transform_exact": (
                transform_exists
                and all(
                    item.transform_script_relative_path
                    == transform_relative_path
                    and item.transform_script_hash == transform_script_hash
                    and item.transform_params_hash
                    == sha256_value(item.transform_params)
                    for item in ledger.entries
                )
            ),
            "transform_execution_receipt_bound": (
                transform_receipt.workspace_spec_hash
                == workspace.spec.spec_hash
                and transform_receipt.raw_baseline_hash
                == raw_baseline.baseline_hash
                and transform_receipt.s2_attempt
                == raw_baseline.s2_attempt
                and transform_receipt.task_id == snapshot.task_id
                and transform_receipt.input_hash
                in {
                    item.raw_response_hash
                    for item in ledger.entries
                    if item.raw_response_hash is not None
                }
                and transform_receipt.transform_relative_path
                == transform_relative_path
                and transform_receipt.transform_hash
                == transform_script_hash
                and transform_receipt.output_relative_path
                == processed_snapshot_relative_path
                and transform_receipt.output_hash == processed_file_hash
            ),
            "source_precedes_processing": accessed_after_source,
            "fixture_scope_consistent": (
                source_contract.fixture_only
                == source_receipt.fixture_only
                == source_verification.fixture_only
                == snapshot.fixture_only
                == source_acquisition_receipt.fixture_only
                == s2_source_reverification_receipt.fixture_only
                and s2_source_reverification_receipt.official_live_transport_authenticated
                == (not snapshot.fixture_only)
            ),
        }
    )
    reasons = sorted(
        check_id for check_id, passed in checks.items() if not passed
    )
    status: ProvenanceStatusV62 = "PASS" if not reasons else "FAIL"
    scientific_status: ScientificProvenanceStatusV62 = (
        "FAIL"
        if reasons
        else "NOT_RUN"
        if snapshot.fixture_only
        else "HUMAN"
    )
    return DataProvenanceBindingV62.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        s1_gate_hash=raw_baseline.s1_gate_hash,
        s2_attempt=raw_baseline.s2_attempt,
        raw_baseline_hash=(
            raw_baseline.baseline_hash
            if raw_baseline.baseline_hash is not None
            else "0" * 64
        ),
        raw_tree_hash=raw_baseline.raw_tree_hash,
        ledger_hash=ledger.ledger_hash if ledger.ledger_hash else "0" * 64,
        processed_snapshot_hash=(
            snapshot.snapshot_hash if snapshot.snapshot_hash else "0" * 64
        ),
        processed_file_hash=processed_file_hash,
        transform_script_hash=transform_script_hash,
        transform_receipt_hash=(
            transform_receipt.receipt_hash
            if transform_receipt.receipt_hash
            else "0" * 64
        ),
        transform_params_hashes=transform_params_hashes,
        source_contract_hash=(
            source_contract.contract_hash
            if source_contract.contract_hash
            else "0" * 64
        ),
        source_receipt_hash=(
            source_receipt.receipt_hash
            if source_receipt.receipt_hash
            else "0" * 64
        ),
        source_verification_hash=(
            source_verification.verification_hash
            if source_verification.verification_hash
            else "0" * 64
        ),
        source_acquisition_authority_receipt_hash=(
            source_acquisition_receipt.receipt_hash
            if source_acquisition_receipt.receipt_hash
            else "0" * 64
        ),
        s2_source_reverification_receipt_hash=(
            s2_source_reverification_receipt.receipt_hash
            if s2_source_reverification_receipt.receipt_hash
            else "0" * 64
        ),
        source_acquisition_authority_key_id=(
            source_acquisition_receipt.authority_key_id
        ),
        source_reverification_authority_key_id=(
            s2_source_reverification_receipt.authority_key_id
        ),
        source_acquisition_authority_mode=(
            source_acquisition_receipt.authority_mode
        ),
        source_reverification_authority_mode=(
            s2_source_reverification_receipt.authority_mode
        ),
        source_transport_mode=source_acquisition_receipt.transport_mode,
        official_live_transport_authenticated=(
            s2_source_reverification_receipt.official_live_transport_authenticated
        ),
        source_raw_hash=source_raw_hash,
        measurement_schema_hash=(
            measurement_schema.schema_hash
            if measurement_schema.schema_hash
            else "0" * 64
        ),
        fixture_only=snapshot.fixture_only,
        status=status,
        scientific_provenance_status=scientific_status,
        checks=checks,
        reason_codes=reasons,
    )


__all__ = [
    "DataProvenanceBindingV62",
    "MEASUREMENT_SCHEMA_PATH",
    "MeasurementSchemaV62",
    "PROVENANCE_BINDING_PATH",
    "PROCESSED_SNAPSHOT_PATH",
    "S2TransformReceiptV62",
    "TRANSFORM_PATH",
    "build_data_provenance_binding_v62",
]
