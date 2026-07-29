"""Narrow, auditable S2--S6 Studio runtime for positive scalar ODE series.

This is an additive vertical slice.  It does not pretend to execute arbitrary
S1 mathematics: the selected S1 candidate must explicitly describe an
autonomous ODE family, and the supplied data must satisfy the registered V5.2
adapter contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import numpy as np
import scipy
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.check_registry import (
    AdapterContextV50,
    AdapterOutcomeV50,
    CheckRegistryV50,
)
from fma.v5.paper import build_paper
from fma.v5.stage_workspace import POLICIES, StageWorkspaceV50, _tree_hash
from fma.v5.workspace_schemas import (
    CandidateFormalizationV50,
    CandidateSetV50,
    CodeManifestV50,
    DataLedgerEntryV50,
    DataLedgerV50,
    DecisionAssertionV50,
    DecisionDossierV50,
    ModelSpecV50,
    ProcessedArtifactV50,
    ProcessedManifestV50,
    ResultIndexV50,
    ResultRecordV50,
    StageId,
    UQClaimV50,
    UQSummaryV50,
    ValidationObligationV50,
    ValidationPlanV50,
)
from fma.v5_1.codex_stage_driver import (
    RoleProcessOutcomeV51,
    StageRoleDriverV51,
    commit_generator_outcome_v51,
)
from fma.v5_2.ode_system import (
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
    run_ode_replays_v52,
)
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_7.adaptive_positive_series import (
    AdaptivePositiveSeriesBundleV57,
    AdaptiveReplayAuthorityV57,
    AdaptiveThresholdsV57,
    build_adaptive_positive_series_bundle_v57,
    run_authenticated_adaptive_replays_v57,
)
from fma.v6.decision_value import (
    DECISION_CONTRACT_PATH,
    DECISION_EVIDENCE_PATH,
    DECISION_INTENT_PATH,
    DecisionValueContractV62,
    DecisionValueEvidenceV62,
    DecisionValueIntentV62,
    decision_contract_from_intent_v62,
    evaluate_decision_value_v62,
)
from fma.v6.executable_candidate import (
    EXECUTABLE_CANDIDATE_INTENT_PATH,
    EXECUTABLE_CANDIDATE_IR_PATH,
    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
    ExecutableCandidateReceiptV62,
    ExecutableCandidateResolutionV62,
    RegisteredFamilySearchIRV62,
    RegisteredFamilySearchIntentV62,
    build_executable_candidate_receipt_v62,
    resolve_executable_candidate_v62,
    verify_executable_candidate_receipt_v62,
)
from fma.v6.measurement_study_design import (
    MEASUREMENT_STUDY_DESIGN_PATH_V67,
    MeasurementStudyDesignContractV67,
)
from fma.v6.predata_protocol import (
    CANDIDATE_EXECUTION_BINDING_PATH_V67,
    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
    CandidateExecutionBindingV67,
    PreDataExecutionProtocolV67,
    bind_candidate_to_predata_protocol_v67,
    registered_positive_series_capability_pack_v67,
    verify_predata_execution_protocol_v67,
)
from fma.v6.provenance import (
    MEASUREMENT_SCHEMA_PATH,
    PROVENANCE_BINDING_PATH,
    DataProvenanceBindingV62,
    MeasurementSchemaV62,
    S2TransformReceiptV62,
    build_data_provenance_binding_v62,
)
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    SOURCE_RAW_PATH,
    SOURCE_RECEIPT_PATH,
    SOURCE_VERIFICATION_PATH,
    SourceVerificationV62,
    WorldBankSourceContractV62,
    WorldBankSourceReceiptV62,
)
from fma.v6.recovery_kernel import (
    ProblemSignatureV60,
    RecoveryKernelV60,
    default_capability_registry_v60,
)
from fma.v6.scientific_success import (
    ROLLING_CONFIRMATION_PATH,
    RollingConfirmationV61,
    SUCCESS_CONTRACT_PATH,
    ScientificSuccessContractV61,
    default_scientific_success_contract_v61,
    evaluate_rolling_confirmation_v61,
)
from fma.v6.source_auth import (
    S2_SOURCE_REVERIFICATION_PATH,
    SOURCE_ACQUISITION_AUTH_PATH,
    S2SourceReverificationReceiptV62,
    SourceAcquisitionReceiptV62,
    SourceTransportAuthorityV62,
)


EventCallback = Callable[
    [
        str,
        Literal["accepted", "running", "succeeded", "failed", "blocked"],
        str,
        dict[str, Any],
    ],
    None,
]
DriverFactory = Callable[[], StageRoleDriverV51]

ODE_ADAPTER_ID = "scalar_autonomous_ode_v52"
ADAPTIVE_ADAPTER_ID = "adaptive_positive_series_v57"
RAW_RELATIVE_PATH = "data/raw/ode_series.json"
PROCESSED_RELATIVE_PATH = "data/processed/ode_snapshot.json"
ADAPTER_BINDING_PATH = "docs/adapter_binding.json"
BUNDLE_PATH = "results/ode_scientific_bundle.json"
REPLAY_INPUT_PATH = "checks/ode_replay_input.json"
ADAPTIVE_BUNDLE_PATH = "results/adaptive_positive_series_bundle.json"
ADAPTIVE_REPLAY_INPUT_PATH = "checks/adaptive_replay_input.json"
ADAPTIVE_REPLAY_RECEIPTS_PATH = "checks/adaptive_replay_receipts.json"
ADAPTIVE_REPLAY_SUMMARY_PATH = "checks/adaptive_replay_receipt.json"
S2_TRANSFORM_RECEIPT_PATH = "checks/s2_data_transform_receipt.json"
EXECUTABLE_CANDIDATE_RESOLUTION_PATH_V67 = (
    "docs/executable_candidate_resolution_v67.json"
)


class BackhalfRuntimeError(RuntimeError):
    pass


class V67S2CompatibilityFailure(StrictModel):
    """Typed, claim-limited evidence for a pre-write V6.7 rejection."""

    schema_version: Literal["6.7-s2-compatibility-failure"] = (
        "6.7-s2-compatibility-failure"
    )
    stage: Literal["S2"] = "S2"
    failure_owner: Literal["data_contract", "capability"]
    compatibility_phase: Literal[
        "artifact_replay",
        "pre_acquisition",
        "pre_raw_materialization",
        "s2_replay",
    ]
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256 | None = None
    measurement_contract_hash: Sha256 | None = None
    predata_protocol_hash: Sha256 | None = None
    reason_codes: list[Identifier] = Field(min_length=1)
    checks: dict[Identifier, bool] = Field(min_length=1)
    observation_values_included: Literal[False] = False
    raw_ode_data_written_by_backhalf: Literal[False] = False
    scientific_failure_established: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    failure_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> "V67S2CompatibilityFailure":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError(
                "V6.7 S2 failure reason codes must be sorted and unique"
            )
        if list(self.checks) != sorted(self.checks):
            raise ValueError("V6.7 S2 failure checks must be sorted")
        expected_reasons = sorted(
            check_id for check_id, passed in self.checks.items() if not passed
        )
        if self.reason_codes != expected_reasons:
            raise ValueError(
                "V6.7 S2 failure reasons differ from failed checks"
            )
        if self.failure_hash and self.failure_hash != self.content_hash():
            raise ValueError("V6.7 S2 failure hash differs")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"failure_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "V67S2CompatibilityFailure":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"failure_hash"})
        payload["failure_hash"] = draft.content_hash()
        return cls(**payload)


class ExecutableCandidateResolutionV67(StrictModel):
    """S2 compatibility projection over the pre-data V6.7 authority."""

    schema_version: Literal["6.7-executable-candidate-resolution"] = (
        "6.7-executable-candidate-resolution"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    s1_gate_hash: Sha256
    s2_attempt: int = Field(ge=1)
    candidate_id: Identifier
    candidate_structural_hash: Sha256
    legacy_v62_resolution_hash: Sha256
    measurement_contract_hash: Sha256
    predata_protocol_hash: Sha256
    source_contract_hash: Sha256
    candidate_execution_binding_hash: Sha256
    adapter_id: Identifier
    adapter_version: Identifier
    capability_pack_hash: Sha256
    threshold_hashes: list[Sha256] = Field(min_length=2)
    adapter_resolution_stage: Literal["pre_data_compiler"] = (
        "pre_data_compiler"
    )
    s2_role: Literal["compatibility_validation_only"] = (
        "compatibility_validation_only"
    )
    silent_adapter_substitution_permitted: Literal[False] = False
    recovery_requires_new_graph_attempt: Literal[True] = True
    recovery_requires_successor_protocol: Literal[True] = True
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    resolution_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> "ExecutableCandidateResolutionV67":
        if self.threshold_hashes != sorted(set(self.threshold_hashes)):
            raise ValueError(
                "V6.7 resolution threshold hashes must be sorted and unique"
            )
        if self.resolution_hash and self.resolution_hash != self.content_hash():
            raise ValueError("V6.7 executable resolution hash differs")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"resolution_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "ExecutableCandidateResolutionV67":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"resolution_hash"})
        payload["resolution_hash"] = draft.content_hash()
        return cls(**payload)


class V67S2CompatibilityError(BackhalfRuntimeError):
    """Fail-closed V6.7 exception carrying machine-readable graph evidence."""

    def __init__(self, evidence: V67S2CompatibilityFailure) -> None:
        self.evidence = evidence
        super().__init__(
            f"V6.7 {evidence.failure_owner} compatibility failure: "
            + ", ".join(evidence.reason_codes)
        )


@dataclass(frozen=True)
class V67S2ContractContext:
    """Authenticated pre-data authority for the current S1 lineage."""

    measurement: MeasurementStudyDesignContractV67
    protocol: PreDataExecutionProtocolV67
    source_contract: WorldBankSourceContractV62
    candidate_binding: CandidateExecutionBindingV67


class StudioODEDataRequestV59(StrictModel):
    schema_version: Literal["5.9"] = "5.9"
    adapter_id: Literal[
        "scalar_autonomous_ode_v52",
        "adaptive_positive_series_v57",
    ] = ODE_ADAPTER_ID
    time_unit: Identifier
    state_unit: Identifier
    times: list[float] = Field(min_length=12, max_length=4096)
    observations: list[float] = Field(min_length=12, max_length=4096)
    source_id: str = Field(min_length=3, max_length=300)
    license_status: str = Field(min_length=2, max_length=300)
    fixture_only: bool = False

    @model_validator(mode="after")
    def validate_series(self) -> "StudioODEDataRequestV59":
        if len(self.times) != len(self.observations):
            raise ValueError("times and observations must have equal length")
        if any(not math.isfinite(value) for value in self.times):
            raise ValueError("times must be finite")
        if any(
            right <= left for left, right in zip(self.times, self.times[1:])
        ):
            raise ValueError("times must be strictly increasing")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.observations
        ):
            raise ValueError("ODE observations must be finite and positive")
        return self


class DataMappingDraftV59(StrictModel):
    schema_version: Literal["5.9"] = "5.9"
    data_requirement_ids: list[Identifier] = Field(min_length=1, max_length=8)
    semantic_name: str = Field(min_length=5, max_length=300)
    units: str = Field(min_length=1, max_length=100)
    transform_rule: str = Field(min_length=10, max_length=600)
    quality_flags: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_ids(self) -> "DataMappingDraftV59":
        if self.data_requirement_ids != sorted(set(self.data_requirement_ids)):
            raise ValueError("data requirement IDs must be sorted and unique")
        return self


class DecisionNarrativeDraftV59(StrictModel):
    schema_version: Literal["5.9"] = "5.9"
    statement: str = Field(min_length=20, max_length=1200)
    limitations: list[str] = Field(min_length=1, max_length=8)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_stage_file_admitted(
    workspace: StageWorkspaceV50,
    *,
    stage: StageId,
    relative_path: str,
) -> bool:
    """Return whether the current authenticated stage admits this exact file."""

    try:
        path = workspace.root / relative_path
        certificate = workspace._certificate_for_current_node(stage)
        if (
            not path.is_file()
            or certificate is None
            or workspace.current_gate(stage) != certificate.certificate_hash
            or not workspace.verify_certificate(certificate)
        ):
            return False
        binding = next(
            (
                item
                for item in certificate.manifest.files
                if item.relative_path == relative_path
            ),
            None,
        )
        return bool(
            binding is not None
            and binding.size_bytes == path.stat().st_size
            and binding.sha256 == _sha(path)
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _raise_v67_compatibility(
    workspace: StageWorkspaceV50,
    *,
    failure_owner: Literal["data_contract", "capability"],
    compatibility_phase: Literal[
        "artifact_replay",
        "pre_acquisition",
        "pre_raw_materialization",
        "s2_replay",
    ],
    checks: dict[str, bool],
    measurement: MeasurementStudyDesignContractV67 | None = None,
    protocol: PreDataExecutionProtocolV67 | None = None,
) -> None:
    ordered_checks = dict(sorted(checks.items()))
    reason_codes = sorted(
        check_id
        for check_id, passed in ordered_checks.items()
        if not passed
    )
    if not reason_codes:
        raise ValueError("V6.7 compatibility failure has no failed check")
    evidence = V67S2CompatibilityFailure.seal(
        failure_owner=failure_owner,
        compatibility_phase=compatibility_phase,
        workspace_spec_hash=str(workspace.spec.spec_hash),
        s0_gate_hash=workspace.current_gate("S0"),
        measurement_contract_hash=(
            measurement.contract_hash if measurement is not None else None
        ),
        predata_protocol_hash=(
            protocol.protocol_hash if protocol is not None else None
        ),
        reason_codes=reason_codes,
        checks=ordered_checks,
    )
    raise V67S2CompatibilityError(evidence)


def load_v67_s2_contract_v67(
    workspace: StageWorkspaceV50,
) -> V67S2ContractContext | None:
    """Load and replay the current authenticated V6.7 pre-data authority.

    The absence of both V6.7 files selects the legacy path.  Any partial,
    stale, unsealed, unadmitted, or non-replayable V6.7 state fails closed.
    """

    root = workspace.root
    measurement_path = root / MEASUREMENT_STUDY_DESIGN_PATH_V67
    protocol_path = root / PREDATA_EXECUTION_PROTOCOL_PATH_V67
    source_contract_path = root / SOURCE_CONTRACT_PATH
    candidate_binding_path = root / CANDIDATE_EXECUTION_BINDING_PATH_V67
    measurement_present = measurement_path.is_file()
    protocol_present = protocol_path.is_file()
    if not measurement_present and not protocol_present:
        return None
    if not measurement_present or not protocol_present:
        _raise_v67_compatibility(
            workspace,
            failure_owner="capability",
            compatibility_phase="artifact_replay",
            checks={
                "measurement_contract_present": measurement_present,
                "predata_protocol_present": protocol_present,
            },
        )
    if not source_contract_path.is_file():
        _raise_v67_compatibility(
            workspace,
            failure_owner="data_contract",
            compatibility_phase="artifact_replay",
            checks={"source_contract_present": False},
        )
    if not candidate_binding_path.is_file():
        _raise_v67_compatibility(
            workspace,
            failure_owner="capability",
            compatibility_phase="artifact_replay",
            checks={"candidate_execution_binding_present": False},
        )
    try:
        measurement = MeasurementStudyDesignContractV67.model_validate_json(
            measurement_path.read_text(encoding="utf-8")
        )
        protocol = PreDataExecutionProtocolV67.model_validate_json(
            protocol_path.read_text(encoding="utf-8")
        )
        source_contract = WorldBankSourceContractV62.model_validate_json(
            source_contract_path.read_text(encoding="utf-8")
        )
        candidate_binding = CandidateExecutionBindingV67.model_validate_json(
            candidate_binding_path.read_text(encoding="utf-8")
        )
        measurement.assert_sealed()
        protocol.assert_sealed()
        source_contract.assert_sealed()
        candidate_binding.assert_sealed()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        try:
            _raise_v67_compatibility(
                workspace,
                failure_owner="capability",
                compatibility_phase="artifact_replay",
                checks={"v67_artifacts_schema_and_seals_valid": False},
            )
        except V67S2CompatibilityError as failure:
            raise failure from exc

    current_s0 = workspace.current_gate("S0")
    current_s1 = workspace.current_gate("S1")
    s1_certificate = workspace._certificate_for_current_node("S1")
    lineage_checks = {
        "measurement_bound_to_current_s0": (
            current_s0 is not None
            and measurement.s0_gate_hash == current_s0
        ),
        "measurement_bound_to_workspace": (
            measurement.workspace_spec_hash == workspace.spec.spec_hash
        ),
        "measurement_contract_admitted_by_current_s1": (
            _current_stage_file_admitted(
                workspace,
                stage="S1",
                relative_path=MEASUREMENT_STUDY_DESIGN_PATH_V67,
            )
        ),
        "candidate_execution_binding_admitted_by_current_s1": (
            _current_stage_file_admitted(
                workspace,
                stage="S1",
                relative_path=CANDIDATE_EXECUTION_BINDING_PATH_V67,
            )
        ),
        "predata_bound_to_current_s0": (
            current_s0 is not None and protocol.s0_gate_hash == current_s0
        ),
        "predata_bound_to_measurement": (
            protocol.measurement_contract_id == measurement.contract_id
            and protocol.measurement_contract_hash
            == measurement.contract_hash
            and protocol.measurement_id
            == measurement.measurement.measurement_id
        ),
        "predata_bound_to_workspace": (
            protocol.workspace_spec_hash == workspace.spec.spec_hash
        ),
        "predata_protocol_admitted_by_current_s1": (
            _current_stage_file_admitted(
                workspace,
                stage="S1",
                relative_path=PREDATA_EXECUTION_PROTOCOL_PATH_V67,
            )
        ),
        "source_contract_admitted_by_current_s1": (
            _current_stage_file_admitted(
                workspace,
                stage="S1",
                relative_path=SOURCE_CONTRACT_PATH,
            )
        ),
        "source_contract_bound_to_measurement": (
            measurement.source_contract_id == source_contract.contract_id
            and measurement.source_contract_hash
            == source_contract.contract_hash
        ),
        "source_contract_bound_to_predata_protocol": (
            protocol.source_contract_id == source_contract.contract_id
            and protocol.source_contract_hash
            == source_contract.contract_hash
        ),
        "s1_certificate_current_and_authenticated": (
            current_s1 is not None
            and s1_certificate is not None
            and current_s1 == s1_certificate.certificate_hash
            and workspace.verify_certificate(s1_certificate)
        ),
        "s1_predecessor_is_current_s0": (
            current_s0 is not None
            and s1_certificate is not None
            and s1_certificate.manifest.predecessor_gate_hash == current_s0
            and s1_certificate.upstream_gate_hashes == [current_s0]
        ),
    }
    if not all(lineage_checks.values()):
        _raise_v67_compatibility(
            workspace,
            failure_owner="capability",
            compatibility_phase="artifact_replay",
            checks=lineage_checks,
            measurement=measurement,
            protocol=protocol,
        )
    try:
        pack = registered_positive_series_capability_pack_v67(
            protocol.adapter_binding.adapter_id
        )
        replay_valid = verify_predata_execution_protocol_v67(
            measurement_contract=measurement,
            capability_pack=pack,
            protocol=protocol,
        )
    except (TypeError, ValueError):
        replay_valid = False
    try:
        model, selected_payload = _selected_candidate(workspace)
        selected_candidate = CandidateFormalizationV50.model_validate(
            selected_payload
        )
        execution_intent = RegisteredFamilySearchIntentV62.model_validate_json(
            (root / EXECUTABLE_CANDIDATE_INTENT_PATH).read_text(
                encoding="utf-8"
            )
        )
        execution_ir = RegisteredFamilySearchIRV62.model_validate_json(
            (root / EXECUTABLE_CANDIDATE_IR_PATH).read_text(
                encoding="utf-8"
            )
        )
        expected_candidate_binding = bind_candidate_to_predata_protocol_v67(
            candidate=selected_candidate,
            execution_intent=execution_intent,
            execution_ir=execution_ir,
            protocol=protocol,
        )
        candidate_binding_replay_valid = bool(
            candidate_binding == expected_candidate_binding
            and model.selected_candidate_id == candidate_binding.candidate_id
            and model.selected_candidate_structural_hash
            == candidate_binding.candidate_structural_hash
            and candidate_binding.workspace_spec_hash
            == workspace.spec.spec_hash
            and candidate_binding.s0_gate_hash == current_s0
            and candidate_binding.source_contract_hash
            == source_contract.contract_hash
            and candidate_binding.measurement_contract_hash
            == measurement.contract_hash
            and candidate_binding.predata_protocol_hash
            == protocol.protocol_hash
            and candidate_binding.selected_adapter_id
            == protocol.adapter_binding.adapter_id
            and candidate_binding.selected_adapter_version
            == protocol.adapter_binding.adapter_version
            and candidate_binding.capability_pack_hash
            == protocol.adapter_binding.capability_pack_hash
        )
    except (OSError, TypeError, UnicodeDecodeError, ValueError):
        candidate_binding_replay_valid = False
    replay_checks = {
        "adapter_resolution_frozen_before_data": (
            protocol.adapter_resolution.adapter_resolution_stage
            == "pre_data_compiler"
            and protocol.adapter_resolution.s2_role
            == "compatibility_validation_only"
            and not protocol.adapter_resolution.silent_adapter_substitution_permitted
            and not protocol.adapter_resolution.same_protocol_fallback_permitted
        ),
        "candidate_execution_binding_exact_replay": (
            candidate_binding_replay_valid
        ),
        "predata_protocol_exact_replay": replay_valid,
    }
    if not all(replay_checks.values()):
        _raise_v67_compatibility(
            workspace,
            failure_owner="capability",
            compatibility_phase="artifact_replay",
            checks=replay_checks,
            measurement=measurement,
            protocol=protocol,
        )
    return V67S2ContractContext(
        measurement=measurement,
        protocol=protocol,
        source_contract=source_contract,
        candidate_binding=candidate_binding,
    )


def validate_v67_pre_acquisition_v67(
    workspace: StageWorkspaceV50,
    *,
    adapter_id: str,
    source_contract: WorldBankSourceContractV62,
    measurement_unit: str,
    time_basis: str,
    missing_value_policy: str,
) -> V67S2ContractContext | None:
    """Reject a mismatched V6.7 acquisition request before network access."""

    context = load_v67_s2_contract_v67(workspace)
    if context is None:
        return None
    try:
        source_contract.assert_sealed()
    except ValueError:
        source_sealed = False
    else:
        source_sealed = True
    protocol = context.protocol
    measurement = context.measurement
    interval_count = (
        source_contract.end_year - source_contract.start_year + 1
    )
    minimum_required = max(
        measurement.sampling.minimum_sample_size,
        protocol.compatibility.minimum_execution_observation_count,
        protocol.compatibility.minimum_confirmation_observation_count,
    )
    checks = {
        "adapter_matches_frozen_protocol": (
            adapter_id == protocol.adapter_binding.adapter_id
        ),
        "measurement_unit_matches_frozen_contract": (
            measurement_unit
            == measurement.measurement.unit
            == protocol.compatibility.exact_measurement_unit_required
            == source_contract.state_unit
        ),
        "missingness_policy_matches_frozen_contract": (
            missing_value_policy
            in {"reject", "reject_incomplete_series"}
            and measurement.missingness.handling_policy
            == "reject_incomplete_series"
            and protocol.compatibility.missing_value_policy
            == "reject_incomplete_series"
        ),
        "source_contract_exactly_matches_frozen_source": (
            source_sealed and source_contract == context.source_contract
        ),
        "source_interval_can_meet_frozen_sample_minimum": (
            interval_count >= minimum_required
            and source_contract.minimum_observations >= minimum_required
        ),
        "time_basis_matches_frozen_contract": (
            time_basis
            == measurement.measurement.time_basis
            == protocol.compatibility.exact_time_basis_required
        ),
    }
    if not all(checks.values()):
        owner: Literal["data_contract", "capability"] = (
            "capability"
            if not checks["adapter_matches_frozen_protocol"]
            else "data_contract"
        )
        _raise_v67_compatibility(
            workspace,
            failure_owner=owner,
            compatibility_phase="pre_acquisition",
            checks=checks,
            measurement=measurement,
            protocol=protocol,
        )
    return context


def _effectively_regular_cadence(
    times: list[float],
    *,
    maximum_relative_deviation: float,
) -> bool:
    if len(times) < 2:
        return False
    deltas = [
        right - left for left, right in zip(times, times[1:])
    ]
    reference = deltas[0]
    if not math.isfinite(reference) or reference <= 0:
        return False
    return all(
        math.isfinite(delta)
        and delta > 0
        and abs(delta - reference) / abs(reference)
        <= maximum_relative_deviation
        for delta in deltas
    )


def validate_v67_data_compatibility_v67(
    workspace: StageWorkspaceV50,
    request: StudioODEDataRequestV59,
    *,
    source_contract: WorldBankSourceContractV62 | None = None,
    source_receipt: WorldBankSourceReceiptV62 | None = None,
    source_acquisition_receipt: SourceAcquisitionReceiptV62 | None = None,
    measurement_schema: MeasurementSchemaV62 | None = None,
    source_raw_body: bytes | None = None,
    require_source_evidence: bool = True,
    compatibility_phase: Literal[
        "pre_raw_materialization", "s2_replay"
    ] = "pre_raw_materialization",
) -> V67S2ContractContext | None:
    """Evaluate all frozen V6.7 predicates without changing workspace state."""

    context = load_v67_s2_contract_v67(workspace)
    if context is None:
        return None
    contract = source_contract or context.source_contract
    protocol = context.protocol
    measurement = context.measurement
    minimum_required = max(
        measurement.sampling.minimum_sample_size,
        protocol.compatibility.minimum_execution_observation_count,
        protocol.compatibility.minimum_confirmation_observation_count,
    )
    cadence_limit = (
        protocol.compatibility.maximum_cadence_relative_deviation
    )
    expected_times = [
        float(year)
        for year in range(contract.start_year, contract.end_year + 1)
    ]
    data_checks = {
        "adapter_matches_frozen_protocol": (
            request.adapter_id == protocol.adapter_binding.adapter_id
        ),
        "all_observations_present": (
            len(request.times) == len(request.observations)
            and len(request.observations) == len(expected_times)
        ),
        "cadence_matches_frozen_policy": (
            not protocol.compatibility.effectively_regular_cadence_required
            or (
                cadence_limit is not None
                and _effectively_regular_cadence(
                    request.times,
                    maximum_relative_deviation=cadence_limit,
                )
            )
        ),
        "finite_positive_values_match_frozen_policy": (
            (
                all(
                    math.isfinite(value) and value > 0
                    for value in request.observations
                )
                if protocol.compatibility.finite_positive_values_required
                else all(math.isfinite(value) for value in request.observations)
            )
        ),
        "observation_count_meets_frozen_minimum": (
            len(request.observations) >= minimum_required
        ),
        "source_contract_exactly_matches_frozen_source": (
            contract == context.source_contract
        ),
        "source_fixture_scope_matches": (
            request.fixture_only == contract.fixture_only
        ),
        "source_id_matches_frozen_source": (
            request.source_id
            == (
                f"world-bank:{contract.country_code}:"
                f"{contract.indicator_id}:{contract.start_year}-"
                f"{contract.end_year}"
            )
        ),
        "state_unit_matches_frozen_measurement": (
            request.state_unit
            == contract.state_unit
            == measurement.measurement.unit
            == protocol.compatibility.exact_measurement_unit_required
        ),
        "strictly_increasing_time_matches_frozen_policy": (
            not protocol.compatibility.strictly_increasing_time_required
            or all(
                right > left
                for left, right in zip(
                    request.times, request.times[1:]
                )
            )
        ),
        "time_grid_matches_frozen_source": request.times == expected_times,
        "time_unit_matches_frozen_source": (
            request.time_unit == contract.time_unit
        ),
    }
    if not all(data_checks.values()):
        owner: Literal["data_contract", "capability"] = (
            "capability"
            if not data_checks["adapter_matches_frozen_protocol"]
            else "data_contract"
        )
        _raise_v67_compatibility(
            workspace,
            failure_owner=owner,
            compatibility_phase=compatibility_phase,
            checks=data_checks,
            measurement=measurement,
            protocol=protocol,
        )

    source_items_present = all(
        item is not None
        for item in (
            source_receipt,
            source_acquisition_receipt,
            measurement_schema,
            source_raw_body,
        )
    )
    if require_source_evidence and not source_items_present:
        _raise_v67_compatibility(
            workspace,
            failure_owner="data_contract",
            compatibility_phase=compatibility_phase,
            checks={"authenticated_source_evidence_complete": False},
            measurement=measurement,
            protocol=protocol,
        )
    if source_items_present:
        assert source_receipt is not None
        assert source_acquisition_receipt is not None
        assert measurement_schema is not None
        assert source_raw_body is not None
        try:
            source_receipt.assert_sealed()
            measurement_schema.assert_sealed()
            snapshot = ODETimeSeriesSnapshotV52.seal(
                task_id=workspace.spec.workspace_id,
                time_unit=request.time_unit,
                state_unit=request.state_unit,
                times=request.times,
                observations=request.observations,
                source_id=request.source_id,
                fixture_only=request.fixture_only,
            )
            authority = SourceTransportAuthorityV62.from_stage_workspace(
                workspace
            )
            acquisition_valid = authority.verify_acquisition(
                workspace_spec=workspace.spec,
                contract=contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                raw_body=source_raw_body,
                receipt=source_acquisition_receipt,
            )
            source_models_valid = True
        except (OSError, TypeError, ValueError):
            acquisition_valid = False
            source_models_valid = False
        source_checks = {
            "measurement_schema_matches_frozen_measurement": (
                source_models_valid
                and measurement_schema.measurement_id
                == measurement.measurement.measurement_id
                and measurement_schema.source_contract_hash
                == contract.contract_hash
                and measurement_schema.indicator_id
                == contract.indicator_id
                and measurement_schema.observation_time_basis
                == measurement.measurement.time_basis
                and measurement_schema.time_unit == request.time_unit
                and measurement_schema.state_unit
                == measurement.measurement.unit
                and measurement_schema.missing_value_policy == "reject"
                and measurement_schema.transformation_kind == "identity"
            ),
            "source_acquisition_replay_authenticated": acquisition_valid,
            "source_receipt_matches_frozen_request": (
                source_models_valid
                and source_receipt.contract_hash == contract.contract_hash
                and source_receipt.source_id == request.source_id
                and source_receipt.observation_count
                == len(request.observations)
                and source_receipt.first_year == contract.start_year
                and source_receipt.last_year == contract.end_year
                and source_receipt.fixture_only == request.fixture_only
            ),
        }
        if not all(source_checks.values()):
            _raise_v67_compatibility(
                workspace,
                failure_owner="data_contract",
                compatibility_phase=compatibility_phase,
                checks=source_checks,
                measurement=measurement,
                protocol=protocol,
            )
    return context


def _write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise BackhalfRuntimeError(
            f"refusing to overwrite existing artifact: {path.as_posix()}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _s2_transform_source() -> str:
    """Return the exact standalone transform executed by S2."""

    return (
        '"""Deterministic raw Studio request to sealed ODE snapshot."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import hashlib\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def _canonical(value: object) -> str:\n"
        "    return json.dumps(value, ensure_ascii=False, sort_keys=True, "
        'separators=(",", ":"), allow_nan=False)\n'
        "\n"
        "\n"
        "def transform(payload: dict, *, task_id: str) -> dict:\n"
        "    snapshot = {\n"
        '        "schema_version": "5.2",\n'
        '        "task_id": task_id,\n'
        '        "time_unit": payload["time_unit"],\n'
        '        "state_unit": payload["state_unit"],\n'
        '        "times": payload["times"],\n'
        '        "observations": payload["observations"],\n'
        '        "source_id": payload["source_id"],\n'
        '        "fixture_only": payload["fixture_only"],\n'
        "    }\n"
        '    snapshot["snapshot_hash"] = hashlib.sha256(\n'
        '        _canonical(snapshot).encode("utf-8")\n'
        "    ).hexdigest()\n"
        "    return snapshot\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        raise SystemExit(2)\n"
        "    input_path = Path(sys.argv[1])\n"
        "    output_path = Path(sys.argv[2])\n"
        '    payload = json.loads(input_path.read_text(encoding="utf-8"))\n'
        "    result = transform(payload, task_id=sys.argv[3])\n"
        "    output_path.write_text(\n"
        "        json.dumps(\n"
        "            result,\n"
        "            ensure_ascii=False,\n"
        "            sort_keys=True,\n"
        "            indent=2,\n"
        "            allow_nan=False,\n"
        "        )\n"
        '        + "\\n",\n'
        '        encoding="utf-8",\n'
        '        newline="\\n",\n'
        "    )\n"
        "    return 0\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


def _artifact_map(outcome: RoleProcessOutcomeV51) -> dict[str, str]:
    artifacts = {
        item.artifact_type: item.content
        for item in outcome.draft.proposed_artifacts
    }
    if len(artifacts) != len(outcome.draft.proposed_artifacts):
        raise BackhalfRuntimeError("role returned duplicate artifact types")
    return artifacts


def _artifact_json(
    outcome: RoleProcessOutcomeV51,
    artifact_type: str,
) -> dict[str, Any]:
    artifacts = _artifact_map(outcome)
    if set(artifacts) != {artifact_type}:
        raise BackhalfRuntimeError(
            f"{outcome.request.role_name} must return only {artifact_type}"
        )
    try:
        payload = json.loads(artifacts[artifact_type])
    except json.JSONDecodeError as exc:
        raise BackhalfRuntimeError(
            f"{artifact_type} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise BackhalfRuntimeError(f"{artifact_type} must be a JSON object")
    return payload


def _selected_candidate(
    workspace: StageWorkspaceV50,
) -> tuple[ModelSpecV50, dict[str, Any]]:
    model = ModelSpecV50.model_validate_json(
        (workspace.root / "docs" / "model_spec.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_payload = json.loads(
        (workspace.root / "docs" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = candidate_payload.get("candidates", [])
    selected = next(
        (
            item
            for item in candidates
            if item.get("candidate_id") == model.selected_candidate_id
        ),
        None,
    )
    if not isinstance(selected, dict):
        raise BackhalfRuntimeError("selected S1 candidate is unavailable")
    return model, selected


def _assert_ode_compatible(
    workspace: StageWorkspaceV50,
    model: ModelSpecV50,
    selected: dict[str, Any],
) -> None:
    if selected.get("candidate_id") != model.selected_candidate_id:
        raise BackhalfRuntimeError(
            "selected S1 candidate does not match the frozen model spec"
        )
    try:
        candidate = CandidateFormalizationV50.model_validate(selected)
        intent = RegisteredFamilySearchIntentV62.model_validate_json(
            (workspace.root / EXECUTABLE_CANDIDATE_INTENT_PATH).read_text(
                encoding="utf-8"
            )
        )
        execution_ir = RegisteredFamilySearchIRV62.model_validate_json(
            (workspace.root / EXECUTABLE_CANDIDATE_IR_PATH).read_text(
                encoding="utf-8"
            )
        )
        execution_ir.assert_sealed()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BackhalfRuntimeError(
            "selected S1 candidate lacks a valid typed executable IR"
        ) from exc
    if (
        not model.model_hash
        or model.model_hash != model.content_hash()
        or model.selected_candidate_structural_hash
        != candidate.structural_hash()
        or execution_ir.candidate_id != candidate.candidate_id
        or execution_ir.candidate_structural_hash
        != candidate.structural_hash()
        or execution_ir.model_intent_hash != intent.content_hash()
        or intent.candidate_id != candidate.candidate_id
    ):
        raise BackhalfRuntimeError(
            "selected S1 candidate and typed executable IR are not exactly bound"
        )


def _effective_adapter_id(
    workspace: StageWorkspaceV50,
    request: StudioODEDataRequestV59,
) -> str:
    v67_context = load_v67_s2_contract_v67(workspace)
    if v67_context is not None:
        frozen_adapter = v67_context.protocol.adapter_binding.adapter_id
        if request.adapter_id != frozen_adapter:
            _raise_v67_compatibility(
                workspace,
                failure_owner="capability",
                compatibility_phase="pre_raw_materialization",
                checks={"adapter_matches_frozen_protocol": False},
                measurement=v67_context.measurement,
                protocol=v67_context.protocol,
            )
        return frozen_adapter
    requested = request.adapter_id
    recovery_state = RecoveryKernelV60(workspace).load_state()
    if (
        requested == ODE_ADAPTER_ID
        and recovery_state.last_action == "BRANCH"
        and request.observations
        and len(request.observations) >= 26
    ):
        requested = ADAPTIVE_ADAPTER_ID
    signature = ProblemSignatureV60(
        state_kind="scalar",
        time_kind="continuous",
        dynamics_kind="autonomous",
        observation_kind="complete",
        task_kind="prediction",
        observation_count=len(request.observations),
        positive_observations=all(
            math.isfinite(value) and value > 0
            for value in request.observations
        ),
        strictly_increasing_time=all(
            right > left
            for left, right in zip(request.times, request.times[1:])
        ),
    )
    decision = default_capability_registry_v60().route(signature)
    if requested not in decision.compatible_pack_ids:
        reasons = decision.incompatibilities.get(
            requested, ["capability pack is not registered"]
        )
        raise BackhalfRuntimeError(
            f"{requested} is incompatible with the frozen problem signature: "
            + ", ".join(reasons)
        )
    return requested


def _frozen_adaptive_thresholds(
    workspace: StageWorkspaceV50 | None = None,
) -> tuple[
    HybridODEThresholdsV56,
    AdaptiveThresholdsV57,
]:
    if workspace is not None:
        context = load_v67_s2_contract_v67(workspace)
        if context is not None:
            primary = context.protocol.thresholds.hybrid_thresholds
            adaptive = context.protocol.thresholds.adaptive_thresholds
            if primary is None or adaptive is None:
                _raise_v67_compatibility(
                    workspace,
                    failure_owner="capability",
                    compatibility_phase="s2_replay",
                    checks={
                        "adaptive_thresholds_present_in_frozen_protocol": False
                    },
                    measurement=context.measurement,
                    protocol=context.protocol,
                )
            primary.assert_sealed()
            adaptive.assert_sealed()
            return primary, adaptive
    repository_root = Path(__file__).resolve().parents[2]
    primary_path = repository_root / "V5_6_HYBRID_THRESHOLDS.json"
    adaptive_path = repository_root / "V5_7_ADAPTIVE_THRESHOLDS.json"
    if not primary_path.is_file() or not adaptive_path.is_file():
        raise BackhalfRuntimeError(
            "frozen V5.6/V5.7 threshold files are unavailable"
        )
    primary = HybridODEThresholdsV56.seal(
        **json.loads(primary_path.read_text(encoding="utf-8"))
    )
    adaptive = AdaptiveThresholdsV57.seal(
        **json.loads(adaptive_path.read_text(encoding="utf-8"))
    )
    primary.assert_sealed()
    adaptive.assert_sealed()
    return primary, adaptive


def _validate_v67_request_from_workspace(
    workspace: StageWorkspaceV50,
    request: StudioODEDataRequestV59,
    *,
    compatibility_phase: Literal[
        "pre_raw_materialization", "s2_replay"
    ],
) -> V67S2ContractContext | None:
    context = load_v67_s2_contract_v67(workspace)
    if context is None:
        return None
    root = workspace.root
    required_paths = (
        SOURCE_RECEIPT_PATH,
        SOURCE_ACQUISITION_AUTH_PATH,
        MEASUREMENT_SCHEMA_PATH,
        SOURCE_RAW_PATH,
    )
    presence = {
        relative_path: (root / relative_path).is_file()
        for relative_path in required_paths
    }
    if not all(presence.values()):
        _raise_v67_compatibility(
            workspace,
            failure_owner="data_contract",
            compatibility_phase=compatibility_phase,
            checks={
                "measurement_schema_present": presence[
                    MEASUREMENT_SCHEMA_PATH
                ],
                "source_acquisition_receipt_present": presence[
                    SOURCE_ACQUISITION_AUTH_PATH
                ],
                "source_raw_response_present": presence[SOURCE_RAW_PATH],
                "source_receipt_present": presence[SOURCE_RECEIPT_PATH],
            },
            measurement=context.measurement,
            protocol=context.protocol,
        )
    try:
        source_receipt = WorldBankSourceReceiptV62.model_validate_json(
            (root / SOURCE_RECEIPT_PATH).read_text(encoding="utf-8")
        )
        source_acquisition_receipt = (
            SourceAcquisitionReceiptV62.model_validate_json(
                (root / SOURCE_ACQUISITION_AUTH_PATH).read_text(
                    encoding="utf-8"
                )
            )
        )
        measurement_schema = MeasurementSchemaV62.model_validate_json(
            (root / MEASUREMENT_SCHEMA_PATH).read_text(encoding="utf-8")
        )
        source_raw_body = (root / SOURCE_RAW_PATH).read_bytes()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        try:
            _raise_v67_compatibility(
                workspace,
                failure_owner="data_contract",
                compatibility_phase=compatibility_phase,
                checks={"source_evidence_schema_valid": False},
                measurement=context.measurement,
                protocol=context.protocol,
            )
        except V67S2CompatibilityError as failure:
            raise failure from exc
    return validate_v67_data_compatibility_v67(
        workspace,
        request,
        source_contract=context.source_contract,
        source_receipt=source_receipt,
        source_acquisition_receipt=source_acquisition_receipt,
        measurement_schema=measurement_schema,
        source_raw_body=source_raw_body,
        require_source_evidence=True,
        compatibility_phase=compatibility_phase,
    )


def ingest_ode_data_v59(
    workspace: StageWorkspaceV50,
    request: StudioODEDataRequestV59,
) -> Path:
    if workspace.current_gate("S1") is None:
        raise BackhalfRuntimeError("ODE data intake requires an open S1 gate")
    if workspace.current_gate("S2") is not None:
        raise BackhalfRuntimeError("S2 is already frozen")
    _, selected = _selected_candidate(workspace)
    model = ModelSpecV50.model_validate_json(
        (workspace.root / "docs" / "model_spec.json").read_text(
            encoding="utf-8"
        )
    )
    _assert_ode_compatible(workspace, model, selected)
    _validate_v67_request_from_workspace(
        workspace,
        request,
        compatibility_phase="pre_raw_materialization",
    )
    _effective_adapter_id(workspace, request)
    path = workspace.root / RAW_RELATIVE_PATH
    _write_json_new(path, request.model_dump(mode="json"))
    return path


def _read_bound_file(context: AdapterContextV50, relative_path: str) -> bytes:
    binding = next(
        (
            item
            for item in context.manifest.files
            if item.relative_path == relative_path
        ),
        None,
    )
    if binding is None:
        raise ValueError(f"{relative_path} is absent from the frozen manifest")
    path = (context.workspace_root / relative_path).resolve()
    root = context.workspace_root.resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(relative_path)
    payload = path.read_bytes()
    if (
        len(payload) != binding.size_bytes
        or hashlib.sha256(payload).hexdigest() != binding.sha256
    ):
        raise ValueError(f"{relative_path} differs from the frozen manifest")
    return payload


class StudioODEObligationAdapterV59:
    adapter_id = "studio_scalar_ode_obligation_adapter"
    adapter_version = "5.9"

    def __init__(self, obligation: ValidationObligationV50) -> None:
        self.check_id = obligation.check_id
        self.level = obligation.level

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = ODEScientificBundleV52.model_validate_json(
            _read_bound_file(context, BUNDLE_PATH)
        )
        binding = json.loads(
            _read_bound_file(context, ADAPTER_BINDING_PATH).decode("utf-8")
        )
        if binding.get("adapter_id") != ODE_ADAPTER_ID:
            raise ValueError("S2 adapter binding is not scalar ODE V5.2")
        evidence = next(
            item for item in bundle.levels if item.level == self.level
        )
        adapter_binding = next(
            item
            for item in context.manifest.files
            if item.relative_path == ADAPTER_BINDING_PATH
        )
        payload: dict[str, Any] = {
            "adapter_binding_hash": adapter_binding.sha256,
            "bundle_hash": bundle.bundle_hash,
            "level_evidence": evidence.model_dump(mode="json"),
            "fixture_only": bundle.fixture_only,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        if self.level == "L0":
            code = CodeManifestV50.model_validate_json(
                _read_bound_file(
                    context,
                    "results/code_manifest.json",
                )
            )
            payload["computation_artifact_sha256"] = code.replay_receipt_hash
        return AdapterOutcomeV50(
            status="PASS" if evidence.status == "PASS" else "FAIL",
            reason_code=(
                "studio_scalar_ode_level_passed"
                if evidence.status == "PASS"
                else f"studio_scalar_ode_level_{evidence.status.lower()}"
            ),
            thresholds=evidence.thresholds,
            metrics=evidence.metrics,
            evidence_payloads=[payload],
        )


class StudioAdaptiveObligationAdapterV60:
    adapter_id = "studio_adaptive_positive_series_obligation_adapter"
    adapter_version = "6.0"

    def __init__(self, obligation: ValidationObligationV50) -> None:
        self.check_id = obligation.check_id
        self.level = obligation.level

    def run(self, context: AdapterContextV50) -> AdapterOutcomeV50:
        bundle = AdaptivePositiveSeriesBundleV57.model_validate_json(
            _read_bound_file(context, ADAPTIVE_BUNDLE_PATH)
        )
        binding = json.loads(
            _read_bound_file(context, ADAPTER_BINDING_PATH).decode("utf-8")
        )
        if binding.get("adapter_id") != ADAPTIVE_ADAPTER_ID:
            raise ValueError("S2 adapter binding is not adaptive V5.7")
        evidence = next(
            item for item in bundle.levels if item.level == self.level
        )
        payload: dict[str, Any] = {
            "adapter_binding_hash": next(
                item.sha256
                for item in context.manifest.files
                if item.relative_path == ADAPTER_BINDING_PATH
            ),
            "bundle_hash": bundle.bundle_hash,
            "candidate_graph_hash": bundle.graph.graph_hash,
            "selected_branch": bundle.graph.selected_branch,
            "selected_model_id": bundle.graph.selected_model_id,
            "level_evidence": evidence.model_dump(mode="json"),
            "fixture_only": bundle.fixture_only,
            "causal_mechanism_identified": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        if self.level == "L0":
            code = CodeManifestV50.model_validate_json(
                _read_bound_file(context, "results/code_manifest.json")
            )
            payload["computation_artifact_sha256"] = code.replay_receipt_hash
        return AdapterOutcomeV50(
            status="PASS" if evidence.status == "PASS" else "FAIL",
            reason_code=(
                "studio_adaptive_positive_series_level_passed"
                if evidence.status == "PASS"
                else (
                    "studio_adaptive_positive_series_level_"
                    f"{evidence.status.lower()}"
                )
            ),
            thresholds=evidence.thresholds,
            metrics=evidence.metrics,
            evidence_payloads=[payload],
        )


class StudioBackhalfOrchestratorV59:
    """Drive the registered scalar ODE path from S2 through S6."""

    def __init__(
        self,
        *,
        workspace: StageWorkspaceV50,
        task_id: str,
        driver_factory: DriverFactory,
        event_callback: EventCallback,
    ) -> None:
        self.workspace = workspace
        self.task_id = task_id
        self.driver_factory = driver_factory
        self.event_callback = event_callback
        if workspace.current_gate("S1") is None:
            raise BackhalfRuntimeError("back-half execution requires an open S1 gate")

    def _event(
        self,
        event_type: str,
        status: Literal["accepted", "running", "succeeded", "failed", "blocked"],
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.event_callback(event_type, status, message, details or {})

    def _run_role(
        self,
        *,
        stage: StageId,
        role_name: str,
        role_kind: Literal["generator", "reviewer"],
        subject_id: str,
        objective: str,
        public_inputs: dict[str, Any],
        allowed_candidate_ids: list[str],
    ) -> RoleProcessOutcomeV51:
        return self.driver_factory().run(
            task_id=self.task_id,
            stage=stage,
            role_name=role_name,
            role_kind=role_kind,
            subject_id=subject_id,
            objective=objective,
            public_inputs=public_inputs,
            allowed_candidate_ids=allowed_candidate_ids,
        )

    def _commit_review(
        self,
        *,
        stage: StageId,
        role: str,
        reviewer: RoleProcessOutcomeV51,
        producer_run_id: str,
        producer_context_id: str,
    ) -> None:
        manifest = self.workspace._manifest_for_stage(stage)
        checks = self.workspace._latest_checks(
            stage,
            str(manifest.manifest_hash),
        )
        allowed_inputs = sorted(
            {item.sha256 for item in manifest.files}
            | {
                str(item.result_hash)
                for item in checks.values()
                if item.result_hash is not None
            }
        )
        finding_ids = sorted(
            {
                f"finding-{hashlib.sha256(item.encode('utf-8')).hexdigest()[:16]}"
                for item in reviewer.draft.findings
            }
        )
        trace = self.workspace.commit_evidence(
            "codex_review_transport_trace_v59",
            {
                "stage": stage,
                "role": role,
                "producer_run_id": producer_run_id,
                "reviewer_run_id": reviewer.request.run_id,
                "producer_context_id": producer_context_id,
                "reviewer_context_id": reviewer.request.context_id,
                "context_isolation_attested": True,
                "allowed_input_hashes": allowed_inputs,
                "process_receipt": reviewer.receipt.model_dump(mode="json"),
            },
        )
        output = self.workspace.commit_evidence(
            "codex_review_output_v59",
            {
                "stage": stage,
                "role": role,
                "verdict": reviewer.draft.verdict,
                "finding_ids": finding_ids,
                "draft": reviewer.draft.model_dump(mode="json"),
            },
        )
        self.workspace.issue_review(
            stage=stage,
            review_id=f"review-{reviewer.request.run_id}",
            role=role,
            producer_run_id=producer_run_id,
            reviewer_run_id=reviewer.request.run_id,
            producer_context_id=producer_context_id,
            reviewer_context_id=reviewer.request.context_id,
            prompt_hash=reviewer.receipt.prompt_hash,
            output_schema_hash=reviewer.receipt.output_schema_hash,
            allowed_input_hashes=allowed_inputs,
            transport_trace_hash=trace.sha256,
            output_artifact_hash=output.sha256,
            verdict=reviewer.draft.verdict,
            finding_ids=finding_ids,
            issued_by="verifier",
        )

    def _review_stage(
        self,
        *,
        stage: StageId,
        role: str,
        producer_run_id: str,
        producer_context_id: str,
        summary: dict[str, Any],
    ) -> RoleProcessOutcomeV51:
        manifest = self.workspace._manifest_for_stage(stage)
        checks = self.workspace._latest_checks(
            stage,
            str(manifest.manifest_hash),
        )
        reviewer = self._run_role(
            stage=stage,
            role_name=f"{stage.lower()}_{role}",
            role_kind="reviewer",
            subject_id=f"{stage.lower()}-work",
            objective=(
                f"Independently audit {stage} evidence against its frozen "
                "workflow and scientific boundary."
            ),
            public_inputs={
                "manifest": manifest.model_dump(mode="json"),
                "checks": {
                    key: value.model_dump(mode="json")
                    for key, value in checks.items()
                },
                "summary": summary,
                "gate_policy_hash": POLICIES[stage].policy_hash,
                "review_rule": (
                    "APPROVE only when the supplied frozen evidence is coherent "
                    "and every reserved authority remains false. Scientific "
                    "FAIL or NOT_RUN must not be reinterpreted as PASS."
                ),
            },
            allowed_candidate_ids=[],
        )
        if reviewer.draft.authority_claimed:
            raise BackhalfRuntimeError(f"{stage} reviewer claimed authority")
        if reviewer.draft.proposed_artifacts:
            raise BackhalfRuntimeError(f"{stage} reviewer proposed artifacts")
        self._commit_review(
            stage=stage,
            role=role,
            reviewer=reviewer,
            producer_run_id=producer_run_id,
            producer_context_id=producer_context_id,
        )
        return reviewer

    def _evaluate(
        self,
        *,
        stage: StageId,
        producer_run_id: str,
        producer_context_id: str,
        summary: dict[str, Any],
        extra_paths: list[str] | None = None,
        scientific_obligations: list[ValidationObligationV50] | None = None,
    ) -> str:
        actor: Literal["model", "harness"] = (
            "model" if stage in {"S2", "S5"} else "harness"
        )
        manifest = self.workspace.submit_stage(
            stage,
            actor=actor,
            extra_paths=extra_paths or [],
        )
        mechanical = self.workspace.run_mechanical_check(stage)
        if mechanical.status != "PASS":
            self._event(
                f"{stage.lower()}_mechanical_check_failed",
                "blocked",
                f"{stage} failed its harness-owned mechanical check",
                {
                    "reasons": mechanical.metrics,
                    "manifest_hash": manifest.manifest_hash,
                },
            )
            return "BLOCKED"
        if scientific_obligations:
            registry = CheckRegistryV50()
            binding_payload = json.loads(
                (self.workspace.root / ADAPTER_BINDING_PATH).read_text(
                    encoding="utf-8"
                )
            )
            adapter_id = binding_payload.get("adapter_id")
            for obligation in scientific_obligations:
                if obligation.applicability == "applicable":
                    if adapter_id == ODE_ADAPTER_ID:
                        registry.register(
                            StudioODEObligationAdapterV59(obligation)
                        )
                    elif adapter_id == ADAPTIVE_ADAPTER_ID:
                        registry.register(
                            StudioAdaptiveObligationAdapterV60(obligation)
                        )
                    else:
                        raise BackhalfRuntimeError(
                            "S2 adapter binding is not registered in Studio"
                        )
            for obligation in scientific_obligations:
                registry.execute(self.workspace, obligation)
        reviewers: dict[str, str] = {}
        for role in POLICIES[stage].required_review_roles:
            outcome = self._review_stage(
                stage=stage,
                role=role,
                producer_run_id=producer_run_id,
                producer_context_id=producer_context_id,
                summary=summary,
            )
            reviewers[role] = outcome.draft.verdict
        evaluation = self.workspace.evaluate_gate(stage)
        self._event(
            f"{stage.lower()}_gate_evaluated",
            "succeeded" if evaluation.decision == "OPEN" else "blocked",
            (
                f"{stage} gate opened"
                if evaluation.decision == "OPEN"
                else f"{stage} stopped: {evaluation.decision}"
            ),
            {
                "decision": evaluation.decision,
                "reasons": evaluation.reasons,
                "review_verdicts": reviewers,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        return evaluation.decision

    def run_s2(self) -> str:
        if self.workspace.current_gate("S2"):
            return "OPEN"
        root = self.workspace.root
        raw_path = root / RAW_RELATIVE_PATH
        if not raw_path.is_file():
            raise BackhalfRuntimeError(
                "S2 requires user-supplied scalar ODE data"
            )
        for relative in (
            "data/ledger.json",
            "data/processed/manifest.json",
            PROCESSED_RELATIVE_PATH,
            ADAPTER_BINDING_PATH,
            SUCCESS_CONTRACT_PATH,
            DECISION_CONTRACT_PATH,
            EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
            EXECUTABLE_CANDIDATE_RESOLUTION_PATH_V67,
        ):
            if (root / relative).exists():
                raise BackhalfRuntimeError(
                    "S2 contains partial artifacts; automatic rerun is blocked"
                )
        request = StudioODEDataRequestV59.model_validate_json(
            raw_path.read_text(encoding="utf-8")
        )
        v67_context = _validate_v67_request_from_workspace(
            self.workspace,
            request,
            compatibility_phase="s2_replay",
        )
        source_paths = (
            SOURCE_CONTRACT_PATH,
            SOURCE_RAW_PATH,
            SOURCE_RECEIPT_PATH,
            SOURCE_ACQUISITION_AUTH_PATH,
            MEASUREMENT_SCHEMA_PATH,
        )
        source_presence = [
            (root / relative_path).is_file()
            for relative_path in source_paths
        ]
        if any(source_presence) and not all(source_presence):
            raise BackhalfRuntimeError(
                "official-source evidence is incomplete"
            )
        official_source = all(source_presence)
        source_contract = (
            WorldBankSourceContractV62.model_validate_json(
                (root / SOURCE_CONTRACT_PATH).read_text(encoding="utf-8")
            )
            if official_source
            else None
        )
        source_receipt = (
            WorldBankSourceReceiptV62.model_validate_json(
                (root / SOURCE_RECEIPT_PATH).read_text(encoding="utf-8")
            )
            if official_source
            else None
        )
        source_acquisition_receipt = (
            SourceAcquisitionReceiptV62.model_validate_json(
                (root / SOURCE_ACQUISITION_AUTH_PATH).read_text(
                    encoding="utf-8"
                )
            )
            if official_source
            else None
        )
        measurement_schema = (
            MeasurementSchemaV62.model_validate_json(
                (root / MEASUREMENT_SCHEMA_PATH).read_text(
                    encoding="utf-8"
                )
            )
            if official_source
            else None
        )
        model, selected = _selected_candidate(self.workspace)
        _assert_ode_compatible(self.workspace, model, selected)
        adapter_id = _effective_adapter_id(self.workspace, request)
        success_contract = default_scientific_success_contract_v61(
            workspace_spec_hash=str(self.workspace.spec.spec_hash),
            adapter_id=adapter_id,
        )
        if (
            v67_context is not None
            and success_contract.thresholds
            != v67_context.protocol.thresholds.scientific_success_thresholds
        ):
            _raise_v67_compatibility(
                self.workspace,
                failure_owner="capability",
                compatibility_phase="s2_replay",
                checks={
                    "scientific_success_thresholds_match_frozen_protocol": False
                },
                measurement=v67_context.measurement,
                protocol=v67_context.protocol,
            )
        baseline = self.workspace._raw_baseline_for_current_s2()
        if baseline is None:
            baseline = self.workspace.freeze_raw_inputs(actor="harness")
        candidate_set = CandidateSetV50.model_validate_json(
            (root / "docs" / "candidates.json").read_text(encoding="utf-8")
        )
        execution_ir = RegisteredFamilySearchIRV62.model_validate_json(
            (root / EXECUTABLE_CANDIDATE_IR_PATH).read_text(encoding="utf-8")
        )
        execution_resolution = resolve_executable_candidate_v62(
            workspace=self.workspace,
            execution_ir=execution_ir,
            candidate_set=candidate_set,
            model_spec=model,
            adapter_id=adapter_id,
        )
        _write_json_new(
            root / EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
            execution_resolution.model_dump(mode="json"),
        )
        execution_resolution_v67: ExecutableCandidateResolutionV67 | None = None
        if v67_context is not None:
            execution_resolution_v67 = ExecutableCandidateResolutionV67.seal(
                workspace_spec_hash=str(self.workspace.spec.spec_hash),
                s0_gate_hash=v67_context.measurement.s0_gate_hash,
                s1_gate_hash=str(self.workspace.current_gate("S1")),
                s2_attempt=baseline.s2_attempt,
                candidate_id=model.selected_candidate_id,
                candidate_structural_hash=(
                    model.selected_candidate_structural_hash
                ),
                legacy_v62_resolution_hash=str(
                    execution_resolution.resolution_hash
                ),
                measurement_contract_hash=str(
                    v67_context.measurement.contract_hash
                ),
                predata_protocol_hash=str(
                    v67_context.protocol.protocol_hash
                ),
                source_contract_hash=str(
                    v67_context.source_contract.contract_hash
                ),
                candidate_execution_binding_hash=str(
                    v67_context.candidate_binding.binding_hash
                ),
                adapter_id=adapter_id,
                adapter_version=(
                    v67_context.protocol.adapter_binding.adapter_version
                ),
                capability_pack_hash=(
                    v67_context.protocol.adapter_binding.capability_pack_hash
                ),
                threshold_hashes=(
                    v67_context.protocol.thresholds.threshold_hashes()
                ),
            )
            _write_json_new(
                root / EXECUTABLE_CANDIDATE_RESOLUTION_PATH_V67,
                execution_resolution_v67.model_dump(mode="json"),
            )
        snapshot = ODETimeSeriesSnapshotV52.seal(
            task_id=self.task_id,
            time_unit=request.time_unit,
            state_unit=request.state_unit,
            times=request.times,
            observations=request.observations,
            source_id=request.source_id,
            fixture_only=request.fixture_only,
        )
        source_verification: SourceVerificationV62 | None = None
        source_reverification: S2SourceReverificationReceiptV62 | None = None
        if (
            source_contract is not None
            and source_receipt is not None
            and source_acquisition_receipt is not None
        ):
            if (
                (root / SOURCE_VERIFICATION_PATH).exists()
                or (root / S2_SOURCE_REVERIFICATION_PATH).exists()
            ):
                raise BackhalfRuntimeError(
                    "S2 source re-verification artifacts already exist"
                )
            source_authority = (
                SourceTransportAuthorityV62.from_stage_workspace(
                    self.workspace
                )
            )
            replay = source_authority.reverify_world_bank_source_at_s2(
                workspace=self.workspace,
                raw_baseline=baseline,
                contract=source_contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                acquisition_receipt=source_acquisition_receipt,
            )
            source_verification = replay.verification
            source_reverification = replay.authority_receipt
            if not source_authority.is_s2_reverification_admissible(
                workspace=self.workspace,
                raw_baseline=baseline,
                contract=source_contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                acquisition_receipt=source_acquisition_receipt,
                verification=source_verification,
                receipt=source_reverification,
            ):
                raise BackhalfRuntimeError(
                    "official source failed authenticated current-S2 replay"
                )
            _write_json_new(
                root / SOURCE_VERIFICATION_PATH,
                source_verification.model_dump(mode="json"),
            )
            _write_json_new(
                root / S2_SOURCE_REVERIFICATION_PATH,
                source_reverification.model_dump(mode="json"),
            )
        required_ids = sorted(set(model.data_requirement_ids))
        producer = self._run_role(
            stage="S2",
            role_name="s2_data_steward",
            role_kind="generator",
            subject_id=model.selected_candidate_id,
            objective=(
                "Map the harness-frozen positive scalar series to every selected "
                "model data requirement without changing bytes or claiming quality."
            ),
            public_inputs={
                "objective": self.workspace.spec.objective,
                "selected_candidate": selected,
                "raw_baseline_hash": baseline.baseline_hash,
                "data_summary": {
                    "adapter_id": adapter_id,
                    "point_count": len(request.times),
                    "time_unit": request.time_unit,
                    "state_unit": request.state_unit,
                    "source_id": request.source_id,
                    "fixture_only": request.fixture_only,
                },
                "required_data_requirement_ids": required_ids,
                "required_artifacts": {
                    "data_mapping": DataMappingDraftV59.model_json_schema()
                },
            },
            allowed_candidate_ids=[model.selected_candidate_id],
        )
        if producer.draft.authority_claimed:
            raise BackhalfRuntimeError("S2 data steward claimed authority")
        mapping = DataMappingDraftV59.model_validate(
            _artifact_json(producer, "data_mapping")
        )
        if mapping.data_requirement_ids != required_ids:
            raise BackhalfRuntimeError(
                "S2 data mapping does not cover the selected model requirements"
            )
        commit_generator_outcome_v51(
            self.workspace,
            producer,
            execution_role="modeler",
            input_authority_hash=str(self.workspace.current_gate("S1")),
        )
        transform_path = root / "src" / "models" / "prepare_ode_data.py"
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        if transform_path.exists():
            raise BackhalfRuntimeError("S2 transform already exists")
        transform_path.write_text(
            _s2_transform_source(),
            encoding="utf-8",
            newline="\n",
        )
        processed_path = root / PROCESSED_RELATIVE_PATH
        if processed_path.exists():
            raise BackhalfRuntimeError("S2 processed snapshot already exists")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_processed = processed_path.with_name(
            f".{processed_path.name}.{uuid4().hex}.tmp"
        )
        command = [
            sys.executable,
            str(transform_path),
            str(raw_path),
            str(temporary_processed),
            self.task_id,
        ]
        transform_started = datetime.now(timezone.utc)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            temporary_processed.unlink(missing_ok=True)
            raise BackhalfRuntimeError(
                "S2 transform fresh-process execution timed out"
            ) from exc
        transform_finished = datetime.now(timezone.utc)
        if completed.returncode != 0 or not temporary_processed.is_file():
            temporary_processed.unlink(missing_ok=True)
            raise BackhalfRuntimeError(
                "S2 transform fresh-process execution failed"
            )
        replayed_snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
            temporary_processed.read_text(encoding="utf-8")
        )
        replayed_snapshot.assert_sealed()
        if replayed_snapshot.model_dump(mode="json") != snapshot.model_dump(
            mode="json"
        ):
            temporary_processed.unlink(missing_ok=True)
            raise BackhalfRuntimeError(
                "S2 transform output differs from code-owned snapshot"
            )
        temporary_processed.replace(processed_path)
        processed_hash = _sha(processed_path)
        transform_receipt = S2TransformReceiptV62.seal(
            workspace_spec_hash=self.workspace.spec.spec_hash,
            raw_baseline_hash=baseline.baseline_hash,
            s2_attempt=baseline.s2_attempt,
            task_id=self.task_id,
            input_relative_path=RAW_RELATIVE_PATH,
            input_hash=_sha(raw_path),
            transform_relative_path="src/models/prepare_ode_data.py",
            transform_hash=_sha(transform_path),
            output_relative_path=PROCESSED_RELATIVE_PATH,
            output_hash=processed_hash,
            command=command,
            runtime_identity=(
                f"{platform.python_implementation()} "
                f"{platform.python_version()} @ {sys.executable}"
            ),
            stdout_hash=hashlib.sha256(completed.stdout).hexdigest(),
            stderr_hash=hashlib.sha256(completed.stderr).hexdigest(),
            started_at=transform_started,
            finished_at=transform_finished,
        )
        _write_json_new(
            root / S2_TRANSFORM_RECEIPT_PATH,
            transform_receipt.model_dump(mode="json"),
        )
        transform_params = {
            "adapter_id": adapter_id,
            "task_id": self.task_id,
            "input_schema": "5.9",
            "output_schema": "5.2",
            "identity_observations": True,
            "drop_missing": False,
        }
        accessed_at = datetime.now(timezone.utc)
        entries = [
            DataLedgerEntryV50(
                data_item_id=requirement_id,
                semantic_name=f"{mapping.semantic_name}: {requirement_id}",
                units=mapping.units,
                source_kind="official" if official_source else "user",
                source_ref=(
                    source_receipt.source_id
                    if source_receipt is not None
                    else request.source_id
                ),
                raw_relative_path=RAW_RELATIVE_PATH,
                accessed_at=accessed_at,
                license_status=(
                    (
                        f"{source_contract.license_id};"
                        "independent_license_review_absent"
                    )
                    if source_contract is not None
                    else request.license_status
                ),
                raw_response_hash=_sha(raw_path),
                transform_script_relative_path="src/models/prepare_ode_data.py",
                transform_script_hash=_sha(transform_path),
                transform_params=transform_params,
                transform_params_hash=sha256_value(transform_params),
                processed_artifact_hash=processed_hash,
                quality_flags=[
                    *mapping.quality_flags,
                    "positive_scalar_series_contract",
                    *(
                        ["official_source_transport_verified_v62"]
                        if official_source
                        else []
                    ),
                ],
            )
            for requirement_id in required_ids
        ]
        ledger = DataLedgerV50.seal(
            entries=entries,
            raw_baseline_tree_hash=baseline.raw_tree_hash,
        )
        _write_json_new(
            root / "data" / "ledger.json",
            ledger.model_dump(mode="json"),
        )
        _write_json_new(
            root / "data" / "processed" / "manifest.json",
            ProcessedManifestV50(
                raw_baseline_tree_hash=baseline.raw_tree_hash,
                artifacts=[
                    ProcessedArtifactV50(
                        data_item_id=requirement_id,
                        relative_path=PROCESSED_RELATIVE_PATH,
                        artifact_hash=processed_hash,
                    )
                    for requirement_id in required_ids
                ],
            ).model_dump(mode="json"),
        )
        _write_json_new(
            root / ADAPTER_BINDING_PATH,
            {
                "schema_version": (
                    "6.7" if v67_context is not None else "6.0"
                ),
                "adapter_id": adapter_id,
                **(
                    {
                        "adapter_resolution_authority": (
                            "predata_execution_protocol_v67"
                        ),
                        "adapter_resolution_stage": "pre_data_compiler",
                        "adapter_version": (
                            v67_context.protocol.adapter_binding.adapter_version
                        ),
                        "capability_pack_hash": (
                            v67_context.protocol.adapter_binding.capability_pack_hash
                        ),
                        "candidate_execution_binding_hash": (
                            v67_context.candidate_binding.binding_hash
                        ),
                        "measurement_contract_hash": (
                            v67_context.measurement.contract_hash
                        ),
                        "predata_protocol_hash": (
                            v67_context.protocol.protocol_hash
                        ),
                        "executable_resolution_v67_hash": (
                            execution_resolution_v67.resolution_hash
                            if execution_resolution_v67 is not None
                            else None
                        ),
                        "source_contract_hash": (
                            v67_context.source_contract.contract_hash
                        ),
                        "threshold_hashes": (
                            v67_context.protocol.thresholds.threshold_hashes()
                        ),
                        "s2_role": "compatibility_validation_only",
                        "silent_adapter_substitution_permitted": False,
                        "recovery_requires_new_graph_attempt": True,
                        "recovery_requires_successor_protocol": True,
                    }
                    if v67_context is not None
                    else {}
                ),
                "selected_candidate_id": model.selected_candidate_id,
                "selected_candidate_structural_hash": (
                    model.selected_candidate_structural_hash
                ),
                "registered_families": (
                    [
                        "constant",
                        "exponential",
                        "gompertz",
                        "logistic",
                    ]
                    if adapter_id == ODE_ADAPTER_ID
                    else [
                        "constant",
                        "exponential",
                        "gompertz",
                        "logistic",
                        "log_random_walk_drift",
                        "log_growth_ar1",
                    ]
                ),
                "raw_baseline_hash": baseline.baseline_hash,
                "execution_ir_hash": execution_ir.ir_hash,
                "execution_resolution_hash": (
                    execution_resolution.resolution_hash
                ),
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        _write_json_new(
            root / SUCCESS_CONTRACT_PATH,
            success_contract.model_dump(mode="json"),
        )
        decision_contract: DecisionValueContractV62 | None = None
        decision_intent_path = root / DECISION_INTENT_PATH
        if decision_intent_path.is_file():
            decision_intent = DecisionValueIntentV62.model_validate_json(
                decision_intent_path.read_text(encoding="utf-8")
            )
            decision_intent.assert_sealed()
            if decision_intent.action_unit != snapshot.state_unit:
                raise BackhalfRuntimeError(
                    "decision intent action unit differs from the frozen "
                    "observation unit"
                )
            decision_contract = decision_contract_from_intent_v62(
                workspace_spec_hash=str(self.workspace.spec.spec_hash),
                success_contract=success_contract,
                intent=decision_intent,
            )
            _write_json_new(
                root / DECISION_CONTRACT_PATH,
                decision_contract.model_dump(mode="json"),
            )
        provenance_binding: DataProvenanceBindingV62 | None = None
        if (
            source_contract is not None
            and source_receipt is not None
            and source_verification is not None
            and source_acquisition_receipt is not None
            and source_reverification is not None
            and measurement_schema is not None
        ):
            provenance_binding = build_data_provenance_binding_v62(
                workspace=self.workspace,
                raw_baseline=baseline,
                ledger=ledger,
                snapshot=snapshot,
                source_contract=source_contract,
                source_receipt=source_receipt,
                source_verification=source_verification,
                source_acquisition_receipt=source_acquisition_receipt,
                s2_source_reverification_receipt=source_reverification,
                source_authority=(
                    SourceTransportAuthorityV62.from_stage_workspace(
                        self.workspace
                    )
                ),
                measurement_schema=measurement_schema,
                transform_receipt=transform_receipt,
                processed_snapshot_relative_path=PROCESSED_RELATIVE_PATH,
            )
            _write_json_new(
                root / PROVENANCE_BINDING_PATH,
                provenance_binding.model_dump(mode="json"),
            )
            if provenance_binding.status != "PASS":
                raise BackhalfRuntimeError(
                    "official source failed S2 provenance binding: "
                    + ", ".join(provenance_binding.reason_codes)
                )
        self._event(
            "s2_data_materialized",
            "succeeded",
            "Frozen user data were mapped to the selected registered capability pack",
            {
                "point_count": len(request.times),
                "data_requirement_count": len(required_ids),
                "adapter_id": adapter_id,
                "raw_baseline_hash": baseline.baseline_hash,
                "fixture_only": request.fixture_only,
                "source_integrity_status": (
                    provenance_binding.status
                    if provenance_binding is not None
                    else "NOT_RUN"
                ),
                "scientific_provenance_status": (
                    provenance_binding.scientific_provenance_status
                    if provenance_binding is not None
                    else "HUMAN"
                ),
                "decision_value_contract_frozen": (
                    decision_contract is not None
                ),
                "executable_candidate_resolution_hash": (
                    execution_resolution.resolution_hash
                ),
                "measurement_contract_hash_v67": (
                    v67_context.measurement.contract_hash
                    if v67_context is not None
                    else None
                ),
                "predata_protocol_hash_v67": (
                    v67_context.protocol.protocol_hash
                    if v67_context is not None
                    else None
                ),
                "executable_candidate_resolution_hash_v67": (
                    execution_resolution_v67.resolution_hash
                    if execution_resolution_v67 is not None
                    else None
                ),
            },
        )
        extra_paths = [
            ADAPTER_BINDING_PATH,
            SUCCESS_CONTRACT_PATH,
            S2_TRANSFORM_RECEIPT_PATH,
            EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
        ]
        if decision_contract is not None:
            extra_paths.append(DECISION_CONTRACT_PATH)
        if v67_context is not None:
            extra_paths.extend(
                [
                    MEASUREMENT_STUDY_DESIGN_PATH_V67,
                    PREDATA_EXECUTION_PROTOCOL_PATH_V67,
                    SOURCE_CONTRACT_PATH,
                    CANDIDATE_EXECUTION_BINDING_PATH_V67,
                    EXECUTABLE_CANDIDATE_RESOLUTION_PATH_V67,
                ]
            )
        if provenance_binding is not None:
            extra_paths.extend(
                [
                    *source_paths,
                    SOURCE_VERIFICATION_PATH,
                    S2_SOURCE_REVERIFICATION_PATH,
                    PROVENANCE_BINDING_PATH,
                ]
            )
        return self._evaluate(
            stage="S2",
            producer_run_id=producer.request.run_id,
            producer_context_id=producer.request.context_id,
            summary={
                "adapter_id": adapter_id,
                "point_count": len(request.times),
                "fixture_only": request.fixture_only,
                "mapping": mapping.model_dump(mode="json"),
            },
            extra_paths=extra_paths,
        )

    def _current_adapter_id(self) -> str:
        payload = json.loads(
            (self.workspace.root / ADAPTER_BINDING_PATH).read_text(
                encoding="utf-8"
            )
        )
        adapter_id = str(payload.get("adapter_id", ""))
        if adapter_id not in {ODE_ADAPTER_ID, ADAPTIVE_ADAPTER_ID}:
            raise BackhalfRuntimeError("S2 selected an unknown capability pack")
        return adapter_id

    def _materialize_adaptive_s3(
        self,
    ) -> tuple[AdaptivePositiveSeriesBundleV57, ValidationPlanV50]:
        root = self.workspace.root
        for relative in (
            ADAPTIVE_BUNDLE_PATH,
            ADAPTIVE_REPLAY_INPUT_PATH,
            ADAPTIVE_REPLAY_RECEIPTS_PATH,
            ADAPTIVE_REPLAY_SUMMARY_PATH,
            EXECUTABLE_CANDIDATE_RECEIPT_PATH,
            "results/index.json",
            "results/code_manifest.json",
        ):
            if (root / relative).exists():
                raise BackhalfRuntimeError(
                    "S3 contains partial artifacts; automatic rerun is blocked"
                )
        snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
            (root / PROCESSED_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        primary, adaptive = _frozen_adaptive_thresholds(self.workspace)
        _write_json_new(
            root / ADAPTIVE_REPLAY_INPUT_PATH,
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "primary_thresholds": primary.model_dump(mode="json"),
                "adaptive_thresholds": adaptive.model_dump(mode="json"),
            },
        )
        replay_authority = AdaptiveReplayAuthorityV57(
            key_id=f"{self.workspace.authority_key_id}.adaptive",
            secret=self.workspace._authority_key,
        )
        receipts = run_authenticated_adaptive_replays_v57(
            root / ADAPTIVE_REPLAY_INPUT_PATH,
            authority=replay_authority,
        )
        _write_json_new(
            root / ADAPTIVE_REPLAY_RECEIPTS_PATH,
            [item.model_dump(mode="json") for item in receipts],
        )
        bundle = build_adaptive_positive_series_bundle_v57(
            snapshot=snapshot,
            primary_thresholds=primary,
            adaptive_thresholds=adaptive,
            replay_receipts=receipts,
            replay_authority=replay_authority,
        )
        _write_json_new(
            root / ADAPTIVE_BUNDLE_PATH,
            bundle.model_dump(mode="json"),
        )
        source_path = (
            root / "src" / "models" / "run_adaptive_positive_series.py"
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            raise BackhalfRuntimeError("S3 adaptive adapter source already exists")
        source_path.write_text(
            "\"\"\"Registered execution entrypoint: "
            "fma.v5_7.adaptive_positive_series.\"\"\"\n"
            f'ADAPTER_ID = "{ADAPTIVE_ADAPTER_ID}"\n',
            encoding="utf-8",
            newline="\n",
        )
        environment_path = root / "results" / "environment.json"
        fermi_path = root / "results" / "fermi_estimate.json"
        toy_path = root / "checks" / "ode_toy_oracle.json"
        _write_json_new(
            environment_path,
            {
                "schema_version": "6.0-environment",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "adapter_id": ADAPTIVE_ADAPTER_ID,
            },
        )
        candidate_count = (
            len(bundle.primary_bundle.candidates)
            + len(bundle.growth_candidates)
        )
        _write_json_new(
            fermi_path,
            {
                "schema_version": "6.0-fermi",
                "observation_count": len(snapshot.times),
                "registered_family_count": candidate_count,
                "fit_scale": len(snapshot.times) * candidate_count,
                "candidate_graph_hash": bundle.graph.graph_hash,
            },
        )
        l2 = next(item for item in bundle.levels if item.level == "L2")
        _write_json_new(toy_path, l2.model_dump(mode="json"))
        replay_command = (
            "python -m fma.v5_7.adaptive_positive_series replay "
            "checks/adaptive_replay_input.json"
        )
        source_tree_hash = _tree_hash(root / "src")
        _write_json_new(
            root / ADAPTIVE_REPLAY_SUMMARY_PATH,
            {
                "schema_version": "6.0-adaptive-replay",
                "replay_command": replay_command,
                "source_tree_hash": source_tree_hash,
                "environment_hash": _sha(environment_path),
                "random_seed": adaptive.bootstrap_seed,
                "exit_code": 0,
                "passed": (
                    len(receipts) == 2
                    and len(
                        {
                            item.deterministic_output_hash
                            for item in receipts
                        }
                    )
                    == 1
                    and all(replay_authority.verify(item) for item in receipts)
                ),
                "authenticated_receipts_ref": (
                    ADAPTIVE_REPLAY_RECEIPTS_PATH
                ),
                "authenticated_receipts_hash": _sha(
                    root / ADAPTIVE_REPLAY_RECEIPTS_PATH
                ),
                "authenticated_receipt_hashes": [
                    item.receipt_hash for item in receipts
                ],
                "deterministic_output_hashes": [
                    item.deterministic_output_hash for item in receipts
                ],
            },
        )
        _write_json_new(
            root / "results" / "code_manifest.json",
            CodeManifestV50(
                source_tree_hash=source_tree_hash,
                environment_ref="results/environment.json",
                environment_hash=_sha(environment_path),
                replay_command=replay_command,
                replay_receipt_ref=ADAPTIVE_REPLAY_SUMMARY_PATH,
                replay_receipt_hash=_sha(
                    root / ADAPTIVE_REPLAY_SUMMARY_PATH
                ),
                random_seed=adaptive.bootstrap_seed,
                tolerance_policy=(
                    "Frozen V5.6 primary and V5.7 adaptive thresholds"
                ),
                fermi_estimate_ref="results/fermi_estimate.json",
                fermi_estimate_hash=_sha(fermi_path),
                toy_oracle_refs=["checks/ode_toy_oracle.json"],
                toy_oracle_hashes={
                    "checks/ode_toy_oracle.json": _sha(toy_path)
                },
            ).model_dump(mode="json"),
        )
        if bundle.graph.selected_branch == "log_growth":
            selected = next(
                item
                for item in bundle.growth_candidates
                if item.candidate_id == bundle.graph.selected_model_id
            )
            forecast = selected.forecast_value
        elif bundle.graph.selected_branch == "hybrid_ode":
            selected_primary = next(
                item
                for item in bundle.primary_bundle.candidates
                if item.candidate_id == bundle.graph.selected_model_id
            )
            forecast = selected_primary.forecast_value
        else:
            diagnostic_growth = next(
                (
                    item
                    for item in bundle.growth_candidates
                    if item.candidate_id == bundle.graph.selected_model_id
                ),
                None,
            )
            if diagnostic_growth is not None:
                forecast = diagnostic_growth.forecast_value
            else:
                selected_primary = next(
                    item
                    for item in bundle.primary_bundle.candidates
                    if item.candidate_id == bundle.graph.selected_model_id
                )
                forecast = selected_primary.forecast_value
        l4 = next(item for item in bundle.levels if item.level == "L4")
        low = l4.metrics.get("forecast_interval_low")
        high = l4.metrics.get("forecast_interval_high")
        candidate_forecasts = [
            item.forecast_value for item in bundle.primary_bundle.candidates
        ] + [item.forecast_value for item in bundle.growth_candidates]
        if not isinstance(low, (int, float)) or not isinstance(
            high, (int, float)
        ):
            low = min(candidate_forecasts)
            high = max(candidate_forecasts)
        point_path = root / "results" / "artifacts" / "forecast.json"
        interval_path = (
            root / "results" / "artifacts" / "forecast_interval.json"
        )
        _write_json_new(
            point_path,
            {
                "schema_version": "6.0-result",
                "result_id": "forecast",
                "value": forecast,
                "interval_low": None,
                "interval_high": None,
                "units": snapshot.state_unit,
            },
        )
        _write_json_new(
            interval_path,
            {
                "schema_version": "6.0-result",
                "result_id": "forecast_interval",
                "value": None,
                "interval_low": float(low),
                "interval_high": float(high),
                "units": snapshot.state_unit,
            },
        )
        _write_json_new(
            root / "results" / "index.json",
            ResultIndexV50(
                records=[
                    ResultRecordV50(
                        result_id="forecast",
                        relative_path="results/artifacts/forecast.json",
                        artifact_hash=_sha(point_path),
                        value=forecast,
                        units=snapshot.state_unit,
                    ),
                    ResultRecordV50(
                        result_id="forecast_interval",
                        relative_path=(
                            "results/artifacts/forecast_interval.json"
                        ),
                        artifact_hash=_sha(interval_path),
                        interval_low=float(low),
                        interval_high=float(high),
                        units=snapshot.state_unit,
                    ),
                ]
            ).model_dump(mode="json"),
        )
        plan = ValidationPlanV50.model_validate_json(
            (root / "docs" / "validation_plan.json").read_text(
                encoding="utf-8"
            )
        )
        return bundle, plan

    def _materialize_s3(self) -> tuple[ODEScientificBundleV52, ValidationPlanV50]:
        root = self.workspace.root
        for relative in (
            BUNDLE_PATH,
            REPLAY_INPUT_PATH,
            EXECUTABLE_CANDIDATE_RECEIPT_PATH,
            "results/index.json",
            "results/code_manifest.json",
        ):
            if (root / relative).exists():
                raise BackhalfRuntimeError(
                    "S3 contains partial artifacts; automatic rerun is blocked"
                )
        snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
            (root / PROCESSED_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        v67_context = load_v67_s2_contract_v67(self.workspace)
        if v67_context is not None:
            frozen_ode_thresholds = (
                v67_context.protocol.thresholds.ode_thresholds
            )
            if frozen_ode_thresholds is None:
                _raise_v67_compatibility(
                    self.workspace,
                    failure_owner="capability",
                    compatibility_phase="s2_replay",
                    checks={
                        "ode_thresholds_present_in_frozen_protocol": False
                    },
                    measurement=v67_context.measurement,
                    protocol=v67_context.protocol,
                )
            thresholds = frozen_ode_thresholds
        else:
            thresholds = ODEThresholdsV52.seal(bootstrap_replicates=40)
        thresholds.assert_sealed()
        _write_json_new(
            root / REPLAY_INPUT_PATH,
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "thresholds": thresholds.model_dump(mode="json"),
            },
        )
        replay_hashes = run_ode_replays_v52(root / REPLAY_INPUT_PATH)
        bundle = build_ode_bundle_v52(
            snapshot=snapshot,
            thresholds=thresholds,
            replay_output_hashes=replay_hashes,
        )
        _write_json_new(root / BUNDLE_PATH, bundle.model_dump(mode="json"))
        source_path = root / "src" / "models" / "run_scalar_ode.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.exists():
            raise BackhalfRuntimeError("S3 adapter source already exists")
        source_path.write_text(
            "\"\"\"Registered execution entrypoint: fma.v5_2.ode_system.\"\"\"\n"
            "ADAPTER_ID = \"scalar_autonomous_ode_v52\"\n",
            encoding="utf-8",
            newline="\n",
        )
        environment_path = root / "results" / "environment.json"
        fermi_path = root / "results" / "fermi_estimate.json"
        toy_path = root / "checks" / "ode_toy_oracle.json"
        replay_receipt_path = root / "checks" / "ode_replay_receipt.json"
        _write_json_new(
            environment_path,
            {
                "schema_version": "5.9-environment",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "adapter_id": ODE_ADAPTER_ID,
            },
        )
        _write_json_new(
            fermi_path,
            {
                "schema_version": "5.9-fermi",
                "observation_count": len(snapshot.times),
                "registered_family_count": len(bundle.candidates),
                "fit_scale": len(snapshot.times) * len(bundle.candidates),
            },
        )
        l2 = next(item for item in bundle.levels if item.level == "L2")
        _write_json_new(toy_path, l2.model_dump(mode="json"))
        replay_command = (
            "python -m fma.v5_2.ode_system replay "
            "checks/ode_replay_input.json"
        )
        source_tree_hash = _tree_hash(root / "src")
        _write_json_new(
            replay_receipt_path,
            {
                "schema_version": "5.9-replay",
                "replay_command": replay_command,
                "source_tree_hash": source_tree_hash,
                "environment_hash": _sha(environment_path),
                "random_seed": 104729,
                "exit_code": 0,
                "passed": len(replay_hashes) == 2
                and len(set(replay_hashes)) == 1,
                "deterministic_output_hashes": replay_hashes,
            },
        )
        _write_json_new(
            root / "results" / "code_manifest.json",
            CodeManifestV50(
                source_tree_hash=source_tree_hash,
                environment_ref="results/environment.json",
                environment_hash=_sha(environment_path),
                replay_command=replay_command,
                replay_receipt_ref="checks/ode_replay_receipt.json",
                replay_receipt_hash=_sha(replay_receipt_path),
                random_seed=104729,
                tolerance_policy=(
                    "Frozen ODE V5.2 thresholds and deterministic replay hashes"
                ),
                fermi_estimate_ref="results/fermi_estimate.json",
                fermi_estimate_hash=_sha(fermi_path),
                toy_oracle_refs=["checks/ode_toy_oracle.json"],
                toy_oracle_hashes={
                    "checks/ode_toy_oracle.json": _sha(toy_path)
                },
            ).model_dump(mode="json"),
        )
        selected = next(
            item
            for item in bundle.candidates
            if item.candidate_id == bundle.selected_candidate_id
        )
        l4 = next(item for item in bundle.levels if item.level == "L4")
        low = l4.metrics.get("forecast_interval_low")
        high = l4.metrics.get("forecast_interval_high")
        if not isinstance(low, (int, float)) or not isinstance(
            high, (int, float)
        ):
            candidate_forecasts = [
                item.forecast_value for item in bundle.candidates
            ]
            low = min(candidate_forecasts)
            high = max(candidate_forecasts)
        point_path = root / "results" / "artifacts" / "forecast.json"
        interval_path = (
            root / "results" / "artifacts" / "forecast_interval.json"
        )
        _write_json_new(
            point_path,
            {
                "schema_version": "5.9-result",
                "result_id": "forecast",
                "value": selected.forecast_value,
                "interval_low": None,
                "interval_high": None,
                "units": snapshot.state_unit,
            },
        )
        _write_json_new(
            interval_path,
            {
                "schema_version": "5.9-result",
                "result_id": "forecast_interval",
                "value": None,
                "interval_low": float(low),
                "interval_high": float(high),
                "units": snapshot.state_unit,
            },
        )
        _write_json_new(
            root / "results" / "index.json",
            ResultIndexV50(
                records=[
                    ResultRecordV50(
                        result_id="forecast",
                        relative_path=(
                            "results/artifacts/forecast.json"
                        ),
                        artifact_hash=_sha(point_path),
                        value=selected.forecast_value,
                        units=snapshot.state_unit,
                    ),
                    ResultRecordV50(
                        result_id="forecast_interval",
                        relative_path=(
                            "results/artifacts/forecast_interval.json"
                        ),
                        artifact_hash=_sha(interval_path),
                        interval_low=float(low),
                        interval_high=float(high),
                        units=snapshot.state_unit,
                    ),
                ]
            ).model_dump(mode="json"),
        )
        plan = ValidationPlanV50.model_validate_json(
            (root / "docs" / "validation_plan.json").read_text(
                encoding="utf-8"
            )
        )
        return bundle, plan

    def _materialize_executable_candidate_receipt(
        self,
        bundle: ODEScientificBundleV52 | AdaptivePositiveSeriesBundleV57,
    ) -> ExecutableCandidateReceiptV62:
        root = self.workspace.root
        resolution = ExecutableCandidateResolutionV62.model_validate_json(
            (root / EXECUTABLE_CANDIDATE_RESOLUTION_PATH).read_text(
                encoding="utf-8"
            )
        )
        receipt = build_executable_candidate_receipt_v62(
            workspace=self.workspace,
            resolution=resolution,
            bundle=bundle,
        )
        if not verify_executable_candidate_receipt_v62(
            workspace=self.workspace,
            resolution=resolution,
            bundle=bundle,
            receipt=receipt,
        ):
            raise BackhalfRuntimeError(
                "S3 executable candidate receipt failed deterministic replay"
            )
        _write_json_new(
            root / EXECUTABLE_CANDIDATE_RECEIPT_PATH,
            receipt.model_dump(mode="json"),
        )
        return receipt

    def run_s3(self) -> str:
        if self.workspace.current_gate("S3"):
            return "OPEN"
        if self.workspace.current_gate("S2") is None:
            raise BackhalfRuntimeError("S3 requires an open S2 gate")
        if self._current_adapter_id() == ADAPTIVE_ADAPTER_ID:
            adaptive_bundle, plan = self._materialize_adaptive_s3()
            execution_receipt = (
                self._materialize_executable_candidate_receipt(
                    adaptive_bundle
                )
            )
            obligations = [
                item for item in plan.obligations if item.stage == "S3"
            ]
            self._event(
                "s3_computation_completed",
                "succeeded",
                "Adaptive candidate graph was fitted and independently replayed",
                {
                    "selected_branch": adaptive_bundle.graph.selected_branch,
                    "selected_family": adaptive_bundle.graph.selected_model_id,
                    "recovery_triggered": (
                        adaptive_bundle.graph.recovery_triggered
                    ),
                    "bundle_hash": adaptive_bundle.bundle_hash,
                    "level_statuses": {
                        item.level: item.status
                        for item in adaptive_bundle.levels
                    },
                    "scientific_acceptance": (
                        adaptive_bundle.scientific_acceptance
                    ),
                    "fixture_only": adaptive_bundle.fixture_only,
                    "executable_candidate_receipt_hash": (
                        execution_receipt.receipt_hash
                    ),
                },
            )
            return self._evaluate(
                stage="S3",
                producer_run_id="s3-harness-adaptive-executor",
                producer_context_id=f"s3-harness-{uuid4().hex[:16]}",
                summary={
                    "adapter_id": ADAPTIVE_ADAPTER_ID,
                    "bundle_hash": adaptive_bundle.bundle_hash,
                    "selected_branch": (
                        adaptive_bundle.graph.selected_branch
                    ),
                    "selected_model_id": (
                        adaptive_bundle.graph.selected_model_id
                    ),
                    "levels": {
                        item.level: item.status
                        for item in adaptive_bundle.levels
                    },
                },
                extra_paths=[
                    ADAPTIVE_BUNDLE_PATH,
                    ADAPTER_BINDING_PATH,
                    ADAPTIVE_REPLAY_RECEIPTS_PATH,
                    ADAPTIVE_REPLAY_SUMMARY_PATH,
                    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
                ],
                scientific_obligations=obligations,
            )
        bundle, plan = self._materialize_s3()
        execution_receipt = self._materialize_executable_candidate_receipt(
            bundle
        )
        obligations = [
            item for item in plan.obligations if item.stage == "S3"
        ]
        self._event(
            "s3_computation_completed",
            "succeeded",
            "Registered scalar ODE candidates were fitted and replayed",
            {
                "selected_family": bundle.selected_candidate_id,
                "bundle_hash": bundle.bundle_hash,
                "level_statuses": {
                    item.level: item.status for item in bundle.levels
                },
                "scientific_acceptance": bundle.scientific_acceptance,
                "fixture_only": bundle.fixture_only,
                "executable_candidate_receipt_hash": (
                    execution_receipt.receipt_hash
                ),
            },
        )
        return self._evaluate(
            stage="S3",
            producer_run_id="s3-harness-ode-executor",
            producer_context_id=f"s3-harness-{uuid4().hex[:16]}",
            summary={
                "adapter_id": ODE_ADAPTER_ID,
                "bundle_hash": bundle.bundle_hash,
                "levels": {
                    item.level: item.status for item in bundle.levels
                },
            },
            extra_paths=[
                BUNDLE_PATH,
                ADAPTER_BINDING_PATH,
                EXECUTABLE_CANDIDATE_RECEIPT_PATH,
            ],
            scientific_obligations=obligations,
        )

    def run_s4(self) -> str:
        if self.workspace.current_gate("S4"):
            return "OPEN"
        if self.workspace.current_gate("S3") is None:
            raise BackhalfRuntimeError("S4 requires an open S3 gate")
        root = self.workspace.root
        for relative in (
            "results/verification_summary.json",
            "results/uq_summary.json",
            ROLLING_CONFIRMATION_PATH,
        ):
            if (root / relative).exists():
                raise BackhalfRuntimeError(
                    "S4 contains partial artifacts; automatic rerun is blocked"
                )
        adapter_id = self._current_adapter_id()
        if adapter_id == ADAPTIVE_ADAPTER_ID:
            bundle = AdaptivePositiveSeriesBundleV57.model_validate_json(
                (root / ADAPTIVE_BUNDLE_PATH).read_text(encoding="utf-8")
            )
            bundle_path = ADAPTIVE_BUNDLE_PATH
            selected_summary = {
                "selected_branch": bundle.graph.selected_branch,
                "selected_model_id": bundle.graph.selected_model_id,
                "recovery_triggered": bundle.graph.recovery_triggered,
            }
        else:
            bundle = ODEScientificBundleV52.model_validate_json(
                (root / BUNDLE_PATH).read_text(encoding="utf-8")
            )
            bundle_path = BUNDLE_PATH
            selected_summary = {
                "selected_model_id": bundle.selected_candidate_id,
                "recovery_triggered": False,
            }
        plan = ValidationPlanV50.model_validate_json(
            (root / "docs" / "validation_plan.json").read_text(
                encoding="utf-8"
            )
        )
        obligations = [
            item for item in plan.obligations if item.stage == "S4"
        ]
        _write_json_new(
            root / "results" / "verification_summary.json",
            {
                "schema_version": "5.9",
                "validation_plan_hash": plan.plan_hash,
                "check_ids": [item.check_id for item in obligations],
                "adapter_id": adapter_id,
                "bundle_hash": bundle.bundle_hash,
                "level_statuses": {
                    item.level: item.status for item in bundle.levels
                },
                "scientific_acceptance": bundle.scientific_acceptance,
                "scientific_qualification_granted": False,
            },
        )
        l4 = next(item for item in bundle.levels if item.level == "L4")
        disagreement = l4.metrics.get(
            "ensemble_forecast_coefficient_of_variation",
            l4.metrics.get("forecast_interval_relative_width"),
        )
        _write_json_new(
            root / "results" / "uq_summary.json",
            UQSummaryV50(
                claims=[
                    UQClaimV50(
                        claim_id="forecast_claim",
                        result_id="forecast",
                        interval_result_id="forecast_interval",
                        support_status=(
                            "in_support"
                            if l4.status == "PASS"
                            else "unknown"
                        ),
                        ensemble_disagreement=(
                            float(disagreement)
                            if isinstance(disagreement, (int, float))
                            and math.isfinite(float(disagreement))
                            else 1.0
                        ),
                    )
                ]
            ).model_dump(mode="json"),
        )
        success_contract = ScientificSuccessContractV61.model_validate_json(
            (root / SUCCESS_CONTRACT_PATH).read_text(encoding="utf-8")
        )
        snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
            (root / PROCESSED_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        rolling_confirmation = evaluate_rolling_confirmation_v61(
            snapshot=snapshot,
            contract=success_contract,
        )
        _write_json_new(
            root / ROLLING_CONFIRMATION_PATH,
            rolling_confirmation.model_dump(mode="json"),
        )
        self._event(
            "s4_verification_materialized",
            "succeeded",
            "Holdout and uncertainty evidence were projected from the capability bundle",
            {
                "l3_status": next(
                    item.status for item in bundle.levels if item.level == "L3"
                ),
                "l4_status": l4.status,
                "scientific_acceptance": bundle.scientific_acceptance,
                "rolling_confirmation_status": rolling_confirmation.status,
                "rolling_confirmation_hash": (
                    rolling_confirmation.evidence_hash
                ),
                **selected_summary,
            },
        )
        return self._evaluate(
            stage="S4",
            producer_run_id=f"s4-harness-{adapter_id}-verifier",
            producer_context_id=f"s4-harness-{uuid4().hex[:16]}",
            summary={
                "adapter_id": adapter_id,
                "bundle_hash": bundle.bundle_hash,
                "scientific_acceptance": bundle.scientific_acceptance,
                "fixture_only": bundle.fixture_only,
                "rolling_confirmation_status": rolling_confirmation.status,
                **selected_summary,
            },
            extra_paths=[
                bundle_path,
                ADAPTER_BINDING_PATH,
                ROLLING_CONFIRMATION_PATH,
            ],
            scientific_obligations=obligations,
        )

    def run_s5(self) -> str:
        if self.workspace.current_gate("S5"):
            return "OPEN"
        if self.workspace.current_gate("S4") is None:
            raise BackhalfRuntimeError("S5 requires an open S4 gate")
        root = self.workspace.root
        dossier_path = root / "results" / "decision_dossier.json"
        decision_evidence_path = root / DECISION_EVIDENCE_PATH
        if dossier_path.exists() or decision_evidence_path.exists():
            raise BackhalfRuntimeError(
                "S5 contains partial artifacts; automatic rerun is blocked"
            )
        results = ResultIndexV50.model_validate_json(
            (root / "results" / "index.json").read_text(encoding="utf-8")
        )
        uq = UQSummaryV50.model_validate_json(
            (root / "results" / "uq_summary.json").read_text(
                encoding="utf-8"
            )
        )
        plan = ValidationPlanV50.model_validate_json(
            (root / "docs" / "validation_plan.json").read_text(
                encoding="utf-8"
            )
        )
        model = ModelSpecV50.model_validate_json(
            (root / "docs" / "model_spec.json").read_text(
                encoding="utf-8"
            )
        )
        decision_evidence = None
        decision_contract_path = root / DECISION_CONTRACT_PATH
        if decision_contract_path.is_file():
            decision_contract = DecisionValueContractV62.model_validate_json(
                decision_contract_path.read_text(encoding="utf-8")
            )
            success_contract = (
                ScientificSuccessContractV61.model_validate_json(
                    (root / SUCCESS_CONTRACT_PATH).read_text(encoding="utf-8")
                )
            )
            snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
                (root / PROCESSED_RELATIVE_PATH).read_text(encoding="utf-8")
            )
            decision_evidence = evaluate_decision_value_v62(
                snapshot=snapshot,
                success_contract=success_contract,
                decision_contract=decision_contract,
            )
            _write_json_new(
                decision_evidence_path,
                decision_evidence.model_dump(mode="json"),
            )
        producer = self._run_role(
            stage="S5",
            role_name="s5_decision_writer",
            role_kind="generator",
            subject_id=model.selected_candidate_id,
            objective=(
                "Draft one bounded report-only interpretation of the frozen "
                "result and uncertainty evidence."
            ),
            public_inputs={
                "objective": self.workspace.spec.objective,
                "selected_candidate_id": model.selected_candidate_id,
                "results": results.model_dump(mode="json"),
                "uq": uq.model_dump(mode="json"),
                "decision_value_evidence": (
                    decision_evidence.model_dump(mode="json")
                    if decision_evidence is not None
                    else {
                        "status": "NOT_RUN",
                        "reason_codes": [
                            "decision_value_contract_absent"
                        ],
                    }
                ),
                "authority_rule": (
                    "Narrative only. The harness owns bindings, next_action, "
                    "prediction registration, and every external action."
                ),
                "required_artifacts": {
                    "decision_narrative": (
                        DecisionNarrativeDraftV59.model_json_schema()
                    )
                },
            },
            allowed_candidate_ids=[model.selected_candidate_id],
        )
        if producer.draft.authority_claimed:
            raise BackhalfRuntimeError("S5 writer claimed authority")
        narrative = DecisionNarrativeDraftV59.model_validate(
            _artifact_json(producer, "decision_narrative")
        )
        commit_generator_outcome_v51(
            self.workspace,
            producer,
            execution_role="writer",
            input_authority_hash=str(self.workspace.current_gate("S4")),
        )
        high_disagreement = any(
            item.ensemble_disagreement
            >= plan.ensemble_disagreement_threshold
            for item in uq.claims
        )
        unsupported = any(
            item.support_status != "in_support" for item in uq.claims
        )
        next_action = (
            "return_to_data_acquisition"
            if high_disagreement or unsupported
            else "request_human_decision"
            if decision_evidence is not None
            else "draft_report_only"
        )
        _write_json_new(
            dossier_path,
            DecisionDossierV50(
                assertions=[
                    DecisionAssertionV50(
                        assertion_id="forecast_interpretation",
                        statement=narrative.statement,
                        result_ids=["forecast"],
                        uq_claim_ids=["forecast_claim"],
                    )
                ],
                high_disagreement_detected=high_disagreement,
                next_action=next_action,
                real_world_action_authorized=False,
            ).model_dump(mode="json"),
        )
        self._event(
            "s5_decision_dossier_materialized",
            "succeeded",
            "A bounded decision dossier was bound to results and UQ",
            {
                "next_action": next_action,
                "high_disagreement_detected": high_disagreement,
                "limitations": narrative.limitations,
                "decision_value_status": (
                    decision_evidence.status
                    if decision_evidence is not None
                    else "NOT_RUN"
                ),
                "scientific_decision_status": (
                    decision_evidence.scientific_decision_status
                    if decision_evidence is not None
                    else "NOT_RUN"
                ),
                "real_world_action_authorized": False,
            },
        )
        s5_extra_paths = (
            [DECISION_EVIDENCE_PATH]
            if decision_evidence is not None
            else []
        )
        return self._evaluate(
            stage="S5",
            producer_run_id=producer.request.run_id,
            producer_context_id=producer.request.context_id,
            summary={
                "next_action": next_action,
                "high_disagreement_detected": high_disagreement,
                "decision_value_status": (
                    decision_evidence.status
                    if decision_evidence is not None
                    else "NOT_RUN"
                ),
                "scientific_decision_status": (
                    decision_evidence.scientific_decision_status
                    if decision_evidence is not None
                    else "NOT_RUN"
                ),
                "real_world_action_authorized": False,
            },
            extra_paths=s5_extra_paths,
        )

    def run_s6(self) -> str:
        if self.workspace.current_gate("S6"):
            return "OPEN"
        if self.workspace.current_gate("S5") is None:
            raise BackhalfRuntimeError("S6 requires an open S5 gate")
        root = self.workspace.root
        required = (
            "results/values.json",
            "paper/main.template.tex",
            "paper/build/main.tex",
            "paper/build/main.pdf",
            "paper/build/build_receipt.json",
        )
        if any((root / relative).exists() for relative in required):
            raise BackhalfRuntimeError(
                "S6 contains partial artifacts; automatic rerun is blocked"
            )
        results = ResultIndexV50.model_validate_json(
            (root / "results" / "index.json").read_text(encoding="utf-8")
        )
        values: dict[str, float] = {}
        for record in results.records:
            if record.value is not None:
                values[record.result_id] = record.value
            if (
                record.interval_low is not None
                and record.interval_high is not None
            ):
                values[f"{record.result_id}_low"] = record.interval_low
                values[f"{record.result_id}_high"] = record.interval_high
        _write_json_new(root / "results" / "values.json", values)
        template_path = root / "paper" / "main.template.tex"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\section*{FMA Modeling Report}\n"
            "Registered scalar model forecast: {{result.forecast}}.\\\\\n"
            "Frozen uncertainty interval: "
            "{{result.forecast_interval_low}} to "
            "{{result.forecast_interval_high}}.\\\\\n"
            "This report grants neither scientific qualification nor "
            "real-world action authority.\n"
            "\\end{document}\n",
            encoding="utf-8",
            newline="\n",
        )
        receipt = build_paper(root)
        self._event(
            "s6_paper_built",
            "succeeded",
            "The report PDF was built from machine-readable result values",
            {
                "receipt_hash": receipt.receipt_hash,
                "scientific_correctness_established": False,
                "real_world_action_authorized": False,
            },
        )
        return self._evaluate(
            stage="S6",
            producer_run_id="s6-harness-paper-builder",
            producer_context_id=f"s6-harness-{uuid4().hex[:16]}",
            summary={
                "paper_build_receipt_hash": receipt.receipt_hash,
                "scientific_correctness_established": False,
                "real_world_action_authorized": False,
            },
        )

    def run(self) -> dict[str, str]:
        decisions: dict[str, str] = {}
        for stage, runner in (
            ("S2", self.run_s2),
            ("S3", self.run_s3),
            ("S4", self.run_s4),
            ("S5", self.run_s5),
            ("S6", self.run_s6),
        ):
            decision = runner()
            decisions[stage] = decision
            if decision != "OPEN":
                break
        return decisions


def backhalf_summary_v59(workspace: StageWorkspaceV50) -> dict[str, Any]:
    root = workspace.root
    binding_path = root / ADAPTER_BINDING_PATH
    binding = (
        json.loads(binding_path.read_text(encoding="utf-8"))
        if binding_path.is_file()
        else {}
    )
    adapter_id = str(binding.get("adapter_id", ODE_ADAPTER_ID))
    bundle_path = root / BUNDLE_PATH
    ode_bundle = (
        ODEScientificBundleV52.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )
        if bundle_path.is_file()
        else None
    )
    adaptive_path = root / ADAPTIVE_BUNDLE_PATH
    adaptive_bundle = (
        AdaptivePositiveSeriesBundleV57.model_validate_json(
            adaptive_path.read_text(encoding="utf-8")
        )
        if adaptive_path.is_file()
        else None
    )
    bundle = adaptive_bundle or ode_bundle
    provenance_path = root / PROVENANCE_BINDING_PATH
    provenance_admitted = _current_stage_file_admitted(
        workspace,
        stage="S2",
        relative_path=PROVENANCE_BINDING_PATH,
    )
    try:
        provenance = (
            DataProvenanceBindingV62.model_validate_json(
                provenance_path.read_text(encoding="utf-8")
            )
            if provenance_admitted
            else None
        )
    except (OSError, UnicodeDecodeError, ValueError):
        provenance = None
        provenance_admitted = False
    rolling_admitted = _current_stage_file_admitted(
        workspace,
        stage="S4",
        relative_path=ROLLING_CONFIRMATION_PATH,
    )
    decision_admitted = _current_stage_file_admitted(
        workspace,
        stage="S5",
        relative_path=DECISION_EVIDENCE_PATH,
    )
    execution_resolution_admitted = _current_stage_file_admitted(
        workspace,
        stage="S2",
        relative_path=EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
    )
    execution_receipt_admitted = _current_stage_file_admitted(
        workspace,
        stage="S3",
        relative_path=EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    )
    try:
        rolling = (
            RollingConfirmationV61.model_validate_json(
                (root / ROLLING_CONFIRMATION_PATH).read_text(
                    encoding="utf-8"
                )
            )
            if rolling_admitted
            else None
        )
    except (OSError, UnicodeDecodeError, ValueError):
        rolling = None
        rolling_admitted = False
    try:
        decision_evidence = (
            DecisionValueEvidenceV62.model_validate_json(
                (root / DECISION_EVIDENCE_PATH).read_text(encoding="utf-8")
            )
            if decision_admitted
            else None
        )
    except (OSError, UnicodeDecodeError, ValueError):
        decision_evidence = None
        decision_admitted = False
    execution_receipt: ExecutableCandidateReceiptV62 | None = None
    execution_replayed = False
    if execution_resolution_admitted and execution_receipt_admitted:
        try:
            execution_resolution = (
                ExecutableCandidateResolutionV62.model_validate_json(
                    (
                        root / EXECUTABLE_CANDIDATE_RESOLUTION_PATH
                    ).read_text(encoding="utf-8")
                )
            )
            execution_receipt = (
                ExecutableCandidateReceiptV62.model_validate_json(
                    (root / EXECUTABLE_CANDIDATE_RECEIPT_PATH).read_text(
                        encoding="utf-8"
                    )
                )
            )
            execution_bundle = (
                AdaptivePositiveSeriesBundleV57.model_validate_json(
                    (root / ADAPTIVE_BUNDLE_PATH).read_text(encoding="utf-8")
                )
                if execution_resolution.adapter_id == ADAPTIVE_ADAPTER_ID
                else ODEScientificBundleV52.model_validate_json(
                    (root / BUNDLE_PATH).read_text(encoding="utf-8")
                )
            )
            execution_replayed = verify_executable_candidate_receipt_v62(
                workspace=workspace,
                resolution=execution_resolution,
                bundle=execution_bundle,
                receipt=execution_receipt,
            )
        except (OSError, UnicodeDecodeError, ValueError):
            execution_receipt = None
            execution_replayed = False
    return {
        "schema_version": "6.0",
        "adapter_id": adapter_id,
        "data_received": (root / RAW_RELATIVE_PATH).is_file(),
        "workflow_complete": workspace.current_gate("S6") is not None,
        "selected_scientific_family": (
            (
                adaptive_bundle.graph.selected_model_id
                if adaptive_bundle is not None
                else ode_bundle.selected_candidate_id
                if ode_bundle is not None
                else None
            )
        ),
        "selected_branch": (
            adaptive_bundle.graph.selected_branch
            if adaptive_bundle is not None
            else "autonomous_ode"
            if ode_bundle is not None
            else None
        ),
        "recovery_triggered": (
            adaptive_bundle.graph.recovery_triggered
            if adaptive_bundle is not None
            else False
        ),
        "level_statuses": (
            {item.level: item.status for item in bundle.levels}
            if bundle is not None
            else {}
        ),
        "scientific_acceptance": (
            bundle.scientific_acceptance if bundle is not None else False
        ),
        "fixture_only": bundle.fixture_only if bundle is not None else None,
        "source_integrity_status": (
            provenance.status if provenance is not None else "NOT_RUN"
        ),
        "scientific_provenance_status": (
            provenance.scientific_provenance_status
            if provenance is not None
            else "NOT_RUN"
        ),
        "source_stage_admission_status": (
            "PASS" if provenance_admitted and provenance is not None else "NOT_RUN"
        ),
        "rolling_confirmation_admission_status": (
            "PASS" if rolling_admitted and rolling is not None else "NOT_RUN"
        ),
        "rolling_confirmation_status": (
            rolling.status if rolling is not None else "NOT_RUN"
        ),
        "decision_evidence_admission_status": (
            "PASS"
            if decision_admitted and decision_evidence is not None
            else "NOT_RUN"
        ),
        "decision_evidence_status": (
            decision_evidence.status
            if decision_evidence is not None
            else "NOT_RUN"
        ),
        "scientific_decision_status": (
            decision_evidence.scientific_decision_status
            if decision_evidence is not None
            else "NOT_RUN"
        ),
        "executable_candidate_admission_status": (
            "PASS"
            if (
                execution_resolution_admitted
                and execution_receipt_admitted
                and execution_replayed
            )
            else "FAIL"
            if execution_resolution_admitted or execution_receipt_admitted
            else "NOT_RUN"
        ),
        "executable_candidate_status": (
            execution_receipt.local_execution_status
            if execution_receipt is not None and execution_replayed
            else "NOT_RUN"
        ),
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


__all__ = [
    "BackhalfRuntimeError",
    "ADAPTIVE_ADAPTER_ID",
    "DataMappingDraftV59",
    "DecisionNarrativeDraftV59",
    "ODE_ADAPTER_ID",
    "StudioBackhalfOrchestratorV59",
    "StudioODEDataRequestV59",
    "backhalf_summary_v59",
    "ingest_ode_data_v59",
]
