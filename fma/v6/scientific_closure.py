"""Additive V6.2 scientific-closure composition over a V6.1 report.

V6.1 remains the authority for its local workflow and rolling-confirmation
semantics.  This module only composes that sealed result with a code-owned
source-to-S2 provenance binding, optional retrospective decision evidence, and an
optional *local-only* external-artifact binding.

The local external binding deliberately cannot represent a verified
independent signature.  Consequently this module can show that local
prerequisites are ready for external evaluation, but it can never grant
scientific qualification or authorize a real-world action.  A later additive
version must accept a cryptographically verified certificate produced by an
independent authority before either claim may change.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import StageId
from fma.v5_2.ode_system import ODEScientificBundleV52
from fma.v5_7.adaptive_positive_series import (
    AdaptivePositiveSeriesBundleV57,
)

from .decision_value import DECISION_EVIDENCE_PATH, DecisionValueEvidenceV62
from .executable_candidate import (
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    EXECUTABLE_CANDIDATE_INTENT_PATH,
    EXECUTABLE_CANDIDATE_IR_PATH,
    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
    SCALAR_ODE_ADAPTER_ID,
    ExecutableCandidateReceiptV62,
    ExecutableCandidateResolutionV62,
    RegisteredFamilySearchIRV62,
    RegisteredFamilySearchIntentV62,
    verify_executable_candidate_receipt_v62,
)
from .provenance import (
    PROVENANCE_BINDING_PATH,
    DataProvenanceBindingV62,
)
from .scientific_success import (
    ClaimCeilingV61,
    ClaimKindV61,
    DimensionStatusV61,
    SUCCESS_PROJECTION_PATH,
    ScientificSuccessDimensionV61,
    ScientificSuccessReportV61,
)


_STAGES: tuple[StageId, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
ROLLING_CONFIRMATION_ADMISSION_PATH = (
    "results/rolling_confirmation_v61.json"
)
DECISION_EVIDENCE_ADMISSION_PATH = "results/decision_value_evidence_v62.json"
ODE_SCIENTIFIC_BUNDLE_ADMISSION_PATH = "results/ode_scientific_bundle.json"
ADAPTIVE_SCIENTIFIC_BUNDLE_ADMISSION_PATH = (
    "results/adaptive_positive_series_bundle.json"
)
SCIENTIFIC_CLOSURE_ROOT = ".fma/scientific_closure_v62"
_DIMENSION_IDS: tuple[str, ...] = (
    "workflow_integrity",
    "data_provenance",
    "local_adapter_checks",
    "leakage_safe_confirmation",
    "decision_value",
    "mechanism_identification",
    "external_generalization",
    "scientific_qualification",
)

_LOCAL_REQUIREMENTS: dict[ClaimKindV61, frozenset[str]] = {
    "descriptive": frozenset(
        {
            "workflow_integrity",
            "data_provenance",
            "local_adapter_checks",
        }
    ),
    "predictive": frozenset(
        {
            "workflow_integrity",
            "data_provenance",
            "local_adapter_checks",
            "leakage_safe_confirmation",
        }
    ),
    "mechanistic": frozenset(
        {
            "workflow_integrity",
            "data_provenance",
            "local_adapter_checks",
            "leakage_safe_confirmation",
            "mechanism_identification",
        }
    ),
    "prescriptive": frozenset(
        {
            "workflow_integrity",
            "data_provenance",
            "local_adapter_checks",
            "leakage_safe_confirmation",
            "decision_value",
        }
    ),
    "generalization": frozenset(
        {
            "workflow_integrity",
            "data_provenance",
            "local_adapter_checks",
            "leakage_safe_confirmation",
        }
    ),
}


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_evidence_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in relative_path
    ):
        raise ValueError("stage-admission evidence path is unsafe")
    candidate = (root / Path(*pure.parts)).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("stage-admission evidence path escapes workspace")
    return candidate


def _current_stage_file_admitted(
    workspace: StageWorkspaceV50,
    *,
    stage: StageId,
    relative_path: str,
) -> bool:
    try:
        path = _safe_evidence_path(workspace.root.resolve(), relative_path)
        certificate = workspace._certificate_for_current_node(stage)
        if (
            certificate is None
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
            and path.is_file()
            and binding.sha256 == _file_hash(path)
            and binding.size_bytes == path.stat().st_size
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _local_required_ids(claim_kind: ClaimKindV61) -> list[str]:
    return sorted(_LOCAL_REQUIREMENTS[claim_kind])


def _closure_required_ids(claim_kind: ClaimKindV61) -> list[str]:
    required = set(_LOCAL_REQUIREMENTS[claim_kind])
    # Descriptive evidence need not claim transport to a new population.
    # Every empirical predictive, mechanistic, prescriptive, or explicit
    # generalization claim does.
    if claim_kind != "descriptive":
        required.add("external_generalization")
    # Qualification is an independent authority decision, not a model output.
    required.add("scientific_qualification")
    return sorted(required)


class StageEvidenceAdmissionV62(StrictModel):
    """Code-derived proof that component evidence preceded current gates."""

    schema_version: Literal["6.2-stage-evidence-admission"] = (
        "6.2-stage-evidence-admission"
    )
    workspace_spec_hash: Sha256
    v61_report_hash: Sha256
    current_gate_hashes: dict[StageId, Sha256]
    s2_attempt: Annotated[int, Field(ge=1)]
    stage_manifest_hashes: dict[StageId, Sha256]
    evidence_relative_paths: dict[Identifier, str]
    evidence_file_hashes: dict[Identifier, Sha256]
    executable_candidate_intent_hash: Sha256
    executable_candidate_ir_hash: Sha256
    executable_candidate_resolution_hash: Sha256
    executable_candidate_receipt_hash: Sha256
    scientific_bundle_hash: Sha256
    provenance_binding_hash: Sha256
    rolling_confirmation_hash: Sha256
    decision_evidence_hash: Sha256 | None = None
    status: Literal["PASS", "FAIL"]
    checks: dict[Identifier, bool]
    reason_codes: list[Identifier]
    admission_mode: Literal["current_authenticated_stage_manifests"] = (
        "current_authenticated_stage_manifests"
    )
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    admission_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_admission(self) -> "StageEvidenceAdmissionV62":
        expected_keys = {
            "execution_intent",
            "execution_ir",
            "execution_resolution",
            "execution_receipt",
            "scientific_bundle",
            "provenance",
            "rolling_confirmation",
        }
        expected_stages: set[StageId] = {"S1", "S2", "S3", "S4"}
        if self.decision_evidence_hash is not None:
            expected_keys.add("decision")
            expected_stages.add("S5")
        if set(self.evidence_relative_paths) != expected_keys:
            raise ValueError("stage-admission evidence path keys differ")
        if set(self.evidence_file_hashes) != expected_keys:
            raise ValueError("stage-admission evidence hash keys differ")
        if not set(self.stage_manifest_hashes).issubset(expected_stages):
            raise ValueError("stage-admission contains an unrelated manifest")
        for relative_path in self.evidence_relative_paths.values():
            pure = PurePosixPath(relative_path)
            if (
                pure.is_absolute()
                or not pure.parts
                or any(part in {"", ".", ".."} for part in pure.parts)
                or "\\" in relative_path
            ):
                raise ValueError("stage-admission contains an unsafe path")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("stage-admission reasons must be sorted and unique")
        expected_pass = bool(self.checks) and all(self.checks.values())
        if (self.status == "PASS") != expected_pass:
            raise ValueError("stage-admission status differs from checks")
        if self.status == "PASS" and (
            set(self.stage_manifest_hashes) != expected_stages
        ):
            raise ValueError("passing admission lacks a required stage manifest")
        if self.admission_hash and self.admission_hash != self.content_hash():
            raise ValueError("stage-admission hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "admission_hash")

    def assert_sealed(self) -> None:
        if (
            not self.admission_hash
            or self.admission_hash != self.content_hash()
        ):
            raise ValueError("stage-evidence admission is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "StageEvidenceAdmissionV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"admission_hash"})
        payload["admission_hash"] = draft.content_hash()
        return cls(**payload)


def build_stage_evidence_admission_v62(
    *,
    workspace: StageWorkspaceV50,
    v61_report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    decision_evidence: DecisionValueEvidenceV62 | None = None,
    execution_intent_relative_path: str = EXECUTABLE_CANDIDATE_INTENT_PATH,
    execution_ir_relative_path: str = EXECUTABLE_CANDIDATE_IR_PATH,
    execution_resolution_relative_path: str = (
        EXECUTABLE_CANDIDATE_RESOLUTION_PATH
    ),
    execution_receipt_relative_path: str = EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    scientific_bundle_relative_path: str | None = None,
    provenance_relative_path: str = PROVENANCE_BINDING_PATH,
    rolling_confirmation_relative_path: str = (
        ROLLING_CONFIRMATION_ADMISSION_PATH
    ),
    decision_relative_path: str = DECISION_EVIDENCE_ADMISSION_PATH,
) -> StageEvidenceAdmissionV62:
    """Inspect authenticated current manifests and exact workspace files.

    The aggregate V6.1 report is intentionally post-S6.  Its executable
    semantics must have been frozen in S1, resolved with provenance in S2,
    replayed in S3, and joined to rolling-confirmation and optional decision
    evidence by S4 and S5 respectively.
    """

    root = workspace.root.resolve()
    current_gate_hashes = {
        stage: str(gate)
        for stage in _STAGES
        if (gate := workspace.current_gate(stage)) is not None
    }
    if scientific_bundle_relative_path is None:
        scientific_bundle_relative_path = (
            ADAPTIVE_SCIENTIFIC_BUNDLE_ADMISSION_PATH
            if v61_report.adapter_id
            == ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID
            else ODE_SCIENTIFIC_BUNDLE_ADMISSION_PATH
        )
    evidence_paths = {
        "execution_intent": execution_intent_relative_path,
        "execution_ir": execution_ir_relative_path,
        "execution_resolution": execution_resolution_relative_path,
        "execution_receipt": execution_receipt_relative_path,
        "scientific_bundle": scientific_bundle_relative_path,
        "provenance": provenance_relative_path,
        "rolling_confirmation": rolling_confirmation_relative_path,
    }
    expected_objects: dict[str, StrictModel] = {
        "provenance": provenance,
        "rolling_confirmation": v61_report.rolling_confirmation,
    }
    stage_by_key: dict[str, StageId] = {
        "execution_intent": "S1",
        "execution_ir": "S1",
        "execution_resolution": "S2",
        "execution_receipt": "S3",
        "scientific_bundle": "S3",
        "provenance": "S2",
        "rolling_confirmation": "S4",
    }
    if decision_evidence is not None:
        evidence_paths["decision"] = decision_relative_path
        expected_objects["decision"] = decision_evidence
        stage_by_key["decision"] = "S5"

    required_stages = sorted(set(stage_by_key.values()))
    certificates = {}
    stage_manifest_hashes: dict[StageId, str] = {}
    for stage in required_stages:
        certificate = workspace._certificate_for_current_node(stage)
        if (
            certificate is not None
            and workspace.current_gate(stage) == certificate.certificate_hash
            and workspace.verify_certificate(certificate)
            and certificate.manifest.manifest_hash
        ):
            certificates[stage] = certificate
            stage_manifest_hashes[stage] = certificate.manifest.manifest_hash

    evidence_types: dict[str, type[StrictModel]] = {
        "execution_intent": RegisteredFamilySearchIntentV62,
        "execution_ir": RegisteredFamilySearchIRV62,
        "execution_resolution": ExecutableCandidateResolutionV62,
        "execution_receipt": ExecutableCandidateReceiptV62,
        "scientific_bundle": (
            AdaptivePositiveSeriesBundleV57
            if v61_report.adapter_id
            == ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID
            else ODEScientificBundleV52
        ),
        "provenance": DataProvenanceBindingV62,
        "rolling_confirmation": type(v61_report.rolling_confirmation),
    }
    if decision_evidence is not None:
        evidence_types["decision"] = DecisionValueEvidenceV62

    evidence_file_hashes: dict[str, str] = {}
    parsed_objects: dict[str, StrictModel | None] = {}
    checks = {
        "workspace_verified": workspace.verify(),
        "execution_adapter_supported": v61_report.adapter_id
        in {
            SCALAR_ODE_ADAPTER_ID,
            ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
        },
        "v61_report_current": (
            bool(v61_report.report_hash)
            and v61_report.report_hash == v61_report.content_hash()
            and v61_report.workspace_spec_hash == workspace.spec.spec_hash
            and v61_report.current_gate_hashes == current_gate_hashes
        ),
        "required_current_certificates_present": (
            set(certificates) == set(required_stages)
        ),
    }
    for key, relative_path in evidence_paths.items():
        try:
            path = _safe_evidence_path(root, relative_path)
            file_hash = _file_hash(path)
            file_size = path.stat().st_size
            parsed = evidence_types[key].model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            file_hash = "0" * 64
            file_size = -1
            parsed = None
        parsed_objects[key] = parsed
        evidence_file_hashes[key] = file_hash
        stage = stage_by_key[key]
        certificate = certificates.get(stage)
        manifest_binding = (
            next(
                (
                    item
                    for item in certificate.manifest.files
                    if item.relative_path == relative_path
                ),
                None,
            )
            if certificate is not None
            else None
        )
        checks[f"{key}_file_exact"] = (
            parsed is not None
            and (
                key not in expected_objects
                or parsed == expected_objects[key]
            )
            and manifest_binding is not None
            and manifest_binding.sha256 == file_hash
            and manifest_binding.size_bytes == file_size
        )

    execution_intent = parsed_objects.get("execution_intent")
    execution_ir = parsed_objects.get("execution_ir")
    execution_resolution = parsed_objects.get("execution_resolution")
    execution_receipt = parsed_objects.get("execution_receipt")
    scientific_bundle = parsed_objects.get("scientific_bundle")
    intent_hash = (
        execution_intent.content_hash()
        if isinstance(execution_intent, RegisteredFamilySearchIntentV62)
        else "0" * 64
    )
    ir_hash = (
        execution_ir.ir_hash
        if isinstance(execution_ir, RegisteredFamilySearchIRV62)
        and execution_ir.ir_hash
        else "0" * 64
    )
    resolution_hash = (
        execution_resolution.resolution_hash
        if isinstance(
            execution_resolution, ExecutableCandidateResolutionV62
        )
        and execution_resolution.resolution_hash
        else "0" * 64
    )
    receipt_hash = (
        execution_receipt.receipt_hash
        if isinstance(execution_receipt, ExecutableCandidateReceiptV62)
        and execution_receipt.receipt_hash
        else "0" * 64
    )
    bundle_hash = (
        scientific_bundle.bundle_hash
        if isinstance(
            scientific_bundle,
            (ODEScientificBundleV52, AdaptivePositiveSeriesBundleV57),
        )
        and scientific_bundle.bundle_hash
        else "0" * 64
    )
    execution_types_valid = (
        isinstance(
            execution_intent, RegisteredFamilySearchIntentV62
        )
        and isinstance(execution_ir, RegisteredFamilySearchIRV62)
        and isinstance(
            execution_resolution, ExecutableCandidateResolutionV62
        )
        and isinstance(
            execution_receipt, ExecutableCandidateReceiptV62
        )
        and isinstance(
            scientific_bundle,
            (ODEScientificBundleV52, AdaptivePositiveSeriesBundleV57),
        )
    )
    checks["execution_semantic_artifacts_sealed"] = bool(
        execution_types_valid
        and ir_hash != "0" * 64
        and resolution_hash != "0" * 64
        and receipt_hash != "0" * 64
        and bundle_hash != "0" * 64
    )
    checks["execution_intent_ir_bound"] = bool(
        execution_types_valid
        and execution_ir.model_intent_hash == intent_hash
        and execution_ir.candidate_id
        == execution_resolution.selected_candidate_id
        and execution_ir.candidate_structural_hash
        == execution_resolution.selected_candidate_structural_hash
    )
    checks["execution_resolution_receipt_bound"] = bool(
        execution_types_valid
        and execution_resolution.execution_ir_hash == ir_hash
        and execution_receipt.resolution_hash == resolution_hash
        and execution_receipt.selected_candidate_structural_hash
        == execution_resolution.selected_candidate_structural_hash
        and execution_receipt.adapter_id == execution_resolution.adapter_id
    )
    checks["execution_bundle_report_bound"] = bool(
        execution_types_valid
        and execution_resolution.adapter_id == v61_report.adapter_id
        and execution_receipt.bundle_hash == bundle_hash
        and bundle_hash == v61_report.scientific_bundle_hash
        and execution_receipt.fixture_only == v61_report.fixture_only
        and scientific_bundle.snapshot_hash
        == provenance.processed_snapshot_hash
    )
    checks["execution_receipt_replayed"] = bool(
        execution_types_valid
        and verify_executable_candidate_receipt_v62(
            workspace=workspace,
            resolution=execution_resolution,
            bundle=scientific_bundle,
            receipt=execution_receipt,
        )
    )

    s2_certificate = certificates.get("S2")
    current_s2_attempt = (
        s2_certificate.attempt if s2_certificate is not None else 1
    )
    checks["s2_attempt_current"] = (
        s2_certificate is not None
        and provenance.s2_attempt == current_s2_attempt
        and isinstance(
            execution_resolution, ExecutableCandidateResolutionV62
        )
        and execution_resolution.s2_attempt == current_s2_attempt
        and isinstance(
            execution_receipt, ExecutableCandidateReceiptV62
        )
        and execution_receipt.s2_attempt == current_s2_attempt
    )
    reasons = sorted(
        check_id for check_id, passed in checks.items() if not passed
    )
    return StageEvidenceAdmissionV62.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        v61_report_hash=(
            v61_report.report_hash if v61_report.report_hash else "0" * 64
        ),
        current_gate_hashes=current_gate_hashes,
        s2_attempt=current_s2_attempt,
        stage_manifest_hashes=stage_manifest_hashes,
        evidence_relative_paths=evidence_paths,
        evidence_file_hashes=evidence_file_hashes,
        executable_candidate_intent_hash=intent_hash,
        executable_candidate_ir_hash=ir_hash,
        executable_candidate_resolution_hash=resolution_hash,
        executable_candidate_receipt_hash=receipt_hash,
        scientific_bundle_hash=bundle_hash,
        provenance_binding_hash=(
            provenance.binding_hash if provenance.binding_hash else "0" * 64
        ),
        rolling_confirmation_hash=(
            v61_report.rolling_confirmation.evidence_hash
            if v61_report.rolling_confirmation.evidence_hash
            else "0" * 64
        ),
        decision_evidence_hash=(
            decision_evidence.evidence_hash if decision_evidence else None
        ),
        status="PASS" if not reasons else "FAIL",
        checks=checks,
        reason_codes=reasons,
    )


class LocalExternalQualificationBindingV62(StrictModel):
    """Hash binding for untrusted external artifacts, never qualification.

    This type is useful before a real external verifier is integrated: it
    prevents evidence from another attempt from being silently substituted,
    while its literal fields make caller-supplied ``True`` authority
    unrepresentable.
    """

    schema_version: Literal["6.2-local-external-binding"] = (
        "6.2-local-external-binding"
    )
    workspace_spec_hash: Sha256
    v61_report_hash: Sha256
    claim_kind: ClaimKindV61
    current_gate_hashes: dict[StageId, Sha256]
    s2_attempt: Annotated[int, Field(ge=1)]
    source_snapshot_hash: Sha256
    source_verification_hash: Sha256
    provenance_binding_hash: Sha256
    stage_admission_hash: Sha256
    decision_evidence_hash: Sha256 | None = None
    external_artifact_hash: Sha256 | None = None
    evaluation_scope: Literal["local_binding_only"] = "local_binding_only"
    external_generalization_status: Literal["NOT_RUN"] = "NOT_RUN"
    scientific_qualification_status: Literal["NOT_RUN"] = "NOT_RUN"
    independent_signature_verified: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    reason_codes: list[Identifier] = Field(
        default_factory=lambda: ["independent_external_signature_absent"]
    )
    binding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "LocalExternalQualificationBindingV62":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError(
                "local external-binding reasons must be sorted and unique"
            )
        if "independent_external_signature_absent" not in self.reason_codes:
            raise ValueError(
                "local external binding must disclose absent signature"
            )
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("local external binding hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    def assert_sealed(self) -> None:
        if not self.binding_hash or self.binding_hash != self.content_hash():
            raise ValueError("local external binding is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "LocalExternalQualificationBindingV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"binding_hash"})
        payload["binding_hash"] = draft.content_hash()
        return cls(**payload)


class ScientificClosureDimensionV62(StrictModel):
    dimension_id: Identifier
    status: DimensionStatusV61
    required_for_claim: bool
    reason_codes: list[Identifier]
    evidence_refs: list[Sha256]
    metrics: dict[Identifier, float | int | bool | None] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_dimension(self) -> "ScientificClosureDimensionV62":
        if self.dimension_id not in _DIMENSION_IDS:
            raise ValueError("unknown V6.2 scientific-closure dimension")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("closure dimension reasons must be sorted and unique")
        if self.evidence_refs != sorted(set(self.evidence_refs)):
            raise ValueError(
                "closure dimension evidence refs must be sorted and unique"
            )
        return self


class ScientificClosureReportV62(StrictModel):
    """Current, hash-bound projection of local and external readiness."""

    schema_version: Literal["6.2-scientific-closure-report"] = (
        "6.2-scientific-closure-report"
    )
    workspace_spec_hash: Sha256
    v61_contract_hash: Sha256
    v61_report_hash: Sha256
    claim_kind: ClaimKindV61
    current_gate_hashes: dict[StageId, Sha256]
    s2_attempt: Annotated[int, Field(ge=1)]
    source_snapshot_hash: Sha256
    source_verification_hash: Sha256
    provenance_binding_hash: Sha256
    stage_admission_hash: Sha256
    decision_evidence_hash: Sha256 | None = None
    external_binding_hash: Sha256 | None = None
    fixture_only: bool
    local_required_dimension_ids: list[Identifier]
    closure_required_dimension_ids: list[Identifier]
    dimensions: list[ScientificClosureDimensionV62]
    local_evidence_status: DimensionStatusV61
    scientific_closure_status: DimensionStatusV61
    claim_ceiling: ClaimCeilingV61
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ScientificClosureReportV62":
        if [item.dimension_id for item in self.dimensions] != sorted(
            _DIMENSION_IDS
        ):
            raise ValueError(
                "closure report must contain every ordered dimension"
            )
        if self.local_required_dimension_ids != _local_required_ids(
            self.claim_kind
        ):
            raise ValueError("local requirements differ from claim policy")
        if self.closure_required_dimension_ids != _closure_required_ids(
            self.claim_kind
        ):
            raise ValueError("closure requirements differ from claim policy")
        required = set(self.closure_required_dimension_ids)
        if any(
            item.required_for_claim != (item.dimension_id in required)
            for item in self.dimensions
        ):
            raise ValueError("dimension required flags differ from claim policy")
        by_id = {item.dimension_id: item for item in self.dimensions}
        if by_id["scientific_qualification"].status != "NOT_RUN":
            raise ValueError(
                "V6.2 local closure cannot establish qualification"
            )
        if by_id["external_generalization"].status != "NOT_RUN":
            raise ValueError(
                "V6.2 local external binding cannot establish generalization"
            )
        if self.scientific_closure_status == "PASS":
            raise ValueError(
                "V6.2 local closure cannot pass without signed external evidence"
            )
        local_expected = _aggregate(
            self.dimensions, set(self.local_required_dimension_ids)
        )
        if self.local_evidence_status != local_expected:
            raise ValueError("local evidence status differs from dimensions")
        closure_expected = _aggregate(
            self.dimensions, set(self.closure_required_dimension_ids)
        )
        if self.scientific_closure_status != closure_expected:
            raise ValueError("closure status differs from dimensions")
        if self.fixture_only:
            expected_ceiling: ClaimCeilingV61 = "fixture_protocol_only"
        elif local_expected == "PASS" and (
            "leakage_safe_confirmation"
            in self.local_required_dimension_ids
        ):
            expected_ceiling = "local_leakage_safe_predictive_evidence"
        elif (
            by_id["local_adapter_checks"].status == "PASS"
            and by_id["data_provenance"].status == "PASS"
        ):
            expected_ceiling = "local_retrospective_adapter_evidence"
        elif by_id["workflow_integrity"].status == "PASS":
            expected_ceiling = "workflow_integrity_only"
        else:
            expected_ceiling = "no_scientific_claim"
        if self.claim_ceiling != expected_ceiling:
            raise ValueError("claim ceiling differs from dimension evidence")
        if self.fixture_only and self.claim_ceiling != "fixture_protocol_only":
            raise ValueError("fixture closure exceeds fixture claim ceiling")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("V6.2 scientific-closure report hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V6.2 scientific-closure report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ScientificClosureReportV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"report_hash"})
        payload["report_hash"] = draft.content_hash()
        return cls(**payload)


class ScientificClosureVerificationV62(StrictModel):
    """Replay result for both evidence hashes and current graph bindings."""

    schema_version: Literal["6.2-scientific-closure-verification"] = (
        "6.2-scientific-closure-verification"
    )
    report_hash: Sha256 | None
    current_binding_hash: Sha256
    status: Literal["PASS", "FAIL"]
    checks: dict[Identifier, bool]
    reason_codes: list[Identifier]
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    verification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_verification(self) -> "ScientificClosureVerificationV62":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError(
                "closure-verification reasons must be sorted and unique"
            )
        expected = bool(self.checks) and all(self.checks.values())
        if (self.status == "PASS") != expected:
            raise ValueError("closure-verification status differs from checks")
        if self.verification_hash and (
            self.verification_hash != self.content_hash()
        ):
            raise ValueError("closure-verification hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "verification_hash")

    @classmethod
    def seal(cls, **data: object) -> "ScientificClosureVerificationV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"verification_hash"})
        payload["verification_hash"] = draft.content_hash()
        return cls(**payload)


def _dimension(
    *,
    dimension_id: str,
    status: DimensionStatusV61,
    required_ids: set[str],
    reason_codes: list[str],
    evidence_refs: list[str],
    metrics: dict[str, float | int | bool | None] | None = None,
) -> ScientificClosureDimensionV62:
    return ScientificClosureDimensionV62(
        dimension_id=dimension_id,
        status=status,
        required_for_claim=dimension_id in required_ids,
        reason_codes=sorted(set(reason_codes)),
        evidence_refs=sorted(set(evidence_refs)),
        metrics=metrics or {},
    )


def _aggregate(
    dimensions: list[ScientificClosureDimensionV62],
    required_ids: set[str],
) -> DimensionStatusV61:
    statuses = [
        item.status
        for item in dimensions
        if item.dimension_id in required_ids
    ]
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "NOT_RUN" for status in statuses):
        return "NOT_RUN"
    if any(status == "HUMAN" for status in statuses):
        return "HUMAN"
    return "PASS"


def _assert_v61_current(
    *,
    v61_report: ScientificSuccessReportV61,
    current_workspace_spec_hash: str,
    current_gate_hashes: Mapping[StageId, str],
) -> None:
    if (
        not v61_report.report_hash
        or v61_report.report_hash != v61_report.content_hash()
    ):
        raise ValueError("V6.1 report is not sealed")
    if v61_report.workspace_spec_hash != current_workspace_spec_hash:
        raise ValueError("V6.1 report belongs to another workspace")
    if v61_report.current_gate_hashes != dict(current_gate_hashes):
        raise ValueError("V6.1 report is stale for current gate hashes")


def _assert_provenance_bound(
    *,
    provenance: DataProvenanceBindingV62,
    v61_report: ScientificSuccessReportV61,
    current_workspace_spec_hash: str,
    current_gate_hashes: Mapping[StageId, str],
    current_s2_attempt: int,
    current_snapshot_hash: str,
) -> None:
    provenance.assert_sealed()
    if provenance.workspace_spec_hash != current_workspace_spec_hash:
        raise ValueError("provenance binding belongs to another workspace")
    if provenance.processed_snapshot_hash != current_snapshot_hash:
        raise ValueError("provenance binding references another snapshot")
    if provenance.fixture_only != v61_report.fixture_only:
        raise ValueError("provenance and V6.1 fixture scopes differ")
    if provenance.s1_gate_hash != current_gate_hashes.get("S1"):
        raise ValueError("provenance binding is stale for the current S1 gate")
    if provenance.s2_attempt != current_s2_attempt:
        raise ValueError("provenance binding is stale for the current S2 attempt")


def _assert_decision_bound(
    *,
    decision_evidence: DecisionValueEvidenceV62,
    v61_report: ScientificSuccessReportV61,
    current_snapshot_hash: str,
) -> None:
    if (
        not decision_evidence.evidence_hash
        or decision_evidence.evidence_hash
        != decision_evidence.content_hash()
    ):
        raise ValueError("decision evidence is not sealed")
    if decision_evidence.success_contract_hash != v61_report.contract_hash:
        raise ValueError("decision evidence binds another V6.1 contract")
    if decision_evidence.adapter_id != v61_report.adapter_id:
        raise ValueError("decision evidence binds another adapter")
    if decision_evidence.snapshot_hash != current_snapshot_hash:
        raise ValueError("decision evidence binds another snapshot")
    if decision_evidence.fixture_only != v61_report.fixture_only:
        raise ValueError("decision evidence and V6.1 fixture scopes differ")
    if decision_evidence.status in {"PASS", "FAIL"}:
        if not all(
            (
                decision_evidence.actual_values_hash,
                decision_evidence.model_actions_hash,
                decision_evidence.baseline_actions_hash,
            )
        ):
            raise ValueError("completed decision evidence lacks value hashes")
        if (
            decision_evidence.actual_values_hash
            != v61_report.rolling_confirmation.actual_values_hash
        ):
            raise ValueError(
                "decision evidence and V6.1 confirmation use different actuals"
            )


def _assert_stage_admission_bound(
    *,
    admission: StageEvidenceAdmissionV62,
    v61_report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    decision_evidence: DecisionValueEvidenceV62 | None,
    current_workspace_spec_hash: str,
    current_gate_hashes: Mapping[StageId, str],
    current_s2_attempt: int,
) -> None:
    admission.assert_sealed()
    if admission.status != "PASS":
        raise ValueError("component evidence was not admitted by current stages")
    if admission.workspace_spec_hash != current_workspace_spec_hash:
        raise ValueError("stage admission belongs to another workspace")
    if admission.v61_report_hash != v61_report.report_hash:
        raise ValueError("stage admission references another V6.1 report")
    if admission.current_gate_hashes != dict(current_gate_hashes):
        raise ValueError("stage admission is stale for current gates")
    if admission.s2_attempt != current_s2_attempt:
        raise ValueError("stage admission is stale for current S2 attempt")
    if admission.provenance_binding_hash != provenance.binding_hash:
        raise ValueError("stage admission references another provenance binding")
    if (
        admission.rolling_confirmation_hash
        != v61_report.rolling_confirmation.evidence_hash
    ):
        raise ValueError("stage admission references another confirmation")
    if admission.scientific_bundle_hash != v61_report.scientific_bundle_hash:
        raise ValueError("stage admission references another scientific bundle")
    if not all(
        admission.checks.get(check_id, False)
        for check_id in (
            "execution_semantic_artifacts_sealed",
            "execution_intent_ir_bound",
            "execution_resolution_receipt_bound",
            "execution_bundle_report_bound",
            "execution_receipt_replayed",
        )
    ):
        raise ValueError("stage admission lacks replayed executable semantics")
    expected_decision_hash = (
        decision_evidence.evidence_hash if decision_evidence else None
    )
    if admission.decision_evidence_hash != expected_decision_hash:
        raise ValueError("stage admission references other decision evidence")


def _assert_code_derived_admission(
    *,
    workspace: StageWorkspaceV50,
    admission: StageEvidenceAdmissionV62,
    v61_report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    decision_evidence: DecisionValueEvidenceV62 | None,
) -> None:
    derived_gate_hashes = {
        stage: str(gate)
        for stage in _STAGES
        if (gate := workspace.current_gate(stage)) is not None
    }
    if not workspace.verify():
        raise ValueError("stage-admission workspace does not verify")
    if workspace.spec.spec_hash != admission.workspace_spec_hash:
        raise ValueError("stage-admission workspace hash is not current")
    if derived_gate_hashes != admission.current_gate_hashes:
        raise ValueError("stage-admission gate hashes are not current")
    s2_certificate = workspace._certificate_for_current_node("S2")
    if (
        s2_certificate is None
        or workspace.current_gate("S2") != s2_certificate.certificate_hash
        or s2_certificate.attempt != admission.s2_attempt
    ):
        raise ValueError("current S2 attempt was not code-derived")
    rebuilt = build_stage_evidence_admission_v62(
        workspace=workspace,
        v61_report=v61_report,
        provenance=provenance,
        decision_evidence=decision_evidence,
        execution_intent_relative_path=admission.evidence_relative_paths[
            "execution_intent"
        ],
        execution_ir_relative_path=admission.evidence_relative_paths[
            "execution_ir"
        ],
        execution_resolution_relative_path=(
            admission.evidence_relative_paths["execution_resolution"]
        ),
        execution_receipt_relative_path=admission.evidence_relative_paths[
            "execution_receipt"
        ],
        scientific_bundle_relative_path=admission.evidence_relative_paths[
            "scientific_bundle"
        ],
        provenance_relative_path=admission.evidence_relative_paths[
            "provenance"
        ],
        rolling_confirmation_relative_path=admission.evidence_relative_paths[
            "rolling_confirmation"
        ],
        decision_relative_path=admission.evidence_relative_paths.get(
            "decision", DECISION_EVIDENCE_ADMISSION_PATH
        ),
    )
    if rebuilt != admission:
        raise ValueError(
            "stage admission differs from current authenticated manifests"
        )


def _assert_external_bound(
    *,
    binding: LocalExternalQualificationBindingV62,
    v61_report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    stage_admission: StageEvidenceAdmissionV62,
    decision_evidence: DecisionValueEvidenceV62 | None,
    current_s2_attempt: int,
    current_snapshot_hash: str,
) -> None:
    binding.assert_sealed()
    expected_decision_hash = (
        decision_evidence.evidence_hash if decision_evidence else None
    )
    if binding.workspace_spec_hash != v61_report.workspace_spec_hash:
        raise ValueError("external binding belongs to another workspace")
    if binding.v61_report_hash != v61_report.report_hash:
        raise ValueError("external binding references another V6.1 report")
    if binding.claim_kind != v61_report.claim_kind:
        raise ValueError("external binding references another claim kind")
    if binding.current_gate_hashes != v61_report.current_gate_hashes:
        raise ValueError("external binding is stale for current gates")
    if binding.s2_attempt != current_s2_attempt:
        raise ValueError("external binding is stale for the current S2 attempt")
    if binding.source_snapshot_hash != current_snapshot_hash:
        raise ValueError("external binding references another snapshot")
    if binding.source_verification_hash != provenance.source_verification_hash:
        raise ValueError("external binding references another source verification")
    if binding.provenance_binding_hash != provenance.binding_hash:
        raise ValueError("external binding references another provenance binding")
    if binding.stage_admission_hash != stage_admission.admission_hash:
        raise ValueError("external binding references another stage admission")
    if binding.decision_evidence_hash != expected_decision_hash:
        raise ValueError("external binding references different decision evidence")


def evaluate_scientific_closure_v62(
    *,
    workspace: StageWorkspaceV50,
    v61_report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    stage_admission: StageEvidenceAdmissionV62,
    decision_evidence: DecisionValueEvidenceV62 | None = None,
    external_binding: LocalExternalQualificationBindingV62 | None = None,
) -> ScientificClosureReportV62:
    """Compose a current V6.2 report without expanding claim authority."""

    _assert_code_derived_admission(
        workspace=workspace,
        admission=stage_admission,
        v61_report=v61_report,
        provenance=provenance,
        decision_evidence=decision_evidence,
    )
    current_workspace_spec_hash = stage_admission.workspace_spec_hash
    current_gate_hashes = stage_admission.current_gate_hashes
    current_s2_attempt = stage_admission.s2_attempt
    current_snapshot_hash = provenance.processed_snapshot_hash
    _assert_v61_current(
        v61_report=v61_report,
        current_workspace_spec_hash=current_workspace_spec_hash,
        current_gate_hashes=current_gate_hashes,
    )
    _assert_provenance_bound(
        provenance=provenance,
        v61_report=v61_report,
        current_workspace_spec_hash=current_workspace_spec_hash,
        current_gate_hashes=current_gate_hashes,
        current_s2_attempt=current_s2_attempt,
        current_snapshot_hash=current_snapshot_hash,
    )
    if decision_evidence is not None:
        _assert_decision_bound(
            decision_evidence=decision_evidence,
            v61_report=v61_report,
            current_snapshot_hash=current_snapshot_hash,
        )
    _assert_stage_admission_bound(
        admission=stage_admission,
        v61_report=v61_report,
        provenance=provenance,
        decision_evidence=decision_evidence,
        current_workspace_spec_hash=current_workspace_spec_hash,
        current_gate_hashes=current_gate_hashes,
        current_s2_attempt=current_s2_attempt,
    )
    if external_binding is not None:
        _assert_external_bound(
            binding=external_binding,
            v61_report=v61_report,
            provenance=provenance,
            stage_admission=stage_admission,
            decision_evidence=decision_evidence,
            current_s2_attempt=current_s2_attempt,
            current_snapshot_hash=current_snapshot_hash,
        )

    v61_dimensions: dict[str, ScientificSuccessDimensionV61] = {
        item.dimension_id: item for item in v61_report.dimensions
    }
    local_required = set(_local_required_ids(v61_report.claim_kind))
    closure_required = set(_closure_required_ids(v61_report.claim_kind))

    dimensions: list[ScientificClosureDimensionV62] = []
    for dimension_id in (
        "workflow_integrity",
        "local_adapter_checks",
        "leakage_safe_confirmation",
        "mechanism_identification",
    ):
        prior = v61_dimensions[dimension_id]
        dimensions.append(
            _dimension(
                dimension_id=dimension_id,
                status=prior.status,
                required_ids=closure_required,
                reason_codes=prior.reason_codes,
                evidence_refs=prior.evidence_refs,
                metrics=prior.metrics,
            )
        )

    provenance_status: DimensionStatusV61 = (
        provenance.scientific_provenance_status
    )
    provenance_reasons = list(provenance.reason_codes)
    if provenance.status == "PASS" and v61_report.fixture_only:
        provenance_reasons.append("fixture_only")
    elif provenance.status == "PASS":
        provenance_reasons.append(
            "independent_measurement_review_required"
        )
    dimensions.append(
        _dimension(
            dimension_id="data_provenance",
            status=provenance_status,
            required_ids=closure_required,
            reason_codes=provenance_reasons,
            evidence_refs=[
                str(provenance.binding_hash),
                provenance.source_verification_hash,
                provenance.processed_snapshot_hash,
            ],
            metrics={
                "fixture_only": v61_report.fixture_only,
                "mechanical_provenance_checks_pass": (
                    provenance.status == "PASS"
                ),
                "independent_measurement_review": (
                    provenance.independent_measurement_review
                ),
            },
        )
    )

    if decision_evidence is None:
        decision_status: DimensionStatusV61 = "NOT_RUN"
        decision_reasons = ["decision_value_evidence_absent"]
        decision_refs: list[str] = []
        decision_metrics: dict[str, float | int | bool | None] = {}
    else:
        decision_status = decision_evidence.scientific_decision_status
        decision_reasons = list(decision_evidence.reason_codes)
        decision_reasons.append("local_retrospective_only")
        decision_refs = [str(decision_evidence.evidence_hash)]
        decision_metrics = {
            **decision_evidence.metrics,
            "mechanical_decision_checks_pass": (
                decision_evidence.status == "PASS"
            ),
            "prospective_trial_completed": (
                decision_evidence.prospective_trial_completed
            ),
        }
    dimensions.append(
        _dimension(
            dimension_id="decision_value",
            status=decision_status,
            required_ids=closure_required,
            reason_codes=decision_reasons,
            evidence_refs=decision_refs,
            metrics=decision_metrics,
        )
    )

    external_reasons = (
        list(external_binding.reason_codes)
        if external_binding is not None
        else ["external_evidence_absent"]
    )
    external_refs = (
        [
            str(external_binding.binding_hash),
            *(
                [external_binding.external_artifact_hash]
                if external_binding.external_artifact_hash
                else []
            ),
        ]
        if external_binding is not None
        else []
    )
    dimensions.extend(
        [
            _dimension(
                dimension_id="external_generalization",
                status="NOT_RUN",
                required_ids=closure_required,
                reason_codes=external_reasons,
                evidence_refs=external_refs,
                metrics={"independent_signature_verified": False},
            ),
            _dimension(
                dimension_id="scientific_qualification",
                status="NOT_RUN",
                required_ids=closure_required,
                reason_codes=external_reasons,
                evidence_refs=external_refs,
                metrics={"independent_signature_verified": False},
            ),
        ]
    )
    dimensions.sort(key=lambda item: item.dimension_id)

    local_status = _aggregate(dimensions, local_required)
    closure_status = _aggregate(dimensions, closure_required)
    if v61_report.fixture_only:
        ceiling: ClaimCeilingV61 = "fixture_protocol_only"
    elif local_status == "PASS" and (
        "leakage_safe_confirmation" in local_required
    ):
        ceiling = "local_leakage_safe_predictive_evidence"
    elif (
        v61_dimensions["local_adapter_checks"].status == "PASS"
        and provenance_status == "PASS"
    ):
        ceiling = "local_retrospective_adapter_evidence"
    elif v61_dimensions["workflow_integrity"].status == "PASS":
        ceiling = "workflow_integrity_only"
    else:
        ceiling = "no_scientific_claim"

    return ScientificClosureReportV62.seal(
        workspace_spec_hash=v61_report.workspace_spec_hash,
        v61_contract_hash=v61_report.contract_hash,
        v61_report_hash=v61_report.report_hash,
        claim_kind=v61_report.claim_kind,
        current_gate_hashes=dict(current_gate_hashes),
        s2_attempt=current_s2_attempt,
        source_snapshot_hash=current_snapshot_hash,
        source_verification_hash=provenance.source_verification_hash,
        provenance_binding_hash=provenance.binding_hash,
        stage_admission_hash=stage_admission.admission_hash,
        decision_evidence_hash=(
            decision_evidence.evidence_hash if decision_evidence else None
        ),
        external_binding_hash=(
            external_binding.binding_hash if external_binding else None
        ),
        fixture_only=v61_report.fixture_only,
        local_required_dimension_ids=sorted(local_required),
        closure_required_dimension_ids=sorted(closure_required),
        dimensions=dimensions,
        local_evidence_status=local_status,
        scientific_closure_status=closure_status,
        claim_ceiling=ceiling,
    )


def verify_scientific_closure_v62(
    *,
    workspace: StageWorkspaceV50,
    report: ScientificClosureReportV62,
    v61_report: ScientificSuccessReportV61,
    provenance: DataProvenanceBindingV62,
    stage_admission: StageEvidenceAdmissionV62,
    decision_evidence: DecisionValueEvidenceV62 | None = None,
    external_binding: LocalExternalQualificationBindingV62 | None = None,
) -> ScientificClosureVerificationV62:
    """Recompute the report and fail closed on stale or substituted evidence."""

    expected_decision_hash = (
        decision_evidence.evidence_hash if decision_evidence else None
    )
    expected_external_hash = (
        external_binding.binding_hash if external_binding else None
    )
    try:
        _assert_code_derived_admission(
            workspace=workspace,
            admission=stage_admission,
            v61_report=v61_report,
            provenance=provenance,
            decision_evidence=decision_evidence,
        )
    except (KeyError, OSError, TypeError, ValueError):
        code_derived_admission = False
    else:
        code_derived_admission = True
    current_workspace_spec_hash = workspace.spec.spec_hash
    current_gate_hashes = {
        stage: str(gate)
        for stage in _STAGES
        if (gate := workspace.current_gate(stage)) is not None
    }
    s2_certificate = workspace._certificate_for_current_node("S2")
    current_s2_attempt = (
        s2_certificate.attempt if s2_certificate is not None else 0
    )
    current_snapshot_hash = provenance.processed_snapshot_hash
    checks = {
        "report_self_hash": bool(report.report_hash)
        and report.report_hash == report.content_hash(),
        "v61_report_self_hash": bool(v61_report.report_hash)
        and v61_report.report_hash == v61_report.content_hash(),
        "v61_report_binding": report.v61_report_hash
        == v61_report.report_hash,
        "workspace_binding_current": (
            report.workspace_spec_hash == current_workspace_spec_hash
            and v61_report.workspace_spec_hash == current_workspace_spec_hash
        ),
        "gate_binding_current": (
            report.current_gate_hashes == dict(current_gate_hashes)
            and v61_report.current_gate_hashes == dict(current_gate_hashes)
        ),
        "s2_attempt_binding_current": (
            report.s2_attempt == current_s2_attempt
            and provenance.s2_attempt == current_s2_attempt
        ),
        "snapshot_binding_current": (
            report.source_snapshot_hash == current_snapshot_hash
            and provenance.processed_snapshot_hash == current_snapshot_hash
        ),
        "provenance_binding_self_hash": bool(provenance.binding_hash)
        and provenance.binding_hash == provenance.content_hash(),
        "source_verification_binding": report.source_verification_hash
        == provenance.source_verification_hash,
        "provenance_binding": report.provenance_binding_hash
        == provenance.binding_hash,
        "stage_admission_self_hash": bool(stage_admission.admission_hash)
        and stage_admission.admission_hash == stage_admission.content_hash(),
        "stage_admission_binding": (
            report.stage_admission_hash == stage_admission.admission_hash
        ),
        "stage_admission_passed": stage_admission.status == "PASS",
        "stage_admission_code_derived": code_derived_admission,
        "executable_semantics_admitted": (
            stage_admission.scientific_bundle_hash
            == v61_report.scientific_bundle_hash
            and all(
                stage_admission.checks.get(check_id, False)
                for check_id in (
                    "execution_semantic_artifacts_sealed",
                    "execution_intent_ir_bound",
                    "execution_resolution_receipt_bound",
                    "execution_bundle_report_bound",
                    "execution_receipt_replayed",
                )
            )
        ),
        "decision_evidence_binding": report.decision_evidence_hash
        == expected_decision_hash,
        "external_binding": report.external_binding_hash
        == expected_external_hash,
        "qualification_not_locally_granted": (
            not report.scientific_qualification_granted
        ),
        "real_world_action_not_authorized": (
            not report.real_world_action_authorized
        ),
        "deterministic_recomputation": False,
    }
    try:
        recomputed = evaluate_scientific_closure_v62(
            workspace=workspace,
            v61_report=v61_report,
            provenance=provenance,
            stage_admission=stage_admission,
            decision_evidence=decision_evidence,
            external_binding=external_binding,
        )
    except (KeyError, TypeError, ValueError):
        pass
    else:
        checks["deterministic_recomputation"] = recomputed == report

    reasons = sorted(
        check_id for check_id, passed in checks.items() if not passed
    )
    return ScientificClosureVerificationV62.seal(
        report_hash=report.report_hash,
        current_binding_hash=sha256_value(
            {
                "workspace_spec_hash": current_workspace_spec_hash,
                "current_gate_hashes": dict(current_gate_hashes),
                "current_s2_attempt": current_s2_attempt,
                "current_snapshot_hash": current_snapshot_hash,
            }
        ),
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        reason_codes=reasons,
    )


def _closure_attempt_root(workspace: StageWorkspaceV50) -> Path:
    certificate = workspace._certificate_for_current_node("S6")
    if (
        certificate is None
        or workspace.current_gate("S6") != certificate.certificate_hash
        or not workspace.verify_certificate(certificate)
    ):
        raise ValueError("scientific closure requires a current authenticated S6")
    return workspace.root / SCIENTIFIC_CLOSURE_ROOT / f"a{certificate.attempt}"


def _write_projection_exact(path: Path, model: StrictModel) -> None:
    payload = canonical_json(model) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(
                f"existing scientific-closure projection differs: {path.name}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        payload,
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def materialize_scientific_closure_v62(
    workspace: StageWorkspaceV50,
) -> tuple[
    StageEvidenceAdmissionV62,
    ScientificClosureReportV62,
    ScientificClosureVerificationV62,
    dict[str, str],
]:
    """Materialize a post-S6 local closure without entering any stage manifest."""

    root = workspace.root
    v61_report = ScientificSuccessReportV61.model_validate_json(
        (root / SUCCESS_PROJECTION_PATH).read_text(encoding="utf-8")
    )
    provenance = DataProvenanceBindingV62.model_validate_json(
        (root / PROVENANCE_BINDING_PATH).read_text(encoding="utf-8")
    )
    decision_path = root / DECISION_EVIDENCE_PATH
    decision_evidence = (
        DecisionValueEvidenceV62.model_validate_json(
            decision_path.read_text(encoding="utf-8")
        )
        if decision_path.is_file()
        else None
    )
    admission = build_stage_evidence_admission_v62(
        workspace=workspace,
        v61_report=v61_report,
        provenance=provenance,
        decision_evidence=decision_evidence,
    )
    if admission.status != "PASS":
        raise ValueError(
            "current stage evidence is not admissible: "
            + ", ".join(admission.reason_codes)
        )
    report = evaluate_scientific_closure_v62(
        workspace=workspace,
        v61_report=v61_report,
        provenance=provenance,
        stage_admission=admission,
        decision_evidence=decision_evidence,
    )
    verification = verify_scientific_closure_v62(
        workspace=workspace,
        report=report,
        v61_report=v61_report,
        provenance=provenance,
        stage_admission=admission,
        decision_evidence=decision_evidence,
    )
    if verification.status != "PASS":
        raise ValueError(
            "scientific closure replay failed: "
            + ", ".join(verification.reason_codes)
        )

    evidence_hashes = {
        "admission": workspace.commit_evidence(
            "stage_evidence_admission_v62",
            admission.model_dump(mode="json"),
        ).sha256,
        "report": workspace.commit_evidence(
            "scientific_closure_report_v62",
            report.model_dump(mode="json"),
        ).sha256,
        "verification": workspace.commit_evidence(
            "scientific_closure_verification_v62",
            verification.model_dump(mode="json"),
        ).sha256,
    }
    attempt_root = _closure_attempt_root(workspace)
    _write_projection_exact(attempt_root / "admission.json", admission)
    _write_projection_exact(attempt_root / "report.json", report)
    _write_projection_exact(attempt_root / "verification.json", verification)
    return admission, report, verification, evidence_hashes


def _empty_closure_summary(
    *,
    reason_codes: list[str] | None = None,
    source_integrity_status: str = "NOT_RUN",
    scientific_provenance_status: str = "NOT_RUN",
) -> dict[str, object]:
    return {
        "schema_version": "6.2",
        "evaluated": False,
        "source_integrity_status": source_integrity_status,
        "scientific_provenance_status": scientific_provenance_status,
        "decision_evidence_status": "NOT_RUN",
        "scientific_decision_status": "NOT_RUN",
        "stage_admission_status": "NOT_RUN",
        "closure_verification_status": "NOT_RUN",
        "local_evidence_status": "NOT_RUN",
        "scientific_closure_status": "NOT_RUN",
        "claim_ceiling": "no_scientific_claim",
        "fixture_only": None,
        "dimensions": {},
        "reason_codes": sorted(set(reason_codes or [])),
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


def scientific_closure_summary_v62(
    workspace: StageWorkspaceV50,
) -> dict[str, object]:
    """Read and replay the current attempt-scoped closure projection."""

    root = workspace.root
    provenance_path = root / PROVENANCE_BINDING_PATH
    if not _current_stage_file_admitted(
        workspace,
        stage="S2",
        relative_path=PROVENANCE_BINDING_PATH,
    ):
        return _empty_closure_summary(
            reason_codes=["current_s2_provenance_not_admitted"]
        )
    try:
        provenance = DataProvenanceBindingV62.model_validate_json(
            provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return _empty_closure_summary(
            reason_codes=["authenticated_provenance_binding_absent"]
        )
    source_status = provenance.status
    scientific_source_status = provenance.scientific_provenance_status
    try:
        attempt_root = _closure_attempt_root(workspace)
    except ValueError:
        return _empty_closure_summary(
            reason_codes=["current_s6_certificate_absent"],
            source_integrity_status=source_status,
            scientific_provenance_status=scientific_source_status,
        )
    required = {
        "admission": attempt_root / "admission.json",
        "report": attempt_root / "report.json",
        "verification": attempt_root / "verification.json",
    }
    if not all(path.is_file() for path in required.values()):
        return _empty_closure_summary(
            reason_codes=["current_closure_projection_absent"],
            source_integrity_status=source_status,
            scientific_provenance_status=scientific_source_status,
        )
    try:
        admission = StageEvidenceAdmissionV62.model_validate_json(
            required["admission"].read_text(encoding="utf-8")
        )
        report = ScientificClosureReportV62.model_validate_json(
            required["report"].read_text(encoding="utf-8")
        )
        stored_verification = ScientificClosureVerificationV62.model_validate_json(
            required["verification"].read_text(encoding="utf-8")
        )
        v61_report = ScientificSuccessReportV61.model_validate_json(
            (root / SUCCESS_PROJECTION_PATH).read_text(encoding="utf-8")
        )
        decision_path = root / DECISION_EVIDENCE_PATH
        decision_evidence = (
            DecisionValueEvidenceV62.model_validate_json(
                decision_path.read_text(encoding="utf-8")
            )
            if decision_path.is_file()
            else None
        )
        replayed = verify_scientific_closure_v62(
            workspace=workspace,
            report=report,
            v61_report=v61_report,
            provenance=provenance,
            stage_admission=admission,
            decision_evidence=decision_evidence,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _empty_closure_summary(
            reason_codes=[
                "current_closure_projection_invalid",
                type(exc).__name__.lower(),
            ],
            source_integrity_status=source_status,
            scientific_provenance_status=scientific_source_status,
        )
    verification_current = (
        stored_verification == replayed and replayed.status == "PASS"
    )
    return {
        "schema_version": "6.2",
        "evaluated": True,
        "source_integrity_status": source_status,
        "scientific_provenance_status": scientific_source_status,
        "decision_evidence_status": (
            decision_evidence.status if decision_evidence else "NOT_RUN"
        ),
        "scientific_decision_status": (
            decision_evidence.scientific_decision_status
            if decision_evidence
            else "NOT_RUN"
        ),
        "stage_admission_status": admission.status,
        "closure_verification_status": (
            "PASS" if verification_current else "FAIL"
        ),
        "local_evidence_status": report.local_evidence_status,
        "scientific_closure_status": report.scientific_closure_status,
        "claim_ceiling": report.claim_ceiling,
        "fixture_only": report.fixture_only,
        "dimensions": {
            item.dimension_id: item.model_dump(mode="json")
            for item in report.dimensions
        },
        "reason_codes": (
            []
            if verification_current
            else sorted(
                set(
                    stored_verification.reason_codes
                    + replayed.reason_codes
                    + ["stored_closure_verification_stale"]
                )
            )
        ),
        "report_hash": report.report_hash,
        "admission_hash": admission.admission_hash,
        "verification_hash": stored_verification.verification_hash,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


__all__ = [
    "ADAPTIVE_SCIENTIFIC_BUNDLE_ADMISSION_PATH",
    "DECISION_EVIDENCE_ADMISSION_PATH",
    "LocalExternalQualificationBindingV62",
    "ODE_SCIENTIFIC_BUNDLE_ADMISSION_PATH",
    "ROLLING_CONFIRMATION_ADMISSION_PATH",
    "SCIENTIFIC_CLOSURE_ROOT",
    "ScientificClosureDimensionV62",
    "ScientificClosureReportV62",
    "ScientificClosureVerificationV62",
    "StageEvidenceAdmissionV62",
    "build_stage_evidence_admission_v62",
    "evaluate_scientific_closure_v62",
    "materialize_scientific_closure_v62",
    "scientific_closure_summary_v62",
    "verify_scientific_closure_v62",
]
