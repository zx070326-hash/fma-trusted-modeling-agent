"""V6.3 external qualification for one narrow predictive claim.

This module is additive over V6.2.  It does not reinterpret local closure
artifacts and it does not authorize deployment.  A qualification can only be
issued after replaying the current V6.2 closure from a ``StageWorkspaceV50``,
binding an immutable S4 prediction seal, verifying four role-separated
Ed25519 signatures, and consuming exactly one aggregate external evaluation.

The adapter deliberately supports only aggregate normalized RMSE for the
registered positive-series predictive path.  It is not a generic scientific
or causal qualification mechanism.
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import PredictionSealV50, RoleExecutionReceiptV50

from .scientific_closure import (
    ScientificClosureReportV62,
    StageEvidenceAdmissionV62,
    _closure_attempt_root,
    scientific_closure_summary_v62,
)
from .executable_candidate import (
    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    ExecutableCandidateReceiptV62,
)


FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FinitePositive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
PredictiveClaimCeilingV63 = Literal[
    "no_scientific_claim",
    "workflow_integrity_only",
    "local_retrospective_adapter_evidence",
    "local_leakage_safe_predictive_evidence",
    "fixture_protocol_only",
    "externally_qualified_predictive_evidence",
]
LocalClaimCeilingV63 = Literal[
    "no_scientific_claim",
    "workflow_integrity_only",
    "local_retrospective_adapter_evidence",
    "local_leakage_safe_predictive_evidence",
]
QualificationStatusV63 = Literal[
    "NOT_RUN",
    "REJECTED",
    "EXTERNALLY_QUALIFIED",
]
_CONSUMPTION_KIND = "external_evaluation_consumption_v63"
_FORECAST_INPUT_KIND = "external_forecast_input_v63"
_PREDICTION_BINDING_KIND = "current_model_prediction_binding_v63"
_PREDICTION_VECTOR_KIND = "external_prediction_vector_v63"
_RESERVATION_KIND = "external_evaluation_reservation_v63"
_PREDICTION_OUTPUT_SCHEMA_HASH = sha256_value(
    {
        "schema": "external_positive_series_prediction_vector",
        "schema_version": "6.3",
        "values": "ordered_finite_predictions_without_targets",
    }
)


class ExternalQualificationError(RuntimeError):
    """A fail-closed V6.3 chain validation error."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _unsigned_bytes(model: StrictModel, hash_field: str) -> bytes:
    return canonical_json(
        model.model_dump(
            mode="json",
            exclude={"signature_base64", hash_field},
        )
    ).encode("utf-8")


def _load_private_key(private_key_pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("V6.3 signing key must be Ed25519")
    return key


def _load_public_key(public_key_pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("V6.3 trusted key must be Ed25519")
    return key


def external_qualification_key_fingerprint_v63(
    public_key_pem: bytes,
) -> str:
    key = _load_public_key(public_key_pem)
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _verify_signature(
    *,
    model: StrictModel,
    key_id: str,
    signature_base64: str | None,
    trusted_public_keys: Mapping[str, bytes],
    hash_field: str,
) -> bool:
    public_key_pem = trusted_public_keys.get(key_id)
    if public_key_pem is None or not signature_base64:
        return False
    try:
        signature = base64.b64decode(
            signature_base64.encode("ascii"),
            validate=True,
        )
        _load_public_key(public_key_pem).verify(
            signature,
            _unsigned_bytes(model, hash_field),
        )
    except (InvalidSignature, TypeError, ValueError):
        return False
    return True


def _sign_model(
    *,
    model_type: type[StrictModel],
    data: dict[str, object],
    private_key_pem: bytes,
    hash_field: str,
) -> StrictModel:
    unsigned = model_type(**data)
    signature = _load_private_key(private_key_pem).sign(
        _unsigned_bytes(unsigned, hash_field)
    )
    tagged_payload = unsigned.model_dump(mode="json")
    tagged_payload["signature_base64"] = base64.b64encode(signature).decode(
        "ascii"
    )
    tagged = model_type(**tagged_payload)
    final_payload = tagged.model_dump(mode="json")
    final_payload[hash_field] = _hash_without(tagged, hash_field)
    return model_type(**final_payload)


def _commit_unique_qualification_artifact(
    *,
    workspace: StageWorkspaceV50,
    kind: str,
    model: StrictModel,
    model_type: type[Any],
    allow_create: bool = True,
) -> str:
    """Persist one exact role envelope per qualification, atomically."""

    qualification_id = getattr(model, "qualification_id", None)
    if not isinstance(qualification_id, str):
        raise ExternalQualificationError(
            f"{kind} lacks a qualification identity"
        )
    task_id = getattr(model, "task_id", None)
    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    transaction = (
        transaction_factory()
        if callable(transaction_factory)
        else nullcontext()
    )
    with transaction:
        try:
            prior = [
                (reference, item)
                for reference, item in workspace._artifacts_of_kind(
                    kind,
                    model_type,
                )
                if (
                    item.qualification_id == qualification_id
                    or (
                        isinstance(task_id, str)
                        and getattr(item, "task_id", None) == task_id
                    )
                )
            ]
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ExternalQualificationError(
                f"{kind} ledger could not be replayed"
            ) from exc
        exact = [
            reference.sha256
            for reference, item in prior
            if item == model
        ]
        if exact:
            if len(prior) != 1 or len(exact) != 1:
                raise ExternalQualificationError(
                    f"{kind} ledger contains duplicates"
                )
            return exact[0]
        if prior:
            raise ExternalQualificationError(
                f"{kind} is immutable for this qualification"
            )
        if not allow_create:
            raise ExternalQualificationError(
                f"{kind} is not committed"
            )
        reference = workspace.commit_evidence(
            kind,
            model.model_dump(mode="json"),
        )
        return reference.sha256


def _exact_committed_artifact_hash(
    *,
    workspace: StageWorkspaceV50,
    kind: str,
    model: StrictModel,
    model_type: type[Any],
) -> str:
    try:
        exact = [
            reference.sha256
            for reference, item in workspace._artifacts_of_kind(
                kind,
                model_type,
            )
            if item == model
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationError(
            f"{kind} ledger could not be replayed"
        ) from exc
    if len(exact) != 1:
        raise ExternalQualificationError(
            f"{kind} exact committed envelope is absent or duplicated"
        )
    return exact[0]


class PredictiveLocalContextV63(StrictModel):
    """Code-derived view of the current V6.2 predictive prerequisites."""

    schema_version: Literal["6.3-predictive-local-context"] = (
        "6.3-predictive-local-context"
    )
    workspace_spec_hash: Sha256
    v62_report_hash: Sha256
    v62_admission_hash: Sha256
    v62_verification_hash: Sha256
    s4_gate_hash: Sha256
    s6_gate_hash: Sha256
    scientific_bundle_hash: Sha256
    processed_snapshot_hash: Sha256
    executable_candidate_receipt_hash: Sha256
    selected_model_id: Identifier
    selected_model_identity_hash: Sha256
    closure_summary_hash: Sha256
    claim_kind: Literal["predictive"] = "predictive"
    workflow_integrity_status: Literal["PASS"] = "PASS"
    local_adapter_status: Literal["PASS"] = "PASS"
    rolling_confirmation_status: Literal["PASS"] = "PASS"
    source_integrity_status: Literal["PASS"] = "PASS"
    fixture_only: Literal[False] = False
    prior_claim_ceiling: LocalClaimCeilingV63
    context_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_context(self) -> "PredictiveLocalContextV63":
        if self.context_hash and self.context_hash != self.content_hash():
            raise ValueError("V6.3 local-context hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "context_hash")

    @classmethod
    def seal(cls, **data: object) -> "PredictiveLocalContextV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"context_hash"})
        payload["context_hash"] = draft.content_hash()
        return cls(**payload)


def derive_predictive_local_context_v63(
    workspace: StageWorkspaceV50,
) -> PredictiveLocalContextV63:
    """Replay V6.2 and derive, rather than accept, local PASS flags."""

    try:
        summary = scientific_closure_summary_v62(workspace)
        s4_gate_hash = workspace.current_gate("S4")
        s6_gate_hash = workspace.current_gate("S6")
        dimensions = summary.get("dimensions")
        if not isinstance(dimensions, dict):
            raise ExternalQualificationError("V6.2 dimensions are absent")

        def dimension_status(dimension_id: str) -> object:
            value = dimensions.get(dimension_id)
            return value.get("status") if isinstance(value, dict) else None

        claim_kind = summary.get("claim_kind")
        model_binding_values = {
            key: summary.get(key)
            for key in (
                "scientific_bundle_hash",
                "processed_snapshot_hash",
                "executable_candidate_receipt_hash",
                "selected_model_id",
                "selected_model_identity_hash",
            )
        }
        if claim_kind is None or any(
            value is None for value in model_binding_values.values()
        ):
            report_path = _closure_attempt_root(workspace) / "report.json"
            closure_report = ScientificClosureReportV62.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
            closure_report.assert_sealed()
            if closure_report.report_hash != summary.get("report_hash"):
                raise ExternalQualificationError(
                    "V6.2 summary and closure report hashes differ"
                )
            claim_kind = closure_report.claim_kind
            admission = StageEvidenceAdmissionV62.model_validate_json(
                (
                    _closure_attempt_root(workspace) / "admission.json"
                ).read_text(encoding="utf-8")
            )
            admission.assert_sealed()
            if (
                closure_report.stage_admission_hash
                != admission.admission_hash
                or summary.get("admission_hash")
                != admission.admission_hash
            ):
                raise ExternalQualificationError(
                    "V6.2 report and stage admission differ"
                )
            execution_receipt = (
                ExecutableCandidateReceiptV62.model_validate_json(
                    (
                        workspace.root
                        / EXECUTABLE_CANDIDATE_RECEIPT_PATH
                    ).read_text(encoding="utf-8")
                )
            )
            execution_receipt.assert_sealed()
            if (
                execution_receipt.receipt_hash
                != admission.executable_candidate_receipt_hash
                or execution_receipt.bundle_hash
                != admission.scientific_bundle_hash
                or not execution_receipt.bundle_scientific_acceptance
                or execution_receipt.fixture_only
            ):
                raise ExternalQualificationError(
                    "current executable candidate receipt is not qualifying"
                )
            selected_identity_hash = sha256_value(
                {
                    "selected_model_id": execution_receipt.selected_model_id,
                    "selected_candidate_structural_hash": (
                        execution_receipt.selected_candidate_structural_hash
                    ),
                    "executable_candidate_receipt_hash": (
                        execution_receipt.receipt_hash
                    ),
                    "scientific_bundle_hash": execution_receipt.bundle_hash,
                }
            )
            model_binding_values = {
                "scientific_bundle_hash": admission.scientific_bundle_hash,
                "processed_snapshot_hash": (
                    closure_report.source_snapshot_hash
                ),
                "executable_candidate_receipt_hash": (
                    execution_receipt.receipt_hash
                ),
                "selected_model_id": execution_receipt.selected_model_id,
                "selected_model_identity_hash": selected_identity_hash,
            }
        checks = {
            "workspace_verified": bool(workspace.verify()),
            "closure_evaluated": summary.get("evaluated") is True,
            "closure_verification_pass": (
                summary.get("closure_verification_status") == "PASS"
            ),
            "stage_admission_pass": (
                summary.get("stage_admission_status") == "PASS"
            ),
            "non_fixture": summary.get("fixture_only") is False,
            "source_integrity_pass": (
                summary.get("source_integrity_status") == "PASS"
            ),
            "workflow_integrity_pass": (
                dimension_status("workflow_integrity") == "PASS"
            ),
            "local_adapter_pass": (
                dimension_status("local_adapter_checks") == "PASS"
            ),
            "rolling_confirmation_pass": (
                dimension_status("leakage_safe_confirmation") == "PASS"
            ),
            "predictive_claim_scope": claim_kind == "predictive",
            "current_s4_present": bool(s4_gate_hash),
            "current_s6_present": bool(s6_gate_hash),
        }
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            raise ExternalQualificationError(
                "V6.3 local predictive prerequisites rejected: "
                + ",".join(failed)
            )
        report_hash = summary.get("report_hash")
        admission_hash = summary.get("admission_hash")
        verification_hash = summary.get("verification_hash")
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (report_hash, admission_hash, verification_hash)
        ):
            raise ExternalQualificationError(
                "V6.2 closure hashes are absent or malformed"
            )
        hash_binding_keys = {
            "scientific_bundle_hash",
            "processed_snapshot_hash",
            "executable_candidate_receipt_hash",
            "selected_model_identity_hash",
        }
        if not all(
            isinstance(model_binding_values[key], str)
            and len(model_binding_values[key]) == 64
            for key in hash_binding_keys
        ) or not isinstance(
            model_binding_values["selected_model_id"],
            str,
        ):
            raise ExternalQualificationError(
                "V6.2 current-model binding is absent or malformed"
            )
        prior_claim_ceiling = summary.get("claim_ceiling")
        if prior_claim_ceiling not in {
            "no_scientific_claim",
            "workflow_integrity_only",
            "local_retrospective_adapter_evidence",
            "local_leakage_safe_predictive_evidence",
        }:
            raise ExternalQualificationError(
                "V6.2 predictive claim ceiling is invalid"
            )
        return PredictiveLocalContextV63.seal(
            workspace_spec_hash=workspace.spec.spec_hash,
            v62_report_hash=report_hash,
            v62_admission_hash=admission_hash,
            v62_verification_hash=verification_hash,
            s4_gate_hash=s4_gate_hash,
            s6_gate_hash=s6_gate_hash,
            scientific_bundle_hash=model_binding_values[
                "scientific_bundle_hash"
            ],
            processed_snapshot_hash=model_binding_values[
                "processed_snapshot_hash"
            ],
            executable_candidate_receipt_hash=model_binding_values[
                "executable_candidate_receipt_hash"
            ],
            selected_model_id=model_binding_values["selected_model_id"],
            selected_model_identity_hash=model_binding_values[
                "selected_model_identity_hash"
            ],
            closure_summary_hash=sha256_value(summary),
            prior_claim_ceiling=prior_claim_ceiling,
        )
    except ExternalQualificationError:
        raise
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        raise ExternalQualificationError(
            "V6.3 could not replay current V6.2 closure"
        ) from exc


class PredictiveExternalQualificationContractV63(StrictModel):
    """Threshold and graph bindings frozen before private evaluation."""

    schema_version: Literal["6.3-predictive-qualification-contract"] = (
        "6.3-predictive-qualification-contract"
    )
    qualification_id: Identifier
    task_id: Identifier
    local_context_hash: Sha256
    workspace_spec_hash: Sha256
    v62_report_hash: Sha256
    s4_gate_hash: Sha256
    s6_gate_hash: Sha256
    scientific_bundle_hash: Sha256
    processed_snapshot_hash: Sha256
    executable_candidate_receipt_hash: Sha256
    selected_model_id: Identifier
    selected_model_identity_hash: Sha256
    prediction_output_schema_hash: Sha256 = _PREDICTION_OUTPUT_SCHEMA_HASH
    metric: Literal["normalized_root_mean_squared_error"] = (
        "normalized_root_mean_squared_error"
    )
    metric_formula: Literal[
        "sqrt(squared_error_sum/target_squared_value_sum)"
    ] = (
        "sqrt(squared_error_sum/target_squared_value_sum)"
    )
    normalization_scale_policy: Literal[
        "external_target_rms_from_precommitted_target_square_sum"
    ] = "external_target_rms_from_precommitted_target_square_sum"
    maximum_metric_value: FiniteNonNegative
    minimum_external_observation_count: Annotated[int, Field(ge=3)]
    maximum_external_evaluations: Literal[1] = 1
    aggregate_feedback_only: Literal[True] = True
    predictive_claim_only: Literal[True] = True
    frozen_before_private_evaluation: Literal[True] = True
    trusted_authority_key_ids: dict[Identifier, Identifier]
    trusted_authority_key_fingerprints: dict[Identifier, Sha256]
    trusted_authority_set_hash: Sha256
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    frozen_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "PredictiveExternalQualificationContractV63":
        expected_roles = {"custody", "registry", "evaluator", "promotion"}
        if (
            self.prediction_output_schema_hash
            != _PREDICTION_OUTPUT_SCHEMA_HASH
        ):
            raise ValueError("V6.3 prediction output schema hash differs")
        if set(self.trusted_authority_key_ids) != expected_roles:
            raise ValueError("V6.3 contract must pin all four authority key IDs")
        if set(self.trusted_authority_key_fingerprints) != expected_roles:
            raise ValueError(
                "V6.3 contract must pin all four authority fingerprints"
            )
        if len(set(self.trusted_authority_key_ids.values())) != 4:
            raise ValueError("V6.3 authority key IDs must be pairwise distinct")
        if len(set(self.trusted_authority_key_fingerprints.values())) != 4:
            raise ValueError(
                "V6.3 authorities must use distinct physical keys"
            )
        expected_authority_hash = sha256_value(
            {
                "key_ids": self.trusted_authority_key_ids,
                "fingerprints": self.trusted_authority_key_fingerprints,
            }
        )
        if self.trusted_authority_set_hash != expected_authority_hash:
            raise ValueError("V6.3 trusted authority set hash differs")
        if self.coordinator_host_id == self.generator_host_id:
            raise ValueError("coordinator and generator hosts must differ")
        _assert_aware(self.frozen_at, "frozen_at")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("V6.3 qualification-contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("V6.3 qualification contract is not sealed")

    @classmethod
    def seal(
        cls, **data: object
    ) -> "PredictiveExternalQualificationContractV63":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalForecastInputV63(StrictModel):
    """Public, target-ordered forecast coordinates frozen before custody.

    The artifact contains no target values.  Its semantic hash is the
    ``external_snapshot_hash`` used by custody, prediction, registration, and
    evaluation.  This prevents a caller from substituting an opaque hash for
    the actual public prediction inputs.
    """

    schema_version: Literal["6.3-external-forecast-input"] = (
        "6.3-external-forecast-input"
    )
    qualification_id: Identifier
    task_id: Identifier
    contract_hash: Sha256
    local_context_hash: Sha256
    processed_snapshot_hash: Sha256
    target_ids: Annotated[list[Identifier], Field(min_length=1)]
    target_order_hash: Sha256
    forecast_times: Annotated[list[FiniteNumber], Field(min_length=1)]
    input_semantics: Literal[
        "future_time_coordinates_without_private_targets"
    ] = "future_time_coordinates_without_private_targets"
    private_target_values_included: Literal[False] = False
    frozen_at: datetime
    input_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "ExternalForecastInputV63":
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("external forecast target IDs must be unique")
        if len(self.forecast_times) != len(self.target_ids):
            raise ValueError(
                "external forecast times and target IDs have different lengths"
            )
        if any(
            right <= left
            for left, right in zip(
                self.forecast_times,
                self.forecast_times[1:],
            )
        ):
            raise ValueError("external forecast times must be strictly increasing")
        if self.target_order_hash != sha256_value(self.target_ids):
            raise ValueError("external forecast target order hash differs")
        _assert_aware(self.frozen_at, "frozen_at")
        if self.input_hash and self.input_hash != self.content_hash():
            raise ValueError("external forecast input hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "input_hash")

    def assert_sealed(self) -> None:
        if not self.input_hash or self.input_hash != self.content_hash():
            raise ValueError("external forecast input is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExternalForecastInputV63":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"input_hash"})
        payload["input_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalPredictionVectorV63(StrictModel):
    """Typed target-ordered predictions without private target values."""

    schema_version: Literal["6.3-external-prediction-vector"] = (
        "6.3-external-prediction-vector"
    )
    qualification_id: Identifier
    local_context_hash: Sha256
    selected_model_identity_hash: Sha256
    external_snapshot_hash: Sha256
    target_ids: Annotated[list[Identifier], Field(min_length=1)]
    target_order_hash: Sha256
    predictions: Annotated[list[FinitePositive], Field(min_length=1)]
    prediction_values_hash: Sha256
    vector_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_vector(self) -> "ExternalPredictionVectorV63":
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("external prediction target IDs must be unique")
        if len(self.predictions) != len(self.target_ids):
            raise ValueError(
                "external predictions and target IDs have different lengths"
            )
        if self.target_order_hash != sha256_value(self.target_ids):
            raise ValueError("external prediction target order hash differs")
        if self.prediction_values_hash != sha256_value(self.predictions):
            raise ValueError("external prediction values hash differs")
        if self.vector_hash and self.vector_hash != self.content_hash():
            raise ValueError("external prediction vector hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "vector_hash")

    def assert_sealed(self) -> None:
        if not self.vector_hash or self.vector_hash != self.content_hash():
            raise ValueError("external prediction vector is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExternalPredictionVectorV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"vector_hash"})
        payload["vector_hash"] = draft.content_hash()
        return cls(**payload)


class CurrentModelPredictionBindingV63(StrictModel):
    """Harness-authenticated output of the current-model forecast generator."""

    schema_version: Literal["6.3-current-model-prediction-binding"] = (
        "6.3-current-model-prediction-binding"
    )
    qualification_id: Identifier
    contract_hash: Sha256
    workspace_spec_hash: Sha256
    local_context_hash: Sha256
    scientific_bundle_hash: Sha256
    processed_snapshot_hash: Sha256
    executable_candidate_receipt_hash: Sha256
    selected_model_id: Identifier
    selected_model_identity_hash: Sha256
    external_snapshot_hash: Sha256
    holdout_observation_count: Annotated[int, Field(ge=1)]
    target_order_hash: Sha256
    prediction_output_schema_hash: Sha256 = _PREDICTION_OUTPUT_SCHEMA_HASH
    prediction_artifact_hash: Sha256
    prediction_vector_hash: Sha256
    generator_execution_receipt_hash: Sha256
    generator_adapter_id: Literal[
        "current_v62_selected_positive_series_forecast_v63"
    ] = "current_v62_selected_positive_series_forecast_v63"
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    binding_hash: Sha256 | None = None
    private_holdout_targets_accessed: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_binding(self) -> "CurrentModelPredictionBindingV63":
        if (
            self.prediction_output_schema_hash
            != _PREDICTION_OUTPUT_SCHEMA_HASH
        ):
            raise ValueError("prediction binding output schema hash differs")
        if self.authority_auth_tag and not self.binding_hash:
            raise ValueError(
                "authenticated prediction binding requires binding hash"
            )
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("current-model prediction binding hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self,
            "authority_auth_tag",
            "binding_hash",
        )

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.binding_hash
            or self.binding_hash != self.content_hash()
        ):
            raise ValueError(
                "current-model prediction binding is not sealed"
            )


def freeze_predictive_external_qualification_contract_v63(
    *,
    workspace: StageWorkspaceV50,
    qualification_id: str,
    task_id: str,
    maximum_metric_value: float,
    minimum_external_observation_count: int,
    coordinator_host_id: str,
    generator_host_id: str,
    custody_key_id: str,
    registry_key_id: str,
    evaluator_key_id: str,
    promotion_key_id: str,
    trusted_public_keys: Mapping[str, bytes],
    frozen_at: datetime | None = None,
) -> PredictiveExternalQualificationContractV63:
    context = derive_predictive_local_context_v63(workspace)
    authority_key_ids = {
        "custody": custody_key_id,
        "registry": registry_key_id,
        "evaluator": evaluator_key_id,
        "promotion": promotion_key_id,
    }
    try:
        authority_fingerprints = {
            role: external_qualification_key_fingerprint_v63(
                trusted_public_keys[key_id]
            )
            for role, key_id in authority_key_ids.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalQualificationError(
            "V6.3 protocol cannot pin the requested authority keys"
        ) from exc
    contract = PredictiveExternalQualificationContractV63.seal(
        qualification_id=qualification_id,
        task_id=task_id,
        local_context_hash=context.context_hash,
        workspace_spec_hash=context.workspace_spec_hash,
        v62_report_hash=context.v62_report_hash,
        s4_gate_hash=context.s4_gate_hash,
        s6_gate_hash=context.s6_gate_hash,
        scientific_bundle_hash=context.scientific_bundle_hash,
        processed_snapshot_hash=context.processed_snapshot_hash,
        executable_candidate_receipt_hash=(
            context.executable_candidate_receipt_hash
        ),
        selected_model_id=context.selected_model_id,
        selected_model_identity_hash=context.selected_model_identity_hash,
        maximum_metric_value=maximum_metric_value,
        minimum_external_observation_count=minimum_external_observation_count,
        trusted_authority_key_ids=authority_key_ids,
        trusted_authority_key_fingerprints=authority_fingerprints,
        trusted_authority_set_hash=sha256_value(
            {
                "key_ids": authority_key_ids,
                "fingerprints": authority_fingerprints,
            }
        ),
        coordinator_host_id=coordinator_host_id,
        generator_host_id=generator_host_id,
        frozen_at=frozen_at or _utc_now(),
    )
    _commit_unique_qualification_artifact(
        workspace=workspace,
        kind="predictive_external_qualification_contract_v63",
        model=contract,
        model_type=PredictiveExternalQualificationContractV63,
    )
    return contract


class ExternalEvidenceCustodyV63(StrictModel):
    """Signed custody plus measurement-review statement."""

    schema_version: Literal["6.3-external-evidence-custody"] = (
        "6.3-external-evidence-custody"
    )
    qualification_id: Identifier
    contract_hash: Sha256
    local_context_hash: Sha256
    v62_report_hash: Sha256
    external_snapshot_hash: Sha256
    holdout_commitment_hash: Sha256
    normalization_scale_commitment_hash: Sha256
    target_order_hash: Sha256
    holdout_observation_count: Annotated[int, Field(ge=1)]
    fixture_only: bool
    measurement_protocol_hash: Sha256
    measurement_review_hash: Sha256
    external_environment_hash: Sha256
    strict_unseen_verified: bool
    independent_measurement_review_passed: bool
    external_environment_verified: bool
    holdout_frozen_before_prediction: bool
    private_values_disclosed: bool = False
    custodian_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    attested_at: datetime
    custody_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    custody_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_custody(self) -> "ExternalEvidenceCustodyV63":
        if self.coordinator_host_id == self.generator_host_id:
            raise ValueError("coordinator and generator hosts must differ")
        if self.custodian_host_id in {
            self.coordinator_host_id,
            self.generator_host_id,
        }:
            raise ValueError(
                "custodian host must differ from coordinator/generator"
            )
        _assert_aware(self.attested_at, "attested_at")
        if self.custody_hash and (
            not self.signature_base64
            or self.custody_hash != self.content_hash()
        ):
            raise ValueError("V6.3 custody signature envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return _unsigned_bytes(self, "custody_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "custody_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.custody_hash
            or self.custody_hash != self.content_hash()
        ):
            raise ValueError("V6.3 custody evidence is not sealed")


def sign_external_evidence_custody_v63(
    *,
    private_key_pem: bytes,
    **data: object,
) -> ExternalEvidenceCustodyV63:
    data.setdefault("attested_at", _utc_now())
    return _sign_model(
        model_type=ExternalEvidenceCustodyV63,
        data=dict(data),
        private_key_pem=private_key_pem,
        hash_field="custody_hash",
    )  # type: ignore[return-value]


class ExternalPredictionRegistrationV63(StrictModel):
    """Append-only external registry statement, made before holdout access."""

    schema_version: Literal["6.3-external-prediction-registration"] = (
        "6.3-external-prediction-registration"
    )
    qualification_id: Identifier
    task_id: Identifier
    contract_hash: Sha256
    local_context_hash: Sha256
    custody_hash: Sha256
    current_model_prediction_binding_hash: Sha256
    generator_execution_receipt_hash: Sha256
    s4_gate_hash: Sha256
    training_snapshot_hash: Sha256
    candidate_hash: Sha256
    prediction_artifact_hash: Sha256
    external_snapshot_hash: Sha256
    holdout_commitment_hash: Sha256
    normalization_scale_commitment_hash: Sha256
    target_order_hash: Sha256
    holdout_observation_count: Annotated[int, Field(ge=1)]
    append_only_registry_verified: Literal[True] = True
    registered_before_private_holdout_access: Literal[True] = True
    private_values_disclosed: Literal[False] = False
    registry_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    registered_at: datetime
    registry_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    registration_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_registration(self) -> "ExternalPredictionRegistrationV63":
        if self.coordinator_host_id == self.generator_host_id:
            raise ValueError("coordinator and generator hosts must differ")
        if self.registry_host_id in {
            self.coordinator_host_id,
            self.generator_host_id,
        }:
            raise ValueError(
                "registry host must differ from coordinator/generator"
            )
        _assert_aware(self.registered_at, "registered_at")
        if self.registration_hash and (
            not self.signature_base64
            or self.registration_hash != self.content_hash()
        ):
            raise ValueError("V6.3 registration signature envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return _unsigned_bytes(self, "registration_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "registration_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.registration_hash
            or self.registration_hash != self.content_hash()
        ):
            raise ValueError("V6.3 prediction registration is not sealed")


def sign_external_prediction_registration_v63(
    *,
    private_key_pem: bytes,
    **data: object,
) -> ExternalPredictionRegistrationV63:
    data.setdefault("registered_at", _utc_now())
    return _sign_model(
        model_type=ExternalPredictionRegistrationV63,
        data=dict(data),
        private_key_pem=private_key_pem,
        hash_field="registration_hash",
    )  # type: ignore[return-value]


def _assert_contract_current(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
) -> PredictiveLocalContextV63:
    contract.assert_sealed()
    try:
        committed_contracts = [
            item
            for _, item in workspace._artifacts_of_kind(
                "predictive_external_qualification_contract_v63",
                PredictiveExternalQualificationContractV63,
            )
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalQualificationError(
            "V6.3 frozen-contract ledger could not be replayed"
        ) from exc
    if contract not in committed_contracts:
        raise ExternalQualificationError(
            "V6.3 qualification contract was not frozen in the workspace"
        )
    current = derive_predictive_local_context_v63(workspace)
    expected = {
        "local_context": (
            contract.local_context_hash,
            current.context_hash,
        ),
        "workspace": (
            contract.workspace_spec_hash,
            current.workspace_spec_hash,
        ),
        "v62_report": (contract.v62_report_hash, current.v62_report_hash),
        "s4_gate": (contract.s4_gate_hash, current.s4_gate_hash),
        "s6_gate": (contract.s6_gate_hash, current.s6_gate_hash),
    }
    if any(left != right for left, right in expected.values()):
        raise ExternalQualificationError(
            "V6.3 qualification contract is stale for current closure"
        )
    return current


def commit_external_forecast_input_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    target_ids: list[str],
    forecast_times: list[float],
    frozen_at: datetime | None = None,
) -> ExternalForecastInputV63:
    """Freeze the exact public forecast coordinates before custody.

    No target values are accepted by this API.  Repeating the exact call is
    idempotent; a different input for the same qualification is rejected.
    """

    _assert_contract_current(workspace=workspace, contract=contract)
    effective_frozen_at = frozen_at or _utc_now()
    if effective_frozen_at < contract.frozen_at:
        raise ExternalQualificationError(
            "external forecast input predates the qualification contract"
        )
    if len(target_ids) < contract.minimum_external_observation_count:
        raise ExternalQualificationError(
            "external forecast input has too few target coordinates"
        )
    forecast_input = ExternalForecastInputV63.seal(
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        processed_snapshot_hash=contract.processed_snapshot_hash,
        target_ids=target_ids,
        target_order_hash=sha256_value(target_ids),
        forecast_times=forecast_times,
        frozen_at=effective_frozen_at,
    )
    _commit_unique_qualification_artifact(
        workspace=workspace,
        kind=_FORECAST_INPUT_KIND,
        model=forecast_input,
        model_type=ExternalForecastInputV63,
    )
    return forecast_input


def _current_external_forecast_input(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
) -> tuple[ExternalForecastInputV63, str]:
    try:
        matches = [
            (reference, item)
            for reference, item in workspace._artifacts_of_kind(
                _FORECAST_INPUT_KIND,
                ExternalForecastInputV63,
            )
            if item.qualification_id == contract.qualification_id
            or item.task_id == contract.task_id
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationError(
            "external forecast-input ledger could not be replayed"
        ) from exc
    if len(matches) != 1:
        raise ExternalQualificationError(
            "one exact external forecast input is required"
        )
    reference, forecast_input = matches[0]
    try:
        forecast_input.assert_sealed()
    except ValueError as exc:
        raise ExternalQualificationError(
            "external forecast input envelope rejected"
        ) from exc
    if (
        forecast_input.qualification_id != contract.qualification_id
        or forecast_input.task_id != contract.task_id
        or forecast_input.contract_hash != contract.contract_hash
        or forecast_input.local_context_hash != contract.local_context_hash
        or forecast_input.processed_snapshot_hash
        != contract.processed_snapshot_hash
        or forecast_input.frozen_at < contract.frozen_at
    ):
        raise ExternalQualificationError(
            "external forecast input contract binding rejected"
        )
    exact_hash = _exact_committed_artifact_hash(
        workspace=workspace,
        kind=_FORECAST_INPUT_KIND,
        model=forecast_input,
        model_type=ExternalForecastInputV63,
    )
    if exact_hash != reference.sha256:
        raise ExternalQualificationError(
            "external forecast input artifact differs"
        )
    return forecast_input, exact_hash


def _external_forecast_input_reason_codes(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: "ExternalEvidenceCustodyV63",
) -> list[str]:
    try:
        forecast_input, _ = _current_external_forecast_input(
            workspace=workspace,
            contract=contract,
        )
    except ExternalQualificationError:
        return ["external_forecast_input_invalid"]
    reasons: list[str] = []
    if custody.external_snapshot_hash != forecast_input.input_hash:
        reasons.append("external_snapshot_not_typed_forecast_input")
    if custody.target_order_hash != forecast_input.target_order_hash:
        reasons.append("forecast_target_order_differs")
    if custody.holdout_observation_count != len(forecast_input.target_ids):
        reasons.append("forecast_target_count_differs")
    if custody.attested_at < forecast_input.frozen_at:
        reasons.append("custody_predates_forecast_input")
    return sorted(set(reasons))


def _assert_trusted_authority_set(
    *,
    contract: PredictiveExternalQualificationContractV63,
    trusted_public_keys: Mapping[str, bytes],
) -> None:
    runtime_fingerprints: dict[str, str] = {}
    try:
        for role, key_id in contract.trusted_authority_key_ids.items():
            runtime_fingerprints[role] = (
                external_qualification_key_fingerprint_v63(
                    trusted_public_keys[key_id]
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalQualificationError(
            "runtime authority set differs from frozen protocol"
        ) from exc
    if runtime_fingerprints != contract.trusted_authority_key_fingerprints:
        raise ExternalQualificationError(
            "runtime authority fingerprints differ from frozen protocol"
        )
    runtime_set_hash = sha256_value(
        {
            "key_ids": contract.trusted_authority_key_ids,
            "fingerprints": runtime_fingerprints,
        }
    )
    if runtime_set_hash != contract.trusted_authority_set_hash:
        raise ExternalQualificationError(
            "runtime authority set hash differs from frozen protocol"
        )


def _require_signed(
    *,
    model: StrictModel,
    key_id: str,
    signature_base64: str | None,
    trusted_public_keys: Mapping[str, bytes],
    hash_field: str,
    label: str,
) -> None:
    try:
        getattr(model, "assert_sealed")()
    except (AttributeError, ValueError) as exc:
        raise ExternalQualificationError(
            f"{label} envelope rejected"
        ) from exc
    if not _verify_signature(
        model=model,
        key_id=key_id,
        signature_base64=signature_base64,
        trusted_public_keys=trusted_public_keys,
        hash_field=hash_field,
    ):
        raise ExternalQualificationError(f"{label} signature rejected")


def _custody_reason_codes(
    *,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
) -> list[str]:
    reasons: list[str] = []
    if (
        custody.qualification_id != contract.qualification_id
        or custody.contract_hash != contract.contract_hash
        or custody.local_context_hash != contract.local_context_hash
        or custody.v62_report_hash != contract.v62_report_hash
        or custody.coordinator_host_id != contract.coordinator_host_id
        or custody.generator_host_id != contract.generator_host_id
        or custody.holdout_observation_count
        < contract.minimum_external_observation_count
    ):
        reasons.append("custody_contract_binding_invalid")
    for reason, passed in {
        "external_fixture_only": not custody.fixture_only,
        "strict_unseen_not_verified": custody.strict_unseen_verified,
        "independent_measurement_review_failed": (
            custody.independent_measurement_review_passed
        ),
        "external_environment_not_verified": (
            custody.external_environment_verified
        ),
        "holdout_not_frozen_before_prediction": (
            custody.holdout_frozen_before_prediction
        ),
        "private_values_disclosed": not custody.private_values_disclosed,
    }.items():
        if not passed:
            reasons.append(reason)
    return sorted(set(reasons))


def _assert_custody_bound(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
) -> None:
    reasons = sorted(
        set(
            _custody_reason_codes(
                contract=contract,
                custody=custody,
            )
            + _external_forecast_input_reason_codes(
                workspace=workspace,
                contract=contract,
                custody=custody,
            )
        )
    )
    if reasons:
        raise ExternalQualificationError(
            "external custody rejected: " + ",".join(reasons)
        )


class ExternalCustodyAdmissionV63(StrictModel):
    schema_version: Literal["6.3-external-custody-admission"] = (
        "6.3-external-custody-admission"
    )
    qualification_id: Identifier
    contract_hash: Sha256
    custody_hash: Sha256
    custody_artifact_hash: Sha256
    status: Literal["VERIFIED", "REJECTED"]
    reason_codes: list[Identifier]
    signature_verified: Literal[True] = True
    prediction_seal_issued: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    admission_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_admission(self) -> "ExternalCustodyAdmissionV63":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("custody admission reasons differ")
        if (self.status == "VERIFIED") != (not self.reason_codes):
            raise ValueError("custody admission status differs from reasons")
        if self.admission_hash and self.admission_hash != self.content_hash():
            raise ValueError("custody admission hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "admission_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExternalCustodyAdmissionV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"admission_hash"})
        payload["admission_hash"] = draft.content_hash()
        return cls(**payload)


def admit_external_evidence_custody_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    trusted_public_keys: Mapping[str, bytes],
) -> ExternalCustodyAdmissionV63:
    """Persist a signed custody result, including honest negative evidence."""

    _assert_contract_current(workspace=workspace, contract=contract)
    _assert_trusted_authority_set(
        contract=contract,
        trusted_public_keys=trusted_public_keys,
    )
    if (
        custody.custody_key_id
        != contract.trusted_authority_key_ids["custody"]
    ):
        raise ExternalQualificationError(
            "custody authority differs from frozen protocol"
        )
    _require_signed(
        model=custody,
        key_id=custody.custody_key_id,
        signature_base64=custody.signature_base64,
        trusted_public_keys=trusted_public_keys,
        hash_field="custody_hash",
        label="custody",
    )
    custody_artifact_hash = _commit_unique_qualification_artifact(
        workspace=workspace,
        kind="external_evidence_custody_v63",
        model=custody,
        model_type=ExternalEvidenceCustodyV63,
    )
    reasons = sorted(
        set(
            _custody_reason_codes(
                contract=contract,
                custody=custody,
            )
            + _external_forecast_input_reason_codes(
                workspace=workspace,
                contract=contract,
                custody=custody,
            )
        )
    )
    admission = ExternalCustodyAdmissionV63.seal(
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        custody_hash=custody.custody_hash,
        custody_artifact_hash=custody_artifact_hash,
        status="REJECTED" if reasons else "VERIFIED",
        reason_codes=reasons,
    )
    _commit_unique_qualification_artifact(
        workspace=workspace,
        kind="external_custody_admission_v63",
        model=admission,
        model_type=ExternalCustodyAdmissionV63,
    )
    return admission


def _assert_current_model_prediction_binding(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    binding: CurrentModelPredictionBindingV63,
) -> RoleExecutionReceiptV50:
    forecast_input, _ = _current_external_forecast_input(
        workspace=workspace,
        contract=contract,
    )
    try:
        binding.assert_sealed()
    except ValueError as exc:
        raise ExternalQualificationError(
            "current-model prediction binding envelope rejected"
        ) from exc
    if (
        binding.authority_key_id != workspace.authority_key_id
        or not workspace._verify_mac(
            _PREDICTION_BINDING_KIND,
            binding.unsigned_hash(),
            binding.authority_auth_tag,
        )
    ):
        raise ExternalQualificationError(
            "current-model prediction binding authority rejected"
        )
    expected = {
        "qualification": (
            binding.qualification_id,
            contract.qualification_id,
        ),
        "contract": (binding.contract_hash, contract.contract_hash),
        "workspace": (
            binding.workspace_spec_hash,
            contract.workspace_spec_hash,
        ),
        "context": (
            binding.local_context_hash,
            contract.local_context_hash,
        ),
        "bundle": (
            binding.scientific_bundle_hash,
            contract.scientific_bundle_hash,
        ),
        "snapshot": (
            binding.processed_snapshot_hash,
            contract.processed_snapshot_hash,
        ),
        "execution_receipt": (
            binding.executable_candidate_receipt_hash,
            contract.executable_candidate_receipt_hash,
        ),
        "selected_model": (
            binding.selected_model_id,
            contract.selected_model_id,
        ),
        "selected_model_identity": (
            binding.selected_model_identity_hash,
            contract.selected_model_identity_hash,
        ),
        "external_snapshot": (
            binding.external_snapshot_hash,
            custody.external_snapshot_hash,
        ),
        "holdout_count": (
            binding.holdout_observation_count,
            custody.holdout_observation_count,
        ),
        "target_order": (
            binding.target_order_hash,
            custody.target_order_hash,
        ),
        "output_schema": (
            binding.prediction_output_schema_hash,
            contract.prediction_output_schema_hash,
        ),
    }
    if any(left != right for left, right in expected.values()):
        raise ExternalQualificationError(
            "prediction artifact is not bound to the current selected model"
        )
    try:
        committed_bindings = workspace._artifacts_of_kind(
            _PREDICTION_BINDING_KIND,
            CurrentModelPredictionBindingV63,
        )
        if not any(item == binding for _, item in committed_bindings):
            raise ExternalQualificationError(
                "current-model prediction binding is not committed"
            )
        role_receipts = [
            item
            for _, item in workspace._artifacts_of_kind(
                "role_execution_receipt_v50",
                RoleExecutionReceiptV50,
            )
            if item.receipt_hash
            == binding.generator_execution_receipt_hash
        ]
        if len(role_receipts) != 1:
            raise ExternalQualificationError(
                "prediction generator execution receipt is absent"
            )
        role_receipt = role_receipts[0]
        prediction_vector = ExternalPredictionVectorV63.model_validate(
            workspace._artifact_payload_by_hash(
                binding.prediction_artifact_hash
            )
        )
        prediction_vector.assert_sealed()
        prediction_artifact_hash = _exact_committed_artifact_hash(
            workspace=workspace,
            kind=_PREDICTION_VECTOR_KIND,
            model=prediction_vector,
            model_type=ExternalPredictionVectorV63,
        )
        if (
            prediction_artifact_hash != binding.prediction_artifact_hash
            or prediction_vector.vector_hash
            != binding.prediction_vector_hash
            or prediction_vector.qualification_id
            != contract.qualification_id
            or prediction_vector.local_context_hash
            != contract.local_context_hash
            or prediction_vector.selected_model_identity_hash
            != contract.selected_model_identity_hash
            or prediction_vector.external_snapshot_hash
            != custody.external_snapshot_hash
            or prediction_vector.external_snapshot_hash
            != forecast_input.input_hash
            or prediction_vector.target_ids != forecast_input.target_ids
            or prediction_vector.target_order_hash
            != custody.target_order_hash
            or len(prediction_vector.predictions)
            != custody.holdout_observation_count
        ):
            raise ExternalQualificationError(
                "typed prediction vector does not bind the held-out targets"
            )
        expected_input_authority = sha256_value(
            {
                "contract_hash": contract.contract_hash,
                "local_context_hash": contract.local_context_hash,
                "scientific_bundle_hash": contract.scientific_bundle_hash,
                "processed_snapshot_hash": contract.processed_snapshot_hash,
                "executable_candidate_receipt_hash": (
                    contract.executable_candidate_receipt_hash
                ),
                "selected_model_identity_hash": (
                    contract.selected_model_identity_hash
                ),
                "external_snapshot_hash": custody.external_snapshot_hash,
                "holdout_observation_count": (
                    custody.holdout_observation_count
                ),
            }
        )
        if (
            not workspace.verify_role_execution(role_receipt)
            or role_receipt.stage != "S4"
            or role_receipt.role != "modeler"
            or role_receipt.subject_id != contract.task_id
            or role_receipt.issued_at < custody.attested_at
            or role_receipt.input_authority_hash
            != expected_input_authority
            or role_receipt.output_schema_hash
            != contract.prediction_output_schema_hash
            or role_receipt.output_artifact_hash
            != binding.prediction_artifact_hash
            or binding.prediction_artifact_hash
            not in workspace._committed_artifact_hashes()
        ):
            raise ExternalQualificationError(
                "prediction generator receipt does not bind current inputs"
            )
    except ExternalQualificationError:
        raise
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationError(
            "current-model prediction evidence could not be replayed"
        ) from exc
    return role_receipt


def issue_current_model_prediction_binding_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    prediction_vector: ExternalPredictionVectorV63,
    generator_execution_receipt: RoleExecutionReceiptV50,
) -> CurrentModelPredictionBindingV63:
    """Authenticate one exact generator output without exposing a sign oracle.

    Every authoritative field is derived from the current contract, verified
    custody, typed forecast input, typed prediction vector, and an existing
    harness-issued role receipt.  Callers cannot supply model identities,
    artifact hashes, authority tags, or claim flags.
    """

    _assert_contract_current(workspace=workspace, contract=contract)
    _assert_custody_bound(
        workspace=workspace,
        contract=contract,
        custody=custody,
    )
    _verified_custody_admission(
        workspace=workspace,
        contract=contract,
        custody=custody,
    )
    forecast_input, _ = _current_external_forecast_input(
        workspace=workspace,
        contract=contract,
    )
    try:
        prediction_vector.assert_sealed()
    except ValueError as exc:
        raise ExternalQualificationError(
            "typed prediction vector envelope rejected"
        ) from exc
    prediction_artifact_hash = _exact_committed_artifact_hash(
        workspace=workspace,
        kind=_PREDICTION_VECTOR_KIND,
        model=prediction_vector,
        model_type=ExternalPredictionVectorV63,
    )
    if (
        prediction_vector.qualification_id != contract.qualification_id
        or prediction_vector.local_context_hash != contract.local_context_hash
        or prediction_vector.selected_model_identity_hash
        != contract.selected_model_identity_hash
        or prediction_vector.external_snapshot_hash
        != forecast_input.input_hash
        or prediction_vector.target_ids != forecast_input.target_ids
        or prediction_vector.target_order_hash
        != forecast_input.target_order_hash
        or len(prediction_vector.predictions)
        != len(forecast_input.forecast_times)
    ):
        raise ExternalQualificationError(
            "typed prediction vector differs from frozen forecast input"
        )
    expected_input_authority = sha256_value(
        {
            "contract_hash": contract.contract_hash,
            "local_context_hash": contract.local_context_hash,
            "scientific_bundle_hash": contract.scientific_bundle_hash,
            "processed_snapshot_hash": contract.processed_snapshot_hash,
            "executable_candidate_receipt_hash": (
                contract.executable_candidate_receipt_hash
            ),
            "selected_model_identity_hash": (
                contract.selected_model_identity_hash
            ),
            "external_snapshot_hash": forecast_input.input_hash,
            "holdout_observation_count": len(forecast_input.target_ids),
        }
    )
    try:
        RoleExecutionReceiptV50.model_validate(
            generator_execution_receipt.model_dump(mode="json")
        )
        exact_role_receipts = [
            item
            for _, item in workspace._artifacts_of_kind(
                "role_execution_receipt_v50",
                RoleExecutionReceiptV50,
            )
            if item.receipt_hash == generator_execution_receipt.receipt_hash
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationError(
            "prediction generator execution receipt could not be replayed"
        ) from exc
    if (
        not generator_execution_receipt.receipt_hash
        or len(exact_role_receipts) != 1
        or exact_role_receipts[0] != generator_execution_receipt
        or not workspace.verify_role_execution(generator_execution_receipt)
        or generator_execution_receipt.stage != "S4"
        or generator_execution_receipt.role != "modeler"
        or generator_execution_receipt.subject_id != contract.task_id
        or generator_execution_receipt.issued_at < custody.attested_at
        or generator_execution_receipt.input_authority_hash
        != expected_input_authority
        or generator_execution_receipt.output_schema_hash
        != contract.prediction_output_schema_hash
        or generator_execution_receipt.output_artifact_hash
        != prediction_artifact_hash
    ):
        raise ExternalQualificationError(
            "prediction generator execution receipt binding rejected"
        )
    unsigned = CurrentModelPredictionBindingV63(
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        workspace_spec_hash=contract.workspace_spec_hash,
        local_context_hash=contract.local_context_hash,
        scientific_bundle_hash=contract.scientific_bundle_hash,
        processed_snapshot_hash=contract.processed_snapshot_hash,
        executable_candidate_receipt_hash=(
            contract.executable_candidate_receipt_hash
        ),
        selected_model_id=contract.selected_model_id,
        selected_model_identity_hash=contract.selected_model_identity_hash,
        external_snapshot_hash=forecast_input.input_hash,
        holdout_observation_count=len(forecast_input.target_ids),
        target_order_hash=forecast_input.target_order_hash,
        prediction_output_schema_hash=contract.prediction_output_schema_hash,
        prediction_artifact_hash=prediction_artifact_hash,
        prediction_vector_hash=prediction_vector.vector_hash,
        generator_execution_receipt_hash=(
            generator_execution_receipt.receipt_hash
        ),
        authority_key_id=workspace.authority_key_id,
    )
    payload = unsigned.model_dump(mode="json")
    payload["authority_auth_tag"] = workspace._mac(
        _PREDICTION_BINDING_KIND,
        unsigned.unsigned_hash(),
    )
    payload["binding_hash"] = sha256_value(
        {
            key: value
            for key, value in payload.items()
            if key != "binding_hash"
        }
    )
    binding = CurrentModelPredictionBindingV63.model_validate(payload)
    _commit_unique_qualification_artifact(
        workspace=workspace,
        kind=_PREDICTION_BINDING_KIND,
        model=binding,
        model_type=CurrentModelPredictionBindingV63,
    )
    _assert_current_model_prediction_binding(
        workspace=workspace,
        contract=contract,
        custody=custody,
        binding=binding,
    )
    return binding


def register_external_prediction_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    prediction_binding: CurrentModelPredictionBindingV63,
    registration: ExternalPredictionRegistrationV63,
    trusted_public_keys: Mapping[str, bytes],
) -> PredictionSealV50:
    """Verify the external registry chain and issue the existing S4 seal."""

    custody_admission = admit_external_evidence_custody_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        trusted_public_keys=trusted_public_keys,
    )
    if custody_admission.status != "VERIFIED":
        raise ExternalQualificationError(
            "external custody rejected: "
            + ",".join(custody_admission.reason_codes)
        )
    if (
        registration.registry_key_id
        != contract.trusted_authority_key_ids["registry"]
    ):
        raise ExternalQualificationError(
            "registry authority differs from frozen protocol"
        )
    generator_receipt = _assert_current_model_prediction_binding(
        workspace=workspace,
        contract=contract,
        custody=custody,
        binding=prediction_binding,
    )
    _require_signed(
        model=registration,
        key_id=registration.registry_key_id,
        signature_base64=registration.signature_base64,
        trusted_public_keys=trusted_public_keys,
        hash_field="registration_hash",
        label="prediction registration",
    )
    if custody.custody_key_id == registration.registry_key_id:
        raise ExternalQualificationError(
            "custody and registry key IDs must be distinct"
        )
    if custody.custodian_host_id == registration.registry_host_id:
        raise ExternalQualificationError(
            "custody and registry hosts must be distinct"
        )
    if not (
        contract.frozen_at
        <= custody.attested_at
        <= generator_receipt.issued_at
        <= registration.registered_at
    ):
        raise ExternalQualificationError(
            "external registration chronology rejected"
        )
    if (
        registration.qualification_id != contract.qualification_id
        or registration.task_id != contract.task_id
        or registration.contract_hash != contract.contract_hash
        or registration.local_context_hash != contract.local_context_hash
        or registration.custody_hash != custody.custody_hash
        or registration.current_model_prediction_binding_hash
        != prediction_binding.binding_hash
        or registration.generator_execution_receipt_hash
        != prediction_binding.generator_execution_receipt_hash
        or registration.s4_gate_hash != contract.s4_gate_hash
        or registration.training_snapshot_hash
        != contract.processed_snapshot_hash
        or registration.candidate_hash
        != contract.selected_model_identity_hash
        or registration.prediction_artifact_hash
        != prediction_binding.prediction_artifact_hash
        or registration.external_snapshot_hash
        != custody.external_snapshot_hash
        or registration.holdout_commitment_hash
        != custody.holdout_commitment_hash
        or registration.normalization_scale_commitment_hash
        != custody.normalization_scale_commitment_hash
        or registration.target_order_hash != custody.target_order_hash
        or registration.holdout_observation_count
        != custody.holdout_observation_count
        or registration.coordinator_host_id != contract.coordinator_host_id
        or registration.generator_host_id != contract.generator_host_id
    ):
        raise ExternalQualificationError(
            "external prediction registration binding rejected"
        )
    _commit_unique_qualification_artifact(
        workspace=workspace,
        kind="external_prediction_registration_v63",
        model=registration,
        model_type=ExternalPredictionRegistrationV63,
    )
    return workspace.issue_prediction_seal(
        task_id=contract.task_id,
        training_snapshot_hash=registration.training_snapshot_hash,
        candidate_hash=registration.candidate_hash,
        prediction_artifact_hash=registration.prediction_artifact_hash,
        external_registration_hash=registration.registration_hash,
        external_snapshot_hash=registration.external_snapshot_hash,
        holdout_commitment_hash=registration.holdout_commitment_hash,
    )


class ExternalEvaluationReservationV63(StrictModel):
    """Code-owned authority to dispatch exactly one external evaluation."""

    schema_version: Literal["6.3-external-evaluation-reservation"] = (
        "6.3-external-evaluation-reservation"
    )
    qualification_id: Identifier
    task_id: Identifier
    contract_hash: Sha256
    local_context_hash: Sha256
    custody_hash: Sha256
    custody_admission_hash: Sha256
    current_model_prediction_binding_hash: Sha256
    registration_hash: Sha256
    prediction_seal_hash: Sha256
    prediction_artifact_hash: Sha256
    reserved_evaluation_sequence: Literal[1] = 1
    evaluator_key_id: Identifier
    evaluator_host_id: Identifier
    custodian_host_id: Identifier
    registry_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    reserved_at: datetime
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    reservation_hash: Sha256 | None = None
    private_holdout_targets_accessed: Literal[False] = False
    real_world_action_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_reservation(self) -> "ExternalEvaluationReservationV63":
        hosts = {
            self.coordinator_host_id,
            self.generator_host_id,
            self.custodian_host_id,
            self.registry_host_id,
            self.evaluator_host_id,
        }
        if len(hosts) != 5:
            raise ValueError(
                "reservation roles must use five distinct hosts"
            )
        _assert_aware(self.reserved_at, "reserved_at")
        if self.authority_auth_tag and not self.reservation_hash:
            raise ValueError(
                "authenticated reservation requires reservation hash"
            )
        if (
            self.reservation_hash
            and self.reservation_hash != self.content_hash()
        ):
            raise ValueError("external evaluation reservation hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self,
            "authority_auth_tag",
            "reservation_hash",
        )

    def content_hash(self) -> str:
        return _hash_without(self, "reservation_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authority_auth_tag
            or not self.reservation_hash
            or self.reservation_hash != self.content_hash()
        ):
            raise ValueError(
                "external evaluation reservation is not sealed"
            )


def _verified_custody_admission(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
) -> tuple[ExternalCustodyAdmissionV63, str]:
    try:
        matches = [
            (reference, admission)
            for reference, admission in workspace._artifacts_of_kind(
                "external_custody_admission_v63",
                ExternalCustodyAdmissionV63,
            )
            if admission.qualification_id == contract.qualification_id
            and admission.contract_hash == contract.contract_hash
            and admission.custody_hash == custody.custody_hash
        ]
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ExternalQualificationError(
            "external custody-admission ledger could not be replayed"
        ) from exc
    if len(matches) != 1:
        raise ExternalQualificationError(
            "verified external custody admission is absent or duplicated"
        )
    reference, admission = matches[0]
    if (
        not admission.admission_hash
        or admission.admission_hash != admission.content_hash()
        or admission.status != "VERIFIED"
        or admission.reason_codes
        or admission.custody_artifact_hash
        != _exact_committed_artifact_hash(
            workspace=workspace,
            kind="external_evidence_custody_v63",
            model=custody,
            model_type=ExternalEvidenceCustodyV63,
        )
    ):
        raise ExternalQualificationError(
            "external custody admission is not a verified exact envelope"
        )
    exact_hash = _exact_committed_artifact_hash(
        workspace=workspace,
        kind="external_custody_admission_v63",
        model=admission,
        model_type=ExternalCustodyAdmissionV63,
    )
    if exact_hash != reference.sha256:
        raise ExternalQualificationError(
            "external custody-admission artifact differs"
        )
    return admission, exact_hash


def _assert_registered_prediction_chain(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    prediction_binding: CurrentModelPredictionBindingV63,
    registration: ExternalPredictionRegistrationV63,
    prediction_seal: PredictionSealV50,
) -> None:
    _assert_custody_bound(
        workspace=workspace,
        contract=contract,
        custody=custody,
    )
    generator_receipt = _assert_current_model_prediction_binding(
        workspace=workspace,
        contract=contract,
        custody=custody,
        binding=prediction_binding,
    )
    if not (
        contract.frozen_at
        <= custody.attested_at
        <= generator_receipt.issued_at
        <= registration.registered_at
        <= prediction_seal.registered_at
    ):
        raise ExternalQualificationError(
            "external registration chronology rejected"
        )
    seal_valid = (
        workspace.verify_prediction_seal(prediction_seal)
        and prediction_seal.workspace_spec_hash
        == contract.workspace_spec_hash
        and prediction_seal.s4_gate_hash == workspace.current_gate("S4")
        and prediction_seal.s4_gate_hash == contract.s4_gate_hash
        and prediction_seal.task_id == contract.task_id
        and prediction_seal.training_snapshot_hash
        == registration.training_snapshot_hash
        and prediction_seal.candidate_hash == registration.candidate_hash
        and prediction_seal.prediction_artifact_hash
        == registration.prediction_artifact_hash
        and prediction_seal.external_registration_hash
        == registration.registration_hash
        and prediction_seal.external_snapshot_hash
        == custody.external_snapshot_hash
        and prediction_seal.holdout_commitment_hash
        == custody.holdout_commitment_hash
    )
    if not seal_valid:
        raise ExternalQualificationError(
            "current S4 prediction seal binding rejected"
        )
    if (
        registration.qualification_id != contract.qualification_id
        or registration.task_id != contract.task_id
        or registration.contract_hash != contract.contract_hash
        or registration.local_context_hash != contract.local_context_hash
        or registration.custody_hash != custody.custody_hash
        or registration.current_model_prediction_binding_hash
        != prediction_binding.binding_hash
        or registration.generator_execution_receipt_hash
        != prediction_binding.generator_execution_receipt_hash
        or registration.s4_gate_hash != contract.s4_gate_hash
        or registration.training_snapshot_hash
        != contract.processed_snapshot_hash
        or registration.candidate_hash
        != contract.selected_model_identity_hash
        or registration.prediction_artifact_hash
        != prediction_binding.prediction_artifact_hash
        or registration.external_snapshot_hash
        != custody.external_snapshot_hash
        or registration.holdout_commitment_hash
        != custody.holdout_commitment_hash
        or registration.normalization_scale_commitment_hash
        != custody.normalization_scale_commitment_hash
        or registration.target_order_hash != custody.target_order_hash
        or registration.holdout_observation_count
        != custody.holdout_observation_count
        or registration.coordinator_host_id
        != contract.coordinator_host_id
        or registration.generator_host_id != contract.generator_host_id
    ):
        raise ExternalQualificationError(
            "prediction registration chain binding rejected"
        )
    _exact_committed_artifact_hash(
        workspace=workspace,
        kind="external_prediction_registration_v63",
        model=registration,
        model_type=ExternalPredictionRegistrationV63,
    )
    _exact_committed_artifact_hash(
        workspace=workspace,
        kind="prediction_seal_v50",
        model=prediction_seal,
        model_type=PredictionSealV50,
    )


def _assert_evaluation_reservation(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    custody_admission: ExternalCustodyAdmissionV63,
    prediction_binding: CurrentModelPredictionBindingV63,
    registration: ExternalPredictionRegistrationV63,
    prediction_seal: PredictionSealV50,
    reservation: ExternalEvaluationReservationV63,
) -> str:
    try:
        reservation.assert_sealed()
    except (AttributeError, ValueError) as exc:
        raise ExternalQualificationError(
            "external evaluation reservation envelope rejected"
        ) from exc
    if (
        reservation.authority_key_id != workspace.authority_key_id
        or not workspace._verify_mac(
            _RESERVATION_KIND,
            reservation.unsigned_hash(),
            reservation.authority_auth_tag,
        )
    ):
        raise ExternalQualificationError(
            "external evaluation reservation authority rejected"
        )
    if (
        reservation.qualification_id != contract.qualification_id
        or reservation.task_id != contract.task_id
        or reservation.contract_hash != contract.contract_hash
        or reservation.local_context_hash != contract.local_context_hash
        or reservation.custody_hash != custody.custody_hash
        or reservation.custody_admission_hash
        != custody_admission.admission_hash
        or reservation.current_model_prediction_binding_hash
        != prediction_binding.binding_hash
        or reservation.registration_hash != registration.registration_hash
        or reservation.prediction_seal_hash != prediction_seal.seal_hash
        or reservation.prediction_artifact_hash
        != registration.prediction_artifact_hash
        or reservation.evaluator_key_id
        != contract.trusted_authority_key_ids["evaluator"]
        or reservation.custodian_host_id != custody.custodian_host_id
        or reservation.registry_host_id != registration.registry_host_id
        or reservation.coordinator_host_id
        != contract.coordinator_host_id
        or reservation.generator_host_id != contract.generator_host_id
        or reservation.reserved_at < prediction_seal.registered_at
    ):
        raise ExternalQualificationError(
            "external evaluation reservation binding rejected"
        )
    return _exact_committed_artifact_hash(
        workspace=workspace,
        kind=_RESERVATION_KIND,
        model=reservation,
        model_type=ExternalEvaluationReservationV63,
    )


def reserve_external_evaluation_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    prediction_binding: CurrentModelPredictionBindingV63,
    registration: ExternalPredictionRegistrationV63,
    prediction_seal: PredictionSealV50,
    evaluator_key_id: str,
    evaluator_host_id: str,
    reserved_at: datetime | None = None,
) -> ExternalEvaluationReservationV63:
    """Atomically reserve the sole evaluator dispatch for this task."""

    _assert_contract_current(workspace=workspace, contract=contract)
    _assert_registered_prediction_chain(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=prediction_binding,
        registration=registration,
        prediction_seal=prediction_seal,
    )
    custody_admission, _ = _verified_custody_admission(
        workspace=workspace,
        contract=contract,
        custody=custody,
    )
    if evaluator_key_id != contract.trusted_authority_key_ids["evaluator"]:
        raise ExternalQualificationError(
            "reserved evaluator authority differs from frozen protocol"
        )
    if evaluator_host_id in {
        contract.coordinator_host_id,
        contract.generator_host_id,
        custody.custodian_host_id,
        registration.registry_host_id,
    }:
        raise ExternalQualificationError(
            "reserved evaluator host is not independent"
        )
    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    transaction = (
        transaction_factory()
        if callable(transaction_factory)
        else nullcontext()
    )
    with transaction:
        try:
            prior = [
                item
                for _, item in workspace._artifacts_of_kind(
                    _RESERVATION_KIND,
                    ExternalEvaluationReservationV63,
                )
                if item.qualification_id == contract.qualification_id
                or item.task_id == contract.task_id
            ]
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ExternalQualificationError(
                "external evaluation reservation ledger could not be replayed"
            ) from exc
        if prior:
            if len(prior) != 1:
                raise ExternalQualificationError(
                    "external evaluation reservation ledger has duplicates"
                )
            existing = prior[0]
            _assert_evaluation_reservation(
                workspace=workspace,
                contract=contract,
                custody=custody,
                custody_admission=custody_admission,
                prediction_binding=prediction_binding,
                registration=registration,
                prediction_seal=prediction_seal,
                reservation=existing,
            )
            if (
                existing.evaluator_key_id != evaluator_key_id
                or existing.evaluator_host_id != evaluator_host_id
                or (
                    reserved_at is not None
                    and existing.reserved_at != reserved_at
                )
            ):
                raise ExternalQualificationError(
                    "external evaluation reservation is immutable"
                )
            return existing
        effective_reserved_at = reserved_at or _utc_now()
        if effective_reserved_at < prediction_seal.registered_at:
            raise ExternalQualificationError(
                "external evaluation reservation predates prediction seal"
            )
        unsigned = ExternalEvaluationReservationV63(
            qualification_id=contract.qualification_id,
            task_id=contract.task_id,
            contract_hash=contract.contract_hash,
            local_context_hash=contract.local_context_hash,
            custody_hash=custody.custody_hash,
            custody_admission_hash=custody_admission.admission_hash,
            current_model_prediction_binding_hash=prediction_binding.binding_hash,
            registration_hash=registration.registration_hash,
            prediction_seal_hash=prediction_seal.seal_hash,
            prediction_artifact_hash=registration.prediction_artifact_hash,
            evaluator_key_id=evaluator_key_id,
            evaluator_host_id=evaluator_host_id,
            custodian_host_id=custody.custodian_host_id,
            registry_host_id=registration.registry_host_id,
            coordinator_host_id=contract.coordinator_host_id,
            generator_host_id=contract.generator_host_id,
            reserved_at=effective_reserved_at,
            authority_key_id=workspace.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = workspace._mac(
            _RESERVATION_KIND,
            unsigned.unsigned_hash(),
        )
        payload["reservation_hash"] = sha256_value(
            {
                key: value
                for key, value in payload.items()
                if key != "reservation_hash"
            }
        )
        reservation = ExternalEvaluationReservationV63.model_validate(payload)
        workspace.commit_evidence(
            _RESERVATION_KIND,
            reservation.model_dump(mode="json"),
        )
        return reservation


class ExternalAggregateEvaluationV63(StrictModel):
    """One aggregate-only score returned by the independent evaluator."""

    schema_version: Literal["6.3-external-aggregate-evaluation"] = (
        "6.3-external-aggregate-evaluation"
    )
    evaluation_id: Identifier
    qualification_id: Identifier
    contract_hash: Sha256
    local_context_hash: Sha256
    custody_hash: Sha256
    registration_hash: Sha256
    prediction_seal_hash: Sha256
    reservation_hash: Sha256
    prediction_artifact_hash: Sha256
    external_snapshot_hash: Sha256
    holdout_commitment_hash: Sha256
    normalization_scale_commitment_hash: Sha256
    target_order_hash: Sha256
    holdout_observation_count: Annotated[int, Field(ge=1)]
    metric: Literal["normalized_root_mean_squared_error"] = (
        "normalized_root_mean_squared_error"
    )
    squared_error_sum: FiniteNonNegative
    target_squared_value_sum: FinitePositive
    aggregate_metric_value: FiniteNonNegative
    evaluation_sequence: Literal[1] = 1
    aggregate_only: Literal[True] = True
    per_observation_feedback_released: Literal[False] = False
    private_values_disclosed: Literal[False] = False
    evaluator_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    evaluated_at: datetime
    evaluator_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    evaluation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evaluation(self) -> "ExternalAggregateEvaluationV63":
        if self.coordinator_host_id == self.generator_host_id:
            raise ValueError("coordinator and generator hosts must differ")
        if self.evaluator_host_id in {
            self.coordinator_host_id,
            self.generator_host_id,
        }:
            raise ValueError(
                "evaluator host must differ from coordinator/generator"
            )
        _assert_aware(self.evaluated_at, "evaluated_at")
        recomputed = _recomputed_external_metric(self)
        if not math.isclose(
            self.aggregate_metric_value,
            recomputed,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "aggregate metric differs from sufficient statistics"
            )
        if self.normalization_scale_commitment_hash != sha256_value(
            {
                "holdout_observation_count": (
                    self.holdout_observation_count
                ),
                "target_squared_value_sum": (
                    self.target_squared_value_sum
                ),
            }
        ):
            raise ValueError(
                "normalization scale differs from custody commitment"
            )
        if self.evaluation_hash and (
            not self.signature_base64
            or self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.3 evaluation signature envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return _unsigned_bytes(self, "evaluation_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "evaluation_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.evaluation_hash
            or self.evaluation_hash != self.content_hash()
        ):
            raise ValueError("V6.3 external evaluation is not sealed")


def sign_external_aggregate_evaluation_v63(
    *,
    private_key_pem: bytes,
    **data: object,
) -> ExternalAggregateEvaluationV63:
    data.setdefault("evaluated_at", _utc_now())
    return _sign_model(
        model_type=ExternalAggregateEvaluationV63,
        data=dict(data),
        private_key_pem=private_key_pem,
        hash_field="evaluation_hash",
    )  # type: ignore[return-value]


def _recomputed_external_metric(
    evaluation: ExternalAggregateEvaluationV63,
) -> float:
    return math.sqrt(
        evaluation.squared_error_sum
        / evaluation.target_squared_value_sum
    )


def _threshold_reasons(
    *,
    contract: PredictiveExternalQualificationContractV63,
    evaluation: ExternalAggregateEvaluationV63,
) -> list[str]:
    reasons: list[str] = []
    recomputed_metric = _recomputed_external_metric(evaluation)
    if (
        evaluation.holdout_observation_count
        < contract.minimum_external_observation_count
    ):
        reasons.append("external_observation_count_below_threshold")
    if recomputed_metric > contract.maximum_metric_value:
        reasons.append("external_metric_threshold_failed")
    return sorted(reasons)


class ExternalPredictivePromotionV63(StrictModel):
    """Independent promotion bound to the aggregate result and threshold."""

    schema_version: Literal["6.3-external-predictive-promotion"] = (
        "6.3-external-predictive-promotion"
    )
    qualification_id: Identifier
    contract_hash: Sha256
    local_context_hash: Sha256
    custody_hash: Sha256
    registration_hash: Sha256
    prediction_seal_hash: Sha256
    reservation_hash: Sha256
    evaluation_hash: Sha256
    threshold_recomputed_pass: bool
    integrity_incident_free: bool
    decision: Literal["QUALIFY", "REJECT"]
    reason_codes: list[Identifier]
    qualification_granted: bool
    private_values_disclosed: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    promotion_host_id: Identifier
    coordinator_host_id: Identifier
    generator_host_id: Identifier
    decided_at: datetime
    promotion_key_id: Identifier
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    promotion_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_promotion(self) -> "ExternalPredictivePromotionV63":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("promotion reasons must be sorted and unique")
        if self.promotion_host_id in {
            self.coordinator_host_id,
            self.generator_host_id,
        }:
            raise ValueError(
                "promotion host must differ from coordinator/generator"
            )
        expected = (
            self.threshold_recomputed_pass
            and self.integrity_incident_free
            and not self.reason_codes
        )
        if self.qualification_granted != expected:
            raise ValueError("promotion qualification differs from evidence")
        if self.decision != (
            "QUALIFY" if self.qualification_granted else "REJECT"
        ):
            raise ValueError("promotion decision differs from qualification")
        _assert_aware(self.decided_at, "decided_at")
        if self.promotion_hash and (
            not self.signature_base64
            or self.promotion_hash != self.content_hash()
        ):
            raise ValueError("V6.3 promotion signature envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return _unsigned_bytes(self, "promotion_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "promotion_hash")

    def assert_sealed(self) -> None:
        if (
            not self.signature_base64
            or not self.promotion_hash
            or self.promotion_hash != self.content_hash()
        ):
            raise ValueError("V6.3 external promotion is not sealed")


def sign_external_predictive_promotion_v63(
    *,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    registration: ExternalPredictionRegistrationV63,
    prediction_seal: PredictionSealV50,
    evaluation: ExternalAggregateEvaluationV63,
    integrity_incident_free: bool,
    promotion_host_id: str,
    promotion_key_id: str,
    private_key_pem: bytes,
    decided_at: datetime | None = None,
) -> ExternalPredictivePromotionV63:
    """External helper derives QUALIFY/REJECT; callers cannot choose it."""

    reasons = _threshold_reasons(contract=contract, evaluation=evaluation)
    if not integrity_incident_free:
        reasons.append("integrity_incident_present")
    reasons = sorted(set(reasons))
    return _sign_model(
        model_type=ExternalPredictivePromotionV63,
        data={
            "qualification_id": contract.qualification_id,
            "contract_hash": contract.contract_hash,
            "local_context_hash": contract.local_context_hash,
            "custody_hash": custody.custody_hash,
            "registration_hash": registration.registration_hash,
            "prediction_seal_hash": prediction_seal.seal_hash,
            "reservation_hash": evaluation.reservation_hash,
            "evaluation_hash": evaluation.evaluation_hash,
            "threshold_recomputed_pass": not _threshold_reasons(
                contract=contract,
                evaluation=evaluation,
            ),
            "integrity_incident_free": integrity_incident_free,
            "decision": "QUALIFY" if not reasons else "REJECT",
            "reason_codes": reasons,
            "qualification_granted": not reasons,
            "promotion_host_id": promotion_host_id,
            "coordinator_host_id": contract.coordinator_host_id,
            "generator_host_id": contract.generator_host_id,
            "decided_at": decided_at or _utc_now(),
            "promotion_key_id": promotion_key_id,
        },
        private_key_pem=private_key_pem,
        hash_field="promotion_hash",
    )  # type: ignore[return-value]


class ExternalEvaluationConsumptionV63(StrictModel):
    schema_version: Literal["6.3-external-evaluation-consumption"] = (
        "6.3-external-evaluation-consumption"
    )
    qualification_id: Identifier
    task_id: Identifier
    contract_hash: Sha256
    evaluation_id: Identifier
    evaluation_hash: Sha256
    prediction_seal_hash: Sha256
    reservation_hash: Sha256
    consumption_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_consumption(self) -> "ExternalEvaluationConsumptionV63":
        if (
            self.consumption_hash
            and self.consumption_hash != self.content_hash()
        ):
            raise ValueError("external evaluation consumption hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "consumption_hash")

    @classmethod
    def seal(cls, **data: object) -> "ExternalEvaluationConsumptionV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"consumption_hash"})
        payload["consumption_hash"] = draft.content_hash()
        return cls(**payload)


class ExternalPredictiveQualificationReceiptV63(StrictModel):
    """Code-owned final status; action authority is permanently false."""

    schema_version: Literal["6.3-external-predictive-qualification"] = (
        "6.3-external-predictive-qualification"
    )
    qualification_id: Identifier
    task_id: Identifier
    workspace_spec_hash: Sha256
    local_context_hash: Sha256
    contract_hash: Sha256
    custody_hash: Sha256
    custody_admission_hash: Sha256
    current_model_prediction_binding_hash: Sha256
    registration_hash: Sha256
    prediction_seal_hash: Sha256
    reservation_hash: Sha256
    evaluation_hash: Sha256
    consumption_hash: Sha256
    promotion_hash: Sha256 | None = None
    authority_artifact_hashes: dict[Identifier, Sha256]
    status: QualificationStatusV63
    reason_codes: list[Identifier]
    checks: dict[Identifier, bool]
    external_metric_value: FiniteNonNegative
    maximum_metric_value: FiniteNonNegative
    predictive_qualification_granted: bool
    scientific_qualification_granted: bool
    claim_ceiling: PredictiveClaimCeilingV63
    mechanistic_qualification_granted: Literal[False] = False
    prescriptive_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ExternalPredictiveQualificationReceiptV63":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("qualification reasons must be sorted and unique")
        expected_artifact_roles = {
            "contract",
            "forecast_input",
            "custody",
            "custody_admission",
            "prediction_binding",
            "prediction_vector",
            "prediction_seal",
            "registration",
            "evaluation_reservation",
            "evaluation",
            "evaluation_consumption",
        }
        if self.promotion_hash is not None:
            expected_artifact_roles.add("promotion")
        if set(self.authority_artifact_hashes) != expected_artifact_roles:
            raise ValueError(
                "qualification authority-artifact roles differ"
            )
        qualified = self.status == "EXTERNALLY_QUALIFIED"
        if (
            self.predictive_qualification_granted != qualified
            or self.scientific_qualification_granted != qualified
        ):
            raise ValueError("qualification flags differ from status")
        if qualified and (
            not self.promotion_hash
            or not self.checks
            or not all(self.checks.values())
            or self.reason_codes
            or self.claim_ceiling
            != "externally_qualified_predictive_evidence"
        ):
            raise ValueError("qualified receipt lacks mandatory evidence")
        if not qualified and (
            self.claim_ceiling == "externally_qualified_predictive_evidence"
        ):
            raise ValueError("unqualified receipt exceeds its claim ceiling")
        if self.status == "NOT_RUN" and self.promotion_hash is not None:
            raise ValueError("NOT_RUN receipt cannot bind a promotion")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("V6.3 qualification receipt hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("V6.3 qualification receipt is not sealed")

    @classmethod
    def seal(
        cls, **data: object
    ) -> "ExternalPredictiveQualificationReceiptV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


def _commit_qualification_receipt(
    workspace: StageWorkspaceV50,
    receipt: ExternalPredictiveQualificationReceiptV63,
) -> ExternalPredictiveQualificationReceiptV63:
    kind = "external_predictive_qualification_v63"
    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    transaction = (
        transaction_factory()
        if callable(transaction_factory)
        else nullcontext()
    )
    with transaction:
        try:
            prior = [
                item
                for _, item in workspace._artifacts_of_kind(
                    kind,
                    ExternalPredictiveQualificationReceiptV63,
                )
                if item.qualification_id == receipt.qualification_id
                or item.task_id == receipt.task_id
            ]
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise ExternalQualificationError(
                "qualification receipt ledger could not be replayed"
            ) from exc
        terminal = [item for item in prior if item.status != "NOT_RUN"]
        if terminal:
            if len(terminal) == 1 and terminal[0] == receipt:
                return terminal[0]
            raise ExternalQualificationError(
                "external predictive qualification is already final"
            )
        if receipt not in prior:
            workspace.commit_evidence(
                kind,
                receipt.model_dump(mode="json"),
            )
    return receipt


def _resolve_qualification_receipt(
    *,
    workspace: StageWorkspaceV50,
    receipt: ExternalPredictiveQualificationReceiptV63,
    persist: bool,
) -> ExternalPredictiveQualificationReceiptV63:
    if persist:
        return _commit_qualification_receipt(workspace, receipt)
    _exact_committed_artifact_hash(
        workspace=workspace,
        kind="external_predictive_qualification_v63",
        model=receipt,
        model_type=ExternalPredictiveQualificationReceiptV63,
    )
    return receipt


def _existing_consumptions(
    workspace: StageWorkspaceV50,
) -> list[ExternalEvaluationConsumptionV63]:
    try:
        return [
            item
            for _, item in workspace._artifacts_of_kind(
                _CONSUMPTION_KIND,
                ExternalEvaluationConsumptionV63,
            )
        ]
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExternalQualificationError(
            "external evaluation ledger could not be replayed"
        ) from exc


def _consume_evaluation_once(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    evaluation: ExternalAggregateEvaluationV63,
    prediction_seal: PredictionSealV50,
    reservation: ExternalEvaluationReservationV63,
    allow_create: bool = True,
) -> ExternalEvaluationConsumptionV63:
    expected = ExternalEvaluationConsumptionV63.seal(
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        evaluation_id=evaluation.evaluation_id,
        evaluation_hash=evaluation.evaluation_hash,
        prediction_seal_hash=prediction_seal.seal_hash,
        reservation_hash=reservation.reservation_hash,
    )
    transaction_factory = getattr(
        getattr(getattr(workspace, "graph", None), "store", None),
        "writer_transaction",
        None,
    )
    transaction = (
        transaction_factory()
        if callable(transaction_factory)
        else nullcontext()
    )
    with transaction:
        prior = [
            item
            for item in _existing_consumptions(workspace)
            if item.qualification_id == contract.qualification_id
            or item.task_id == contract.task_id
        ]
        if prior:
            if len(prior) == 1 and prior[0] == expected:
                return prior[0]
            raise ExternalQualificationError(
                "duplicate external evaluation rejected"
            )
        if not allow_create:
            raise ExternalQualificationError(
                "external evaluation consumption is not committed"
            )
        workspace.commit_evidence(
            _CONSUMPTION_KIND,
            expected.model_dump(mode="json"),
        )
    return expected


def _assert_chain_bindings(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    custody_admission: ExternalCustodyAdmissionV63,
    prediction_binding: CurrentModelPredictionBindingV63,
    registration: ExternalPredictionRegistrationV63,
    prediction_seal: PredictionSealV50,
    reservation: ExternalEvaluationReservationV63,
    evaluation: ExternalAggregateEvaluationV63,
) -> None:
    _assert_registered_prediction_chain(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=prediction_binding,
        registration=registration,
        prediction_seal=prediction_seal,
    )
    _assert_evaluation_reservation(
        workspace=workspace,
        contract=contract,
        custody=custody,
        custody_admission=custody_admission,
        prediction_binding=prediction_binding,
        registration=registration,
        prediction_seal=prediction_seal,
        reservation=reservation,
    )
    if not (
        contract.frozen_at
        <= custody.attested_at
        <= registration.registered_at
        <= prediction_seal.registered_at
        <= reservation.reserved_at
        <= evaluation.evaluated_at
    ):
        raise ExternalQualificationError(
            "external evidence chronology rejected"
        )
    if (
        evaluation.qualification_id != contract.qualification_id
        or evaluation.contract_hash != contract.contract_hash
        or evaluation.local_context_hash != contract.local_context_hash
        or evaluation.custody_hash != custody.custody_hash
        or evaluation.registration_hash != registration.registration_hash
        or evaluation.prediction_seal_hash != prediction_seal.seal_hash
        or evaluation.reservation_hash != reservation.reservation_hash
        or evaluation.prediction_artifact_hash
        != registration.prediction_artifact_hash
        or evaluation.external_snapshot_hash
        != custody.external_snapshot_hash
        or evaluation.holdout_commitment_hash
        != custody.holdout_commitment_hash
        or evaluation.normalization_scale_commitment_hash
        != custody.normalization_scale_commitment_hash
        or evaluation.target_order_hash != custody.target_order_hash
        or evaluation.holdout_observation_count
        != custody.holdout_observation_count
        or evaluation.metric != contract.metric
        or evaluation.evaluation_sequence
        != reservation.reserved_evaluation_sequence
        or evaluation.evaluator_key_id != reservation.evaluator_key_id
        or evaluation.evaluator_host_id != reservation.evaluator_host_id
    ):
        raise ExternalQualificationError(
            "external aggregate evaluation reservation binding rejected; "
            "external hosts must be distinct and reserved"
        )


def assess_external_predictive_qualification_v63(
    *,
    workspace: StageWorkspaceV50,
    contract: PredictiveExternalQualificationContractV63,
    custody: ExternalEvidenceCustodyV63,
    prediction_binding: CurrentModelPredictionBindingV63,
    registration: ExternalPredictionRegistrationV63,
    prediction_seal: PredictionSealV50,
    reservation: ExternalEvaluationReservationV63,
    evaluation: ExternalAggregateEvaluationV63,
    promotion: ExternalPredictivePromotionV63 | None,
    trusted_public_keys: Mapping[str, bytes],
    _persist: bool = True,
) -> ExternalPredictiveQualificationReceiptV63:
    """Replay the full chain and fail closed on stale or substituted evidence."""

    if not isinstance(reservation, ExternalEvaluationReservationV63):
        raise ExternalQualificationError(
            "formal assessment requires an external evaluation reservation"
        )
    current = _assert_contract_current(workspace=workspace, contract=contract)
    _assert_trusted_authority_set(
        contract=contract,
        trusted_public_keys=trusted_public_keys,
    )
    expected_runtime_ids = {
        "custody": custody.custody_key_id,
        "registry": registration.registry_key_id,
        "evaluator": evaluation.evaluator_key_id,
        "promotion": (
            promotion.promotion_key_id
            if promotion
            else contract.trusted_authority_key_ids["promotion"]
        ),
    }
    if expected_runtime_ids != contract.trusted_authority_key_ids:
        raise ExternalQualificationError(
            "external authority IDs differ from frozen protocol"
        )
    signed_items = [
        (
            custody,
            custody.custody_key_id,
            custody.signature_base64,
            "custody_hash",
            "custody",
        ),
        (
            registration,
            registration.registry_key_id,
            registration.signature_base64,
            "registration_hash",
            "prediction registration",
        ),
        (
            evaluation,
            evaluation.evaluator_key_id,
            evaluation.signature_base64,
            "evaluation_hash",
            "external evaluation",
        ),
    ]
    for model, key_id, signature, hash_field, label in signed_items:
        _require_signed(
            model=model,
            key_id=key_id,
            signature_base64=signature,
            trusted_public_keys=trusted_public_keys,
            hash_field=hash_field,
            label=label,
        )
    custody_admission, custody_admission_artifact_hash = (
        _verified_custody_admission(
            workspace=workspace,
            contract=contract,
            custody=custody,
        )
    )
    _, forecast_input_artifact_hash = (
        _current_external_forecast_input(
            workspace=workspace,
            contract=contract,
        )
    )
    try:
        prediction_vector = ExternalPredictionVectorV63.model_validate(
            workspace._artifact_payload_by_hash(
                prediction_binding.prediction_artifact_hash
            )
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ExternalQualificationError(
            "typed prediction vector could not be loaded"
        ) from exc
    prediction_vector_artifact_hash = _exact_committed_artifact_hash(
        workspace=workspace,
        kind=_PREDICTION_VECTOR_KIND,
        model=prediction_vector,
        model_type=ExternalPredictionVectorV63,
    )
    if (
        prediction_vector_artifact_hash
        != prediction_binding.prediction_artifact_hash
    ):
        raise ExternalQualificationError(
            "typed prediction vector artifact kind differs"
        )
    authority_artifact_hashes = {
        "contract": _commit_unique_qualification_artifact(
            workspace=workspace,
            kind="predictive_external_qualification_contract_v63",
            model=contract,
            model_type=PredictiveExternalQualificationContractV63,
            allow_create=False,
        ),
        "forecast_input": forecast_input_artifact_hash,
        "custody": _commit_unique_qualification_artifact(
            workspace=workspace,
            kind="external_evidence_custody_v63",
            model=custody,
            model_type=ExternalEvidenceCustodyV63,
            allow_create=False,
        ),
        "custody_admission": custody_admission_artifact_hash,
        "prediction_binding": _exact_committed_artifact_hash(
            workspace=workspace,
            kind=_PREDICTION_BINDING_KIND,
            model=prediction_binding,
            model_type=CurrentModelPredictionBindingV63,
        ),
        "prediction_vector": prediction_vector_artifact_hash,
        "prediction_seal": _exact_committed_artifact_hash(
            workspace=workspace,
            kind="prediction_seal_v50",
            model=prediction_seal,
            model_type=PredictionSealV50,
        ),
        "registration": _commit_unique_qualification_artifact(
            workspace=workspace,
            kind="external_prediction_registration_v63",
            model=registration,
            model_type=ExternalPredictionRegistrationV63,
            allow_create=False,
        ),
        "evaluation_reservation": _exact_committed_artifact_hash(
            workspace=workspace,
            kind=_RESERVATION_KIND,
            model=reservation,
            model_type=ExternalEvaluationReservationV63,
        ),
    }
    _assert_chain_bindings(
        workspace=workspace,
        contract=contract,
        custody=custody,
        custody_admission=custody_admission,
        prediction_binding=prediction_binding,
        registration=registration,
        prediction_seal=prediction_seal,
        reservation=reservation,
        evaluation=evaluation,
    )

    key_ids = [
        custody.custody_key_id,
        registration.registry_key_id,
        evaluation.evaluator_key_id,
        *([promotion.promotion_key_id] if promotion else []),
    ]
    if len(key_ids) != len(set(key_ids)):
        raise ExternalQualificationError(
            "external authority key IDs must be pairwise distinct"
        )
    try:
        fingerprints = [
            external_qualification_key_fingerprint_v63(
                trusted_public_keys[key_id]
            )
            for key_id in key_ids
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalQualificationError(
            "external authority key is not pinned Ed25519"
        ) from exc
    if len(fingerprints) != len(set(fingerprints)):
        raise ExternalQualificationError(
            "external authorities reuse physical signing keys"
        )

    role_hosts = {
        "custodian": custody.custodian_host_id,
        "registry": registration.registry_host_id,
        "evaluator": evaluation.evaluator_host_id,
    }
    if any(
        host
        in {contract.coordinator_host_id, contract.generator_host_id}
        for host in role_hosts.values()
    ):
        raise ExternalQualificationError(
            "external evidence host overlaps coordinator/generator"
        )
    if len(set(role_hosts.values())) != len(role_hosts):
        raise ExternalQualificationError(
            "custody, registry, and evaluator hosts must be distinct"
        )

    authority_artifact_hashes["evaluation"] = (
        _commit_unique_qualification_artifact(
            workspace=workspace,
            kind="external_aggregate_evaluation_v63",
            model=evaluation,
            model_type=ExternalAggregateEvaluationV63,
            allow_create=_persist,
        )
    )
    consumption = _consume_evaluation_once(
        workspace=workspace,
        contract=contract,
        evaluation=evaluation,
        prediction_seal=prediction_seal,
        reservation=reservation,
        allow_create=_persist,
    )
    authority_artifact_hashes["evaluation_consumption"] = (
        _exact_committed_artifact_hash(
            workspace=workspace,
            kind=_CONSUMPTION_KIND,
            model=consumption,
            model_type=ExternalEvaluationConsumptionV63,
        )
    )
    threshold_reasons = _threshold_reasons(
        contract=contract,
        evaluation=evaluation,
    )
    if promotion is None:
        receipt = ExternalPredictiveQualificationReceiptV63.seal(
            qualification_id=contract.qualification_id,
            task_id=contract.task_id,
            workspace_spec_hash=current.workspace_spec_hash,
            local_context_hash=current.context_hash,
            contract_hash=contract.contract_hash,
            custody_hash=custody.custody_hash,
            custody_admission_hash=custody_admission.admission_hash,
            current_model_prediction_binding_hash=(
                prediction_binding.binding_hash
            ),
            registration_hash=registration.registration_hash,
            prediction_seal_hash=prediction_seal.seal_hash,
            reservation_hash=reservation.reservation_hash,
            evaluation_hash=evaluation.evaluation_hash,
            consumption_hash=consumption.consumption_hash,
            authority_artifact_hashes=authority_artifact_hashes,
            status="NOT_RUN",
            reason_codes=["external_promotion_missing"],
            checks={
                "current_v62_closure_replayed": True,
                "external_evaluation_threshold_pass": not threshold_reasons,
                "promotion_present": False,
            },
            external_metric_value=_recomputed_external_metric(evaluation),
            maximum_metric_value=contract.maximum_metric_value,
            predictive_qualification_granted=False,
            scientific_qualification_granted=False,
            claim_ceiling=current.prior_claim_ceiling,
        )
        return _resolve_qualification_receipt(
            workspace=workspace,
            receipt=receipt,
            persist=_persist,
        )

    _require_signed(
        model=promotion,
        key_id=promotion.promotion_key_id,
        signature_base64=promotion.signature_base64,
        trusted_public_keys=trusted_public_keys,
        hash_field="promotion_hash",
        label="external promotion",
    )
    if promotion.promotion_host_id in {
        contract.coordinator_host_id,
        contract.generator_host_id,
        custody.custodian_host_id,
        registration.registry_host_id,
        evaluation.evaluator_host_id,
    }:
        raise ExternalQualificationError(
            "promotion host is not independent"
        )
    if promotion.decided_at < evaluation.evaluated_at:
        raise ExternalQualificationError(
            "promotion predates the external evaluation"
        )
    expected_promotion_binding = (
        promotion.qualification_id == contract.qualification_id
        and promotion.contract_hash == contract.contract_hash
        and promotion.local_context_hash == contract.local_context_hash
        and promotion.custody_hash == custody.custody_hash
        and promotion.registration_hash == registration.registration_hash
        and promotion.prediction_seal_hash == prediction_seal.seal_hash
        and promotion.reservation_hash == reservation.reservation_hash
        and promotion.evaluation_hash == evaluation.evaluation_hash
        and promotion.coordinator_host_id == contract.coordinator_host_id
        and promotion.generator_host_id == contract.generator_host_id
    )
    if not expected_promotion_binding:
        raise ExternalQualificationError(
            "external promotion chain binding rejected"
        )
    recomputed_pass = not threshold_reasons
    if (
        promotion.threshold_recomputed_pass != recomputed_pass
        or promotion.qualification_granted
        != (
            recomputed_pass
            and promotion.integrity_incident_free
            and not promotion.reason_codes
        )
    ):
        raise ExternalQualificationError(
            "promotion differs from code-recomputed threshold"
        )
    authority_artifact_hashes["promotion"] = (
        _commit_unique_qualification_artifact(
            workspace=workspace,
            kind="external_predictive_promotion_v63",
            model=promotion,
            model_type=ExternalPredictivePromotionV63,
            allow_create=_persist,
        )
    )
    reasons = sorted(
        set(
            threshold_reasons
            + list(promotion.reason_codes)
            + (
                []
                if promotion.integrity_incident_free
                else ["integrity_incident_present"]
            )
        )
    )
    qualified = bool(
        promotion.decision == "QUALIFY"
        and promotion.qualification_granted
        and recomputed_pass
        and not reasons
    )
    checks = {
        "current_v62_closure_replayed": True,
        "current_s4_prediction_seal_verified": True,
        "non_fixture_external_custody_verified": not custody.fixture_only,
        "strict_unseen_verified": custody.strict_unseen_verified,
        "independent_measurement_review_passed": (
            custody.independent_measurement_review_passed
        ),
        "external_environment_verified": (
            custody.external_environment_verified
        ),
        "aggregate_evaluation_only": evaluation.aggregate_only
        and not evaluation.per_observation_feedback_released
        and not evaluation.private_values_disclosed,
        "external_evaluation_threshold_pass": recomputed_pass,
        "promotion_signature_verified": True,
        "promotion_integrity_incident_free": (
            promotion.integrity_incident_free
        ),
    }
    if not all(checks.values()):
        qualified = False
        reasons.extend(
            key for key, passed in checks.items() if not passed
        )
        reasons = sorted(set(reasons))
    receipt = ExternalPredictiveQualificationReceiptV63.seal(
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        workspace_spec_hash=current.workspace_spec_hash,
        local_context_hash=current.context_hash,
        contract_hash=contract.contract_hash,
        custody_hash=custody.custody_hash,
        custody_admission_hash=custody_admission.admission_hash,
        current_model_prediction_binding_hash=prediction_binding.binding_hash,
        registration_hash=registration.registration_hash,
        prediction_seal_hash=prediction_seal.seal_hash,
        reservation_hash=reservation.reservation_hash,
        evaluation_hash=evaluation.evaluation_hash,
        consumption_hash=consumption.consumption_hash,
        promotion_hash=promotion.promotion_hash,
        authority_artifact_hashes=authority_artifact_hashes,
        status="EXTERNALLY_QUALIFIED" if qualified else "REJECTED",
        reason_codes=[] if qualified else reasons,
        checks=checks,
        external_metric_value=_recomputed_external_metric(evaluation),
        maximum_metric_value=contract.maximum_metric_value,
        predictive_qualification_granted=qualified,
        scientific_qualification_granted=qualified,
        claim_ceiling=(
            "externally_qualified_predictive_evidence"
            if qualified
            else current.prior_claim_ceiling
        ),
    )
    return _resolve_qualification_receipt(
        workspace=workspace,
        receipt=receipt,
        persist=_persist,
    )


class ExternalPredictiveQualificationReplayV63(StrictModel):
    schema_version: Literal["6.3-external-qualification-replay"] = (
        "6.3-external-qualification-replay"
    )
    receipt_hash: Sha256 | None
    status: Literal["PASS", "FAIL"]
    checks: dict[Identifier, bool]
    reason_codes: list[Identifier]
    real_world_action_authorized: Literal[False] = False
    replay_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_replay(self) -> "ExternalPredictiveQualificationReplayV63":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("qualification replay reasons differ")
        expected = bool(self.checks) and all(self.checks.values())
        if (self.status == "PASS") != expected:
            raise ValueError("qualification replay status differs from checks")
        if self.replay_hash and self.replay_hash != self.content_hash():
            raise ValueError("qualification replay hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "replay_hash")

    @classmethod
    def seal(
        cls, **data: object
    ) -> "ExternalPredictiveQualificationReplayV63":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"replay_hash"})
        payload["replay_hash"] = draft.content_hash()
        return cls(**payload)


def verify_external_predictive_qualification_v63(
    *,
    workspace: StageWorkspaceV50,
    receipt: ExternalPredictiveQualificationReceiptV63,
    trusted_public_keys: Mapping[str, bytes],
) -> ExternalPredictiveQualificationReplayV63:
    """Reload every signed envelope from the authority store and replay."""

    checks: dict[str, bool] = {
        "receipt_self_hash": bool(receipt.receipt_hash)
        and receipt.receipt_hash == receipt.content_hash(),
        "receipt_committed": False,
        "all_authority_artifacts_load": False,
        "full_chain_recomputed": False,
    }
    reasons: list[str] = []
    try:
        receipt.assert_sealed()
        checks["receipt_committed"] = bool(
            _exact_committed_artifact_hash(
                workspace=workspace,
                kind="external_predictive_qualification_v63",
                model=receipt,
                model_type=ExternalPredictiveQualificationReceiptV63,
            )
        )
        artifact_types: dict[
            str,
            tuple[str, type[StrictModel]],
        ] = {
            "contract": (
                "predictive_external_qualification_contract_v63",
                PredictiveExternalQualificationContractV63,
            ),
            "forecast_input": (
                _FORECAST_INPUT_KIND,
                ExternalForecastInputV63,
            ),
            "custody": (
                "external_evidence_custody_v63",
                ExternalEvidenceCustodyV63,
            ),
            "custody_admission": (
                "external_custody_admission_v63",
                ExternalCustodyAdmissionV63,
            ),
            "prediction_binding": (
                _PREDICTION_BINDING_KIND,
                CurrentModelPredictionBindingV63,
            ),
            "prediction_vector": (
                _PREDICTION_VECTOR_KIND,
                ExternalPredictionVectorV63,
            ),
            "prediction_seal": (
                "prediction_seal_v50",
                PredictionSealV50,
            ),
            "registration": (
                "external_prediction_registration_v63",
                ExternalPredictionRegistrationV63,
            ),
            "evaluation_reservation": (
                _RESERVATION_KIND,
                ExternalEvaluationReservationV63,
            ),
            "evaluation": (
                "external_aggregate_evaluation_v63",
                ExternalAggregateEvaluationV63,
            ),
            "evaluation_consumption": (
                _CONSUMPTION_KIND,
                ExternalEvaluationConsumptionV63,
            ),
            "promotion": (
                "external_predictive_promotion_v63",
                ExternalPredictivePromotionV63,
            ),
        }
        loaded: dict[str, StrictModel] = {}
        for role, artifact_hash in receipt.authority_artifact_hashes.items():
            kind, model_type = artifact_types[role]
            model = model_type.model_validate(
                workspace._artifact_payload_by_hash(artifact_hash)
            )
            exact_hash = _exact_committed_artifact_hash(
                workspace=workspace,
                kind=kind,
                model=model,
                model_type=model_type,
            )
            if exact_hash != artifact_hash:
                raise ExternalQualificationError(
                    f"{role} authority artifact hash differs"
                )
            loaded[role] = model
        checks["all_authority_artifacts_load"] = True
        recomputed = assess_external_predictive_qualification_v63(
            workspace=workspace,
            contract=loaded["contract"],  # type: ignore[arg-type]
            custody=loaded["custody"],  # type: ignore[arg-type]
            prediction_binding=loaded["prediction_binding"],  # type: ignore[arg-type]
            registration=loaded["registration"],  # type: ignore[arg-type]
            prediction_seal=loaded["prediction_seal"],  # type: ignore[arg-type]
            reservation=loaded["evaluation_reservation"],  # type: ignore[arg-type]
            evaluation=loaded["evaluation"],  # type: ignore[arg-type]
            promotion=loaded.get("promotion"),  # type: ignore[arg-type]
            trusted_public_keys=trusted_public_keys,
            _persist=False,
        )
        checks["full_chain_recomputed"] = recomputed == receipt
        if recomputed != receipt:
            reasons.append("qualification_recomputation_differs")
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        reasons.extend(
            ["qualification_replay_failed", type(exc).__name__.lower()]
        )
    for check_id, passed in checks.items():
        if not passed:
            reasons.append(check_id)
    reasons = sorted(set(reasons))
    return ExternalPredictiveQualificationReplayV63.seal(
        receipt_hash=receipt.receipt_hash,
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        reason_codes=reasons,
    )


__all__ = [
    "CurrentModelPredictionBindingV63",
    "ExternalAggregateEvaluationV63",
    "ExternalCustodyAdmissionV63",
    "ExternalEvidenceCustodyV63",
    "ExternalEvaluationConsumptionV63",
    "ExternalEvaluationReservationV63",
    "ExternalForecastInputV63",
    "ExternalPredictionVectorV63",
    "ExternalPredictionRegistrationV63",
    "ExternalPredictivePromotionV63",
    "ExternalPredictiveQualificationReplayV63",
    "ExternalPredictiveQualificationReceiptV63",
    "ExternalQualificationError",
    "PredictiveExternalQualificationContractV63",
    "PredictiveLocalContextV63",
    "admit_external_evidence_custody_v63",
    "assess_external_predictive_qualification_v63",
    "commit_external_forecast_input_v63",
    "derive_predictive_local_context_v63",
    "external_qualification_key_fingerprint_v63",
    "freeze_predictive_external_qualification_contract_v63",
    "issue_current_model_prediction_binding_v63",
    "register_external_prediction_v63",
    "reserve_external_evaluation_v63",
    "sign_external_aggregate_evaluation_v63",
    "sign_external_evidence_custody_v63",
    "sign_external_prediction_registration_v63",
    "sign_external_predictive_promotion_v63",
    "verify_external_predictive_qualification_v63",
]
