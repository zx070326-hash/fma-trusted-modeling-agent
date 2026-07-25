"""Public registration and V5 graph binding for an I32-style campaign."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from fma._file_lock import exclusive_file_lock
from fma.hashing import canonical_json, sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.external_harness import PredictionDocumentV50
from fma.v5.stage_workspace import StageWorkspaceError, StageWorkspaceV50
from fma.v5.workspace_schemas import PredictionSealV50

from .custody import (
    CustodyVerificationReceiptV53,
    ExternalCustodyAttestationV53,
    PrivateScoreContractV53,
)
from .ode_forecast import ODEForecastBundleV53


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


_V53_SCIENTIFIC_CHECK_STAGES = {
    "scalar_ode_forecast_v53_l0": "S3",
    "scalar_ode_forecast_v53_l1": "S3",
    "scalar_ode_forecast_v53_l2": "S3",
    "scalar_ode_forecast_v53_l3": "S4",
    "scalar_ode_forecast_v53_l4": "S4",
}


def _v53_scientific_checks_bind_bundle(
    workspace: StageWorkspaceV50,
    forecast_bundle: ODEForecastBundleV53,
) -> bool:
    for check_id, stage in _V53_SCIENTIFIC_CHECK_STAGES.items():
        if workspace.current_gate(stage) is None:
            return False
        manifest = workspace._manifest_for_stage(stage)
        checks = workspace._latest_checks(stage, str(manifest.manifest_hash))
        result = checks.get(check_id)
        if (
            result is None
            or result.status != "PASS"
            or result.adapter_id != "scalar_ode_forecast_v53_scientific_adapter"
            or result.adapter_version != "5.3"
            or not result.evidence_refs
        ):
            return False
        bound = False
        for evidence_hash in result.evidence_refs:
            payload = workspace._artifact_payload_by_hash(evidence_hash)
            if (
                isinstance(payload, dict)
                and payload.get("check_id") == check_id
                and payload.get("adapter_id")
                == "scalar_ode_forecast_v53_scientific_adapter"
                and isinstance(payload.get("payload"), dict)
                and payload["payload"].get("forecast_bundle_hash")
                == forecast_bundle.bundle_hash
            ):
                bound = True
                break
        if not bound:
            return False
    return True


class PredictionRegistrationV53(StrictModel):
    """Immutable public registration; it contains no private target value."""

    schema_version: Literal["5.3"] = "5.3"
    registration_id: Identifier
    case_id: Identifier
    score_contract_hash: Sha256
    forecast_plan_hash: Sha256
    forecast_bundle_hash: Sha256
    custody_attestation_hash: Sha256
    custody_verification_receipt_hash: Sha256
    prediction_snapshot_hash: Sha256
    prediction_semantic_hash: Sha256
    snapshot_relative_path: Annotated[str, Field(min_length=1)]
    registry_event_hash: Sha256
    external_anchor_receipt_hash: Sha256
    fixture_only: bool
    private_holdout_accessed_before_registration: Literal[False] = False
    private_acceptance_data_exposed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    registered_at: datetime
    registration_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_registration(self) -> "PredictionRegistrationV53":
        if self.registered_at.utcoffset() is None:
            raise ValueError("registered_at must be timezone-aware")
        if self.registration_hash and (self.registration_hash != self.content_hash()):
            raise ValueError("prediction registration hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "registration_hash")

    def assert_sealed(self) -> None:
        if not self.registration_hash or self.registration_hash != self.content_hash():
            raise ValueError("prediction registration is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PredictionRegistrationV53":
        data.setdefault("registered_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"registration_hash"})
        payload["registration_hash"] = draft.content_hash()
        return cls(**payload)


class PredictionRegistryEventV53(StrictModel):
    schema_version: Literal["5.3"] = "5.3"
    sequence: Annotated[int, Field(ge=1)]
    event_type: Literal["prediction_registered"]
    case_id: Identifier
    payload: dict[str, Any]
    previous_event_hash: Sha256 | None = None
    created_at: datetime
    event_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_event(self) -> "PredictionRegistryEventV53":
        if self.sequence == 1 and self.previous_event_hash is not None:
            raise ValueError("first registry event cannot have a predecessor")
        if self.sequence > 1 and self.previous_event_hash is None:
            raise ValueError("later registry event needs a predecessor")
        if self.event_hash and self.event_hash != self.content_hash():
            raise ValueError("registry event hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "event_hash")

    @classmethod
    def seal(cls, **data: object) -> "PredictionRegistryEventV53":
        data.setdefault("created_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"event_hash"})
        payload["event_hash"] = draft.content_hash()
        return cls(**payload)


class PublicPredictionRegistryV53:
    """Code-owned, single-writer public registry outside the task workspace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.event_path = self.root / "events.jsonl"
        self.lock_path = self.root / ".events.writer.lock"

    def _case_root(self, case_id: str) -> Path:
        if not case_id or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in case_id
        ):
            raise ValueError("unsafe case identifier")
        path = (self.root / "cases" / case_id).resolve()
        if self.root not in path.parents:
            raise ValueError("case path escapes registry root")
        return path

    def _validated_events(self) -> list[PredictionRegistryEventV53]:
        if not self.event_path.exists():
            return []
        events: list[PredictionRegistryEventV53] = []
        previous: str | None = None
        for index, line in enumerate(
            self.event_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            event = PredictionRegistryEventV53.model_validate_json(line)
            if event.sequence != index:
                raise ValueError("prediction registry sequence differs")
            if event.previous_event_hash != previous:
                raise ValueError("prediction registry chain differs")
            if not event.event_hash or event.event_hash != event.content_hash():
                raise ValueError("prediction registry event is unsealed")
            events.append(event)
            previous = event.event_hash
        return events

    def verify_event_chain(self) -> bool:
        try:
            self._validated_events()
            return True
        except (OSError, ValueError):
            return False

    def register(
        self,
        *,
        registration_id: str,
        prediction_path: str | Path,
        workspace_root: str | Path,
        forecast_bundle: ODEForecastBundleV53,
        score_contract: PrivateScoreContractV53,
        custody_attestation: ExternalCustodyAttestationV53,
        custody_verification: CustodyVerificationReceiptV53,
        external_anchor_receipt_hash: str,
    ) -> PredictionRegistrationV53:
        if not self.verify_event_chain():
            raise RuntimeError("prediction registry event chain is invalid")
        forecast_bundle.forecast_plan.assert_sealed()
        score_contract.assert_forecast_plan(forecast_bundle.forecast_plan)
        custody_attestation.assert_sealed()
        if not forecast_bundle.scientific_acceptance:
            raise ValueError(
                "public prediction registration requires passing V5.3 evidence"
            )
        if custody_verification.status != "VERIFIED":
            raise ValueError("external custody verification did not pass")
        if (
            custody_verification.attestation_hash
            != custody_attestation.attestation_hash
            or custody_verification.score_contract_hash != score_contract.contract_hash
        ):
            raise ValueError("custody verification binds another campaign")
        if (
            custody_attestation.score_contract_hash != score_contract.contract_hash
            or custody_attestation.forecast_plan_hash
            != forecast_bundle.forecast_plan.plan_hash
        ):
            raise ValueError("custody attestation binds another campaign")

        workspace = Path(workspace_root).resolve()
        if _paths_overlap(self.root, workspace):
            raise ValueError("prediction registry and task workspace overlap")
        prediction_file = Path(prediction_path).resolve()
        if workspace not in prediction_file.parents or not prediction_file.is_file():
            raise ValueError("prediction must be a file inside the task workspace")
        raw = prediction_file.read_bytes()
        if len(raw) > 1_048_576:
            raise ValueError("prediction document exceeds 1 MiB")
        prediction = PredictionDocumentV50.model_validate_json(raw)
        expected = forecast_bundle.final_refit.prediction_document(
            score_contract.case_id
        )
        if prediction != expected:
            raise ValueError(
                "prediction document differs from the certified final refit"
            )

        with exclusive_file_lock(self.lock_path):
            events = self._validated_events()
            case_root = self._case_root(score_contract.case_id)
            registration_path = case_root / "registration.json"
            if registration_path.exists():
                raise RuntimeError("prediction registration is immutable and unique")
            snapshot_hash = _sha256_bytes(raw)
            snapshot_path = case_root / "snapshots" / f"{snapshot_hash}.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with snapshot_path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            event = PredictionRegistryEventV53.seal(
                sequence=len(events) + 1,
                event_type="prediction_registered",
                case_id=score_contract.case_id,
                payload={
                    "registration_id": registration_id,
                    "score_contract_hash": score_contract.contract_hash,
                    "forecast_bundle_hash": forecast_bundle.bundle_hash,
                    "prediction_snapshot_hash": snapshot_hash,
                    "prediction_semantic_hash": sha256_value(prediction),
                    "custody_attestation_hash": (custody_attestation.attestation_hash),
                    "external_anchor_receipt_hash": (external_anchor_receipt_hash),
                },
                previous_event_hash=(events[-1].event_hash if events else None),
            )
            registration = PredictionRegistrationV53.seal(
                registration_id=registration_id,
                case_id=score_contract.case_id,
                score_contract_hash=score_contract.contract_hash,
                forecast_plan_hash=forecast_bundle.forecast_plan.plan_hash,
                forecast_bundle_hash=forecast_bundle.bundle_hash,
                custody_attestation_hash=custody_attestation.attestation_hash,
                custody_verification_receipt_hash=(custody_verification.receipt_hash),
                prediction_snapshot_hash=snapshot_hash,
                prediction_semantic_hash=sha256_value(prediction),
                snapshot_relative_path=snapshot_path.relative_to(self.root).as_posix(),
                registry_event_hash=event.event_hash,
                external_anchor_receipt_hash=external_anchor_receipt_hash,
                fixture_only=forecast_bundle.fixture_only,
            )
            registration_path.parent.mkdir(parents=True, exist_ok=True)
            with registration_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(registration) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return registration

    def verify_registration(self, registration: PredictionRegistrationV53) -> bool:
        try:
            registration.assert_sealed()
            if not self.verify_event_chain():
                return False
            case_root = self._case_root(registration.case_id)
            stored = PredictionRegistrationV53.model_validate_json(
                (case_root / "registration.json").read_text(encoding="utf-8")
            )
            if stored != registration:
                return False
            snapshot = (self.root / registration.snapshot_relative_path).resolve()
            if self.root not in snapshot.parents:
                return False
            raw = snapshot.read_bytes()
            if _sha256_bytes(raw) != registration.prediction_snapshot_hash:
                return False
            prediction = PredictionDocumentV50.model_validate_json(raw)
            if sha256_value(prediction) != registration.prediction_semantic_hash:
                return False
            matching = [
                event
                for event in self._validated_events()
                if event.event_hash == registration.registry_event_hash
            ]
            return (
                len(matching) == 1
                and matching[0].case_id == registration.case_id
                and matching[0].payload.get("forecast_bundle_hash")
                == registration.forecast_bundle_hash
                and matching[0].payload.get("prediction_snapshot_hash")
                == registration.prediction_snapshot_hash
            )
        except (OSError, ValueError, RuntimeError):
            return False


class I32GraphBindingV53(StrictModel):
    """Binding receipt over real V5 graph artifacts and an authenticated S4 gate."""

    schema_version: Literal["5.3"] = "5.3"
    workspace_spec_hash: Sha256
    graph_id: Identifier
    s4_gate_hash: Sha256
    graph_snapshot_before_binding_hash: Sha256
    graph_event_tip_before_binding: Sha256
    forecast_bundle_artifact_hash: Sha256
    custody_attestation_artifact_hash: Sha256
    custody_verification_artifact_hash: Sha256
    prediction_registration_artifact_hash: Sha256
    prediction_seal_hash: Sha256
    forecast_bundle_hash: Sha256
    score_contract_hash: Sha256
    custody_attestation_hash: Sha256
    prediction_registration_hash: Sha256
    status: Literal["GRAPH_BOUND_AWAITING_PRIVATE_EVALUATION"] = (
        "GRAPH_BOUND_AWAITING_PRIVATE_EVALUATION"
    )
    private_evaluation_status: Literal["NOT_RUN"] = "NOT_RUN"
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    bound_at: datetime
    binding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "I32GraphBindingV53":
        if self.bound_at.utcoffset() is None:
            raise ValueError("bound_at must be timezone-aware")
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("I32 graph binding hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    @classmethod
    def seal(cls, **data: object) -> "I32GraphBindingV53":
        data.setdefault("bound_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"binding_hash"})
        payload["binding_hash"] = draft.content_hash()
        return cls(**payload)


class I32GraphBindingResultV53(StrictModel):
    binding: I32GraphBindingV53
    binding_artifact: ArtifactRef
    prediction_seal: PredictionSealV50


def bind_i32_public_campaign_to_v5_v53(
    *,
    workspace: StageWorkspaceV50,
    forecast_bundle: ODEForecastBundleV53,
    score_contract: PrivateScoreContractV53,
    custody_attestation: ExternalCustodyAttestationV53,
    custody_verification: CustodyVerificationReceiptV53,
    prediction_registration: PredictionRegistrationV53,
) -> I32GraphBindingResultV53:
    """Bind accepted public evidence after S4; never run or promote private data."""

    if not workspace.verify():
        raise StageWorkspaceError("V5 workspace failed verification")
    s4_gate_hash = workspace.current_gate("S4")
    if s4_gate_hash is None:
        raise StageWorkspaceError("I32 binding requires a current S4 gate")
    if not forecast_bundle.scientific_acceptance:
        raise ValueError("I32 binding requires passing V5.3 public evidence")
    if not _v53_scientific_checks_bind_bundle(workspace, forecast_bundle):
        raise StageWorkspaceError(
            "I32 binding requires authenticated V5.3 L0-L4 checks "
            "over the frozen S3/S4 bundle"
        )
    score_contract.assert_forecast_plan(forecast_bundle.forecast_plan)
    custody_attestation.assert_sealed()
    prediction_registration.assert_sealed()
    if custody_verification.status != "VERIFIED":
        raise ValueError("I32 binding requires verified external custody")
    expected = {
        "score_contract_hash": score_contract.contract_hash,
        "forecast_plan_hash": forecast_bundle.forecast_plan.plan_hash,
        "forecast_bundle_hash": forecast_bundle.bundle_hash,
        "custody_attestation_hash": custody_attestation.attestation_hash,
        "custody_verification_receipt_hash": custody_verification.receipt_hash,
    }
    actual = {
        "score_contract_hash": prediction_registration.score_contract_hash,
        "forecast_plan_hash": prediction_registration.forecast_plan_hash,
        "forecast_bundle_hash": prediction_registration.forecast_bundle_hash,
        "custody_attestation_hash": (prediction_registration.custody_attestation_hash),
        "custody_verification_receipt_hash": (
            prediction_registration.custody_verification_receipt_hash
        ),
    }
    if actual != expected:
        raise ValueError("prediction registration binds another I32 campaign")

    bundle_ref = workspace.commit_evidence("i32_forecast_bundle_v53", forecast_bundle)
    custody_ref = workspace.commit_evidence(
        "i32_custody_attestation_v53", custody_attestation
    )
    verification_ref = workspace.commit_evidence(
        "i32_custody_verification_v53", custody_verification
    )
    registration_ref = workspace.commit_evidence(
        "i32_prediction_registration_v53", prediction_registration
    )
    prediction_seal = workspace.issue_prediction_seal(
        task_id=forecast_bundle.task_id,
        training_snapshot_hash=forecast_bundle.public_snapshot_hash,
        candidate_hash=sha256_value(
            {
                "selected_candidate_id": (
                    forecast_bundle.final_refit.selected_candidate_id
                ),
                "final_fit_hash": forecast_bundle.final_refit.final_fit.fit_hash,
            }
        ),
        prediction_artifact_hash=(prediction_registration.prediction_snapshot_hash),
        external_registration_hash=(prediction_registration.registration_hash),
        external_snapshot_hash=(prediction_registration.prediction_snapshot_hash),
        holdout_commitment_hash=custody_attestation.capsule_commitment,
    )
    state = workspace.graph.project_state()
    binding = I32GraphBindingV53.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        graph_id=workspace.spec.graph_id,
        s4_gate_hash=s4_gate_hash,
        graph_snapshot_before_binding_hash=state.snapshot.snapshot_hash,
        graph_event_tip_before_binding=state.snapshot.last_graph_event_hash,
        forecast_bundle_artifact_hash=bundle_ref.sha256,
        custody_attestation_artifact_hash=custody_ref.sha256,
        custody_verification_artifact_hash=verification_ref.sha256,
        prediction_registration_artifact_hash=registration_ref.sha256,
        prediction_seal_hash=prediction_seal.seal_hash,
        forecast_bundle_hash=forecast_bundle.bundle_hash,
        score_contract_hash=score_contract.contract_hash,
        custody_attestation_hash=custody_attestation.attestation_hash,
        prediction_registration_hash=(prediction_registration.registration_hash),
    )
    binding_ref = workspace.commit_evidence("i32_graph_binding_v53", binding)
    return I32GraphBindingResultV53(
        binding=binding,
        binding_artifact=binding_ref,
        prediction_seal=prediction_seal,
    )


def verify_i32_graph_binding_v53(
    *,
    workspace: StageWorkspaceV50,
    result: I32GraphBindingResultV53,
) -> bool:
    try:
        binding = result.binding
        if (
            not workspace.verify()
            or workspace.spec.spec_hash != binding.workspace_spec_hash
            or workspace.spec.graph_id != binding.graph_id
            or workspace.current_gate("S4") != binding.s4_gate_hash
            or not workspace.verify_prediction_seal(result.prediction_seal)
            or result.prediction_seal.seal_hash != binding.prediction_seal_hash
        ):
            return False
        committed = workspace._committed_artifact_hashes()
        required = {
            binding.forecast_bundle_artifact_hash,
            binding.custody_attestation_artifact_hash,
            binding.custody_verification_artifact_hash,
            binding.prediction_registration_artifact_hash,
            result.binding_artifact.sha256,
        }
        return required.issubset(committed)
    except (OSError, RuntimeError, ValueError, StageWorkspaceError):
        return False


__all__ = [
    "I32GraphBindingResultV53",
    "I32GraphBindingV53",
    "PredictionRegistrationV53",
    "PredictionRegistryEventV53",
    "PublicPredictionRegistryV53",
    "bind_i32_public_campaign_to_v5_v53",
    "verify_i32_graph_binding_v53",
]
