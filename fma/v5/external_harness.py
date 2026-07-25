from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from fma._file_lock import exclusive_file_lock
from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
SECRECY_MODE = "logical_projection_plus_canary"
PREDICTION_RELATIVE_PATH = Path("predictions") / "registered.json"
SUPPORTED_MECHANISMS = frozenset(
    {
        "BUILDPAPER",
        "CHECKS_L3_L4",
        "COMPETE",
        "ENSEMBLE",
        "GATE",
        "REDTEAM",
        "SKILLS",
    }
)


class HarnessProtocolError(RuntimeError):
    """The caller attempted a transition forbidden by the H0 protocol."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _assert_sealed(model: StrictModel, field: str, expected: str) -> None:
    if not expected or expected != _hash_without(model, field):
        raise ValueError(f"{model.__class__.__name__} is not sealed")


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


class PublicCaseSpecV50(StrictModel):
    """Only this case projection may enter the task workspace."""

    schema_version: Literal["5.0"] = "5.0"
    case_id: Identifier
    title: Annotated[str, Field(min_length=3)]
    objective: Annotated[str, Field(min_length=5)]
    public_payload: dict[str, Any]
    supported_mechanisms: list[Identifier] = Field(default_factory=list)
    case_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "PublicCaseSpecV50":
        if self.supported_mechanisms != sorted(set(self.supported_mechanisms)):
            raise ValueError("supported_mechanisms must be sorted and unique")
        unknown = set(self.supported_mechanisms) - SUPPORTED_MECHANISMS
        if unknown:
            raise ValueError(f"unsupported mechanisms declared: {sorted(unknown)}")
        if self.case_hash and self.case_hash != self.content_hash():
            raise ValueError("case_hash does not match public case content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "case_hash")

    def assert_sealed(self) -> None:
        _assert_sealed(self, "case_hash", str(self.case_hash or ""))

    @classmethod
    def seal(cls, **data: object) -> "PublicCaseSpecV50":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"case_hash"}),
            case_hash=draft.content_hash(),
        )


class PrivateTargetV50(StrictModel):
    target_id: Identifier
    value: FiniteNumber


class PrivateCaseCapsuleV50(StrictModel):
    """Private evaluation state stored only below the external harness root."""

    schema_version: Literal["5.0-private"] = "5.0-private"
    case_id: Identifier
    public_case_hash: Sha256
    holdout: Annotated[list[PrivateTargetV50], Field(min_length=1)]
    quality_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    secrecy_canary: Annotated[str, Field(min_length=16)]
    capsule_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_capsule(self) -> "PrivateCaseCapsuleV50":
        target_ids = [target.target_id for target in self.holdout]
        if target_ids != sorted(set(target_ids)):
            raise ValueError("private target IDs must be sorted and unique")
        if self.capsule_hash and self.capsule_hash != self.content_hash():
            raise ValueError("capsule_hash does not match private capsule content")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "capsule_hash")

    def assert_sealed(self) -> None:
        _assert_sealed(self, "capsule_hash", str(self.capsule_hash or ""))

    @classmethod
    def seal(cls, **data: object) -> "PrivateCaseCapsuleV50":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"capsule_hash"}),
            capsule_hash=draft.content_hash(),
        )


class PredictionPointV50(StrictModel):
    target_id: Identifier
    value: FiniteNumber


class PredictionDocumentV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    case_id: Identifier
    predictions: Annotated[list[PredictionPointV50], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_predictions(self) -> "PredictionDocumentV50":
        target_ids = [point.target_id for point in self.predictions]
        if target_ids != sorted(set(target_ids)):
            raise ValueError("prediction target IDs must be sorted and unique")
        return self


class PreparationReceiptV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    case_id: Identifier
    workspace_path: Annotated[str, Field(min_length=1)]
    public_case_hash: Sha256
    private_capsule_hash: Sha256
    public_workspace_relative_path: Literal["problem/public_case.json"] = (
        "problem/public_case.json"
    )
    secrecy_mode: Literal["logical_projection_plus_canary"] = SECRECY_MODE
    canary_absent_at_preparation: Literal[True] = True
    host_secrecy_attested: Literal[False] = False
    capability_claim_permitted: Literal[False] = False
    prepared_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "PreparationReceiptV50":
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match preparation receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        _assert_sealed(self, "receipt_hash", str(self.receipt_hash or ""))

    @classmethod
    def seal(cls, **data: object) -> "PreparationReceiptV50":
        data.setdefault("prepared_at", _utc_now())
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class PredictionRegistrationV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    case_id: Identifier
    public_case_hash: Sha256
    prediction_source_relative_path: Literal["predictions/registered.json"] = (
        "predictions/registered.json"
    )
    source_bytes_hash: Sha256
    prediction_semantic_hash: Sha256
    snapshot_bytes_hash: Sha256
    snapshot_relative_path: Annotated[str, Field(min_length=1)]
    secrecy_mode: Literal["logical_projection_plus_canary"] = SECRECY_MODE
    private_holdout_accessed_before_registration: Literal[False] = False
    host_secrecy_attested: Literal[False] = False
    capability_claim_permitted: Literal[False] = False
    registered_at: datetime
    registration_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_registration(self) -> "PredictionRegistrationV50":
        if self.source_bytes_hash != self.snapshot_bytes_hash:
            raise ValueError("registered source and frozen snapshot hashes differ")
        if (
            self.registration_hash
            and self.registration_hash != self.content_hash()
        ):
            raise ValueError("registration_hash does not match registration")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "registration_hash")

    def assert_sealed(self) -> None:
        _assert_sealed(
            self, "registration_hash", str(self.registration_hash or "")
        )

    @classmethod
    def seal(cls, **data: object) -> "PredictionRegistrationV50":
        data.setdefault("registered_at", _utc_now())
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"registration_hash"}),
            registration_hash=draft.content_hash(),
        )


class PrivateHoldoutRevealV50(StrictModel):
    """A private return value; it is never serialized into the workspace."""

    schema_version: Literal["5.0-private-reveal"] = "5.0-private-reveal"
    case_id: Identifier
    registration_hash: Sha256
    holdout: Annotated[list[PrivateTargetV50], Field(min_length=1)]
    revealed_at: datetime


class ScoreReportV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    case_id: Identifier
    registration_hash: Sha256
    scored_snapshot_bytes_hash: Sha256
    scored_prediction_semantic_hash: Sha256
    current_source_bytes_hash: Sha256 | None = None
    score_source: Literal["frozen_content_addressed_snapshot"] = (
        "frozen_content_addressed_snapshot"
    )
    metric: Literal["mean_absolute_error"] = "mean_absolute_error"
    raw_mae: FiniteNumber | None = None
    quality_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    integrity_valid: bool
    prediction_domain_valid: bool
    source_mutated_after_registration: bool
    public_projection_unchanged: bool
    canary_leak_detected: bool
    reasons: list[Annotated[str, Field(min_length=1)]]
    secrecy_mode: Literal["logical_projection_plus_canary"] = SECRECY_MODE
    host_secrecy_attested: Literal[False] = False
    capability_claim_permitted: Literal[False] = False
    scored_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "ScoreReportV50":
        if not self.integrity_valid and self.quality_score != 0:
            raise ValueError("invalid integrity requires quality_score=0")
        if not self.prediction_domain_valid and self.quality_score != 0:
            raise ValueError("invalid prediction domain requires quality_score=0")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match score report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        _assert_sealed(self, "report_hash", str(self.report_hash or ""))

    @classmethod
    def seal(cls, **data: object) -> "ScoreReportV50":
        data.setdefault("scored_at", _utc_now())
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class MechanismArmV50(StrictModel):
    mechanism_id: Identifier
    enabled: bool
    implementation_hash: Sha256
    behavior_fingerprint: Sha256
    run_receipt_hash: Sha256


class AblationAssessmentV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    case_id: Identifier
    mechanism_id: Identifier
    valid_ablation: bool
    unsupported_mechanism: bool
    no_op_detected: bool
    reasons: list[Annotated[str, Field(min_length=1)]]
    control_run_receipt_hash: Sha256
    treatment_run_receipt_hash: Sha256
    capability_claim_permitted: Literal[False] = False
    assessed_at: datetime
    assessment_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> "AblationAssessmentV50":
        if self.valid_ablation and (
            self.unsupported_mechanism or self.no_op_detected or self.reasons
        ):
            raise ValueError("a valid ablation cannot contain invalidity reasons")
        if (
            self.assessment_hash
            and self.assessment_hash != self.content_hash()
        ):
            raise ValueError("assessment_hash does not match ablation assessment")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "assessment_hash")

    def assert_sealed(self) -> None:
        _assert_sealed(
            self, "assessment_hash", str(self.assessment_hash or "")
        )

    @classmethod
    def seal(cls, **data: object) -> "AblationAssessmentV50":
        data.setdefault("assessed_at", _utc_now())
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"assessment_hash"}),
            assessment_hash=draft.content_hash(),
        )


class HarnessEventV50(StrictModel):
    schema_version: Literal["5.0"] = "5.0"
    sequence: Annotated[int, Field(ge=1)]
    event_type: Literal[
        "case_prepared",
        "prediction_registered",
        "holdout_revealed",
        "prediction_scored",
        "ablation_assessed",
    ]
    case_id: Identifier
    payload: dict[str, Any]
    previous_event_hash: Sha256 | None = None
    created_at: datetime
    event_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_event(self) -> "HarnessEventV50":
        if self.sequence == 1 and self.previous_event_hash is not None:
            raise ValueError("first event cannot have a predecessor")
        if self.sequence > 1 and self.previous_event_hash is None:
            raise ValueError("non-first event needs a predecessor")
        if self.event_hash and self.event_hash != self.content_hash():
            raise ValueError("event_hash does not match event")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "event_hash")

    def assert_sealed(self) -> None:
        _assert_sealed(self, "event_hash", str(self.event_hash or ""))

    @classmethod
    def seal(cls, **data: object) -> "HarnessEventV50":
        data.setdefault("created_at", _utc_now())
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"event_hash"}),
            event_hash=draft.content_hash(),
        )


class ExternalHarnessV50:
    """External, fail-closed H0 evaluator.

    This is a logical information-flow boundary with a leak canary, not an OS
    sandbox.  Accordingly every receipt keeps host and capability claims false.
    """

    def __init__(self, harness_root: str | Path) -> None:
        self.root = Path(harness_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.event_path = self.root / "events.jsonl"
        self._writer_lock_path = self.root / ".events.writer.lock"

    def prepare_case(
        self,
        public_case: PublicCaseSpecV50,
        private_capsule: PrivateCaseCapsuleV50,
        workspace: str | Path,
    ) -> PreparationReceiptV50:
        if not self.verify_event_chain():
            raise HarnessProtocolError(
                "refusing to prepare a case on an invalid event chain"
            )
        public_case.assert_sealed()
        private_capsule.assert_sealed()
        if private_capsule.case_id != public_case.case_id:
            raise ValueError("public and private case IDs differ")
        if private_capsule.public_case_hash != public_case.case_hash:
            raise ValueError("private capsule is bound to another public case")

        workspace_path = Path(workspace).resolve()
        if _paths_overlap(self.root, workspace_path):
            raise ValueError("harness root and task workspace must not overlap")
        if workspace_path.exists():
            raise HarnessProtocolError("prepare_case requires a fresh workspace path")
        case_dir = self._case_dir(public_case.case_id)
        if case_dir.exists():
            raise HarnessProtocolError("case was already prepared")

        public_bytes = canonical_json(public_case).encode("utf-8")
        canary_bytes = private_capsule.secrecy_canary.encode("utf-8")
        if canary_bytes in public_bytes:
            raise ValueError("private secrecy canary appears in public projection")

        (case_dir / "public").mkdir(parents=True)
        (case_dir / "private").mkdir()
        (case_dir / "control").mkdir()
        (case_dir / "snapshots" / "predictions").mkdir(parents=True)
        (case_dir / "scores").mkdir()
        (case_dir / "ablations").mkdir()
        (workspace_path / "problem").mkdir(parents=True)
        (workspace_path / "predictions").mkdir()

        self._write_json(case_dir / "public" / "case.json", public_case)
        self._write_json(
            case_dir / "private" / "capsule.json", private_capsule
        )
        self._write_json(
            workspace_path / "problem" / "public_case.json", public_case
        )
        canary_absent = not self._workspace_contains(
            workspace_path, canary_bytes
        )
        if not canary_absent:
            raise HarnessProtocolError("private canary leaked during preparation")

        receipt = PreparationReceiptV50.seal(
            case_id=public_case.case_id,
            workspace_path=str(workspace_path),
            public_case_hash=public_case.case_hash,
            private_capsule_hash=private_capsule.capsule_hash,
        )
        self._write_json(case_dir / "control" / "preparation.json", receipt)
        self._append_event(
            "case_prepared",
            public_case.case_id,
            {
                "preparation_receipt_hash": receipt.receipt_hash,
                "public_case_hash": public_case.case_hash,
                "private_capsule_hash": private_capsule.capsule_hash,
                "secrecy_mode": SECRECY_MODE,
                "host_secrecy_attested": False,
                "capability_claim_permitted": False,
            },
        )
        return receipt

    def register_prediction(self, case_id: str) -> PredictionRegistrationV50:
        receipt, public_case, private_capsule = self._load_case(case_id)
        case_dir = self._case_dir(case_id)
        registration_path = case_dir / "control" / "registration.json"
        if registration_path.exists():
            raise HarnessProtocolError("prediction registration is immutable and unique")

        workspace = Path(receipt.workspace_path)
        self._assert_public_projection_unchanged(workspace, public_case)
        if self._workspace_contains(
            workspace, private_capsule.secrecy_canary.encode("utf-8")
        ):
            raise HarnessProtocolError("private canary is present in task workspace")

        source_path = workspace / PREDICTION_RELATIVE_PATH
        try:
            raw = source_path.read_bytes()
        except FileNotFoundError as exc:
            raise HarnessProtocolError(
                "predictions/registered.json does not exist"
            ) from exc
        if len(raw) > 1_048_576:
            raise HarnessProtocolError("prediction document exceeds 1 MiB")
        try:
            prediction = PredictionDocumentV50.model_validate_json(raw)
        except Exception as exc:
            raise HarnessProtocolError("prediction document is invalid") from exc
        if prediction.case_id != case_id:
            raise HarnessProtocolError("prediction belongs to another case")

        bytes_hash = _sha256_bytes(raw)
        semantic_hash = sha256_value(prediction)
        snapshot_path = (
            case_dir / "snapshots" / "predictions" / f"{bytes_hash}.json"
        )
        try:
            with snapshot_path.open("xb") as handle:
                handle.write(raw)
        except FileExistsError as exc:
            raise HarnessProtocolError("prediction snapshot already exists") from exc

        registration = PredictionRegistrationV50.seal(
            case_id=case_id,
            public_case_hash=public_case.case_hash,
            source_bytes_hash=bytes_hash,
            prediction_semantic_hash=semantic_hash,
            snapshot_bytes_hash=bytes_hash,
            snapshot_relative_path=str(snapshot_path.relative_to(self.root)),
        )
        self._write_json(registration_path, registration)
        self._append_event(
            "prediction_registered",
            case_id,
            {
                "registration_hash": registration.registration_hash,
                "snapshot_bytes_hash": bytes_hash,
                "prediction_semantic_hash": semantic_hash,
            },
        )
        return registration

    def reveal_holdout(self, case_id: str) -> PrivateHoldoutRevealV50:
        _, _, private_capsule = self._load_case(case_id)
        registration = self._load_registration(case_id)
        reveal = PrivateHoldoutRevealV50(
            case_id=case_id,
            registration_hash=registration.registration_hash,
            holdout=private_capsule.holdout,
            revealed_at=_utc_now(),
        )
        self._append_event(
            "holdout_revealed",
            case_id,
            {
                "registration_hash": registration.registration_hash,
                "private_reveal_hash": sha256_value(reveal),
            },
        )
        return reveal

    def score(self, case_id: str) -> ScoreReportV50:
        receipt, public_case, private_capsule = self._load_case(case_id)
        registration = self._load_registration(case_id)
        workspace = Path(receipt.workspace_path)

        snapshot_path = self.root / registration.snapshot_relative_path
        try:
            snapshot_bytes = snapshot_path.read_bytes()
        except FileNotFoundError as exc:
            raise HarnessProtocolError("registered snapshot is missing") from exc
        if _sha256_bytes(snapshot_bytes) != registration.snapshot_bytes_hash:
            raise HarnessProtocolError("registered prediction snapshot was altered")
        try:
            frozen_prediction = PredictionDocumentV50.model_validate_json(
                snapshot_bytes
            )
        except Exception as exc:
            raise HarnessProtocolError("registered snapshot is invalid") from exc
        if sha256_value(frozen_prediction) != registration.prediction_semantic_hash:
            raise HarnessProtocolError("registered prediction semantics changed")

        source_path = workspace / PREDICTION_RELATIVE_PATH
        try:
            current_source_hash = (
                _sha256_bytes(source_path.read_bytes())
                if source_path.is_file()
                else None
            )
        except OSError:
            current_source_hash = None
        source_mutated = current_source_hash != registration.source_bytes_hash
        public_unchanged = self._public_projection_unchanged(
            workspace, public_case
        )
        canary_leaked = self._workspace_contains(
            workspace, private_capsule.secrecy_canary.encode("utf-8")
        )

        prediction_by_id = {
            point.target_id: point.value for point in frozen_prediction.predictions
        }
        target_by_id = {
            target.target_id: target.value for target in private_capsule.holdout
        }
        domain_valid = (
            frozen_prediction.case_id == case_id
            and set(prediction_by_id) == set(target_by_id)
        )
        raw_mae: float | None = None
        if domain_valid:
            raw_mae = sum(
                abs(prediction_by_id[target_id] - target_by_id[target_id])
                for target_id in sorted(target_by_id)
            ) / len(target_by_id)

        reasons: list[str] = []
        if source_mutated:
            reasons.append("prediction_source_mutated_after_registration")
        if not public_unchanged:
            reasons.append("public_projection_changed_after_preparation")
        if canary_leaked:
            reasons.append("private_canary_detected_in_workspace")
        if not domain_valid:
            reasons.append("frozen_prediction_domain_does_not_match_holdout")
        integrity_valid = not source_mutated and public_unchanged and not canary_leaked
        if integrity_valid and domain_valid and raw_mae is not None:
            quality_score = max(
                0.0, min(1.0, 1.0 - raw_mae / private_capsule.quality_scale)
            )
        else:
            quality_score = 0.0

        report = ScoreReportV50.seal(
            case_id=case_id,
            registration_hash=registration.registration_hash,
            scored_snapshot_bytes_hash=registration.snapshot_bytes_hash,
            scored_prediction_semantic_hash=registration.prediction_semantic_hash,
            current_source_bytes_hash=current_source_hash,
            raw_mae=raw_mae,
            quality_score=quality_score,
            integrity_valid=integrity_valid,
            prediction_domain_valid=domain_valid,
            source_mutated_after_registration=source_mutated,
            public_projection_unchanged=public_unchanged,
            canary_leak_detected=canary_leaked,
            reasons=reasons,
        )
        self._write_json(
            self._case_dir(case_id) / "scores" / f"{report.report_hash}.json",
            report,
        )
        self._append_event(
            "prediction_scored",
            case_id,
            {
                "registration_hash": registration.registration_hash,
                "score_report_hash": report.report_hash,
                "integrity_valid": report.integrity_valid,
                "quality_score": report.quality_score,
            },
        )
        return report

    def assess_ablation(
        self,
        case_id: str,
        control: MechanismArmV50,
        treatment: MechanismArmV50,
    ) -> AblationAssessmentV50:
        _, public_case, _ = self._load_case(case_id)
        reasons: list[str] = []
        same_mechanism = control.mechanism_id == treatment.mechanism_id
        mechanism_id = control.mechanism_id
        if not same_mechanism:
            reasons.append("arms_name_different_mechanisms")
        supported = (
            same_mechanism
            and mechanism_id in SUPPORTED_MECHANISMS
            and mechanism_id in public_case.supported_mechanisms
        )
        if not supported:
            reasons.append("mechanism_not_supported_for_case")
        if control.enabled == treatment.enabled:
            reasons.append("mechanism_state_not_flipped")
        no_op = control.behavior_fingerprint == treatment.behavior_fingerprint
        if no_op:
            reasons.append("behavior_fingerprint_unchanged_no_op")
        if control.run_receipt_hash == treatment.run_receipt_hash:
            reasons.append("arms_reference_same_run_receipt")
        if control.implementation_hash != treatment.implementation_hash:
            reasons.append("implementation_hash_differs_confound")
        # V5.0 can validate an arm declaration and detect obvious no-ops, but
        # it does not yet execute both mechanism assignments through a bound
        # runner.  Caller-supplied hashes are therefore never sufficient to
        # certify a causal mechanism ablation.
        reasons.append("mechanism_runtime_not_bound_to_receipts")

        assessment = AblationAssessmentV50.seal(
            case_id=case_id,
            mechanism_id=mechanism_id,
            valid_ablation=False,
            unsupported_mechanism=not supported,
            no_op_detected=no_op,
            reasons=reasons,
            control_run_receipt_hash=control.run_receipt_hash,
            treatment_run_receipt_hash=treatment.run_receipt_hash,
        )
        self._write_json(
            self._case_dir(case_id)
            / "ablations"
            / f"{assessment.assessment_hash}.json",
            assessment,
        )
        self._append_event(
            "ablation_assessed",
            case_id,
            {
                "ablation_assessment_hash": assessment.assessment_hash,
                "mechanism_id": mechanism_id,
                "valid_ablation": assessment.valid_ablation,
            },
        )
        return assessment

    def verify_event_chain(self) -> bool:
        try:
            self._validated_events()
        except (HarnessProtocolError, OSError, ValueError):
            return False
        return True

    def _validated_events(self) -> list[HarnessEventV50]:
        if not self.event_path.exists():
            return []
        serialized = self.event_path.read_text(encoding="utf-8")
        if serialized and not serialized.endswith("\n"):
            raise HarnessProtocolError("harness event chain ends with a partial record")
        previous_hash: str | None = None
        expected_sequence = 1
        events: list[HarnessEventV50] = []
        for line in serialized.splitlines():
            if not line.strip():
                raise HarnessProtocolError(
                    "harness event chain contains a blank record"
                )
            event = HarnessEventV50.model_validate_json(line)
            event.assert_sealed()
            if event.sequence != expected_sequence:
                raise HarnessProtocolError(
                    "harness event sequence is not contiguous"
                )
            if event.previous_event_hash != previous_hash:
                raise HarnessProtocolError(
                    "harness event predecessor hash does not match"
                )
            events.append(event)
            previous_hash = event.event_hash
            expected_sequence += 1
        return events

    def _load_case(
        self, case_id: str
    ) -> tuple[
        PreparationReceiptV50, PublicCaseSpecV50, PrivateCaseCapsuleV50
    ]:
        if not self.verify_event_chain():
            raise HarnessProtocolError("harness event chain is invalid")
        case_dir = self._case_dir(case_id)
        try:
            receipt = PreparationReceiptV50.model_validate_json(
                (case_dir / "control" / "preparation.json").read_bytes()
            )
            public_case = PublicCaseSpecV50.model_validate_json(
                (case_dir / "public" / "case.json").read_bytes()
            )
            private_capsule = PrivateCaseCapsuleV50.model_validate_json(
                (case_dir / "private" / "capsule.json").read_bytes()
            )
        except FileNotFoundError as exc:
            raise HarnessProtocolError("case was not prepared") from exc
        receipt.assert_sealed()
        public_case.assert_sealed()
        private_capsule.assert_sealed()
        if (
            receipt.case_id != case_id
            or public_case.case_id != case_id
            or private_capsule.case_id != case_id
        ):
            raise HarnessProtocolError("case control files disagree")
        if receipt.public_case_hash != public_case.case_hash:
            raise HarnessProtocolError("prepared public case changed")
        if receipt.private_capsule_hash != private_capsule.capsule_hash:
            raise HarnessProtocolError("prepared private capsule changed")
        if not self._event_references(
            "case_prepared", case_id, "preparation_receipt_hash", receipt.receipt_hash
        ):
            raise HarnessProtocolError("preparation is absent from event chain")
        return receipt, public_case, private_capsule

    def _load_registration(
        self, case_id: str
    ) -> PredictionRegistrationV50:
        path = self._case_dir(case_id) / "control" / "registration.json"
        try:
            registration = PredictionRegistrationV50.model_validate_json(
                path.read_bytes()
            )
        except FileNotFoundError as exc:
            raise HarnessProtocolError(
                "holdout access and scoring require prior registration"
            ) from exc
        registration.assert_sealed()
        if registration.case_id != case_id:
            raise HarnessProtocolError("registration belongs to another case")
        if not self._event_references(
            "prediction_registered",
            case_id,
            "registration_hash",
            registration.registration_hash,
        ):
            raise HarnessProtocolError("registration is absent from event chain")
        return registration

    def _append_event(
        self,
        event_type: Literal[
            "case_prepared",
            "prediction_registered",
            "holdout_revealed",
            "prediction_scored",
            "ablation_assessed",
        ],
        case_id: str,
        payload: dict[str, Any],
    ) -> HarnessEventV50:
        with exclusive_file_lock(self._writer_lock_path):
            try:
                existing = self._validated_events()
            except (OSError, ValueError) as exc:
                raise HarnessProtocolError(
                    "refusing to append to an invalid event chain"
                ) from exc
            previous_hash = existing[-1].event_hash if existing else None
            event = HarnessEventV50.seal(
                sequence=len(existing) + 1,
                event_type=event_type,
                case_id=case_id,
                payload=payload,
                previous_event_hash=previous_hash,
            )
            with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(event))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def _read_events(self) -> list[HarnessEventV50]:
        return self._validated_events()

    def _event_references(
        self,
        event_type: str,
        case_id: str,
        key: str,
        value: str | None,
    ) -> bool:
        return any(
            event.event_type == event_type
            and event.case_id == case_id
            and event.payload.get(key) == value
            for event in self._read_events()
        )

    def _case_dir(self, case_id: str) -> Path:
        if not case_id or not case_id[0].isalpha() or any(
            not (character.isalnum() or character in "_.-")
            for character in case_id
        ):
            raise ValueError("invalid case_id")
        return self.root / "cases" / case_id

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    @staticmethod
    def _workspace_contains(workspace: Path, needle: bytes) -> bool:
        if not workspace.exists():
            return False
        for path in workspace.rglob("*"):
            if path.is_file():
                try:
                    if needle in path.read_bytes():
                        return True
                except OSError:
                    return True
        return False

    @staticmethod
    def _public_projection_unchanged(
        workspace: Path, expected: PublicCaseSpecV50
    ) -> bool:
        path = workspace / "problem" / "public_case.json"
        try:
            current = PublicCaseSpecV50.model_validate_json(path.read_bytes())
            current.assert_sealed()
        except (OSError, ValueError):
            return False
        return current.case_hash == expected.case_hash

    def _assert_public_projection_unchanged(
        self, workspace: Path, expected: PublicCaseSpecV50
    ) -> None:
        if not self._public_projection_unchanged(workspace, expected):
            raise HarnessProtocolError("public case projection changed")


__all__ = [
    "AblationAssessmentV50",
    "ExternalHarnessV50",
    "HarnessEventV50",
    "HarnessProtocolError",
    "MechanismArmV50",
    "PredictionDocumentV50",
    "PredictionPointV50",
    "PredictionRegistrationV50",
    "PreparationReceiptV50",
    "PrivateCaseCapsuleV50",
    "PrivateHoldoutRevealV50",
    "PrivateTargetV50",
    "PublicCaseSpecV50",
    "ScoreReportV50",
    "SECRECY_MODE",
    "SUPPORTED_MECHANISMS",
]
