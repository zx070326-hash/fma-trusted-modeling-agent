"""Authenticated gold-stage injection and executable mechanism ablations."""

from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class GoldFileV51(StrictModel):
    relative_path: Annotated[str, Field(min_length=1, max_length=300)]
    content_base64: Annotated[str, Field(min_length=1)]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_file(self) -> "GoldFileV51":
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or self.relative_path != path.as_posix()
        ):
            raise ValueError("gold file path must be safe canonical POSIX-relative")
        payload = base64.b64decode(self.content_base64, validate=True)
        if hashlib.sha256(payload).hexdigest() != self.content_sha256:
            raise ValueError("gold file content hash differs")
        return self


class GoldStagePackageV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    package_id: Identifier
    task_id: Identifier
    protocol_hash: Sha256
    through_stage: Literal["S0", "S1", "S2", "S3", "S4", "S5"]
    predecessor_package_hash: Sha256 | None
    files: Annotated[list[GoldFileV51], Field(min_length=1)]
    private_acceptance_data_included: Literal[False] = False
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    package_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_package(self) -> "GoldStagePackageV51":
        paths = [item.relative_path for item in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("gold file paths must be sorted and unique")
        if self.authority_auth_tag and not self.package_hash:
            raise ValueError("authenticated gold package requires package_hash")
        if self.package_hash and self.package_hash != self.content_hash():
            raise ValueError("package_hash does not match gold package")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "package_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "package_hash")


class GoldInjectionReceiptV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    package_hash: Sha256
    task_id: Identifier
    through_stage: Literal["S0", "S1", "S2", "S3", "S4", "S5"]
    target_root_hash_before: Sha256
    injected_file_hashes: dict[str, Sha256]
    target_root_hash_after: Sha256
    private_acceptance_data_injected: Literal[False] = False
    injected_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "GoldInjectionReceiptV51":
        _assert_timezone(self.injected_at, "injected_at")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match gold injection")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "GoldInjectionReceiptV51":
        data.setdefault("injected_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class GoldAuthorityV51:
    """External HMAC authority; key material is never serialized."""

    def __init__(self, key_id: str, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("gold authority secret must contain at least 32 bytes")
        self.key_id = key_id
        self._secret = bytes(secret)

    def _mac(self, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            f"gold_stage_package_v51:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def seal_package(self, **data: object) -> GoldStagePackageV51:
        data["authority_key_id"] = self.key_id
        unsigned = GoldStagePackageV51(**data)
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(unsigned.unsigned_hash())
        payload["package_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "package_hash"}
        )
        return GoldStagePackageV51.model_validate(payload)

    def verify(self, package: GoldStagePackageV51) -> bool:
        try:
            GoldStagePackageV51.model_validate(package.model_dump(mode="json"))
        except ValueError:
            return False
        return (
            package.authority_key_id == self.key_id
            and package.package_hash == package.content_hash()
            and package.authority_auth_tag is not None
            and hmac.compare_digest(
                package.authority_auth_tag,
                self._mac(package.unsigned_hash()),
            )
        )


def _tree_hash(root: Path) -> str:
    if not root.exists():
        return sha256_value([])
    rows: list[dict[str, str | int]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    return sha256_value(rows)


def inject_gold_stage_v51(
    package: GoldStagePackageV51,
    *,
    authority: GoldAuthorityV51,
    target_root: str | Path,
) -> GoldInjectionReceiptV51:
    """Inject an authenticated prefix into an empty, isolated task root."""

    if not authority.verify(package):
        raise PermissionError("gold package authentication failed")
    root = Path(target_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("gold injection target must be empty")
    before = _tree_hash(root)
    injected: dict[str, str] = {}
    for item in package.files:
        target = (root / Path(*PurePosixPath(item.relative_path).parts)).resolve()
        if root not in target.parents:
            raise ValueError("gold injection target escaped task root")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = base64.b64decode(item.content_base64, validate=True)
        target.write_bytes(payload)
        injected[item.relative_path] = hashlib.sha256(payload).hexdigest()
    return GoldInjectionReceiptV51.seal(
        package_hash=package.package_hash,
        task_id=package.task_id,
        through_stage=package.through_stage,
        target_root_hash_before=before,
        injected_file_hashes=dict(sorted(injected.items())),
        target_root_hash_after=_tree_hash(root),
    )


class MechanismProfileV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    competition: bool = True
    independent_review: bool = True
    backward_revision: bool = True
    scientific_adapters: bool = True
    gold_through_stage: Literal["NONE", "S0", "S1", "S2", "S3", "S4", "S5"] = (
        "NONE"
    )


class NuisanceIdentityV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    task_hash: Sha256
    development_data_hash: Sha256
    candidate_registry_hash: Sha256
    role_prompt_pack_hash: Sha256
    requested_model: str | None
    seed: int
    maximum_role_calls: Annotated[int, Field(ge=1)]
    maximum_input_tokens: Annotated[int, Field(ge=1)]
    wall_time_limit_seconds: Annotated[int, Field(ge=1)]


class MechanismRunReceiptV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    run_id: Identifier
    nuisance_identity: NuisanceIdentityV51
    profile: MechanismProfileV51
    observed_mechanism_events: list[Identifier]
    terminal_state: Literal[
        "SCIENTIFICALLY_REJECTED",
        "HOLDOUT_SCORED_NOT_QUALIFIED",
        "INTEGRITY_FAILURE",
        "NOT_RUN",
    ]
    development_score: float | None
    holdout_score: float | None
    role_call_count: Annotated[int, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    wall_time_seconds: Annotated[float, Field(ge=0)]
    output_artifact_hash: Sha256
    completed_at: datetime
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_run(self) -> "MechanismRunReceiptV51":
        _assert_timezone(self.completed_at, "completed_at")
        if self.observed_mechanism_events != sorted(
            set(self.observed_mechanism_events)
        ):
            raise ValueError("observed mechanism events must be sorted and unique")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match mechanism run")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "MechanismRunReceiptV51":
        data.setdefault("completed_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class AblationComparisonV51(StrictModel):
    schema_version: Literal["5.1"] = "5.1"
    mechanism_id: Literal[
        "competition",
        "independent_review",
        "backward_revision",
        "scientific_adapters",
    ]
    control_run_receipt_hash: Sha256
    treatment_run_receipt_hash: Sha256
    nuisance_identity_hash: Sha256
    exactly_one_mechanism_changed: bool
    observed_execution_path_delta: bool
    no_op_detected: bool
    valid_ablation: bool
    development_score_delta: float | None
    holdout_score_delta: float | None
    causal_claim_permitted: Literal[False] = False
    reasons: list[str]
    comparison_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_comparison(self) -> "AblationComparisonV51":
        if self.valid_ablation and (
            not self.exactly_one_mechanism_changed
            or not self.observed_execution_path_delta
            or self.no_op_detected
            or self.reasons
        ):
            raise ValueError("valid ablation contains an invalidity condition")
        if self.comparison_hash and self.comparison_hash != self.content_hash():
            raise ValueError("comparison_hash does not match ablation comparison")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "comparison_hash")

    @classmethod
    def seal(cls, **data: object) -> "AblationComparisonV51":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"comparison_hash"}),
            comparison_hash=draft.content_hash(),
        )


def compare_ablation_runs_v51(
    control: MechanismRunReceiptV51,
    treatment: MechanismRunReceiptV51,
    *,
    mechanism_id: Literal[
        "competition",
        "independent_review",
        "backward_revision",
        "scientific_adapters",
    ],
) -> AblationComparisonV51:
    nuisance_equal = control.nuisance_identity == treatment.nuisance_identity
    control_profile = control.profile.model_dump(mode="json")
    treatment_profile = treatment.profile.model_dump(mode="json")
    changed = sorted(
        key
        for key in control_profile
        if key != "schema_version"
        and control_profile[key] != treatment_profile[key]
    )
    exactly_one = changed == [mechanism_id]
    event_name = f"{mechanism_id}_executed"
    expected_control_event = bool(control_profile[mechanism_id])
    expected_treatment_event = bool(treatment_profile[mechanism_id])
    observed_control_event = event_name in control.observed_mechanism_events
    observed_treatment_event = event_name in treatment.observed_mechanism_events
    path_delta = (
        expected_control_event != expected_treatment_event
        and observed_control_event == expected_control_event
        and observed_treatment_event == expected_treatment_event
        and (
            control.observed_mechanism_events
            != treatment.observed_mechanism_events
            or control.output_artifact_hash != treatment.output_artifact_hash
        )
    )
    no_op = exactly_one and not path_delta
    reasons: list[str] = []
    if not nuisance_equal:
        reasons.append("nuisance_identity_differs")
    if not exactly_one:
        reasons.append("not_exactly_one_mechanism_changed")
    if no_op:
        reasons.append("declared_toggle_was_no_op")

    def delta(first: float | None, second: float | None) -> float | None:
        return None if first is None or second is None else second - first

    return AblationComparisonV51.seal(
        mechanism_id=mechanism_id,
        control_run_receipt_hash=control.receipt_hash,
        treatment_run_receipt_hash=treatment.receipt_hash,
        nuisance_identity_hash=sha256_value(control.nuisance_identity),
        exactly_one_mechanism_changed=exactly_one,
        observed_execution_path_delta=path_delta,
        no_op_detected=no_op,
        valid_ablation=nuisance_equal and exactly_one and path_delta and not no_op,
        development_score_delta=delta(
            control.development_score, treatment.development_score
        ),
        holdout_score_delta=delta(control.holdout_score, treatment.holdout_score),
        reasons=sorted(reasons),
    )


__all__ = [
    "AblationComparisonV51",
    "GoldAuthorityV51",
    "GoldFileV51",
    "GoldInjectionReceiptV51",
    "GoldStagePackageV51",
    "MechanismProfileV51",
    "MechanismRunReceiptV51",
    "NuisanceIdentityV51",
    "compare_ablation_runs_v51",
    "inject_gold_stage_v51",
]
