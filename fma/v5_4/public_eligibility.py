"""Prospective public-evidence gate for private forecast evaluation.

V5.4 does not reinterpret any V5.3 artifact.  It adds a code-owned gate that
asks a narrower question before a private evaluation can be requested:

    Is the selected candidate's public advantage over the frozen baseline
    stable across origins, time blocks, and forecast horizons?

The gate operates on paired public losses only.  It cannot read private
targets, grant scientific qualification, or authorize real-world action.
"""

from __future__ import annotations

import base64
import hashlib
import math
from datetime import datetime, timezone
from typing import Annotated, Literal

import numpy as np
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
from fma.v5_3.external_private import PrivateEvaluationRequestV53


FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
EligibilityDecisionV54 = Literal["ELIGIBLE", "ABSTAIN"]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_private_key(private_key_pem: bytes) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("public eligibility authority key must be Ed25519")
    return key


def _load_public_key(public_key_pem: bytes) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("public eligibility authority public key must be Ed25519")
    return key


def public_eligibility_key_fingerprint_v54(public_key_pem: bytes) -> str:
    key = _load_public_key(public_key_pem)
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()


class PublicEligibilityContractV54(StrictModel):
    """Frozen, pre-modeling policy for public-to-private eligibility."""

    schema_version: Literal["5.4-public-eligibility-contract"] = (
        "5.4-public-eligibility-contract"
    )
    contract_id: Identifier
    task_id: Identifier
    baseline_id: Identifier
    candidate_selection_rule_hash: Sha256
    expected_horizons: Annotated[list[int], Field(min_length=1)]
    minimum_origin_count: Annotated[int, Field(ge=6)]
    contiguous_time_block_count: Annotated[int, Field(ge=2)]
    recent_origin_count: Annotated[int, Field(ge=2)]
    minimum_overall_advantage: NonNegativeFinite = 0.0
    minimum_time_block_advantage: NonNegativeFinite = 0.0
    minimum_recent_advantage: NonNegativeFinite = 0.0
    minimum_horizon_advantage: NonNegativeFinite = 0.0
    minimum_origin_win_fraction: Annotated[
        float, Field(ge=0.5, le=1.0, allow_inf_nan=False)
    ] = 0.6
    bootstrap_confidence: Annotated[
        float, Field(gt=0.8, lt=1.0, allow_inf_nan=False)
    ] = 0.95
    bootstrap_replicates: Annotated[int, Field(ge=1000)] = 8192
    bootstrap_block_length: Annotated[int, Field(ge=2)]
    multiplicity_correction_count: Annotated[int, Field(ge=1)]
    bootstrap_seed: Annotated[int, Field(ge=0, le=4_294_967_295)] = 1729
    requires_public_scientific_acceptance: Literal[True] = True
    fixture_allowed: Literal[False] = False
    frozen_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "PublicEligibilityContractV54":
        if self.expected_horizons != sorted(set(self.expected_horizons)):
            raise ValueError("expected horizons must be sorted and unique")
        if any(item < 1 for item in self.expected_horizons):
            raise ValueError("expected horizons must be positive")
        if self.minimum_origin_count < 2 * self.contiguous_time_block_count:
            raise ValueError("each contiguous time block needs at least two origins")
        if self.recent_origin_count > self.minimum_origin_count:
            raise ValueError("recent origin count exceeds minimum origin count")
        if self.bootstrap_block_length > self.minimum_origin_count:
            raise ValueError("bootstrap block length exceeds minimum origin count")
        adjusted_alpha = (1.0 - self.bootstrap_confidence) / (
            self.multiplicity_correction_count
        )
        if self.bootstrap_replicates * adjusted_alpha < 10:
            raise ValueError(
                "bootstrap replicates do not resolve the multiplicity-adjusted tail"
            )
        if self.frozen_at.utcoffset() is None:
            raise ValueError("frozen_at must be timezone-aware")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("public eligibility contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("public eligibility contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicEligibilityContractV54":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


class PairedForecastLossV54(StrictModel):
    """One paired candidate/baseline loss on the same public target."""

    origin: Annotated[int, Field(ge=1)]
    horizon: Annotated[int, Field(ge=1)]
    candidate_loss: NonNegativeFinite
    baseline_loss: NonNegativeFinite


class PublicEligibilityInputV54(StrictModel):
    """Sealed public evidence supplied to the code-owned gate."""

    schema_version: Literal["5.4-public-eligibility-input"] = (
        "5.4-public-eligibility-input"
    )
    task_id: Identifier
    contract_hash: Sha256
    candidate_id: Identifier
    baseline_id: Identifier
    candidate_search_count: Annotated[int, Field(ge=1)]
    loss_name: Literal["normalized_absolute_error"] = "normalized_absolute_error"
    public_scientific_acceptance_verified: bool
    fixture_only: bool
    source_artifact_hashes: Annotated[list[Sha256], Field(min_length=1)]
    rows: Annotated[list[PairedForecastLossV54], Field(min_length=1)]
    input_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "PublicEligibilityInputV54":
        if self.source_artifact_hashes != sorted(set(self.source_artifact_hashes)):
            raise ValueError("source artifact hashes must be sorted and unique")
        coordinates = [(item.origin, item.horizon) for item in self.rows]
        if coordinates != sorted(set(coordinates)):
            raise ValueError("paired loss coordinates must be sorted and unique")
        if self.input_hash and self.input_hash != self.content_hash():
            raise ValueError("public eligibility input hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "input_hash")

    def assert_sealed(self) -> None:
        if not self.input_hash or self.input_hash != self.content_hash():
            raise ValueError("public eligibility input is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicEligibilityInputV54":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"input_hash"})
        payload["input_hash"] = draft.content_hash()
        return cls(**payload)


class PublicEligibilityMetricsV54(StrictModel):
    origin_count: Annotated[int, Field(ge=1)]
    horizons: Annotated[list[int], Field(min_length=1)]
    paired_row_count: Annotated[int, Field(ge=1)]
    mean_advantage: FiniteNumber
    selection_adjusted_bootstrap_lower_bound: FiniteNumber
    adjusted_alpha: PositiveFinite
    origin_win_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    contiguous_time_block_means: Annotated[list[FiniteNumber], Field(min_length=1)]
    recent_mean_advantage: FiniteNumber
    horizon_mean_advantages: dict[int, FiniteNumber]
    recent_horizon_mean_advantages: dict[int, FiniteNumber]


class PublicEligibilityAssessmentV54(StrictModel):
    """Deterministic assessment; authentication is a separate harness act."""

    schema_version: Literal["5.4-public-eligibility-assessment"] = (
        "5.4-public-eligibility-assessment"
    )
    task_id: Identifier
    contract_hash: Sha256
    input_hash: Sha256
    candidate_id: Identifier
    baseline_id: Identifier
    metrics: PublicEligibilityMetricsV54
    checks: dict[Identifier, bool]
    decision: EligibilityDecisionV54
    public_gate_eligible: bool
    private_evaluation_performed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    assessment_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> "PublicEligibilityAssessmentV54":
        expected = bool(self.checks) and all(self.checks.values())
        if self.public_gate_eligible != expected:
            raise ValueError("public eligibility flag differs from checks")
        if self.decision != ("ELIGIBLE" if expected else "ABSTAIN"):
            raise ValueError("public eligibility decision differs from checks")
        if self.assessment_hash and self.assessment_hash != self.content_hash():
            raise ValueError("public eligibility assessment hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "assessment_hash")

    @classmethod
    def seal(cls, **data: object) -> "PublicEligibilityAssessmentV54":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"assessment_hash"})
        payload["assessment_hash"] = draft.content_hash()
        return cls(**payload)


class PublicEligibilityReceiptV54(StrictModel):
    """Harness-authenticated binding for one deterministic assessment."""

    schema_version: Literal["5.4-public-eligibility-receipt"] = (
        "5.4-public-eligibility-receipt"
    )
    receipt_id: Identifier
    assessment_hash: Sha256
    contract_hash: Sha256
    input_hash: Sha256
    decision: EligibilityDecisionV54
    authority_key_id: Identifier
    authority_public_key_fingerprint: Sha256
    signature_base64: Annotated[str, Field(min_length=40)] | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "PublicEligibilityReceiptV54":
        if self.receipt_hash and (
            not self.signature_base64 or self.receipt_hash != self.content_hash()
        ):
            raise ValueError("public eligibility receipt envelope differs")
        return self

    def unsigned_bytes(self) -> bytes:
        return canonical_json(
            self.model_dump(
                mode="json",
                exclude={"signature_base64", "receipt_hash"},
            )
        ).encode()

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class PublicEligibilityAuthorityV54:
    """Ed25519 signer whose private key must stay outside model context."""

    def __init__(self, *, key_id: str, private_key_pem: bytes) -> None:
        self.key_id = key_id
        self._private_key = _load_private_key(private_key_pem)
        self.public_key_pem = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.public_key_fingerprint = public_eligibility_key_fingerprint_v54(
            self.public_key_pem
        )

    def issue(
        self,
        *,
        receipt_id: str,
        assessment: PublicEligibilityAssessmentV54,
    ) -> PublicEligibilityReceiptV54:
        if not assessment.assessment_hash:
            raise ValueError("assessment must be sealed before authentication")
        unsigned = PublicEligibilityReceiptV54(
            receipt_id=receipt_id,
            assessment_hash=assessment.assessment_hash,
            contract_hash=assessment.contract_hash,
            input_hash=assessment.input_hash,
            decision=assessment.decision,
            authority_key_id=self.key_id,
            authority_public_key_fingerprint=self.public_key_fingerprint,
        )
        payload = unsigned.model_dump(mode="json")
        payload["signature_base64"] = base64.b64encode(
            self._private_key.sign(unsigned.unsigned_bytes())
        ).decode("ascii")
        authenticated = PublicEligibilityReceiptV54(**payload)
        final_payload = authenticated.model_dump(mode="json")
        final_payload["receipt_hash"] = authenticated.content_hash()
        return PublicEligibilityReceiptV54(**final_payload)

    def verify(
        self,
        *,
        assessment: PublicEligibilityAssessmentV54,
        receipt: PublicEligibilityReceiptV54,
    ) -> bool:
        return (
            receipt.authority_key_id == self.key_id
            and verify_public_eligibility_receipt_v54(
                assessment=assessment,
                receipt=receipt,
                authority_public_key_pem=self.public_key_pem,
            )
        )


def verify_public_eligibility_receipt_v54(
    *,
    assessment: PublicEligibilityAssessmentV54,
    receipt: PublicEligibilityReceiptV54,
    authority_public_key_pem: bytes,
) -> bool:
    """Verify an eligibility receipt using only a pinned public key."""

    try:
        expected_fingerprint = public_eligibility_key_fingerprint_v54(
            authority_public_key_pem
        )
        if not (
            assessment.assessment_hash
            and assessment.assessment_hash == assessment.content_hash()
            and receipt.assessment_hash == assessment.assessment_hash
            and receipt.contract_hash == assessment.contract_hash
            and receipt.input_hash == assessment.input_hash
            and receipt.decision == assessment.decision
            and receipt.authority_public_key_fingerprint == expected_fingerprint
            and receipt.signature_base64
            and receipt.receipt_hash
            and receipt.receipt_hash == receipt.content_hash()
        ):
            return False
        _load_public_key(authority_public_key_pem).verify(
            base64.b64decode(receipt.signature_base64, validate=True),
            receipt.unsigned_bytes(),
        )
        return True
    except (InvalidSignature, TypeError, ValueError):
        return False


class PrivateEvaluationAuthorizationV54(StrictModel):
    """Public, sealed authorization to present one V5.3 request to V5.4 worker."""

    schema_version: Literal["5.4-private-evaluation-authorization"] = (
        "5.4-private-evaluation-authorization"
    )
    authorization_id: Identifier
    request_hash: Sha256
    contract_hash: Sha256
    input_hash: Sha256
    assessment_hash: Sha256
    eligibility_receipt_hash: Sha256
    eligibility_authority_key_id: Identifier
    eligibility_authority_public_key_fingerprint: Sha256
    public_gate_decision: Literal["ELIGIBLE"] = "ELIGIBLE"
    private_evaluation_count_at_issue: Literal[0] = 0
    issued_at: datetime
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authorization_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_authorization(self) -> "PrivateEvaluationAuthorizationV54":
        if self.issued_at.utcoffset() is None:
            raise ValueError("authorization issued_at must be timezone-aware")
        if self.authorization_hash and self.authorization_hash != self.content_hash():
            raise ValueError("private evaluation authorization hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "authorization_hash")

    def assert_sealed(self) -> None:
        if (
            not self.authorization_hash
            or self.authorization_hash != self.content_hash()
        ):
            raise ValueError("private evaluation authorization is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateEvaluationAuthorizationV54":
        data.setdefault("issued_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"authorization_hash"})
        payload["authorization_hash"] = draft.content_hash()
        return cls(**payload)


def authorize_private_evaluation_request_v54(
    *,
    authorization_id: str,
    request: PrivateEvaluationRequestV53,
    contract: PublicEligibilityContractV54,
    evidence: PublicEligibilityInputV54,
    assessment: PublicEligibilityAssessmentV54,
    receipt: PublicEligibilityReceiptV54,
    authority_public_key_pem: bytes,
    issued_at: datetime | None = None,
) -> PrivateEvaluationAuthorizationV54:
    """Bind a V5.3 request only after a verified V5.4 ELIGIBLE decision."""

    request.assert_sealed()
    contract.assert_sealed()
    evidence.assert_sealed()
    if assessment.decision != "ELIGIBLE" or not assessment.public_gate_eligible:
        raise ValueError("private evaluation requires an eligible public assessment")
    if not verify_public_eligibility_receipt_v54(
        assessment=assessment,
        receipt=receipt,
        authority_public_key_pem=authority_public_key_pem,
    ):
        raise ValueError("public eligibility receipt is not authenticated")
    if not (
        assessment.contract_hash == contract.contract_hash == evidence.contract_hash
        and assessment.input_hash == evidence.input_hash
        and assessment.task_id == contract.task_id == evidence.task_id
    ):
        raise ValueError("public eligibility authorization bindings differ")
    return PrivateEvaluationAuthorizationV54.seal(
        authorization_id=authorization_id,
        request_hash=request.request_hash,
        contract_hash=contract.contract_hash,
        input_hash=evidence.input_hash,
        assessment_hash=assessment.assessment_hash,
        eligibility_receipt_hash=receipt.receipt_hash,
        eligibility_authority_key_id=receipt.authority_key_id,
        eligibility_authority_public_key_fingerprint=(
            receipt.authority_public_key_fingerprint
        ),
        issued_at=issued_at or _utc_now(),
    )


def verify_private_evaluation_authorization_v54(
    *,
    authorization: PrivateEvaluationAuthorizationV54,
    request: PrivateEvaluationRequestV53,
    contract: PublicEligibilityContractV54,
    evidence: PublicEligibilityInputV54,
    assessment: PublicEligibilityAssessmentV54,
    receipt: PublicEligibilityReceiptV54,
    authority_public_key_pem: bytes,
) -> bool:
    """Fail-closed verification used by a V5.4 private worker."""

    try:
        authorization.assert_sealed()
        request.assert_sealed()
        contract.assert_sealed()
        evidence.assert_sealed()
        return bool(
            assessment.decision == "ELIGIBLE"
            and assessment.public_gate_eligible
            and verify_public_eligibility_receipt_v54(
                assessment=assessment,
                receipt=receipt,
                authority_public_key_pem=authority_public_key_pem,
            )
            and authorization.request_hash == request.request_hash
            and authorization.contract_hash
            == contract.contract_hash
            == evidence.contract_hash
            == assessment.contract_hash
            == receipt.contract_hash
            and authorization.input_hash
            == evidence.input_hash
            == assessment.input_hash
            == receipt.input_hash
            and authorization.assessment_hash
            == assessment.assessment_hash
            == receipt.assessment_hash
            and authorization.eligibility_receipt_hash == receipt.receipt_hash
            and authorization.eligibility_authority_key_id == receipt.authority_key_id
            and authorization.eligibility_authority_public_key_fingerprint
            == receipt.authority_public_key_fingerprint
        )
    except (TypeError, ValueError):
        return False


def _moving_block_bootstrap_lower_bound(
    values: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    confidence: float,
    multiplicity_count: int,
    seed: int,
) -> tuple[float, float]:
    adjusted_alpha = (1.0 - confidence) / multiplicity_count
    starts = np.arange(values.size - block_length + 1)
    blocks_per_sample = math.ceil(values.size / block_length)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_starts = rng.choice(
            starts,
            size=blocks_per_sample,
            replace=True,
        )
        sampled = np.concatenate(
            [values[start : start + block_length] for start in sampled_starts]
        )[: values.size]
        means[index] = float(np.mean(sampled))
    means.sort()
    order_index = max(0, math.floor(adjusted_alpha * (replicates + 1)) - 1)
    return float(means[order_index]), adjusted_alpha


def assess_public_eligibility_v54(
    *,
    contract: PublicEligibilityContractV54,
    evidence: PublicEligibilityInputV54,
) -> PublicEligibilityAssessmentV54:
    """Assess paired public evidence without reading private outcomes."""

    contract.assert_sealed()
    evidence.assert_sealed()
    if evidence.task_id != contract.task_id:
        raise ValueError("eligibility evidence belongs to another task")
    if evidence.contract_hash != contract.contract_hash:
        raise ValueError("eligibility evidence is bound to another contract")
    if evidence.baseline_id != contract.baseline_id:
        raise ValueError("eligibility evidence uses another baseline")
    if evidence.candidate_search_count > contract.multiplicity_correction_count:
        raise ValueError("candidate search count exceeds the frozen correction budget")

    origins = sorted({row.origin for row in evidence.rows})
    horizons = sorted({row.horizon for row in evidence.rows})
    expected_coordinates = [
        (origin, horizon)
        for origin in origins
        for horizon in contract.expected_horizons
    ]
    observed_coordinates = [(row.origin, row.horizon) for row in evidence.rows]
    rectangular_coverage = (
        horizons == contract.expected_horizons
        and observed_coordinates == expected_coordinates
    )
    if not rectangular_coverage:
        raise ValueError("paired losses do not form the frozen origin-horizon grid")

    advantages = {
        (row.origin, row.horizon): row.baseline_loss - row.candidate_loss
        for row in evidence.rows
    }
    origin_advantages = np.asarray(
        [
            np.mean(
                [
                    advantages[(origin, horizon)]
                    for horizon in contract.expected_horizons
                ]
            )
            for origin in origins
        ],
        dtype=float,
    )
    block_means = [
        float(np.mean(block))
        for block in np.array_split(
            origin_advantages,
            contract.contiguous_time_block_count,
        )
    ]
    recent_origin_values = origin_advantages[-contract.recent_origin_count :]
    horizon_means = {
        horizon: float(np.mean([advantages[(origin, horizon)] for origin in origins]))
        for horizon in contract.expected_horizons
    }
    recent_origins = origins[-contract.recent_origin_count :]
    recent_horizon_means = {
        horizon: float(
            np.mean([advantages[(origin, horizon)] for origin in recent_origins])
        )
        for horizon in contract.expected_horizons
    }
    bootstrap_lower, adjusted_alpha = _moving_block_bootstrap_lower_bound(
        origin_advantages,
        block_length=contract.bootstrap_block_length,
        replicates=contract.bootstrap_replicates,
        confidence=contract.bootstrap_confidence,
        multiplicity_count=contract.multiplicity_correction_count,
        seed=contract.bootstrap_seed,
    )
    mean_advantage = float(np.mean(origin_advantages))
    recent_mean = float(np.mean(recent_origin_values))
    win_fraction = float(np.mean(origin_advantages > 0.0))

    metrics = PublicEligibilityMetricsV54(
        origin_count=len(origins),
        horizons=horizons,
        paired_row_count=len(evidence.rows),
        mean_advantage=mean_advantage,
        selection_adjusted_bootstrap_lower_bound=bootstrap_lower,
        adjusted_alpha=adjusted_alpha,
        origin_win_fraction=win_fraction,
        contiguous_time_block_means=block_means,
        recent_mean_advantage=recent_mean,
        horizon_mean_advantages=horizon_means,
        recent_horizon_mean_advantages=recent_horizon_means,
    )
    checks = {
        "nonfixture_public_evidence": not evidence.fixture_only,
        "public_scientific_acceptance": (
            evidence.public_scientific_acceptance_verified
        ),
        "candidate_search_within_frozen_budget": (
            evidence.candidate_search_count <= contract.multiplicity_correction_count
        ),
        "minimum_origin_count": len(origins) >= contract.minimum_origin_count,
        "complete_origin_horizon_grid": rectangular_coverage,
        "overall_advantage": (mean_advantage > contract.minimum_overall_advantage),
        "selection_adjusted_bootstrap_lower_bound": (
            bootstrap_lower > contract.minimum_overall_advantage
        ),
        "origin_win_fraction": (win_fraction >= contract.minimum_origin_win_fraction),
        "all_contiguous_time_blocks": all(
            value > contract.minimum_time_block_advantage for value in block_means
        ),
        "recent_window_advantage": (recent_mean > contract.minimum_recent_advantage),
        "all_horizon_advantages": all(
            value > contract.minimum_horizon_advantage
            for value in horizon_means.values()
        ),
        "all_recent_horizon_advantages": all(
            value > contract.minimum_horizon_advantage
            for value in recent_horizon_means.values()
        ),
    }
    eligible = all(checks.values())
    return PublicEligibilityAssessmentV54.seal(
        task_id=evidence.task_id,
        contract_hash=contract.contract_hash,
        input_hash=evidence.input_hash,
        candidate_id=evidence.candidate_id,
        baseline_id=evidence.baseline_id,
        metrics=metrics,
        checks=checks,
        decision="ELIGIBLE" if eligible else "ABSTAIN",
        public_gate_eligible=eligible,
    )
