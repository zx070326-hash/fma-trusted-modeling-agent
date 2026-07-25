"""Code-owned materialization of one frozen campaign protocol.

The baseline identifier is accepted exactly once, in the prospective protocol.
Every downstream policy and V5.4 contract is derived from that sealed value.
There is no downstream baseline argument to normalize, rename, or mistype.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_4.public_eligibility import PublicEligibilityContractV54


NonNegativeFinite = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class PublicEligibilitySettingsV55(StrictModel):
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

    @model_validator(mode="after")
    def validate_settings(self) -> "PublicEligibilitySettingsV55":
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
        return self


class ProspectiveCampaignProtocolV55(StrictModel):
    """Protocol frozen before task selection and downstream materialization."""

    schema_version: Literal["5.5-prospective-campaign-protocol"] = (
        "5.5-prospective-campaign-protocol"
    )
    protocol_id: Identifier
    baseline_id: Identifier
    candidate_families: Annotated[list[Identifier], Field(min_length=3)]
    maximum_candidate_search_count: Annotated[int, Field(ge=3)]
    public_eligibility: PublicEligibilitySettingsV55
    selection_rule: Literal["aggregate_normalized_mae_then_rmse_then_complexity"] = (
        "aggregate_normalized_mae_then_rmse_then_complexity"
    )
    graph_recovery_required_after_initial_family_failure: Literal[True] = True
    selection_uses_public_data_only: Literal[True] = True
    final_family_locked_before_all_public_refit: Literal[True] = True
    private_target_storage: Literal["separate_encrypted_envelope"] = (
        "separate_encrypted_envelope"
    )
    source_provenance_release: Literal["separate_encrypted_envelope_after_closeout"] = (
        "separate_encrypted_envelope_after_closeout"
    )
    distinct_encryption_key_domains: Literal[True] = True
    frozen_before_task_selection: Literal[True] = True
    frozen_at: datetime
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "ProspectiveCampaignProtocolV55":
        if self.candidate_families != sorted(set(self.candidate_families)):
            raise ValueError("candidate families must be sorted and unique")
        if self.maximum_candidate_search_count < len(self.candidate_families):
            raise ValueError("candidate budget is below registered family count")
        if (
            self.public_eligibility.multiplicity_correction_count
            != self.maximum_candidate_search_count
        ):
            raise ValueError(
                "multiplicity correction must equal maximum candidate search count"
            )
        if self.frozen_at.utcoffset() is None:
            raise ValueError("frozen_at must be timezone-aware")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("prospective protocol hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise ValueError("prospective campaign protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ProspectiveCampaignProtocolV55":
        data.setdefault("frozen_at", _utc_now())
        draft = cls(**data)
        payload = draft.model_dump(exclude={"protocol_hash"})
        payload["protocol_hash"] = draft.content_hash()
        return cls(**payload)


class CandidateSelectionPolicyV55(StrictModel):
    """Task-specific policy derived without a caller-supplied baseline."""

    schema_version: Literal["5.5-candidate-selection-policy"] = (
        "5.5-candidate-selection-policy"
    )
    task_id: Identifier
    protocol_hash: Sha256
    baseline_id: Identifier
    candidate_families: Annotated[list[Identifier], Field(min_length=3)]
    maximum_candidate_search_count: Annotated[int, Field(ge=3)]
    selection_rule: Literal["aggregate_normalized_mae_then_rmse_then_complexity"]
    graph_recovery_required_after_initial_family_failure: Literal[True]
    selection_uses_public_data_only: Literal[True]
    final_family_locked_before_all_public_refit: Literal[True]
    private_target_access_forbidden: Literal[True] = True
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "CandidateSelectionPolicyV55":
        if self.candidate_families != sorted(set(self.candidate_families)):
            raise ValueError("candidate families must be sorted and unique")
        if self.maximum_candidate_search_count < len(self.candidate_families):
            raise ValueError("candidate budget is below registered family count")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("candidate selection policy hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("candidate selection policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CandidateSelectionPolicyV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"policy_hash"})
        payload["policy_hash"] = draft.content_hash()
        return cls(**payload)


class PublicLaunchBindingV55(StrictModel):
    """Code-owned receipt binding exact protocol, policy, and V5.4 contract."""

    schema_version: Literal["5.5-public-launch-binding"] = "5.5-public-launch-binding"
    task_id: Identifier
    protocol_hash: Sha256
    candidate_policy_hash: Sha256
    public_eligibility_contract_hash: Sha256
    baseline_id: Identifier
    strict_baseline_identity_verified: Literal[True] = True
    all_fields_derived_from_protocol: Literal[True] = True
    private_evaluation_performed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    binding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "PublicLaunchBindingV55":
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("public launch binding hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    def assert_sealed(self) -> None:
        if not self.binding_hash or self.binding_hash != self.content_hash():
            raise ValueError("public launch binding is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PublicLaunchBindingV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"binding_hash"})
        payload["binding_hash"] = draft.content_hash()
        return cls(**payload)


def materialize_public_launch_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    task_id: str,
    eligibility_contract_id: str,
    materialized_at: datetime | None = None,
) -> tuple[
    CandidateSelectionPolicyV55,
    PublicEligibilityContractV54,
    PublicLaunchBindingV55,
]:
    """Derive every baseline-bearing public artifact from one sealed protocol."""

    protocol.assert_sealed()
    settings = protocol.public_eligibility
    policy = CandidateSelectionPolicyV55.seal(
        task_id=task_id,
        protocol_hash=protocol.protocol_hash,
        baseline_id=protocol.baseline_id,
        candidate_families=protocol.candidate_families,
        maximum_candidate_search_count=protocol.maximum_candidate_search_count,
        selection_rule=protocol.selection_rule,
        graph_recovery_required_after_initial_family_failure=(
            protocol.graph_recovery_required_after_initial_family_failure
        ),
        selection_uses_public_data_only=protocol.selection_uses_public_data_only,
        final_family_locked_before_all_public_refit=(
            protocol.final_family_locked_before_all_public_refit
        ),
    )
    contract = PublicEligibilityContractV54.seal(
        contract_id=eligibility_contract_id,
        task_id=task_id,
        baseline_id=protocol.baseline_id,
        candidate_selection_rule_hash=policy.policy_hash,
        expected_horizons=settings.expected_horizons,
        minimum_origin_count=settings.minimum_origin_count,
        contiguous_time_block_count=settings.contiguous_time_block_count,
        recent_origin_count=settings.recent_origin_count,
        minimum_overall_advantage=settings.minimum_overall_advantage,
        minimum_time_block_advantage=settings.minimum_time_block_advantage,
        minimum_recent_advantage=settings.minimum_recent_advantage,
        minimum_horizon_advantage=settings.minimum_horizon_advantage,
        minimum_origin_win_fraction=settings.minimum_origin_win_fraction,
        bootstrap_confidence=settings.bootstrap_confidence,
        bootstrap_replicates=settings.bootstrap_replicates,
        bootstrap_block_length=settings.bootstrap_block_length,
        multiplicity_correction_count=settings.multiplicity_correction_count,
        bootstrap_seed=settings.bootstrap_seed,
        frozen_at=materialized_at or _utc_now(),
    )
    binding = PublicLaunchBindingV55.seal(
        task_id=task_id,
        protocol_hash=protocol.protocol_hash,
        candidate_policy_hash=policy.policy_hash,
        public_eligibility_contract_hash=contract.contract_hash,
        baseline_id=protocol.baseline_id,
    )
    assert_public_launch_binding_v55(
        protocol=protocol,
        policy=policy,
        contract=contract,
        binding=binding,
    )
    return policy, contract, binding


def verify_public_launch_binding_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    policy: CandidateSelectionPolicyV55,
    contract: PublicEligibilityContractV54,
    binding: PublicLaunchBindingV55,
) -> bool:
    try:
        protocol.assert_sealed()
        policy.assert_sealed()
        contract.assert_sealed()
        binding.assert_sealed()
        settings = protocol.public_eligibility
        return bool(
            policy.task_id == contract.task_id == binding.task_id
            and policy.protocol_hash == binding.protocol_hash == protocol.protocol_hash
            and protocol.baseline_id
            == policy.baseline_id
            == contract.baseline_id
            == binding.baseline_id
            and policy.policy_hash
            == contract.candidate_selection_rule_hash
            == binding.candidate_policy_hash
            and contract.contract_hash == binding.public_eligibility_contract_hash
            and policy.candidate_families == protocol.candidate_families
            and policy.maximum_candidate_search_count
            == protocol.maximum_candidate_search_count
            and policy.selection_rule == protocol.selection_rule
            and contract.expected_horizons == settings.expected_horizons
            and contract.minimum_origin_count == settings.minimum_origin_count
            and contract.contiguous_time_block_count
            == settings.contiguous_time_block_count
            and contract.recent_origin_count == settings.recent_origin_count
            and math.isclose(
                contract.minimum_overall_advantage,
                settings.minimum_overall_advantage,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                contract.minimum_time_block_advantage,
                settings.minimum_time_block_advantage,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                contract.minimum_recent_advantage,
                settings.minimum_recent_advantage,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                contract.minimum_horizon_advantage,
                settings.minimum_horizon_advantage,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                contract.minimum_origin_win_fraction,
                settings.minimum_origin_win_fraction,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and math.isclose(
                contract.bootstrap_confidence,
                settings.bootstrap_confidence,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and contract.bootstrap_replicates == settings.bootstrap_replicates
            and contract.bootstrap_block_length == settings.bootstrap_block_length
            and contract.multiplicity_correction_count
            == settings.multiplicity_correction_count
            and contract.bootstrap_seed == settings.bootstrap_seed
        )
    except (TypeError, ValueError):
        return False


def assert_public_launch_binding_v55(
    *,
    protocol: ProspectiveCampaignProtocolV55,
    policy: CandidateSelectionPolicyV55,
    contract: PublicEligibilityContractV54,
    binding: PublicLaunchBindingV55,
) -> None:
    if not verify_public_launch_binding_v55(
        protocol=protocol,
        policy=policy,
        contract=contract,
        binding=binding,
    ):
        raise ValueError("public launch artifacts are not exact protocol derivations")


__all__ = [
    "CandidateSelectionPolicyV55",
    "ProspectiveCampaignProtocolV55",
    "PublicEligibilitySettingsV55",
    "PublicLaunchBindingV55",
    "assert_public_launch_binding_v55",
    "materialize_public_launch_v55",
    "verify_public_launch_binding_v55",
]
