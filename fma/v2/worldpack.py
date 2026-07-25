from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .empirical_schemas import ForecastCandidate, ForecastCandidateSpec, ForecastCandidateSpecV22
from .forecast_evaluator import forecast_one
from .schemas import Identifier, Sha256, _assert_timezone


Mechanism = Literal[
    "local_trend_after_break",
    "level_shift",
    "seasonal_regime",
    "mean_reverting_control",
]
Arm = Literal["direct_generation", "retrieval_evolution_memory"]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class WorldPackSpecV22(StrictModel):
    """Pre-frozen paired ablation protocol; the outer holdout stays private."""

    schema_version: Literal["2.2"] = "2.2"
    pack_id: Identifier
    mechanisms: list[Mechanism] = Field(min_length=4, max_length=4)
    seeds: list[Annotated[int, Field(ge=0, le=2_147_483_647)]] = Field(
        min_length=3, max_length=8
    )
    training_points: Annotated[int, Field(ge=48, le=10_000)]
    inner_validation_points: Annotated[int, Field(ge=12, le=1_000)]
    outer_holdout_points: Annotated[int, Field(ge=12, le=1_000)]
    seasonal_period: Annotated[int, Field(ge=2, le=365)]
    candidate_budget_per_arm: Literal[4] = 4
    confidence_level: Annotated[float, Field(gt=0.8, lt=1, allow_inf_nan=False)]
    bootstrap_replicates: Annotated[int, Field(ge=1_000, le=100_000)]
    bootstrap_seed: Annotated[int, Field(ge=0, le=2_147_483_647)]
    negative_transfer_relative_margin: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False)
    ]
    max_negative_transfer_cases: Annotated[int, Field(ge=0)]
    minimum_win_fraction: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "WorldPackSpecV22":
        _assert_timezone(self.frozen_at, "frozen_at")
        if set(self.mechanisms) != {
            "local_trend_after_break",
            "level_shift",
            "seasonal_regime",
            "mean_reverting_control",
        }:
            raise ValueError("WorldPack needs the four frozen mechanisms exactly once")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("WorldPack seeds must be unique")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match WorldPack spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("WorldPack spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackSpecV22":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class WorldPackCandidateDefinitionV22(StrictModel):
    candidate_id: Identifier
    family: Literal[
        "last_value",
        "mean_level",
        "linear_trend",
        "seasonal_naive",
        "window_linear_trend",
        "exponential_smoothing",
    ]
    seasonal_period: Annotated[int, Field(ge=2, le=365)] | None = None
    window_length: Annotated[int, Field(ge=4, le=10_000)] | None = None
    smoothing_alpha: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> "WorldPackCandidateDefinitionV22":
        if (self.family == "seasonal_naive") != (self.seasonal_period is not None):
            raise ValueError("only seasonal_naive needs seasonal_period")
        if (self.family == "window_linear_trend") != (self.window_length is not None):
            raise ValueError("only window_linear_trend needs window_length")
        if (self.family == "exponential_smoothing") != (
            self.smoothing_alpha is not None
        ):
            raise ValueError("only exponential_smoothing needs smoothing_alpha")
        return self


class WorldPackArmPolicyV22(StrictModel):
    """The exact candidate-and-selector policy under test, independent of cases."""

    schema_version: Literal["2.2"] = "2.2"
    policy_id: Identifier
    arm: Arm
    selection_rule: Literal["minimum_inner_rolling_mae"] = "minimum_inner_rolling_mae"
    candidates: list[WorldPackCandidateDefinitionV22] = Field(min_length=4, max_length=4)
    policy_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_policy(self) -> "WorldPackArmPolicyV22":
        ids = [candidate.candidate_id for candidate in self.candidates]
        families = [candidate.family for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("WorldPack policy candidate IDs must be unique")
        if families.count("last_value") != 1:
            raise ValueError("WorldPack policy needs exactly one last-value baseline")
        if self.policy_hash and self.policy_hash != self.content_hash():
            raise ValueError("policy_hash does not match WorldPack arm policy")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "policy_hash")

    def assert_sealed(self) -> None:
        if not self.policy_hash or self.policy_hash != self.content_hash():
            raise ValueError("WorldPack arm policy is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackArmPolicyV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"policy_hash"}),
            policy_hash=draft.content_hash(),
        )


class HiddenWorldCaseV22(StrictModel):
    schema_version: Literal["2.2"] = "2.2"
    case_id: Identifier
    pack_spec_hash: Sha256
    mechanism: Mechanism
    seed: int
    values: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        min_length=72
    )
    case_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_case(self) -> "HiddenWorldCaseV22":
        if self.case_hash and self.case_hash != self.content_hash():
            raise ValueError("case_hash does not match hidden world case")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "case_hash")

    def assert_sealed(self) -> None:
        if not self.case_hash or self.case_hash != self.content_hash():
            raise ValueError("hidden world case is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "HiddenWorldCaseV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"case_hash"}),
            case_hash=draft.content_hash(),
        )

    def public_projection(self, spec: WorldPackSpecV22) -> dict[str, object]:
        """The only case view available to candidate selection."""

        self.assert_sealed()
        spec.assert_sealed()
        if self.pack_spec_hash != spec.spec_hash:
            raise ValueError("hidden case belongs to another WorldPack")
        observed_count = spec.training_points + spec.inner_validation_points
        observed = self.values[:observed_count]
        return {
            "case_id": self.case_id,
            "observed_values": observed,
            "inner_start_index": spec.training_points,
            "outer_holdout_points": spec.outer_holdout_points,
            "seasonal_period": spec.seasonal_period,
            "candidate_budget": spec.candidate_budget_per_arm,
            "observed_data_hash": sha256_value(
                {
                    "case_id": self.case_id,
                    "observed_values": observed,
                    "inner_start_index": spec.training_points,
                    "outer_holdout_points": spec.outer_holdout_points,
                    "seasonal_period": spec.seasonal_period,
                    "candidate_budget": spec.candidate_budget_per_arm,
                }
            ),
        }


class PrivateWorldPackV22(StrictModel):
    schema_version: Literal["2.2"] = "2.2"
    pack_spec_hash: Sha256
    cases: list[HiddenWorldCaseV22] = Field(min_length=12)
    private_pack_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_pack(self) -> "PrivateWorldPackV22":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("WorldPack case IDs must be unique")
        for case in self.cases:
            case.assert_sealed()
            if case.pack_spec_hash != self.pack_spec_hash:
                raise ValueError("WorldPack contains a case from another spec")
        if self.private_pack_hash and self.private_pack_hash != self.content_hash():
            raise ValueError("private_pack_hash does not match WorldPack")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "private_pack_hash")

    def assert_sealed(self) -> None:
        if not self.private_pack_hash or self.private_pack_hash != self.content_hash():
            raise ValueError("private WorldPack is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PrivateWorldPackV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"private_pack_hash"}),
            private_pack_hash=draft.content_hash(),
        )


class WorldPackSelectionReceiptV22(StrictModel):
    schema_version: Literal["2.2"] = "2.2"
    case_id: Identifier
    arm: Arm
    arm_policy_hash: Sha256
    observed_data_hash: Sha256
    candidate_ids: list[Identifier]
    candidate_hashes: dict[Identifier, Sha256]
    inner_mae_by_candidate: dict[Identifier, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    selected_candidate_id: Identifier
    forecast_evaluation_count: Annotated[int, Field(ge=1)]
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "WorldPackSelectionReceiptV22":
        if self.candidate_ids != sorted(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be sorted and unique")
        expected = set(self.candidate_ids)
        if set(self.candidate_hashes) != expected or set(self.inner_mae_by_candidate) != expected:
            raise ValueError("selection receipt candidate maps do not match candidate_ids")
        if self.selected_candidate_id not in expected:
            raise ValueError("selected candidate is absent from the receipt")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match WorldPack selection")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("WorldPack selection receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackSelectionReceiptV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"receipt_hash"}),
            receipt_hash=draft.content_hash(),
        )


class WorldPackSelectionBundleV22(StrictModel):
    schema_version: Literal["2.2"] = "2.2"
    arm: Arm
    arm_policy_hash: Sha256
    receipts: list[WorldPackSelectionReceiptV22] = Field(min_length=12)
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "WorldPackSelectionBundleV22":
        ids = [receipt.case_id for receipt in self.receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("selection bundle case IDs must be unique")
        for receipt in self.receipts:
            receipt.assert_sealed()
            if receipt.arm != self.arm or receipt.arm_policy_hash != self.arm_policy_hash:
                raise ValueError("selection receipt belongs to another arm")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match selection bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("WorldPack selection bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackSelectionBundleV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class WorldPackCaseResultV22(StrictModel):
    case_id: Identifier
    mechanism: Mechanism
    seed: int
    direct_candidate_id: Identifier
    memory_candidate_id: Identifier
    direct_outer_mae: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    memory_outer_mae: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    paired_mae_improvement: Annotated[float, Field(allow_inf_nan=False)]
    relative_improvement: Annotated[float, Field(allow_inf_nan=False)] | None
    outcome: Literal["memory_win", "tie", "memory_loss"]
    negative_transfer: bool
    direct_evaluation_count: Annotated[int, Field(ge=1)]
    memory_evaluation_count: Annotated[int, Field(ge=1)]


class WorldPackAblationReportV22(StrictModel):
    schema_version: Literal["2.2"] = "2.2"
    report_id: Identifier
    pack_spec_hash: Sha256
    private_pack_hash: Sha256
    direct_policy_hash: Sha256
    memory_policy_hash: Sha256
    direct_bundle_hash: Sha256
    memory_bundle_hash: Sha256
    cases: list[WorldPackCaseResultV22] = Field(min_length=12)
    mean_mae_improvement: Annotated[float, Field(allow_inf_nan=False)]
    confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    confidence_upper: Annotated[float, Field(allow_inf_nan=False)]
    win_count: Annotated[int, Field(ge=0)]
    tie_count: Annotated[int, Field(ge=0)]
    loss_count: Annotated[int, Field(ge=0)]
    negative_transfer_count: Annotated[int, Field(ge=0)]
    same_budget: bool
    status: Literal["promoted_for_worldpack_scope", "candidate_rejected"]
    reason_codes: list[Literal[
        "confidence_interval_not_positive",
        "negative_transfer_limit_exceeded",
        "minimum_win_fraction_not_met",
        "candidate_budget_mismatch",
    ]]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "WorldPackAblationReportV22":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if self.win_count + self.tie_count + self.loss_count != len(self.cases):
            raise ValueError("WorldPack outcome counts do not cover all cases")
        if (self.status == "promoted_for_worldpack_scope") == bool(self.reason_codes):
            raise ValueError("promoted reports need no reasons; rejected reports need reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match WorldPack report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("WorldPack ablation report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackAblationReportV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class WorldPackManifestV22(StrictModel):
    schema_version: Literal["2.2"] = "2.2"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    artifact_refs: list[ArtifactRef] = Field(min_length=7, max_length=7)
    terminal_status: Literal["promoted_for_worldpack_scope", "candidate_rejected"]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "WorldPackManifestV22":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != 7:
            raise ValueError("WorldPack manifest references must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match WorldPack manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("WorldPack manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackManifestV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class WorldPackOutcome:
    store: RunStore
    spec: WorldPackSpecV22
    private_pack: PrivateWorldPackV22
    direct_policy: WorldPackArmPolicyV22
    memory_policy: WorldPackArmPolicyV22
    direct_selections: WorldPackSelectionBundleV22
    memory_selections: WorldPackSelectionBundleV22
    report: WorldPackAblationReportV22
    manifest: WorldPackManifestV22


def default_worldpack_spec(*, frozen_at: datetime | None = None) -> WorldPackSpecV22:
    return WorldPackSpecV22.seal(
        pack_id="forecast_adaptation_worldpack",
        mechanisms=[
            "local_trend_after_break",
            "level_shift",
            "seasonal_regime",
            "mean_reverting_control",
        ],
        seeds=[11, 29, 47],
        training_points=84,
        inner_validation_points=24,
        outer_holdout_points=24,
        seasonal_period=12,
        confidence_level=0.95,
        bootstrap_replicates=5_000,
        bootstrap_seed=2_026_072_2,
        negative_transfer_relative_margin=0.05,
        max_negative_transfer_cases=0,
        minimum_win_fraction=0.5,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def default_worldpack_arm_policy(
    arm: Arm, *, seasonal_period: int = 12
) -> WorldPackArmPolicyV22:
    if arm == "direct_generation":
        definitions = [
            WorldPackCandidateDefinitionV22(
                candidate_id="last_value_baseline", family="last_value"
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="global_mean_challenger", family="mean_level"
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="global_trend_challenger", family="linear_trend"
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="seasonal_naive_challenger",
                family="seasonal_naive",
                seasonal_period=seasonal_period,
            ),
        ]
    else:
        definitions = [
            WorldPackCandidateDefinitionV22(
                candidate_id="last_value_baseline", family="last_value"
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="local_trend_challenger",
                family="window_linear_trend",
                window_length=12,
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="exponential_level_challenger",
                family="exponential_smoothing",
                smoothing_alpha=0.3,
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="seasonal_naive_challenger",
                family="seasonal_naive",
                seasonal_period=seasonal_period,
            ),
        ]
    return WorldPackArmPolicyV22.seal(
        policy_id=f"{arm}_policy",
        arm=arm,
        candidates=definitions,
    )


def safe_hybrid_worldpack_policy(*, seasonal_period: int = 12) -> WorldPackArmPolicyV22:
    """Failure-evolved policy: retain global trend, retire SES from this bundle.

    The SES knowledge artifact is not deleted.  It simply loses membership in
    this exact policy after the V2.3 negative-transfer failure signature.
    """

    return WorldPackArmPolicyV22.seal(
        policy_id="retrieval_hybrid_safe_policy",
        arm="retrieval_evolution_memory",
        candidates=[
            WorldPackCandidateDefinitionV22(
                candidate_id="last_value_baseline", family="last_value"
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="global_trend_challenger", family="linear_trend"
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="local_trend_challenger",
                family="window_linear_trend",
                window_length=12,
            ),
            WorldPackCandidateDefinitionV22(
                candidate_id="seasonal_naive_challenger",
                family="seasonal_naive",
                seasonal_period=seasonal_period,
            ),
        ],
    )


def generate_private_worldpack(spec: WorldPackSpecV22) -> PrivateWorldPackV22:
    spec.assert_sealed()
    assert spec.spec_hash is not None
    total = spec.training_points + spec.inner_validation_points + spec.outer_holdout_points
    cases: list[HiddenWorldCaseV22] = []
    for mechanism_index, mechanism in enumerate(spec.mechanisms):
        for seed in spec.seeds:
            values = _generate_values(
                mechanism,
                seed + mechanism_index * 100_000,
                total,
                spec.training_points,
                spec.seasonal_period,
            )
            cases.append(
                HiddenWorldCaseV22.seal(
                    case_id=f"{mechanism}_s{seed}",
                    pack_spec_hash=spec.spec_hash,
                    mechanism=mechanism,
                    seed=seed,
                    values=values,
                )
            )
    return PrivateWorldPackV22.seal(pack_spec_hash=spec.spec_hash, cases=cases)


def select_worldpack_candidate(
    public_case: dict[str, object],
    arm: Arm,
    policy: WorldPackArmPolicyV22 | None = None,
) -> WorldPackSelectionReceiptV22:
    """Select on the visible inner window only; no hidden case fields are accepted."""

    allowed = {
        "case_id",
        "observed_values",
        "inner_start_index",
        "outer_holdout_points",
        "seasonal_period",
        "candidate_budget",
        "observed_data_hash",
    }
    if set(public_case) != allowed:
        raise ValueError("public WorldPack projection contains unexpected fields")
    case_id = str(public_case["case_id"])
    values = [float(value) for value in public_case["observed_values"]]
    inner_start = int(public_case["inner_start_index"])
    seasonal_period = int(public_case["seasonal_period"])
    candidate_budget = int(public_case["candidate_budget"])
    policy = policy or default_worldpack_arm_policy(
        arm, seasonal_period=seasonal_period
    )
    policy.assert_sealed()
    if policy.arm != arm:
        raise ValueError("WorldPack arm policy belongs to another arm")
    data_contract_hash = sha256_value(
        {
            "worldpack_case": case_id,
            "observed_data_hash": public_case["observed_data_hash"],
        }
    )
    candidates = _candidates_for_policy(policy, data_contract_hash)
    if len(candidates) != candidate_budget:
        raise RuntimeError("candidate arm violates the frozen WorldPack budget")
    inner_mae: dict[str, float] = {}
    hashes: dict[str, str] = {}
    for candidate in candidates:
        errors = [
            abs(forecast_one(values[:origin], candidate) - values[origin])
            for origin in range(inner_start, len(values))
        ]
        inner_mae[candidate.candidate_id] = sum(errors) / len(errors)
        assert candidate.candidate_hash is not None
        hashes[candidate.candidate_id] = candidate.candidate_hash
    selected = min(inner_mae, key=lambda candidate_id: (inner_mae[candidate_id], candidate_id))
    candidate_ids = sorted(inner_mae)
    return WorldPackSelectionReceiptV22.seal(
        case_id=case_id,
        arm=arm,
        arm_policy_hash=policy.policy_hash,
        observed_data_hash=str(public_case["observed_data_hash"]),
        candidate_ids=candidate_ids,
        candidate_hashes={candidate_id: hashes[candidate_id] for candidate_id in candidate_ids},
        inner_mae_by_candidate={candidate_id: inner_mae[candidate_id] for candidate_id in candidate_ids},
        selected_candidate_id=selected,
        forecast_evaluation_count=len(candidates) * (len(values) - inner_start),
    )


def evaluate_worldpack_ablation(
    spec: WorldPackSpecV22,
    private_pack: PrivateWorldPackV22,
    direct_policy: WorldPackArmPolicyV22,
    memory_policy: WorldPackArmPolicyV22,
    direct: WorldPackSelectionBundleV22,
    memory: WorldPackSelectionBundleV22,
    *,
    evaluated_at: datetime | None = None,
) -> WorldPackAblationReportV22:
    spec.assert_sealed()
    private_pack.assert_sealed()
    direct_policy.assert_sealed()
    memory_policy.assert_sealed()
    direct.assert_sealed()
    memory.assert_sealed()
    if private_pack.pack_spec_hash != spec.spec_hash:
        raise ValueError("private WorldPack is bound to another spec")
    if direct.arm != "direct_generation" or memory.arm != "retrieval_evolution_memory":
        raise ValueError("WorldPack bundles are assigned to the wrong arms")
    if (
        direct_policy.arm != direct.arm
        or memory_policy.arm != memory.arm
        or direct.arm_policy_hash != direct_policy.policy_hash
        or memory.arm_policy_hash != memory_policy.policy_hash
    ):
        raise ValueError("WorldPack selection bundle is bound to another arm policy")
    direct_by_case = {receipt.case_id: receipt for receipt in direct.receipts}
    memory_by_case = {receipt.case_id: receipt for receipt in memory.receipts}
    case_ids = {case.case_id for case in private_pack.cases}
    if set(direct_by_case) != case_ids or set(memory_by_case) != case_ids:
        raise ValueError("selection bundles do not cover the private WorldPack")

    results: list[WorldPackCaseResultV22] = []
    for case in private_pack.cases:
        public = case.public_projection(spec)
        direct_receipt = direct_by_case[case.case_id]
        memory_receipt = memory_by_case[case.case_id]
        expected_direct = select_worldpack_candidate(
            public, "direct_generation", direct_policy
        )
        expected_memory = select_worldpack_candidate(
            public, "retrieval_evolution_memory", memory_policy
        )
        if direct_receipt.receipt_hash != expected_direct.receipt_hash:
            raise ValueError("direct selection does not replay from the public projection")
        if memory_receipt.receipt_hash != expected_memory.receipt_hash:
            raise ValueError("memory selection does not replay from the public projection")
        direct_candidate = _candidate_by_id(
            direct_policy, direct_receipt.selected_candidate_id, public
        )
        memory_candidate = _candidate_by_id(
            memory_policy, memory_receipt.selected_candidate_id, public
        )
        observed_count = spec.training_points + spec.inner_validation_points
        direct_errors = [
            abs(forecast_one(case.values[:origin], direct_candidate) - case.values[origin])
            for origin in range(observed_count, len(case.values))
        ]
        memory_errors = [
            abs(forecast_one(case.values[:origin], memory_candidate) - case.values[origin])
            for origin in range(observed_count, len(case.values))
        ]
        direct_mae = sum(direct_errors) / len(direct_errors)
        memory_mae = sum(memory_errors) / len(memory_errors)
        improvement = direct_mae - memory_mae
        tolerance = max(1e-12, direct_mae * 1e-9)
        if improvement > tolerance:
            outcome = "memory_win"
        elif improvement < -tolerance:
            outcome = "memory_loss"
        else:
            outcome = "tie"
        negative_transfer = memory_mae > direct_mae * (
            1.0 + spec.negative_transfer_relative_margin
        ) + 1e-12
        results.append(
            WorldPackCaseResultV22(
                case_id=case.case_id,
                mechanism=case.mechanism,
                seed=case.seed,
                direct_candidate_id=direct_candidate.candidate_id,
                memory_candidate_id=memory_candidate.candidate_id,
                direct_outer_mae=direct_mae,
                memory_outer_mae=memory_mae,
                paired_mae_improvement=improvement,
                relative_improvement=(None if direct_mae <= 1e-15 else improvement / direct_mae),
                outcome=outcome,
                negative_transfer=negative_transfer,
                direct_evaluation_count=direct_receipt.forecast_evaluation_count,
                memory_evaluation_count=memory_receipt.forecast_evaluation_count,
            )
        )
    improvements = [result.paired_mae_improvement for result in results]
    lower, upper = _paired_case_bootstrap_interval(improvements, spec)
    mean_improvement = float(np.mean(improvements))
    win_count = sum(result.outcome == "memory_win" for result in results)
    tie_count = sum(result.outcome == "tie" for result in results)
    loss_count = sum(result.outcome == "memory_loss" for result in results)
    negative_count = sum(result.negative_transfer for result in results)
    same_budget = all(
        result.direct_evaluation_count == result.memory_evaluation_count for result in results
    )
    reasons: list[str] = []
    if lower <= 0:
        reasons.append("confidence_interval_not_positive")
    if negative_count > spec.max_negative_transfer_cases:
        reasons.append("negative_transfer_limit_exceeded")
    if win_count / len(results) < spec.minimum_win_fraction:
        reasons.append("minimum_win_fraction_not_met")
    if not same_budget:
        reasons.append("candidate_budget_mismatch")
    assert spec.spec_hash is not None
    assert private_pack.private_pack_hash is not None
    assert direct_policy.policy_hash is not None
    assert memory_policy.policy_hash is not None
    assert direct.bundle_hash is not None
    assert memory.bundle_hash is not None
    return WorldPackAblationReportV22.seal(
        report_id="forecast_memory_paired_ablation",
        pack_spec_hash=spec.spec_hash,
        private_pack_hash=private_pack.private_pack_hash,
        direct_policy_hash=direct_policy.policy_hash,
        memory_policy_hash=memory_policy.policy_hash,
        direct_bundle_hash=direct.bundle_hash,
        memory_bundle_hash=memory.bundle_hash,
        cases=results,
        mean_mae_improvement=mean_improvement,
        confidence_lower=lower,
        confidence_upper=upper,
        win_count=win_count,
        tie_count=tie_count,
        loss_count=loss_count,
        negative_transfer_count=negative_count,
        same_budget=same_budget,
        status="candidate_rejected" if reasons else "promoted_for_worldpack_scope",
        reason_codes=reasons,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_worldpack_ablation(
    output_root: str | Path,
    *,
    spec: WorldPackSpecV22 | None = None,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> WorldPackOutcome:
    at = evaluated_at or datetime.now(timezone.utc)
    spec = spec or default_worldpack_spec(frozen_at=at)
    private_pack = generate_private_worldpack(spec)
    direct_policy = default_worldpack_arm_policy(
        "direct_generation", seasonal_period=spec.seasonal_period
    )
    memory_policy = default_worldpack_arm_policy(
        "retrieval_evolution_memory", seasonal_period=spec.seasonal_period
    )
    assert direct_policy.policy_hash is not None
    assert memory_policy.policy_hash is not None
    direct = WorldPackSelectionBundleV22.seal(
        arm="direct_generation",
        arm_policy_hash=direct_policy.policy_hash,
        receipts=[
            select_worldpack_candidate(
                case.public_projection(spec), "direct_generation", direct_policy
            )
            for case in private_pack.cases
        ],
    )
    memory = WorldPackSelectionBundleV22.seal(
        arm="retrieval_evolution_memory",
        arm_policy_hash=memory_policy.policy_hash,
        receipts=[
            select_worldpack_candidate(
                case.public_projection(spec),
                "retrieval_evolution_memory",
                memory_policy,
            )
            for case in private_pack.cases
        ],
    )
    report = evaluate_worldpack_ablation(
        spec,
        private_pack,
        direct_policy,
        memory_policy,
        direct,
        memory,
        evaluated_at=at,
    )
    store = RunStore(
        output_root,
        run_id=run_id or f"worldpack-ablation-{uuid4().hex[:12]}",
    )
    refs = [
        store.put_artifact("worldpack_spec_v22", spec),
        store.put_artifact("private_worldpack_v22", private_pack),
        store.put_artifact("worldpack_direct_policy_v22", direct_policy),
        store.put_artifact("worldpack_memory_policy_v22", memory_policy),
        store.put_artifact("worldpack_direct_selections_v22", direct),
        store.put_artifact("worldpack_memory_selections_v22", memory),
        store.put_artifact("worldpack_ablation_report_v22", report),
    ]
    manifest = WorldPackManifestV22.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("worldpack_manifest_v22", manifest)
    store.emit(
        "worldpack_ablation_completed",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_worldpack_run(store.run_directory):
        raise RuntimeError("WorldPack run failed independent verification")
    return WorldPackOutcome(
        store,
        spec,
        private_pack,
        direct_policy,
        memory_policy,
        direct,
        memory,
        report,
        manifest,
    )


def verify_worldpack_run(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in refs:
            store.load_artifact(ref)
        manifests = [ref for ref in refs if ref.kind == "worldpack_manifest_v22"]
        if len(manifests) != 1:
            return False
        manifest = WorldPackManifestV22.model_validate(store.load_artifact(manifests[0]))
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            matches = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(matches) != 1:
                raise RuntimeError(f"manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(matches[0]))

        spec = load_one("worldpack_spec_v22", WorldPackSpecV22)
        private_pack = load_one("private_worldpack_v22", PrivateWorldPackV22)
        direct_policy = load_one("worldpack_direct_policy_v22", WorldPackArmPolicyV22)
        memory_policy = load_one("worldpack_memory_policy_v22", WorldPackArmPolicyV22)
        direct = load_one("worldpack_direct_selections_v22", WorldPackSelectionBundleV22)
        memory = load_one("worldpack_memory_selections_v22", WorldPackSelectionBundleV22)
        report = load_one("worldpack_ablation_report_v22", WorldPackAblationReportV22)
        recomputed = evaluate_worldpack_ablation(
            spec,
            private_pack,
            direct_policy,
            memory_policy,
            direct,
            memory,
            evaluated_at=report.evaluated_at,
        )
        return (
            recomputed.report_hash == report.report_hash
            and manifest.terminal_status == report.status
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ):
        return False


def _candidates_for_arm(
    arm: Arm, data_contract_hash: str, seasonal_period: int
) -> list[ForecastCandidate]:
    return _candidates_for_policy(
        default_worldpack_arm_policy(arm, seasonal_period=seasonal_period),
        data_contract_hash,
    )


def _candidates_for_policy(
    policy: WorldPackArmPolicyV22, data_contract_hash: str
) -> list[ForecastCandidate]:
    policy.assert_sealed()
    candidates: list[ForecastCandidate] = []
    for definition in policy.candidates:
        assumptions = {
            "last_value": "The most recent level persists for one step",
            "mean_level": "The entire observed history shares one level",
            "linear_trend": "One linear trend spans the entire observed history",
            "seasonal_naive": "The frozen seasonal period repeats",
            "window_linear_trend": "Only the recent local trend persists for one step",
            "exponential_smoothing": "Recent observations deserve geometrically greater weight",
        }
        common = {
            "candidate_id": definition.candidate_id,
            "data_contract_hash": data_contract_hash,
            "family": definition.family,
            "assumptions": [assumptions[definition.family]],
            "role": "baseline" if definition.family == "last_value" else "challenger",
        }
        if policy.arm == "direct_generation" or definition.family in {
            "mean_level",
            "linear_trend",
        }:
            candidates.append(
                ForecastCandidateSpec.seal(
                    **common,
                    seasonal_period=definition.seasonal_period,
                )
            )
        else:
            candidates.append(
                ForecastCandidateSpecV22.seal(
                    **common,
                    seasonal_period=definition.seasonal_period,
                    window_length=definition.window_length,
                    smoothing_alpha=definition.smoothing_alpha,
                )
            )
    return candidates


def _candidate_by_id(
    policy: WorldPackArmPolicyV22,
    candidate_id: str,
    public_case: dict[str, object],
) -> ForecastCandidate:
    data_contract_hash = sha256_value(
        {
            "worldpack_case": public_case["case_id"],
            "observed_data_hash": public_case["observed_data_hash"],
        }
    )
    candidates = _candidates_for_policy(policy, data_contract_hash)
    matches = [candidate for candidate in candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError("selected candidate is absent from its frozen arm")
    return matches[0]


def _paired_case_bootstrap_interval(
    improvements: list[float], spec: WorldPackSpecV22
) -> tuple[float, float]:
    random = Random(spec.bootstrap_seed)
    count = len(improvements)
    draws = [
        sum(improvements[random.randrange(count)] for _ in range(count)) / count
        for _ in range(spec.bootstrap_replicates)
    ]
    alpha = (1.0 - spec.confidence_level) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _generate_values(
    mechanism: Mechanism,
    seed: int,
    total: int,
    training_points: int,
    seasonal_period: int,
) -> list[float]:
    random = Random(seed)
    values: list[float] = []
    if mechanism == "mean_reverting_control":
        current = 50.0
        for _ in range(total):
            current = 50.0 + 0.6 * (current - 50.0) + random.gauss(0.0, 0.8)
            values.append(max(0.0, current))
        return values
    break_point = training_points - 12
    for index in range(total):
        if mechanism == "local_trend_after_break":
            level = 50.0 + 0.02 * index
            if index >= break_point:
                level += 0.35 * (index - break_point)
            noise = random.gauss(0.0, 0.25)
        elif mechanism == "level_shift":
            level = 45.0 + (8.0 if index >= training_points else 0.0)
            noise = random.gauss(0.0, 0.35)
        else:
            angle = 2.0 * math.pi * index / seasonal_period
            level = 50.0 + 4.0 * math.sin(angle) + 1.5 * math.cos(angle)
            noise = random.gauss(0.0, 0.3)
        values.append(max(0.0, level + noise))
    return values
