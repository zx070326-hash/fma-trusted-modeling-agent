from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import beta

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .forecast_evaluator import forecast_one
from .schemas import Identifier, Sha256, _assert_timezone
from .worldpack import (
    Arm,
    Mechanism,
    PrivateWorldPackV22,
    WorldPackArmPolicyV22,
    WorldPackCaseResultV22,
    WorldPackSelectionBundleV22,
    _candidate_by_id,
    default_worldpack_arm_policy,
    generate_private_worldpack,
    select_worldpack_candidate,
)


INITIAL_CONFIRMATION_SEEDS = (
    101,
    137,
    173,
    211,
    251,
    293,
    337,
    383,
    431,
    479,
    521,
    569,
    617,
    661,
    709,
    757,
    809,
    857,
    907,
    953,
)
SAFE_POLICY_CONFIRMATION_SEEDS = (
    1009,
    1061,
    1117,
    1171,
    1223,
    1277,
    1327,
    1381,
    1433,
    1487,
    1543,
    1597,
    1657,
    1709,
    1759,
    1811,
    1871,
    1931,
    1993,
    2053,
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class WorldPackConfirmationSpecV23(StrictModel):
    """Prospective confirmation protocol designed after, but disjoint from, V2.2."""

    schema_version: Literal["2.3"] = "2.3"
    pack_id: Identifier
    prior_exploratory_report_hash: Sha256
    mechanisms: list[Mechanism] = Field(min_length=4, max_length=4)
    seeds: list[Annotated[int, Field(ge=0, le=2_147_483_647)]] = Field(
        min_length=20, max_length=64
    )
    training_points: Annotated[int, Field(ge=48, le=10_000)]
    inner_validation_points: Annotated[int, Field(ge=12, le=1_000)]
    outer_holdout_points: Annotated[int, Field(ge=12, le=1_000)]
    seasonal_period: Annotated[int, Field(ge=2, le=365)]
    candidate_budget_per_arm: Literal[4] = 4
    confidence_level: Annotated[float, Field(gt=0.8, lt=1, allow_inf_nan=False)]
    bootstrap_replicates: Annotated[int, Field(ge=5_000, le=100_000)]
    bootstrap_seed: Annotated[int, Field(ge=0, le=2_147_483_647)]
    negative_transfer_relative_margin: Annotated[
        float, Field(gt=0, lt=1, allow_inf_nan=False)
    ]
    maximum_negative_transfer_rate_upper: Annotated[
        float, Field(gt=0, lt=0.5, allow_inf_nan=False)
    ]
    mechanism_noninferiority_margin: Annotated[
        float, Field(gt=0, lt=0.5, allow_inf_nan=False)
    ]
    frozen_at: datetime
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "WorldPackConfirmationSpecV23":
        _assert_timezone(self.frozen_at, "frozen_at")
        if set(self.mechanisms) != {
            "local_trend_after_break",
            "level_shift",
            "seasonal_regime",
            "mean_reverting_control",
        }:
            raise ValueError("V2.3 confirmation needs the four frozen mechanisms")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("V2.3 confirmation seeds must be unique")
        if {11, 29, 47} & set(self.seeds):
            raise ValueError("V2.3 confirmation seeds overlap the V2.2 exploratory seeds")
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("spec_hash does not match V2.3 confirmation spec")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("V2.3 confirmation spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackConfirmationSpecV23":
        data.setdefault("frozen_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"spec_hash"}),
            spec_hash=draft.content_hash(),
        )


class MechanismConfirmationResultV23(StrictModel):
    mechanism: Mechanism
    case_count: Annotated[int, Field(ge=20)]
    mean_relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    confidence_upper: Annotated[float, Field(allow_inf_nan=False)]
    noninferiority_margin: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    noninferior: bool

    @model_validator(mode="after")
    def validate_interval(self) -> "MechanismConfirmationResultV23":
        if self.confidence_lower > self.confidence_upper:
            raise ValueError("mechanism confidence interval is reversed")
        return self


class WorldPackConfirmationReportV23(StrictModel):
    schema_version: Literal["2.3"] = "2.3"
    report_id: Identifier
    confirmation_spec_hash: Sha256
    prior_exploratory_report_hash: Sha256
    private_pack_hash: Sha256
    direct_policy_hash: Sha256
    memory_policy_hash: Sha256
    direct_bundle_hash: Sha256
    memory_bundle_hash: Sha256
    cases: list[WorldPackCaseResultV22] = Field(min_length=80)
    mechanism_results: list[MechanismConfirmationResultV23] = Field(
        min_length=4, max_length=4
    )
    macro_mean_relative_improvement: Annotated[float, Field(allow_inf_nan=False)]
    macro_confidence_lower: Annotated[float, Field(allow_inf_nan=False)]
    macro_confidence_upper: Annotated[float, Field(allow_inf_nan=False)]
    win_count: Annotated[int, Field(ge=0)]
    tie_count: Annotated[int, Field(ge=0)]
    loss_count: Annotated[int, Field(ge=0)]
    negative_transfer_count: Annotated[int, Field(ge=0)]
    negative_transfer_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    negative_transfer_rate_upper: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    same_budget: bool
    status: Literal["promoted_for_worldpack_scope_v23", "candidate_rejected_v23"]
    reason_codes: list[Literal[
        "macro_improvement_interval_not_positive",
        "mechanism_noninferiority_failed",
        "negative_transfer_rate_bound_failed",
        "candidate_budget_mismatch",
    ]]
    evaluated_at: datetime
    report_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> "WorldPackConfirmationReportV23":
        _assert_timezone(self.evaluated_at, "evaluated_at")
        if len({result.mechanism for result in self.mechanism_results}) != 4:
            raise ValueError("V2.3 report needs four unique mechanism results")
        if self.win_count + self.tie_count + self.loss_count != len(self.cases):
            raise ValueError("V2.3 outcome counts do not cover all cases")
        if (self.status == "promoted_for_worldpack_scope_v23") == bool(
            self.reason_codes
        ):
            raise ValueError("promoted reports need no reasons; rejected reports need reasons")
        if self.report_hash and self.report_hash != self.content_hash():
            raise ValueError("report_hash does not match V2.3 confirmation report")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "report_hash")

    def assert_sealed(self) -> None:
        if not self.report_hash or self.report_hash != self.content_hash():
            raise ValueError("V2.3 confirmation report is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackConfirmationReportV23":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"report_hash"}),
            report_hash=draft.content_hash(),
        )


class WorldPackConfirmationManifestV23(StrictModel):
    schema_version: Literal["2.3"] = "2.3"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    artifact_refs: list[ArtifactRef] = Field(min_length=7, max_length=7)
    terminal_status: Literal[
        "promoted_for_worldpack_scope_v23", "candidate_rejected_v23"
    ]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "WorldPackConfirmationManifestV23":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != 7:
            raise ValueError("V2.3 manifest references must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match V2.3 confirmation manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("V2.3 confirmation manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WorldPackConfirmationManifestV23":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


class OperatorQualificationV23(StrictModel):
    """Code-owned qualification for one exact policy and one limited scope."""

    schema_version: Literal["2.3"] = "2.3"
    qualification_id: Identifier
    policy_hash: Sha256
    confirmation_report_hash: Sha256
    qualification_scope: Literal["synthetic_forecast_worldpack_v23"] = (
        "synthetic_forecast_worldpack_v23"
    )
    status: Literal["qualified"] = "qualified"
    limitations: list[Annotated[str, Field(min_length=3)]] = Field(min_length=3)
    qualified_at: datetime
    qualification_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_qualification(self) -> "OperatorQualificationV23":
        _assert_timezone(self.qualified_at, "qualified_at")
        if self.qualification_hash and self.qualification_hash != self.content_hash():
            raise ValueError("qualification_hash does not match operator qualification")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "qualification_hash")

    def assert_sealed(self) -> None:
        if not self.qualification_hash or self.qualification_hash != self.content_hash():
            raise ValueError("operator qualification is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "OperatorQualificationV23":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"qualification_hash"}),
            qualification_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class WorldPackConfirmationOutcome:
    store: RunStore
    spec: WorldPackConfirmationSpecV23
    private_pack: PrivateWorldPackV22
    direct_policy: WorldPackArmPolicyV22
    memory_policy: WorldPackArmPolicyV22
    direct_selections: WorldPackSelectionBundleV22
    memory_selections: WorldPackSelectionBundleV22
    report: WorldPackConfirmationReportV23
    manifest: WorldPackConfirmationManifestV23


def qualify_worldpack_policy_v23(
    policy: WorldPackArmPolicyV22,
    report: WorldPackConfirmationReportV23,
) -> OperatorQualificationV23:
    policy.assert_sealed()
    report.assert_sealed()
    if report.status != "promoted_for_worldpack_scope_v23":
        raise ValueError("a rejected WorldPack policy cannot be qualified")
    if report.memory_policy_hash != policy.policy_hash:
        raise ValueError("WorldPack report is bound to another memory policy")
    assert policy.policy_hash is not None
    assert report.report_hash is not None
    return OperatorQualificationV23.seal(
        qualification_id=f"{policy.policy_id}_qualification",
        policy_hash=policy.policy_hash,
        confirmation_report_hash=report.report_hash,
        limitations=[
            "valid only for the frozen synthetic forecast WorldPack V2.3 scope",
            "does not establish real-data external validity or decision eligibility",
            "requires active source lineage and an independently replayable report",
        ],
        qualified_at=report.evaluated_at,
    )


def default_confirmation_spec_v23(
    *,
    prior_exploratory_report_hash: str,
    frozen_at: datetime | None = None,
) -> WorldPackConfirmationSpecV23:
    return WorldPackConfirmationSpecV23.seal(
        pack_id="forecast_adaptation_confirmation_v23",
        prior_exploratory_report_hash=prior_exploratory_report_hash,
        mechanisms=[
            "local_trend_after_break",
            "level_shift",
            "seasonal_regime",
            "mean_reverting_control",
        ],
        seeds=list(INITIAL_CONFIRMATION_SEEDS),
        training_points=84,
        inner_validation_points=24,
        outer_holdout_points=24,
        seasonal_period=12,
        confidence_level=0.95,
        bootstrap_replicates=10_000,
        bootstrap_seed=230_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def safe_policy_confirmation_spec_v23(
    *,
    prior_confirmation_report_hash: str,
    frozen_at: datetime | None = None,
) -> WorldPackConfirmationSpecV23:
    """Second, disjoint confirmation tranche for the failure-evolved policy.

    ``prior_exploratory_report_hash`` is retained as the V2.3 wire field for
    hash compatibility; in this run it points to the immediately preceding
    confirmation report that supplied the failure signature.
    """

    if set(INITIAL_CONFIRMATION_SEEDS) & set(SAFE_POLICY_CONFIRMATION_SEEDS):
        raise RuntimeError("safe-policy confirmation seeds overlap prior confirmation")
    return WorldPackConfirmationSpecV23.seal(
        pack_id="forecast_safe_policy_confirmation_v23",
        prior_exploratory_report_hash=prior_confirmation_report_hash,
        mechanisms=[
            "local_trend_after_break",
            "level_shift",
            "seasonal_regime",
            "mean_reverting_control",
        ],
        seeds=list(SAFE_POLICY_CONFIRMATION_SEEDS),
        training_points=84,
        inner_validation_points=24,
        outer_holdout_points=24,
        seasonal_period=12,
        confidence_level=0.95,
        bootstrap_replicates=10_000,
        bootstrap_seed=240_722,
        negative_transfer_relative_margin=0.05,
        maximum_negative_transfer_rate_upper=0.05,
        mechanism_noninferiority_margin=0.05,
        frozen_at=frozen_at or datetime.now(timezone.utc),
    )


def evaluate_worldpack_confirmation_v23(
    spec: WorldPackConfirmationSpecV23,
    private_pack: PrivateWorldPackV22,
    direct_policy: WorldPackArmPolicyV22,
    memory_policy: WorldPackArmPolicyV22,
    direct: WorldPackSelectionBundleV22,
    memory: WorldPackSelectionBundleV22,
    *,
    evaluated_at: datetime | None = None,
) -> WorldPackConfirmationReportV23:
    spec.assert_sealed()
    private_pack.assert_sealed()
    direct_policy.assert_sealed()
    memory_policy.assert_sealed()
    direct.assert_sealed()
    memory.assert_sealed()
    if private_pack.pack_spec_hash != spec.spec_hash:
        raise ValueError("V2.3 private WorldPack is bound to another spec")
    if (
        direct_policy.arm != "direct_generation"
        or memory_policy.arm != "retrieval_evolution_memory"
        or direct.arm_policy_hash != direct_policy.policy_hash
        or memory.arm_policy_hash != memory_policy.policy_hash
    ):
        raise ValueError("V2.3 selections are bound to another arm policy")
    direct_by_case = {receipt.case_id: receipt for receipt in direct.receipts}
    memory_by_case = {receipt.case_id: receipt for receipt in memory.receipts}
    case_ids = {case.case_id for case in private_pack.cases}
    if set(direct_by_case) != case_ids or set(memory_by_case) != case_ids:
        raise ValueError("V2.3 selections do not cover the private WorldPack")

    results: list[WorldPackCaseResultV22] = []
    for case in private_pack.cases:
        public = case.public_projection(spec)  # compatible sealed projection contract
        direct_receipt = direct_by_case[case.case_id]
        memory_receipt = memory_by_case[case.case_id]
        expected_direct = select_worldpack_candidate(
            public, "direct_generation", direct_policy
        )
        expected_memory = select_worldpack_candidate(
            public, "retrieval_evolution_memory", memory_policy
        )
        if direct_receipt.receipt_hash != expected_direct.receipt_hash:
            raise ValueError("V2.3 direct selection does not replay")
        if memory_receipt.receipt_hash != expected_memory.receipt_hash:
            raise ValueError("V2.3 memory selection does not replay")
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
        relative = improvement / direct_mae if direct_mae > 1e-15 else 0.0
        tolerance = max(1e-12, direct_mae * 1e-9)
        outcome = (
            "memory_win"
            if improvement > tolerance
            else "memory_loss"
            if improvement < -tolerance
            else "tie"
        )
        negative = memory_mae > direct_mae * (
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
                relative_improvement=relative,
                outcome=outcome,
                negative_transfer=negative,
                direct_evaluation_count=direct_receipt.forecast_evaluation_count,
                memory_evaluation_count=memory_receipt.forecast_evaluation_count,
            )
        )

    grouped: dict[str, list[float]] = {mechanism: [] for mechanism in spec.mechanisms}
    for result in results:
        assert result.relative_improvement is not None
        grouped[result.mechanism].append(result.relative_improvement)
    macro_draws = _stratified_macro_bootstrap(grouped, spec)
    alpha = (1.0 - spec.confidence_level) / 2.0
    macro_lower, macro_upper = np.quantile(
        macro_draws, [alpha, 1.0 - alpha], method="linear"
    )
    mechanism_results: list[MechanismConfirmationResultV23] = []
    for index, mechanism in enumerate(spec.mechanisms):
        values = grouped[mechanism]
        draws = _bootstrap_means(
            values,
            spec.bootstrap_replicates,
            spec.bootstrap_seed + 10_000 + index,
        )
        lower, upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
        mechanism_results.append(
            MechanismConfirmationResultV23(
                mechanism=mechanism,
                case_count=len(values),
                mean_relative_improvement=float(np.mean(values)),
                confidence_lower=float(lower),
                confidence_upper=float(upper),
                noninferiority_margin=spec.mechanism_noninferiority_margin,
                noninferior=float(lower) >= -spec.mechanism_noninferiority_margin,
            )
        )
    negative_count = sum(result.negative_transfer for result in results)
    total = len(results)
    negative_rate = negative_count / total
    negative_upper = _one_sided_binomial_upper(
        negative_count, total, spec.confidence_level
    )
    same_budget = all(
        result.direct_evaluation_count == result.memory_evaluation_count
        for result in results
    )
    reasons: list[str] = []
    if float(macro_lower) <= 0:
        reasons.append("macro_improvement_interval_not_positive")
    if not all(result.noninferior for result in mechanism_results):
        reasons.append("mechanism_noninferiority_failed")
    if negative_upper > spec.maximum_negative_transfer_rate_upper:
        reasons.append("negative_transfer_rate_bound_failed")
    if not same_budget:
        reasons.append("candidate_budget_mismatch")
    assert spec.spec_hash is not None
    assert private_pack.private_pack_hash is not None
    assert direct_policy.policy_hash is not None
    assert memory_policy.policy_hash is not None
    assert direct.bundle_hash is not None
    assert memory.bundle_hash is not None
    return WorldPackConfirmationReportV23.seal(
        report_id="forecast_memory_prospective_confirmation",
        confirmation_spec_hash=spec.spec_hash,
        prior_exploratory_report_hash=spec.prior_exploratory_report_hash,
        private_pack_hash=private_pack.private_pack_hash,
        direct_policy_hash=direct_policy.policy_hash,
        memory_policy_hash=memory_policy.policy_hash,
        direct_bundle_hash=direct.bundle_hash,
        memory_bundle_hash=memory.bundle_hash,
        cases=results,
        mechanism_results=mechanism_results,
        macro_mean_relative_improvement=float(
            np.mean([np.mean(grouped[mechanism]) for mechanism in spec.mechanisms])
        ),
        macro_confidence_lower=float(macro_lower),
        macro_confidence_upper=float(macro_upper),
        win_count=sum(result.outcome == "memory_win" for result in results),
        tie_count=sum(result.outcome == "tie" for result in results),
        loss_count=sum(result.outcome == "memory_loss" for result in results),
        negative_transfer_count=negative_count,
        negative_transfer_rate=negative_rate,
        negative_transfer_rate_upper=negative_upper,
        same_budget=same_budget,
        status=(
            "candidate_rejected_v23"
            if reasons
            else "promoted_for_worldpack_scope_v23"
        ),
        reason_codes=reasons,
        evaluated_at=evaluated_at or datetime.now(timezone.utc),
    )


def run_worldpack_confirmation_v23(
    output_root: str | Path,
    *,
    spec: WorldPackConfirmationSpecV23,
    memory_policy: WorldPackArmPolicyV22 | None = None,
    evaluated_at: datetime | None = None,
    run_id: str | None = None,
) -> WorldPackConfirmationOutcome:
    at = evaluated_at or datetime.now(timezone.utc)
    spec.assert_sealed()
    direct_policy = default_worldpack_arm_policy(
        "direct_generation", seasonal_period=spec.seasonal_period
    )
    memory_policy = memory_policy or default_worldpack_arm_policy(
        "retrieval_evolution_memory", seasonal_period=spec.seasonal_period
    )
    memory_policy.assert_sealed()
    if memory_policy.arm != "retrieval_evolution_memory":
        raise ValueError("V2.3 memory policy belongs to another arm")
    store = RunStore(
        output_root,
        run_id=run_id or f"worldpack-confirmation-v23-{uuid4().hex[:10]}",
    )
    spec_ref = store.put_artifact("worldpack_confirmation_spec_v23", spec)
    direct_policy_ref = store.put_artifact("worldpack_direct_policy_v22", direct_policy)
    memory_policy_ref = store.put_artifact("worldpack_memory_policy_v22", memory_policy)
    store.emit(
        "worldpack_confirmation_protocol_frozen",
        {
            "spec_ref": spec_ref.model_dump(mode="json"),
            "direct_policy_ref": direct_policy_ref.model_dump(mode="json"),
            "memory_policy_ref": memory_policy_ref.model_dump(mode="json"),
        },
    )

    private_pack = generate_private_worldpack(spec)  # stable V2.2 case envelope
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
    report = evaluate_worldpack_confirmation_v23(
        spec,
        private_pack,
        direct_policy,
        memory_policy,
        direct,
        memory,
        evaluated_at=at,
    )
    remaining_refs = [
        store.put_artifact("private_worldpack_v22", private_pack),
        store.put_artifact("worldpack_direct_selections_v22", direct),
        store.put_artifact("worldpack_memory_selections_v22", memory),
        store.put_artifact("worldpack_confirmation_report_v23", report),
    ]
    manifest = WorldPackConfirmationManifestV23.seal(
        run_id=store.run_id,
        artifact_refs=[
            spec_ref,
            direct_policy_ref,
            memory_policy_ref,
            *remaining_refs,
        ],
        terminal_status=report.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("worldpack_confirmation_manifest_v23", manifest)
    store.emit(
        "worldpack_confirmation_completed",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_worldpack_confirmation_v23(store.run_directory):
        raise RuntimeError("V2.3 WorldPack confirmation failed independent verification")
    return WorldPackConfirmationOutcome(
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


def verify_worldpack_confirmation_v23(run_directory: str | Path) -> bool:
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
        frozen_events = [
            event
            for event in events
            if event["event_type"] == "worldpack_confirmation_protocol_frozen"
        ]
        if len(frozen_events) != 1:
            return False
        frozen = frozen_events[0]
        frozen_refs = {
            key: ArtifactRef.model_validate(value)
            for key, value in frozen["payload"].items()
        }
        if set(frozen_refs) != {"spec_ref", "direct_policy_ref", "memory_policy_ref"}:
            return False
        private_commits = [
            event
            for event in events
            if event["event_type"] == "artifact_committed"
            and event["payload"].get("kind") == "private_worldpack_v22"
        ]
        if len(private_commits) != 1 or frozen["sequence"] >= private_commits[0]["sequence"]:
            return False

        manifests = [
            ref for ref in refs if ref.kind == "worldpack_confirmation_manifest_v23"
        ]
        if len(manifests) != 1:
            return False
        manifest = WorldPackConfirmationManifestV23.model_validate(
            store.load_artifact(manifests[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            matches = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(matches) != 1:
                raise RuntimeError(f"V2.3 manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(matches[0]))

        spec = load_one("worldpack_confirmation_spec_v23", WorldPackConfirmationSpecV23)
        private_pack = load_one("private_worldpack_v22", PrivateWorldPackV22)
        direct_policy = load_one("worldpack_direct_policy_v22", WorldPackArmPolicyV22)
        memory_policy = load_one("worldpack_memory_policy_v22", WorldPackArmPolicyV22)
        direct = load_one("worldpack_direct_selections_v22", WorldPackSelectionBundleV22)
        memory = load_one("worldpack_memory_selections_v22", WorldPackSelectionBundleV22)
        report = load_one(
            "worldpack_confirmation_report_v23", WorldPackConfirmationReportV23
        )
        if (
            frozen_refs["spec_ref"].sha256
            != next(
                ref.sha256
                for ref in manifest.artifact_refs
                if ref.kind == "worldpack_confirmation_spec_v23"
            )
            or frozen_refs["direct_policy_ref"].sha256
            != next(
                ref.sha256
                for ref in manifest.artifact_refs
                if ref.kind == "worldpack_direct_policy_v22"
            )
            or frozen_refs["memory_policy_ref"].sha256
            != next(
                ref.sha256
                for ref in manifest.artifact_refs
                if ref.kind == "worldpack_memory_policy_v22"
            )
        ):
            return False
        recomputed = evaluate_worldpack_confirmation_v23(
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
        StopIteration,
        json.JSONDecodeError,
    ):
        return False


def _stratified_macro_bootstrap(
    grouped: dict[str, list[float]], spec: WorldPackConfirmationSpecV23
) -> np.ndarray:
    random = Random(spec.bootstrap_seed)
    mechanisms = list(spec.mechanisms)
    draws: list[float] = []
    for _ in range(spec.bootstrap_replicates):
        mechanism_means: list[float] = []
        for mechanism in mechanisms:
            values = grouped[mechanism]
            mechanism_means.append(
                sum(values[random.randrange(len(values))] for _ in values) / len(values)
            )
        draws.append(sum(mechanism_means) / len(mechanism_means))
    return np.asarray(draws, dtype=float)


def _bootstrap_means(values: list[float], replicates: int, seed: int) -> np.ndarray:
    random = Random(seed)
    count = len(values)
    return np.asarray(
        [
            sum(values[random.randrange(count)] for _ in range(count)) / count
            for _ in range(replicates)
        ],
        dtype=float,
    )


def _one_sided_binomial_upper(successes: int, trials: int, confidence: float) -> float:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial counts")
    if successes == trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))
