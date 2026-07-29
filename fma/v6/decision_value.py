"""V6.2 retrospective decision-value adapter for capacity-like actions.

Prediction error is not automatically decision value.  This adapter freezes a
piecewise-linear underage/overage loss before evaluation, replays the same
outer rolling origins used by V6.1, and compares model-derived actions with a
persistence policy.  The result is local retrospective evidence only; it
cannot authorize an action or substitute for a prospective decision trial.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52

from .scientific_success import (
    ADAPTIVE_ADAPTER_ID,
    DimensionStatusV61,
    ScientificSuccessContractV61,
    _fold_forecast,
)

DECISION_INTENT_PATH = "problem/decision_value_intent_v62.json"
DECISION_CONTRACT_PATH = "docs/decision_value_contract_v62.json"
DECISION_EVIDENCE_PATH = "results/decision_value_evidence_v62.json"


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class DecisionValueIntentV62(StrictModel):
    """Value-owner input frozen with S0, before data and model results."""

    schema_version: Literal["6.2-decision-value-intent"] = (
        "6.2-decision-value-intent"
    )
    decision_id: Identifier
    value_owner_ref: Identifier
    action_unit: Identifier
    underage_unit_cost: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ]
    overage_unit_cost: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ]
    minimum_relative_loss_improvement: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.05
    maximum_mean_normalized_regret: Annotated[
        float, Field(gt=0, le=1, allow_inf_nan=False)
    ] = 0.20
    authority_scope: Literal["local_user_supplied"] = "local_user_supplied"
    external_value_owner_signature_verified: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    intent_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_intent(self) -> "DecisionValueIntentV62":
        if self.intent_hash and self.intent_hash != self.content_hash():
            raise ValueError("decision-value intent hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "intent_hash")

    def assert_sealed(self) -> None:
        if not self.intent_hash or self.intent_hash != self.content_hash():
            raise ValueError("decision-value intent is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DecisionValueIntentV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"intent_hash"})
        payload["intent_hash"] = draft.content_hash()
        return cls(**payload)


class DecisionValueContractV62(StrictModel):
    """Frozen value contract for a point forecast used as a capacity action."""

    schema_version: Literal["6.2-decision-value-contract"] = (
        "6.2-decision-value-contract"
    )
    workspace_spec_hash: Sha256
    success_contract_hash: Sha256
    intent_hash: Sha256 | None = None
    decision_id: Identifier
    value_owner_ref: Identifier = "unspecified-local-user"
    decision_kind: Literal["capacity_from_point_forecast"] = (
        "capacity_from_point_forecast"
    )
    action_unit: Identifier
    underage_unit_cost: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ]
    overage_unit_cost: Annotated[
        float, Field(gt=0, allow_inf_nan=False)
    ]
    minimum_relative_loss_improvement: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] = 0.05
    maximum_mean_normalized_regret: Annotated[
        float, Field(gt=0, le=1, allow_inf_nan=False)
    ] = 0.20
    baseline_policy: Literal["persistence"] = "persistence"
    oracle_policy: Literal["perfect_hindsight_capacity"] = (
        "perfect_hindsight_capacity"
    )
    evaluation_method: Literal["nested_rolling_origin_one_step"] = (
        "nested_rolling_origin_one_step"
    )
    private_feedback_permitted: Literal[False] = False
    external_value_owner_signature_verified: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "DecisionValueContractV62":
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("decision-value contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("decision-value contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DecisionValueContractV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


class DecisionValueEvidenceV62(StrictModel):
    schema_version: Literal["6.2-decision-value-evidence"] = (
        "6.2-decision-value-evidence"
    )
    contract_hash: Sha256
    success_contract_hash: Sha256
    snapshot_hash: Sha256
    adapter_id: Identifier
    fixture_only: bool
    status: DimensionStatusV61
    scientific_decision_status: Literal["FAIL", "NOT_RUN", "HUMAN"]
    requested_fold_count: Annotated[int, Field(ge=1)]
    completed_fold_count: Annotated[int, Field(ge=0)]
    admissible_fold_count: Annotated[int, Field(ge=0)]
    completed_origin_indices: list[Annotated[int, Field(ge=1)]]
    training_snapshot_hashes: list[Sha256]
    model_action_hashes: list[Sha256]
    checks: dict[Identifier, bool]
    metrics: dict[Identifier, float | int | None]
    thresholds: dict[Identifier, float | int]
    reason_codes: list[Identifier]
    actual_values_hash: Sha256 | None = None
    model_actions_hash: Sha256 | None = None
    baseline_actions_hash: Sha256 | None = None
    local_retrospective_only: Literal[True] = True
    prospective_trial_completed: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    evidence_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "DecisionValueEvidenceV62":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("decision-value reasons must be sorted and unique")
        if self.completed_fold_count > self.requested_fold_count:
            raise ValueError("completed decision folds exceed requested folds")
        if self.admissible_fold_count > self.completed_fold_count:
            raise ValueError("admissible decision folds exceed completed folds")
        fold_evidence_lengths = {
            len(self.completed_origin_indices),
            len(self.training_snapshot_hashes),
            len(self.model_action_hashes),
        }
        if fold_evidence_lengths != {self.completed_fold_count}:
            raise ValueError(
                "per-fold decision evidence differs from completed folds"
            )
        aggregate_hashes = (
            self.actual_values_hash,
            self.model_actions_hash,
            self.baseline_actions_hash,
        )
        if self.completed_fold_count > 0 and any(
            value is None for value in aggregate_hashes
        ):
            raise ValueError("completed decision evidence lacks aggregate hashes")
        if self.completed_fold_count == 0 and any(
            value is not None for value in aggregate_hashes
        ):
            raise ValueError(
                "empty decision evidence contains aggregate hashes"
            )
        if self.status == "PASS" and (
            not self.checks or not all(self.checks.values())
        ):
            raise ValueError("passing decision evidence contains failed checks")
        if self.status == "PASS" and (
            self.completed_fold_count != self.requested_fold_count
            or self.admissible_fold_count != self.requested_fold_count
        ):
            raise ValueError("passing decision evidence lacks requested folds")
        expected_scientific_status = (
            "NOT_RUN"
            if self.fixture_only
            else "FAIL"
            if self.status == "FAIL"
            else "NOT_RUN"
            if self.status == "NOT_RUN"
            else "HUMAN"
        )
        if self.scientific_decision_status != expected_scientific_status:
            raise ValueError("scientific decision status is overstated")
        if self.evidence_hash and self.evidence_hash != self.content_hash():
            raise ValueError("decision-value evidence hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "evidence_hash")

    @classmethod
    def seal(cls, **data: object) -> "DecisionValueEvidenceV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"evidence_hash"})
        payload["evidence_hash"] = draft.content_hash()
        return cls(**payload)


def _capacity_loss(
    *,
    action: float,
    actual: float,
    underage_unit_cost: float,
    overage_unit_cost: float,
) -> float:
    if action < actual:
        return underage_unit_cost * (actual - action)
    return overage_unit_cost * (action - actual)


def decision_contract_from_intent_v62(
    *,
    workspace_spec_hash: str,
    success_contract: ScientificSuccessContractV61,
    intent: DecisionValueIntentV62,
) -> DecisionValueContractV62:
    success_contract.thresholds.assert_sealed()
    intent.assert_sealed()
    if workspace_spec_hash != success_contract.workspace_spec_hash:
        raise ValueError("decision intent target workspace differs")
    return DecisionValueContractV62.seal(
        workspace_spec_hash=workspace_spec_hash,
        success_contract_hash=success_contract.contract_hash,
        intent_hash=intent.intent_hash,
        decision_id=intent.decision_id,
        value_owner_ref=intent.value_owner_ref,
        action_unit=intent.action_unit,
        underage_unit_cost=intent.underage_unit_cost,
        overage_unit_cost=intent.overage_unit_cost,
        minimum_relative_loss_improvement=(
            intent.minimum_relative_loss_improvement
        ),
        maximum_mean_normalized_regret=(
            intent.maximum_mean_normalized_regret
        ),
    )


def evaluate_decision_value_v62(
    *,
    snapshot: ODETimeSeriesSnapshotV52,
    success_contract: ScientificSuccessContractV61,
    decision_contract: DecisionValueContractV62,
) -> DecisionValueEvidenceV62:
    snapshot.assert_sealed()
    success_contract.thresholds.assert_sealed()
    decision_contract.assert_sealed()
    if (
        not success_contract.contract_hash
        or success_contract.contract_hash != success_contract.content_hash()
    ):
        raise ValueError("success contract is not sealed")
    if decision_contract.workspace_spec_hash != (
        success_contract.workspace_spec_hash
    ):
        raise ValueError("decision contract belongs to another workspace")
    if decision_contract.success_contract_hash != (
        success_contract.contract_hash
    ):
        raise ValueError("decision contract binds another success contract")
    if decision_contract.action_unit != snapshot.state_unit:
        raise ValueError("decision action unit differs from observed state unit")

    thresholds = success_contract.thresholds
    adaptive = success_contract.adapter_id == ADAPTIVE_ADAPTER_ID
    fold_count = (
        thresholds.adaptive_confirmation_folds
        if adaptive
        else thresholds.ode_confirmation_folds
    )
    minimum_history = (
        thresholds.adaptive_minimum_history_points
        if adaptive
        else thresholds.ode_minimum_history_points
    )
    required = minimum_history + fold_count
    common_thresholds = {
        "minimum_observation_count": required,
        "requested_fold_count": fold_count,
        "minimum_relative_loss_improvement": (
            decision_contract.minimum_relative_loss_improvement
        ),
        "maximum_mean_normalized_regret": (
            decision_contract.maximum_mean_normalized_regret
        ),
    }
    observation_count = len(snapshot.observations)
    if observation_count < required:
        return DecisionValueEvidenceV62.seal(
            contract_hash=decision_contract.contract_hash,
            success_contract_hash=success_contract.contract_hash,
            snapshot_hash=snapshot.content_hash(),
            adapter_id=success_contract.adapter_id,
            fixture_only=snapshot.fixture_only,
            status="NOT_RUN",
            scientific_decision_status="NOT_RUN",
            requested_fold_count=fold_count,
            completed_fold_count=0,
            admissible_fold_count=0,
            completed_origin_indices=[],
            training_snapshot_hashes=[],
            model_action_hashes=[],
            checks={},
            metrics={
                "minimum_observation_count": required,
                "observed_observation_count": observation_count,
            },
            thresholds=common_thresholds,
            reason_codes=["insufficient_observations"],
        )

    times = np.asarray(snapshot.times, dtype=float)
    observations = np.asarray(snapshot.observations, dtype=float)
    model_actions: list[float] = []
    baseline_actions: list[float] = []
    actuals: list[float] = []
    model_losses: list[float] = []
    baseline_losses: list[float] = []
    completed_origin_indices: list[int] = []
    training_snapshot_hashes: list[str] = []
    model_action_hashes: list[str] = []
    admissible_count = 0
    failure_count = 0
    first = observation_count - fold_count
    for index in range(first, observation_count):
        prefix = ODETimeSeriesSnapshotV52.seal(
            task_id=f"{snapshot.task_id}-v62-decision-fold-{index}",
            time_unit=snapshot.time_unit,
            state_unit=snapshot.state_unit,
            times=times[:index].tolist(),
            observations=observations[:index].tolist(),
            source_id=snapshot.source_id,
            fixture_only=snapshot.fixture_only,
        )
        try:
            model_action, _, _, _, admissible = _fold_forecast(
                adapter_id=success_contract.adapter_id,
                snapshot=prefix,
            )
        except (ArithmeticError, RuntimeError, ValueError):
            failure_count += 1
            continue
        actual = float(observations[index])
        baseline_action = float(observations[index - 1])
        if (
            not math.isfinite(model_action)
            or model_action <= 0
            or not math.isfinite(actual)
            or actual <= 0
        ):
            failure_count += 1
            continue
        model_actions.append(float(model_action))
        baseline_actions.append(baseline_action)
        actuals.append(actual)
        completed_origin_indices.append(index)
        training_snapshot_hashes.append(prefix.content_hash())
        model_action_hashes.append(sha256_value(float(model_action)))
        model_losses.append(
            _capacity_loss(
                action=float(model_action),
                actual=actual,
                underage_unit_cost=decision_contract.underage_unit_cost,
                overage_unit_cost=decision_contract.overage_unit_cost,
            )
        )
        baseline_losses.append(
            _capacity_loss(
                action=baseline_action,
                actual=actual,
                underage_unit_cost=decision_contract.underage_unit_cost,
                overage_unit_cost=decision_contract.overage_unit_cost,
            )
        )
        admissible_count += int(admissible)

    completed = len(model_actions)
    if completed == 0:
        return DecisionValueEvidenceV62.seal(
            contract_hash=decision_contract.contract_hash,
            success_contract_hash=success_contract.contract_hash,
            snapshot_hash=snapshot.content_hash(),
            adapter_id=success_contract.adapter_id,
            fixture_only=snapshot.fixture_only,
            status="FAIL",
            scientific_decision_status=(
                "NOT_RUN" if snapshot.fixture_only else "FAIL"
            ),
            requested_fold_count=fold_count,
            completed_fold_count=0,
            admissible_fold_count=0,
            completed_origin_indices=[],
            training_snapshot_hashes=[],
            model_action_hashes=[],
            checks={
                "all_requested_folds_completed": False,
                "all_inner_selections_admissible": False,
            },
            metrics={"fold_failure_count": failure_count},
            thresholds=common_thresholds,
            reason_codes=["no_valid_decision_fold"],
        )

    mean_model_loss = float(np.mean(model_losses))
    mean_baseline_loss = float(np.mean(baseline_losses))
    relative_improvement = 1.0 - mean_model_loss / max(
        mean_baseline_loss, 1e-12
    )
    loss_scale = max(
        float(np.mean(np.abs(actuals)))
        * max(
            decision_contract.underage_unit_cost,
            decision_contract.overage_unit_cost,
        ),
        1e-12,
    )
    normalized_regret = mean_model_loss / loss_scale
    checks = {
        "all_requested_folds_completed": completed == fold_count,
        "all_inner_selections_admissible": admissible_count == fold_count,
        "persistence_decision_loss_improved": (
            relative_improvement
            >= decision_contract.minimum_relative_loss_improvement
        ),
        "mean_normalized_regret_bounded": (
            normalized_regret
            <= decision_contract.maximum_mean_normalized_regret
        ),
    }
    reasons = sorted(
        check_id for check_id, passed in checks.items() if not passed
    )
    return DecisionValueEvidenceV62.seal(
        contract_hash=decision_contract.contract_hash,
        success_contract_hash=success_contract.contract_hash,
        snapshot_hash=snapshot.content_hash(),
        adapter_id=success_contract.adapter_id,
        fixture_only=snapshot.fixture_only,
        status="PASS" if all(checks.values()) else "FAIL",
        scientific_decision_status=(
            "NOT_RUN"
            if snapshot.fixture_only
            else "HUMAN"
            if all(checks.values())
            else "FAIL"
        ),
        requested_fold_count=fold_count,
        completed_fold_count=completed,
        admissible_fold_count=admissible_count,
        completed_origin_indices=completed_origin_indices,
        training_snapshot_hashes=training_snapshot_hashes,
        model_action_hashes=model_action_hashes,
        checks=checks,
        metrics={
            "mean_model_loss": mean_model_loss,
            "mean_persistence_loss": mean_baseline_loss,
            "relative_loss_improvement": relative_improvement,
            "mean_normalized_regret": normalized_regret,
            "fold_failure_count": failure_count,
        },
        thresholds=common_thresholds,
        reason_codes=reasons,
        actual_values_hash=sha256_value(actuals),
        model_actions_hash=sha256_value(model_actions),
        baseline_actions_hash=sha256_value(baseline_actions),
    )


__all__ = [
    "DECISION_CONTRACT_PATH",
    "DECISION_EVIDENCE_PATH",
    "DECISION_INTENT_PATH",
    "DecisionValueContractV62",
    "DecisionValueEvidenceV62",
    "DecisionValueIntentV62",
    "decision_contract_from_intent_v62",
    "evaluate_decision_value_v62",
]
