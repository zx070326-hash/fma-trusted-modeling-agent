from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256, _assert_timezone


DecisionTargetV31 = Literal[
    "free_run_prediction",
    "controlled_response_prediction",
]


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


def _switch_count(values: list[list[float]]) -> int:
    return sum(
        1
        for left, right in zip(values, values[1:], strict=False)
        if any(abs(a - b) > 1e-12 for a, b in zip(left, right, strict=True))
    )


class KnownActuatorMapV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    actuator_id: Identifier
    state_names: list[Identifier] = Field(min_length=1, max_length=4)
    input_names: list[Identifier] = Field(min_length=1, max_length=2)
    matrix: list[list[Annotated[float, Field(allow_inf_nan=False)]]]
    source_ref: Annotated[str, Field(min_length=3)]
    actuator_is_known: Literal[True] = True
    inference_of_actuator_permitted: Literal[False] = False
    actuator_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_actuator(self) -> "KnownActuatorMapV31":
        if len(self.matrix) != len(self.state_names):
            raise ValueError("actuator row count must equal state dimension")
        if any(len(row) != len(self.input_names) for row in self.matrix):
            raise ValueError("actuator column count must equal input dimension")
        if not any(abs(value) > 0 for row in self.matrix for value in row):
            raise ValueError("actuator map cannot be identically zero")
        if self.actuator_hash and self.actuator_hash != self.content_hash():
            raise ValueError("actuator_hash does not match known actuator map")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "actuator_hash")

    def assert_sealed(self) -> None:
        if not self.actuator_hash or self.actuator_hash != self.content_hash():
            raise ValueError("known actuator map is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "KnownActuatorMapV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"actuator_hash"}),
            actuator_hash=draft.content_hash(),
        )


class PiecewiseConstantInputActionV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    action_id: Identifier
    actuator_hash: Sha256
    segment_duration: Annotated[float, Field(gt=0, le=2, allow_inf_nan=False)]
    input_values: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=2, max_length=16
    )
    peak_amplitude: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    total_energy: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    switch_count: Annotated[int, Field(ge=1, le=15)]
    action_cost: Literal[1] = 1
    real_world_action_authorized: Literal[False] = False
    action_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "PiecewiseConstantInputActionV31":
        width = len(self.input_values[0])
        if width < 1 or any(len(row) != width for row in self.input_values):
            raise ValueError("piecewise input rows must have one common positive width")
        peak = max(abs(value) for row in self.input_values for value in row)
        energy = self.segment_duration * sum(
            value * value for row in self.input_values for value in row
        )
        switches = _switch_count(self.input_values)
        if not math.isclose(self.peak_amplitude, peak, rel_tol=0, abs_tol=1e-12):
            raise ValueError("declared peak amplitude does not match input values")
        if not math.isclose(self.total_energy, energy, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("declared energy does not match input values")
        if self.switch_count != switches:
            raise ValueError("declared switch count does not match input values")
        if self.action_hash and self.action_hash != self.content_hash():
            raise ValueError("action_hash does not match controlled input action")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "action_hash")

    def assert_sealed(self) -> None:
        if not self.action_hash or self.action_hash != self.content_hash():
            raise ValueError("controlled input action is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PiecewiseConstantInputActionV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"action_hash"}),
            action_hash=draft.content_hash(),
        )


class ExperimentConstraintEnvelopeV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    envelope_id: Identifier
    actuator_hash: Sha256
    state_lower_bounds: list[Annotated[float, Field(allow_inf_nan=False)]]
    state_upper_bounds: list[Annotated[float, Field(allow_inf_nan=False)]]
    required_peak_amplitude: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    required_total_energy: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    required_switch_count: Annotated[int, Field(ge=1)]
    maximum_empirical_prediction_risk: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    action_cost: Literal[1] = 1
    safety_claim: Literal["empirical_proxy_not_formal_guarantee"] = (
        "empirical_proxy_not_formal_guarantee"
    )
    real_world_execution_permitted: Literal[False] = False
    envelope_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_envelope(self) -> "ExperimentConstraintEnvelopeV31":
        if len(self.state_lower_bounds) != len(self.state_upper_bounds):
            raise ValueError("state-bound dimensions differ")
        if any(low >= high for low, high in zip(
            self.state_lower_bounds, self.state_upper_bounds, strict=True
        )):
            raise ValueError("state bounds must be ordered")
        if self.envelope_hash and self.envelope_hash != self.content_hash():
            raise ValueError("envelope_hash does not match experiment envelope")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "envelope_hash")

    def assert_sealed(self) -> None:
        if not self.envelope_hash or self.envelope_hash != self.content_hash():
            raise ValueError("experiment envelope is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExperimentConstraintEnvelopeV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"envelope_hash"}),
            envelope_hash=draft.content_hash(),
        )


class ExperimentAcquisitionReceiptV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    acquisition_id: Identifier
    case_id: Identifier
    action_hash: Sha256
    decision_target: DecisionTargetV31
    d_optimal_gain: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    model_disagreement: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    decision_information: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    action_cost: Literal[1] = 1
    empirical_prediction_risk: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ]
    utility_score: Annotated[float, Field(allow_inf_nan=False)]
    admissible: bool
    gate_codes: list[Literal[
        "known_actuator",
        "peak_equal",
        "energy_equal",
        "switch_equal",
        "cost_equal",
        "empirical_risk_pass",
        "empirical_risk_fail",
    ]] = Field(min_length=6, max_length=6)
    formal_safety_proven: Literal[False] = False
    acquisition_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_acquisition(self) -> "ExperimentAcquisitionReceiptV31":
        risk_pass = "empirical_risk_pass" in self.gate_codes
        if self.admissible != risk_pass:
            raise ValueError("acquisition admissibility must follow risk gate")
        if self.acquisition_hash and self.acquisition_hash != self.content_hash():
            raise ValueError("acquisition_hash does not match receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "acquisition_hash")

    def assert_sealed(self) -> None:
        if not self.acquisition_hash or self.acquisition_hash != self.content_hash():
            raise ValueError("acquisition receipt is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExperimentAcquisitionReceiptV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"acquisition_hash"}),
            acquisition_hash=draft.content_hash(),
        )


class ExperimentPermissionDecisionV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    acquisition_hash: Sha256
    envelope_hash: Sha256
    decision: Literal["allow_synthetic", "deny", "abstain"]
    policy_rule: Literal[
        "allow_bounded_synthetic_experiment",
        "deny_contract_mismatch",
        "deny_data_quality",
        "deny_budget",
        "abstain_no_admissible_action",
    ]
    # The operational receipt is shared by V3.1 (two-step) and the additive
    # V3.1.1 horizon evolution (three-step).  Expanding this validation bound
    # does not change any serialized V3.1 artifact or content hash.
    budget_before: Annotated[int, Field(ge=0, le=3)]
    budget_after: Annotated[int, Field(ge=0, le=3)]
    decided_at: datetime
    permission_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_permission(self) -> "ExperimentPermissionDecisionV31":
        _assert_timezone(self.decided_at, "decided_at")
        consumed = self.budget_before - self.budget_after
        if (self.decision == "allow_synthetic") != (consumed == 1):
            raise ValueError("only allowed synthetic experiments consume one budget unit")
        if self.permission_hash and self.permission_hash != self.content_hash():
            raise ValueError("permission_hash does not match permission decision")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "permission_hash")

    def assert_sealed(self) -> None:
        if not self.permission_hash or self.permission_hash != self.content_hash():
            raise ValueError("experiment permission is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ExperimentPermissionDecisionV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"permission_hash"}),
            permission_hash=draft.content_hash(),
        )


class ControlledObservationReceiptV31(StrictModel):
    schema_version: Literal["3.1"] = "3.1"
    observation_id: Identifier
    case_id: Identifier
    action_hash: Sha256
    actuator_hash: Sha256
    times: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(min_length=9)
    states: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=9
    )
    inputs: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(
        min_length=9
    )
    empirical_peak_state_ratio: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    quality_flags: list[Identifier] = Field(default_factory=list)
    trust_class: Literal["untrusted_synthetic_observation"] = (
        "untrusted_synthetic_observation"
    )
    real_world_action_executed: Literal[False] = False
    observed_at: datetime
    observation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> "ControlledObservationReceiptV31":
        _assert_timezone(self.observed_at, "observed_at")
        if len(self.times) != len(self.states) or len(self.times) != len(self.inputs):
            raise ValueError("observation time, state, and input lengths differ")
        if any(right <= left for left, right in zip(self.times, self.times[1:])):
            raise ValueError("observation times must strictly increase")
        if any(len(row) != len(self.states[0]) for row in self.states):
            raise ValueError("observation state dimensions differ")
        if any(len(row) != len(self.inputs[0]) for row in self.inputs):
            raise ValueError("observation input dimensions differ")
        if self.observation_hash and self.observation_hash != self.content_hash():
            raise ValueError("observation_hash does not match controlled observation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "observation_hash")

    def assert_sealed(self) -> None:
        if not self.observation_hash or self.observation_hash != self.content_hash():
            raise ValueError("controlled observation is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "ControlledObservationReceiptV31":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"observation_hash"}),
            observation_hash=draft.content_hash(),
        )


def validate_action_against_envelope_v31(
    action: PiecewiseConstantInputActionV31,
    actuator: KnownActuatorMapV31,
    envelope: ExperimentConstraintEnvelopeV31,
) -> list[str]:
    action.assert_sealed()
    actuator.assert_sealed()
    envelope.assert_sealed()
    failures: list[str] = []
    if action.actuator_hash != actuator.actuator_hash or envelope.actuator_hash != actuator.actuator_hash:
        failures.append("actuator_mismatch")
    if len(action.input_values[0]) != len(actuator.input_names):
        failures.append("input_dimension_mismatch")
    if not math.isclose(action.peak_amplitude, envelope.required_peak_amplitude, abs_tol=1e-12):
        failures.append("peak_mismatch")
    if not math.isclose(action.total_energy, envelope.required_total_energy, abs_tol=1e-12):
        failures.append("energy_mismatch")
    if action.switch_count != envelope.required_switch_count:
        failures.append("switch_mismatch")
    if action.action_cost != envelope.action_cost:
        failures.append("cost_mismatch")
    return failures
