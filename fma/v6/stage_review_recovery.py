"""V6.6 typed pre-data S0 contract and bounded review-recovery inputs.

This module is deliberately additive.  It does not alter V5 artifacts or make
an S0 workflow gate scientific evidence.  Code freezes the evaluation profile
and canary tolerance; model processes may only propose concise diagnostic
drafts and fixed review codes.

Only defects that the harness can reproduce from the current public S0
artifacts are eligible for one semantic repair.  Reviewer prose, private
evidence, holdout outcomes, and unclassified semantic objections never enter a
repair context.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.workspace_schemas import (
    DecisionFunctionCanaryV50,
    DecisionFunctionSpecV50,
    RegimeDiagnosisV50,
)
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_7.adaptive_positive_series import AdaptiveThresholdsV57
from fma.v6.scientific_success import ScientificSuccessThresholdsV61


S0_EVALUATION_PROFILE_PATH_V66 = "docs/s0_evaluation_profile_v66.json"
S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66 = 3_000
S0_NARRATIVE_MAX_CHARACTERS_V66 = 200
S0_CANARY_TOLERANCE_V66 = 1e-9
S0_MAX_AUTOMATIC_SEMANTIC_REPAIRS_V66 = 1

ODE_ADAPTER_ID_V66 = "scalar_autonomous_ode_v52"
ADAPTIVE_ADAPTER_ID_V66 = "adaptive_positive_series_v57"

# These commitments intentionally fail closed if an older threshold class is
# silently reinterpreted.  Changing one requires a new additive profile.
ODE_THRESHOLDS_HASH_V66 = (
    "949d45224243991382885c0d7bf9453c65eacd14b7d2ae4909df5ffedf3d2861"
)
ADAPTIVE_THRESHOLDS_HASH_V66 = (
    "2beec5a8cedb8cc947ee1dd8e24ca1c5f0a6c87b00edd3d8f8c4d8019989db7f"
)
SCIENTIFIC_SUCCESS_THRESHOLDS_HASH_V66 = (
    "a805614a071c9a56bbfaef82a8870363b88ad1d8ed0d8883aa7e5b5d768208b7"
)

S0ArtifactPathV66 = Literal[
    "problem/contract.json",
    "problem/decision_function.json",
    "docs/regime.json",
    "docs/s0_evaluation_profile_v66.json",
]
S0FindingSeverityV66 = Literal["BLOCKING", "HUMAN"]
S0FindingVerificationV66 = Literal[
    "MECHANICALLY_VERIFIED",
    "HUMAN_ONLY",
    "UNCLASSIFIED",
]
S0RepairOperatorV66 = Literal[
    "COMPLETE_REQUIRED_FIELD",
    "COMPLETE_SENTENCE",
    "RESTORE_FROZEN_PROFILE_BINDING",
    "REMOVE_MODEL_CONTROLLED_TOLERANCE",
    "ALIGN_CANARY_ARITY",
    "DEDUPLICATE_DECISION_INPUTS",
    "COMPRESS_TO_ARTIFACT_ENVELOPE",
]
S0RepairAuthorizationReasonV66 = Literal[
    "AUTHORIZED",
    "HOLDOUT_EXPOSED",
    "PRIVATE_EVIDENCE_USED",
    "NON_AUTOREPAIRABLE_FINDING",
    "REPEATED_FAILURE_SIGNATURE",
    "REPAIR_BUDGET_EXHAUSTED",
]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _adaptive_thresholds_v66() -> AdaptiveThresholdsV57:
    """Reconstruct the V5.7 frozen thresholds without mutable caller input."""

    thresholds = AdaptiveThresholdsV57.seal(
        split_fraction=0.7,
        minimum_points_per_slice=8,
        maximum_validation_relative_rmse=0.15,
        minimum_persistence_relative_improvement=0.1,
        maximum_innovation_absolute_lag1_correlation=0.35,
        maximum_absolute_growth_ar1_phi=0.95,
        minimum_growth_ar1_validation_relative_improvement=0.05,
        maximum_growth_phi_window_range=0.3,
        maximum_growth_drift_window_range_standardized=1.0,
        maximum_innovation_mean_shift_standardized=1.5,
        maximum_single_innovation_standardized=5.0,
        minimum_validation_interval_coverage=0.5,
        maximum_absolute_mean_log_growth=0.5,
        selection_complexity_penalty_per_parameter=0.002,
        bootstrap_replicates=40,
        bootstrap_seed=155921,
        minimum_bootstrap_success_fraction=0.8,
        maximum_forecast_interval_relative_width=2.0,
        maximum_window_sensitivity_relative_range=1.0,
    )
    thresholds.assert_sealed()
    if thresholds.threshold_hash != ADAPTIVE_THRESHOLDS_HASH_V66:
        raise ValueError("V6.6 adaptive threshold commitment differs")
    return thresholds


def _assert_threshold_commitments_v66() -> None:
    ode = ODEThresholdsV52.seal()
    success = ScientificSuccessThresholdsV61.seal()
    ode.assert_sealed()
    success.assert_sealed()
    _adaptive_thresholds_v66()
    if ode.threshold_hash != ODE_THRESHOLDS_HASH_V66:
        raise ValueError("V6.6 ODE threshold commitment differs")
    if success.thresholds_hash != SCIENTIFIC_SUCCESS_THRESHOLDS_HASH_V66:
        raise ValueError("V6.6 scientific-success threshold commitment differs")


class S0EvaluationProfileV66(StrictModel):
    """Code-owned S0 evaluation policy for the two positive-scalar adapters."""

    schema_version: Literal["6.6-s0-evaluation-profile"] = (
        "6.6-s0-evaluation-profile"
    )
    profile_id: Literal["positive-scalar-predictive-v66"] = (
        "positive-scalar-predictive-v66"
    )
    applicable_adapter_ids: list[
        Literal[
            "adaptive_positive_series_v57",
            "scalar_autonomous_ode_v52",
        ]
    ]
    candidate_registry_stage: Literal["S1"] = "S1"
    development_split_policy: Literal["chronological_prefix_suffix"] = (
        "chronological_prefix_suffix"
    )
    confirmation_policy: Literal["nested_rolling_origin_one_step"] = (
        "nested_rolling_origin_one_step"
    )
    adapter_validation_baseline: Literal["constant_state"] = "constant_state"
    confirmation_baseline: Literal["one_step_persistence"] = (
        "one_step_persistence"
    )
    uncertainty_policy: Literal[
        "training_only_bootstrap_and_rolling_empirical_diagnostic"
    ] = "training_only_bootstrap_and_rolling_empirical_diagnostic"
    interval_claim_ceiling: Literal["diagnostic_interval_quality_only"] = (
        "diagnostic_interval_quality_only"
    )
    abstention_policy: Literal[
        "not_run_or_human_when_required_evidence_is_unavailable"
    ] = "not_run_or_human_when_required_evidence_is_unavailable"
    ode_thresholds_hash: Sha256
    adaptive_thresholds_hash: Sha256
    scientific_success_thresholds_hash: Sha256
    private_feedback_permitted: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    profile_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> "S0EvaluationProfileV66":
        _assert_threshold_commitments_v66()
        if self.applicable_adapter_ids != [
            ADAPTIVE_ADAPTER_ID_V66,
            ODE_ADAPTER_ID_V66,
        ]:
            raise ValueError("V6.6 adapter IDs differ from the frozen profile")
        if self.ode_thresholds_hash != ODE_THRESHOLDS_HASH_V66:
            raise ValueError("V6.6 ODE threshold hash differs")
        if self.adaptive_thresholds_hash != ADAPTIVE_THRESHOLDS_HASH_V66:
            raise ValueError("V6.6 adaptive threshold hash differs")
        if (
            self.scientific_success_thresholds_hash
            != SCIENTIFIC_SUCCESS_THRESHOLDS_HASH_V66
        ):
            raise ValueError("V6.6 success threshold hash differs")
        if self.profile_hash and self.profile_hash != self.content_hash():
            raise ValueError("V6.6 evaluation profile hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "profile_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.profile_hash or self.profile_hash != self.content_hash():
            raise ValueError("V6.6 evaluation profile is not sealed")

    @classmethod
    def seal(cls) -> "S0EvaluationProfileV66":
        draft = cls(
            applicable_adapter_ids=[
                ADAPTIVE_ADAPTER_ID_V66,
                ODE_ADAPTER_ID_V66,
            ],
            ode_thresholds_hash=ODE_THRESHOLDS_HASH_V66,
            adaptive_thresholds_hash=ADAPTIVE_THRESHOLDS_HASH_V66,
            scientific_success_thresholds_hash=(
                SCIENTIFIC_SUCCESS_THRESHOLDS_HASH_V66
            ),
        )
        return cls(
            **draft.model_dump(exclude={"profile_hash"}),
            profile_hash=draft.content_hash(),
        )


def frozen_s0_evaluation_profile_v66() -> S0EvaluationProfileV66:
    """Return the only V6.6 profile accepted by this module."""

    return S0EvaluationProfileV66.seal()


def _complete_sentence(value: str, field_name: str) -> str:
    normalized = value.strip()
    if normalized.endswith(("...", "…")):
        raise ValueError(f"{field_name} ends with an ellipsis")
    if not normalized.endswith((".", "!", "?", "。", "！", "？")):
        raise ValueError(f"{field_name} must end with terminal punctuation")
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    if any(normalized.count(left) != normalized.count(right) for left, right in pairs):
        raise ValueError(f"{field_name} contains unbalanced delimiters")
    return normalized


class DecisionFunctionCanaryDraftV66(StrictModel):
    """A model draft without a tolerance field."""

    canary_id: Identifier
    input_values: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1,
        max_length=6,
    )
    expected: Annotated[float, Field(allow_inf_nan=False)]


class DecisionFunctionDraftV66(StrictModel):
    """S0 decision draft; the harness alone injects canary tolerance."""

    schema_version: Literal["6.6-s0-decision-draft"] = (
        "6.6-s0-decision-draft"
    )
    function_id: Identifier
    input_names: list[Identifier] = Field(min_length=1, max_length=6)
    expression: str = Field(min_length=1, max_length=500)
    sense: Literal["minimize", "maximize", "report_only"]
    output_unit: str = Field(min_length=1, max_length=80)
    canaries: list[DecisionFunctionCanaryDraftV66] = Field(
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_draft(self) -> "DecisionFunctionDraftV66":
        if self.input_names != list(dict.fromkeys(self.input_names)):
            raise ValueError("decision input_names must be unique")
        canary_ids = [item.canary_id for item in self.canaries]
        if canary_ids != list(dict.fromkeys(canary_ids)):
            raise ValueError("decision canary IDs must be unique")
        if any(
            len(item.input_values) != len(self.input_names)
            for item in self.canaries
        ):
            raise ValueError("canary input_values must align with input_names")
        if len(canonical_json(self.model_dump(mode="json"))) > (
            S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66
        ):
            raise ValueError("decision draft exceeds the V5.1 artifact envelope")
        return self


class RegimeDiagnosisDraftV66(StrictModel):
    """Concise S0 diagnosis bound to the frozen evaluation profile."""

    schema_version: Literal["6.6-s0-regime-draft"] = "6.6-s0-regime-draft"
    system_boundary: str = Field(
        min_length=10,
        max_length=S0_NARRATIVE_MAX_CHARACTERS_V66,
    )
    state_and_memory: str = Field(
        min_length=10,
        max_length=S0_NARRATIVE_MAX_CHARACTERS_V66,
    )
    uncertainty_and_data: str = Field(
        min_length=10,
        max_length=S0_NARRATIVE_MAX_CHARACTERS_V66,
    )
    decision_and_loss: str = Field(
        min_length=10,
        max_length=S0_NARRATIVE_MAX_CHARACTERS_V66,
    )
    query_type: Literal[
        "explanation",
        "prediction",
        "control",
        "optimization",
        "design",
        "mixed",
    ]
    downstream_decision: str = Field(
        min_length=10,
        max_length=S0_NARRATIVE_MAX_CHARACTERS_V66,
    )
    decision_function_id: Identifier
    computable_decision_function: str = Field(
        min_length=10,
        max_length=S0_NARRATIVE_MAX_CHARACTERS_V66,
    )
    evidence_hashes: list[Sha256] = Field(min_length=1, max_length=8)
    limitations: list[
        Annotated[
            str,
            Field(
                min_length=10,
                max_length=S0_NARRATIVE_MAX_CHARACTERS_V66,
            ),
        ]
    ] = Field(min_length=1, max_length=3)
    evaluation_profile_hash: Sha256

    @field_validator(
        "system_boundary",
        "state_and_memory",
        "uncertainty_and_data",
        "decision_and_loss",
        "downstream_decision",
        "computable_decision_function",
    )
    @classmethod
    def validate_sentence_field(cls, value: str, info: Any) -> str:
        return _complete_sentence(value, info.field_name)

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, values: list[str]) -> list[str]:
        return [
            _complete_sentence(value, f"limitations[{index}]")
            for index, value in enumerate(values)
        ]

    @model_validator(mode="after")
    def validate_diagnosis(self) -> "RegimeDiagnosisDraftV66":
        profile = frozen_s0_evaluation_profile_v66()
        profile.assert_sealed()
        if self.evaluation_profile_hash != profile.profile_hash:
            raise ValueError("regime draft is not bound to the V6.6 profile")
        if self.evaluation_profile_hash not in self.evidence_hashes:
            raise ValueError("profile hash must be included in evidence_hashes")
        if self.evidence_hashes != sorted(set(self.evidence_hashes)):
            raise ValueError("evidence_hashes must be sorted and unique")
        if len(canonical_json(self.model_dump(mode="json"))) > (
            S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66
        ):
            raise ValueError("regime draft exceeds the V5.1 artifact envelope")
        return self


def materialize_decision_function_v66(
    draft: DecisionFunctionDraftV66,
) -> DecisionFunctionSpecV50:
    """Materialize V5.0 with a code-owned, non-model tolerance."""

    validated = DecisionFunctionDraftV66.model_validate(
        draft.model_dump(mode="json")
    )
    return DecisionFunctionSpecV50.seal(
        function_id=validated.function_id,
        input_names=validated.input_names,
        expression=validated.expression,
        sense=validated.sense,
        output_unit=validated.output_unit,
        canaries=[
            DecisionFunctionCanaryV50(
                canary_id=item.canary_id,
                inputs=dict(zip(validated.input_names, item.input_values)),
                expected=item.expected,
                tolerance=S0_CANARY_TOLERANCE_V66,
            )
            for item in validated.canaries
        ],
    )


def materialize_regime_diagnosis_v66(
    draft: RegimeDiagnosisDraftV66,
) -> RegimeDiagnosisV50:
    """Materialize the historical V5.0 projection without reinterpreting it."""

    validated = RegimeDiagnosisDraftV66.model_validate(
        draft.model_dump(mode="json")
    )
    return RegimeDiagnosisV50.seal(
        system_boundary=validated.system_boundary,
        state_and_memory=validated.state_and_memory,
        uncertainty_and_data=validated.uncertainty_and_data,
        decision_and_loss=validated.decision_and_loss,
        query_type=validated.query_type,
        downstream_decision=validated.downstream_decision,
        decision_function_id=validated.decision_function_id,
        computable_decision_function=validated.computable_decision_function,
        evidence_hashes=validated.evidence_hashes,
        limitations=validated.limitations,
    )


class S0ReviewDefectCodeV66(str, Enum):
    REQUIRED_FIELD_MISSING = "REQUIRED_FIELD_MISSING"
    FIELD_TRUNCATED = "FIELD_TRUNCATED"
    INCOMPLETE_SENTENCE = "INCOMPLETE_SENTENCE"
    EVALUATION_PROFILE_BINDING_MISSING = (
        "EVALUATION_PROFILE_BINDING_MISSING"
    )
    EVALUATION_PROFILE_BINDING_MISMATCH = (
        "EVALUATION_PROFILE_BINDING_MISMATCH"
    )
    MODEL_CONTROLLED_CANARY_TOLERANCE = (
        "MODEL_CONTROLLED_CANARY_TOLERANCE"
    )
    DECISION_CANARY_ARITY_MISMATCH = "DECISION_CANARY_ARITY_MISMATCH"
    DECISION_INPUTS_NOT_UNIQUE = "DECISION_INPUTS_NOT_UNIQUE"
    ARTIFACT_ENVELOPE_EXCEEDED = "ARTIFACT_ENVELOPE_EXCEEDED"
    SEMANTIC_BOUNDARY_UNRESOLVED = "SEMANTIC_BOUNDARY_UNRESOLVED"
    OBJECTIVE_OR_LOSS_UNRESOLVED = "OBJECTIVE_OR_LOSS_UNRESOLVED"
    EVIDENCE_OR_MEASUREMENT_UNRESOLVED = (
        "EVIDENCE_OR_MEASUREMENT_UNRESOLVED"
    )
    HUMAN_DECISION_REQUIRED = "HUMAN_DECISION_REQUIRED"
    AUTHORITY_BOUNDARY_VIOLATION = "AUTHORITY_BOUNDARY_VIOLATION"
    PRIVATE_EVIDENCE_REQUIRED = "PRIVATE_EVIDENCE_REQUIRED"
    OTHER_UNCLASSIFIED_REJECT = "OTHER_UNCLASSIFIED_REJECT"


AUTO_REPAIRABLE_S0_CODES_V66: frozenset[S0ReviewDefectCodeV66] = frozenset(
    {
        S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING,
        S0ReviewDefectCodeV66.FIELD_TRUNCATED,
        S0ReviewDefectCodeV66.INCOMPLETE_SENTENCE,
        S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISSING,
        S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISMATCH,
        S0ReviewDefectCodeV66.MODEL_CONTROLLED_CANARY_TOLERANCE,
        S0ReviewDefectCodeV66.DECISION_CANARY_ARITY_MISMATCH,
        S0ReviewDefectCodeV66.DECISION_INPUTS_NOT_UNIQUE,
        S0ReviewDefectCodeV66.ARTIFACT_ENVELOPE_EXCEEDED,
    }
)


@dataclass(frozen=True)
class _FindingRuleV66:
    requirement_id: str
    severity: S0FindingSeverityV66
    observed: str
    expected: str
    repair_operator: S0RepairOperatorV66 | None


_FINDING_RULES_V66: dict[S0ReviewDefectCodeV66, _FindingRuleV66] = {
    S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING: _FindingRuleV66(
        "s0.required-fields",
        "BLOCKING",
        "required_field_absent",
        "required_field_present",
        "COMPLETE_REQUIRED_FIELD",
    ),
    S0ReviewDefectCodeV66.FIELD_TRUNCATED: _FindingRuleV66(
        "s0.complete-sentences",
        "BLOCKING",
        "field_reaches_bound_without_complete_sentence",
        "concise_complete_sentence",
        "COMPLETE_SENTENCE",
    ),
    S0ReviewDefectCodeV66.INCOMPLETE_SENTENCE: _FindingRuleV66(
        "s0.complete-sentences",
        "BLOCKING",
        "terminal_punctuation_absent",
        "complete_sentence",
        "COMPLETE_SENTENCE",
    ),
    S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISSING: (
        _FindingRuleV66(
            "s0.frozen-evaluation-profile",
            "BLOCKING",
            "profile_binding_absent",
            "exact_profile_hash",
            "RESTORE_FROZEN_PROFILE_BINDING",
        )
    ),
    S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISMATCH: (
        _FindingRuleV66(
            "s0.frozen-evaluation-profile",
            "BLOCKING",
            "profile_binding_mismatch",
            "exact_profile_hash",
            "RESTORE_FROZEN_PROFILE_BINDING",
        )
    ),
    S0ReviewDefectCodeV66.MODEL_CONTROLLED_CANARY_TOLERANCE: _FindingRuleV66(
        "s0.harness-owned-canary-policy",
        "BLOCKING",
        "model_supplied_tolerance",
        "harness_injected_tolerance",
        "REMOVE_MODEL_CONTROLLED_TOLERANCE",
    ),
    S0ReviewDefectCodeV66.DECISION_CANARY_ARITY_MISMATCH: _FindingRuleV66(
        "s0.computable-decision",
        "BLOCKING",
        "canary_arity_mismatch",
        "canary_arity_matches_inputs",
        "ALIGN_CANARY_ARITY",
    ),
    S0ReviewDefectCodeV66.DECISION_INPUTS_NOT_UNIQUE: _FindingRuleV66(
        "s0.computable-decision",
        "BLOCKING",
        "duplicate_decision_inputs",
        "unique_decision_inputs",
        "DEDUPLICATE_DECISION_INPUTS",
    ),
    S0ReviewDefectCodeV66.ARTIFACT_ENVELOPE_EXCEEDED: _FindingRuleV66(
        "s0.role-artifact-envelope",
        "BLOCKING",
        "artifact_exceeds_3000_characters",
        "artifact_at_most_3000_characters",
        "COMPRESS_TO_ARTIFACT_ENVELOPE",
    ),
    S0ReviewDefectCodeV66.SEMANTIC_BOUNDARY_UNRESOLVED: _FindingRuleV66(
        "s0.semantic-boundary",
        "HUMAN",
        "semantic_boundary_unresolved",
        "human_resolved_boundary",
        None,
    ),
    S0ReviewDefectCodeV66.OBJECTIVE_OR_LOSS_UNRESOLVED: _FindingRuleV66(
        "s0.objective-and-loss",
        "HUMAN",
        "objective_or_loss_unresolved",
        "value_owner_resolved_objective",
        None,
    ),
    S0ReviewDefectCodeV66.EVIDENCE_OR_MEASUREMENT_UNRESOLVED: (
        _FindingRuleV66(
            "s0.evidence-and-measurement",
            "HUMAN",
            "evidence_or_measurement_unresolved",
            "measurement_owner_resolution",
            None,
        )
    ),
    S0ReviewDefectCodeV66.HUMAN_DECISION_REQUIRED: _FindingRuleV66(
        "s0.human-decision",
        "HUMAN",
        "human_choice_missing",
        "human_choice_recorded",
        None,
    ),
    S0ReviewDefectCodeV66.AUTHORITY_BOUNDARY_VIOLATION: _FindingRuleV66(
        "s0.authority-boundary",
        "HUMAN",
        "authority_boundary_violation",
        "independent_authority_resolution",
        None,
    ),
    S0ReviewDefectCodeV66.PRIVATE_EVIDENCE_REQUIRED: _FindingRuleV66(
        "s0.private-evidence-firewall",
        "HUMAN",
        "private_evidence_required",
        "human_or_external_custody_resolution",
        None,
    ),
    S0ReviewDefectCodeV66.OTHER_UNCLASSIFIED_REJECT: _FindingRuleV66(
        "s0.unclassified-review-reject",
        "HUMAN",
        "unclassified_reviewer_objection",
        "human_triage",
        None,
    ),
}

_REGIME_SENTENCE_FIELDS_V66 = (
    "system_boundary",
    "state_and_memory",
    "uncertainty_and_data",
    "decision_and_loss",
    "downstream_decision",
    "computable_decision_function",
)

_NON_AUTO_TARGETS_V66: dict[
    S0ReviewDefectCodeV66,
    tuple[S0ArtifactPathV66, str],
] = {
    S0ReviewDefectCodeV66.SEMANTIC_BOUNDARY_UNRESOLVED: (
        "docs/regime.json",
        "/system_boundary",
    ),
    S0ReviewDefectCodeV66.OBJECTIVE_OR_LOSS_UNRESOLVED: (
        "problem/decision_function.json",
        "/",
    ),
    S0ReviewDefectCodeV66.EVIDENCE_OR_MEASUREMENT_UNRESOLVED: (
        "docs/regime.json",
        "/uncertainty_and_data",
    ),
    S0ReviewDefectCodeV66.HUMAN_DECISION_REQUIRED: (
        "docs/regime.json",
        "/downstream_decision",
    ),
    S0ReviewDefectCodeV66.AUTHORITY_BOUNDARY_VIOLATION: (
        "docs/regime.json",
        "/",
    ),
    S0ReviewDefectCodeV66.PRIVATE_EVIDENCE_REQUIRED: (
        "docs/regime.json",
        "/uncertainty_and_data",
    ),
    S0ReviewDefectCodeV66.OTHER_UNCLASSIFIED_REJECT: (
        "docs/regime.json",
        "/",
    ),
}


class S0ReviewerFindingCodesV66(StrictModel):
    """Small wire schema available to an independent S0 reviewer."""

    schema_version: Literal["6.6-s0-reviewer-codes"] = (
        "6.6-s0-reviewer-codes"
    )
    codes: list[S0ReviewDefectCodeV66] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_codes(self) -> "S0ReviewerFindingCodesV66":
        values = [item.value for item in self.codes]
        if values != sorted(set(values)):
            raise ValueError("reviewer finding codes must be sorted and unique")
        return self


class S0ReviewFindingDraftV66(StrictModel):
    """A target inferred by the harness, never reviewer prose."""

    code: S0ReviewDefectCodeV66
    artifact_path: S0ArtifactPathV66
    json_pointer: Annotated[
        str,
        Field(pattern=r"^/(?:[A-Za-z0-9_.-]+(?:/[0-9]+)?)?$"),
    ]


def _payload_characters(payload: object) -> int:
    try:
        return len(canonical_json(payload))
    except (TypeError, ValueError):
        return S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66 + 1


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def infer_s0_review_finding_drafts_v66(
    *,
    regime_payload: object,
    decision_payload: object,
    evaluation_profile_payload: object,
) -> list[S0ReviewFindingDraftV66]:
    """Mechanically infer the bounded, deterministic pre-data defect set."""

    findings: dict[
        tuple[str, S0ArtifactPathV66, str],
        S0ReviewFindingDraftV66,
    ] = {}

    def add(
        code: S0ReviewDefectCodeV66,
        artifact_path: S0ArtifactPathV66,
        json_pointer: str,
    ) -> None:
        key = (code.value, artifact_path, json_pointer)
        findings[key] = S0ReviewFindingDraftV66(
            code=code,
            artifact_path=artifact_path,
            json_pointer=json_pointer,
        )

    regime = _mapping(regime_payload)
    decision = _mapping(decision_payload)
    profile = _mapping(evaluation_profile_payload)

    if not profile:
        add(
            S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING,
            S0_EVALUATION_PROFILE_PATH_V66,
            "/",
        )
    else:
        try:
            parsed_profile = S0EvaluationProfileV66.model_validate(profile)
            parsed_profile.assert_sealed()
        except (TypeError, ValueError):
            add(
                S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISMATCH,
                S0_EVALUATION_PROFILE_PATH_V66,
                "/profile_hash",
            )

    expected_profile = frozen_s0_evaluation_profile_v66()
    expected_profile.assert_sealed()
    regime_profile_hash = regime.get("evaluation_profile_hash")
    regime_evidence_hashes = regime.get("evidence_hashes")
    projected_profile_binding = bool(
        isinstance(regime_evidence_hashes, list)
        and expected_profile.profile_hash in regime_evidence_hashes
    )
    if regime_profile_hash is None and not projected_profile_binding:
        add(
            S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISSING,
            "docs/regime.json",
            "/evaluation_profile_hash",
        )
    elif (
        regime_profile_hash is not None
        and regime_profile_hash != expected_profile.profile_hash
    ):
        add(
            S0ReviewDefectCodeV66.EVALUATION_PROFILE_BINDING_MISMATCH,
            "docs/regime.json",
            "/evaluation_profile_hash",
        )

    for field_name in _REGIME_SENTENCE_FIELDS_V66:
        value = regime.get(field_name)
        pointer = f"/{field_name}"
        if not isinstance(value, str) or not value.strip():
            add(
                S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING,
                "docs/regime.json",
                pointer,
            )
            continue
        normalized = value.strip()
        try:
            _complete_sentence(normalized, field_name)
        except ValueError:
            code = (
                S0ReviewDefectCodeV66.FIELD_TRUNCATED
                if len(normalized) >= S0_NARRATIVE_MAX_CHARACTERS_V66
                else S0ReviewDefectCodeV66.INCOMPLETE_SENTENCE
            )
            add(code, "docs/regime.json", pointer)

    limitations = regime.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        add(
            S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING,
            "docs/regime.json",
            "/limitations",
        )
    else:
        for index, value in enumerate(limitations[:3]):
            pointer = f"/limitations/{index}"
            if not isinstance(value, str) or not value.strip():
                add(
                    S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING,
                    "docs/regime.json",
                    pointer,
                )
                continue
            normalized = value.strip()
            try:
                _complete_sentence(normalized, pointer)
            except ValueError:
                code = (
                    S0ReviewDefectCodeV66.FIELD_TRUNCATED
                    if len(normalized) >= S0_NARRATIVE_MAX_CHARACTERS_V66
                    else S0ReviewDefectCodeV66.INCOMPLETE_SENTENCE
                )
                add(code, "docs/regime.json", pointer)

    input_names = decision.get("input_names")
    if not isinstance(input_names, list) or not input_names:
        add(
            S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING,
            "problem/decision_function.json",
            "/input_names",
        )
        input_count = 0
    else:
        input_count = len(input_names)
        if len(set(str(item) for item in input_names)) != len(input_names):
            add(
                S0ReviewDefectCodeV66.DECISION_INPUTS_NOT_UNIQUE,
                "problem/decision_function.json",
                "/input_names",
            )

    canaries = decision.get("canaries")
    if not isinstance(canaries, list) or not canaries:
        add(
            S0ReviewDefectCodeV66.REQUIRED_FIELD_MISSING,
            "problem/decision_function.json",
            "/canaries",
        )
    else:
        for index, raw_canary in enumerate(canaries[:4]):
            canary = _mapping(raw_canary)
            if "tolerance" in canary:
                add(
                    S0ReviewDefectCodeV66.MODEL_CONTROLLED_CANARY_TOLERANCE,
                    "problem/decision_function.json",
                    f"/canaries/{index}",
                )
            values = canary.get("input_values")
            if not isinstance(values, list):
                legacy_inputs = canary.get("inputs")
                values_count = (
                    len(legacy_inputs)
                    if isinstance(legacy_inputs, Mapping)
                    else -1
                )
            else:
                values_count = len(values)
            if values_count != input_count:
                add(
                    S0ReviewDefectCodeV66.DECISION_CANARY_ARITY_MISMATCH,
                    "problem/decision_function.json",
                    f"/canaries/{index}",
                )

    if _payload_characters(regime_payload) > (
        S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66
    ):
        add(
            S0ReviewDefectCodeV66.ARTIFACT_ENVELOPE_EXCEEDED,
            "docs/regime.json",
            "/",
        )
    if _payload_characters(decision_payload) > (
        S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66
    ):
        add(
            S0ReviewDefectCodeV66.ARTIFACT_ENVELOPE_EXCEEDED,
            "problem/decision_function.json",
            "/",
        )

    return [findings[key] for key in sorted(findings)]


class S0ReviewFindingV66(StrictModel):
    schema_version: Literal["6.6-s0-review-finding"] = (
        "6.6-s0-review-finding"
    )
    finding_id: Identifier
    reported_code: S0ReviewDefectCodeV66 | None
    reported_code_hash: Sha256
    code: S0ReviewDefectCodeV66
    requirement_id: Identifier
    artifact_path: S0ArtifactPathV66
    json_pointer: str = Field(
        pattern=r"^/(?:[A-Za-z0-9_.-]+(?:/[0-9]+)?)?$"
    )
    severity: S0FindingSeverityV66
    verification: S0FindingVerificationV66
    observed: Identifier
    expected: Identifier
    auto_repairable: bool
    repair_operator: S0RepairOperatorV66 | None
    finding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_finding(self) -> "S0ReviewFindingV66":
        rule = _FINDING_RULES_V66[self.code]
        if (
            self.requirement_id != rule.requirement_id
            or self.severity != rule.severity
            or self.observed != rule.observed
            or self.expected != rule.expected
            or self.repair_operator != rule.repair_operator
        ):
            raise ValueError("S0 review finding differs from its code-owned rule")
        expected_auto = (
            self.code in AUTO_REPAIRABLE_S0_CODES_V66
            and self.verification == "MECHANICALLY_VERIFIED"
        )
        if self.auto_repairable != expected_auto:
            raise ValueError("S0 auto-repair flag differs from verification")
        expected_id = _finding_id_v66(
            code=self.code,
            artifact_path=self.artifact_path,
            json_pointer=self.json_pointer,
        )
        if self.finding_id != expected_id:
            raise ValueError("S0 finding ID was not harness-derived")
        expected_reported_hash = sha256_value(
            {
                "reviewer_code": (
                    self.reported_code.value
                    if self.reported_code is not None
                    else "UNKNOWN"
                )
            }
        )
        if self.reported_code_hash != expected_reported_hash:
            raise ValueError("S0 reported-code hash differs")
        if self.finding_hash and self.finding_hash != self.content_hash():
            raise ValueError("S0 finding hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "finding_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.finding_hash or self.finding_hash != self.content_hash():
            raise ValueError("S0 review finding is not sealed")


def _finding_id_v66(
    *,
    code: S0ReviewDefectCodeV66,
    artifact_path: S0ArtifactPathV66,
    json_pointer: str,
) -> str:
    digest = sha256_value(
        {
            "schema_version": "6.6-s0-review-finding-id",
            "code": code.value,
            "artifact_path": artifact_path,
            "json_pointer": json_pointer,
        }
    )
    return f"s0f-{digest[:24]}"


def _seal_finding_v66(
    *,
    reported_code: S0ReviewDefectCodeV66 | None,
    effective_code: S0ReviewDefectCodeV66,
    artifact_path: S0ArtifactPathV66,
    json_pointer: str,
    verification: S0FindingVerificationV66,
) -> S0ReviewFindingV66:
    rule = _FINDING_RULES_V66[effective_code]
    draft = S0ReviewFindingV66(
        finding_id=_finding_id_v66(
            code=effective_code,
            artifact_path=artifact_path,
            json_pointer=json_pointer,
        ),
        reported_code=reported_code,
        reported_code_hash=sha256_value(
            {
                "reviewer_code": (
                    reported_code.value
                    if reported_code is not None
                    else "UNKNOWN"
                )
            }
        ),
        code=effective_code,
        requirement_id=rule.requirement_id,
        artifact_path=artifact_path,
        json_pointer=json_pointer,
        severity=rule.severity,
        verification=verification,
        observed=rule.observed,
        expected=rule.expected,
        auto_repairable=(
            effective_code in AUTO_REPAIRABLE_S0_CODES_V66
            and verification == "MECHANICALLY_VERIFIED"
        ),
        repair_operator=rule.repair_operator,
    )
    return S0ReviewFindingV66(
        **draft.model_dump(exclude={"finding_hash"}),
        finding_hash=draft.content_hash(),
    )


class S0ReviewFindingSetV66(StrictModel):
    schema_version: Literal["6.6-s0-review-finding-set"] = (
        "6.6-s0-review-finding-set"
    )
    task_id: Identifier
    attempt_id: Identifier
    reviewer_receipt_hash: Sha256
    evaluation_profile_hash: Sha256
    source_role: Literal["independent_reviewer"] = "independent_reviewer"
    findings: list[S0ReviewFindingV66] = Field(min_length=1, max_length=32)
    private_evidence_used: Literal[False] = False
    holdout_exposed: Literal[False] = False
    failure_signature: Sha256
    finding_set_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_finding_set(self) -> "S0ReviewFindingSetV66":
        profile = frozen_s0_evaluation_profile_v66()
        profile.assert_sealed()
        if self.evaluation_profile_hash != profile.profile_hash:
            raise ValueError("finding set profile binding differs")
        for finding in self.findings:
            finding.assert_sealed()
        keys = [
            (item.code.value, item.artifact_path, item.json_pointer)
            for item in self.findings
        ]
        if keys != sorted(set(keys)):
            raise ValueError("S0 findings must be sorted and unique")
        if self.failure_signature != _failure_signature_v66(
            self.findings,
            self.evaluation_profile_hash,
        ):
            raise ValueError("S0 failure signature was not harness-derived")
        if (
            self.finding_set_hash
            and self.finding_set_hash != self.content_hash()
        ):
            raise ValueError("S0 finding-set hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "finding_set_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if (
            not self.finding_set_hash
            or self.finding_set_hash != self.content_hash()
        ):
            raise ValueError("S0 review finding set is not sealed")


def _failure_signature_v66(
    findings: Sequence[S0ReviewFindingV66],
    evaluation_profile_hash: str,
) -> str:
    return sha256_value(
        {
            "schema_version": "6.6-s0-failure-signature",
            "evaluation_profile_hash": evaluation_profile_hash,
            "defects": [
                {
                    "code": item.code.value,
                    "artifact_path": item.artifact_path,
                    "json_pointer": item.json_pointer,
                }
                for item in findings
            ],
        }
    )


def seal_s0_review_findings_v66(
    *,
    task_id: str,
    attempt_id: str,
    reviewer_receipt_hash: str,
    reviewer_codes: Sequence[str | S0ReviewDefectCodeV66],
    regime_payload: object,
    decision_payload: object,
    evaluation_profile_payload: object,
) -> S0ReviewFindingSetV66:
    """Normalize reviewer codes against mechanically observed public defects.

    A claimed auto code absent from the inferred defect set is downgraded to a
    HUMAN-only unclassified reject.  Unknown strings and an empty REJECT code
    list take the same fail-closed path.
    """

    inferred = infer_s0_review_finding_drafts_v66(
        regime_payload=regime_payload,
        decision_payload=decision_payload,
        evaluation_profile_payload=evaluation_profile_payload,
    )
    inferred_by_code: dict[
        S0ReviewDefectCodeV66,
        list[S0ReviewFindingDraftV66],
    ] = {}
    for item in inferred:
        inferred_by_code.setdefault(item.code, []).append(item)

    findings_by_key: dict[
        tuple[str, S0ArtifactPathV66, str],
        S0ReviewFindingV66,
    ] = {}
    normalized_codes: list[S0ReviewDefectCodeV66 | None] = []
    for raw_code in reviewer_codes:
        try:
            value = (
                raw_code.value
                if isinstance(raw_code, S0ReviewDefectCodeV66)
                else str(raw_code)
            )
            normalized_codes.append(S0ReviewDefectCodeV66(value))
        except ValueError:
            normalized_codes.append(None)
    if not normalized_codes:
        normalized_codes.append(None)

    for reported_code in normalized_codes:
        targets = (
            inferred_by_code.get(reported_code, [])
            if reported_code is not None
            else []
        )
        if (
            reported_code in AUTO_REPAIRABLE_S0_CODES_V66
            and targets
        ):
            for target in targets:
                finding = _seal_finding_v66(
                    reported_code=reported_code,
                    effective_code=reported_code,
                    artifact_path=target.artifact_path,
                    json_pointer=target.json_pointer,
                    verification="MECHANICALLY_VERIFIED",
                )
                key = (
                    finding.code.value,
                    finding.artifact_path,
                    finding.json_pointer,
                )
                findings_by_key[key] = finding
            continue
        if (
            reported_code is not None
            and reported_code not in AUTO_REPAIRABLE_S0_CODES_V66
        ):
            effective_code = reported_code
            verification: S0FindingVerificationV66 = "HUMAN_ONLY"
        else:
            effective_code = S0ReviewDefectCodeV66.OTHER_UNCLASSIFIED_REJECT
            verification = "UNCLASSIFIED"
        artifact_path, json_pointer = _NON_AUTO_TARGETS_V66[effective_code]
        finding = _seal_finding_v66(
            reported_code=reported_code,
            effective_code=effective_code,
            artifact_path=artifact_path,
            json_pointer=json_pointer,
            verification=verification,
        )
        key = (
            finding.code.value,
            finding.artifact_path,
            finding.json_pointer,
        )
        findings_by_key[key] = finding

    findings = [findings_by_key[key] for key in sorted(findings_by_key)]
    profile = frozen_s0_evaluation_profile_v66()
    profile.assert_sealed()
    failure_signature = _failure_signature_v66(
        findings,
        profile.profile_hash,
    )
    draft = S0ReviewFindingSetV66(
        task_id=task_id,
        attempt_id=attempt_id,
        reviewer_receipt_hash=reviewer_receipt_hash,
        evaluation_profile_hash=profile.profile_hash,
        findings=findings,
        failure_signature=failure_signature,
    )
    return S0ReviewFindingSetV66(
        **draft.model_dump(exclude={"finding_set_hash"}),
        finding_set_hash=draft.content_hash(),
    )


class S0SemanticRepairDecisionV66(StrictModel):
    schema_version: Literal["6.6-s0-semantic-repair-decision"] = (
        "6.6-s0-semantic-repair-decision"
    )
    finding_set_hash: Sha256
    failure_signature: Sha256
    repair_attempts_used: int = Field(ge=0)
    maximum_automatic_repairs: Literal[1] = (
        S0_MAX_AUTOMATIC_SEMANTIC_REPAIRS_V66
    )
    previous_failure_signatures: list[Sha256] = Field(max_length=16)
    authorized: bool
    reason: S0RepairAuthorizationReasonV66
    decision_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "S0SemanticRepairDecisionV66":
        if self.previous_failure_signatures != sorted(
            set(self.previous_failure_signatures)
        ):
            raise ValueError("previous failure signatures must be sorted/unique")
        if self.authorized != (self.reason == "AUTHORIZED"):
            raise ValueError("repair authorization and reason differ")
        if self.decision_hash and self.decision_hash != self.content_hash():
            raise ValueError("S0 repair-decision hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "decision_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.decision_hash or self.decision_hash != self.content_hash():
            raise ValueError("S0 semantic repair decision is not sealed")


def authorize_s0_semantic_repair_v66(
    finding_set: S0ReviewFindingSetV66,
    *,
    repair_attempts_used: int,
    previous_failure_signatures: Sequence[str] = (),
    holdout_exposed: bool = False,
    private_evidence_used: bool = False,
) -> S0SemanticRepairDecisionV66:
    """Apply the single-attempt, pre-data, mechanically verified policy."""

    finding_set.assert_sealed()
    prior = sorted(set(previous_failure_signatures))
    if holdout_exposed or finding_set.holdout_exposed:
        reason: S0RepairAuthorizationReasonV66 = "HOLDOUT_EXPOSED"
    elif private_evidence_used or finding_set.private_evidence_used:
        reason = "PRIVATE_EVIDENCE_USED"
    elif any(not item.auto_repairable for item in finding_set.findings):
        reason = "NON_AUTOREPAIRABLE_FINDING"
    elif finding_set.failure_signature in prior:
        reason = "REPEATED_FAILURE_SIGNATURE"
    elif repair_attempts_used >= S0_MAX_AUTOMATIC_SEMANTIC_REPAIRS_V66:
        reason = "REPAIR_BUDGET_EXHAUSTED"
    else:
        reason = "AUTHORIZED"
    draft = S0SemanticRepairDecisionV66(
        finding_set_hash=finding_set.finding_set_hash,
        failure_signature=finding_set.failure_signature,
        repair_attempts_used=repair_attempts_used,
        previous_failure_signatures=prior,
        authorized=reason == "AUTHORIZED",
        reason=reason,
    )
    return S0SemanticRepairDecisionV66(
        **draft.model_dump(exclude={"decision_hash"}),
        decision_hash=draft.content_hash(),
    )


class S0RepairTargetV66(StrictModel):
    code: S0ReviewDefectCodeV66
    artifact_path: S0ArtifactPathV66
    json_pointer: str = Field(
        pattern=r"^/(?:[A-Za-z0-9_.-]+(?:/[0-9]+)?)?$"
    )
    repair_operator: S0RepairOperatorV66


class S0RepairContextV66(StrictModel):
    """The only feedback disclosed to an automatic S0 repair generator."""

    schema_version: Literal["6.6-s0-repair-context"] = (
        "6.6-s0-repair-context"
    )
    task_id: Identifier
    prior_attempt_id: Identifier
    new_attempt_id: Identifier
    evaluation_profile_hash: Sha256
    source_finding_set_hash: Sha256
    failure_signature: Sha256
    authorization_decision_hash: Sha256
    targets: list[S0RepairTargetV66] = Field(min_length=1, max_length=16)
    disclosure_policy: Literal["typed_codes_and_json_pointers_only"] = (
        "typed_codes_and_json_pointers_only"
    )
    reviewer_rationale_included: Literal[False] = False
    private_evidence_included: Literal[False] = False
    holdout_evidence_included: Literal[False] = False
    context_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_context(self) -> "S0RepairContextV66":
        profile = frozen_s0_evaluation_profile_v66()
        profile.assert_sealed()
        if self.evaluation_profile_hash != profile.profile_hash:
            raise ValueError("repair context profile binding differs")
        keys = [
            (
                item.code.value,
                item.artifact_path,
                item.json_pointer,
                item.repair_operator,
            )
            for item in self.targets
        ]
        if keys != sorted(set(keys)):
            raise ValueError("repair targets must be sorted and unique")
        if any(item.code not in AUTO_REPAIRABLE_S0_CODES_V66 for item in self.targets):
            raise ValueError("repair context contains a non-auto defect")
        if self.context_hash and self.context_hash != self.content_hash():
            raise ValueError("S0 repair-context hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "context_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.context_hash or self.context_hash != self.content_hash():
            raise ValueError("S0 repair context is not sealed")


def build_s0_repair_context_v66(
    *,
    finding_set: S0ReviewFindingSetV66,
    authorization: S0SemanticRepairDecisionV66,
    new_attempt_id: str,
) -> S0RepairContextV66:
    """Build a rationale-free repair packet after policy authorization."""

    finding_set.assert_sealed()
    authorization.assert_sealed()
    if not authorization.authorized:
        raise ValueError("S0 semantic repair was not authorized")
    if (
        authorization.finding_set_hash != finding_set.finding_set_hash
        or authorization.failure_signature != finding_set.failure_signature
    ):
        raise ValueError("repair authorization does not bind the finding set")
    targets = [
        S0RepairTargetV66(
            code=item.code,
            artifact_path=item.artifact_path,
            json_pointer=item.json_pointer,
            repair_operator=item.repair_operator,
        )
        for item in finding_set.findings
        if item.auto_repairable and item.repair_operator is not None
    ]
    if len(targets) != len(finding_set.findings):
        raise ValueError("repair context cannot disclose non-auto findings")
    draft = S0RepairContextV66(
        task_id=finding_set.task_id,
        prior_attempt_id=finding_set.attempt_id,
        new_attempt_id=new_attempt_id,
        evaluation_profile_hash=finding_set.evaluation_profile_hash,
        source_finding_set_hash=finding_set.finding_set_hash,
        failure_signature=finding_set.failure_signature,
        authorization_decision_hash=authorization.decision_hash,
        targets=targets,
    )
    return S0RepairContextV66(
        **draft.model_dump(exclude={"context_hash"}),
        context_hash=draft.content_hash(),
    )


__all__ = [
    "ADAPTIVE_ADAPTER_ID_V66",
    "ADAPTIVE_THRESHOLDS_HASH_V66",
    "AUTO_REPAIRABLE_S0_CODES_V66",
    "DecisionFunctionCanaryDraftV66",
    "DecisionFunctionDraftV66",
    "ODE_ADAPTER_ID_V66",
    "ODE_THRESHOLDS_HASH_V66",
    "RegimeDiagnosisDraftV66",
    "S0_CANARY_TOLERANCE_V66",
    "S0_EVALUATION_PROFILE_PATH_V66",
    "S0_MAX_AUTOMATIC_SEMANTIC_REPAIRS_V66",
    "S0_NARRATIVE_MAX_CHARACTERS_V66",
    "S0_ROLE_ARTIFACT_MAX_CHARACTERS_V66",
    "SCIENTIFIC_SUCCESS_THRESHOLDS_HASH_V66",
    "S0EvaluationProfileV66",
    "S0RepairContextV66",
    "S0RepairTargetV66",
    "S0ReviewDefectCodeV66",
    "S0ReviewerFindingCodesV66",
    "S0ReviewFindingDraftV66",
    "S0ReviewFindingSetV66",
    "S0ReviewFindingV66",
    "S0SemanticRepairDecisionV66",
    "authorize_s0_semantic_repair_v66",
    "build_s0_repair_context_v66",
    "frozen_s0_evaluation_profile_v66",
    "infer_s0_review_finding_drafts_v66",
    "materialize_decision_function_v66",
    "materialize_regime_diagnosis_v66",
    "seal_s0_review_findings_v66",
]
