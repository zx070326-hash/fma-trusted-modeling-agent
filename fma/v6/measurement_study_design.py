"""V6.7 pre-data measurement and study-design contract.

S0 defines the modelling question, while candidate generation starts in S1.
This additive S0.5 contract freezes what the target construct means and how it
would be measured and studied before any observation is accessed.  It is a
design artifact, not evidence: all semantic review remains HUMAN, independent
review and data access remain NOT_RUN, and the contract cannot grant a
scientific qualification or authorize a real-world action.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256


MEASUREMENT_STUDY_DESIGN_PATH_V67 = (
    "docs/measurement_study_design_contract_v67.json"
)

ClaimKindV67 = Literal[
    "descriptive",
    "predictive",
    "mechanistic",
    "prescriptive",
    "generalization",
]
ReviewStatusV67 = Literal["HUMAN", "NOT_RUN"]

ShortText = Annotated[str, Field(min_length=3, max_length=300)]
LongText = Annotated[str, Field(min_length=10, max_length=3000)]
TextList = Annotated[list[ShortText], Field(min_length=1)]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _require_unique(values: list[str], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")


class ConstructDefinitionV67(StrictModel):
    construct_id: Identifier
    name: ShortText
    conceptual_definition: LongText
    role: Literal[
        "outcome",
        "state",
        "exposure",
        "input",
        "decision",
        "context",
    ]
    representation: Literal["direct", "proxy", "latent"]
    representation_rationale: LongText


class MeasurementDefinitionV67(StrictModel):
    measurement_id: Identifier
    construct_id: Identifier
    operational_definition: LongText
    unit: Annotated[str, Field(min_length=1, max_length=120)]
    time_basis: ShortText
    aggregation_basis: ShortText
    scale_type: Literal[
        "nominal",
        "ordinal",
        "interval",
        "ratio",
        "count",
        "event_time",
    ]
    source_definition: LongText
    directionality: Literal[
        "higher_is_more",
        "lower_is_more",
        "unordered",
        "not_applicable",
    ]


class PopulationDefinitionV67(StrictModel):
    population_id: Identifier
    target_population: LongText
    unit_of_analysis: ShortText
    spatial_scope: ShortText
    temporal_scope: ShortText
    inclusion_criteria: TextList
    exclusion_criteria: list[ShortText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_criteria(self) -> "PopulationDefinitionV67":
        _require_unique(self.inclusion_criteria, "population inclusion criteria")
        _require_unique(self.exclusion_criteria, "population exclusion criteria")
        return self


class SamplingPlanV67(StrictModel):
    sampling_frame: LongText
    sampling_method: Literal[
        "census",
        "probability",
        "non_probability",
        "administrative_complete_series",
        "experiment_assignment",
        "other_predeclared",
    ]
    selection_rule: LongText
    minimum_sample_size: Annotated[int, Field(ge=1)]
    stopping_rule: LongText
    representativeness_limitations: LongText


class MissingnessPlanV67(StrictModel):
    anticipated_sources: TextList
    mechanism_assumptions: TextList
    handling_policy: Literal[
        "reject_incomplete_series",
        "complete_case",
        "multiple_imputation",
        "model_based",
        "sensitivity_analysis_required",
        "not_applicable_by_design",
    ]
    sensitivity_analysis_plan: LongText
    post_data_diagnosis_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_sources(self) -> "MissingnessPlanV67":
        _require_unique(self.anticipated_sources, "missingness source")
        _require_unique(self.mechanism_assumptions, "missingness assumption")
        return self


class MeasurementErrorPlanV67(StrictModel):
    anticipated_error_sources: TextList
    error_structure_assumption: LongText
    calibration_or_reference_plan: LongText
    propagation_or_sensitivity_plan: LongText
    independent_calibration_status: Literal["NOT_RUN"] = "NOT_RUN"

    @model_validator(mode="after")
    def validate_sources(self) -> "MeasurementErrorPlanV67":
        _require_unique(
            self.anticipated_error_sources,
            "measurement-error source",
        )
        return self


class BiasPlanV67(StrictModel):
    anticipated_biases: TextList
    mitigation_plan: LongText
    residual_bias_policy: LongText

    @model_validator(mode="after")
    def validate_biases(self) -> "BiasPlanV67":
        _require_unique(self.anticipated_biases, "anticipated bias")
        return self


class ConfoundingPlanV67(StrictModel):
    relevance: Literal[
        "required_for_claim",
        "not_applicable_to_noncausal_claim",
    ]
    candidate_confounders: list[ShortText] = Field(default_factory=list)
    identification_or_control_strategy: LongText
    unmeasured_confounding_policy: LongText
    causal_interpretation_authorized: Literal[False] = False

    @model_validator(mode="after")
    def validate_confounders(self) -> "ConfoundingPlanV67":
        _require_unique(self.candidate_confounders, "candidate confounder")
        if (
            self.relevance == "required_for_claim"
            and not self.candidate_confounders
        ):
            raise ValueError(
                "claim-relevant confounding requires candidate confounders"
            )
        if (
            self.relevance == "not_applicable_to_noncausal_claim"
            and self.candidate_confounders
        ):
            raise ValueError(
                "noncausal confounding plan cannot list controlled confounders"
            )
        return self


class StudyDesignV67(StrictModel):
    design_type: Literal[
        "observational_longitudinal",
        "observational_cross_sectional",
        "time_series",
        "panel",
        "randomized_experiment",
        "quasi_experiment",
        "simulation",
    ]
    target_quantity: LongText
    temporal_ordering: LongText
    comparison_strategy: LongText
    validation_design: LongText
    leakage_prevention_plan: LongText


class ApplicabilityBoundaryV67(StrictModel):
    intended_use: LongText
    in_scope_conditions: TextList
    out_of_scope_conditions: TextList
    transport_assumptions: TextList
    abstention_conditions: TextList

    @model_validator(mode="after")
    def validate_boundaries(self) -> "ApplicabilityBoundaryV67":
        for field_name in (
            "in_scope_conditions",
            "out_of_scope_conditions",
            "transport_assumptions",
            "abstention_conditions",
        ):
            _require_unique(getattr(self, field_name), field_name)
        return self


class EthicsBoundaryV67(StrictModel):
    risk_level: Literal["minimal", "elevated", "regulated"]
    human_participant_data_expected: bool
    sensitive_data_expected: bool
    consent_or_legal_basis_plan: LongText
    prohibited_uses: TextList
    ethics_review_required: bool

    @model_validator(mode="after")
    def validate_ethics_boundary(self) -> "EthicsBoundaryV67":
        _require_unique(self.prohibited_uses, "prohibited use")
        if (
            self.risk_level != "minimal"
            or self.human_participant_data_expected
            or self.sensitive_data_expected
        ) and not self.ethics_review_required:
            raise ValueError(
                "non-minimal or human/sensitive-data design requires ethics review"
            )
        return self


class MeasurementStudyDesignContractV67(StrictModel):
    """Sealed S0.5 design boundary containing no observed data."""

    schema_version: Literal["6.7-measurement-study-design"] = (
        "6.7-measurement-study-design"
    )
    contract_id: Identifier
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    source_contract_id: Identifier
    source_contract_hash: Sha256
    claim_kind: ClaimKindV67
    claim_scope: LongText
    construct_definition: ConstructDefinitionV67
    measurement: MeasurementDefinitionV67
    population: PopulationDefinitionV67
    sampling: SamplingPlanV67
    missingness: MissingnessPlanV67
    measurement_error: MeasurementErrorPlanV67
    bias: BiasPlanV67
    confounding: ConfoundingPlanV67
    study_design: StudyDesignV67
    applicability: ApplicabilityBoundaryV67
    ethics: EthicsBoundaryV67
    construct_review_status: Literal["HUMAN"] = "HUMAN"
    measurement_review_status: Literal["HUMAN"] = "HUMAN"
    study_design_review_status: Literal["HUMAN"] = "HUMAN"
    ethics_review_status: ReviewStatusV67 = "NOT_RUN"
    independent_review_status: Literal["NOT_RUN"] = "NOT_RUN"
    data_access_status: Literal["NOT_RUN"] = "NOT_RUN"
    observation_data_included: Literal[False] = False
    observed_statistics_included: Literal[False] = False
    private_acceptance_data_accessed: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "MeasurementStudyDesignContractV67":
        if (
            self.measurement.construct_id
            != self.construct_definition.construct_id
        ):
            raise ValueError("measurement references another construct")
        if (
            self.claim_kind == "mechanistic"
            and self.confounding.relevance != "required_for_claim"
        ) or (
            self.claim_kind in {
                "descriptive",
                "predictive",
                "generalization",
            }
            and self.confounding.relevance
            != "not_applicable_to_noncausal_claim"
        ):
            raise ValueError("confounding plan differs from claim kind")
        expected_ethics_review = (
            "HUMAN" if self.ethics.ethics_review_required else "NOT_RUN"
        )
        if self.ethics_review_status != expected_ethics_review:
            raise ValueError(
                "ethics review status differs from the declared requirement"
            )
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError(
                "V6.7 measurement/study-design contract hash differs"
            )
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if (
            not self.contract_hash
            or self.contract_hash != self.content_hash()
        ):
            raise ValueError(
                "V6.7 measurement/study-design contract is not sealed"
            )

    @classmethod
    def seal(cls, **data: object) -> "MeasurementStudyDesignContractV67":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


__all__ = [
    "ApplicabilityBoundaryV67",
    "BiasPlanV67",
    "ClaimKindV67",
    "ConfoundingPlanV67",
    "ConstructDefinitionV67",
    "EthicsBoundaryV67",
    "MEASUREMENT_STUDY_DESIGN_PATH_V67",
    "MeasurementDefinitionV67",
    "MeasurementErrorPlanV67",
    "MeasurementStudyDesignContractV67",
    "MissingnessPlanV67",
    "PopulationDefinitionV67",
    "ReviewStatusV67",
    "SamplingPlanV67",
    "StudyDesignV67",
]
