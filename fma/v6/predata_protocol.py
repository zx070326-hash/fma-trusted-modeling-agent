"""V6.7 code-owned execution protocol frozen before observation access.

The model-selected mathematical prose remains documentation.  This module
accepts one sealed S0.5 measurement/study-design contract and one *exact*
registered positive-series capability pack, then compiles the executable
rules that the harness will apply later.  Adapter selection therefore happens
before data access; S2 may only validate the frozen compatibility predicates.

The protocol is content sealed but is not scientific evidence or standalone
gate authority.  A future change of adapter, threshold, or recovery direction
requires a new graph attempt and a newly compiled successor protocol.
"""

from __future__ import annotations

import hashlib
import marshal
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.workspace_schemas import CandidateFormalizationV50
from fma.v5_2 import ode_system as ode_system_v52
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_6 import hybrid_ode as hybrid_ode_v56
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_7 import adaptive_positive_series as adaptive_series_v57
from fma.v5_7.adaptive_positive_series import AdaptiveThresholdsV57

from .executable_candidate import (
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    SCALAR_ODE_ADAPTER_ID,
    SUPPORTED_ADAPTER_IDS_V62,
    AdapterIdV62,
    RegisteredFamilySearchIRV62,
    RegisteredFamilySearchIntentV62,
    RegisteredFamilyV62,
    allowed_family_registry_hash_v62,
    registered_families_for_adapter_v62,
)
from .measurement_study_design import MeasurementStudyDesignContractV67
from .recovery_kernel import (
    CapabilityPackV60,
    default_capability_registry_v60,
)
from .scientific_success import ScientificSuccessThresholdsV61
from .stage_review_recovery import (
    _adaptive_thresholds_v66,
    frozen_s0_evaluation_profile_v66,
)


PREDATA_EXECUTION_PROTOCOL_PATH_V67 = "docs/predata_execution_protocol_v67.json"
CANDIDATE_EXECUTION_BINDING_PATH_V67 = "docs/candidate_execution_binding_v67.json"
PREDATA_COMPILER_ID_V67 = "positive-series-predata-compiler-v67"

LevelV67 = Literal["L0", "L1", "L2", "L3", "L4"]
StageForLevelV67 = Literal["S3", "S4"]
FiniteScalar = Annotated[float, Field(allow_inf_nan=False)]
SettingValueV67 = int | float | str
VersionTextV67 = Annotated[
    str,
    Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    ),
]


class PreDataProtocolError(ValueError):
    """The pre-data compiler or replay verifier failed closed."""


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _callable_semantic_hash(callable_object: object) -> str:
    """Bind loaded Python semantics without opening any observation source."""

    code = getattr(callable_object, "__code__", None)
    if code is None:
        raise PreDataProtocolError(
            "pre-data implementation contains an unhashable callable"
        )
    return sha256_value(
        {
            "module": getattr(callable_object, "__module__", None),
            "qualname": getattr(callable_object, "__qualname__", None),
            "marshalled_code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest(),
            "defaults": repr(getattr(callable_object, "__defaults__", None)),
            "keyword_defaults": repr(getattr(callable_object, "__kwdefaults__", None)),
        }
    )


def registered_positive_series_capability_pack_v67(
    adapter_id: str,
) -> CapabilityPackV60:
    """Return a defensive copy of one current code-registered pack."""

    if adapter_id not in SUPPORTED_ADAPTER_IDS_V62:
        raise PreDataProtocolError(
            f"unsupported positive-series capability pack: {adapter_id}"
        )
    registry = default_capability_registry_v60()
    packs = getattr(registry, "_packs", None)
    if not isinstance(packs, dict):
        raise PreDataProtocolError(
            "capability registry does not expose its code-owned pack table"
        )
    pack = packs.get(adapter_id)
    if not isinstance(pack, CapabilityPackV60):
        raise PreDataProtocolError(
            f"registered capability pack is unavailable: {adapter_id}"
        )
    pack.assert_sealed()
    return pack.model_copy(deep=True)


def _exact_registered_pack(pack: CapabilityPackV60) -> CapabilityPackV60:
    try:
        pack.assert_sealed()
    except ValueError as exc:
        raise PreDataProtocolError(
            "pre-data compilation requires a sealed capability pack"
        ) from exc
    expected = registered_positive_series_capability_pack_v67(pack.pack_id)
    if pack != expected:
        raise PreDataProtocolError(
            "capability pack differs from the current code-owned registry"
        )
    return expected


class AdapterBindingV67(StrictModel):
    """Exact executable and scientific adapter identity."""

    schema_version: Literal["6.7-adapter-binding"] = "6.7-adapter-binding"
    adapter_id: AdapterIdV62
    adapter_version: Identifier
    capability_pack_hash: Sha256
    executor_id: Identifier
    scientific_adapter_id: Identifier
    scientific_adapter_version: VersionTextV67
    allowed_families: Annotated[
        list[RegisteredFamilyV62],
        Field(min_length=4, max_length=6),
    ]
    allowed_family_registry_hash: Sha256
    execution_builder_ref: Annotated[str, Field(min_length=10)]
    execution_builder_semantic_hash: Sha256
    scientific_adapter_run_ref: Annotated[str, Field(min_length=10)]
    scientific_adapter_run_semantic_hash: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> "AdapterBindingV67":
        expected_families = list(registered_families_for_adapter_v62(self.adapter_id))
        if self.allowed_families != expected_families:
            raise ValueError("V6.7 adapter binding differs from its family registry")
        if self.allowed_family_registry_hash != (
            allowed_family_registry_hash_v62(self.adapter_id)
        ):
            raise ValueError("V6.7 family-registry hash differs")
        return self


class CompatibilityPolicyV67(StrictModel):
    """Predicates frozen now and evaluated only after S2 binds source bytes."""

    schema_version: Literal["6.7-compatibility-policy"] = "6.7-compatibility-policy"
    adapter_id: AdapterIdV62
    allowed_state_kinds: list[Literal["scalar", "vector", "network", "spatial_field"]]
    allowed_time_kinds: list[Literal["static", "discrete", "continuous"]]
    allowed_dynamics_kinds: list[
        Literal["none", "autonomous", "nonautonomous", "stochastic"]
    ]
    allowed_observation_kinds: list[
        Literal["complete", "partial", "irregular", "aggregate"]
    ]
    minimum_execution_observation_count: Annotated[int, Field(ge=1)]
    minimum_confirmation_observation_count: Annotated[int, Field(ge=1)]
    finite_positive_values_required: bool
    strictly_increasing_time_required: bool
    effectively_regular_cadence_required: bool
    maximum_cadence_relative_deviation: FiniteScalar | None
    exact_measurement_unit_required: Annotated[str, Field(min_length=1)]
    exact_time_basis_required: Annotated[str, Field(min_length=3)]
    missing_value_policy: Literal["reject_incomplete_series"] = (
        "reject_incomplete_series"
    )
    s2_compatibility_rule: Literal[
        "evaluate_all_frozen_predicates_against_bound_s2_snapshot"
    ] = "evaluate_all_frozen_predicates_against_bound_s2_snapshot"
    incompatible_result: Literal[
        "reject_before_fit_and_emit_graph_data_contract_failure"
    ] = "reject_before_fit_and_emit_graph_data_contract_failure"
    compatibility_failure_is_scientific_failure: Literal[False] = False

    @model_validator(mode="after")
    def validate_policy(self) -> "CompatibilityPolicyV67":
        for field_name in (
            "allowed_state_kinds",
            "allowed_time_kinds",
            "allowed_dynamics_kinds",
            "allowed_observation_kinds",
        ):
            values = getattr(self, field_name)
            if values != sorted(set(values)):
                raise ValueError(f"V6.7 {field_name} must be sorted and unique")
        if (
            self.minimum_confirmation_observation_count
            < self.minimum_execution_observation_count
        ):
            raise ValueError("confirmation cannot require less history than execution")
        if self.effectively_regular_cadence_required != (
            self.maximum_cadence_relative_deviation is not None
        ):
            raise ValueError("cadence tolerance differs from regularity requirement")
        return self


class AdapterResolutionPolicyV67(StrictModel):
    """Selection is complete before data; S2 cannot choose another adapter."""

    schema_version: Literal["6.7-adapter-resolution-policy"] = (
        "6.7-adapter-resolution-policy"
    )
    adapter_resolution_stage: Literal["pre_data_compiler"] = "pre_data_compiler"
    selected_adapter_id: AdapterIdV62
    selection_rule: Literal[
        "exact_single_registered_pack_supplied_to_code_owned_compiler"
    ] = "exact_single_registered_pack_supplied_to_code_owned_compiler"
    s2_role: Literal["compatibility_validation_only"] = "compatibility_validation_only"
    silent_adapter_substitution_permitted: Literal[False] = False
    same_protocol_fallback_permitted: Literal[False] = False
    recovery_rule: Literal["new_graph_attempt_and_new_protocol_successor_required"] = (
        "new_graph_attempt_and_new_protocol_successor_required"
    )
    threshold_change_requires_successor_protocol: Literal[True] = True


class ParameterBoundRuleV67(StrictModel):
    """A harness-owned reference to a deterministic parameter-domain rule."""

    family_id: RegisteredFamilyV62
    parameter_names: list[Identifier]
    bound_rule_id: Identifier
    exact_rule: Annotated[str, Field(min_length=10, max_length=1000)]
    model_supplied_bounds_permitted: Literal[False] = False
    rule_text_is_executable: Literal[False] = False

    @model_validator(mode="after")
    def validate_rule(self) -> "ParameterBoundRuleV67":
        if self.parameter_names != sorted(set(self.parameter_names)):
            raise ValueError("V6.7 parameter names must be sorted and unique")
        return self


class FittingPolicyV67(StrictModel):
    """Exact fitting, optimization, bounds, and selection semantics."""

    schema_version: Literal["6.7-fitting-policy"] = "6.7-fitting-policy"
    adapter_id: AdapterIdV62
    fit_implementation_refs: Annotated[
        list[str],
        Field(min_length=1),
    ]
    fit_implementation_semantic_hashes: Annotated[
        dict[Identifier, Sha256],
        Field(min_length=1),
    ]
    fitting_objective: Annotated[str, Field(min_length=10, max_length=1000)]
    optimizer_id: Identifier
    optimizer_settings: dict[Identifier, SettingValueV67]
    parameter_bound_rules: Annotated[
        list[ParameterBoundRuleV67],
        Field(min_length=4, max_length=6),
    ]
    candidate_selection_rule: Annotated[
        str,
        Field(min_length=10, max_length=1500),
    ]
    optimizer_nonconvergence_rule: Literal[
        "candidate_inadmissible_required_check_fails_no_result_tuning"
    ] = "candidate_inadmissible_required_check_fails_no_result_tuning"
    model_family_text_executable: Literal[False] = False
    mathematical_form_text_executable: Literal[False] = False
    arbitrary_code_execution_permitted: Literal[False] = False

    @model_validator(mode="after")
    def validate_fitting(self) -> "FittingPolicyV67":
        if self.fit_implementation_refs != sorted(set(self.fit_implementation_refs)):
            raise ValueError("V6.7 fit implementation refs must be sorted and unique")
        if list(self.fit_implementation_semantic_hashes) != sorted(
            self.fit_implementation_semantic_hashes
        ):
            raise ValueError("V6.7 fit semantic hashes must be key-sorted")
        families = [item.family_id for item in self.parameter_bound_rules]
        expected = list(registered_families_for_adapter_v62(self.adapter_id))
        if families != expected:
            raise ValueError("V6.7 fitting policy must bind every registered family")
        return self


class ThresholdBundleV67(StrictModel):
    """Typed historical thresholds reused without reinterpretation."""

    schema_version: Literal["6.7-threshold-bundle"] = "6.7-threshold-bundle"
    adapter_id: AdapterIdV62
    s0_evaluation_profile_hash: Sha256
    ode_thresholds: ODEThresholdsV52 | None = None
    hybrid_thresholds: HybridODEThresholdsV56 | None = None
    adaptive_thresholds: AdaptiveThresholdsV57 | None = None
    scientific_success_thresholds: ScientificSuccessThresholdsV61

    @model_validator(mode="after")
    def validate_threshold_bundle(self) -> "ThresholdBundleV67":
        self.scientific_success_thresholds.assert_sealed()
        if self.adapter_id == SCALAR_ODE_ADAPTER_ID:
            if (
                self.ode_thresholds is None
                or self.hybrid_thresholds is not None
                or self.adaptive_thresholds is not None
            ):
                raise ValueError("scalar ODE protocol has the wrong threshold bundle")
            self.ode_thresholds.assert_sealed()
        else:
            if (
                self.ode_thresholds is not None
                or self.hybrid_thresholds is None
                or self.adaptive_thresholds is None
            ):
                raise ValueError("adaptive protocol has the wrong threshold bundle")
            self.hybrid_thresholds.assert_sealed()
            self.adaptive_thresholds.assert_sealed()
        return self

    def threshold_hashes(self) -> list[str]:
        values = [self.scientific_success_thresholds.thresholds_hash]
        for threshold in (
            self.ode_thresholds,
            self.hybrid_thresholds,
            self.adaptive_thresholds,
        ):
            if threshold is not None:
                values.append(threshold.threshold_hash)
        return sorted(cast(list[str], values))


class TrainingConfirmationPolicyV67(StrictModel):
    """Development split and leakage-safe confirmation frozen pre-data."""

    schema_version: Literal["6.7-training-confirmation-policy"] = (
        "6.7-training-confirmation-policy"
    )
    adapter_id: AdapterIdV62
    development_split_method: Literal["chronological_prefix_suffix"] = (
        "chronological_prefix_suffix"
    )
    development_split_fraction: Annotated[
        float,
        Field(gt=0.5, lt=0.9, allow_inf_nan=False),
    ]
    development_split_index_rule: Literal["min(max(int(n*split_fraction),2),n-2)"] = (
        "min(max(int(n*split_fraction),2),n-2)"
    )
    minimum_points_per_development_slice: Annotated[int, Field(ge=4)]
    development_baseline_ids: Annotated[
        list[Identifier],
        Field(min_length=1),
    ]
    confirmation_method: Literal["nested_rolling_origin_one_step"] = (
        "nested_rolling_origin_one_step"
    )
    confirmation_implementation_ref: Literal[
        "fma.v6.scientific_success.evaluate_rolling_confirmation_v61"
    ] = "fma.v6.scientific_success.evaluate_rolling_confirmation_v61"
    confirmation_implementation_semantic_hash: Sha256
    confirmation_minimum_history: Annotated[int, Field(ge=1)]
    confirmation_fold_count: Annotated[int, Field(ge=1)]
    confirmation_minimum_total_observations: Annotated[int, Field(ge=1)]
    confirmation_origin_rule: Literal[
        "for_index_in_range(n-fold_count,n)_fit_select_on_prefix_0_to_index_minus_1"
    ] = "for_index_in_range(n-fold_count,n)_fit_select_on_prefix_0_to_index_minus_1"
    refit_each_origin: Literal[True] = True
    replay_candidate_selection_each_origin: Literal[True] = True
    confirmation_baseline_id: Literal["persistence"] = "persistence"
    comparison_metric_ids: list[Identifier]
    future_observation_access_during_fit_permitted: Literal[False] = False
    private_acceptance_feedback_permitted: Literal[False] = False

    @model_validator(mode="after")
    def validate_training(self) -> "TrainingConfirmationPolicyV67":
        if self.development_baseline_ids != sorted(set(self.development_baseline_ids)):
            raise ValueError("V6.7 development baselines must be sorted and unique")
        if self.comparison_metric_ids != sorted(set(self.comparison_metric_ids)):
            raise ValueError("V6.7 confirmation metrics must be sorted and unique")
        if self.confirmation_minimum_total_observations != (
            self.confirmation_minimum_history + self.confirmation_fold_count
        ):
            raise ValueError("V6.7 confirmation total differs from history plus folds")
        return self


class UncertaintyStressPolicyV67(StrictModel):
    """Bootstrap, dependence, interval, and stress-test claim boundary."""

    schema_version: Literal["6.7-uncertainty-stress-policy"] = (
        "6.7-uncertainty-stress-policy"
    )
    adapter_id: AdapterIdV62
    bootstrap_rule: Annotated[str, Field(min_length=10, max_length=1500)]
    bootstrap_replicates_by_component: dict[Identifier, Annotated[int, Field(ge=20)]]
    bootstrap_seed_by_component: dict[
        Identifier,
        Annotated[int, Field(ge=0, le=2**32 - 1)],
    ]
    dependence_rule: Annotated[str, Field(min_length=10, max_length=1500)]
    block_bootstrap_used: Literal[False] = False
    temporal_dependence_coverage_guaranteed: Literal[False] = False
    interval_rule: Annotated[str, Field(min_length=10, max_length=1500)]
    interval_quantiles: list[Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]
    interval_evidence_kind: Literal[
        "training_only_bootstrap_and_rolling_empirical_diagnostic"
    ] = "training_only_bootstrap_and_rolling_empirical_diagnostic"
    interval_claim_ceiling: Literal["diagnostic_interval_quality_only"] = (
        "diagnostic_interval_quality_only"
    )
    finite_sample_coverage_guaranteed: Literal[False] = False
    post_selection_coverage_guaranteed: Literal[False] = False
    stress_rule_ids: Annotated[list[Identifier], Field(min_length=2)]
    stress_rule_refs: Annotated[list[str], Field(min_length=2)]

    @model_validator(mode="after")
    def validate_uncertainty(self) -> "UncertaintyStressPolicyV67":
        for field_name in (
            "bootstrap_replicates_by_component",
            "bootstrap_seed_by_component",
        ):
            values = getattr(self, field_name)
            if list(values) != sorted(values):
                raise ValueError(f"V6.7 {field_name} must be key-sorted")
        if set(self.bootstrap_replicates_by_component) != set(
            self.bootstrap_seed_by_component
        ):
            raise ValueError(
                "V6.7 bootstrap components differ between counts and seeds"
            )
        if self.interval_quantiles != [0.025, 0.5, 0.975]:
            raise ValueError("V6.7 interval quantiles differ")
        if self.stress_rule_ids != sorted(set(self.stress_rule_ids)):
            raise ValueError("V6.7 stress rule IDs must be sorted and unique")
        if self.stress_rule_refs != sorted(set(self.stress_rule_refs)):
            raise ValueError("V6.7 stress rule refs must be sorted and unique")
        return self


class LevelRuleV67(StrictModel):
    """One exact registered L0-L4 adapter rule reference."""

    level: LevelV67
    stage: StageForLevelV67
    check_id: Identifier
    adapter_id: Identifier
    adapter_version: VersionTextV67
    implementation_ref: Annotated[str, Field(min_length=10)]
    implementation_semantic_hash: Sha256
    threshold_hashes: Annotated[list[Sha256], Field(min_length=2)]
    required: Literal[True] = True
    pass_rule: Literal["PASS_only_when_exact_registered_level_evidence_is_PASS"] = (
        "PASS_only_when_exact_registered_level_evidence_is_PASS"
    )
    not_run_counts_as_pass: Literal[False] = False
    file_presence_counts_as_evidence: Literal[False] = False

    @model_validator(mode="after")
    def validate_level(self) -> "LevelRuleV67":
        expected_stage = "S3" if self.level in {"L0", "L1", "L2"} else "S4"
        if self.stage != expected_stage:
            raise ValueError("V6.7 level is assigned to the wrong stage")
        if self.threshold_hashes != sorted(set(self.threshold_hashes)):
            raise ValueError("V6.7 level threshold hashes must be sorted and unique")
        return self


class AbstentionPolicyV67(StrictModel):
    """Fail-closed outcomes without promotion or same-protocol substitution."""

    schema_version: Literal["6.7-abstention-policy"] = "6.7-abstention-policy"
    missing_required_evidence_status: Literal["NOT_RUN"] = "NOT_RUN"
    failed_required_level_action: Literal["ABSTAIN_AND_PRESERVE_NEGATIVE_EVIDENCE"] = (
        "ABSTAIN_AND_PRESERVE_NEGATIVE_EVIDENCE"
    )
    incompatible_data_action: Literal["REJECT_BEFORE_FIT_AND_RETURN_TO_GRAPH"] = (
        "REJECT_BEFORE_FIT_AND_RETURN_TO_GRAPH"
    )
    adaptive_unresolved_branch_action: Literal["ABSTAIN"] = "ABSTAIN"
    human_required_action: Literal["PAUSE_HUMAN"] = "PAUSE_HUMAN"
    threshold_tuning_after_results_permitted: Literal[False] = False
    silent_model_switch_permitted: Literal[False] = False
    recovery_requires_new_graph_attempt: Literal[True] = True
    recovery_requires_new_protocol_successor: Literal[True] = True
    local_workflow_pass_is_scientific_qualification: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False


class PreDataExecutionProtocolV67(StrictModel):
    """Deterministic, sealed projection of the code-owned pre-data compiler."""

    schema_version: Literal["6.7-predata-execution-protocol"] = (
        "6.7-predata-execution-protocol"
    )
    compiler_id: Literal["positive-series-predata-compiler-v67"] = (
        PREDATA_COMPILER_ID_V67
    )
    protocol_id: Identifier
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    source_contract_id: Identifier
    source_contract_hash: Sha256
    measurement_contract_id: Identifier
    measurement_contract_hash: Sha256
    measurement_id: Identifier
    claim_kind: Literal["predictive"] = "predictive"
    allowed_adapter_ids: Annotated[
        list[AdapterIdV62],
        Field(min_length=1, max_length=1),
    ]
    allowed_adapter_versions: Annotated[
        dict[AdapterIdV62, Identifier],
        Field(min_length=1, max_length=1),
    ]
    adapter_binding: AdapterBindingV67
    adapter_resolution: AdapterResolutionPolicyV67
    compatibility: CompatibilityPolicyV67
    thresholds: ThresholdBundleV67
    fitting: FittingPolicyV67
    training_and_confirmation: TrainingConfirmationPolicyV67
    uncertainty_and_stress: UncertaintyStressPolicyV67
    level_rules: Annotated[list[LevelRuleV67], Field(min_length=5, max_length=5)]
    abstention: AbstentionPolicyV67
    observation_values_accessed_during_compilation: Literal[False] = False
    observed_statistics_accessed_during_compilation: Literal[False] = False
    private_acceptance_data_accessed: Literal[False] = False
    model_text_executable: Literal[False] = False
    protocol_is_scientific_evidence: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    protocol_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_protocol(self) -> "PreDataExecutionProtocolV67":
        adapter_id = self.adapter_binding.adapter_id
        if self.allowed_adapter_ids != [adapter_id]:
            raise ValueError("V6.7 protocol must allow exactly its selected adapter")
        if self.allowed_adapter_versions != {
            adapter_id: self.adapter_binding.adapter_version
        }:
            raise ValueError("V6.7 allowed adapter version differs")
        if (
            self.adapter_resolution.selected_adapter_id != adapter_id
            or self.compatibility.adapter_id != adapter_id
            or self.thresholds.adapter_id != adapter_id
            or self.fitting.adapter_id != adapter_id
            or self.training_and_confirmation.adapter_id != adapter_id
            or self.uncertainty_and_stress.adapter_id != adapter_id
        ):
            raise ValueError("V6.7 protocol component adapters differ")
        expected_levels: list[LevelV67] = ["L0", "L1", "L2", "L3", "L4"]
        if [item.level for item in self.level_rules] != expected_levels:
            raise ValueError("V6.7 protocol must contain ordered L0-L4 rules")
        if any(
            item.adapter_id != self.adapter_binding.scientific_adapter_id
            or item.adapter_version != self.adapter_binding.scientific_adapter_version
            for item in self.level_rules
        ):
            raise ValueError("V6.7 L0-L4 adapter identity differs")
        if self.compatibility.exact_measurement_unit_required == "":
            raise ValueError("V6.7 measurement unit is absent")
        if self.protocol_hash and self.protocol_hash != self.content_hash():
            raise ValueError("V6.7 pre-data protocol hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "protocol_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.protocol_hash or self.protocol_hash != self.content_hash():
            raise PreDataProtocolError("V6.7 pre-data execution protocol is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "PreDataExecutionProtocolV67":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"protocol_hash"})
        payload["protocol_hash"] = draft.content_hash()
        return cls(**payload)


class CandidateExecutionBindingV67(StrictModel):
    """Bind selected candidate semantics to the exact pre-data authority."""

    schema_version: Literal["6.7-candidate-execution-binding"] = (
        "6.7-candidate-execution-binding"
    )
    workspace_spec_hash: Sha256
    s0_gate_hash: Sha256
    source_contract_hash: Sha256
    measurement_contract_hash: Sha256
    predata_protocol_hash: Sha256
    candidate_id: Identifier
    candidate_structural_hash: Sha256
    legacy_v62_intent_hash: Sha256
    legacy_v62_ir_hash: Sha256
    selected_adapter_id: AdapterIdV62
    selected_adapter_version: Identifier
    capability_pack_hash: Sha256
    allowed_adapter_ids: Annotated[
        list[AdapterIdV62],
        Field(min_length=1, max_length=1),
    ]
    allowed_family_ids: Annotated[
        list[RegisteredFamilyV62],
        Field(min_length=1),
    ]
    allowed_family_registry_hash: Sha256
    executable_semantics_authority: Literal["predata_execution_protocol_v67"] = (
        "predata_execution_protocol_v67"
    )
    adapter_resolution_stage: Literal["pre_data_compiler"] = "pre_data_compiler"
    s2_role: Literal["compatibility_validation_only"] = "compatibility_validation_only"
    legacy_v62_resolution_is_authority: Literal[False] = False
    silent_adapter_substitution_permitted: Literal[False] = False
    model_text_executable: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    binding_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> "CandidateExecutionBindingV67":
        if self.allowed_adapter_ids != [self.selected_adapter_id]:
            raise ValueError(
                "V6.7 candidate binding must allow exactly its selected adapter"
            )
        if self.allowed_family_ids != sorted(set(self.allowed_family_ids)):
            raise ValueError(
                "V6.7 candidate binding families must be sorted and unique"
            )
        if self.binding_hash and self.binding_hash != self.content_hash():
            raise ValueError("V6.7 candidate execution binding hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "binding_hash")

    def assert_sealed(self) -> None:
        type(self).model_validate(self.model_dump(mode="json"))
        if not self.binding_hash or self.binding_hash != self.content_hash():
            raise PreDataProtocolError("V6.7 candidate execution binding is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CandidateExecutionBindingV67":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"binding_hash"})
        payload["binding_hash"] = draft.content_hash()
        return cls(**payload)


def bind_candidate_to_predata_protocol_v67(
    *,
    candidate: CandidateFormalizationV50,
    execution_intent: RegisteredFamilySearchIntentV62,
    execution_ir: RegisteredFamilySearchIRV62,
    protocol: PreDataExecutionProtocolV67,
) -> CandidateExecutionBindingV67:
    """Compile the additive V6.7 authority over legacy V6.2 artifacts."""

    protocol.assert_sealed()
    execution_ir.assert_sealed()
    structural_hash = candidate.structural_hash()
    intent_hash = execution_intent.content_hash()
    adapter_id = protocol.adapter_binding.adapter_id
    if (
        candidate.candidate_id != execution_intent.candidate_id
        or candidate.candidate_id != execution_ir.candidate_id
        or execution_ir.candidate_structural_hash != structural_hash
        or execution_ir.model_intent_hash != intent_hash
        or adapter_id not in execution_intent.allowed_adapter_ids
        or adapter_id not in execution_ir.allowed_adapter_ids
    ):
        raise PreDataProtocolError(
            "candidate, legacy execution artifacts, and V6.7 protocol differ"
        )
    return CandidateExecutionBindingV67.seal(
        workspace_spec_hash=protocol.workspace_spec_hash,
        s0_gate_hash=protocol.s0_gate_hash,
        source_contract_hash=protocol.source_contract_hash,
        measurement_contract_hash=protocol.measurement_contract_hash,
        predata_protocol_hash=protocol.protocol_hash,
        candidate_id=candidate.candidate_id,
        candidate_structural_hash=structural_hash,
        legacy_v62_intent_hash=intent_hash,
        legacy_v62_ir_hash=execution_ir.ir_hash,
        selected_adapter_id=adapter_id,
        selected_adapter_version=protocol.adapter_binding.adapter_version,
        capability_pack_hash=protocol.adapter_binding.capability_pack_hash,
        allowed_adapter_ids=[adapter_id],
        allowed_family_ids=sorted(protocol.adapter_binding.allowed_families),
        allowed_family_registry_hash=(
            protocol.adapter_binding.allowed_family_registry_hash
        ),
    )


def _adapter_binding(pack: CapabilityPackV60) -> AdapterBindingV67:
    adapter_id = cast(AdapterIdV62, pack.pack_id)
    if adapter_id == SCALAR_ODE_ADAPTER_ID:
        scientific_class = ode_system_v52.ODELevelAdapterV52
        builder = ode_system_v52.build_ode_bundle_v52
        builder_ref = "fma.v5_2.ode_system.build_ode_bundle_v52"
        run_ref = "fma.v5_2.ode_system.ODELevelAdapterV52.run"
    else:
        scientific_class = adaptive_series_v57.AdaptivePositiveSeriesLevelAdapterV57
        builder = adaptive_series_v57.build_adaptive_positive_series_bundle_v57
        builder_ref = (
            "fma.v5_7.adaptive_positive_series."
            "build_adaptive_positive_series_bundle_v57"
        )
        run_ref = (
            "fma.v5_7.adaptive_positive_series."
            "AdaptivePositiveSeriesLevelAdapterV57.run"
        )
    if pack.scientific_adapter_ids != [scientific_class.adapter_id]:
        raise PreDataProtocolError(
            "capability pack and registered scientific adapter differ"
        )
    return AdapterBindingV67(
        adapter_id=adapter_id,
        adapter_version=pack.pack_version,
        capability_pack_hash=cast(str, pack.pack_hash),
        executor_id=pack.executor_id,
        scientific_adapter_id=scientific_class.adapter_id,
        scientific_adapter_version=scientific_class.adapter_version,
        allowed_families=list(registered_families_for_adapter_v62(adapter_id)),
        allowed_family_registry_hash=allowed_family_registry_hash_v62(adapter_id),
        execution_builder_ref=builder_ref,
        execution_builder_semantic_hash=_callable_semantic_hash(builder),
        scientific_adapter_run_ref=run_ref,
        scientific_adapter_run_semantic_hash=_callable_semantic_hash(
            scientific_class.run
        ),
    )


def _threshold_bundle(adapter_id: AdapterIdV62) -> ThresholdBundleV67:
    profile = frozen_s0_evaluation_profile_v66()
    profile.assert_sealed()
    success = ScientificSuccessThresholdsV61.seal()
    success.assert_sealed()
    if adapter_id == SCALAR_ODE_ADAPTER_ID:
        ode = ODEThresholdsV52.seal()
        ode.assert_sealed()
        return ThresholdBundleV67(
            adapter_id=adapter_id,
            s0_evaluation_profile_hash=cast(str, profile.profile_hash),
            ode_thresholds=ode,
            scientific_success_thresholds=success,
        )
    hybrid = HybridODEThresholdsV56.seal()
    adaptive = _adaptive_thresholds_v66()
    hybrid.assert_sealed()
    adaptive.assert_sealed()
    return ThresholdBundleV67(
        adapter_id=adapter_id,
        s0_evaluation_profile_hash=cast(str, profile.profile_hash),
        hybrid_thresholds=hybrid,
        adaptive_thresholds=adaptive,
        scientific_success_thresholds=success,
    )


def minimum_predata_observation_count_v67(adapter_id: str) -> int:
    """Return the frozen minimum that can execute and confirm one adapter."""

    pack = registered_positive_series_capability_pack_v67(adapter_id)
    resolved_adapter_id = cast(AdapterIdV62, pack.pack_id)
    success = _threshold_bundle(resolved_adapter_id).scientific_success_thresholds
    if resolved_adapter_id == ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID:
        confirmation_minimum = (
            success.adaptive_minimum_history_points
            + success.adaptive_confirmation_folds
        )
    else:
        confirmation_minimum = (
            success.ode_minimum_history_points + success.ode_confirmation_folds
        )
    return max(pack.minimum_observations, confirmation_minimum)


def _ode_parameter_rules() -> list[ParameterBoundRuleV67]:
    return [
        ParameterBoundRuleV67(
            family_id="constant",
            parameter_names=[],
            bound_rule_id="v52.constant.no_parameters",
            exact_rule="No fitted parameters; prediction remains at x0.",
        ),
        ParameterBoundRuleV67(
            family_id="exponential",
            parameter_names=["r"],
            bound_rule_id="v52.exponential.rate_bounds",
            exact_rule=(
                "span=max(t_last-t_first,1e-9); rate=1/span; "
                "r in [-10*rate,10*rate]; "
                "r0=max(log(y_last/y_first)/span,0)."
            ),
        ),
        ParameterBoundRuleV67(
            family_id="gompertz",
            parameter_names=["K", "r"],
            bound_rule_id="v52.gompertz.capacity_rate_bounds",
            exact_rule=(
                "span=max(t_last-t_first,1e-9); rate=1/span; "
                "lower_K=max(max_y*1.0001,y0*1.001); "
                "upper_K=max(lower_K*2,max_y*100); "
                "r in [-10*rate,10*rate], K in [lower_K,upper_K]; "
                "initial=(rate,max(max_y*1.5,lower_K*1.1))."
            ),
        ),
        ParameterBoundRuleV67(
            family_id="logistic",
            parameter_names=["K", "r"],
            bound_rule_id="v52.logistic.capacity_rate_bounds",
            exact_rule=(
                "span=max(t_last-t_first,1e-9); rate=1/span; "
                "lower_K=max(max_y*1.0001,y0*1.001); "
                "upper_K=max(lower_K*2,max_y*100); "
                "r in [-10*rate,10*rate], K in [lower_K,upper_K]; "
                "initial=(rate,max(max_y*1.5,lower_K*1.1))."
            ),
        ),
    ]


def _adaptive_parameter_rules() -> list[ParameterBoundRuleV67]:
    primary = [
        ParameterBoundRuleV67(
            family_id="constant",
            parameter_names=[],
            bound_rule_id="v56.constant.no_parameters",
            exact_rule=("No fitted trend parameters; dimensionless trend equals x0."),
        ),
        ParameterBoundRuleV67(
            family_id="exponential",
            parameter_names=["r"],
            bound_rule_id="v56.exponential.dimensionless_rate_bounds",
            exact_rule=(
                "Dimensionless r_times_span in [-10,10]; "
                "initial=clip(log(y_last/y_first),-2,2); "
                "physical r=r_times_span/time_span."
            ),
        ),
        ParameterBoundRuleV67(
            family_id="gompertz",
            parameter_names=["K", "r"],
            bound_rule_id="v56.gompertz.dimensionless_bounds",
            exact_rule=(
                "With y scaled by mean(y), lower_K=max(max_y*1.0001,"
                "x0*1.001), upper_K=max(lower_K*2,max_y*100); "
                "r_times_span in [-10,10], K_over_scale in "
                "[lower_K,upper_K]."
            ),
        ),
        ParameterBoundRuleV67(
            family_id="logistic",
            parameter_names=["K", "r"],
            bound_rule_id="v56.logistic.dimensionless_bounds",
            exact_rule=(
                "With y scaled by mean(y), lower_K=max(max_y*1.0001,"
                "x0*1.001), upper_K=max(lower_K*2,max_y*100); "
                "r_times_span in [-10,10], K_over_scale in "
                "[lower_K,upper_K]."
            ),
        ),
    ]
    return [
        *primary,
        ParameterBoundRuleV67(
            family_id="log_random_walk_drift",
            parameter_names=["mean_log_growth"],
            bound_rule_id="v57.log_growth.drift_closed_form",
            exact_rule=(
                "mean_log_growth=mean(diff(log(training_y))); phi=0; "
                "innovation scale is the sample standard deviation."
            ),
        ),
        ParameterBoundRuleV67(
            family_id="log_growth_ar1",
            parameter_names=["mean_log_growth", "phi"],
            bound_rule_id="v57.log_growth.ar1_closed_form",
            exact_rule=(
                "mean_log_growth=mean(diff(log(training_y))); raw_phi is "
                "centered lag-one least squares with zero fallback when "
                "denominator<=1e-18; effective_phi=clip(raw_phi,-0.999,0.999)."
            ),
        ),
    ]


def _fitting_policy(adapter_id: AdapterIdV62) -> FittingPolicyV67:
    common_optimizer: dict[Identifier, SettingValueV67] = {
        "ftol": 1e-12,
        "gtol": 1e-12,
        "max_nfev": 4000,
        "method": "trf",
        "xtol": 1e-12,
    }
    if adapter_id == SCALAR_ODE_ADAPTER_ID:
        refs = [
            "fma.v5_2.ode_system._bounds",
            "fma.v5_2.ode_system.fit_ode_v52",
        ]
        hashes = {
            "bounds_v52": _callable_semantic_hash(ode_system_v52._bounds),
            "fit_ode_v52": _callable_semantic_hash(ode_system_v52.fit_ode_v52),
        }
        objective = (
            "Minimize least-squares residuals "
            "(predict(family,t,x0,parameters)-y)/max(mean(y),1e-12)."
        )
        selection = (
            "Evaluate constant, exponential, gompertz, and logistic on the "
            "same chronological development split; minimize "
            "validation_relative_rmse+0.005*parameter_count, then "
            "parameter_count, then candidate_id."
        )
        rules = _ode_parameter_rules()
    else:
        refs = [
            "fma.v5_6.hybrid_ode._fit_trend",
            "fma.v5_7.adaptive_positive_series._estimate_growth_process",
        ]
        hashes = {
            "estimate_growth_process_v57": _callable_semantic_hash(
                adaptive_series_v57._estimate_growth_process
            ),
            "fit_trend_v56": _callable_semantic_hash(hybrid_ode_v56._fit_trend),
        }
        objective = (
            "Primary dimensionless ODE trends minimize unscaled "
            "least-squares residuals; growth drift and AR1 parameters use "
            "the frozen closed-form log-growth estimators."
        )
        selection = (
            "Run the complete V5.6 hybrid ODE graph first; trigger both "
            "registered V5.7 log-growth candidates only when a required "
            "primary L1-L4 level is not PASS; among scientifically admissible "
            "growth candidates minimize validation_relative_rmse plus the "
            "frozen per-parameter penalty, then parameter_count and "
            "candidate_id; if none is admissible select unresolved and abstain."
        )
        rules = _adaptive_parameter_rules()
    return FittingPolicyV67(
        adapter_id=adapter_id,
        fit_implementation_refs=refs,
        fit_implementation_semantic_hashes=hashes,
        fitting_objective=objective,
        optimizer_id="scipy.optimize.least_squares.trf",
        optimizer_settings=common_optimizer,
        parameter_bound_rules=rules,
        candidate_selection_rule=selection,
    )


def _compatibility_policy(
    *,
    contract: MeasurementStudyDesignContractV67,
    pack: CapabilityPackV60,
    thresholds: ThresholdBundleV67,
) -> CompatibilityPolicyV67:
    success = thresholds.scientific_success_thresholds
    adaptive = pack.pack_id == ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID
    minimum_history = (
        success.adaptive_minimum_history_points
        if adaptive
        else success.ode_minimum_history_points
    )
    folds = (
        success.adaptive_confirmation_folds
        if adaptive
        else success.ode_confirmation_folds
    )
    return CompatibilityPolicyV67(
        adapter_id=cast(AdapterIdV62, pack.pack_id),
        allowed_state_kinds=pack.state_kinds,
        allowed_time_kinds=pack.time_kinds,
        allowed_dynamics_kinds=pack.dynamics_kinds,
        allowed_observation_kinds=pack.observation_kinds,
        minimum_execution_observation_count=pack.minimum_observations,
        minimum_confirmation_observation_count=minimum_history + folds,
        finite_positive_values_required=pack.requires_positive_observations,
        strictly_increasing_time_required=(pack.requires_strictly_increasing_time),
        effectively_regular_cadence_required=adaptive,
        maximum_cadence_relative_deviation=1e-9 if adaptive else None,
        exact_measurement_unit_required=contract.measurement.unit,
        exact_time_basis_required=contract.measurement.time_basis,
    )


def _training_policy(
    *,
    pack: CapabilityPackV60,
    thresholds: ThresholdBundleV67,
) -> TrainingConfirmationPolicyV67:
    adapter_id = cast(AdapterIdV62, pack.pack_id)
    success = thresholds.scientific_success_thresholds
    if adapter_id == SCALAR_ODE_ADAPTER_ID:
        assert thresholds.ode_thresholds is not None
        split_fraction = thresholds.ode_thresholds.split_fraction
        minimum_slice = thresholds.ode_thresholds.minimum_points_per_slice
        minimum_history = success.ode_minimum_history_points
        folds = success.ode_confirmation_folds
    else:
        assert thresholds.adaptive_thresholds is not None
        split_fraction = thresholds.adaptive_thresholds.split_fraction
        minimum_slice = thresholds.adaptive_thresholds.minimum_points_per_slice
        minimum_history = success.adaptive_minimum_history_points
        folds = success.adaptive_confirmation_folds
    from .scientific_success import evaluate_rolling_confirmation_v61

    return TrainingConfirmationPolicyV67(
        adapter_id=adapter_id,
        development_split_fraction=split_fraction,
        minimum_points_per_development_slice=minimum_slice,
        development_baseline_ids=pack.baseline_ids,
        confirmation_implementation_semantic_hash=(
            _callable_semantic_hash(evaluate_rolling_confirmation_v61)
        ),
        confirmation_minimum_history=minimum_history,
        confirmation_fold_count=folds,
        confirmation_minimum_total_observations=minimum_history + folds,
        comparison_metric_ids=sorted(
            [
                "admissible_fold_fraction",
                "confirmation_interval_coverage",
                "confirmation_relative_rmse",
                "persistence_relative_improvement",
                "residual_lag1_correlation",
            ]
        ),
    )


def _uncertainty_policy(
    thresholds: ThresholdBundleV67,
) -> UncertaintyStressPolicyV67:
    adapter_id = thresholds.adapter_id
    if adapter_id == SCALAR_ODE_ADAPTER_ID:
        assert thresholds.ode_thresholds is not None
        ode = thresholds.ode_thresholds
        counts = {"ode_training_residual": ode.bootstrap_replicates}
        seeds = {"ode_training_residual": ode.bootstrap_seed}
        bootstrap = (
            "For the selected ODE family, sample training residuals IID with "
            "replacement, add them to fitted training states with a positive "
            "floor, refit the same family, reject nonconverged/nonpositive "
            "replicates, and take frozen forecast quantiles."
        )
        dependence = (
            "No block bootstrap is used. Temporal dependence is diagnosed by "
            "absolute lag-one residual correlation and is not claimed to be "
            "covered by the IID residual bootstrap."
        )
        interval = (
            "Development coverage uses selected-fit training RMSE times 1.96; "
            "L4 forecast intervals use the 0.025, 0.5, and 0.975 quantiles of "
            "successful training-residual bootstrap forecasts."
        )
        stress_ids = sorted(
            [
                "candidate_ensemble_disagreement",
                "fit_window_0.65_0.80_1.00",
            ]
        )
        stress_refs = sorted(
            [
                "fma.v5_2.ode_system._l4_evidence:ensemble_forecast_cv",
                "fma.v5_2.ode_system._l4_evidence:window_forecasts_0.65_0.80_1.00",
            ]
        )
    else:
        assert thresholds.hybrid_thresholds is not None
        assert thresholds.adaptive_thresholds is not None
        hybrid = thresholds.hybrid_thresholds
        adaptive = thresholds.adaptive_thresholds
        counts = {
            "adaptive_growth_innovation": adaptive.bootstrap_replicates,
            "hybrid_primary_innovation": hybrid.bootstrap_replicates,
        }
        seeds = {
            "adaptive_growth_innovation": adaptive.bootstrap_seed,
            "hybrid_primary_innovation": hybrid.bootstrap_seed,
        }
        bootstrap = (
            "The V5.6 primary branch resamples fitted innovations IID and "
            "recurses through the frozen AR1 residual process before refit; "
            "the V5.7 growth branch resamples training log-growth innovations "
            "IID, recurses through the selected drift/AR1 growth process, "
            "refits, and takes frozen forecast quantiles."
        )
        dependence = (
            "Serial dependence is represented only by the registered AR1 "
            "residual or log-growth recursion and checked with lag-one, phi, "
            "window, and innovation-shift diagnostics. No block bootstrap or "
            "general temporal-dependence coverage guarantee is supplied."
        )
        interval = (
            "One-step development intervals use the registered innovation "
            "scale times 1.96; L4 intervals use the 0.025, 0.5, and 0.975 "
            "quantiles of successful conditional-innovation bootstrap "
            "forecasts for the graph-selected branch."
        )
        stress_ids = sorted(
            [
                "growth_drift_and_phi_window_stability",
                "innovation_mean_and_single_shock",
                "primary_recovery_graph_ablation",
                "selected_branch_window_0.70_0.85_1.00",
            ]
        )
        stress_refs = sorted(
            [
                "fma.v5_6.hybrid_ode._candidate_evidence:innovation_stress",
                "fma.v5_6.hybrid_ode._l4_evidence:window_forecasts_0.70_0.85_1.00",
                "fma.v5_7.adaptive_positive_series._growth_candidate:window_and_shift_stress",
                "fma.v5_7.adaptive_positive_series.build_adaptive_positive_series_bundle_v57:graph_recovery",
            ]
        )
    return UncertaintyStressPolicyV67(
        adapter_id=adapter_id,
        bootstrap_rule=bootstrap,
        bootstrap_replicates_by_component=counts,
        bootstrap_seed_by_component=seeds,
        dependence_rule=dependence,
        interval_rule=interval,
        interval_quantiles=[0.025, 0.5, 0.975],
        stress_rule_ids=stress_ids,
        stress_rule_refs=stress_refs,
    )


def _level_rules(
    *,
    adapter_binding: AdapterBindingV67,
    thresholds: ThresholdBundleV67,
) -> list[LevelRuleV67]:
    if adapter_binding.adapter_id == SCALAR_ODE_ADAPTER_ID:
        prefix = "scalar_ode"
        implementation = ode_system_v52.ODELevelAdapterV52.run
        ref = "fma.v5_2.ode_system.ODELevelAdapterV52.run"
    else:
        prefix = "adaptive_positive_series"
        implementation = adaptive_series_v57.AdaptivePositiveSeriesLevelAdapterV57.run
        ref = (
            "fma.v5_7.adaptive_positive_series."
            "AdaptivePositiveSeriesLevelAdapterV57.run"
        )
    implementation_hash = _callable_semantic_hash(implementation)
    hashes = thresholds.threshold_hashes()
    levels: list[LevelV67] = ["L0", "L1", "L2", "L3", "L4"]
    return [
        LevelRuleV67(
            level=level,
            stage="S3" if level in {"L0", "L1", "L2"} else "S4",
            check_id=f"{prefix}_{level.lower()}",
            adapter_id=adapter_binding.scientific_adapter_id,
            adapter_version=adapter_binding.scientific_adapter_version,
            implementation_ref=f"{ref}:{level}",
            implementation_semantic_hash=implementation_hash,
            threshold_hashes=hashes,
        )
        for level in levels
    ]


def compile_predata_execution_protocol_v67(
    *,
    measurement_contract: MeasurementStudyDesignContractV67,
    capability_pack: CapabilityPackV60,
) -> PreDataExecutionProtocolV67:
    """Compile one exact pre-data protocol without accepting observation data."""

    try:
        measurement_contract.assert_sealed()
    except ValueError as exc:
        raise PreDataProtocolError(
            "pre-data compilation requires a sealed measurement contract"
        ) from exc
    pack = _exact_registered_pack(capability_pack)
    if measurement_contract.claim_kind != "predictive":
        raise PreDataProtocolError(
            "registered positive-series protocol supports predictive claims only"
        )
    if measurement_contract.study_design.design_type != "time_series":
        raise PreDataProtocolError(
            "registered positive-series protocol requires a time-series design"
        )
    if measurement_contract.measurement.scale_type not in {"ratio", "count"}:
        raise PreDataProtocolError(
            "positive-series protocol requires a ratio or count measurement"
        )
    if measurement_contract.missingness.handling_policy != ("reject_incomplete_series"):
        raise PreDataProtocolError(
            "registered positive-series adapters reject missing observations"
        )

    adapter_id = cast(AdapterIdV62, pack.pack_id)
    thresholds = _threshold_bundle(adapter_id)
    compatibility = _compatibility_policy(
        contract=measurement_contract,
        pack=pack,
        thresholds=thresholds,
    )
    if (
        measurement_contract.sampling.minimum_sample_size
        < compatibility.minimum_confirmation_observation_count
    ):
        raise PreDataProtocolError(
            "measurement design minimum sample cannot run the frozen "
            "rolling-origin confirmation"
        )
    adapter_binding = _adapter_binding(pack)
    protocol_id = (
        "predata-"
        + sha256_value(
            {
                "compiler_id": PREDATA_COMPILER_ID_V67,
                "measurement_contract_hash": (measurement_contract.contract_hash),
                "capability_pack_hash": pack.pack_hash,
            }
        )[:24]
    )
    return PreDataExecutionProtocolV67.seal(
        protocol_id=protocol_id,
        workspace_spec_hash=measurement_contract.workspace_spec_hash,
        s0_gate_hash=measurement_contract.s0_gate_hash,
        source_contract_id=measurement_contract.source_contract_id,
        source_contract_hash=measurement_contract.source_contract_hash,
        measurement_contract_id=measurement_contract.contract_id,
        measurement_contract_hash=cast(str, measurement_contract.contract_hash),
        measurement_id=measurement_contract.measurement.measurement_id,
        allowed_adapter_ids=[adapter_id],
        allowed_adapter_versions={adapter_id: adapter_binding.adapter_version},
        adapter_binding=adapter_binding,
        adapter_resolution=AdapterResolutionPolicyV67(selected_adapter_id=adapter_id),
        compatibility=compatibility,
        thresholds=thresholds,
        fitting=_fitting_policy(adapter_id),
        training_and_confirmation=_training_policy(
            pack=pack,
            thresholds=thresholds,
        ),
        uncertainty_and_stress=_uncertainty_policy(thresholds),
        level_rules=_level_rules(
            adapter_binding=adapter_binding,
            thresholds=thresholds,
        ),
        abstention=AbstentionPolicyV67(),
    )


def verify_predata_execution_protocol_v67(
    *,
    measurement_contract: MeasurementStudyDesignContractV67,
    capability_pack: CapabilityPackV60,
    protocol: PreDataExecutionProtocolV67,
) -> bool:
    """Replay compilation from public pre-data inputs and compare exactly."""

    try:
        protocol.assert_sealed()
        expected = compile_predata_execution_protocol_v67(
            measurement_contract=measurement_contract,
            capability_pack=capability_pack,
        )
    except (PreDataProtocolError, TypeError, ValueError):
        return False
    return protocol == expected


__all__ = [
    "AbstentionPolicyV67",
    "AdapterBindingV67",
    "AdapterResolutionPolicyV67",
    "CANDIDATE_EXECUTION_BINDING_PATH_V67",
    "CandidateExecutionBindingV67",
    "CompatibilityPolicyV67",
    "FittingPolicyV67",
    "LevelRuleV67",
    "PREDATA_COMPILER_ID_V67",
    "PREDATA_EXECUTION_PROTOCOL_PATH_V67",
    "ParameterBoundRuleV67",
    "PreDataExecutionProtocolV67",
    "PreDataProtocolError",
    "ThresholdBundleV67",
    "TrainingConfirmationPolicyV67",
    "UncertaintyStressPolicyV67",
    "compile_predata_execution_protocol_v67",
    "bind_candidate_to_predata_protocol_v67",
    "minimum_predata_observation_count_v67",
    "registered_positive_series_capability_pack_v67",
    "verify_predata_execution_protocol_v67",
]
