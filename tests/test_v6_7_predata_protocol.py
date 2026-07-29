from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from fma.hashing import canonical_json
from fma.v5.workspace_schemas import CandidateFormalizationV50
from fma.v6.executable_candidate import (
    ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
    SCALAR_ODE_ADAPTER_ID,
    RegisteredFamilySearchIntentV62,
    compile_registered_family_search_ir_v62,
)
from fma.v6.measurement_study_design import (
    MeasurementStudyDesignContractV67,
)
from fma.v6.predata_protocol import (
    CandidateExecutionBindingV67,
    PREDATA_COMPILER_ID_V67,
    PreDataExecutionProtocolV67,
    PreDataProtocolError,
    bind_candidate_to_predata_protocol_v67,
    compile_predata_execution_protocol_v67,
    registered_positive_series_capability_pack_v67,
    verify_predata_execution_protocol_v67,
)
from fma.v6.recovery_kernel import CapabilityPackV60
from fma.v6.scientific_success import ScientificSuccessThresholdsV61
from fma.v6.stage_review_recovery import (
    _adaptive_thresholds_v66,
    frozen_s0_evaluation_profile_v66,
)


def _measurement_contract(
    *,
    minimum_sample_size: int = 40,
    missingness_policy: str = "reject_incomplete_series",
    scale_type: str = "ratio",
) -> MeasurementStudyDesignContractV67:
    return MeasurementStudyDesignContractV67.seal(
        contract_id="task-measurement-v67",
        workspace_spec_hash="a" * 64,
        s0_gate_hash="b" * 64,
        source_contract_id="registered-source-v67",
        source_contract_hash="c" * 64,
        claim_kind="predictive",
        claim_scope=(
            "Predict the next value of the same registered positive scalar "
            "series without causal or policy interpretation."
        ),
        construct_definition={
            "construct_id": "registered_positive_state",
            "name": "registered positive state",
            "conceptual_definition": (
                "The positive scalar state represented by the exact "
                "predeclared operational measure."
            ),
            "role": "outcome",
            "representation": "proxy",
            "representation_rationale": (
                "The registered measure is an operational proxy for the "
                "broader target construct."
            ),
        },
        measurement={
            "measurement_id": "positive_state_measure",
            "construct_id": "registered_positive_state",
            "operational_definition": (
                "One annual positive scalar measurement under the unchanged "
                "registered source definition."
            ),
            "unit": "registered_state_unit",
            "time_basis": "calendar year",
            "aggregation_basis": "one aggregate value per calendar year",
            "scale_type": scale_type,
            "source_definition": (
                "The exact authenticated source contract will be bound in S2."
            ),
            "directionality": "higher_is_more",
        },
        population={
            "population_id": "registered_years",
            "target_population": (
                "All eligible calendar years inside the frozen task boundary."
            ),
            "unit_of_analysis": "calendar year",
            "spatial_scope": "registered aggregate",
            "temporal_scope": "frozen historical interval",
            "inclusion_criteria": [
                "Same registered definition and unit",
                "Inside the frozen interval",
            ],
            "exclusion_criteria": ["Duplicate time index"],
        },
        sampling={
            "sampling_frame": (
                "Every eligible annual record exposed by the exact registered "
                "source and frozen interval."
            ),
            "sampling_method": "administrative_complete_series",
            "selection_rule": (
                "Use every eligible year without post-result deletion or "
                "cherry-picking."
            ),
            "minimum_sample_size": minimum_sample_size,
            "stopping_rule": (
                "Stop at the frozen end year and do not extend after results."
            ),
            "representativeness_limitations": (
                "The administrative aggregate does not represent individual "
                "units or another indicator."
            ),
        },
        missingness={
            "anticipated_sources": ["Official source may omit an eligible year"],
            "mechanism_assumptions": [
                "Missingness is not assumed random before post-data checks"
            ],
            "handling_policy": missingness_policy,
            "sensitivity_analysis_plan": (
                "Reject the narrow adapter and require a new frozen graph "
                "attempt before another missing-data treatment."
            ),
        },
        measurement_error={
            "anticipated_error_sources": ["Source revisions"],
            "error_structure_assumption": (
                "Revision error magnitude and dependence are unknown pre-data."
            ),
            "calibration_or_reference_plan": (
                "Use an independently authenticated vintage when available."
            ),
            "propagation_or_sensitivity_plan": (
                "Preserve the unresolved error as a claim limitation."
            ),
        },
        bias={
            "anticipated_biases": ["Frozen-interval selection bias"],
            "mitigation_plan": (
                "Freeze the interval and preserve all failed validations."
            ),
            "residual_bias_policy": (
                "Do not elevate a local forecast to a population claim."
            ),
        },
        confounding={
            "relevance": "not_applicable_to_noncausal_claim",
            "identification_or_control_strategy": (
                "No causal effect is identified in this predictive task."
            ),
            "unmeasured_confounding_policy": (
                "Associations cannot be interpreted as interventions."
            ),
        },
        study_design={
            "design_type": "time_series",
            "target_quantity": (
                "A one-step predictive distribution for the same measure."
            ),
            "temporal_ordering": (
                "Every training prefix precedes its forecast target."
            ),
            "comparison_strategy": (
                "Compare registered candidates and frozen naive baselines."
            ),
            "validation_design": (
                "Use leakage-safe nested rolling-origin confirmation."
            ),
            "leakage_prevention_plan": (
                "Fit and select only on each chronological prefix."
            ),
        },
        applicability={
            "intended_use": (
                "Local retrospective validation and bounded one-step forecast."
            ),
            "in_scope_conditions": ["Same measure and annual time basis"],
            "out_of_scope_conditions": ["Causal or policy recommendation"],
            "transport_assumptions": ["Authenticated source definition is stable"],
            "abstention_conditions": ["Any required registered gate fails"],
        },
        ethics={
            "risk_level": "minimal",
            "human_participant_data_expected": False,
            "sensitive_data_expected": False,
            "consent_or_legal_basis_plan": (
                "Use only the declared aggregate public source."
            ),
            "prohibited_uses": ["Automated consequential action"],
            "ethics_review_required": False,
        },
    )


def _protocol(adapter_id: str = "scalar_autonomous_ode_v52"):
    contract = _measurement_contract()
    pack = registered_positive_series_capability_pack_v67(adapter_id)
    protocol = compile_predata_execution_protocol_v67(
        measurement_contract=contract,
        capability_pack=pack,
    )
    return contract, pack, protocol


def test_scalar_protocol_is_deterministic_sealed_and_observation_free() -> None:
    contract, pack, first = _protocol()
    second = compile_predata_execution_protocol_v67(
        measurement_contract=contract,
        capability_pack=pack,
    )

    first.assert_sealed()
    assert first == second
    assert canonical_json(first) == canonical_json(second)
    assert first.compiler_id == PREDATA_COMPILER_ID_V67
    assert first.measurement_contract_hash == contract.contract_hash
    assert first.workspace_spec_hash == contract.workspace_spec_hash
    assert first.s0_gate_hash == contract.s0_gate_hash
    assert first.source_contract_id == contract.source_contract_id
    assert first.source_contract_hash == contract.source_contract_hash
    assert first.adapter_binding.capability_pack_hash == pack.pack_hash
    assert first.allowed_adapter_ids == ["scalar_autonomous_ode_v52"]
    assert first.allowed_adapter_versions == {"scalar_autonomous_ode_v52": "v5.2"}
    assert first.adapter_resolution.adapter_resolution_stage == "pre_data_compiler"
    assert first.adapter_resolution.s2_role == "compatibility_validation_only"
    assert first.adapter_resolution.silent_adapter_substitution_permitted is False
    assert first.observation_values_accessed_during_compilation is False
    assert first.observed_statistics_accessed_during_compilation is False
    assert first.model_text_executable is False
    assert first.protocol_is_scientific_evidence is False
    assert first.scientific_qualification_granted is False
    assert first.real_world_action_authorized is False
    assert verify_predata_execution_protocol_v67(
        measurement_contract=contract,
        capability_pack=pack,
        protocol=first,
    )

    parameters = set(
        inspect.signature(compile_predata_execution_protocol_v67).parameters
    )
    assert parameters == {"measurement_contract", "capability_pack"}


def test_candidate_binding_makes_v67_single_adapter_authoritative() -> None:
    _contract, _pack, protocol = _protocol()
    candidate = CandidateFormalizationV50(
        candidate_id="candidate.mechanistic",
        model_family="registered positive scalar time-series search",
        mathematical_form="dx/dt = f(x; theta)",
        assumption_ids=["assumption.positive"],
        symbol_ids=["symbol.state", "symbol.time"],
        data_requirement_ids=["data.positive_series"],
        validation_obligation_ids=[
            "s3_l0_structure",
            "s3_l1_structural",
            "s3_l2_numerical",
            "s4_l3_holdout",
            "s4_l4_uncertainty",
        ],
        abandon_criteria=[
            "Abandon when registered validation rejects every family."
        ],
        lineage="blind mechanistic proposal",
    )
    legacy_intent = RegisteredFamilySearchIntentV62(
        candidate_id=candidate.candidate_id,
        allowed_adapter_ids=[
            SCALAR_ODE_ADAPTER_ID,
            ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID,
        ],
    )
    legacy_ir = compile_registered_family_search_ir_v62(
        candidate,
        legacy_intent,
    )

    binding = bind_candidate_to_predata_protocol_v67(
        candidate=candidate,
        execution_intent=legacy_intent,
        execution_ir=legacy_ir,
        protocol=protocol,
    )

    binding.assert_sealed()
    assert binding.allowed_adapter_ids == [SCALAR_ODE_ADAPTER_ID]
    assert binding.selected_adapter_id == SCALAR_ODE_ADAPTER_ID
    assert binding.adapter_resolution_stage == "pre_data_compiler"
    assert binding.s2_role == "compatibility_validation_only"
    assert binding.legacy_v62_resolution_is_authority is False
    assert binding.executable_semantics_authority == (
        "predata_execution_protocol_v67"
    )
    assert binding.silent_adapter_substitution_permitted is False
    assert binding.predata_protocol_hash == protocol.protocol_hash

    tampered = binding.model_dump(mode="json")
    tampered["selected_adapter_id"] = ADAPTIVE_POSITIVE_SERIES_ADAPTER_ID
    with pytest.raises(ValidationError):
        CandidateExecutionBindingV67.model_validate(tampered)


def test_scalar_protocol_freezes_fit_split_baselines_and_exact_l0_l4_refs() -> None:
    _contract, _pack, protocol = _protocol()

    assert protocol.adapter_binding.allowed_families == [
        "constant",
        "exponential",
        "gompertz",
        "logistic",
    ]
    assert protocol.fitting.optimizer_id == ("scipy.optimize.least_squares.trf")
    assert protocol.fitting.optimizer_settings == {
        "ftol": 1e-12,
        "gtol": 1e-12,
        "max_nfev": 4000,
        "method": "trf",
        "xtol": 1e-12,
    }
    assert [
        item.family_id for item in protocol.fitting.parameter_bound_rules
    ] == protocol.adapter_binding.allowed_families
    exponential = protocol.fitting.parameter_bound_rules[1]
    assert "r in [-10*rate,10*rate]" in exponential.exact_rule
    assert exponential.model_supplied_bounds_permitted is False
    assert (
        protocol.training_and_confirmation.development_split_index_rule
        == "min(max(int(n*split_fraction),2),n-2)"
    )
    assert protocol.training_and_confirmation.development_baseline_ids == [
        "constant",
        "persistence",
    ]
    assert (
        protocol.training_and_confirmation.confirmation_method
        == "nested_rolling_origin_one_step"
    )
    assert protocol.training_and_confirmation.confirmation_minimum_history == 17
    assert protocol.training_and_confirmation.confirmation_fold_count == 6
    assert protocol.compatibility.minimum_confirmation_observation_count == 23
    assert [item.level for item in protocol.level_rules] == [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    ]
    assert [item.check_id for item in protocol.level_rules] == [
        "scalar_ode_l0",
        "scalar_ode_l1",
        "scalar_ode_l2",
        "scalar_ode_l3",
        "scalar_ode_l4",
    ]
    assert [item.stage for item in protocol.level_rules] == [
        "S3",
        "S3",
        "S3",
        "S4",
        "S4",
    ]
    assert all(item.not_run_counts_as_pass is False for item in protocol.level_rules)
    assert all(
        len(item.implementation_semantic_hash) == 64 for item in protocol.level_rules
    )


def test_adaptive_protocol_binds_graph_dependence_uq_and_abstention() -> None:
    contract, pack, protocol = _protocol("adaptive_positive_series_v57")

    assert protocol.allowed_adapter_ids == ["adaptive_positive_series_v57"]
    assert protocol.allowed_adapter_versions == {"adaptive_positive_series_v57": "v5.7"}
    assert protocol.adapter_binding.allowed_families == [
        "constant",
        "exponential",
        "gompertz",
        "logistic",
        "log_random_walk_drift",
        "log_growth_ar1",
    ]
    assert protocol.thresholds.ode_thresholds is None
    assert protocol.thresholds.hybrid_thresholds is not None
    assert protocol.thresholds.adaptive_thresholds == _adaptive_thresholds_v66()
    assert (
        protocol.thresholds.scientific_success_thresholds
        == ScientificSuccessThresholdsV61.seal()
    )
    assert (
        protocol.thresholds.s0_evaluation_profile_hash
        == frozen_s0_evaluation_profile_v66().profile_hash
    )
    assert protocol.compatibility.minimum_execution_observation_count == 26
    assert protocol.compatibility.minimum_confirmation_observation_count == 34
    assert protocol.compatibility.effectively_regular_cadence_required is True
    assert protocol.compatibility.maximum_cadence_relative_deviation == 1e-9
    assert set(protocol.uncertainty_and_stress.bootstrap_replicates_by_component) == {
        "adaptive_growth_innovation",
        "hybrid_primary_innovation",
    }
    assert protocol.uncertainty_and_stress.block_bootstrap_used is False
    assert (
        protocol.uncertainty_and_stress.temporal_dependence_coverage_guaranteed is False
    )
    assert protocol.uncertainty_and_stress.interval_quantiles == [0.025, 0.5, 0.975]
    assert (
        protocol.uncertainty_and_stress.interval_claim_ceiling
        == "diagnostic_interval_quality_only"
    )
    assert "unresolved and abstain" in (protocol.fitting.candidate_selection_rule)
    assert protocol.abstention.adaptive_unresolved_branch_action == "ABSTAIN"
    assert protocol.abstention.recovery_requires_new_graph_attempt is True
    assert protocol.abstention.recovery_requires_new_protocol_successor is True
    assert verify_predata_execution_protocol_v67(
        measurement_contract=contract,
        capability_pack=pack,
        protocol=protocol,
    )


def test_compiler_rejects_unsealed_or_nonregistered_pack() -> None:
    contract = _measurement_contract()
    pack = registered_positive_series_capability_pack_v67("scalar_autonomous_ode_v52")
    pack.executor_id = "tampered_executor"
    with pytest.raises(PreDataProtocolError, match="sealed capability pack"):
        compile_predata_execution_protocol_v67(
            measurement_contract=contract,
            capability_pack=pack,
        )

    original = registered_positive_series_capability_pack_v67(
        "scalar_autonomous_ode_v52"
    )
    payload = original.model_dump(exclude={"pack_hash"})
    payload["executor_id"] = "another_executor"
    replacement = CapabilityPackV60.seal(**payload)
    with pytest.raises(PreDataProtocolError, match="code-owned registry"):
        compile_predata_execution_protocol_v67(
            measurement_contract=contract,
            capability_pack=replacement,
        )

    with pytest.raises(PreDataProtocolError, match="unsupported"):
        registered_positive_series_capability_pack_v67("unknown_adapter")


def test_compiler_rejects_unsealed_measurement_and_incompatible_design() -> None:
    pack = registered_positive_series_capability_pack_v67("scalar_autonomous_ode_v52")
    contract = _measurement_contract()
    contract.measurement.unit = "changed_after_seal"
    with pytest.raises(PreDataProtocolError, match="sealed measurement"):
        compile_predata_execution_protocol_v67(
            measurement_contract=contract,
            capability_pack=pack,
        )

    incomplete = _measurement_contract(missingness_policy="complete_case")
    with pytest.raises(PreDataProtocolError, match="reject missing"):
        compile_predata_execution_protocol_v67(
            measurement_contract=incomplete,
            capability_pack=pack,
        )

    too_small = _measurement_contract(minimum_sample_size=22)
    with pytest.raises(
        PreDataProtocolError,
        match="cannot run the frozen rolling-origin confirmation",
    ):
        compile_predata_execution_protocol_v67(
            measurement_contract=too_small,
            capability_pack=pack,
        )


def test_tampering_breaks_seal_and_resealed_policy_fails_replay() -> None:
    contract, pack, protocol = _protocol()
    protocol.compatibility.minimum_execution_observation_count += 1

    with pytest.raises(
        (PreDataProtocolError, ValidationError),
        match="not sealed|protocol hash differs",
    ):
        protocol.assert_sealed()

    payload = protocol.model_dump(exclude={"protocol_hash"})
    resealed = PreDataExecutionProtocolV67.seal(**payload)
    resealed.assert_sealed()
    assert not verify_predata_execution_protocol_v67(
        measurement_contract=contract,
        capability_pack=pack,
        protocol=resealed,
    )


def test_protocol_schema_forbids_observations_and_authority_escalation() -> None:
    _contract, _pack, protocol = _protocol()

    payload = protocol.model_dump(mode="json")
    payload.pop("protocol_hash")
    payload["observations"] = [1.0, 2.0]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PreDataExecutionProtocolV67.seal(**payload)

    for field_name in (
        "model_text_executable",
        "protocol_is_scientific_evidence",
        "scientific_qualification_granted",
        "real_world_action_authorized",
    ):
        overstated = protocol.model_dump(mode="json")
        overstated.pop("protocol_hash")
        overstated[field_name] = True
        with pytest.raises(ValidationError):
            PreDataExecutionProtocolV67.seal(**overstated)
