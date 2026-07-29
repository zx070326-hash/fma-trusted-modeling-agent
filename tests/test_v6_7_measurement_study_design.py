from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fma.v6.measurement_study_design import (
    ApplicabilityBoundaryV67,
    BiasPlanV67,
    ConfoundingPlanV67,
    ConstructDefinitionV67,
    EthicsBoundaryV67,
    MeasurementDefinitionV67,
    MeasurementErrorPlanV67,
    MeasurementStudyDesignContractV67,
    MissingnessPlanV67,
    PopulationDefinitionV67,
    SamplingPlanV67,
    StudyDesignV67,
)


def _contract_data() -> dict[str, object]:
    return {
        "contract_id": "brazil-gdp-per-capita-measurement",
        "workspace_spec_hash": "a" * 64,
        "s0_gate_hash": "b" * 64,
        "source_contract_id": "world-bank-brazil-gdp-per-capita",
        "source_contract_hash": "c" * 64,
        "claim_kind": "predictive",
        "claim_scope": (
            "Predict the next registered annual country-level indicator value "
            "without causal or policy interpretation."
        ),
        "construct_definition": ConstructDefinitionV67(
            construct_id="real_income_per_person",
            name="real income per person",
            conceptual_definition=(
                "Inflation-adjusted economic output allocated per resident "
                "within the declared national accounting boundary."
            ),
            role="outcome",
            representation="proxy",
            representation_rationale=(
                "The registered national-accounts indicator is a proxy for "
                "the broader material-income construct."
            ),
        ),
        "measurement": MeasurementDefinitionV67(
            measurement_id="wb_real_gdp_per_capita",
            construct_id="real_income_per_person",
            operational_definition=(
                "Annual GDP per capita reported in constant 2015 US dollars "
                "under the predeclared official indicator definition."
            ),
            unit="constant_2015_USD_per_person",
            time_basis="calendar year",
            aggregation_basis="country annual aggregate divided by population",
            scale_type="ratio",
            source_definition=(
                "The exact source identifier and transport contract will be "
                "bound later without changing this operational definition."
            ),
            directionality="higher_is_more",
        ),
        "population": PopulationDefinitionV67(
            population_id="brazil_calendar_years",
            target_population=(
                "Calendar years for Brazil within the predeclared modelling "
                "period and the same national accounting regime."
            ),
            unit_of_analysis="country-year",
            spatial_scope="Brazil national territory",
            temporal_scope="predeclared historical and forecast years",
            inclusion_criteria=[
                "Official annual indicator under the registered definition",
                "Year lies within the predeclared source interval",
            ],
            exclusion_criteria=[
                "Duplicate country-year record",
                "Observation outside the registered indicator definition",
            ],
        ),
        "sampling": SamplingPlanV67(
            sampling_frame=(
                "All annual country records exposed by the exact registered "
                "official indicator and year range."
            ),
            sampling_method="administrative_complete_series",
            selection_rule=(
                "Select every eligible year in the frozen interval without "
                "post-result deletion or cherry-picking."
            ),
            minimum_sample_size=40,
            stopping_rule=(
                "Use the frozen end year; do not extend the series after "
                "inspecting model or verifier results."
            ),
            representativeness_limitations=(
                "Administrative annual records do not represent households "
                "or within-country income distributions."
            ),
        ),
        "missingness": MissingnessPlanV67(
            anticipated_sources=[
                "Official series may omit an eligible country-year",
                "A source revision may remove an earlier value",
            ],
            mechanism_assumptions=[
                "Missingness may depend on source production processes",
                "Missingness is not assumed random without post-data evidence",
            ],
            handling_policy="reject_incomplete_series",
            sensitivity_analysis_plan=(
                "Fail the narrow production adapter and require a new frozen "
                "plan before any imputation or alternative handling."
            ),
        ),
        "measurement_error": MeasurementErrorPlanV67(
            anticipated_error_sources=[
                "National-account revisions",
                "Population denominator revisions",
                "Price-deflator methodology changes",
            ],
            error_structure_assumption=(
                "The size, direction, and temporal dependence of revision "
                "error are unknown before observing external revision vintages."
            ),
            calibration_or_reference_plan=(
                "Bind indicator metadata and compare independent vintages only "
                "when an authenticated calibration adapter is available."
            ),
            propagation_or_sensitivity_plan=(
                "Treat uncalibrated measurement error as a claim limitation "
                "and run declared perturbation checks after data binding."
            ),
        ),
        "bias": BiasPlanV67(
            anticipated_biases=[
                "Survivorship of published historical values",
                "Post-selection of the modelling interval",
                "National aggregate masking distributional changes",
            ],
            mitigation_plan=(
                "Freeze source, interval, exclusions, and validation design "
                "before access and preserve all gate failures."
            ),
            residual_bias_policy=(
                "Report unresolved bias explicitly and prevent elevation to "
                "a population, causal, policy, or welfare claim."
            ),
        ),
        "confounding": ConfoundingPlanV67(
            relevance="not_applicable_to_noncausal_claim",
            identification_or_control_strategy=(
                "No causal effect is identified or controlled in this "
                "predictive-only task."
            ),
            unmeasured_confounding_policy=(
                "Unexpected predictive associations cannot be interpreted as "
                "mechanisms or intervention effects."
            ),
        ),
        "study_design": StudyDesignV67(
            design_type="time_series",
            target_quantity=(
                "A next-step predictive distribution for the registered "
                "annual indicator within the declared applicability boundary."
            ),
            temporal_ordering=(
                "Training observations precede each held-out forecast origin."
            ),
            comparison_strategy=(
                "Compare registered candidates against predeclared naive "
                "baselines at identical rolling origins."
            ),
            validation_design=(
                "Use leakage-safe nested rolling-origin evaluation with all "
                "thresholds frozen before source bytes are bound."
            ),
            leakage_prevention_plan=(
                "Fit and select only on each training prefix; do not expose "
                "future folds or private acceptance outcomes to generators."
            ),
        ),
        "applicability": ApplicabilityBoundaryV67(
            intended_use=(
                "Local retrospective evaluation and bounded next-step "
                "forecasting of the same registered annual indicator."
            ),
            in_scope_conditions=[
                "Same indicator definition and unit",
                "Same country aggregation and annual time basis",
            ],
            out_of_scope_conditions=[
                "Causal attribution",
                "Policy recommendation",
                "Household-level inference",
            ],
            transport_assumptions=[
                "Source metadata remain definitionally compatible",
                "Forecast regime is represented by the frozen history",
            ],
            abstention_conditions=[
                "Unit or indicator definition changes",
                "Required history or rolling folds are unavailable",
                "Registered validation gates fail",
            ],
        ),
        "ethics": EthicsBoundaryV67(
            risk_level="minimal",
            human_participant_data_expected=False,
            sensitive_data_expected=False,
            consent_or_legal_basis_plan=(
                "Use only the declared aggregate public indicator under its "
                "recorded source and attribution terms."
            ),
            prohibited_uses=[
                "Individual eligibility or benefit decision",
                "Automated policy action",
                "Causal welfare claim",
            ],
            ethics_review_required=False,
        ),
    }


def _sealed() -> MeasurementStudyDesignContractV67:
    return MeasurementStudyDesignContractV67.seal(**_contract_data())


def test_contract_is_strict_observation_free_and_claim_limited() -> None:
    contract = _sealed()

    contract.assert_sealed()
    assert contract.contract_hash == contract.content_hash()
    assert contract.source_contract_id == "world-bank-brazil-gdp-per-capita"
    assert contract.source_contract_hash == "c" * 64
    assert contract.construct_review_status == "HUMAN"
    assert contract.measurement_review_status == "HUMAN"
    assert contract.study_design_review_status == "HUMAN"
    assert contract.ethics_review_status == "NOT_RUN"
    assert contract.independent_review_status == "NOT_RUN"
    assert contract.data_access_status == "NOT_RUN"
    assert contract.observation_data_included is False
    assert contract.observed_statistics_included is False
    assert contract.private_acceptance_data_accessed is False
    assert contract.scientific_qualification_granted is False
    assert contract.real_world_action_authorized is False

    payload = contract.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True)
    assert '"observations"' not in encoded
    assert '"observed_values"' not in encoded
    assert '"summary_statistics"' not in encoded


def test_contract_round_trip_preserves_exact_seal() -> None:
    contract = _sealed()

    replay = MeasurementStudyDesignContractV67.model_validate_json(
        contract.model_dump_json()
    )

    replay.assert_sealed()
    assert replay == contract


def test_contract_rejects_observations_and_overstated_authority() -> None:
    data = _contract_data()
    data["observations"] = [1.0, 2.0]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MeasurementStudyDesignContractV67.seal(**data)

    for field_name, value in (
        ("construct_review_status", "PASS"),
        ("ethics_review_status", "PASS"),
        ("independent_review_status", "PASS"),
        ("data_access_status", "PASS"),
        ("observation_data_included", True),
        ("scientific_qualification_granted", True),
        ("real_world_action_authorized", True),
    ):
        overstated = _contract_data()
        overstated[field_name] = value
        with pytest.raises(ValidationError):
            MeasurementStudyDesignContractV67.seal(**overstated)


def test_seal_detects_nested_tampering() -> None:
    contract = _sealed()
    contract.measurement.unit = "current_USD_per_person"

    with pytest.raises(ValueError, match="not sealed"):
        contract.assert_sealed()

    with pytest.raises(ValidationError, match="contract hash differs"):
        MeasurementStudyDesignContractV67.model_validate(
            contract.model_dump(mode="json")
        )


def test_measurement_must_reference_declared_construct() -> None:
    data = _contract_data()
    measurement = data["measurement"]
    assert isinstance(measurement, MeasurementDefinitionV67)
    data["measurement"] = measurement.model_copy(
        update={"construct_id": "another_construct"}
    )

    with pytest.raises(ValidationError, match="another construct"):
        MeasurementStudyDesignContractV67.seal(**data)


def test_confounding_plan_is_claim_relative_and_never_authorizes_causality() -> None:
    predictive = _contract_data()
    predictive["confounding"] = ConfoundingPlanV67(
        relevance="required_for_claim",
        candidate_confounders=["Fiscal policy"],
        identification_or_control_strategy=(
            "A predeclared strategy would be required for a causal claim."
        ),
        unmeasured_confounding_policy=(
            "Residual confounding prevents causal interpretation."
        ),
    )
    with pytest.raises(ValidationError, match="differs from claim kind"):
        MeasurementStudyDesignContractV67.seal(**predictive)

    mechanistic = _contract_data()
    mechanistic["claim_kind"] = "mechanistic"
    with pytest.raises(ValidationError, match="differs from claim kind"):
        MeasurementStudyDesignContractV67.seal(**mechanistic)

    mechanistic["confounding"] = ConfoundingPlanV67(
        relevance="required_for_claim",
        candidate_confounders=["Fiscal policy", "Global commodity prices"],
        identification_or_control_strategy=(
            "Freeze an identification strategy before accessing observations."
        ),
        unmeasured_confounding_policy=(
            "Abstain from mechanism claims if residual confounding is not "
            "bounded by independent evidence."
        ),
    )
    contract = MeasurementStudyDesignContractV67.seal(**mechanistic)
    assert contract.confounding.causal_interpretation_authorized is False
    assert contract.scientific_qualification_granted is False


def test_nonminimal_or_sensitive_design_requires_ethics_review() -> None:
    with pytest.raises(ValidationError, match="requires ethics review"):
        EthicsBoundaryV67(
            risk_level="elevated",
            human_participant_data_expected=False,
            sensitive_data_expected=False,
            consent_or_legal_basis_plan=(
                "A legal basis would be established before data access."
            ),
            prohibited_uses=["Automated adverse action"],
            ethics_review_required=False,
        )

    data = _contract_data()
    data["ethics"] = EthicsBoundaryV67(
        risk_level="elevated",
        human_participant_data_expected=False,
        sensitive_data_expected=False,
        consent_or_legal_basis_plan=(
            "A legal basis would be established before data access."
        ),
        prohibited_uses=["Automated adverse action"],
        ethics_review_required=True,
    )
    with pytest.raises(ValidationError, match="ethics review status differs"):
        MeasurementStudyDesignContractV67.seal(**data)
    data["ethics_review_status"] = "HUMAN"
    contract = MeasurementStudyDesignContractV67.seal(**data)
    assert contract.ethics_review_status == "HUMAN"


def test_lists_with_duplicate_design_claims_are_rejected() -> None:
    data = _contract_data()
    bias = data["bias"]
    assert isinstance(bias, BiasPlanV67)
    data["bias"] = bias.model_copy(
        update={
            "anticipated_biases": [
                "Post-selection of the modelling interval",
                "Post-selection of the modelling interval",
            ]
        }
    )

    with pytest.raises(ValidationError, match="must be unique"):
        MeasurementStudyDesignContractV67.seal(**data)
