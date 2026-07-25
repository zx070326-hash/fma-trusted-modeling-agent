from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from fma.v3.evidence_concept_compiler_v313 import (
    DIMENSIONLESS_V313,
    RHS_DIMENSION_V313,
    STATE_DIMENSION_V313,
    ConceptPackageV313,
    DimensionV313,
    OperatorNodeV313,
    ParameterSpecV313,
    append_concept_experience_event_v313,
    compile_concept_package_v313,
    default_concept_evidence_v313,
    default_concept_packages_v313,
    empty_concept_experience_store_v313,
)


V312_REPORT_HASH = "73743c30318d75d17e2d2cd1c8b6451349a008df9bb1f6d7b2ae2e32b430f6a2"
NOW = datetime(2026, 7, 22, 6, 30, tzinfo=timezone.utc)


def _state() -> OperatorNodeV313:
    return OperatorNodeV313(kind="state", state_index=0)


def _parameter(parameter_id: str) -> OperatorNodeV313:
    return OperatorNodeV313(kind="parameter", parameter_id=parameter_id)


def _constant(value: float) -> OperatorNodeV313:
    return OperatorNodeV313(kind="constant", constant_value=value)


def _op(kind: str, *children: OperatorNodeV313) -> OperatorNodeV313:
    return OperatorNodeV313(kind=kind, children=list(children))


def _package(evidence, *, concept_id: str, rhs, parameters):
    return ConceptPackageV313.seal(
        concept_id=concept_id,
        evidence_hash=evidence.evidence_hash,
        supporting_claim_hashes=[evidence.claims[0].claim_hash],
        rhs=rhs,
        parameters=parameters,
        state_domain_lower=0.01,
        state_domain_upper=2.0,
    )


def test_v313_default_packages_compile_without_executable_source() -> None:
    evidence = default_concept_evidence_v313(
        v312_report_hash=V312_REPORT_HASH
    )
    packages = default_concept_packages_v313(evidence)
    compiled = [compile_concept_package_v313(item, evidence) for item in packages]
    assert len(evidence.sources) == 9
    assert len(evidence.claims) == 10
    assert {item.concept_id for item in packages} == {
        "log_capacity_growth",
        "generalized_capacity_growth",
        "hyperbolic_net_growth",
        "affine_rate_decoy",
    }
    assert all(
        item.static_checks_passed
        and item.numeric_checks_passed
        and item.rhs_dimension == RHS_DIMENSION_V313
        and not item.arbitrary_code_executed
        and not item.custom_operator_executed
        for item in compiled
    )
    assert all(
        not source.full_content_snapshot_available
        and not source.execution_permission
        for source in evidence.sources
    )
    monod_support = next(
        item for item in evidence.claims
        if item.claim_id == "claim_monod_hyperbolic_rate"
    )
    limitation = next(
        item for item in evidence.claims
        if item.claim_id == "claim_monod_instantaneous_limit"
    )
    assert monod_support.contradiction_claim_hashes == [limitation.claim_hash]


def test_v313_unit_checker_rejects_log_of_dimensional_state() -> None:
    evidence = default_concept_evidence_v313(
        v312_report_hash=V312_REPORT_HASH
    )
    package = _package(
        evidence,
        concept_id="bad_dimensional_log",
        rhs=_op(
            "multiply",
            _parameter("a"),
            _op("log", _state()),
        ),
        parameters=[
            ParameterSpecV313(
                parameter_id="a",
                dimension=RHS_DIMENSION_V313,
                lower_bound=0.1,
                upper_bound=2.0,
                initial_value=1.0,
            )
        ],
    )
    with pytest.raises(ValueError, match="log requires a dimensionless"):
        compile_concept_package_v313(package, evidence)


def test_v313_unit_checker_rejects_addition_and_power_mismatch() -> None:
    evidence = default_concept_evidence_v313(
        v312_report_hash=V312_REPORT_HASH
    )
    bad_add = _package(
        evidence,
        concept_id="bad_add_units",
        rhs=_op("add", _state(), _parameter("r")),
        parameters=[
            ParameterSpecV313(
                parameter_id="r",
                dimension=DimensionV313(time_power=-1),
                lower_bound=0.1,
                upper_bound=2.0,
                initial_value=1.0,
            )
        ],
    )
    with pytest.raises(ValueError, match="equal dimensions"):
        compile_concept_package_v313(bad_add, evidence)
    bad_power = _package(
        evidence,
        concept_id="bad_variable_power",
        rhs=_op(
            "multiply",
            _parameter("a"),
            _op("power", _state(), _parameter("nu")),
        ),
        parameters=[
            ParameterSpecV313(
                parameter_id="a",
                dimension=RHS_DIMENSION_V313,
                lower_bound=0.1,
                upper_bound=2.0,
                initial_value=1.0,
            ),
            ParameterSpecV313(
                parameter_id="nu",
                dimension=DIMENSIONLESS_V313,
                lower_bound=0.2,
                upper_bound=2.0,
                initial_value=1.0,
            ),
        ],
    )
    with pytest.raises(ValueError, match="dimensionless base"):
        compile_concept_package_v313(bad_power, evidence)


def test_v313_numeric_canary_rejects_zero_divisor() -> None:
    evidence = default_concept_evidence_v313(
        v312_report_hash=V312_REPORT_HASH
    )
    package = _package(
        evidence,
        concept_id="bad_zero_divisor",
        rhs=_op("divide", _parameter("a"), _parameter("d")),
        parameters=[
            ParameterSpecV313(
                parameter_id="a",
                dimension=RHS_DIMENSION_V313,
                lower_bound=0.1,
                upper_bound=2.0,
                initial_value=1.0,
            ),
            ParameterSpecV313(
                parameter_id="d",
                dimension=DIMENSIONLESS_V313,
                lower_bound=0.0,
                upper_bound=1.0,
                initial_value=0.5,
            ),
        ],
    )
    with pytest.raises(ValueError, match="numeric domain canary failed"):
        compile_concept_package_v313(package, evidence)


def test_v313_experience_store_requires_private_adjudication_and_revokes() -> None:
    evidence = default_concept_evidence_v313(
        v312_report_hash=V312_REPORT_HASH
    )
    package = default_concept_packages_v313(evidence)[0]
    compiled = compile_concept_package_v313(package, evidence)
    store = empty_concept_experience_store_v313(evidence, created_at=NOW)
    store = append_concept_experience_event_v313(
        store,
        event_type="proposed",
        package=package,
        compiled=None,
        phase="research",
        created_at=NOW + timedelta(seconds=1),
    )
    store = append_concept_experience_event_v313(
        store,
        event_type="compiled",
        package=package,
        compiled=compiled,
        phase="research",
        created_at=NOW + timedelta(seconds=2),
    )
    with pytest.raises(ValueError, match="needs adjudication"):
        append_concept_experience_event_v313(
            store,
            event_type="privately_admitted",
            package=package,
            compiled=compiled,
            phase="confirmation",
            created_at=NOW + timedelta(seconds=3),
        )
    store = append_concept_experience_event_v313(
        store,
        event_type="privately_admitted",
        package=package,
        compiled=compiled,
        phase="confirmation",
        created_at=NOW + timedelta(seconds=3),
        adjudication_hash="a" * 64,
        private_evaluator_event=True,
    )
    assert store.active_concept_versions == {"log_capacity_growth": 1}
    store = append_concept_experience_event_v313(
        store,
        event_type="revoked",
        package=package,
        compiled=compiled,
        phase="post_confirmation",
        created_at=NOW + timedelta(seconds=4),
        adjudication_hash="b" * 64,
        private_evaluator_event=True,
    )
    assert store.active_concept_versions == {}
    assert len(store.events) == 4
    store.assert_sealed()


def test_v313_schema_rejects_source_code_and_public_admission_flag() -> None:
    evidence = default_concept_evidence_v313(
        v312_report_hash=V312_REPORT_HASH
    )
    package = default_concept_packages_v313(evidence)[0]
    payload = package.model_dump(exclude={"package_hash"})
    payload["arbitrary_code_present"] = True
    with pytest.raises(ValidationError):
        ConceptPackageV313.seal(**payload)
    compiled = compile_concept_package_v313(package, evidence)
    store = empty_concept_experience_store_v313(evidence, created_at=NOW)
    with pytest.raises(ValidationError):
        # The Literal[False] field is code-owned and cannot be overridden.
        from fma.v3.evidence_concept_compiler_v313 import ConceptExperienceEventV313

        ConceptExperienceEventV313.seal(
            event_id="bad_public_admission",
            sequence=1,
            previous_event_hash=None,
            event_type="privately_admitted",
            concept_id=package.concept_id,
            package_hash=package.package_hash,
            compiled_hash=compiled.compiled_hash,
            adjudication_hash="c" * 64,
            phase="confirmation",
            private_evaluator_event=True,
            public_score_used_for_admission=True,
            created_at=NOW,
        )
    assert store.events == []
