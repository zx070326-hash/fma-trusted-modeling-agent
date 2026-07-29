from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from fma.v6.capability_catalog_v68 import (
    POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
    SCALAR_ODE_MANIFEST_ID_V68,
    capability_runtime_definition_v68,
    default_capability_runtime_definitions_v68,
)
from fma.v6.capability_runtime_v68 import (
    CapabilityRuntimeConformanceReportV68,
    CapabilityRuntimeDefinitionV68,
    run_capability_runtime_conformance_v68,
)


@pytest.fixture(scope="module")
def definitions() -> Mapping[str, CapabilityRuntimeDefinitionV68]:
    return default_capability_runtime_definitions_v68()


@pytest.fixture(scope="module")
def reports(
    definitions: Mapping[str, CapabilityRuntimeDefinitionV68],
) -> dict[str, CapabilityRuntimeConformanceReportV68]:
    return {
        key: run_capability_runtime_conformance_v68(definition)
        for key, definition in definitions.items()
    }


def test_runtime_definitions_are_direct_callable_and_exact_manifest_bound(
    definitions: Mapping[str, CapabilityRuntimeDefinitionV68],
) -> None:
    assert list(definitions) == [
        POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68,
        SCALAR_ODE_MANIFEST_ID_V68,
    ]
    for manifest_id, definition in definitions.items():
        definition.manifest.assert_sealed()
        assert definition.runtime_available is True
        assert callable(definition.compiler)
        assert callable(definition.executor)
        assert callable(definition.level_verifier)
        assert definition.dynamic_import_permitted is False
        assert definition.model_supplied_callable_permitted is False
        assert capability_runtime_definition_v68(
            manifest_id=manifest_id,
            manifest_hash=str(definition.manifest.manifest_hash),
        ).definition_hash == definition.definition_hash
        with pytest.raises(KeyError, match="manifest hash mismatch"):
            capability_runtime_definition_v68(
                manifest_id=manifest_id,
                manifest_hash="f" * 64,
            )


def test_positive_pack_passes_shared_executable_conformance(
    reports: Mapping[str, CapabilityRuntimeConformanceReportV68],
) -> None:
    report = reports[POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68]
    report.assert_sealed()

    assert report.status == "PASS"
    assert report.identity_conformance_status == "PASS"
    assert report.benchmark_coverage_status == "PASS"
    assert report.observed_public_case_count == 7
    assert report.observed_adversarial_case_count == 3
    assert report.expected_benchmark_case_ids == (
        report.observed_benchmark_case_ids
    )
    assert all(item.contract_status == "PASS" for item in report.receipts)

    by_id = {item.case_id: item for item in report.receipts}
    drift = by_id["drift_canonical"]
    assert drift.deterministic_execution_confirmed is True
    assert drift.first_execution_hash == drift.second_execution_hash
    assert drift.observed_level_statuses["L0"] == "NOT_RUN"

    fresh = by_id["fresh_replay"]
    assert fresh.observed_level_statuses == {
        "L0": "NOT_RUN",
        "L1": "PASS",
        "L2": "PASS",
        "L3": "PASS",
        "L4": "PASS",
    }

    no_signal = by_id["no_signal_abstain"]
    assert no_signal.observed_level_statuses["L3"] == "FAIL"
    assert no_signal.observed_level_statuses["L4"] == "FAIL"

    for case_id in (
        "nonpositive_rejected",
        "short_series_rejected",
        "typed_ir_tamper_rejected",
    ):
        receipt = by_id[case_id]
        assert receipt.observed_outcome == "REJECTED"
        assert receipt.observed_exception_type == "ValidationError"

    assert report.fixture_only is True
    assert report.report_is_scientific_evidence is False
    assert report.scientific_evidence_status == "NOT_RUN"
    assert report.maturity_promotion_granted is False
    assert report.scientific_qualification_granted is False
    assert report.real_world_action_authorized is False


def test_scalar_ode_uses_same_runtime_but_benchmark_remains_not_run(
    reports: Mapping[str, CapabilityRuntimeConformanceReportV68],
) -> None:
    report = reports[SCALAR_ODE_MANIFEST_ID_V68]
    report.assert_sealed()

    assert report.identity_conformance_status == "PASS"
    assert report.status == "NOT_RUN"
    assert report.benchmark_coverage_status == "NOT_RUN"
    assert report.expected_benchmark_case_ids == []
    assert report.observed_benchmark_case_ids == []
    assert all(item.contract_status == "PASS" for item in report.receipts)

    by_id = {item.case_id: item for item in report.receipts}
    canonical = by_id["scalar_ode_canonical_runtime"]
    assert canonical.observed_outcome == "EXECUTED"
    assert canonical.deterministic_execution_confirmed is True
    assert list(canonical.observed_level_statuses) == [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    ]
    assert (
        by_id["scalar_ode_typed_ir_tamper_rejected"].observed_outcome
        == "REJECTED"
    )
    assert report.report_is_scientific_evidence is False
    assert report.scientific_qualification_granted is False


def test_runtime_report_is_sealed_and_tamper_evident(
    reports: Mapping[str, CapabilityRuntimeConformanceReportV68],
) -> None:
    report = reports[POSITIVE_LOG_INCREMENT_MANIFEST_ID_V68]
    tampered = report.model_dump(mode="json")
    tampered["status"] = "FAIL"
    with pytest.raises(
        ValidationError,
        match="runtime conformance status differs",
    ):
        CapabilityRuntimeConformanceReportV68.model_validate(tampered)
