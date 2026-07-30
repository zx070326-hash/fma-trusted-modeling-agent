from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from fma.studio import service as studio_service_module
from fma.hashing import canonical_json, sha256_value
from fma.storage import RunStore
from fma.v5.workspace_schemas import RegimeDiagnosisV50
from fma.v5_1.codex_stage_driver import FixtureStageRoleTransportV51
from fma.v6.decision_value import (
    DECISION_CONTRACT_PATH,
    DECISION_EVIDENCE_PATH,
    DECISION_INTENT_PATH,
    DecisionValueEvidenceV62,
)
from fma.v6.executable_candidate import (
    EXECUTABLE_CANDIDATE_INTENT_PATH,
    EXECUTABLE_CANDIDATE_IR_PATH,
    EXECUTABLE_CANDIDATE_RECEIPT_PATH,
    EXECUTABLE_CANDIDATE_RESOLUTION_PATH,
    ExecutableCandidateReceiptV62,
)
from fma.v6.provenance import PROVENANCE_BINDING_PATH
from fma.v6.predata_transaction import PREDATA_CONTRACT_PATHS_V67
from fma.v6.public_source import SOURCE_RAW_PATH, SourceHTTPResponseV62
from fma.v6.scientific_closure import SCIENTIFIC_CLOSURE_ROOT
from fma.v6.scientific_success import ROLLING_CONFIRMATION_PATH
from fma.v6.source_auth import (
    S2_SOURCE_REVERIFICATION_PATH,
    SOURCE_ACQUISITION_AUTH_PATH,
)
from fma.v6.stage_gate_outcome import (
    StageGateOutcomeV66,
    record_blocked_stage_gate_v66,
    verify_stage_gate_outcome_v66,
)
from fma.studio.__main__ import _parser as studio_cli_parser
from fma.studio.server import StudioHTTPServer
from fma.studio.service import (
    StudioConflictError,
    StudioTaskService,
    StudioValidationError,
)


AUTHORITY_KEY = b"studio-test-authority-key-" + b"k" * 32
BRIDGE_TOKEN = "studio-test-bridge-token-123456"
OBJECTIVE = (
    "Forecast weekly emergency visits for twelve weeks so staffing can be "
    "planned, while treating understaffing as the larger error."
)


def test_studio_cli_accepts_explicit_model_pin() -> None:
    args = studio_cli_parser().parse_args(
        [
            "--task-root",
            "tasks",
            "--authority-key-file",
            "authority.key",
            "--model",
            "gpt-5.6",
            "--codex-runtime",
            "wsl",
            "--expected-codex-cli-version",
            "0.145.0-alpha.18",
            "--wsl-distribution",
            "Ubuntu",
        ]
    )

    assert args.model == "gpt-5.6"
    assert args.codex_runtime == "wsl"
    assert args.expected_codex_cli_version == "0.145.0-alpha.18"
    assert args.wsl_distribution == "Ubuntu"


def _valid_draft(request):
    if request.role_kind == "reviewer":
        return {
            "schema_version": "5.1",
            "request_hash": request.request_hash,
            "role_name": request.role_name,
            "selected_candidate_id": None,
            "verdict": "APPROVE",
            "rationale": "The S0 task is bounded, computable, and honest.",
            "assumptions": [],
            "findings": [],
            "uncertainties": ["No empirical data has been ingested yet."],
            "proposed_artifacts": [],
            "authority_claimed": False,
        }
    profile_hash = request.public_inputs["frozen_evaluation_profile"]["profile_hash"]
    decision = {
        "schema_version": "6.6-s0-decision-draft",
        "function_id": "asymmetric_staffing_loss",
        "input_names": ["prediction", "target"],
        "expression": ("2 * max(target - prediction, 0) + max(prediction - target, 0)"),
        "sense": "minimize",
        "output_unit": "visit_count",
        "canaries": [
            {
                "canary_id": "exact",
                "input_values": [10.0, 10.0],
                "expected": 0.0,
            },
            {
                "canary_id": "under",
                "input_values": [9.0, 10.0],
                "expected": 2.0,
            },
        ],
    }
    regime = {
        "schema_version": "6.6-s0-regime-draft",
        "system_boundary": (
            "One emergency department and its weekly aggregate visit demand."
        ),
        "state_and_memory": (
            "Observed weekly visit counts with trend, seasonality, and lagged state."
        ),
        "uncertainty_and_data": (
            "No data has been ingested; sampling, drift, and missingness remain open."
        ),
        "decision_and_loss": (
            "A report-only forecast scored by asymmetric staffing loss."
        ),
        "query_type": "prediction",
        "downstream_decision": "Prepare a human-reviewed staffing draft.",
        "decision_function_id": "asymmetric_staffing_loss",
        "computable_decision_function": (
            "Use the specified asymmetric absolute staffing loss."
        ),
        "evidence_hashes": sorted(
            [
                request.public_inputs["evidence_snapshot_hash"],
                profile_hash,
            ]
        ),
        "limitations": [
            "No forecast is usable until data provenance and validation pass."
        ],
        "evaluation_profile_hash": profile_hash,
    }
    return {
        "schema_version": "5.1",
        "request_hash": request.request_hash,
        "role_name": request.role_name,
        "selected_candidate_id": None,
        "verdict": "NOT_APPLICABLE",
        "rationale": "Two typed S0 artifacts are proposed for harness validation.",
        "assumptions": ["Weekly aggregation is meaningful."],
        "findings": [],
        "uncertainties": ["No empirical data has been ingested yet."],
        "proposed_artifacts": [
            {
                "artifact_type": "decision_function",
                "content": canonical_json(decision),
            },
            {
                "artifact_type": "regime_diagnosis",
                "content": canonical_json(regime),
            },
        ],
        "authority_claimed": False,
    }


def _role_draft(
    request,
    *,
    artifacts: list[dict[str, str]],
    selected_candidate_id: str | None = None,
    verdict: str = "NOT_APPLICABLE",
    rationale: str = "Typed S1 evidence is proposed for harness validation.",
):
    return {
        "schema_version": "5.1",
        "request_hash": request.request_hash,
        "role_name": request.role_name,
        "selected_candidate_id": selected_candidate_id,
        "verdict": verdict,
        "rationale": rationale,
        "assumptions": [],
        "findings": [],
        "uncertainties": ["No S2 data has been frozen."],
        "proposed_artifacts": artifacts,
        "authority_claimed": False,
    }


def _s1_draft(request):
    if request.stage == "S0":
        return _valid_draft(request)
    if request.role_kind == "reviewer":
        return _role_draft(
            request,
            artifacts=[],
            verdict="APPROVE",
            rationale=(
                "The candidate frontier, epistemic exchange, and validation "
                "duties are bounded and do not claim scientific acceptance."
            ),
        )
    if request.role_name == "s2_data_steward":
        required_ids = request.public_inputs["required_data_requirement_ids"]
        mapping = {
            "schema_version": "5.9",
            "data_requirement_ids": required_ids,
            "semantic_name": "positive scalar state observations over time",
            "units": request.public_inputs["data_summary"]["state_unit"],
            "transform_rule": (
                "Preserve the frozen time and observation arrays byte-for-byte "
                "inside the registered scalar ODE snapshot."
            ),
            "quality_flags": ["fixture_role_did_not_assess_source_quality"],
        }
        return _role_draft(
            request,
            selected_candidate_id=request.allowed_candidate_ids[0],
            artifacts=[
                {
                    "artifact_type": "data_mapping",
                    "content": canonical_json(mapping),
                }
            ],
        )
    if request.role_name == "s5_decision_writer":
        narrative = {
            "schema_version": "5.9",
            "statement": (
                "The registered scalar ODE ensemble supports only the frozen "
                "reporting target and remains subject to the bound uncertainty."
            ),
            "limitations": [
                "Fixture evidence does not establish external scientific validity.",
                "No real-world action is authorized.",
            ],
        }
        return _role_draft(
            request,
            selected_candidate_id=request.allowed_candidate_ids[0],
            artifacts=[
                {
                    "artifact_type": "decision_narrative",
                    "content": canonical_json(narrative),
                }
            ],
        )
    if request.role_name == "s1_prior_model_scout":
        report = {
            "scope": "supplied_public_inputs_only",
            "candidate_family_hints": [
                "mechanistic state space",
                "null persistence baseline",
                "regularized autoregression",
                "system identification",
            ],
            "source_claims_verified": False,
            "limitations": [
                "No live literature search or source verification was performed."
            ],
        }
        return _role_draft(
            request,
            artifacts=[
                {
                    "artifact_type": "literature_map",
                    "content": canonical_json(report),
                }
            ],
        )
    if request.role_name.startswith("s1_cross_paradigm_translator_"):
        packet = request.public_inputs["disclosure_packet"]
        target = packet["recipient_branch_id"]
        unit_id_by_hash = {
            item["unit_hash"]: item["unit_id"]
            for item in request.public_inputs["disclosed_units"]
        }
        source_id = unit_id_by_hash[packet["disclosed_unit_hashes"][0]]
        transfers = [
            {
                "transfer_id": f"transfer.{target}",
                "source_unit_ids": [source_id],
                "target_interpretation": (
                    f"Use peer falsification logic as a stress test for {target}."
                ),
                "proposed_modification": (
                    f"Add one nested diagnostic to the {target} candidate."
                ),
                "falsification_test": (
                    "Reject the modification if frozen-development loss "
                    "does not improve over the unmodified branch."
                ),
            }
        ]
        return _role_draft(
            request,
            artifacts=[
                {
                    "artifact_type": "transfer_hypotheses",
                    "content": canonical_json(transfers),
                }
            ],
        )
    if request.role_name.endswith("_recipient"):
        assessments = [
            {
                "transfer_id": item["transfer_id"],
                "verdict": "ACCEPT_FOR_TEST",
                "rationale": (
                    "The translation is coherent enough to test but has no "
                    "scientific support before S2-S4."
                ),
                "required_test": (
                    "Compare nested candidates on frozen development data and "
                    "preserve the negative result."
                ),
            }
            for item in request.public_inputs["transfer_hypotheses"]
        ]
        return _role_draft(
            request,
            artifacts=[
                {
                    "artifact_type": "transfer_assessments",
                    "content": canonical_json(assessments),
                }
            ],
        )
    if request.role_name in {
        "s1_candidate_synthesizer",
        "s1_candidate_synthesizer_repair",
    }:
        selected = "candidate.mechanistic"
        selected_candidate_structure = {
            "candidate_id": selected,
            "model_family": "mechanistic state-space model",
            "data_requirement_ids": ["mechanistic.observations"],
            "abandon_criteria": ["Abandon if it cannot beat the frozen null baseline."],
            "lineage": (
                "Graph-guided refinement of candidate.mechanistic with a fixed "
                "state update and frozen estimation window."
            ),
        }
        selected_mathematical_form = {
            "candidate_id": selected,
            "mathematical_form": (
                "For fixed hourly predictors u[t], x[t+1] = x[t] + "
                "gain * (u[t] - x[t]); estimate gain on the frozen training "
                "window and propagate exactly twelve decision horizons."
            ),
        }
        executable_candidate_intent = {
            "schema_version": "6.2-registered-family-search-intent",
            "candidate_id": selected,
            "operation": "registered_family_search",
            "input_domain": "positive_scalar_time_series",
            "allowed_adapter_ids": [
                "scalar_autonomous_ode_v52",
                "adaptive_positive_series_v57",
            ],
            "adapter_resolution_stage": "S2",
            "model_family_text_executable": False,
            "mathematical_form_text_executable": False,
            "arbitrary_code_execution_permitted": False,
        }
        selection = {
            "selected_candidate_id": selected,
            "selection_rationale": (
                "The mechanistic branch is selected for development because it "
                "is falsifiable and decision-linked; alternatives are retained."
            ),
            "declared_conservation_laws": [],
            "declared_limit_cases": [
                "zero recent demand gives the baseline state",
                "zero process gain removes the dynamic adjustment",
            ],
            "identifiability_risks": [
                "Short aggregate series may confound trend and process gain."
            ],
        }
        validation_rules = {
            "rules": [
                {
                    "check_id": "s3_l0_replay",
                    "applicability_rule": (
                        "PASS only when two executions from identical frozen "
                        "inputs produce identical canonical result hashes."
                    ),
                },
                {
                    "check_id": "s3_l1_structural",
                    "applicability_rule": (
                        "PASS only with zero executable unit, bound, domain, "
                        "probability-range, and declared-invariant violations."
                    ),
                },
                {
                    "check_id": "s3_l2_numerical",
                    "applicability_rule": (
                        "PASS only when all declared limits hold and estimates "
                        "remain within a pre-frozen tolerance under refinement."
                    ),
                },
                {
                    "check_id": "s4_l3_holdout",
                    "applicability_rule": (
                        "PASS only when the upper 95 percent block-bootstrap "
                        "bound for loss difference versus the frozen baseline is below zero."
                    ),
                },
                {
                    "check_id": "s4_l4_uncertainty",
                    "applicability_rule": (
                        "PASS only when pre-frozen missingness, parameter, and "
                        "regime stresses do not cause a material decision reversal."
                    ),
                },
            ]
        }
        return _role_draft(
            request,
            selected_candidate_id=selected,
            artifacts=[
                {
                    "artifact_type": "selected_candidate_structure",
                    "content": canonical_json(selected_candidate_structure),
                },
                {
                    "artifact_type": "selected_mathematical_form",
                    "content": canonical_json(selected_mathematical_form),
                },
                {
                    "artifact_type": "executable_candidate_intent",
                    "content": canonical_json(executable_candidate_intent),
                },
                {
                    "artifact_type": "selection",
                    "content": canonical_json(selection),
                },
                *[
                    {
                        "artifact_type": f"validation_rule_{rule['check_id']}",
                        "content": canonical_json(rule),
                    }
                    for rule in validation_rules["rules"]
                ],
            ],
        )

    branch = request.public_inputs["branch_id"]
    candidate_id = f"candidate.{branch}"
    assumption_id = f"{branch}.aggregation"
    forms = {
        "mechanistic": "x[t+1] = x[t] + gain * (u[t] - x[t])",
        "null_baseline": "x[t+1] = x[t]",
        "statistical": "x[t+1] = beta0 + gain * x[t] + epsilon[t]",
        "system_learning": "x[t+1] = f_theta(x[t], u[t])",
    }
    candidate = {
        "candidate_id": candidate_id,
        "model_family": branch.replace("_", " "),
        "mathematical_form": forms[branch],
        "data_requirement_ids": [f"{branch}.observations"],
        "abandon_criteria": ["Abandon if it cannot beat the frozen null baseline."],
        "lineage": f"blind independent {branch} branch",
    }
    assumptions = [
        {
            "assumption_id": assumption_id,
            "statement": "Weekly aggregation preserves decision-relevant variation.",
            "failure_consequence": "The candidate cannot support staffing decisions.",
            "falsification_test": "Compare aggregation levels on frozen data.",
            "abandon_criterion": "Abandon when aggregation reverses the decision.",
        }
    ]
    symbols = [
        {
            "symbol_id": f"{branch}.gain",
            "meaning": "branch response gain",
            "unit": "dimensionless",
            "role": "parameter",
            "lower_bound": 0.0,
            "upper_bound": 2.0,
        },
        {
            "symbol_id": f"{branch}.state",
            "meaning": "weekly emergency visit state",
            "unit": "visits",
            "role": "state",
            "lower_bound": 0.0,
            "upper_bound": None,
        },
    ]
    knowledge = [
        {
            "unit_id": f"{branch}.failure-test",
            "kind": "constraint",
            "statement": (
                f"The {branch} branch must beat persistence under asymmetric loss."
            ),
            "applicability_conditions": [
                "Weekly aggregate visit observations are available."
            ],
            "falsification_test": (
                "Reject if development loss is not lower than persistence."
            ),
            "utility_hint": 0.8,
        }
    ]
    return _role_draft(
        request,
        selected_candidate_id=candidate_id,
        artifacts=[
            {"artifact_type": "candidate", "content": canonical_json(candidate)},
            {
                "artifact_type": "assumptions",
                "content": canonical_json(assumptions),
            },
            {"artifact_type": "symbols", "content": canonical_json(symbols)},
            {
                "artifact_type": "knowledge_units",
                "content": canonical_json(knowledge),
            },
        ],
    )


def _ode_backhalf_draft(request):
    payload = _s1_draft(request)
    if request.role_name in {
        "s1_candidate_synthesizer",
        "s1_candidate_synthesizer_repair",
    }:
        for artifact in payload["proposed_artifacts"]:
            if artifact["artifact_type"] == "selected_candidate_structure":
                structure = json.loads(artifact["content"])
                structure["model_family"] = (
                    "registered scalar autonomous ODE model-selection family"
                )
                structure["data_requirement_ids"] = ["ode.scalar_series"]
                artifact["content"] = canonical_json(structure)
            elif artifact["artifact_type"] == "selected_mathematical_form":
                mathematical_form = json.loads(artifact["content"])
                mathematical_form["mathematical_form"] = (
                    "Select by frozen development loss among dx/dt = 0, "
                    "dx/dt = r*x, dx/dt = r*x*log(K/x), and "
                    "dx/dt = r*x*(1-x/K); fit only on the frozen series and "
                    "evaluate all L0-L4 obligations without post-result tuning."
                )
                artifact["content"] = canonical_json(mathematical_form)
    return payload


def _service(
    tmp_path: Path,
    draft_factory=_valid_draft,
    *,
    world_bank_fetcher=None,
) -> StudioTaskService:
    return StudioTaskService(
        tmp_path / "tasks",
        authority_key=AUTHORITY_KEY,
        authority_key_id="studio-test-v1",
        role_transport_factory=lambda _: FixtureStageRoleTransportV51(draft_factory),
        world_bank_fetcher=world_bank_fetcher,
    )


def _inject_s0_projection_sentence_defect(
    service: StudioTaskService,
    monkeypatch: pytest.MonkeyPatch,
    *,
    every_attempt: bool = False,
) -> None:
    """Fault-inject a schema-valid historical/projection defect after parse."""

    original = service._materialize_s0

    def materialize_with_defect(workspace, outcome):
        decision, regime, profile = original(workspace, outcome)
        if every_attempt or workspace._latest_attempt("S0") == 1:
            payload = dict(regime)
            payload["system_boundary"] = (
                "This materialized projection sentence is incomplete"
            )
            payload.pop("diagnosis_hash", None)
            defective = RegimeDiagnosisV50.seal(**payload)
            defective_payload = defective.model_dump(mode="json")
            (workspace.root / "docs" / "regime.json").write_text(
                canonical_json(defective_payload),
                encoding="utf-8",
            )
            return decision, defective_payload, profile
        return decision, regime, profile

    monkeypatch.setattr(
        service,
        "_materialize_s0",
        materialize_with_defect,
    )


def test_create_task_is_idempotent_and_starts_at_s0(tmp_path: Path) -> None:
    service = _service(tmp_path)
    first = service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "emergency-visits",
            "evidence_scope": "development",
        }
    )
    second = service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "emergency-visits",
            "evidence_scope": "development",
        }
    )

    assert first["task_id"] == second["task_id"] == "emergency-visits"
    assert first["workflow"]["frontier_stages"] == ["S0"]
    assert first["workflow"]["stage_statuses"]["S0"] == "frontier"
    assert first["scientific_success"]["evaluated"] is False
    assert first["scientific_success"]["scientific_success_status"] == ("NOT_RUN")
    assert first["scientific_qualification_granted"] is False
    assert first["real_world_action_authorized"] is False
    assert first["events"][0]["event_type"] == "task_created"


def test_s0_runs_generator_reviewer_check_and_gate(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "emergency-visits"})

    result = service.run_s0("emergency-visits")

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert result["workflow"]["frontier_stages"] == ["S1"]
    assert result["next_valid_actions"] == [
        "inspect_s0",
        "run_s1",
        "prepare_portfolio_v69",
    ]
    assert result["scientific_qualification_granted"] is False
    assert result["real_world_action_authorized"] is False
    assert result["events"][-1]["event_type"] == "s0_gate_evaluated"
    assert result["events"][-1]["details"]["decision"] == "OPEN"
    root = tmp_path / "tasks" / "emergency-visits"
    contract = json.loads(
        (root / "problem" / "contract.json").read_text(encoding="utf-8")
    )
    assert contract["question"] == OBJECTIVE
    assert (root / "problem" / "decision_function.json").is_file()
    assert (root / "docs" / "regime.json").is_file()


def test_v70_async_s0_uses_durable_operator_lease_and_authority_projection(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "operator-s0"})

    accepted = service.start_s0("operator-s0")
    assert accepted["operator_v70"]["live_lease"] is True
    assert accepted["operator_v70"]["latest_work"]["status"] == "LEASED"

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        snapshot = service.snapshot("operator-s0")
        if not snapshot["operator_v70"]["live_lease"]:
            break
        time.sleep(0.1)
    else:
        pytest.fail("operator-backed S0 did not finish within the test budget")

    assert snapshot["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert snapshot["operator_v70"]["latest_work"]["status"] == "ACCEPTED"
    assert snapshot["operator_v70"]["latest_work"]["action"] == "run_s0"
    assert snapshot["scientific_qualification_granted"] is False
    assert snapshot["real_world_action_authorized"] is False


def test_s1_runs_parallel_branches_broker_translation_and_dual_review(
    tmp_path: Path,
) -> None:
    requests = []

    def recording_draft(request):
        requests.append(request)
        return _s1_draft(request)

    service = _service(tmp_path, recording_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "parallel-s1"})
    service.run_s0("parallel-s1")

    result = service.run_s1("parallel-s1")

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert result["workflow"]["frontier_stages"] == ["S2"]
    assert result["next_valid_actions"] == ["inspect_s1", "ingest_ode_data"]
    epistemic = result["epistemic"]
    assert epistemic["schema_version"] == "5.8"
    assert epistemic["branch_count"] == 4
    assert epistemic["effective_independent_branches"] == 4
    assert epistemic["independence_passed"] is True
    assert epistemic["independence_scope"] == "origin_separation_only"
    assert epistemic["scientific_independence_established"] is False
    assert epistemic["disclosure_packet_count"] == 4
    assert epistemic["transfer_count"] == 2
    assert epistemic["transfer_assessment_count"] == 2
    assert epistemic["cross_task_experience_count"] == 0
    assert epistemic["cross_task_use_permitted"] is False
    assert result["scientific_qualification_granted"] is False
    assert result["real_world_action_authorized"] is False
    event_types = [event["event_type"] for event in result["events"]]
    assert "s1_parallel_exploration_started" in event_types
    assert "s1_initial_frontier_frozen" in event_types
    assert "s1_controlled_disclosure_opened" in event_types
    assert "s1_cross_branch_learning_completed" in event_types
    assert event_types[-1] == "s1_gate_evaluated"
    assert result["events"][-1]["details"]["decision"] == "OPEN"
    root = tmp_path / "tasks" / "parallel-s1"
    candidates = json.loads(
        (root / "docs" / "candidates.json").read_text(encoding="utf-8")
    )
    assert len(candidates["candidates"]) == 4
    assert len(candidates["generation_receipt_hashes"]) == 4
    for candidate in candidates["candidates"]:
        branch = candidate["candidate_id"].removeprefix("candidate.")
        assert candidate["assumption_ids"] == [f"{branch}.aggregation"]
        assert candidate["symbol_ids"] == sorted([f"{branch}.gain", f"{branch}.state"])
        assert candidate["validation_obligation_ids"] == [
            "s3_l0_replay",
            "s3_l1_structural",
            "s3_l2_numerical",
            "s4_l3_holdout",
            "s4_l4_uncertainty",
        ]
    validation_plan = json.loads(
        (root / "docs" / "validation_plan.json").read_text(encoding="utf-8")
    )
    holdout_rule = next(
        item["applicability_rule"]
        for item in validation_plan["obligations"]
        if item["check_id"] == "s4_l3_holdout"
    )
    assert "block-bootstrap" in holdout_rule
    translators = [
        request
        for request in requests
        if request.role_name.startswith("s1_cross_paradigm_translator_")
    ]
    assert len(translators) == 2
    for request in translators:
        assert "knowledge_graph" not in request.public_inputs
        packet = request.public_inputs["disclosure_packet"]
        disclosed = request.public_inputs["disclosed_units"]
        assert {item["unit_hash"] for item in disclosed} == set(
            packet["disclosed_unit_hashes"]
        )
    reviewers = [
        request
        for request in requests
        if request.role_name in {"s1_referee", "s1_red_team"}
    ]
    assert len(reviewers) == 2
    for request in reviewers:
        assert "artifacts" not in request.public_inputs
        evidence = request.public_inputs["review_evidence_packet"]
        assert len(evidence["candidates"]) == 4
        assert evidence["selected_model"]["selected_candidate_id"].startswith(
            "candidate."
        )
        assert request.public_inputs["response_budget"]["maximum_findings"] == 8
        proof = request.public_inputs["epistemic_review_proof"]
        assert proof["origin_separation"]["assesses_only_origin_separation"] is True
        assert all(item["status"] == "proposed" for item in proof["transfers"])
        assert all(
            item["scientific_support_established"] is False
            for item in proof["transfer_assessments"]
        )


def test_s1_records_one_typed_branch_repair_in_control_event_chain(
    tmp_path: Path,
) -> None:
    rejected = False

    def repairing_s1_draft(request):
        nonlocal rejected
        payload = _s1_draft(request)
        if request.role_name == "s1_mechanistic_blind" and not rejected:
            rejected = True
            payload["proposed_artifacts"] = payload["proposed_artifacts"][:1]
        return payload

    service = _service(tmp_path, repairing_s1_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "repair-s1"})
    service.run_s0("repair-s1")

    result = service.run_s1("repair-s1")

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    event_types = [event["event_type"] for event in result["events"]]
    assert "s1_branch_attempt_rejected" in event_types
    assert "s1_branch_repaired" in event_types


def test_s1_retries_one_branch_transport_failure(tmp_path: Path) -> None:
    failed = False

    def transient_transport_draft(request):
        nonlocal failed
        if request.role_name == "s1_mechanistic_blind" and not failed:
            failed = True
            raise RuntimeError("transient fixture transport failure")
        return _s1_draft(request)

    service = _service(tmp_path, transient_transport_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "transport-repair-s1"})
    service.run_s0("transport-repair-s1")

    result = service.run_s1("transport-repair-s1")

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    event_types = [event["event_type"] for event in result["events"]]
    assert "s1_branch_transport_failed" in event_types
    assert "s1_branch_repaired" in event_types


def test_s1_retries_one_final_review_transport_failure(tmp_path: Path) -> None:
    failed = False
    referee_calls = 0

    def transient_review_draft(request):
        nonlocal failed, referee_calls
        if request.role_name == "s1_referee":
            referee_calls += 1
            if not failed:
                failed = True
                raise RuntimeError("transient fixture review transport failure")
        return _s1_draft(request)

    service = _service(tmp_path, transient_review_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "review-repair-s1"})
    service.run_s0("review-repair-s1")

    result = service.run_s1("review-repair-s1")

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert referee_calls == 2
    event_types = [event["event_type"] for event in result["events"]]
    assert "s1_review_transport_failed" in event_types
    assert "s1_review_transport_recovered" in event_types


def test_s1_uses_one_pre_freeze_formalization_repair(tmp_path: Path) -> None:
    auditor_calls = 0
    repair_calls = 0

    def advisory_repair_draft(request):
        nonlocal auditor_calls, repair_calls
        payload = _s1_draft(request)
        if request.role_name == "s1_formalization_auditor":
            auditor_calls += 1
            if auditor_calls == 1:
                payload["verdict"] = "REJECT"
                payload["rationale"] = (
                    "The first formalization leaves one decisive rule undefined."
                )
                payload["findings"] = [
                    "Define the holdout uncertainty rule before freezing S1."
                ]
        if request.role_name == "s1_candidate_synthesizer_repair":
            repair_calls += 1
        return payload

    service = _service(tmp_path, advisory_repair_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "preflight-repair"})
    service.run_s0("preflight-repair")

    result = service.run_s1("preflight-repair")

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert auditor_calls == 2
    assert repair_calls == 1
    event_types = [event["event_type"] for event in result["events"]]
    assert "s1_preflight_review_requested_repair" in event_types
    assert "s1_preflight_review_recovered" in event_types


def test_s1_retries_one_pre_freeze_auditor_transport_failure(
    tmp_path: Path,
) -> None:
    failed = False
    auditor_calls = 0

    def transient_preflight_draft(request):
        nonlocal failed, auditor_calls
        if request.role_name == "s1_formalization_auditor":
            auditor_calls += 1
            if not failed:
                failed = True
                raise RuntimeError("transient fixture preflight failure")
        return _s1_draft(request)

    service = _service(tmp_path, transient_preflight_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "preflight-transport"})
    service.run_s0("preflight-transport")

    result = service.run_s1("preflight-transport")

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    assert auditor_calls == 2
    event_types = [event["event_type"] for event in result["events"]]
    assert "s1_preflight_transport_failed" in event_types
    assert "s1_preflight_transport_recovered" in event_types


def test_invalid_agent_artifacts_fail_before_stage_files_exist(
    tmp_path: Path,
) -> None:
    def invalid_draft(request):
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            payload["proposed_artifacts"] = []
        return payload

    service = _service(tmp_path, invalid_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "bad-s0"})

    with pytest.raises(StudioValidationError):
        service.run_s0("bad-s0")

    root = tmp_path / "tasks" / "bad-s0"
    assert not (root / "problem" / "contract.json").exists()
    assert not (root / "problem" / "decision_function.json").exists()
    assert not (root / "docs" / "regime.json").exists()
    assert service.snapshot("bad-s0")["workflow"]["frontier_stages"] == ["S0"]


def _ode_data_payload(*, fixture_only: bool = True) -> dict[str, object]:
    times = [index * 0.5 for index in range(36)]
    observations = [
        100.0
        / (1.0 + 19.0 * math.exp(-0.45 * time))
        * (1.0 + 0.006 * math.sin(index * 1.7))
        for index, time in enumerate(times)
    ]
    return {
        "schema_version": "5.9",
        "adapter_id": "scalar_autonomous_ode_v52",
        "time_unit": "day",
        "state_unit": "count",
        "times": times,
        "observations": observations,
        "source_id": "deterministic-logistic-control",
        "license_status": "test-fixture",
        "fixture_only": fixture_only,
    }


def _adaptive_growth_data_payload() -> dict[str, object]:
    rng = np.random.default_rng(102)
    growths = np.zeros(71, dtype=float)
    growths[0] = 0.04
    for index in range(1, len(growths)):
        growths[index] = 0.04 + rng.normal(0.0, 0.01)
    values = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(growths))))
    return {
        "schema_version": "5.9",
        "adapter_id": "adaptive_positive_series_v57",
        "time_unit": "year",
        "state_unit": "positive_index",
        "times": np.arange(len(values), dtype=float).tolist(),
        "observations": values.tolist(),
        "source_id": "deterministic-log-growth-control",
        "license_status": "test-fixture",
        "fixture_only": True,
    }


def _v67_predata_payload() -> dict[str, object]:
    return {
        "schema_version": "6.2",
        "adapter_id": "scalar_autonomous_ode_v52",
        "contract_id": "v67-authoritative-world-bank",
        "country_code": "CHN",
        "indicator_id": "SP.POP.TOTL",
        "start_year": 2000,
        "end_year": 2024,
        "minimum_observations": 23,
        "state_unit": "persons",
        "attribution": "World Bank Open Data API",
        "semantic_name": "resident population",
        "operational_definition": (
            "Published annual resident population at country aggregation."
        ),
        "observation_time_basis": "calendar year",
        "aggregation_level": "country",
        "fixture_only": True,
    }


def _create_v67_task(
    service: StudioTaskService,
    task_id: str,
) -> dict[str, object]:
    return service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": task_id,
            "evidence_scope": "development",
            "workflow_mode": "v67",
        }
    )


def _assert_v67_projection(
    snapshot: dict[str, object],
    *,
    available: bool,
    prepared: bool,
) -> dict[str, object]:
    projection = snapshot["predata_v67"]
    assert isinstance(projection, dict)
    assert projection["schema_version"] == "6.7"
    assert projection["workflow_mode"] == "v67"
    assert projection["available"] is available
    assert projection["prepared"] is prepared
    assert projection["required_before_v67_s1"] is True
    return projection


def test_v67_snapshot_projects_authoritative_predata_state(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    created = _create_v67_task(service, "v67-projection")

    created_projection = _assert_v67_projection(
        created,
        available=False,
        prepared=False,
    )
    assert created_projection["request_summary"] is None
    assert created["next_valid_actions"] == ["run_s0"]

    s0 = service.run_s0("v67-projection")
    s0_projection = _assert_v67_projection(
        s0,
        available=True,
        prepared=False,
    )
    assert s0_projection["request_summary"] is None
    assert s0["next_valid_actions"] == [
        "inspect_s0",
        "prepare_predata_v67",
    ]

    prepared = service.prepare_predata_v67(
        "v67-projection",
        _v67_predata_payload(),
    )
    _assert_v67_projection(
        prepared,
        available=False,
        prepared=True,
    )
    assert prepared["next_valid_actions"] == ["inspect_s0", "run_s1"]


def test_new_v67_task_policy_rejects_evidence_only_legacy_bypass(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-policy-bypass")
    service.run_s0("v67-policy-bypass")
    workspace = service._workspace("v67-policy-bypass")
    assert workspace.spec.spec_hash is not None
    assert workspace.current_gate("S0") is not None
    bundle = studio_service_module.build_world_bank_predata_bundle_v67(
        request=studio_service_module.StudioWorldBankDataRequestV62.model_validate(
            _v67_predata_payload()
        ),
        workspace_spec_hash=workspace.spec.spec_hash,
        s0_gate_hash=workspace.current_gate("S0"),
    )
    for relative_path, model in zip(
        PREDATA_CONTRACT_PATHS_V67,
        bundle,
        strict=True,
    ):
        studio_service_module._write_json_new(
            workspace.root / relative_path,
            model.model_dump(mode="json"),
        )
    source, measurement, protocol = bundle
    workspace.commit_evidence(
        "predata_preparation_v67",
        studio_service_module.predata_preparation_payload_v67(
            workspace_spec_hash=workspace.spec.spec_hash,
            s0_gate_hash=workspace.current_gate("S0"),
            source_contract=source,
            measurement_contract=measurement,
            predata_protocol=protocol,
        ),
    )

    assert len(workspace._artifacts_of_kind("predata_transaction_policy_v67")) == 1
    with pytest.raises(
        StudioConflictError,
        match="cannot accept legacy pre-data completion",
    ):
        service.snapshot("v67-policy-bypass")


def test_historical_prepolicy_v67_bundle_remains_legacy_compatible(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "v67-historical-prepolicy",
            "evidence_scope": "development",
        }
    )
    service.run_s0("v67-historical-prepolicy")
    workspace = service._workspace("v67-historical-prepolicy")
    assert workspace.spec.spec_hash is not None
    assert workspace.current_gate("S0") is not None
    bundle = studio_service_module.build_world_bank_predata_bundle_v67(
        request=studio_service_module.StudioWorldBankDataRequestV62.model_validate(
            _v67_predata_payload()
        ),
        workspace_spec_hash=workspace.spec.spec_hash,
        s0_gate_hash=workspace.current_gate("S0"),
    )
    for relative_path, model in zip(
        PREDATA_CONTRACT_PATHS_V67,
        bundle,
        strict=True,
    ):
        studio_service_module._write_json_new(
            workspace.root / relative_path,
            model.model_dump(mode="json"),
        )
    source, measurement, protocol = bundle
    workspace.commit_evidence(
        "predata_preparation_v67",
        studio_service_module.predata_preparation_payload_v67(
            workspace_spec_hash=workspace.spec.spec_hash,
            s0_gate_hash=workspace.current_gate("S0"),
            source_contract=source,
            measurement_contract=measurement,
            predata_protocol=protocol,
        ),
    )

    snapshot = service.snapshot("v67-historical-prepolicy")
    assert snapshot["predata_v67"]["workflow_mode"] == "v67"
    assert snapshot["predata_v67"]["transaction_status"] == "LEGACY_COMPLETED"
    assert snapshot["predata_v67"]["prepared"] is True
    assert not workspace._artifacts_of_kind("studio_workflow_mode_v67")
    assert not workspace._artifacts_of_kind("predata_transaction_policy_v67")


def test_v67_task_creation_replay_recovers_missing_policy_while_pristine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    original_commit = studio_service_module.StageWorkspaceV50.commit_evidence

    def interrupt_before_policy(self, kind, payload):
        if kind == "predata_transaction_policy_v67":
            raise KeyboardInterrupt("crash before pre-data transaction policy")
        return original_commit(self, kind, payload)

    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        interrupt_before_policy,
    )
    with pytest.raises(KeyboardInterrupt):
        _create_v67_task(service, "v67-policy-replay-pristine")

    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        original_commit,
    )
    recovered = _create_v67_task(service, "v67-policy-replay-pristine")
    workspace = service._workspace("v67-policy-replay-pristine")

    assert len(workspace._artifacts_of_kind("studio_workflow_mode_v67")) == 1
    assert len(workspace._artifacts_of_kind("predata_transaction_policy_v67")) == 1
    assert recovered["predata_v67"]["transaction_status"] == "NOT_STARTED"
    assert recovered["next_valid_actions"] == ["run_s0"]


def test_v67_task_creation_replay_refuses_policy_install_after_s0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    original_commit = studio_service_module.StageWorkspaceV50.commit_evidence

    def interrupt_before_policy(self, kind, payload):
        if kind == "predata_transaction_policy_v67":
            raise KeyboardInterrupt("crash before pre-data transaction policy")
        return original_commit(self, kind, payload)

    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        interrupt_before_policy,
    )
    with pytest.raises(KeyboardInterrupt):
        _create_v67_task(service, "v67-policy-replay-nonpristine")
    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        original_commit,
    )

    service.run_s0("v67-policy-replay-nonpristine")
    with pytest.raises(
        StudioConflictError,
        match="cannot install missing workflow or pre-data policy authority",
    ):
        _create_v67_task(service, "v67-policy-replay-nonpristine")

    workspace = service._workspace("v67-policy-replay-nonpristine")
    assert not workspace._artifacts_of_kind("predata_transaction_policy_v67")


def test_v67_stale_predata_intent_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-stale-predata")
    service.run_s0("v67-stale-predata")
    service.prepare_predata_v67(
        "v67-stale-predata",
        _v67_predata_payload(),
    )
    workspace = service._workspace("v67-stale-predata")
    recovery = studio_service_module.RecoveryKernelV60(workspace)
    recovery.recover(
        failed_stage="S0",
        category="contract_semantics",
        failure_code="replace_stale_predata_authority",
        evidence_refs=recovery.evidence_refs_for_stage("S0"),
        expected_information_gain=0.8,
    )

    snapshot = service.snapshot("v67-stale-predata")
    projection = snapshot["predata_v67"]
    assert projection["transaction_status"] == "STALE_PENDING"
    assert projection["prepared"] is False
    assert projection["recovery_available"] is False
    assert projection["request_summary"] is not None
    assert snapshot["next_valid_actions"] == ["run_s0"]

    with pytest.raises(StudioConflictError, match="stale S0 authority"):
        service.reconcile_predata_v67("v67-stale-predata")
    with pytest.raises(StudioConflictError, match="stale S0 authority"):
        service.prepare_predata_v67(
            "v67-stale-predata",
            _v67_predata_payload(),
        )
    with pytest.raises(StudioConflictError, match="stale S0 authority"):
        service.run_s1("v67-stale-predata")


@pytest.mark.parametrize("category", ["contract_semantics", "review_rejection"])
def test_public_recovery_refuses_untyped_s0_graph_mutation(
    tmp_path: Path,
    category: str,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    task_id = {
        "contract_semantics": "v67-us0-cs",
        "review_rejection": "v67-us0-rr",
    }[category]
    _create_v67_task(service, task_id)
    before = service.run_s0(task_id)
    workspace = service._workspace(task_id)
    before_gate = workspace.current_gate("S0")
    before_attempt = workspace._latest_attempt("S0")

    with pytest.raises(
        StudioValidationError,
        match="authenticated typed S0 review handoff",
    ):
        service.recover(
            task_id,
            {
                "failed_stage": "S0",
                "category": category,
                "failure_code": f"untyped_{category}",
                "expected_information_gain": 0.8,
            },
        )

    workspace = service._workspace(task_id)
    after = service.snapshot(task_id)
    assert workspace.current_gate("S0") == before_gate
    assert workspace._latest_attempt("S0") == before_attempt
    assert not workspace._artifacts_of_kind("failure_diagnosis_v60")
    assert not workspace._artifacts_of_kind("recovery_plan_v60")
    assert not workspace._artifacts_of_kind("recovery_transition_receipt_v60")
    assert after["events"] == before["events"]


def test_v67_run_s1_requires_predata_bundle(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-s1-order")
    before = service.run_s0("v67-s1-order")
    before_events = list(before["events"])

    with pytest.raises(
        StudioConflictError,
        match="V6.7 pre-data",
    ):
        service.run_s1("v67-s1-order")

    after = service.snapshot("v67-s1-order")
    projection = _assert_v67_projection(
        after,
        available=True,
        prepared=False,
    )
    assert projection["request_summary"] is None
    assert after["workflow"]["stage_statuses"]["S1"] == "frontier"
    assert after["events"] == before_events
    assert after["next_valid_actions"] == [
        "inspect_s0",
        "prepare_predata_v67",
    ]


def test_v67_predata_crash_before_preparation_record_is_reconciled_by_s1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-predata-crash")
    service.run_s0("v67-predata-crash")
    original_commit = studio_service_module.StageWorkspaceV50.commit_evidence

    def interrupt_before_authority_record(self, kind, payload):
        if kind == "predata_preparation_v67":
            raise KeyboardInterrupt("simulated crash before authority record")
        return original_commit(self, kind, payload)

    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        interrupt_before_authority_record,
    )
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(
            "v67-predata-crash",
            _v67_predata_payload(),
        )
    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        original_commit,
    )

    workspace = service._workspace("v67-predata-crash")
    assert workspace._artifacts_of_kind("predata_preparation_v67") == []
    pending = service.snapshot("v67-predata-crash")
    assert pending["predata_v67"]["transaction_status"] == "RECOVERY_PENDING"
    assert pending["predata_v67"]["prepared"] is False
    assert pending["predata_v67"]["recovery_available"] is True
    assert pending["next_valid_actions"] == [
        "inspect_s0",
        "reconcile_predata_v67",
    ]

    recovered = service.run_s1("v67-predata-crash")
    workspace = service._workspace("v67-predata-crash")
    assert recovered["predata_v67"]["transaction_status"] == "COMPLETED"
    assert recovered["predata_v67"]["prepared"] is True
    assert workspace.current_gate("S1") is not None
    assert len(workspace._artifacts_of_kind("predata_preparation_intent_v67")) == 1
    assert len(workspace._artifacts_of_kind("predata_preparation_v67")) == 1
    assert len(workspace._artifacts_of_kind("predata_preparation_completion_v67")) == 1
    assert (
        len(
            [
                event
                for event in service._events("v67-predata-crash")
                if event.event_type == "predata_bundle_prepared_v67"
            ]
        )
        == 1
    )


def test_v67_predata_rejects_legacy_ode_intake_before_write(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-no-legacy-data")
    service.run_s0("v67-no-legacy-data")
    service.prepare_predata_v67(
        "v67-no-legacy-data",
        _v67_predata_payload(),
    )
    service.run_s1("v67-no-legacy-data")
    workspace = service._workspace("v67-no-legacy-data")
    raw_path = workspace.root / "data" / "raw" / "ode_series.json"
    assert not raw_path.exists()

    with pytest.raises(
        StudioConflictError,
        match="V6.7.*official-source",
    ):
        service.ingest_ode_data(
            "v67-no-legacy-data",
            _ode_data_payload(),
        )

    assert not raw_path.exists()
    snapshot = service.snapshot("v67-no-legacy-data")
    _assert_v67_projection(
        snapshot,
        available=False,
        prepared=True,
    )
    assert snapshot["next_valid_actions"] == [
        "inspect_s1",
        "ingest_world_bank_data",
    ]


def test_v67_prepare_identical_request_is_idempotent(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-idempotent-prepare")
    service.run_s0("v67-idempotent-prepare")
    request = _v67_predata_payload()
    first = service.prepare_predata_v67(
        "v67-idempotent-prepare",
        request,
    )
    workspace = service._workspace("v67-idempotent-prepare")
    first_projection = _assert_v67_projection(
        first,
        available=False,
        prepared=True,
    )
    first_events = list(first["events"])
    first_evidence = workspace._artifacts_of_kind("predata_preparation_v67")

    replay = service.prepare_predata_v67(
        "v67-idempotent-prepare",
        request,
    )

    assert replay["events"] == first_events
    assert replay["predata_v67"] == first_projection
    assert workspace._artifacts_of_kind("predata_preparation_v67") == first_evidence

    changed = dict(request)
    changed["semantic_name"] = "different population construct"
    with pytest.raises(
        StudioConflictError,
        match="differs from the frozen V6.7",
    ):
        service.prepare_predata_v67(
            "v67-idempotent-prepare",
            changed,
        )
    assert service.snapshot("v67-idempotent-prepare")["events"] == first_events


def test_v67_reopen_restores_public_request_summary_and_actions(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-reopen-summary")
    service.run_s0("v67-reopen-summary")
    request = _v67_predata_payload()
    service.prepare_predata_v67("v67-reopen-summary", request)

    reopened = _service(tmp_path, _ode_backhalf_draft)
    before_s1 = reopened.snapshot("v67-reopen-summary")
    projection = _assert_v67_projection(
        before_s1,
        available=False,
        prepared=True,
    )
    summary = projection["request_summary"]
    assert isinstance(summary, dict)
    for key, value in request.items():
        assert summary[key] == value
    required_hashes = {
        "source_contract_hash",
        "measurement_contract_hash",
        "protocol_hash",
        "preparation_evidence_hash",
    }
    assert required_hashes.issubset(summary)
    assert all(
        isinstance(summary[key], str) and len(summary[key]) == 64
        for key in required_hashes
    )
    allowed_summary_fields = (
        set(request)
        | required_hashes
        | {
            "capability_pack_hash",
            "intent_hash",
            "completion_hash",
        }
    )
    assert set(summary).issubset(allowed_summary_fields)
    assert {
        "times",
        "observations",
        "raw_body",
        "response_body",
        "private_targets",
    }.isdisjoint(summary)
    assert before_s1["next_valid_actions"] == ["inspect_s0", "run_s1"]

    after_s1 = reopened.run_s1("v67-reopen-summary")
    _assert_v67_projection(
        after_s1,
        available=False,
        prepared=True,
    )
    assert after_s1["next_valid_actions"] == [
        "inspect_s1",
        "ingest_world_bank_data",
    ]


def _assert_exact_completed_predata_transaction(
    service: StudioTaskService,
    task_id: str,
) -> None:
    snapshot = service.snapshot(task_id)
    projection = snapshot["predata_v67"]
    assert projection["transaction_status"] == "COMPLETED"
    assert projection["prepared"] is True
    assert projection["recovery_available"] is False
    assert isinstance(projection["intent_hash"], str)
    assert isinstance(projection["completion_hash"], str)
    workspace = service._workspace(task_id)
    assert workspace.verify()
    assert workspace.current_gate("S1") is None
    assert workspace.status().stage_statuses["S1"] == "frontier"
    assert all(
        (workspace.root / relative_path).is_file()
        for relative_path in PREDATA_CONTRACT_PATHS_V67
    )
    assert len(workspace._artifacts_of_kind("predata_preparation_intent_v67")) == 1
    assert len(workspace._artifacts_of_kind("predata_preparation_v67")) == 1
    assert len(workspace._artifacts_of_kind("predata_preparation_completion_v67")) == 1
    assert not (workspace.root / "data/raw/ode_series.json").exists()
    assert not (workspace.root / SOURCE_RAW_PATH).exists()
    assert (
        len(
            [
                event
                for event in service._events(task_id)
                if event.event_type == "predata_bundle_prepared_v67"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("completed_file_writes", [0, 1, 2, 3])
def test_v67_predata_reconciles_crash_after_each_projection_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_file_writes: int,
) -> None:
    task_id = f"v67-projection-crash-{completed_file_writes}"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_write = studio_service_module._write_bytes_new
    writes = 0

    def interrupted_write(path, payload):
        nonlocal writes
        if completed_file_writes == 0:
            raise KeyboardInterrupt("crash after intent before first projection")
        original_write(path, payload)
        writes += 1
        if writes == completed_file_writes:
            raise KeyboardInterrupt("crash after exact projection write")

    monkeypatch.setattr(
        studio_service_module,
        "_write_bytes_new",
        interrupted_write,
    )
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(
        studio_service_module,
        "_write_bytes_new",
        original_write,
    )

    pending = service.snapshot(task_id)
    assert pending["predata_v67"]["transaction_status"] == "RECOVERY_PENDING"
    assert pending["predata_v67"]["prepared"] is False
    assert pending["predata_v67"]["request_summary"] is not None
    assert (
        pending["predata_v67"]["request_summary"]["preparation_evidence_hash"] is None
    )
    service.reconcile_predata_v67(task_id)
    _assert_exact_completed_predata_transaction(service, task_id)


@pytest.mark.parametrize(
    ("artifact_kind", "expected_pending_status", "resume_with_prepare"),
    [
        ("predata_preparation_intent_v67", "NOT_STARTED", True),
        ("predata_preparation_v67", "RECOVERY_PENDING", False),
        ("predata_preparation_completion_v67", "RECOVERY_PENDING", False),
    ],
)
def test_v67_predata_ignores_orphan_artifact_bodies_and_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
    expected_pending_status: str,
    resume_with_prepare: bool,
) -> None:
    task_id = {
        "predata_preparation_intent_v67": "v67-orphan-intent",
        "predata_preparation_v67": "v67-orphan-evidence",
        "predata_preparation_completion_v67": "v67-orphan-completion",
    }[artifact_kind]
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_emit = RunStore._emit_locked
    interrupted = False

    def interrupt_artifact_commit(self, event_type, payload):
        nonlocal interrupted
        if (
            not interrupted
            and event_type == "artifact_committed"
            and isinstance(payload, dict)
            and payload.get("kind") == artifact_kind
        ):
            interrupted = True
            raise KeyboardInterrupt("crash after artifact body before commit event")
        return original_emit(self, event_type, payload)

    monkeypatch.setattr(RunStore, "_emit_locked", interrupt_artifact_commit)
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(RunStore, "_emit_locked", original_emit)

    assert interrupted is True
    assert (
        service.snapshot(task_id)["predata_v67"]["transaction_status"]
        == expected_pending_status
    )
    if resume_with_prepare:
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    else:
        service.reconcile_predata_v67(task_id)
    _assert_exact_completed_predata_transaction(service, task_id)


def test_v67_predata_reconciles_after_preparation_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "v67-prep-before-done"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_commit = studio_service_module.StageWorkspaceV50.commit_evidence

    def interrupt_completion(self, kind, payload):
        if kind == "predata_preparation_completion_v67":
            raise KeyboardInterrupt("crash before completion commit")
        return original_commit(self, kind, payload)

    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        interrupt_completion,
    )
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        original_commit,
    )

    pending = service.snapshot(task_id)
    assert pending["predata_v67"]["transaction_status"] == "RECOVERY_PENDING"
    assert isinstance(
        pending["predata_v67"]["request_summary"]["preparation_evidence_hash"],
        str,
    )
    service.reconcile_predata_v67(task_id)
    _assert_exact_completed_predata_transaction(service, task_id)


def test_v67_predata_reconciles_completion_before_studio_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "v67-completion-before-event"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_append = service._append_predata_completion_event_v67

    def interrupt_event(task, state):
        raise KeyboardInterrupt("crash before Studio projection event")

    monkeypatch.setattr(
        service,
        "_append_predata_completion_event_v67",
        interrupt_event,
    )
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(
        service,
        "_append_predata_completion_event_v67",
        original_append,
    )

    completed = service.snapshot(task_id)
    assert completed["predata_v67"]["transaction_status"] == "COMPLETED"
    assert not [
        event
        for event in service._events(task_id)
        if event.event_type == "predata_bundle_prepared_v67"
    ]
    service.reconcile_predata_v67(task_id)
    _assert_exact_completed_predata_transaction(service, task_id)


def test_v67_predata_reconcile_does_not_duplicate_event_after_return_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "v67-event-before-return"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_snapshot = service.snapshot

    def interrupt_return(requested_task_id):
        assert requested_task_id == task_id
        raise KeyboardInterrupt("crash after Studio event before response")

    monkeypatch.setattr(service, "snapshot", interrupt_return)
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(service, "snapshot", original_snapshot)

    service.reconcile_predata_v67(task_id)
    _assert_exact_completed_predata_transaction(service, task_id)


@pytest.mark.parametrize("missing_relative_path", PREDATA_CONTRACT_PATHS_V67)
def test_v67_completed_predata_rebuilds_each_missing_projection_without_new_authority(
    tmp_path: Path,
    missing_relative_path: str,
) -> None:
    task_id = "v67-missing-" + str(
        PREDATA_CONTRACT_PATHS_V67.index(missing_relative_path)
    )
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    service.prepare_predata_v67(task_id, _v67_predata_payload())
    workspace = service._workspace(task_id)
    before_refs = {
        kind: workspace._artifacts_of_kind(kind)
        for kind in (
            "predata_preparation_intent_v67",
            "predata_preparation_v67",
            "predata_preparation_completion_v67",
        )
    }
    before_events = [
        event
        for event in service._events(task_id)
        if event.event_type == "predata_bundle_prepared_v67"
    ]
    missing_path = workspace.root / missing_relative_path
    expected_bytes = missing_path.read_bytes()
    missing_path.unlink()

    pending = service.snapshot(task_id)
    assert pending["predata_v67"]["transaction_status"] == "RECOVERY_PENDING"
    assert pending["predata_v67"]["prepared"] is False
    assert pending["predata_v67"]["completion_hash"] is not None
    service.reconcile_predata_v67(task_id)

    assert missing_path.read_bytes() == expected_bytes
    workspace = service._workspace(task_id)
    for kind, references in before_refs.items():
        assert workspace._artifacts_of_kind(kind) == references
    assert [
        event
        for event in service._events(task_id)
        if event.event_type == "predata_bundle_prepared_v67"
    ] == before_events
    _assert_exact_completed_predata_transaction(service, task_id)


def test_v67_run_s1_repairs_missing_completion_event_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "v67-consumer-event-repair"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_append = service._append_predata_completion_event_v67

    def interrupt_event(task, state):
        raise KeyboardInterrupt("crash before Studio projection event")

    monkeypatch.setattr(
        service,
        "_append_predata_completion_event_v67",
        interrupt_event,
    )
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(
        service,
        "_append_predata_completion_event_v67",
        original_append,
    )

    result = service.run_s1(task_id)

    assert result["workflow"]["stage_statuses"]["S1"] == "gate_open"
    completion_events = [
        event
        for event in service._events(task_id)
        if event.event_type == "predata_bundle_prepared_v67"
    ]
    assert len(completion_events) == 1


def test_v67_predata_recovery_refuses_to_overwrite_conflicting_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "v67-conflicting-partial"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_write = studio_service_module._write_bytes_new

    def interrupt_first_write(path, payload):
        raise KeyboardInterrupt("crash before first projection")

    monkeypatch.setattr(
        studio_service_module,
        "_write_bytes_new",
        interrupt_first_write,
    )
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(
        studio_service_module,
        "_write_bytes_new",
        original_write,
    )
    workspace = service._workspace(task_id)
    conflict_path = workspace.root / PREDATA_CONTRACT_PATHS_V67[0]
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.write_bytes(b'{"conflicting":true}\n')

    with pytest.raises(StudioConflictError, match="differs|overwrite"):
        service.reconcile_predata_v67(task_id)
    assert conflict_path.read_bytes() == b'{"conflicting":true}\n'
    assert workspace._artifacts_of_kind("predata_preparation_v67") == []
    assert workspace._artifacts_of_kind("predata_preparation_completion_v67") == []


def test_v67_predata_atomic_install_never_overwrites_racing_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "v67-install-race"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_link = studio_service_module.os.link
    conflict = b'{"racing":"conflict"}\n'
    injected = False

    def inject_conflict_before_link(source, destination):
        nonlocal injected
        destination_path = Path(destination)
        if (
            not injected
            and destination_path.name == Path(PREDATA_CONTRACT_PATHS_V67[0]).name
        ):
            destination_path.write_bytes(conflict)
            injected = True
        return original_link(source, destination)

    monkeypatch.setattr(
        studio_service_module.os,
        "link",
        inject_conflict_before_link,
    )
    with pytest.raises(StudioConflictError, match="concurrently created"):
        service.prepare_predata_v67(task_id, _v67_predata_payload())
    monkeypatch.setattr(studio_service_module.os, "link", original_link)

    workspace = service._workspace(task_id)
    conflict_path = workspace.root / PREDATA_CONTRACT_PATHS_V67[0]
    assert injected is True
    assert conflict_path.read_bytes() == conflict
    assert workspace._artifacts_of_kind("predata_preparation_v67") == []
    assert workspace._artifacts_of_kind("predata_preparation_completion_v67") == []


def test_json_new_atomic_install_never_overwrites_racing_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "document.json"
    conflict = b'{"owner":"other-writer"}\n'
    original_link = studio_service_module.os.link

    def inject_conflict_before_link(source, destination):
        Path(destination).write_bytes(conflict)
        return original_link(source, destination)

    monkeypatch.setattr(
        studio_service_module.os,
        "link",
        inject_conflict_before_link,
    )
    with pytest.raises(StudioConflictError, match="concurrently created"):
        studio_service_module._write_json_new(path, {"owner": "service"})

    assert path.read_bytes() == conflict


def test_studio_event_append_is_atomic_when_replace_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, "v67-event-atomic")
    before_events = service._events("v67-event-atomic")
    event_path = service._event_path("v67-event-atomic")
    before_bytes = event_path.read_bytes()
    original_replace = studio_service_module.os.replace

    def interrupt_event_replace(source, destination):
        if Path(destination) == event_path:
            raise KeyboardInterrupt("crash before atomic event projection swap")
        return original_replace(source, destination)

    monkeypatch.setattr(
        studio_service_module.os,
        "replace",
        interrupt_event_replace,
    )
    with pytest.raises(KeyboardInterrupt):
        service._append_event(
            "v67-event-atomic",
            event_type="atomic_event_probe",
            status="succeeded",
            message="Atomic event projection probe",
        )
    monkeypatch.setattr(
        studio_service_module.os,
        "replace",
        original_replace,
    )

    assert event_path.read_bytes() == before_bytes
    assert service._events("v67-event-atomic") == before_events


def test_v67_pending_predata_rejects_changed_request_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "v67-pending-request"
    service = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(service, task_id)
    service.run_s0(task_id)
    original_write = studio_service_module._write_bytes_new

    def interrupt_first_write(path, payload):
        raise KeyboardInterrupt("crash before first projection")

    monkeypatch.setattr(
        studio_service_module,
        "_write_bytes_new",
        interrupt_first_write,
    )
    request = _v67_predata_payload()
    with pytest.raises(KeyboardInterrupt):
        service.prepare_predata_v67(task_id, request)
    monkeypatch.setattr(
        studio_service_module,
        "_write_bytes_new",
        original_write,
    )
    changed = dict(request)
    changed["semantic_name"] = "a conflicting registered construct"

    with pytest.raises(
        StudioConflictError,
        match="differs from the frozen V6.7 pre-data bundle",
    ):
        service.prepare_predata_v67(task_id, changed)

    workspace = service._workspace(task_id)
    assert len(workspace._artifacts_of_kind("predata_preparation_intent_v67")) == 1
    assert workspace._artifacts_of_kind("predata_preparation_v67") == []
    assert all(
        not (workspace.root / relative_path).exists()
        for relative_path in PREDATA_CONTRACT_PATHS_V67
    )
    service.reconcile_predata_v67(task_id)
    _assert_exact_completed_predata_transaction(service, task_id)


def test_v67_two_services_coalesce_identical_predata_request(
    tmp_path: Path,
) -> None:
    task_id = "v67-concurrent-same"
    first = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(first, task_id)
    first.run_s0(task_id)
    second = _service(tmp_path, _ode_backhalf_draft)
    barrier = threading.Barrier(2)
    request = _v67_predata_payload()

    def prepare(service):
        barrier.wait()
        return service.prepare_predata_v67(task_id, request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(prepare, (first, second)))

    assert all(
        result["predata_v67"]["transaction_status"] == "COMPLETED" for result in results
    )
    _assert_exact_completed_predata_transaction(first, task_id)


def test_v67_two_services_make_different_request_first_writer_wins(
    tmp_path: Path,
) -> None:
    task_id = "v67-concurrent-different"
    first = _service(tmp_path, _ode_backhalf_draft)
    _create_v67_task(first, task_id)
    first.run_s0(task_id)
    second = _service(tmp_path, _ode_backhalf_draft)
    barrier = threading.Barrier(2)
    original = _v67_predata_payload()
    changed = dict(original)
    changed["semantic_name"] = "alternate registered population construct"

    def prepare(service, request):
        barrier.wait()
        try:
            service.prepare_predata_v67(task_id, request)
        except Exception as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(prepare, first, original),
            executor.submit(prepare, second, changed),
        )
        outcomes = [future.result() for future in futures]

    assert sum(outcome is None for outcome in outcomes) == 1
    failures = [outcome for outcome in outcomes if outcome is not None]
    assert len(failures) == 1
    assert isinstance(failures[0], StudioConflictError)
    assert "differs from the frozen" in str(failures[0])
    summary = first.snapshot(task_id)["predata_v67"]["request_summary"]
    assert summary["semantic_name"] in {
        original["semantic_name"],
        changed["semantic_name"],
    }
    _assert_exact_completed_predata_transaction(first, task_id)


def test_registered_ode_path_runs_s2_through_s6_without_claiming_authority(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task(
        {
            "objective": (
                "Forecast the next positive scalar population observation from "
                "a frozen time series and report the uncertainty interval."
            ),
            "workspace_id": "ode-backhalf",
        }
    )
    service.run_s0("ode-backhalf")
    service.run_s1("ode-backhalf")

    intake = service.ingest_ode_data(
        "ode-backhalf",
        _ode_data_payload(),
    )
    assert intake["backhalf"]["data_received"] is True
    assert intake["workflow"]["stage_statuses"]["S2"] == "frontier"
    assert intake["next_valid_actions"] == ["inspect_s1", "run_backhalf"]

    result = service.run_backhalf("ode-backhalf")

    assert all(
        result["workflow"]["stage_statuses"][stage] == "gate_open"
        for stage in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
    )
    assert result["backhalf"]["workflow_complete"] is True
    assert result["backhalf"]["selected_scientific_family"] == "logistic"
    assert result["backhalf"]["level_statuses"] == {
        "L0": "PASS",
        "L1": "PASS",
        "L2": "PASS",
        "L3": "PASS",
        "L4": "PASS",
    }
    assert result["backhalf"]["scientific_acceptance"] is True
    assert result["backhalf"]["fixture_only"] is True
    assert result["scientific_success"]["evaluated"] is True
    assert result["scientific_success"]["local_predictive_gate_status"] == ("PASS")
    assert result["scientific_success"]["scientific_success_status"] == ("NOT_RUN")
    assert result["scientific_success"]["claim_ceiling"] == ("fixture_protocol_only")
    assert (
        result["scientific_success"]["dimensions"]["data_provenance"]["status"]
        == "NOT_RUN"
    )
    assert (
        result["scientific_success"]["dimensions"]["leakage_safe_confirmation"][
            "status"
        ]
        == "PASS"
    )
    assert (
        result["scientific_success"]["dimensions"]["decision_value"]["status"]
        == "NOT_RUN"
    )
    assert "scientific_success_evaluated_v61" in [
        event["event_type"] for event in result["events"]
    ]
    assert result["scientific_qualification_granted"] is False
    assert result["real_world_action_authorized"] is False
    root = tmp_path / "tasks" / "ode-backhalf"
    workspace = service._workspace("ode-backhalf")
    assert EXECUTABLE_CANDIDATE_INTENT_PATH in {
        item.relative_path for item in workspace._manifest_for_stage("S1").files
    }
    assert EXECUTABLE_CANDIDATE_IR_PATH in {
        item.relative_path for item in workspace._manifest_for_stage("S1").files
    }
    assert EXECUTABLE_CANDIDATE_RESOLUTION_PATH in {
        item.relative_path for item in workspace._manifest_for_stage("S2").files
    }
    assert EXECUTABLE_CANDIDATE_RECEIPT_PATH in {
        item.relative_path for item in workspace._manifest_for_stage("S3").files
    }
    execution_receipt = ExecutableCandidateReceiptV62.model_validate_json(
        (root / EXECUTABLE_CANDIDATE_RECEIPT_PATH).read_text(encoding="utf-8")
    )
    assert execution_receipt.local_execution_status == "PASS"
    assert execution_receipt.scientific_qualification_status == "NOT_RUN"
    assert execution_receipt.scientific_qualification_granted is False
    assert (root / "paper" / "build" / "main.pdf").is_file()
    assert (root / "paper" / "build" / "build_receipt.json").is_file()


def test_decision_value_lifecycle_is_frozen_before_data_and_admitted_by_stage(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    request = {
        "objective": (
            "Forecast the next positive scalar count and evaluate whether the "
            "forecast improves a frozen asymmetric capacity decision."
        ),
        "workspace_id": "ode-decision-lifecycle",
        "decision_use": {
            "schema_version": "6.2",
            "decision_id": "capacity-loss",
            "value_owner_ref": "local-operations-owner",
            "action_unit": "count",
            "underage_unit_cost": 2.0,
            "overage_unit_cost": 1.0,
            "minimum_relative_loss_improvement": 0.05,
            "maximum_mean_normalized_regret": 0.20,
        },
    }
    service.create_task(request)
    service.run_s0("ode-decision-lifecycle")
    service.run_s1("ode-decision-lifecycle")
    service.ingest_ode_data(
        "ode-decision-lifecycle",
        _ode_data_payload(),
    )
    result = service.run_backhalf("ode-decision-lifecycle")

    workspace = service._workspace("ode-decision-lifecycle")
    expected = {
        "S0": DECISION_INTENT_PATH,
        "S2": DECISION_CONTRACT_PATH,
        "S4": ROLLING_CONFIRMATION_PATH,
        "S5": DECISION_EVIDENCE_PATH,
    }
    for stage, relative_path in expected.items():
        manifest = workspace._manifest_for_stage(stage)
        assert relative_path in {item.relative_path for item in manifest.files}
    root = tmp_path / "tasks" / "ode-decision-lifecycle"
    evidence = DecisionValueEvidenceV62.model_validate_json(
        (root / DECISION_EVIDENCE_PATH).read_text(encoding="utf-8")
    )
    assert evidence.status == "PASS"
    assert evidence.scientific_decision_status == "NOT_RUN"
    assert evidence.fixture_only is True
    assert result["backhalf"]["rolling_confirmation_admission_status"] == "PASS"
    assert result["backhalf"]["rolling_confirmation_status"] == "PASS"
    assert result["backhalf"]["decision_evidence_admission_status"] == "PASS"
    assert result["backhalf"]["decision_evidence_status"] == "PASS"
    assert result["backhalf"]["scientific_decision_status"] == "NOT_RUN"
    dossier = json.loads(
        (root / "results" / "decision_dossier.json").read_text(encoding="utf-8")
    )
    assert dossier["next_action"] in {
        "request_human_decision",
        "return_to_data_acquisition",
    }
    assert dossier["next_action"] != "draft_report_only"
    assert result["real_world_action_authorized"] is False

    changed = dict(request)
    changed["decision_use"] = {
        **request["decision_use"],
        "underage_unit_cost": 3.0,
    }
    with pytest.raises(StudioConflictError, match="another mission"):
        service.create_task(changed)


def test_world_bank_fixture_intake_is_bound_to_current_s2_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    values = list(_ode_data_payload()["observations"])

    def fixture_fetcher(contract):
        records = [
            {
                "indicator": {
                    "id": contract.indicator_id,
                    "value": "Population, total",
                },
                "country": {"id": "NZ", "value": "New Zealand"},
                "countryiso3code": contract.country_code,
                "date": str(year),
                "value": values[year - contract.start_year],
                "unit": "",
                "obs_status": "",
                "decimal": 0,
            }
            for year in range(contract.end_year, contract.start_year - 1, -1)
        ]
        body = json.dumps(
            [
                {
                    "page": 1,
                    "pages": 1,
                    "per_page": 1000,
                    "total": len(records),
                    "sourceid": "2",
                    "lastupdated": "2026-07-13",
                },
                records,
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        return SourceHTTPResponseV62(
            status=200,
            final_url=contract.exact_url,
            content_type="application/json",
            body=body,
        )

    service = _service(
        tmp_path,
        _ode_backhalf_draft,
        world_bank_fetcher=fixture_fetcher,
    )
    service.create_task(
        {
            "objective": (
                "Forecast one registered public population indicator while "
                "preserving exact source provenance and claim limits."
            ),
            "workspace_id": "world-bank-s2",
            "evidence_scope": "public_data",
            "decision_use": {
                "schema_version": "6.2",
                "decision_id": "population-capacity-control",
                "value_owner_ref": "fixture-value-owner",
                "action_unit": "count",
                "underage_unit_cost": 2.0,
                "overage_unit_cost": 1.0,
            },
        }
    )
    service.run_s0("world-bank-s2")
    service.run_s1("world-bank-s2")
    intake = service.ingest_world_bank_data(
        "world-bank-s2",
        {
            "schema_version": "6.2",
            "adapter_id": "scalar_autonomous_ode_v52",
            "contract_id": "nzl-population-fixture-control",
            "country_code": "NZL",
            "indicator_id": "SP.POP.TOTL",
            "start_year": 1988,
            "end_year": 2023,
            "minimum_observations": 23,
            "state_unit": "count",
            "attribution": (
                "World Bank, World Development Indicators, SP.POP.TOTL, "
                "fixture transport control only."
            ),
            "semantic_name": "Population, total",
            "operational_definition": (
                "Annual total population represented by the registered "
                "World Bank indicator in this fixture transport control."
            ),
            "observation_time_basis": "calendar year",
            "aggregation_level": "country total",
            "fixture_only": True,
        },
    )
    assert intake["backhalf"]["data_received"] is True
    workspace = service._workspace("world-bank-s2")
    decision = service._backhalf_orchestrator("world-bank-s2", workspace).run_s2()

    assert decision == "OPEN"
    snapshot = service.snapshot("world-bank-s2")
    assert snapshot["backhalf"]["source_integrity_status"] == "PASS"
    assert snapshot["backhalf"]["scientific_provenance_status"] == "NOT_RUN"
    root = tmp_path / "tasks" / "world-bank-s2"
    binding = json.loads((root / PROVENANCE_BINDING_PATH).read_text(encoding="utf-8"))
    assert binding["status"] == "PASS"
    assert binding["scientific_provenance_status"] == "NOT_RUN"
    assert binding["fixture_only"] is True
    manifest = workspace._manifest_for_stage("S2")
    manifest_paths = {item.relative_path for item in manifest.files}
    assert PROVENANCE_BINDING_PATH in manifest_paths
    assert "data/source_provenance_v62/raw_response.json" in manifest_paths
    assert SOURCE_ACQUISITION_AUTH_PATH in manifest_paths
    assert S2_SOURCE_REVERIFICATION_PATH in manifest_paths
    assert "checks/s2_data_transform_receipt.json" in manifest_paths
    acquisition = json.loads(
        (root / SOURCE_ACQUISITION_AUTH_PATH).read_text(encoding="utf-8")
    )
    reverification = json.loads(
        (root / S2_SOURCE_REVERIFICATION_PATH).read_text(encoding="utf-8")
    )
    assert (
        binding["source_acquisition_authority_receipt_hash"]
        == (acquisition["receipt_hash"])
    )
    assert (
        binding["s2_source_reverification_receipt_hash"]
        == (reverification["receipt_hash"])
    )
    assert binding["checks"]["source_acquisition_authority_authenticated"] is True
    assert binding["checks"]["current_s2_source_reverification_authenticated"] is True
    assert snapshot["scientific_qualification_granted"] is False
    assert snapshot["real_world_action_authorized"] is False

    result = service.run_backhalf("world-bank-s2")
    closure = result["scientific_closure"]
    assert closure["evaluated"] is True
    assert closure["source_integrity_status"] == "PASS"
    assert closure["scientific_provenance_status"] == "NOT_RUN"
    assert closure["decision_evidence_status"] == "PASS"
    assert closure["scientific_decision_status"] == "NOT_RUN"
    assert closure["stage_admission_status"] == "PASS"
    assert closure["closure_verification_status"] == "PASS"
    assert closure["local_evidence_status"] == "NOT_RUN"
    assert closure["scientific_closure_status"] == "NOT_RUN"
    assert closure["claim_ceiling"] == "fixture_protocol_only"
    assert closure["scientific_qualification_granted"] is False
    assert closure["real_world_action_authorized"] is False
    attempt_root = root / SCIENTIFIC_CLOSURE_ROOT / "a1"
    assert (attempt_root / "admission.json").is_file()
    assert (attempt_root / "report.json").is_file()
    assert (attempt_root / "verification.json").is_file()


def test_registered_adaptive_path_runs_candidate_graph_through_s6(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task(
        {
            "objective": (
                "Forecast the next positive scalar observation and recover "
                "from a failed autonomous ODE family using a frozen graph."
            ),
            "workspace_id": "adaptive-backhalf",
        }
    )
    service.run_s0("adaptive-backhalf")
    service.run_s1("adaptive-backhalf")
    service.ingest_ode_data(
        "adaptive-backhalf",
        _adaptive_growth_data_payload(),
    )

    result = service.run_backhalf("adaptive-backhalf")

    assert all(
        result["workflow"]["stage_statuses"][stage] == "gate_open"
        for stage in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
    )
    assert result["backhalf"]["adapter_id"] == ("adaptive_positive_series_v57")
    assert result["backhalf"]["selected_branch"] == "log_growth"
    assert result["backhalf"]["selected_scientific_family"] == ("log_random_walk_drift")
    assert result["backhalf"]["recovery_triggered"] is True
    assert result["backhalf"]["level_statuses"] == {
        "L0": "PASS",
        "L1": "PASS",
        "L2": "PASS",
        "L3": "PASS",
        "L4": "PASS",
    }
    assert result["backhalf"]["scientific_acceptance"] is True
    assert result["scientific_success"]["evaluated"] is True
    assert result["scientific_success"]["local_predictive_gate_status"] == ("FAIL")
    assert result["scientific_success"]["scientific_success_status"] == "FAIL"
    assert result["scientific_success"]["confirmation"]["status"] == "FAIL"
    assert result["scientific_qualification_granted"] is False
    assert result["real_world_action_authorized"] is False


def test_backhalf_recovers_one_partial_s2_projection_and_replays(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task(
        {
            "objective": (
                "Forecast a positive scalar series after recovering one "
                "interrupted S2 projection."
            ),
            "workspace_id": "ode-s2-recovery",
        }
    )
    service.run_s0("ode-s2-recovery")
    service.run_s1("ode-s2-recovery")
    service.ingest_ode_data("ode-s2-recovery", _ode_data_payload())
    root = tmp_path / "tasks" / "ode-s2-recovery"
    (root / "data" / "ledger.json").write_text(
        '{"partial":true}\n',
        encoding="utf-8",
        newline="\n",
    )

    result = service.run_backhalf("ode-s2-recovery")

    assert result["backhalf"]["workflow_complete"] is True
    assert result["recovery"]["same_attempt_retries"] == 1
    transition = next(
        event
        for event in result["events"]
        if event["event_type"] == "recovery_transition_v60"
    )
    assert transition["details"]["transition_status"] == ("SAME_ATTEMPT_RETRY_READY")
    assert "data/ledger.json" in transition["details"]["quarantined_paths"]
    assert (root / "data" / "raw" / "ode_series.json").is_file()


def test_unresolved_adaptive_graph_pauses_at_capability_gap(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task(
        {
            "objective": (
                "Forecast a positive oscillating series and change direction "
                "when the registered autonomous ODE family fails."
            ),
            "workspace_id": "ode-scientific-recovery",
        }
    )
    service.run_s0("ode-scientific-recovery")
    service.run_s1("ode-scientific-recovery")
    payload = _ode_data_payload()
    payload["observations"] = [
        25.0 if index % 2 == 0 else 125.0 for index in range(len(payload["times"]))
    ]
    service.ingest_ode_data("ode-scientific-recovery", payload)

    result = service.run_backhalf("ode-scientific-recovery")

    assert result["backhalf"]["workflow_complete"] is False
    assert result["recovery"]["scientific_attempts_started"] == 2
    assert result["recovery"]["human_required"] is True
    assert result["recovery"]["human_reason"].startswith("s4_")
    assert result["workflow"]["stage_statuses"]["S4"] == ("awaiting_gate_evidence")
    transition = next(
        event
        for event in reversed(result["events"])
        if event["event_type"] == "recovery_transition_v60"
    )
    assert transition["details"]["action"] == "HUMAN"
    assert transition["details"]["revoke_from"] is None
    assert transition["details"]["transition_status"] == "HUMAN_REQUIRED"
    root = tmp_path / "tasks" / "ode-scientific-recovery"
    assert (root / "data" / "raw" / "ode_series.json").is_file()
    assert (root / "docs" / "model_spec.json").is_file()


def test_second_attempt_switches_from_failed_ode_to_adaptive_graph(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task(
        {
            "objective": (
                "Forecast the next positive scalar observation and change "
                "direction after a falsified autonomous ODE."
            ),
            "workspace_id": "ode-to-adaptive",
        }
    )
    service.run_s0("ode-to-adaptive")
    service.run_s1("ode-to-adaptive")
    payload = _adaptive_growth_data_payload()
    payload["adapter_id"] = "scalar_autonomous_ode_v52"
    service.ingest_ode_data("ode-to-adaptive", payload)

    second = service.run_backhalf("ode-to-adaptive")

    assert second["recovery"]["scientific_attempts_started"] == 2
    assert second["recovery"]["last_action"] == "BRANCH"
    assert second["recovery"]["last_revoke_from"] == "S1"
    assert all(
        second["workflow"]["stage_statuses"][stage] == "gate_open"
        for stage in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
    )
    assert second["backhalf"]["adapter_id"] == ("adaptive_positive_series_v57")
    assert second["backhalf"]["selected_branch"] == "log_growth"
    assert second["backhalf"]["selected_scientific_family"] == ("log_random_walk_drift")
    assert second["backhalf"]["recovery_triggered"] is True
    assert all(
        status == "PASS" for status in second["backhalf"]["level_statuses"].values()
    )
    assert second["backhalf"]["scientific_acceptance"] is True
    assert second["scientific_qualification_granted"] is False
    assert second["real_world_action_authorized"] is False
    event_types = [event["event_type"] for event in second["events"]]
    assert "registered_branch_resume_started" in event_types


def test_backhalf_fails_closed_without_data(tmp_path: Path) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "ode-no-data"})
    service.run_s0("ode-no-data")
    service.run_s1("ode-no-data")

    with pytest.raises(StudioValidationError, match="user-supplied"):
        service.run_backhalf("ode-no-data")


def test_ode_intake_rejects_a_missing_typed_execution_ir(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "not-ode"})
    service.run_s0("not-ode")
    service.run_s1("not-ode")
    (tmp_path / "tasks" / "not-ode" / EXECUTABLE_CANDIDATE_IR_PATH).unlink()

    with pytest.raises(StudioValidationError, match="open S1 gate"):
        service.ingest_ode_data("not-ode", _ode_data_payload())


def test_s0_uses_one_bounded_repair_after_typed_rejection(
    tmp_path: Path,
) -> None:
    generator_calls = 0

    def repairing_draft(request):
        nonlocal generator_calls
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
            if generator_calls == 1:
                payload["proposed_artifacts"] = []
        return payload

    service = _service(tmp_path, repairing_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "repair-s0"})

    result = service.run_s0("repair-s0")

    assert generator_calls == 2
    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert "s0_generator_rejected" in [
        event["event_type"] for event in result["events"]
    ]
    completed = next(
        event
        for event in result["events"]
        if event["event_type"] == "s0_generator_completed"
    )
    assert completed["details"]["generator_attempts"] == 2


def test_s0_repairs_non_executable_decision_expression_before_commit(
    tmp_path: Path,
) -> None:
    generator_calls = 0

    def repairing_expression_draft(request):
        nonlocal generator_calls
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
            if generator_calls == 1:
                artifact = next(
                    item
                    for item in payload["proposed_artifacts"]
                    if item["artifact_type"] == "decision_function"
                )
                decision = json.loads(artifact["content"])
                decision["expression"] += ", followed by explanatory prose"
                artifact["content"] = canonical_json(decision)
        return payload

    service = _service(tmp_path, repairing_expression_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "expression-repair"})

    result = service.run_s0("expression-repair")

    assert generator_calls == 2
    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    rejected = next(
        event
        for event in result["events"]
        if event["event_type"] == "s0_generator_rejected"
    )
    assert "safe arithmetic evaluator" in rejected["details"]["failure_signature"]


def test_s0_review_rejection_creates_one_graph_native_repair_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []
    generator_calls = 0
    reviewer_calls = 0

    def review_repair_draft(request):
        nonlocal generator_calls, reviewer_calls
        requests.append(request)
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
        else:
            reviewer_calls += 1
            if reviewer_calls == 1:
                payload["verdict"] = "REJECT"
                payload["findings"] = ["INCOMPLETE_SENTENCE"]
                payload["rationale"] = (
                    "The first contract has a mechanically incomplete field."
                )
        return payload

    service = _service(tmp_path, review_repair_draft)
    _inject_s0_projection_sentence_defect(service, monkeypatch)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "s0-review-repair"})

    result = service.run_s0("s0-review-repair")
    workspace = service._workspace("s0-review-repair")
    state = workspace.graph.project_state()

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert workspace._latest_attempt("S0") == 2
    assert generator_calls == reviewer_calls == 2
    assert (
        state.snapshot.node_statuses[workspace._node_map()[("S0", 1, "work")].node_hash]
        == "revoked"
    )
    assert (
        state.snapshot.node_statuses[workspace._node_map()[("S0", 1, "gate")].node_hash]
        == "revoked"
    )
    historical_outcomes = [
        item
        for _, item in workspace._artifacts_of_kind(
            "stage_gate_outcome_v66",
            StageGateOutcomeV66,
        )
    ]
    assert len(historical_outcomes) == 1
    assert verify_stage_gate_outcome_v66(workspace, historical_outcomes[0])
    forged_payload = historical_outcomes[0].model_dump(mode="json")
    forged_payload["authority_auth_tag"] = "f" * 64
    forged_payload["outcome_hash"] = sha256_value(
        {key: value for key, value in forged_payload.items() if key != "outcome_hash"}
    )
    forged = StageGateOutcomeV66.model_validate(forged_payload)
    assert not verify_stage_gate_outcome_v66(workspace, forged)
    quarantine = (
        workspace.root
        / ".fma"
        / "recovery_v60"
        / "attempts"
        / "a1"
        / "s0"
        / "quarantine"
    )
    assert (quarantine / "docs" / "regime.json").is_file()
    assert (quarantine / "docs" / "s0_evaluation_profile_v66.json").is_file()
    generators = [item for item in requests if item.role_kind == "generator"]
    reviewers = [item for item in requests if item.role_kind == "reviewer"]
    assert "semantic_repair_context" not in generators[0].public_inputs
    repair_context = generators[1].public_inputs["semantic_repair_context"]
    assert repair_context["reviewer_rationale_included"] is False
    assert repair_context["private_evidence_included"] is False
    assert "reviewer_rationale" not in repair_context
    assert "semantic_repair_context" not in reviewers[1].public_inputs
    assert reviewers[0].context_id != reviewers[1].context_id
    gate_events = [
        event
        for event in result["events"]
        if event["event_type"] == "s0_gate_evaluated"
    ]
    assert [item["details"]["decision"] for item in gate_events] == [
        "BLOCKED",
        "OPEN",
    ]
    assert gate_events[0]["details"]["gate_outcome_hash"]
    assert workspace.verify()


def test_s0_reopens_after_crash_between_blocked_gate_and_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_calls = 0
    reviewer_calls = 0

    def interrupted_draft(request):
        nonlocal generator_calls, reviewer_calls
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
        else:
            reviewer_calls += 1
            if reviewer_calls == 1:
                payload["verdict"] = "REJECT"
                payload["findings"] = ["INCOMPLETE_SENTENCE"]
                payload["rationale"] = "A deterministic incomplete sentence remains."
        return payload

    service = _service(tmp_path, interrupted_draft)
    _inject_s0_projection_sentence_defect(service, monkeypatch)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "s0-blocked-crash"})
    original = service._execute_s0_rejection_recovery

    def crash_before_recovery(*args, **kwargs):
        raise KeyboardInterrupt("simulated process loss before recovery")

    monkeypatch.setattr(
        service,
        "_execute_s0_rejection_recovery",
        crash_before_recovery,
    )
    with pytest.raises(KeyboardInterrupt):
        service.run_s0("s0-blocked-crash")
    workspace = service._workspace("s0-blocked-crash")
    assert workspace.status().stage_statuses["S0"] == "blocked"
    assert workspace._latest_attempt("S0") == 1

    monkeypatch.setattr(
        service,
        "_execute_s0_rejection_recovery",
        original,
    )
    result = service.run_s0("s0-blocked-crash")

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert workspace._latest_attempt("S0") == 2
    assert generator_calls == reviewer_calls == 2
    assert "s0_interrupted_recovery_resumed" in [
        item["event_type"] for item in result["events"]
    ]
    assert workspace.verify()


def test_s0_replays_authenticated_review_after_finding_set_commit_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_calls = 0
    reviewer_calls = 0

    def interrupted_draft(request):
        nonlocal generator_calls, reviewer_calls
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
        else:
            reviewer_calls += 1
            if reviewer_calls == 1:
                payload["verdict"] = "REJECT"
                payload["findings"] = ["INCOMPLETE_SENTENCE"]
                payload["rationale"] = "A deterministic incomplete sentence remains."
        return payload

    service = _service(tmp_path, interrupted_draft)
    _inject_s0_projection_sentence_defect(service, monkeypatch)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "s0-review-replay"})
    original_commit = studio_service_module.StageWorkspaceV50.commit_evidence
    interrupted = False

    def crash_before_finding_set(self, kind, payload):
        nonlocal interrupted
        if kind == "s0_review_finding_set_v66" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("simulated process loss before finding-set commit")
        return original_commit(self, kind, payload)

    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        crash_before_finding_set,
    )
    with pytest.raises(KeyboardInterrupt):
        service.run_s0("s0-review-replay")
    workspace = service._workspace("s0-review-replay")
    assert workspace.status().stage_statuses["S0"] == "awaiting_gate_evidence"
    assert len(workspace._artifacts_of_kind("independent_review_receipt_v50")) == 1
    assert workspace._artifacts_of_kind("s0_review_finding_set_v66") == []
    assert generator_calls == reviewer_calls == 1

    monkeypatch.setattr(
        studio_service_module.StageWorkspaceV50,
        "commit_evidence",
        original_commit,
    )
    result = service.run_s0("s0-review-replay")

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert workspace._latest_attempt("S0") == 2
    assert generator_calls == reviewer_calls == 2
    resumed = [
        item
        for item in result["events"]
        if item["event_type"] == "s0_gate_evaluated"
        and item["details"].get("resumed_authenticated_review")
    ]
    assert len(resumed) == 1
    assert resumed[0]["details"]["new_reviewer_invoked"] is False
    assert workspace.verify()


def test_s0_opens_from_authenticated_approve_without_reviewer_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_calls = 0
    reviewer_calls = 0

    def counted_draft(request):
        nonlocal generator_calls, reviewer_calls
        if request.role_kind == "generator":
            generator_calls += 1
        else:
            reviewer_calls += 1
        return _valid_draft(request)

    service = _service(tmp_path, counted_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "s0-approve-replay"})
    original_transition = service._apply_s0_review_gate_transition

    def crash_before_gate(*args, **kwargs):
        raise KeyboardInterrupt("simulated process loss after authenticated approval")

    monkeypatch.setattr(
        service,
        "_apply_s0_review_gate_transition",
        crash_before_gate,
    )
    with pytest.raises(KeyboardInterrupt):
        service.run_s0("s0-approve-replay")
    workspace = service._workspace("s0-approve-replay")
    assert workspace.status().stage_statuses["S0"] == "awaiting_gate_evidence"
    assert generator_calls == reviewer_calls == 1

    monkeypatch.setattr(
        service,
        "_apply_s0_review_gate_transition",
        original_transition,
    )
    result = service.run_s0("s0-approve-replay")

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert workspace._latest_attempt("S0") == 1
    assert generator_calls == reviewer_calls == 1
    resumed = [
        item for item in result["events"] if item["event_type"] == "s0_gate_evaluated"
    ]
    assert len(resumed) == 1
    assert resumed[0]["details"]["resumed_authenticated_review"] is True
    assert resumed[0]["details"]["new_reviewer_invoked"] is False
    assert workspace.verify()


def test_approving_s0_review_cannot_be_recorded_as_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    service.create_task(
        {"objective": OBJECTIVE, "workspace_id": "s0-approve-not-blocked"}
    )
    original_transition = service._apply_s0_review_gate_transition

    def stop_after_authenticated_review(*args, **kwargs):
        raise KeyboardInterrupt("stop before approved gate evaluation")

    monkeypatch.setattr(
        service,
        "_apply_s0_review_gate_transition",
        stop_after_authenticated_review,
    )
    with pytest.raises(KeyboardInterrupt):
        service.run_s0("s0-approve-not-blocked")
    workspace = service._workspace("s0-approve-not-blocked")
    manifest = workspace._manifest_for_stage("S0")
    checks = workspace._latest_checks("S0", str(manifest.manifest_hash))
    reviews = workspace._latest_reviews("S0", str(manifest.manifest_hash))
    irrelevant = workspace.commit_evidence(
        "irrelevant_evidence_v66",
        {"not": "a finding set"},
    )

    with pytest.raises(
        PermissionError,
        match="requires an authenticated rejecting review",
    ):
        record_blocked_stage_gate_v66(
            workspace,
            stage="S0",
            manifest_hash=str(manifest.manifest_hash),
            policy_hash=studio_service_module.POLICIES["S0"].policy_hash,
            check_result_hashes=[
                str(item.result_hash)
                for item in checks.values()
                if item.result_hash is not None
            ],
            review_receipt_hashes=[str(reviews["referee"].receipt_hash)],
            finding_set_hash=irrelevant.sha256,
            reason_codes=["independent_review_rejected"],
        )

    assert workspace.status().stage_statuses["S0"] == "awaiting_gate_evidence"
    monkeypatch.setattr(
        service,
        "_apply_s0_review_gate_transition",
        original_transition,
    )
    result = service.run_s0("s0-approve-not-blocked")
    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert workspace.verify()


def test_s0_rebuilds_context_after_crash_following_graph_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_calls = 0
    reviewer_calls = 0

    def interrupted_draft(request):
        nonlocal generator_calls, reviewer_calls
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
        else:
            reviewer_calls += 1
            if reviewer_calls == 1:
                payload["verdict"] = "REJECT"
                payload["findings"] = ["INCOMPLETE_SENTENCE"]
                payload["rationale"] = "A deterministic incomplete sentence remains."
        return payload

    service = _service(tmp_path, interrupted_draft)
    _inject_s0_projection_sentence_defect(service, monkeypatch)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "s0-context-crash"})
    original_builder = studio_service_module.build_s0_repair_context_v66

    def crash_before_context(*args, **kwargs):
        raise KeyboardInterrupt("simulated process loss before context commit")

    monkeypatch.setattr(
        studio_service_module,
        "build_s0_repair_context_v66",
        crash_before_context,
    )
    with pytest.raises(KeyboardInterrupt):
        service.run_s0("s0-context-crash")
    workspace = service._workspace("s0-context-crash")
    assert workspace.status().stage_statuses["S0"] == "frontier"
    assert workspace._latest_attempt("S0") == 2
    assert service._current_s0_repair_context(workspace) is None

    monkeypatch.setattr(
        studio_service_module,
        "build_s0_repair_context_v66",
        original_builder,
    )
    result = service.run_s0("s0-context-crash")

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert workspace._latest_attempt("S0") == 2
    assert generator_calls == reviewer_calls == 2
    assert "s0_repair_context_rebuilt" in [
        item["event_type"] for item in result["events"]
    ]
    assert workspace.verify()


def test_s0_repeated_review_failure_stops_without_a_third_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_calls = 0
    reviewer_calls = 0

    def repeated_reject_draft(request):
        nonlocal generator_calls, reviewer_calls
        payload = _valid_draft(request)
        if request.role_kind == "generator":
            generator_calls += 1
        else:
            reviewer_calls += 1
            payload["verdict"] = "REJECT"
            payload["findings"] = ["INCOMPLETE_SENTENCE"]
            payload["rationale"] = "The same mechanically incomplete field remains."
        return payload

    service = _service(tmp_path, repeated_reject_draft)
    _inject_s0_projection_sentence_defect(
        service,
        monkeypatch,
        every_attempt=True,
    )
    service.create_task({"objective": OBJECTIVE, "workspace_id": "s0-repeat-stop"})

    result = service.run_s0("s0-repeat-stop")
    calls_before_reopen = (generator_calls, reviewer_calls)
    replay = service.run_s0("s0-repeat-stop")
    workspace = service._workspace("s0-repeat-stop")

    assert workspace._latest_attempt("S0") == 2
    assert result["workflow"]["stage_statuses"]["S0"] == "blocked"
    assert result["recovery"]["stopped"] is True
    assert result["recovery"]["last_action"] == "ABSTAIN"
    assert result["next_valid_actions"] == ["inspect_s0"]
    assert (generator_calls, reviewer_calls) == (2, 2)
    assert (generator_calls, reviewer_calls) == calls_before_reopen
    assert replay["events"] == result["events"]
    assert workspace.verify()


def test_s0_semantic_rejection_requires_human_without_graph_retry(
    tmp_path: Path,
) -> None:
    calls = 0

    def human_semantic_draft(request):
        nonlocal calls
        payload = _valid_draft(request)
        if request.role_kind == "reviewer":
            calls += 1
            payload["verdict"] = "REJECT"
            payload["findings"] = ["SEMANTIC_BOUNDARY_UNRESOLVED"]
            payload["rationale"] = (
                "The system boundary needs an accountable human definition."
            )
        return payload

    service = _service(tmp_path, human_semantic_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "s0-human-semantic"})

    result = service.run_s0("s0-human-semantic")
    replay = service.run_s0("s0-human-semantic")
    workspace = service._workspace("s0-human-semantic")

    assert workspace._latest_attempt("S0") == 1
    assert result["workflow"]["stage_statuses"]["S0"] == "blocked"
    assert result["recovery"]["human_required"] is True
    assert result["recovery"]["last_action"] == "HUMAN"
    assert result["next_valid_actions"] == ["inspect_s0"]
    assert calls == 1
    assert replay["events"] == result["events"]
    assert workspace.verify()


def test_http_bridge_requires_token_for_mutation(tmp_path: Path) -> None:
    service = _service(tmp_path, _s1_draft)
    server = StudioHTTPServer(
        ("127.0.0.1", 0),
        service,
        token=BRIDGE_TOKEN,
        allowed_origins={"http://localhost:3001"},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = urllib.request.Request(
            base + "/api/v1/health",
            headers={"Origin": "http://localhost:3001"},
        )
        with urllib.request.urlopen(health, timeout=5) as response:
            assert json.load(response)["authority_key_exposed"] is False

        body = json.dumps({"objective": OBJECTIVE, "workspace_id": "http-task"}).encode(
            "utf-8"
        )
        unauthenticated = urllib.request.Request(
            base + "/api/v1/tasks",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(unauthenticated, timeout=5)
        assert denied.value.code == 401

        authenticated = urllib.request.Request(
            base + "/api/v1/tasks",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001",
                "X-FMA-Bridge-Token": BRIDGE_TOKEN,
            },
            method="POST",
        )
        with urllib.request.urlopen(authenticated, timeout=5) as response:
            payload = json.load(response)
        assert payload["task_id"] == "http-task"
        assert payload["workflow"]["frontier_stages"] == ["S0"]

        service.run_s0("http-task")
        run_s1 = urllib.request.Request(
            base + "/api/v1/tasks/http-task/run-s1",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001",
                "X-FMA-Bridge-Token": BRIDGE_TOKEN,
            },
            method="POST",
        )
        with urllib.request.urlopen(run_s1, timeout=5) as response:
            accepted = json.load(response)
            assert response.status == 202
        assert accepted["activity"] in {"accepted", "running"}
        for _ in range(100):
            snapshot = service.snapshot("http-task")
            if snapshot["activity"] not in {"accepted", "running"}:
                break
            time.sleep(0.02)
        assert snapshot["workflow"]["stage_statuses"]["S1"] == "gate_open"
        assert snapshot["epistemic"]["branch_count"] == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_bridge_reconciles_v67_predata_with_empty_body() -> None:
    service = MagicMock()
    service.reconcile_predata_v67.return_value = {
        "status": "success",
        "task_id": "http-predata-recovery",
        "predata_v67": {
            "transaction_status": "COMPLETED",
            "recovery_available": False,
            "prepared": True,
        },
    }
    server = StudioHTTPServer(
        ("127.0.0.1", 0),
        service,
        token=BRIDGE_TOKEN,
        allowed_origins={"http://localhost:3001"},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        reconcile = urllib.request.Request(
            (base + "/api/v1/tasks/http-predata-recovery/reconcile-predata"),
            headers={
                "Origin": "http://localhost:3001",
                "X-FMA-Bridge-Token": BRIDGE_TOKEN,
            },
            method="POST",
        )
        with urllib.request.urlopen(reconcile, timeout=5) as response:
            snapshot = json.load(response)
            assert response.status == 200

        assert snapshot["predata_v67"]["transaction_status"] == "COMPLETED"
        service.reconcile_predata_v67.assert_called_once_with("http-predata-recovery")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_bridge_prepares_v67_predata_before_s1(tmp_path: Path) -> None:
    service = _service(tmp_path, _s1_draft)
    service.create_task(
        {
            "objective": OBJECTIVE,
            "workspace_id": "http-predata",
            "evidence_scope": "development",
            "workflow_mode": "v67",
        }
    )
    service.run_s0("http-predata")
    server = StudioHTTPServer(
        ("127.0.0.1", 0),
        service,
        token=BRIDGE_TOKEN,
        allowed_origins={"http://localhost:3001"},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    request_payload = {
        "schema_version": "6.2",
        "adapter_id": "scalar_autonomous_ode_v52",
        "contract_id": "http-predata-world-bank",
        "country_code": "CHN",
        "indicator_id": "SP.POP.TOTL",
        "start_year": 2000,
        "end_year": 2024,
        "minimum_observations": 23,
        "state_unit": "persons",
        "attribution": "World Bank Open Data API",
        "semantic_name": "resident population",
        "operational_definition": (
            "Published annual resident population at country aggregation."
        ),
        "observation_time_basis": "calendar year",
        "aggregation_level": "country",
        "fixture_only": True,
    }
    try:
        invalid_payload = dict(request_payload)
        invalid_payload.pop("operational_definition")
        invalid = urllib.request.Request(
            base + "/api/v1/tasks/http-predata/prepare-predata",
            data=json.dumps(invalid_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001",
                "X-FMA-Bridge-Token": BRIDGE_TOKEN,
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(invalid, timeout=5)
        assert rejected.value.code == 400
        assert json.load(rejected.value)["type"] == "invalid_arguments"

        prepare = urllib.request.Request(
            base + "/api/v1/tasks/http-predata/prepare-predata",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": "http://localhost:3001",
                "X-FMA-Bridge-Token": BRIDGE_TOKEN,
            },
            method="POST",
        )
        with urllib.request.urlopen(prepare, timeout=5) as response:
            snapshot = json.load(response)
            assert response.status == 201

        assert snapshot["workflow"]["stage_statuses"]["S0"] == "gate_open"
        assert snapshot["workflow"]["stage_statuses"]["S1"] == "frontier"
        assert "predata_bundle_prepared_v67" in [
            event["event_type"] for event in snapshot["events"]
        ]
        workspace = service._workspace("http-predata")
        assert (
            workspace.root / "docs" / "measurement_study_design_contract_v67.json"
        ).is_file()
        assert (
            workspace.root / "docs" / "predata_execution_protocol_v67.json"
        ).is_file()
        assert workspace.verify()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
