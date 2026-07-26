from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fma.hashing import canonical_json
from fma.v5_1.codex_stage_driver import FixtureStageRoleTransportV51
from fma.studio.server import StudioHTTPServer
from fma.studio.service import (
    StudioTaskService,
    StudioValidationError,
)


AUTHORITY_KEY = b"studio-test-authority-key-" + b"k" * 32
BRIDGE_TOKEN = "studio-test-bridge-token-123456"
OBJECTIVE = (
    "Forecast weekly emergency visits for twelve weeks so staffing can be "
    "planned, while treating understaffing as the larger error."
)


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
    decision = {
        "schema_version": "5.0",
        "function_id": "asymmetric_staffing_loss",
        "input_names": ["prediction", "target"],
        "expression": ("2 * max(target - prediction, 0) + max(prediction - target, 0)"),
        "sense": "minimize",
        "output_unit": "visit_count",
        "canaries": [
            {
                "canary_id": "exact",
                "inputs": {"prediction": 10.0, "target": 10.0},
                "expected": 0.0,
                "tolerance": 1e-09,
            },
            {
                "canary_id": "under",
                "inputs": {"prediction": 9.0, "target": 10.0},
                "expected": 2.0,
                "tolerance": 1e-09,
            },
        ],
        "function_hash": None,
    }
    regime = {
        "schema_version": "5.0",
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
        "computable_decision_function": "asymmetric absolute staffing loss",
        "evidence_hashes": [request.public_inputs["evidence_snapshot_hash"]],
        "limitations": [
            "No forecast is usable until data provenance and validation pass."
        ],
        "diagnosis_hash": None,
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
            "abandon_criteria": [
                "Abandon if it cannot beat the frozen null baseline."
            ],
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


def _service(tmp_path: Path, draft_factory=_valid_draft) -> StudioTaskService:
    return StudioTaskService(
        tmp_path / "tasks",
        authority_key=AUTHORITY_KEY,
        authority_key_id="studio-test-v1",
        role_transport_factory=lambda _: FixtureStageRoleTransportV51(draft_factory),
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
    assert first["scientific_qualification_granted"] is False
    assert first["real_world_action_authorized"] is False
    assert first["events"][0]["event_type"] == "task_created"


def test_s0_runs_generator_reviewer_check_and_gate(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "emergency-visits"})

    result = service.run_s0("emergency-visits")

    assert result["workflow"]["stage_statuses"]["S0"] == "gate_open"
    assert result["workflow"]["frontier_stages"] == ["S1"]
    assert result["next_valid_actions"] == ["inspect_s0", "run_s1"]
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
    assert result["scientific_qualification_granted"] is False
    assert result["real_world_action_authorized"] is False
    root = tmp_path / "tasks" / "ode-backhalf"
    assert (root / "paper" / "build" / "main.pdf").is_file()
    assert (root / "paper" / "build" / "build_receipt.json").is_file()


def test_backhalf_fails_closed_without_data(tmp_path: Path) -> None:
    service = _service(tmp_path, _ode_backhalf_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "ode-no-data"})
    service.run_s0("ode-no-data")
    service.run_s1("ode-no-data")

    with pytest.raises(StudioValidationError, match="user-supplied"):
        service.run_backhalf("ode-no-data")


def test_ode_intake_rejects_an_incompatible_s1_candidate(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _s1_draft)
    service.create_task({"objective": OBJECTIVE, "workspace_id": "not-ode"})
    service.run_s0("not-ode")
    service.run_s1("not-ode")

    with pytest.raises(StudioValidationError, match="not compatible"):
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
