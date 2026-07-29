from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fma.studio.s1_runtime import (
    CandidateMathematicalFormDraftV58,
    CandidateStructureDraftV58,
    S1SelectionDraftV58,
    ValidationRuleDraftV58,
)
from fma.v6.executable_candidate import RegisteredFamilySearchIntentV62
from fma.studio.backhalf_runtime import (
    DataMappingDraftV59,
    DecisionNarrativeDraftV59,
)
from fma.studio.service import DecisionFunctionDraftV58
from fma.v5.workspace_schemas import RegimeDiagnosisV50
from fma.v5_8.epistemic import (
    EpistemicGraphBuilderV58,
    EpistemicGraphStoreV58,
    ExperienceProjectionV58,
    KnowledgeBrokerV58,
    KnowledgeUnitV58,
    TransferAssessmentV58,
    TransferHypothesisV58,
)
from fma.v5_8.stage_driver import (
    CodexStageRoleTransportV58,
    role_draft_schema_v58,
)
from fma.v5_1.codex_stage_driver import RoleRequestV51


NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)
S0_GATE = "a" * 64
SPEC_HASH = "b" * 64


def _unit(branch: str, index: int, origin: str) -> KnowledgeUnitV58:
    return KnowledgeUnitV58.seal(
        unit_id=f"{branch}.knowledge.{index}",
        task_id="epistemic-fixture",
        branch_id=branch,
        kind="counterexample" if index == 1 else "hypothesis",
        statement=f"{branch} branch proposes a falsifiable structural fact {index}.",
        applicability_conditions=["Only under the declared S0 system boundary."],
        falsification_test="Reject on the frozen development residual test.",
        shared_input_hashes=[S0_GATE],
        independent_origin_hashes=[origin],
        evidence_refs=sorted([S0_GATE, origin]),
        ancestor_unit_hashes=[],
        status="mechanically_valid",
        utility_hint=0.8 - index * 0.1,
        created_by="model",
        created_at=NOW,
    )


def _initial_builder() -> EpistemicGraphBuilderV58:
    builder = EpistemicGraphBuilderV58(
        task_id="epistemic-fixture",
        workspace_spec_hash=SPEC_HASH,
        s0_gate_hash=S0_GATE,
    )
    for branch, origin in (
        ("mechanistic", "1" * 64),
        ("null_baseline", "2" * 64),
        ("statistical", "3" * 64),
    ):
        builder.add_units([_unit(branch, 1, origin)])
    builder.freeze_initial_frontier()
    return builder


def test_independence_excludes_shared_s0_input_but_detects_reused_origin() -> None:
    builder = _initial_builder()
    graph = builder.build()

    assert graph.independence.passed is True
    assert graph.independence.effective_independent_branches == 3
    assert graph.independence.shared_input_hashes == [S0_GATE]

    reused = EpistemicGraphBuilderV58(
        task_id="epistemic-fixture",
        workspace_spec_hash=SPEC_HASH,
        s0_gate_hash=S0_GATE,
    )
    reused.add_units(
        [
            _unit("mechanistic", 1, "1" * 64),
            _unit("statistical", 1, "1" * 64),
            _unit("null_baseline", 1, "2" * 64),
        ]
    )
    reused.freeze_initial_frontier()
    correlated = reused.build()

    assert correlated.independence.passed is False
    assert correlated.independence.cross_branch_overlap_pairs == [
        "mechanistic|statistical"
    ]


def test_broker_never_echoes_recipient_knowledge_and_preserves_diversity() -> None:
    builder = _initial_builder()
    graph = builder.build()
    packet = KnowledgeBrokerV58().disclose(
        graph,
        recipient_branch_id="mechanistic",
        limit=2,
    )
    unit_by_hash = {unit.unit_hash: unit for unit in graph.knowledge_units}

    assert packet.recipient_branch_id not in packet.source_branch_ids
    assert len(packet.source_branch_ids) == 2
    assert {unit_by_hash[item].branch_id for item in packet.disclosed_unit_hashes} == {
        "null_baseline",
        "statistical",
    }


def test_transfer_is_hypothesis_and_recipient_assessment_is_not_science() -> None:
    builder = _initial_builder()
    initial = builder.build()
    units = {unit.unit_id: unit for unit in initial.knowledge_units}
    source = units["statistical.knowledge.1"]
    transfer = TransferHypothesisV58.seal(
        transfer_id="transfer.statistical.to.mechanistic",
        source_unit_hashes=[source.unit_hash],
        source_branch_ids=["statistical"],
        target_branch_id="mechanistic",
        target_interpretation=(
            "Treat the statistical counterexample as a missing forcing candidate."
        ),
        proposed_modification=(
            "Add one explicit forcing term without claiming a causal mechanism."
        ),
        falsification_test="Reject unless the frozen residual obligation improves.",
        translator_receipt_hash="4" * 64,
    )
    builder.add_transfers([transfer])
    assessment = TransferAssessmentV58.seal(
        transfer_hash=transfer.transfer_hash,
        target_branch_id="mechanistic",
        verdict="ACCEPT_FOR_TEST",
        rationale=("The proposal is coherent with the branch but remains untested."),
        required_test="Compare against the original branch on frozen data.",
        assessor_receipt_hash="5" * 64,
    )
    builder.add_transfer_assessments([assessment])
    graph = builder.build()

    assert graph.transfers[0].status == "proposed"
    assert graph.transfer_assessments[0].scientific_support_established is False


def test_s1_knowledge_is_quarantined_from_cross_task_experience() -> None:
    graph = _initial_builder().build()
    projection = graph.experience_projection

    assert projection.eligible_unit_hashes == []
    assert projection.cross_task_use_permitted is False
    assert set(projection.quarantined_unit_hashes) == {
        unit.unit_hash for unit in graph.knowledge_units
    }

    supported = graph.knowledge_units[0].model_copy(
        update={
            "status": "independently_supported",
            "independent_support_count": 2,
            "unit_hash": None,
        }
    )
    supported = KnowledgeUnitV58.seal(**supported.model_dump(exclude={"unit_hash"}))
    still_quarantined = ExperienceProjectionV58.project(
        [supported],
        promotion_authority_present=False,
    )
    assert still_quarantined.cross_task_use_permitted is False


def test_graph_store_is_content_addressed_and_rejects_tampering(tmp_path) -> None:
    graph = _initial_builder().build()
    store = EpistemicGraphStoreV58(tmp_path)
    path = store.save(graph)

    assert path.name == f"{graph.graph_hash}.json"
    assert store.load_head() == graph
    assert store.summary()["independence_passed"] is True

    payload = graph.model_dump(mode="json")
    payload["knowledge_units"][0]["statement"] = "tampered knowledge statement"
    with pytest.raises(ValidationError, match="knowledge unit hash differs"):
        type(graph).model_validate(payload)


def test_v58_wire_schema_constrains_branch_artifact_registry(tmp_path) -> None:
    artifact_types = [
        "assumptions",
        "candidate",
        "knowledge_units",
        "symbols",
    ]
    required_artifacts = {
        artifact_type: {
            "additionalProperties": False,
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "type": "object",
        }
        for artifact_type in artifact_types
    }
    request = RoleRequestV51.seal(
        request_id="request-wire-schema",
        task_id="epistemic-fixture",
        stage="S1",
        role_name="s1_mechanistic_blind",
        role_kind="generator",
        subject_id="candidate.mechanistic",
        objective="Develop one falsifiable mechanistic candidate branch.",
        public_inputs={"required_artifacts": required_artifacts},
        allowed_candidate_ids=["candidate.mechanistic"],
        authority_denials=["cannot_sign_gate"],
    )

    schema = role_draft_schema_v58(request)
    artifacts = schema["properties"]["proposed_artifacts"]

    assert artifacts["minItems"] == artifacts["maxItems"] == 4
    variants = artifacts["items"]["anyOf"]
    assert [
        item["properties"]["artifact_type"]["const"] for item in variants
    ] == artifact_types
    assert all(item["properties"]["content"]["type"] == "object" for item in variants)
    assert schema["properties"]["request_hash"]["const"] == request.request_hash
    assert schema["properties"]["verdict"]["const"] == "NOT_APPLICABLE"
    assert schema["properties"]["rationale"]["maxLength"] == 800
    assert schema["properties"]["findings"]["maxItems"] == 8
    assert schema["properties"]["findings"]["items"]["maxLength"] == 400

    raw = {
        "schema_version": "5.1",
        "request_hash": request.request_hash,
        "role_name": request.role_name,
        "selected_candidate_id": "candidate.mechanistic",
        "verdict": "NOT_APPLICABLE",
        "rationale": "Typed nested artifacts are proposed for validation.",
        "assumptions": [],
        "findings": [],
        "uncertainties": [],
        "proposed_artifacts": [
            {
                "artifact_type": artifact_type,
                "content": {"value": artifact_type},
            }
            for artifact_type in artifact_types
        ],
        "authority_claimed": False,
    }
    normalized = CodexStageRoleTransportV58(tmp_path)._parse_draft(raw, request)
    assert json.loads(normalized.proposed_artifacts[0].content) == {
        "value": "assumptions"
    }


def test_v58_wire_schema_uses_nested_typed_s0_artifacts(tmp_path) -> None:
    required_artifacts = {
        "decision_function": DecisionFunctionDraftV58.model_json_schema(),
        "regime_diagnosis": RegimeDiagnosisV50.model_json_schema(),
    }
    request = RoleRequestV51.seal(
        request_id="request-s0-wire-schema",
        task_id="epistemic-fixture",
        stage="S0",
        role_name="problem_formulator",
        role_kind="generator",
        subject_id="s0_problem_contract",
        objective="Formalize a falsifiable S0 decision contract.",
        public_inputs={"required_artifacts": required_artifacts},
        allowed_candidate_ids=[],
        authority_denials=["cannot_sign_gate"],
    )

    schema = role_draft_schema_v58(request)
    artifacts = schema["properties"]["proposed_artifacts"]
    variants = artifacts["items"]["anyOf"]

    assert artifacts["minItems"] == artifacts["maxItems"] == 2
    assert [
        item["properties"]["artifact_type"]["const"] for item in variants
    ] == ["decision_function", "regime_diagnosis"]
    assert all(item["properties"]["content"]["type"] == "object" for item in variants)
    assert "patternProperties" not in json.dumps(schema)


def test_v58_synthesizer_rules_fit_historical_artifact_envelope() -> None:
    request = RoleRequestV51.seal(
        request_id="request-synth-wire-schema",
        task_id="epistemic-fixture",
        stage="S1",
        role_name="s1_candidate_synthesizer",
        role_kind="generator",
        subject_id="s1-selection",
        objective="Select and formalize one candidate for bounded S1 review.",
        public_inputs={
            "required_artifacts": {
                "selection": S1SelectionDraftV58.model_json_schema(),
                "selected_candidate_structure": (
                    CandidateStructureDraftV58.model_json_schema()
                ),
                "selected_mathematical_form": (
                    CandidateMathematicalFormDraftV58.model_json_schema()
                ),
                "executable_candidate_intent": (
                    RegisteredFamilySearchIntentV62.model_json_schema()
                ),
                **{
                    f"validation_rule_{check_id}": (
                        ValidationRuleDraftV58.model_json_schema()
                    )
                    for check_id in [
                        "s3_l0_replay",
                        "s3_l1_structural",
                        "s3_l2_numerical",
                        "s4_l3_holdout",
                        "s4_l4_uncertainty",
                    ]
                },
            }
        },
        allowed_candidate_ids=["candidate.mechanistic"],
        authority_denials=["cannot_sign_gate"],
    )

    schema = role_draft_schema_v58(request)
    variants = schema["properties"]["proposed_artifacts"]["items"]["anyOf"]
    rule_variant = next(
        item
        for item in variants
        if item["properties"]["artifact_type"]["const"]
        == "validation_rule_s4_l3_holdout"
    )
    rule_schema = rule_variant["properties"]["content"]
    applicability = rule_schema["properties"]["applicability_rule"]
    math_variant = next(
        item
        for item in variants
        if item["properties"]["artifact_type"]["const"]
        == "selected_mathematical_form"
    )
    math_form = math_variant["properties"]["content"]["properties"][
        "mathematical_form"
    ]

    assert applicability["maxLength"] == 1200
    assert math_form["maxLength"] == 2600
    artifacts = schema["properties"]["proposed_artifacts"]
    assert artifacts["minItems"] == artifacts["maxItems"] == 9


@pytest.mark.parametrize(
    ("stage", "role_name", "artifact_type", "artifact_schema"),
    [
        ("S2", "s2_data_steward", "data_mapping", DataMappingDraftV59),
        (
            "S5",
            "s5_decision_writer",
            "decision_narrative",
            DecisionNarrativeDraftV59,
        ),
    ],
)
def test_v59_wire_schema_constrains_backhalf_role_artifacts(
    stage,
    role_name,
    artifact_type,
    artifact_schema,
) -> None:
    request = RoleRequestV51.seal(
        request_id=f"request-{role_name}",
        task_id="epistemic-fixture",
        stage=stage,
        role_name=role_name,
        role_kind="generator",
        subject_id="candidate.mechanistic",
        objective="Produce one bounded back-half draft for harness validation.",
        public_inputs={
            "required_artifacts": {
                artifact_type: artifact_schema.model_json_schema()
            }
        },
        allowed_candidate_ids=["candidate.mechanistic"],
        authority_denials=["cannot_sign_gate"],
    )

    schema = role_draft_schema_v58(request)
    artifacts = schema["properties"]["proposed_artifacts"]

    assert artifacts["minItems"] == artifacts["maxItems"] == 1
    variant = artifacts["items"]["anyOf"][0]
    assert variant["properties"]["artifact_type"]["const"] == artifact_type
    assert variant["properties"]["content"]["type"] == "object"


def test_v59_wire_schema_forbids_reviewer_artifacts_after_s1() -> None:
    request = RoleRequestV51.seal(
        request_id="request-s4-red-team",
        task_id="epistemic-fixture",
        stage="S4",
        role_name="s4_red_team",
        role_kind="reviewer",
        subject_id="s4-work",
        objective="Independently audit frozen S4 evidence.",
        public_inputs={},
        allowed_candidate_ids=[],
        authority_denials=["cannot_sign_gate"],
    )

    schema = role_draft_schema_v58(request)
    artifacts = schema["properties"]["proposed_artifacts"]

    assert artifacts["minItems"] == artifacts["maxItems"] == 0
