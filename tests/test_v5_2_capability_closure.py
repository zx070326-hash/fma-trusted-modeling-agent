from __future__ import annotations

import base64
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fma.hashing import canonical_json, sha256_value
from fma.v5.check_registry import (
    AdapterContextV50,
    CheckRegistryV50,
)
from fma.v5.external_harness import (
    PredictionDocumentV50,
    PrivateCaseCapsuleV50,
    PublicCaseSpecV50,
)
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import (
    FileBindingV50,
    StageArtifactManifestV50,
    TaskWorkspaceSpecV50,
    ValidationObligationV50,
    WorkflowProfileV50,
)
from fma.v5_1.evaluation_harness import (
    GoldAuthorityV51,
    GoldFileV51,
    inject_gold_stage_v51,
)
from fma.v5_2.candidate_space import (
    CandidateAdmissionAuthorityV52,
    CandidateAdmissionPolicyV52,
    ExpressionNodeV52,
    GeneratedCandidateV52,
    GovernedCandidateRegistryV52,
)
from fma.v5_2.cross_domain_evaluation import (
    GoldTaskObservationV52,
    compare_ablation_arms_v52,
    run_fixture_ablation_arm_v52,
    summarize_cross_domain_ablation_v52,
    summarize_gold_coverage_v52,
)
from fma.v5_2.evolution_controller import (
    EvolutionProposalV52,
    GraphEvolutionControllerV52,
    RecoveryAuthorityV52,
    RecoveryBudgetV52,
    RecoveryEvidenceV52,
    RecoveryPolicyV52,
    RecoveryStateV52,
    TransitionFileV52,
)
from fma.v5_2.ode_system import (
    ODELevelAdapterV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
    register_ode_adapters_v52,
    run_ode_replays_v52,
)
from fma.v5_2.private_qualification import (
    PrivateEvaluationRequestV52,
    PrivatePromotionAuthorityV52,
    PrivateWorkerAuthorityV52,
    run_local_private_worker_v52,
)


AUTHORITY_KEY = b"v5.2-stage-authority-key-material"


def _workspace(tmp_path: Path) -> tuple[Path, StageWorkspaceV50]:
    root = scaffold_task_workspace(
        tmp_path / "task",
        "v52fixture",
        "Exercise one graph-native recovery transition",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="v52fixture",
        graph_id="v5-v52fixture",
        objective="Exercise one graph-native recovery transition",
        mission_hash="1" * 64,
        evidence_snapshot_hash="2" * 64,
        evaluator_epoch="v52-test-epoch",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="synthetic_fixture",
        max_nodes=96,
        max_outcomes=96,
    )
    workspace = StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id="v52-stage-key",
    )
    return root, workspace


def _graph_hash(workspace: StageWorkspaceV50) -> str:
    return sha256_value(workspace.graph.project_state().model_dump(mode="json"))


def test_p0_controller_executes_real_revocation_and_new_attempt(
    tmp_path: Path,
) -> None:
    root, workspace = _workspace(tmp_path)
    budget = RecoveryBudgetV52.seal(
        max_attempts=4,
        max_same_skeleton_patches=2,
        max_candidate_switches=2,
        max_generated_candidates=2,
        max_model_calls=20,
        max_wall_time_seconds=3600,
        max_repeated_failure=2,
    )
    policy = RecoveryPolicyV52.seal(
        policy_id="v52-recovery",
        evaluator_epoch="v52-test-epoch",
        budget=budget,
        minimum_score_improvement=0.01,
        patchable_failure_codes=["residual_autocorrelation"],
        task_invalid_failure_codes=["invalid_target"],
    )
    state = RecoveryStateV52.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        policy_hash=policy.policy_hash,
        attempt_count=1,
        model_call_count=2,
    )
    evidence = RecoveryEvidenceV52.seal(
        workspace_spec_hash=workspace.spec.spec_hash,
        graph_state_hash=_graph_hash(workspace),
        evaluator_epoch="v52-test-epoch",
        failed_stage="S3",
        failed_check_hashes=["3" * 64],
        failure_codes=["residual_autocorrelation"],
        selected_candidate_hash="4" * 64,
        candidate_registry_hash="5" * 64,
        candidate_scores={"4" * 64: 0.5},
        public_evidence_hashes=["6" * 64],
    )
    transition_file = TransitionFileV52.from_text(
        "src/models/residual_patch.py",
        "PATCH_LEVEL = 1\n",
    )
    proposal = EvolutionProposalV52.seal(
        proposal_id="patch1",
        action="PATCH_SAME_SKELETON",
        proposer_role_receipt_hash="7" * 64,
        source_candidate_hash="4" * 64,
        projection_files=[transition_file],
        rationale="Correct the observed residual autocorrelation failure.",
        expected_failure_codes_addressed=["residual_autocorrelation"],
    )
    authority = RecoveryAuthorityV52("recovery-key", b"r" * 32)
    controller = GraphEvolutionControllerV52(
        authority=authority, policy=policy
    )
    decision = controller.authorize(
        workspace=workspace,
        state=state,
        evidence=evidence,
        proposal=proposal,
        registered_candidate_hashes={"4" * 64},
        admitted_candidate_receipt_hashes=set(),
    )
    receipt = controller.execute(
        workspace=workspace, decision=decision, proposal=proposal
    )

    assert receipt.status == "ATTEMPT_CREATED"
    assert receipt.earliest_affected_stage == "S3"
    assert receipt.predecessor_attempt == 1
    assert receipt.successor_attempt == 2
    assert receipt.affected_node_hashes
    assert receipt.before_graph_state_hash != receipt.after_graph_state_hash
    assert (
        root / "src" / "models" / "residual_patch.py"
    ).read_text(encoding="utf-8") == "PATCH_LEVEL = 1\n"
    assert workspace.verify()
    next_state = controller.advance_state(state, evidence, decision)
    assert next_state.attempt_count == 2
    assert next_state.same_skeleton_patch_count == 1


def _exponential_candidate(
    *,
    candidate_id: str,
    generator_hash: str,
) -> GeneratedCandidateV52:
    nodes = [
        ExpressionNodeV52(
            node_id="x",
            op="state",
            semantic_role="population",
            dimension={"count": 1},
        ),
        ExpressionNodeV52(
            node_id="r",
            op="parameter",
            semantic_role="growth_rate",
            dimension={"time": -1},
            lower_bound=-2.0,
            upper_bound=2.0,
        ),
        ExpressionNodeV52(
            node_id="rhs",
            op="multiply",
            inputs=["r", "x"],
            dimension={"count": 1, "time": -1},
        ),
        ExpressionNodeV52(
            node_id="prediction",
            op="output",
            inputs=["rhs"],
            dimension={"count": 1, "time": -1},
        ),
    ]
    return GeneratedCandidateV52.seal(
        candidate_id=candidate_id,
        domain_id="scalar_ode",
        family="exponential",
        generation=0,
        nodes=nodes,
        output_node_id="prediction",
        assumptions=["The scalar state is positive over the frozen support."],
        data_requirements=["positive_time_series"],
        limit_cases=[
            {
                "parameter_node_id": "r",
                "limit_value": 0.0,
                "reduces_to_family": "constant",
                "executable_check_id": "zero_rate_limit",
            }
        ],
        identifiability_obligations=[
            {
                "obligation_id": "identify_r",
                "parameter_node_ids": ["r"],
                "executable_check_id": "profile_r",
                "failure_consequence": "Reject the candidate when rate is unidentified.",
            }
        ],
        expected_failure_modes=["Rate estimates may be unstable on short windows."],
        generator_process_receipt_hash=generator_hash,
    )


def _logistic_candidate(
    *,
    parent_hash: str,
    generator_hash: str,
) -> GeneratedCandidateV52:
    nodes = [
        ExpressionNodeV52(
            node_id="x",
            op="state",
            semantic_role="population",
            dimension={"count": 1},
        ),
        ExpressionNodeV52(
            node_id="K",
            op="parameter",
            semantic_role="capacity",
            dimension={"count": 1},
            lower_bound=1.0,
            upper_bound=1000000.0,
        ),
        ExpressionNodeV52(
            node_id="r",
            op="parameter",
            semantic_role="growth_rate",
            dimension={"time": -1},
            lower_bound=-2.0,
            upper_bound=2.0,
        ),
        ExpressionNodeV52(
            node_id="one",
            op="constant",
            constant_value=1.0,
            dimension={},
        ),
        ExpressionNodeV52(
            node_id="ratio",
            op="divide",
            inputs=["x", "K"],
            dimension={},
        ),
        ExpressionNodeV52(
            node_id="capacity_term",
            op="subtract",
            inputs=["one", "ratio"],
            dimension={},
        ),
        ExpressionNodeV52(
            node_id="rx",
            op="multiply",
            inputs=["r", "x"],
            dimension={"count": 1, "time": -1},
        ),
        ExpressionNodeV52(
            node_id="rhs",
            op="multiply",
            inputs=["rx", "capacity_term"],
            dimension={"count": 1, "time": -1},
        ),
        ExpressionNodeV52(
            node_id="prediction",
            op="output",
            inputs=["rhs"],
            dimension={"count": 1, "time": -1},
        ),
    ]
    return GeneratedCandidateV52.seal(
        candidate_id="logistic-generated",
        domain_id="scalar_ode",
        family="logistic",
        generation=1,
        nodes=nodes,
        output_node_id="prediction",
        assumptions=["Capacity is constant over the frozen observation window."],
        data_requirements=["positive_time_series"],
        limit_cases=[
            {
                "parameter_node_id": "r",
                "limit_value": 0.0,
                "reduces_to_family": "constant",
                "executable_check_id": "zero_rate_limit",
            }
        ],
        identifiability_obligations=[
            {
                "obligation_id": "identify_capacity_rate",
                "parameter_node_ids": ["K", "r"],
                "executable_check_id": "profile_capacity_rate",
                "failure_consequence": "Reject forecasts when capacity and rate confound.",
            }
        ],
        expected_failure_modes=["Capacity is weakly identified before curvature appears."],
        parent_candidate_hashes=[parent_hash],
        operator_ids=["add_capacity_feedback"],
        generator_process_receipt_hash=generator_hash,
    )


def test_p1_generated_candidate_is_admitted_and_renamed_duplicate_rejected() -> None:
    baseline = _exponential_candidate(
        candidate_id="exp-baseline", generator_hash="8" * 64
    )
    generated = _logistic_candidate(
        parent_hash=baseline.candidate_hash, generator_hash="9" * 64
    )
    policy = CandidateAdmissionPolicyV52.seal(
        policy_id="ode-candidate-policy",
        allowed_domain_ids=["scalar_ode"],
        allowed_operators=[
            "constant",
            "divide",
            "multiply",
            "output",
            "parameter",
            "state",
            "subtract",
        ],
        available_check_ids=[
            "profile_capacity_rate",
            "profile_r",
            "zero_rate_limit",
        ],
        required_baseline_candidate_hashes=[baseline.candidate_hash],
        max_candidates=8,
        max_nodes_per_candidate=32,
        max_parameters_per_candidate=4,
    )
    authority = CandidateAdmissionAuthorityV52(
        "candidate-key", b"c" * 32
    )
    registry = GovernedCandidateRegistryV52(
        policy=policy, authority=authority, initial_candidates=[baseline]
    )
    accepted = registry.admit(
        generated, observed_generator_receipt_hashes={"9" * 64}
    )
    assert accepted.status == "ADMITTED"
    assert authority.verify(accepted)

    renamed_payload = generated.model_dump(mode="json")
    renamed_payload["candidate_id"] = "logistic-renamed"
    renamed_payload["candidate_hash"] = None
    renamed = GeneratedCandidateV52.seal(**renamed_payload)
    rejected = registry.admit(
        renamed, observed_generator_receipt_hashes={"9" * 64}
    )
    assert rejected.status == "REJECTED"
    assert "structurally_novel" in rejected.reasons
    assert len(registry.candidates) == 2


def test_p1_candidate_dimension_language_fails_closed() -> None:
    candidate = _exponential_candidate(
        candidate_id="bad-dimension", generator_hash="a" * 64
    )
    payload = candidate.model_dump(mode="json")
    payload["candidate_hash"] = None
    payload["nodes"][2]["dimension"] = {"count": 1}
    with pytest.raises(ValidationError, match="dimension"):
        GeneratedCandidateV52(**payload)


def test_p1_local_private_worker_is_signed_but_cannot_qualify(
    tmp_path: Path,
) -> None:
    public = PublicCaseSpecV50.seal(
        case_id="privatecase",
        title="Private score fixture",
        objective="Score one frozen private prediction",
        public_payload={"target_ids": ["y1", "y2"]},
        supported_mechanisms=[],
    )
    capsule = PrivateCaseCapsuleV50.seal(
        case_id="privatecase",
        public_case_hash=public.case_hash,
        holdout=[
            {"target_id": "y1", "value": 1.0},
            {"target_id": "y2", "value": 2.0},
        ],
        quality_scale=1.0,
        secrecy_canary="private-canary-do-not-disclose",
    )
    prediction = PredictionDocumentV50(
        case_id="privatecase",
        predictions=[
            {"target_id": "y1", "value": 1.0},
            {"target_id": "y2", "value": 2.0},
        ],
    )
    prediction_path = tmp_path / "prediction.json"
    prediction_bytes = (canonical_json(prediction) + "\n").encode()
    prediction_path.write_bytes(prediction_bytes)
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_text(
        canonical_json(capsule) + "\n", encoding="utf-8"
    )
    request = PrivateEvaluationRequestV52.seal(
        request_id="private-request",
        case_id="privatecase",
        public_case_hash=public.case_hash,
        registration_hash="b" * 64,
        prediction_snapshot_hash=hashlib.sha256(
            prediction_bytes
        ).hexdigest(),
        prediction_semantic_hash=sha256_value(prediction),
        private_capsule_commitment=capsule.capsule_hash,
        evaluator_epoch="private-epoch",
        minimum_quality_score=0.9,
    )
    worker_secret = b"w" * 32
    worker_receipt, process_receipt = run_local_private_worker_v52(
        request=request,
        prediction_path=prediction_path,
        private_capsule_path=capsule_path,
        worker_secret=worker_secret,
        worker_key_id="private-worker-key",
        worker_id="private-worker",
        worker_host_id="same-host",
        output_directory=tmp_path / "worker-output",
    )
    worker_authority = PrivateWorkerAuthorityV52(
        key_id="private-worker-key", secret=worker_secret
    )
    promotion = PrivatePromotionAuthorityV52(
        key_id="promotion-key",
        secret=b"p" * 32,
        coordinator_host_id="same-host",
    )
    qualification = promotion.assess(
        request=request,
        worker_receipt=worker_receipt,
        worker_authority=worker_authority,
    )

    assert worker_authority.verify(worker_receipt)
    assert process_receipt.fresh_process
    assert worker_receipt.worker_process_id != 0
    assert worker_receipt.quality_score == 1.0
    assert qualification.status == "LOCAL_PROTOCOL_VALIDATED"
    assert qualification.qualification_granted is False
    assert promotion.verify(qualification)
    serialized = canonical_json(worker_receipt)
    assert capsule.secrecy_canary not in serialized
    assert '"value":1.0' not in serialized


def _ode_fixture() -> ODETimeSeriesSnapshotV52:
    times = [index * 0.5 for index in range(36)]
    observations = []
    for index, time in enumerate(times):
        base = 100.0 / (1.0 + 19.0 * math.exp(-0.45 * time))
        observations.append(base * (1.0 + 0.006 * math.sin(index * 1.7)))
    return ODETimeSeriesSnapshotV52.seal(
        task_id="ode-fixture",
        time_unit="day",
        state_unit="count",
        times=times,
        observations=observations,
        source_id="deterministic-logistic-control",
        fixture_only=True,
    )


def test_p2_ode_adapter_runs_deterministic_l0_to_l4(
    tmp_path: Path,
) -> None:
    snapshot = _ode_fixture()
    thresholds = ODEThresholdsV52.seal(bootstrap_replicates=20)
    replay_input = tmp_path / "replay.json"
    replay_input.write_text(
        json.dumps(
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "thresholds": thresholds.model_dump(mode="json"),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    replay_hashes = run_ode_replays_v52(replay_input)
    bundle = build_ode_bundle_v52(
        snapshot=snapshot,
        thresholds=thresholds,
        replay_output_hashes=replay_hashes,
    )
    assert len(set(replay_hashes)) == 1
    assert bundle.selected_candidate_id == "logistic"
    assert [item.level for item in bundle.levels] == [
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    ]
    assert all(item.status == "PASS" for item in bundle.levels)
    assert bundle.scientific_acceptance
    assert bundle.fixture_only
    assert bundle.scientific_qualification_granted is False

    registry = CheckRegistryV50()
    register_ode_adapters_v52(registry)
    assert sorted(registry._adapters) == [
        "scalar_ode_l0",
        "scalar_ode_l1",
        "scalar_ode_l2",
        "scalar_ode_l3",
        "scalar_ode_l4",
    ]


def test_p2_ode_adapter_reads_only_frozen_manifest(
    tmp_path: Path,
) -> None:
    bundle = build_ode_bundle_v52(
        snapshot=_ode_fixture(),
        thresholds=ODEThresholdsV52.seal(bootstrap_replicates=20),
        replay_output_hashes=["c" * 64, "c" * 64],
    )
    bundle_path = tmp_path / "results" / "ode_scientific_bundle.json"
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_text(canonical_json(bundle) + "\n", encoding="utf-8")
    payload = bundle_path.read_bytes()
    manifest = StageArtifactManifestV50.seal(
        workspace_spec_hash="d" * 64,
        stage="S4",
        attempt=1,
        predecessor_gate_hash="e" * 64,
        files=[
            FileBindingV50(
                relative_path="results/ode_scientific_bundle.json",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                snapshot_artifact_hash="f" * 64,
            )
        ],
    )
    obligation = ValidationObligationV50(
        check_id="scalar_ode_l3",
        stage="S4",
        level="L3",
        evidence_class="scientific_computation",
        applicability_rule="Scalar autonomous ODE bundle is present.",
    )
    outcome = ODELevelAdapterV52("L3").run(
        AdapterContextV50(
            workspace_root=tmp_path,
            manifest=manifest,
            obligation=obligation,
        )
    )
    assert outcome.status == "PASS"
    bundle_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        ODELevelAdapterV52("L3").run(
            AdapterContextV50(
                workspace_root=tmp_path,
                manifest=manifest,
                obligation=obligation,
            )
        )


def test_p2_cross_domain_repeated_ablation_has_interval_but_no_fixture_claim(
    tmp_path: Path,
) -> None:
    observations = []
    for domain in ("event_process", "scalar_ode"):
        for case in ("casea", "caseb"):
            nuisance = sha256_value({"domain": domain, "case": case})
            for repetition in (1, 2, 3):
                control = run_fixture_ablation_arm_v52(
                    domain_id=domain,
                    case_id=case,
                    repetition=repetition,
                    mechanism_id="backward_revision",
                    mechanism_enabled=False,
                    nuisance_identity_hash=nuisance,
                    fixture_seed=17,
                    output_directory=tmp_path,
                )
                treatment = run_fixture_ablation_arm_v52(
                    domain_id=domain,
                    case_id=case,
                    repetition=repetition,
                    mechanism_id="backward_revision",
                    mechanism_enabled=True,
                    nuisance_identity_hash=nuisance,
                    fixture_seed=17,
                    output_directory=tmp_path,
                )
                observations.append(
                    compare_ablation_arms_v52(control, treatment)
                )
    summary = summarize_cross_domain_ablation_v52(observations)
    assert summary.total_observations == 12
    assert summary.valid_observations == 12
    assert summary.cross_domain_coverage_satisfied
    assert summary.repeated_measurement_satisfied
    assert summary.confidence_interval_95_low is not None
    assert summary.confidence_interval_95_high is not None
    assert (
        summary.confidence_interval_95_low
        < summary.mean_score_delta
        < summary.confidence_interval_95_high
    )
    assert summary.inference_ready_for_external_review is False
    assert summary.general_causal_claim_permitted is False
    assert summary.reason_codes == ["fixture_only_observations"]


def test_p2_gold_coverage_spans_domains_and_stages_without_claim(
    tmp_path: Path,
) -> None:
    authority = GoldAuthorityV51("gold-v52", b"g" * 32)
    observations = []
    definitions = [
        ("event_process", "eventcase", "S1"),
        ("scalar_ode", "odecase", "S3"),
    ]
    for index, (domain, task, stage) in enumerate(definitions):
        payload = (
            json.dumps({"domain": domain, "stage": stage}, sort_keys=True)
            + "\n"
        ).encode()
        package = authority.seal_package(
            package_id=f"gold-{task}",
            task_id=task,
            protocol_hash=sha256_value({"task": task, "stage": stage}),
            through_stage=stage,
            predecessor_package_hash=None,
            files=[
                GoldFileV51(
                    relative_path=f"docs/gold_{stage.lower()}.json",
                    content_base64=base64.b64encode(payload).decode(),
                    content_sha256=hashlib.sha256(payload).hexdigest(),
                )
            ],
        )
        receipt = inject_gold_stage_v51(
            package,
            authority=authority,
            target_root=tmp_path / f"gold-{index}",
        )
        observations.append(
            GoldTaskObservationV52.from_injection(
                domain_id=domain,
                receipt=receipt,
                isolated_target=True,
                process_receipt_hashes=[
                    sha256_value({"task": task, "process": index})
                ],
                fixture_only=True,
            )
        )
    summary = summarize_gold_coverage_v52(observations)
    assert summary.multi_domain_coverage_satisfied
    assert summary.multi_stage_coverage_satisfied
    assert summary.all_targets_isolated
    assert summary.process_receipts_globally_disjoint
    assert summary.general_gold_effect_claim_permitted is False
