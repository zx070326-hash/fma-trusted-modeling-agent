from __future__ import annotations

import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fma.hashing import canonical_json
from fma.v5.check_registry import CheckRegistryV50
from fma.v5.external_harness import PrivateTargetV50
from fma.v5.stage_workspace import POLICIES, StageWorkspaceError
from fma.v5.workspace_schemas import (
    ValidationObligationV50,
    ValidationPlanV50,
)
from fma.v5_2.ode_system import ODEThresholdsV52, ODETimeSeriesSnapshotV52
from fma.v5_3.campaign import (
    PublicPredictionRegistryV53,
    bind_i32_public_campaign_to_v5_v53,
    verify_i32_graph_binding_v53,
)
from fma.v5_3.custody import (
    PrivateScoreContractV53,
    create_external_capsule_and_attestation_v53,
    verify_external_custody_attestation_v53,
)
from fma.v5_3.ode_forecast import (
    ODEForecastPlanV53,
    ODEForecastReplayAuthorityV53,
    ODEForecastTargetV53,
    build_ode_forecast_bundle_v53,
    register_ode_forecast_adapters_v53,
    run_authenticated_ode_forecast_replays_v53,
)
from fma.v5_3.external_private import (
    build_private_evaluation_request_v53,
    evaluate_external_private_inputs_v53,
    sign_worker_host_attestation_v53,
    verify_external_private_run_v53,
)
from fma.v5_3.promotion import (
    assess_scientific_qualification_v53,
    sign_external_promotion_decision_v53,
)
from tests.test_v5_stage_workspace import (
    FixtureScientificAdapter,
    _approve_reviews,
    _new_workspace,
    _open_stage,
    _open_through_s2,
    _write_s0,
    _write_s1,
    _write_s2,
    _write_s3,
    _write_s4,
)


NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


def _snapshot(*, fixture_only: bool = True) -> ODETimeSeriesSnapshotV52:
    times = [index * 0.5 for index in range(36)]
    observations: list[float] = []
    for index, time in enumerate(times):
        base = 100.0 / (1.0 + 19.0 * math.exp(-0.45 * time))
        observations.append(base * (1.0 + 0.006 * math.sin(index * 1.7)))
    return ODETimeSeriesSnapshotV52.seal(
        task_id="i32-ode-control",
        time_unit="day",
        state_unit="count",
        times=times,
        observations=observations,
        source_id="deterministic-logistic-control",
        fixture_only=fixture_only,
    )


def _plan(
    snapshot: ODETimeSeriesSnapshotV52,
    thresholds: ODEThresholdsV52,
) -> ODEForecastPlanV53:
    return ODEForecastPlanV53.seal(
        plan_id="i32-public-plan",
        task_id=snapshot.task_id,
        public_snapshot_hash=snapshot.snapshot_hash,
        threshold_hash=thresholds.threshold_hash,
        targets=[
            ODEForecastTargetV53(target_id="target-a", time=18.0),
            ODEForecastTargetV53(target_id="target-b", time=20.0),
        ],
        state_unit=snapshot.state_unit,
        time_unit=snapshot.time_unit,
        frozen_at=NOW,
    )


def _accepted_bundle():
    snapshot = _snapshot()
    thresholds = ODEThresholdsV52.seal(bootstrap_replicates=20)
    plan = _plan(snapshot, thresholds)
    bundle = build_ode_forecast_bundle_v53(
        snapshot=snapshot,
        thresholds=thresholds,
        forecast_plan=plan,
        replay_output_hashes=["1" * 64, "1" * 64],
    )
    assert bundle.scientific_acceptance
    return snapshot, thresholds, plan, bundle


def _key_pair() -> tuple[bytes, bytes]:
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _custody(plan: ODEForecastPlanV53):
    private_pem, public_pem = _key_pair()
    score_contract = PrivateScoreContractV53.seal(
        contract_id="i32-score-contract",
        case_id="i32-case",
        protocol_hash="2" * 64,
        public_case_hash="3" * 64,
        forecast_plan_hash=plan.plan_hash,
        target_ids=[item.target_id for item in plan.targets],
        quality_scale=100.0,
        minimum_quality_score=0.8,
        frozen_at=NOW,
    )
    capsule, capsule_bytes, attestation = create_external_capsule_and_attestation_v53(
        score_contract=score_contract,
        private_targets=[
            PrivateTargetV50(target_id="target-a", value=95.0),
            PrivateTargetV50(target_id="target-b", value=98.0),
        ],
        secrecy_canary="i32-private-canary-do-not-disclose",
        private_source_manifest_hash="4" * 64,
        external_anchor_receipt_hash="5" * 64,
        custodian_host_id="external-custodian",
        coordinator_host_id="coordinator-host",
        generator_host_id="generator-host",
        attestation_id="i32-custody-attestation",
        attester_key_id="external-custody-key",
        private_key_pem=private_pem,
        attested_at=NOW,
    )
    verification = verify_external_custody_attestation_v53(
        attestation=attestation,
        score_contract=score_contract,
        forecast_plan=plan,
        trusted_public_keys={"external-custody-key": public_pem},
        expected_coordinator_host_id="coordinator-host",
        expected_generator_host_id="generator-host",
    )
    assert capsule.capsule_hash == attestation.capsule_commitment
    assert capsule.secrecy_canary.encode() in capsule_bytes
    assert capsule.secrecy_canary not in canonical_json(attestation)
    assert verification.status == "VERIFIED"
    assert verification.external_anchor_content_verified is False
    assert verification.qualification_granted is False
    return score_contract, attestation, verification, public_pem, capsule


def _register(
    tmp_path: Path,
    *,
    workspace_root: Path,
    bundle,
    score_contract,
    attestation,
    verification,
):
    prediction_path = workspace_root / "predictions" / "registered.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        canonical_json(bundle.final_refit.prediction_document(score_contract.case_id))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    registry = PublicPredictionRegistryV53(tmp_path / "public-registry")
    registration = registry.register(
        registration_id="i32-public-registration",
        prediction_path=prediction_path,
        workspace_root=workspace_root,
        forecast_bundle=bundle,
        score_contract=score_contract,
        custody_attestation=attestation,
        custody_verification=verification,
        external_anchor_receipt_hash="6" * 64,
    )
    assert registry.verify_registration(registration)
    return registry, registration, prediction_path


def _v53_workspace_through_s4(
    tmp_path: Path,
    bundle,
):
    root, workspace = _new_workspace(tmp_path)
    _write_s0(root)
    _open_stage(workspace, "S0", actor="model")
    base_plan = _write_s1(root, workspace)
    v53_obligations = [
        ValidationObligationV50(
            check_id=f"scalar_ode_forecast_v53_l{level}",
            stage="S3" if level <= 2 else "S4",
            level=f"L{level}",
            evidence_class="scientific_computation",
            applicability_rule=(
                "The exact frozen V5.3 ODE forecast bundle is present."
            ),
        )
        for level in range(5)
    ]
    plan = ValidationPlanV50.seal(
        obligations=[*base_plan.obligations, *v53_obligations],
        frozen_by="verifier",
    )
    (root / "docs" / "validation_plan.json").write_text(
        canonical_json(plan) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _open_stage(workspace, "S1", actor="model")
    _write_s2(root, plan)
    workspace.freeze_raw_inputs(actor="harness")
    _open_stage(workspace, "S2", actor="model")

    bundle_path = root / "results" / "ode_forecast_bundle_v53.json"
    bundle_path.write_text(
        canonical_json(bundle) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_s3(root)
    workspace.submit_stage(
        "S3",
        actor="harness",
        extra_paths=["results/ode_forecast_bundle_v53.json"],
    )
    assert workspace.run_mechanical_check("S3").status == "PASS"
    registry = CheckRegistryV50()
    for obligation in plan.obligations:
        if obligation.stage == "S3" and (
            not obligation.check_id.startswith("scalar_ode_forecast_v53_")
            and obligation.applicability == "applicable"
        ):
            registry.register(FixtureScientificAdapter(obligation))
    register_ode_forecast_adapters_v53(registry)
    for obligation in plan.obligations:
        if obligation.stage == "S3":
            registry.execute(workspace, obligation)
    _approve_reviews(workspace, "S3", POLICIES["S3"].required_review_roles)
    s3_evaluation = workspace.evaluate_gate("S3")
    assert s3_evaluation.decision == "OPEN", s3_evaluation

    _write_s4(root, plan)
    verification_summary_path = root / "results" / "verification_summary.json"
    verification_summary = json.loads(
        verification_summary_path.read_text(encoding="utf-8")
    )
    verification_summary["check_ids"] = sorted(
        item.check_id for item in plan.obligations if item.stage == "S4"
    )
    verification_summary_path.write_text(
        canonical_json(verification_summary) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    workspace.submit_stage(
        "S4",
        actor="harness",
        extra_paths=["results/ode_forecast_bundle_v53.json"],
    )
    assert workspace.run_mechanical_check("S4").status == "PASS"
    registry = CheckRegistryV50()
    for obligation in plan.obligations:
        if obligation.stage == "S4" and (
            not obligation.check_id.startswith("scalar_ode_forecast_v53_")
            and obligation.applicability == "applicable"
        ):
            registry.register(FixtureScientificAdapter(obligation))
    register_ode_forecast_adapters_v53(registry)
    for obligation in plan.obligations:
        if obligation.stage == "S4":
            registry.execute(workspace, obligation)
    _approve_reviews(workspace, "S4", POLICIES["S4"].required_review_roles)
    s4_evaluation = workspace.evaluate_gate("S4")
    assert s4_evaluation.decision == "OPEN", s4_evaluation
    return root, workspace, plan


def test_horizon_plan_rejects_observed_or_unordered_targets() -> None:
    snapshot = _snapshot()
    thresholds = ODEThresholdsV52.seal(bootstrap_replicates=20)
    plan = ODEForecastPlanV53.seal(
        plan_id="bad-support",
        task_id=snapshot.task_id,
        public_snapshot_hash=snapshot.snapshot_hash,
        threshold_hash=thresholds.threshold_hash,
        targets=[
            ODEForecastTargetV53(target_id="target-a", time=17.5),
        ],
        state_unit=snapshot.state_unit,
        time_unit=snapshot.time_unit,
        frozen_at=NOW,
    )
    with pytest.raises(ValueError, match="beyond the public"):
        plan.assert_compatible(snapshot, thresholds)
    with pytest.raises(ValueError, match="sorted and unique"):
        ODEForecastPlanV53.seal(
            plan_id="bad-order",
            task_id=snapshot.task_id,
            public_snapshot_hash=snapshot.snapshot_hash,
            threshold_hash=thresholds.threshold_hash,
            targets=[
                ODEForecastTargetV53(target_id="target-b", time=18.0),
                ODEForecastTargetV53(target_id="target-a", time=20.0),
            ],
            state_unit=snapshot.state_unit,
            time_unit=snapshot.time_unit,
            frozen_at=NOW,
        )


def test_v53_replays_and_recertifies_every_final_refit_horizon(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    thresholds = ODEThresholdsV52.seal(bootstrap_replicates=20)
    plan = _plan(snapshot, thresholds)
    replay_input = tmp_path / "replay.json"
    replay_input.write_text(
        canonical_json(
            {
                "snapshot": snapshot.model_dump(mode="json"),
                "thresholds": thresholds.model_dump(mode="json"),
                "forecast_plan": plan.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    replay_authority = ODEForecastReplayAuthorityV53(
        key_id="fixture-replay-authority",
        secret=b"fixture-replay-authority-secret-material",
    )
    replay_receipts = run_authenticated_ode_forecast_replays_v53(
        replay_input, authority=replay_authority
    )
    bundle = build_ode_forecast_bundle_v53(
        snapshot=snapshot,
        thresholds=thresholds,
        forecast_plan=plan,
        replay_receipts=replay_receipts,
        replay_authority=replay_authority,
    )

    replay_hashes = [item.deterministic_output_hash for item in replay_receipts]
    assert len(set(replay_hashes)) == 1
    assert len({item.process_id for item in replay_receipts}) == 2
    assert all(replay_authority.verify(item) for item in replay_receipts)
    assert bundle.replay_receipt_hashes == [
        item.receipt_hash for item in replay_receipts
    ]
    assert bundle.scientific_acceptance
    assert bundle.development_bundle.levels[0].status == "NOT_RUN"
    assert bundle.development_assessment.status == "PASS"
    assert bundle.final_refit.status == "PASS"
    assert [item.target_id for item in bundle.final_refit.horizons] == [
        "target-a",
        "target-b",
    ]
    assert all(item.status == "PASS" for item in bundle.final_refit.horizons)
    assert bundle.final_refit.final_fit.fit_hash != (
        bundle.development_assessment.selected_fit_hash
    )
    assert bundle.scientific_qualification_granted is False


def test_one_failed_horizon_rejects_entire_public_bundle() -> None:
    snapshot = _snapshot()
    thresholds = ODEThresholdsV52.seal(
        bootstrap_replicates=20,
        maximum_window_sensitivity_relative_range=0.001,
    )
    plan = _plan(snapshot, thresholds)
    bundle = build_ode_forecast_bundle_v53(
        snapshot=snapshot,
        thresholds=thresholds,
        forecast_plan=plan,
        replay_output_hashes=["1" * 64, "1" * 64],
    )
    assert any(item.status == "FAIL" for item in bundle.final_refit.horizons)
    assert bundle.final_refit.status == "FAIL"
    assert bundle.scientific_acceptance is False


def test_non_fixture_bundle_rejects_unauthenticated_replay_hashes() -> None:
    payload = json.loads(
        Path("experiments/iteration_31/PUBLIC_REPLAY_INPUT.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot = ODETimeSeriesSnapshotV52.model_validate(payload["snapshot"])
    thresholds = ODEThresholdsV52.model_validate(payload["thresholds"])
    plan = ODEForecastPlanV53.seal(
        plan_id="nonfixture-replay-auth-control",
        task_id=snapshot.task_id,
        public_snapshot_hash=snapshot.snapshot_hash,
        threshold_hash=thresholds.threshold_hash,
        targets=[
            ODEForecastTargetV53(target_id="target-a", time=190.0),
            ODEForecastTargetV53(target_id="target-b", time=200.0),
        ],
        state_unit=snapshot.state_unit,
        time_unit=snapshot.time_unit,
        frozen_at=NOW,
    )
    bundle = build_ode_forecast_bundle_v53(
        snapshot=snapshot,
        thresholds=thresholds,
        forecast_plan=plan,
        replay_output_hashes=["9" * 64, "9" * 64],
    )
    assert bundle.replay_authentication_required
    assert bundle.l0_checks["authenticated_replay_receipts_or_fixture_control"] is False
    assert bundle.scientific_acceptance is False


def test_external_custody_uses_pinned_signature_and_never_qualifies() -> None:
    snapshot, thresholds, plan, bundle = _accepted_bundle()
    del snapshot, thresholds, bundle
    score_contract, attestation, verification, public_pem, _ = _custody(plan)
    assert verification.status == "VERIFIED"

    rejected = verify_external_custody_attestation_v53(
        attestation=attestation,
        score_contract=score_contract,
        forecast_plan=plan,
        trusted_public_keys={"external-custody-key": public_pem},
        expected_coordinator_host_id="wrong-coordinator",
        expected_generator_host_id="generator-host",
    )
    assert rejected.status == "REJECTED"
    assert "separate_host_claim_invalid" in rejected.reason_codes
    assert rejected.qualification_granted is False


def test_public_registration_is_exact_immutable_and_private_free(
    tmp_path: Path,
) -> None:
    _, _, plan, bundle = _accepted_bundle()
    score_contract, attestation, verification, _, _ = _custody(plan)
    workspace = tmp_path / "task"
    registry, registration, prediction_path = _register(
        tmp_path,
        workspace_root=workspace,
        bundle=bundle,
        score_contract=score_contract,
        attestation=attestation,
        verification=verification,
    )
    prediction_path.write_text("{}\n", encoding="utf-8")
    assert registry.verify_registration(registration)
    prediction_path.write_text(
        canonical_json(bundle.final_refit.prediction_document(score_contract.case_id))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(RuntimeError, match="immutable and unique"):
        registry.register(
            registration_id="second-registration",
            prediction_path=prediction_path,
            workspace_root=workspace,
            forecast_bundle=bundle,
            score_contract=score_contract,
            custody_attestation=attestation,
            custody_verification=verification,
            external_anchor_receipt_hash="6" * 64,
        )
    assert registration.private_holdout_accessed_before_registration is False
    assert registration.scientific_qualification_granted is False


def test_i32_binding_requires_and_uses_real_v5_s4_gate(
    tmp_path: Path,
) -> None:
    _, _, plan, bundle = _accepted_bundle()
    score_contract, attestation, verification, _, _ = _custody(plan)
    root, workspace, validation_plan = _open_through_s2(tmp_path / "v5")
    _write_s3(root)
    _open_stage(
        workspace,
        "S3",
        actor="harness",
        scientific_checks=[
            item for item in validation_plan.obligations if item.stage == "S3"
        ],
    )
    _write_s4(root, validation_plan)

    early_registry, early_registration, _ = _register(
        tmp_path / "early",
        workspace_root=root,
        bundle=bundle,
        score_contract=score_contract,
        attestation=attestation,
        verification=verification,
    )
    assert early_registry.verify_registration(early_registration)
    with pytest.raises(StageWorkspaceError, match="current S4 gate"):
        bind_i32_public_campaign_to_v5_v53(
            workspace=workspace,
            forecast_bundle=bundle,
            score_contract=score_contract,
            custody_attestation=attestation,
            custody_verification=verification,
            prediction_registration=early_registration,
        )

    bound_root, bound_workspace, _ = _v53_workspace_through_s4(
        tmp_path / "bound-v5", bundle
    )
    result = bind_i32_public_campaign_to_v5_v53(
        workspace=bound_workspace,
        forecast_bundle=bundle,
        score_contract=score_contract,
        custody_attestation=attestation,
        custody_verification=verification,
        prediction_registration=early_registration,
    )
    assert bound_workspace.current_gate("S4") == result.binding.s4_gate_hash
    assert verify_i32_graph_binding_v53(workspace=bound_workspace, result=result)
    assert result.binding.private_evaluation_status == "NOT_RUN"
    assert result.binding.scientific_qualification_granted is False

    (bound_root / "results" / "verification_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )
    assert bound_workspace.current_gate("S4") is None
    assert not verify_i32_graph_binding_v53(workspace=bound_workspace, result=result)


def test_external_private_worker_path_is_attested_and_still_not_promotion(
    tmp_path: Path,
) -> None:
    _, _, plan, bundle = _accepted_bundle()
    (
        score_contract,
        custody_attestation,
        custody_verification,
        _,
        capsule,
    ) = _custody(plan)
    root, workspace, validation_plan = _v53_workspace_through_s4(
        tmp_path / "v5", bundle
    )
    registry, registration, prediction_path = _register(
        tmp_path / "registry",
        workspace_root=root,
        bundle=bundle,
        score_contract=score_contract,
        attestation=custody_attestation,
        verification=custody_verification,
    )
    assert registry.verify_registration(registration)
    graph_result = bind_i32_public_campaign_to_v5_v53(
        workspace=workspace,
        forecast_bundle=bundle,
        score_contract=score_contract,
        custody_attestation=custody_attestation,
        custody_verification=custody_verification,
        prediction_registration=registration,
    )
    request = build_private_evaluation_request_v53(
        request_id="i32-private-request",
        evaluator_epoch="i32-private-epoch",
        score_contract=score_contract,
        forecast_bundle=bundle,
        custody_attestation=custody_attestation,
        custody_verification=custody_verification,
        prediction_registration=registration,
        graph_binding=graph_result.binding,
        created_at=NOW,
    )

    worker_private, worker_public = _key_pair()
    host_private, host_public = _key_pair()
    prediction_bytes = prediction_path.read_bytes()
    prediction = bundle.final_refit.prediction_document(score_contract.case_id)
    worker_receipt = evaluate_external_private_inputs_v53(
        request=request,
        score_contract=score_contract,
        prediction=prediction,
        prediction_bytes_hash=hashlib.sha256(prediction_bytes).hexdigest(),
        capsule=capsule,
        worker_id="external-private-worker",
        worker_host_id="external-custodian",
        worker_executable_hash="7" * 64,
        runner_source_hash="8" * 64,
        worker_key_id="external-worker-key",
        worker_private_key_pem=worker_private,
        evaluated_at=NOW,
    )
    host_attestation = sign_worker_host_attestation_v53(
        worker_receipt=worker_receipt,
        worker_public_key_pem=worker_public,
        coordinator_host_id="coordinator-host",
        generator_host_id="generator-host",
        attestation_id="i32-worker-host-attestation",
        host_attester_key_id="external-host-key",
        host_attester_private_key_pem=host_private,
        attested_at=NOW,
    )
    verification = verify_external_private_run_v53(
        request=request,
        score_contract=score_contract,
        custody_attestation=custody_attestation,
        custody_verification=custody_verification,
        worker_receipt=worker_receipt,
        host_attestation=host_attestation,
        trusted_worker_public_keys={"external-worker-key": worker_public},
        trusted_host_public_keys={"external-host-key": host_public},
        expected_coordinator_host_id="coordinator-host",
        expected_generator_host_id="generator-host",
    )
    assert worker_receipt.private_evaluation_count == 1
    assert worker_receipt.private_values_disclosed is False
    assert worker_receipt.per_target_feedback_disclosed is False
    assert verification.status == "VERIFIED"
    assert verification.private_threshold_passed
    assert verification.scientific_qualification_granted is False

    not_run = assess_scientific_qualification_v53(
        campaign_id="i32-fixture-campaign",
        workspace=workspace,
        graph_binding=graph_result,
        registry=registry,
        prediction_registration=registration,
        forecast_bundle=bundle,
        custody_attestation=custody_attestation,
        custody_verification=custody_verification,
        worker_receipt=worker_receipt,
        worker_host_attestation=host_attestation,
        private_run_verification=verification,
        promotion_decision=None,
        trusted_promotion_public_keys={},
    )
    assert not_run.status == "NOT_RUN"
    assert not_run.qualification_granted is False

    promotion_private, promotion_public = _key_pair()
    promotion = sign_external_promotion_decision_v53(
        campaign_id="i32-fixture-campaign",
        graph_binding=graph_result,
        prediction_registration=registration,
        custody_verification=custody_verification,
        private_run_verification=verification,
        forecast_bundle=bundle,
        graph_binding_current_verified=True,
        prediction_registration_verified=True,
        external_anchor_content_verified=True,
        integrity_incident_free=True,
        promotion_key_id="external-promotion-key",
        promotion_private_key_pem=promotion_private,
        decided_at=NOW,
    )
    assert promotion.decision == "REJECT"
    assert "fixture_only_evidence" in promotion.reason_codes
    final = assess_scientific_qualification_v53(
        campaign_id="i32-fixture-campaign",
        workspace=workspace,
        graph_binding=graph_result,
        registry=registry,
        prediction_registration=registration,
        forecast_bundle=bundle,
        custody_attestation=custody_attestation,
        custody_verification=custody_verification,
        worker_receipt=worker_receipt,
        worker_host_attestation=host_attestation,
        private_run_verification=verification,
        promotion_decision=promotion,
        trusted_promotion_public_keys={"external-promotion-key": promotion_public},
    )
    assert final.status == "REJECTED"
    assert final.qualification_granted is False

    wrong_worker_private, wrong_worker_public = _key_pair()
    del wrong_worker_private
    rejected = verify_external_private_run_v53(
        request=request,
        score_contract=score_contract,
        custody_attestation=custody_attestation,
        custody_verification=custody_verification,
        worker_receipt=worker_receipt,
        host_attestation=host_attestation,
        trusted_worker_public_keys={"external-worker-key": wrong_worker_public},
        trusted_host_public_keys={"external-host-key": host_public},
        expected_coordinator_host_id="coordinator-host",
        expected_generator_host_id="generator-host",
    )
    assert rejected.status == "REJECTED"
    assert "worker_signature_invalid" in rejected.reason_codes
