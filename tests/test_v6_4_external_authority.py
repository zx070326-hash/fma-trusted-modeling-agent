from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from fma.hashing import sha256_value
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v6 import external_qualification as qualification
from fma.v6.external_authority import (
    DeploymentIdentityV64,
    DeploymentTrustProviderV64,
    ExternalAuthorityManifestV64,
    ExternalAuthorityRevocationSnapshotV64,
    ExternalRoleHostAttestationV64,
    assess_external_authority_v64,
    register_external_authority_manifest_v64,
    sign_external_authority_manifest_v64,
    sign_external_authority_revocations_v64,
    sign_external_role_host_attestation_v64,
    verify_external_authority_v64,
)
from fma.v6.external_prediction_runtime import (
    run_current_model_external_prediction_v63,
)
from fma.v6.provenance import PROCESSED_SNAPSHOT_PATH
from tests import test_v6_3_external_prediction_runtime as runtime_support


ROLE_HOST_IDS = {
    "custody": "custodian-host",
    "registry": "registry-host",
    "evaluator": "evaluator-host",
    "promotion": "promotion-host",
}


def _key_pair() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ),
    )


def _complete_v63_chain(
    *,
    workspace: StageWorkspaceV50,
    contract: Any,
    forecast_input: Any,
    custody: Any,
    private_keys: dict[str, bytes],
    public_keys: dict[str, bytes],
) -> tuple[Any, Any]:
    runtime = run_current_model_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
    )
    registration = qualification.sign_external_prediction_registration_v63(
        private_key_pem=private_keys["registry-key"],
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        custody_hash=custody.custody_hash,
        current_model_prediction_binding_hash=runtime.binding.binding_hash,
        generator_execution_receipt_hash=(
            runtime.binding.generator_execution_receipt_hash
        ),
        s4_gate_hash=contract.s4_gate_hash,
        training_snapshot_hash=contract.processed_snapshot_hash,
        candidate_hash=contract.selected_model_identity_hash,
        prediction_artifact_hash=(
            runtime.binding.prediction_artifact_hash
        ),
        external_snapshot_hash=custody.external_snapshot_hash,
        holdout_commitment_hash=custody.holdout_commitment_hash,
        normalization_scale_commitment_hash=(
            custody.normalization_scale_commitment_hash
        ),
        target_order_hash=custody.target_order_hash,
        holdout_observation_count=custody.holdout_observation_count,
        registry_host_id=ROLE_HOST_IDS["registry"],
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        registry_key_id="registry-key",
    )
    prediction_seal = qualification.register_external_prediction_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=runtime.binding,
        registration=registration,
        trusted_public_keys=public_keys,
    )
    reservation = qualification.reserve_external_evaluation_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=runtime.binding,
        registration=registration,
        prediction_seal=prediction_seal,
        evaluator_key_id="evaluator-key",
        evaluator_host_id=ROLE_HOST_IDS["evaluator"],
        reserved_at=prediction_seal.registered_at,
    )
    evaluation = qualification.sign_external_aggregate_evaluation_v63(
        private_key_pem=private_keys["evaluator-key"],
        evaluation_id="evaluation.authority-control",
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        custody_hash=custody.custody_hash,
        registration_hash=registration.registration_hash,
        prediction_seal_hash=prediction_seal.seal_hash,
        reservation_hash=reservation.reservation_hash,
        prediction_artifact_hash=registration.prediction_artifact_hash,
        external_snapshot_hash=custody.external_snapshot_hash,
        holdout_commitment_hash=custody.holdout_commitment_hash,
        normalization_scale_commitment_hash=(
            custody.normalization_scale_commitment_hash
        ),
        target_order_hash=custody.target_order_hash,
        holdout_observation_count=custody.holdout_observation_count,
        squared_error_sum=12.0,
        target_squared_value_sum=1200.0,
        aggregate_metric_value=0.10,
        evaluator_host_id=ROLE_HOST_IDS["evaluator"],
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        evaluated_at=reservation.reserved_at,
        evaluator_key_id="evaluator-key",
    )
    promotion = qualification.sign_external_predictive_promotion_v63(
        contract=contract,
        custody=custody,
        registration=registration,
        prediction_seal=prediction_seal,
        evaluation=evaluation,
        integrity_incident_free=True,
        promotion_host_id=ROLE_HOST_IDS["promotion"],
        promotion_key_id="promotion-key",
        private_key_pem=private_keys["promotion-key"],
        decided_at=evaluation.evaluated_at,
    )
    receipt = qualification.assess_external_predictive_qualification_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        prediction_binding=runtime.binding,
        registration=registration,
        prediction_seal=prediction_seal,
        reservation=reservation,
        evaluation=evaluation,
        promotion=promotion,
        trusted_public_keys=public_keys,
    )
    assert receipt.status == "EXTERNALLY_QUALIFIED"
    return runtime, receipt


@pytest.fixture
def authority_control(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Build one local mechanism control, not an external qualification run."""

    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("v64-authority-control")
    workspace = runtime_support._new_workspace(root / "task")
    _, _, _, summary = runtime_support._install_real_ode_model(workspace)
    private_keys, public_keys = runtime_support._keys()
    monkeypatch.setattr(
        qualification,
        "scientific_closure_summary_v62",
        lambda _workspace: summary,
    )
    monkeypatch.setattr(
        StageWorkspaceV50,
        "current_gate",
        lambda _workspace, stage: runtime_support.GATES.get(stage),
    )
    contract = (
        qualification.freeze_predictive_external_qualification_contract_v63(
            workspace=workspace,
            qualification_id="qual.prediction-runtime",
            task_id=workspace.spec.workspace_id,
            maximum_metric_value=0.20,
            minimum_external_observation_count=12,
            coordinator_host_id="coordinator-host",
            generator_host_id="generator-host",
            custody_key_id="custody-key",
            registry_key_id="registry-key",
            evaluator_key_id="evaluator-key",
            promotion_key_id="promotion-key",
            trusted_public_keys=public_keys,
            frozen_at=runtime_support.T0,
        )
    )
    root_private, root_public = _key_pair()
    attester_private, attester_public = _key_pair()
    manifest = sign_external_authority_manifest_v64(
        private_key_pem=root_private,
        manifest_id="manifest.authority-control",
        trust_domain_id="trust.production-control",
        qualification_id=contract.qualification_id,
        task_id=contract.task_id,
        v63_contract_hash=contract.contract_hash,
        authority_key_fingerprints=(
            contract.trusted_authority_key_fingerprints
        ),
        host_attester_key_id="host-attester-key",
        host_attester_key_fingerprint=(
            qualification.external_qualification_key_fingerprint_v63(
                attester_public
            )
        ),
        coordinator_identity=DeploymentIdentityV64(
            v63_host_id=contract.coordinator_host_id,
            host_identity_commitment=sha256_value(
                {"host": "coordinator-physical"}
            ),
            execution_boundary="boundary.coordinator",
            management_domain="domain.coordinator",
        ),
        generator_identity=DeploymentIdentityV64(
            v63_host_id=contract.generator_host_id,
            host_identity_commitment=sha256_value(
                {"host": "generator-physical"}
            ),
            execution_boundary="boundary.generator",
            management_domain="domain.generator",
        ),
        issued_at=contract.frozen_at + timedelta(seconds=30),
        valid_from=contract.frozen_at,
        valid_until=contract.frozen_at + timedelta(days=3650),
        root_key_id="root-key",
    )
    register_external_authority_manifest_v64(
        workspace=workspace,
        manifest=manifest,
    )
    target_ids = [f"target.{index:02d}" for index in range(12)]
    forecast_input = qualification.commit_external_forecast_input_v63(
        workspace=workspace,
        contract=contract,
        target_ids=target_ids,
        forecast_times=[36.0 + index for index in range(12)],
        frozen_at=contract.frozen_at + timedelta(minutes=1),
    )
    custody = qualification.sign_external_evidence_custody_v63(
        private_key_pem=private_keys["custody-key"],
        qualification_id=contract.qualification_id,
        contract_hash=contract.contract_hash,
        local_context_hash=contract.local_context_hash,
        v62_report_hash=contract.v62_report_hash,
        external_snapshot_hash=forecast_input.input_hash,
        holdout_commitment_hash=sha256_value({"private": "holdout"}),
        normalization_scale_commitment_hash=sha256_value(
            {
                "holdout_observation_count": 12,
                "target_squared_value_sum": 1200.0,
            }
        ),
        target_order_hash=forecast_input.target_order_hash,
        holdout_observation_count=12,
        fixture_only=False,
        measurement_protocol_hash=sha256_value(
            {"measurement": "protocol"}
        ),
        measurement_review_hash=sha256_value(
            {"review": "independent"}
        ),
        external_environment_hash=sha256_value(
            {"environment": "external"}
        ),
        strict_unseen_verified=True,
        independent_measurement_review_passed=True,
        external_environment_verified=True,
        holdout_frozen_before_prediction=True,
        custodian_host_id=ROLE_HOST_IDS["custody"],
        coordinator_host_id=contract.coordinator_host_id,
        generator_host_id=contract.generator_host_id,
        attested_at=contract.frozen_at + timedelta(minutes=2),
        custody_key_id="custody-key",
    )
    admission = qualification.admit_external_evidence_custody_v63(
        workspace=workspace,
        contract=contract,
        custody=custody,
        trusted_public_keys=public_keys,
    )
    assert admission.status == "VERIFIED"
    runtime, v63_receipt = _complete_v63_chain(
        workspace=workspace,
        contract=contract,
        forecast_input=forecast_input,
        custody=custody,
        private_keys=private_keys,
        public_keys=public_keys,
    )
    promotion_payload = workspace._artifact_payload_by_hash(
        v63_receipt.authority_artifact_hashes["promotion"]
    )
    assert isinstance(promotion_payload, dict)
    promotion_decided_at = datetime.fromisoformat(
        promotion_payload["decided_at"]
    )
    assessment_time = datetime.now(timezone.utc)
    snapshot = sign_external_authority_revocations_v64(
        private_key_pem=root_private,
        snapshot_id="revocations.epoch-two",
        trust_domain_id=manifest.trust_domain_id,
        epoch=2,
        previous_snapshot_hash=sha256_value(
            {"revocations": "epoch-one"}
        ),
        revoked_manifest_hashes=[],
        revoked_key_fingerprints=[],
        revoked_host_identity_commitments=[],
        effective_at=promotion_decided_at,
        valid_until=manifest.valid_until,
        root_key_id="root-key",
    )
    control = {
        "workspace": workspace,
        "workspace_root": workspace.root,
        "contract": contract,
        "forecast_input": forecast_input,
        "custody": custody,
        "runtime": runtime,
        "v63_receipt": v63_receipt,
        "private_keys": private_keys,
        "public_keys": public_keys,
        "root_private": root_private,
        "root_public": root_public,
        "attester_private": attester_private,
        "attester_public": attester_public,
        "assessed_at": assessment_time,
        "manifest": manifest,
        "snapshot": snapshot,
    }
    control["attestations"] = _attestations(control, manifest=manifest)
    try:
        yield control
    finally:
        monkeypatch.undo()


def _provider(
    control: dict[str, Any],
) -> DeploymentTrustProviderV64:
    snapshot = control["snapshot"]
    return DeploymentTrustProviderV64(
        provider_id="provider.test-deployment-control",
        trust_domain_id=control["manifest"].trust_domain_id,
        root_public_keys={"root-key": control["root_public"]},
        authority_public_keys=control["public_keys"],
        host_attester_public_keys={
            "host-attester-key": control["attester_public"]
        },
        pinned_revocation_epoch=snapshot.epoch,
        pinned_revocation_snapshot_hash=snapshot.snapshot_hash,
        pinned_previous_snapshot_hash=snapshot.previous_snapshot_hash,
        pinned_revoked_manifest_hashes=snapshot.revoked_manifest_hashes,
        pinned_revoked_key_fingerprints=(
            snapshot.revoked_key_fingerprints
        ),
        pinned_revoked_host_identity_commitments=(
            snapshot.revoked_host_identity_commitments
        ),
    )


def _attestations(
    control: dict[str, Any],
    *,
    manifest: ExternalAuthorityManifestV64,
    same_host_roles: tuple[str, str] | None = None,
) -> list[ExternalRoleHostAttestationV64]:
    identities = {
        role: {
            "host_identity_commitment": sha256_value(
                {"host": f"{role}-physical"}
            ),
            "execution_boundary": f"boundary.{role}",
            "management_domain": f"domain.{role}",
        }
        for role in ROLE_HOST_IDS
    }
    if same_host_roles:
        left, right = same_host_roles
        identities[right]["host_identity_commitment"] = identities[left][
            "host_identity_commitment"
        ]
    receipt = control["v63_receipt"]
    artifact_roles = {
        "custody": "custody",
        "registry": "registration",
        "evaluator": "evaluation",
        "promotion": "promotion",
    }
    return [
        sign_external_role_host_attestation_v64(
            private_key_pem=control["attester_private"],
            attestation_id=f"attestation.{role}",
            trust_domain_id=manifest.trust_domain_id,
            manifest_hash=manifest.manifest_hash,
            qualification_id=manifest.qualification_id,
            task_id=manifest.task_id,
            v63_contract_hash=manifest.v63_contract_hash,
            role=role,
            role_key_fingerprint=manifest.authority_key_fingerprints[role],
            v63_role_evidence_hash=receipt.authority_artifact_hashes[
                artifact_roles[role]
            ],
            v63_declared_host_id=ROLE_HOST_IDS[role],
            **identities[role],
            attested_at=control["snapshot"].effective_at,
            valid_until=control["manifest"].valid_until,
            attester_key_id=manifest.host_attester_key_id,
        )
        for role in ROLE_HOST_IDS
    ]


def _anchored_assessment(
    control: dict[str, Any],
    *,
    provider: DeploymentTrustProviderV64 | None = None,
    contract: Any | None = None,
    receipt: Any | None = None,
    manifest: ExternalAuthorityManifestV64 | None = None,
    snapshot: ExternalAuthorityRevocationSnapshotV64 | None = None,
    attestations: list[ExternalRoleHostAttestationV64] | None = None,
    workspace: StageWorkspaceV50 | None = None,
) -> Any:
    return assess_external_authority_v64(
        workspace=workspace or control["workspace"],
        contract=contract or control["contract"],
        v63_receipt=receipt or control["v63_receipt"],
        provider=provider or _provider(control),
        manifest=manifest or control["manifest"],
        revocation_snapshot=snapshot or control["snapshot"],
        host_attestations=(
            control["attestations"]
            if attestations is None
            else attestations
        ),
        assessment_mode="anchored",
        assessed_at=control["assessed_at"],
    )


def _resign_manifest(
    control: dict[str, Any],
    *,
    private_key: bytes | None = None,
    **updates: object,
) -> ExternalAuthorityManifestV64:
    data = control["manifest"].model_dump(
        mode="python",
        exclude={"signature_base64", "manifest_hash"},
    )
    data.update(updates)
    return sign_external_authority_manifest_v64(
        private_key_pem=private_key or control["root_private"],
        **data,
    )


def _revocations(
    control: dict[str, Any],
    *,
    epoch: int = 2,
    **updates: object,
) -> ExternalAuthorityRevocationSnapshotV64:
    data: dict[str, object] = {
        "snapshot_id": f"revocations.epoch-{epoch}",
        "trust_domain_id": control["manifest"].trust_domain_id,
        "epoch": epoch,
        "previous_snapshot_hash": sha256_value(
            {"revocations": f"epoch-{epoch - 1}"}
        ),
        "revoked_manifest_hashes": [],
        "revoked_key_fingerprints": [],
        "revoked_host_identity_commitments": [],
        "effective_at": control["snapshot"].effective_at,
        "valid_until": control["manifest"].valid_until,
        "root_key_id": "root-key",
    }
    data.update(updates)
    return sign_external_authority_revocations_v64(
        private_key_pem=control["root_private"],
        **data,
    )


def test_default_mode_is_unanchored_not_run(
    authority_control: dict[str, Any],
) -> None:
    receipt = assess_external_authority_v64(
        workspace=authority_control["workspace"],
        contract=authority_control["contract"],
        v63_receipt=authority_control["v63_receipt"],
        provider=_provider(authority_control),
        manifest=authority_control["manifest"],
        revocation_snapshot=authority_control["snapshot"],
        host_attestations=authority_control["attestations"],
        assessed_at=authority_control["assessed_at"],
    )

    assert receipt.status == "NOT_RUN"
    assert receipt.anchor_status == "UNANCHORED_REHEARSAL"
    assert receipt.real_world_action_authorized is False


def test_complete_local_mechanism_control_is_protocol_pass_not_run(
    authority_control: dict[str, Any],
) -> None:
    """This proves mechanism behavior only, not a real external deployment."""

    provider = _provider(authority_control)
    receipt = _anchored_assessment(
        authority_control,
        provider=provider,
    )
    event_bytes = (
        authority_control["workspace"].graph.store.event_path.read_bytes()
    )
    replay = verify_external_authority_v64(
        workspace=authority_control["workspace"],
        receipt=receipt,
        provider=provider,
    )

    assert receipt.status == "NOT_RUN"
    assert receipt.anchor_status == "UNANCHORED_REHEARSAL"
    assert receipt.anchor_protocol_status == "PASS"
    assert receipt.checks["deployment_anchor_current"] is False
    assert receipt.real_world_action_authorized is False
    assert replay.status == "PASS"
    assert (
        authority_control["workspace"].graph.store.event_path.read_bytes()
        == event_bytes
    )


def test_manifest_approved_after_forecast_input_is_rejected(
    authority_control: dict[str, Any],
) -> None:
    late_manifest = _resign_manifest(
        authority_control,
        issued_at=authority_control["forecast_input"].frozen_at
        + timedelta(seconds=1),
    )

    receipt = _anchored_assessment(
        authority_control,
        manifest=late_manifest,
        attestations=_attestations(
            authority_control,
            manifest=late_manifest,
        ),
    )

    assert receipt.status == "REJECTED"
    assert (
        receipt.checks["trust_approval_frozen_before_external_evidence"]
        is False
    )


def test_rogue_self_signed_root_is_rejected(
    authority_control: dict[str, Any],
) -> None:
    rogue_private, _ = _key_pair()
    rogue_manifest = _resign_manifest(
        authority_control,
        private_key=rogue_private,
    )
    attestations = _attestations(
        authority_control,
        manifest=rogue_manifest,
    )

    receipt = _anchored_assessment(
        authority_control,
        manifest=rogue_manifest,
        attestations=attestations,
    )

    assert receipt.status == "REJECTED"
    assert receipt.checks["manifest_signature_current"] is False


def test_replacement_authority_key_is_rejected(
    authority_control: dict[str, Any],
) -> None:
    replacement_fingerprints = dict(
        authority_control["manifest"].authority_key_fingerprints
    )
    _, replacement_public = _key_pair()
    replacement_fingerprints["registry"] = (
        qualification.external_qualification_key_fingerprint_v63(
            replacement_public
        )
    )
    manifest = _resign_manifest(
        authority_control,
        authority_key_fingerprints=replacement_fingerprints,
    )
    attestations = _attestations(authority_control, manifest=manifest)

    receipt = _anchored_assessment(
        authority_control,
        manifest=manifest,
        attestations=attestations,
    )

    assert receipt.status == "REJECTED"
    assert receipt.checks["manifest_role_keys_current"] is False


def test_same_host_commitment_is_rejected(
    authority_control: dict[str, Any],
) -> None:
    attestations = _attestations(
        authority_control,
        manifest=authority_control["manifest"],
        same_host_roles=("custody", "registry"),
    )

    receipt = _anchored_assessment(
        authority_control,
        attestations=attestations,
    )

    assert receipt.status == "REJECTED"
    assert receipt.checks["external_role_isolation"] is False


def test_provider_rejects_revocation_epoch_rollback(
    authority_control: dict[str, Any],
) -> None:
    provider = _provider(authority_control)
    assert provider.accept_current_revocation_snapshot(
        snapshot=authority_control["snapshot"],
        assessed_at=datetime.now(timezone.utc),
    )
    old_snapshot = _revocations(authority_control, epoch=1)

    receipt = _anchored_assessment(
        authority_control,
        provider=provider,
        snapshot=old_snapshot,
    )

    assert receipt.status == "REJECTED"
    assert receipt.checks["revocation_snapshot_current"] is False


@pytest.mark.parametrize("revoked_kind", ["manifest", "key", "host"])
def test_revoked_authority_material_is_rejected(
    authority_control: dict[str, Any],
    revoked_kind: str,
) -> None:
    updates: dict[str, object] = {}
    if revoked_kind == "manifest":
        updates["revoked_manifest_hashes"] = [
            authority_control["manifest"].manifest_hash
        ]
    elif revoked_kind == "key":
        updates["revoked_key_fingerprints"] = [
            authority_control[
                "manifest"
            ].authority_key_fingerprints["evaluator"]
        ]
    else:
        updates["revoked_host_identity_commitments"] = [
            authority_control["attestations"][
                0
            ].host_identity_commitment
        ]
    snapshot = _revocations(authority_control, **updates)

    receipt = _anchored_assessment(
        authority_control,
        snapshot=snapshot,
    )

    assert receipt.status == "REJECTED"
    assert receipt.checks["no_revoked_authority_material"] is False


def test_expired_manifest_and_missing_attestation_are_rejected(
    authority_control: dict[str, Any],
) -> None:
    expired_manifest = _resign_manifest(
        authority_control,
        valid_until=authority_control["assessed_at"] - timedelta(seconds=1),
    )
    expired_receipt = _anchored_assessment(
        authority_control,
        manifest=expired_manifest,
        attestations=_attestations(
            authority_control,
            manifest=expired_manifest,
        ),
    )
    missing_receipt = _anchored_assessment(
        authority_control,
        attestations=authority_control["attestations"][:-1],
    )

    assert expired_receipt.status == "REJECTED"
    assert expired_receipt.checks["manifest_signature_current"] is False
    assert missing_receipt.status == "REJECTED"
    assert missing_receipt.checks["host_attestation_set_complete"] is False


def test_tampered_v63_receipt_and_action_true_are_rejected(
    authority_control: dict[str, Any],
) -> None:
    tampered = authority_control["v63_receipt"].model_copy(
        update={"contract_hash": sha256_value({"tampered": "contract"})}
    )
    action_true = authority_control["v63_receipt"].model_copy(
        update={"real_world_action_authorized": True}
    )

    tampered_result = _anchored_assessment(
        authority_control,
        receipt=tampered,
    )
    action_result = _anchored_assessment(
        authority_control,
        receipt=action_true,
    )

    assert tampered_result.status == "REJECTED"
    assert tampered_result.checks["v63_exact_binding"] is False
    assert action_result.status == "REJECTED"
    assert action_result.checks["v63_action_forbidden"] is False
    assert action_result.real_world_action_authorized is False


def test_missing_current_model_artifact_cannot_be_anchored(
    authority_control: dict[str, Any],
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "missing-model"
    shutil.copytree(authority_control["workspace_root"], copied_root)
    (copied_root / PROCESSED_SNAPSHOT_PATH).unlink()
    workspace = StageWorkspaceV50.open_existing(
        copied_root,
        authority_key=runtime_support.AUTHORITY_KEY,
        authority_key_id=runtime_support.AUTHORITY_KEY_ID,
    )

    receipt = _anchored_assessment(
        authority_control,
        workspace=workspace,
    )

    assert receipt.status == "REJECTED"
    assert receipt.checks["current_model_prediction_recomputed"] is False
