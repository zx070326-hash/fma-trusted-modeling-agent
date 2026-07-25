from __future__ import annotations

from pathlib import Path

import pytest

from fma.v5.external_harness import (
    ExternalHarnessV50,
    HarnessProtocolError,
    MechanismArmV50,
    PredictionDocumentV50,
    PredictionPointV50,
    PrivateCaseCapsuleV50,
    PrivateTargetV50,
    PublicCaseSpecV50,
)


CANARY = "H0_PRIVATE_CANARY_DO_NOT_PROJECT_7f3a"


def _prepared(
    tmp_path: Path,
) -> tuple[
    ExternalHarnessV50,
    Path,
    PublicCaseSpecV50,
    PrivateCaseCapsuleV50,
]:
    public_case = PublicCaseSpecV50.seal(
        case_id="case.alpha",
        title="Synthetic hidden-vector fixture",
        objective="Predict the values for the public target identifiers.",
        public_payload={"target_ids": ["target.a", "target.b"]},
        supported_mechanisms=["GATE"],
    )
    private_capsule = PrivateCaseCapsuleV50.seal(
        case_id=public_case.case_id,
        public_case_hash=public_case.case_hash,
        holdout=[
            PrivateTargetV50(target_id="target.a", value=987654.321),
            PrivateTargetV50(target_id="target.b", value=-123456.789),
        ],
        quality_scale=10.0,
        secrecy_canary=CANARY,
    )
    harness = ExternalHarnessV50(tmp_path / "external-control")
    workspace = tmp_path / "task-workspace"
    harness.prepare_case(public_case, private_capsule, workspace)
    return harness, workspace, public_case, private_capsule


def _write_prediction(
    workspace: Path, first: float = 987654.321, second: float = -123456.789
) -> None:
    prediction = PredictionDocumentV50(
        case_id="case.alpha",
        predictions=[
            PredictionPointV50(target_id="target.a", value=first),
            PredictionPointV50(target_id="target.b", value=second),
        ],
    )
    (workspace / "predictions" / "registered.json").write_text(
        prediction.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _workspace_bytes(workspace: Path) -> bytes:
    return b"\n".join(
        path.read_bytes() for path in workspace.rglob("*") if path.is_file()
    )


def test_prepare_uses_fresh_external_workspace_and_projects_no_private_data(
    tmp_path: Path,
) -> None:
    harness, workspace, _, _ = _prepared(tmp_path)

    projected = _workspace_bytes(workspace)
    assert CANARY.encode() not in projected
    assert b"987654.321" not in projected
    assert b"-123456.789" not in projected
    assert b"5.0-private" not in projected
    assert not (workspace / "private").exists()

    receipt = harness._load_case("case.alpha")[0]
    assert receipt.secrecy_mode == "logical_projection_plus_canary"
    assert receipt.host_secrecy_attested is False
    assert receipt.capability_claim_permitted is False
    assert receipt.canary_absent_at_preparation is True
    assert harness.verify_event_chain()

    existing_empty_workspace = tmp_path / "already-exists"
    existing_empty_workspace.mkdir()
    public_case = PublicCaseSpecV50.seal(
        case_id="case.beta",
        title="Second fixture",
        objective="Demonstrate the fresh-workspace refusal.",
        public_payload={},
        supported_mechanisms=[],
    )
    private_capsule = PrivateCaseCapsuleV50.seal(
        case_id="case.beta",
        public_case_hash=public_case.case_hash,
        holdout=[PrivateTargetV50(target_id="target.a", value=1.0)],
        secrecy_canary="ANOTHER_PRIVATE_CANARY_12345",
    )
    with pytest.raises(HarnessProtocolError, match="fresh workspace"):
        harness.prepare_case(
            public_case, private_capsule, existing_empty_workspace
        )


def test_holdout_and_score_are_blocked_until_first_legal_registration(
    tmp_path: Path,
) -> None:
    harness, workspace, _, _ = _prepared(tmp_path)

    with pytest.raises(HarnessProtocolError, match="prior registration"):
        harness.reveal_holdout("case.alpha")
    with pytest.raises(HarnessProtocolError, match="prior registration"):
        harness.score("case.alpha")

    prediction_path = workspace / "predictions" / "registered.json"
    prediction_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(HarnessProtocolError, match="invalid"):
        harness.register_prediction("case.alpha")

    _write_prediction(workspace)
    registration = harness.register_prediction("case.alpha")
    snapshot = harness.root / registration.snapshot_relative_path
    assert snapshot.is_file()
    assert harness.root in snapshot.parents
    assert workspace not in snapshot.parents

    reveal = harness.reveal_holdout("case.alpha")
    assert reveal.registration_hash == registration.registration_hash
    assert [point.value for point in reveal.holdout] == [
        987654.321,
        -123456.789,
    ]
    with pytest.raises(HarnessProtocolError, match="immutable and unique"):
        harness.register_prediction("case.alpha")


def test_score_reads_frozen_snapshot_and_tamper_forces_zero_quality(
    tmp_path: Path,
) -> None:
    harness, workspace, _, _ = _prepared(tmp_path)
    _write_prediction(workspace, first=987653.321, second=-123455.789)
    registration = harness.register_prediction("case.alpha")

    clean = harness.score("case.alpha")
    assert clean.integrity_valid is True
    assert clean.raw_mae == pytest.approx(1.0)
    assert clean.quality_score == pytest.approx(0.9)
    assert clean.scored_snapshot_bytes_hash == registration.snapshot_bytes_hash

    _write_prediction(workspace, first=0.0, second=0.0)
    tampered = harness.score("case.alpha")
    assert tampered.score_source == "frozen_content_addressed_snapshot"
    assert tampered.raw_mae == pytest.approx(1.0)
    assert tampered.source_mutated_after_registration is True
    assert tampered.integrity_valid is False
    assert tampered.quality_score == 0.0
    assert tampered.capability_claim_permitted is False
    assert "prediction_source_mutated_after_registration" in tampered.reasons


def test_canary_leak_after_registration_invalidates_score(tmp_path: Path) -> None:
    harness, workspace, _, _ = _prepared(tmp_path)
    _write_prediction(workspace)
    harness.register_prediction("case.alpha")

    leak_path = workspace / "results" / "leak.txt"
    leak_path.parent.mkdir()
    leak_path.write_text(CANARY, encoding="utf-8")
    report = harness.score("case.alpha")

    assert report.canary_leak_detected is True
    assert report.integrity_valid is False
    assert report.quality_score == 0.0
    assert report.host_secrecy_attested is False


def test_unsupported_and_no_op_mechanisms_cannot_form_valid_ablation(
    tmp_path: Path,
) -> None:
    harness, _, _, _ = _prepared(tmp_path)
    supported_control = MechanismArmV50(
        mechanism_id="GATE",
        enabled=False,
        implementation_hash="1" * 64,
        behavior_fingerprint="2" * 64,
        run_receipt_hash="3" * 64,
    )
    supported_treatment = MechanismArmV50(
        mechanism_id="GATE",
        enabled=True,
        implementation_hash="1" * 64,
        behavior_fingerprint="5" * 64,
        run_receipt_hash="6" * 64,
    )
    declared_only = harness.assess_ablation(
        "case.alpha", supported_control, supported_treatment
    )
    assert declared_only.valid_ablation is False
    assert (
        "mechanism_runtime_not_bound_to_receipts"
        in declared_only.reasons
    )
    assert declared_only.capability_claim_permitted is False

    unsupported = harness.assess_ablation(
        "case.alpha",
        supported_control.model_copy(update={"mechanism_id": "UNKNOWN"}),
        supported_treatment.model_copy(update={"mechanism_id": "UNKNOWN"}),
    )
    assert unsupported.valid_ablation is False
    assert unsupported.unsupported_mechanism is True

    no_op = harness.assess_ablation(
        "case.alpha",
        supported_control,
        supported_treatment.model_copy(
            update={"behavior_fingerprint": supported_control.behavior_fingerprint}
        ),
    )
    assert no_op.valid_ablation is False
    assert no_op.no_op_detected is True
    assert "behavior_fingerprint_unchanged_no_op" in no_op.reasons


def test_event_chain_detects_mutation_and_fails_closed(tmp_path: Path) -> None:
    harness, workspace, _, _ = _prepared(tmp_path)
    _write_prediction(workspace)
    original = harness.event_path.read_text(encoding="utf-8")
    harness.event_path.write_text(
        original.replace("case_prepared", "prediction_scored", 1),
        encoding="utf-8",
    )

    assert harness.verify_event_chain() is False
    with pytest.raises(HarnessProtocolError, match="event chain"):
        harness.register_prediction("case.alpha")
