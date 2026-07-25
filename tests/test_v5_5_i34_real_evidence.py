from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fma.hashing import sha256_value
from fma.v5_3.ode_forecast import (
    ODEForecastBundleV53,
    ODEForecastReplayReceiptV53,
)
from fma.v5_4.public_eligibility import (
    PublicEligibilityAssessmentV54,
    PublicEligibilityInputV54,
    PublicEligibilityReceiptV54,
    verify_public_eligibility_receipt_v54,
)
from fma.v5_5.public_ode_campaign import (
    PublicODECampaignResultV55,
    verify_public_launch_v55,
)
from fma.v5_5.split_custody import (
    CampaignCloseoutAuthorizationV55,
    SourceProvenanceDisclosureReceiptV55,
    SourceProvenanceRecordV55,
    verify_campaign_closeout_authorization_v55,
)


ROOT = Path(__file__).resolve().parents[1]
I34 = ROOT / "experiments" / "iteration_34"
PUBLIC = I34 / "campaign_public"
RUN = I34 / "public_run_v55"
CLOSEOUT = I34 / "closeout_public_v55"
PROVENANCE = I34 / "provenance_release_v55"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_i34_real_public_launch_and_result_are_replayable() -> None:
    launch = verify_public_launch_v55(PUBLIC)
    assert launch.snapshot.fixture_only is False
    assert launch.manifest.external_host_established is False

    result = PublicODECampaignResultV55.model_validate_json(
        (RUN / "public_ode_campaign_result_v55.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.result_hash == result.content_hash()
    assert result.public_manifest_hash == launch.manifest.manifest_hash
    assert result.rolling_selected_family == "logistic"
    assert result.v5_3_selected_family == "logistic"
    assert result.selected_family_alignment is True
    assert result.public_scientific_acceptance_verified is False
    assert result.public_gate_decision == "ABSTAIN"
    assert result.private_evaluation_status == "NOT_AUTHORIZED_NOT_RUN"
    assert result.private_evaluations_consumed == 0

    result_manifest = _load_json(RUN / "result_manifest_v55.json")
    expected_manifest_hash = sha256_value(
        {
            key: value
            for key, value in result_manifest.items()
            if key != "manifest_hash"
        }
    )
    assert result_manifest["manifest_hash"] == expected_manifest_hash
    declared = {
        str(item["path"]): item
        for item in result_manifest["files"]  # type: ignore[index]
    }
    actual = {
        path.name
        for path in RUN.iterdir()
        if path.is_file() and path.name != "result_manifest_v55.json"
    }
    assert set(declared) == actual
    for name, entry in declared.items():
        payload = (RUN / name).read_bytes()
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]

    candidate = _load_json(RUN / "candidate_evidence_v55.json")
    assert sha256_value(candidate) == result.candidate_evidence_hash
    assert candidate["selected_family"] == "logistic"
    assert candidate["candidate_search_count"] == 4
    assert candidate["graph"]["unregistered_family_introduced"] is False  # type: ignore[index]
    assert candidate["graph"]["trailing_window_variant_introduced"] is False  # type: ignore[index]
    assert candidate["graph"]["post_result_threshold_change"] is False  # type: ignore[index]

    bundle = ODEForecastBundleV53.model_validate_json(
        (RUN / "forecast_bundle_v53.json").read_text(encoding="utf-8")
    )
    assert bundle.bundle_hash == bundle.content_hash() == result.forecast_bundle_hash
    assert all(bundle.l0_checks.values())
    assert bundle.development_assessment.status == "FAIL"
    assert (
        bundle.development_assessment.checks["l3_residual_lag_bounded"]
        is False
    )
    assert bundle.final_refit.status == "PASS"
    assert bundle.scientific_acceptance is False

    replay_receipts = [
        ODEForecastReplayReceiptV53.model_validate(item)
        for item in _load_json(RUN / "replay_receipts_v53.json")
    ]
    assert len({item.process_id for item in replay_receipts}) == 2
    assert len({item.deterministic_output_hash for item in replay_receipts}) == 1
    assert [item.receipt_hash for item in replay_receipts] == (
        bundle.replay_receipt_hashes
    )


def test_i34_public_gate_and_provenance_only_closeout_verify() -> None:
    launch = verify_public_launch_v55(PUBLIC)
    evidence = PublicEligibilityInputV54.model_validate_json(
        (RUN / "public_eligibility_input_v54.json").read_text(encoding="utf-8")
    )
    assessment = PublicEligibilityAssessmentV54.model_validate_json(
        (RUN / "public_eligibility_assessment_v54.json").read_text(
            encoding="utf-8"
        )
    )
    receipt = PublicEligibilityReceiptV54.model_validate_json(
        (RUN / "public_eligibility_receipt_v54.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence.input_hash == evidence.content_hash()
    assert assessment.assessment_hash == assessment.content_hash()
    assert assessment.input_hash == evidence.input_hash
    assert assessment.decision == "ABSTAIN"
    assert assessment.checks["public_scientific_acceptance"] is False
    assert verify_public_eligibility_receipt_v54(
        assessment=assessment,
        receipt=receipt,
        authority_public_key_pem=(
            RUN / "public_eligibility_authority_public_key.pem"
        ).read_bytes(),
    )

    authorization = CampaignCloseoutAuthorizationV55.model_validate_json(
        (CLOSEOUT / "closeout_authorization_v55.json").read_text(
            encoding="utf-8"
        )
    )
    terminal_evidence = RUN / "public_eligibility_assessment_v54.json"
    assert authorization.terminal_status == "ABSTAIN"
    assert authorization.terminal_evidence_hash == hashlib.sha256(
        terminal_evidence.read_bytes()
    ).hexdigest()
    assert authorization.private_target_release_authorized is False
    assert verify_campaign_closeout_authorization_v55(
        authorization=authorization,
        protocol=launch.protocol,
        attestation=launch.split_attestation,
        closeout_public_keys={
            authorization.closeout_authority_key_id: (
                CLOSEOUT / "closeout_authority_public_key.pem"
            ).read_bytes()
        },
    )

    source = SourceProvenanceRecordV55.model_validate_json(
        (PROVENANCE / "source_provenance_record_v55.json").read_text(
            encoding="utf-8"
        )
    )
    disclosure = SourceProvenanceDisclosureReceiptV55.model_validate_json(
        (
            PROVENANCE / "source_provenance_disclosure_receipt_v55.json"
        ).read_text(encoding="utf-8")
    )
    source.assert_sealed()
    disclosure.assert_sealed()
    assert source.source_title == "Urban population - Viet Nam"
    assert source.public_period_start == "1990"
    assert source.public_period_end == "2017"
    assert source.private_period_start == "2018"
    assert source.private_period_end == "2021"
    assert disclosure.source_record_hash == source.record_hash
    assert disclosure.private_target_envelope_accessed is False
    assert disclosure.private_target_key_accessed is False
    assert disclosure.private_evaluation_performed is False
    assert disclosure.scientific_qualification_granted is False
    assert disclosure.real_world_action_authorized is False
