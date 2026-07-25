"""External-host CLI for one authorized encrypted V5.5 private evaluation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from fma.hashing import canonical_json
from fma.v5.external_harness import PredictionDocumentV50
from fma.v5_3.custody import (
    ExternalCustodyAttestationV53,
    PrivateScoreContractV53,
)
from fma.v5_3.external_private import PrivateEvaluationRequestV53
from fma.v5_4.public_eligibility import (
    PrivateEvaluationAuthorizationV54,
    PublicEligibilityAssessmentV54,
    PublicEligibilityContractV54,
    PublicEligibilityInputV54,
    PublicEligibilityReceiptV54,
)

from .authorized_private import (
    LegacyCustodyBridgeV55,
    assert_authorized_encrypted_private_preconditions_v55,
    claim_private_evaluation_budget_v55,
    evaluate_authorized_encrypted_private_inputs_v55,
)
from .campaign_protocol import (
    CandidateSelectionPolicyV55,
    ProspectiveCampaignProtocolV55,
    PublicLaunchBindingV55,
)
from .split_custody import (
    EncryptedCustodyEnvelopeV55,
    SplitCustodyAttestationV55,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidate-policy", required=True)
    parser.add_argument("--public-launch-binding", required=True)
    parser.add_argument("--eligibility-contract", required=True)
    parser.add_argument("--eligibility-input", required=True)
    parser.add_argument("--eligibility-assessment", required=True)
    parser.add_argument("--eligibility-receipt", required=True)
    parser.add_argument("--private-authorization", required=True)
    parser.add_argument("--eligibility-authority-public-key", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--score-contract", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--v53-custody-attestation", required=True)
    parser.add_argument("--private-target-envelope", required=True)
    parser.add_argument("--source-provenance-envelope", required=True)
    parser.add_argument("--split-custody-attestation", required=True)
    parser.add_argument("--legacy-custody-bridge", required=True)
    parser.add_argument("--custody-public-key", required=True)
    parser.add_argument("--expected-coordinator-host-id", required=True)
    parser.add_argument("--expected-generator-host-id", required=True)
    parser.add_argument("--private-target-key", required=True)
    parser.add_argument("--budget-ledger-id", required=True)
    parser.add_argument("--budget-ledger-root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker-host-id", required=True)
    parser.add_argument("--worker-key-id", required=True)
    parser.add_argument("--worker-private-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-only", action="store_true")
    return parser


def _load_model(path: str, model_type: type):
    return model_type.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # Public artifacts and ciphertexts are loaded first. The target AES key is
    # deliberately not touched until authorization, custody, and budget pass.
    protocol = _load_model(args.protocol, ProspectiveCampaignProtocolV55)
    candidate_policy = _load_model(
        args.candidate_policy,
        CandidateSelectionPolicyV55,
    )
    public_launch_binding = _load_model(
        args.public_launch_binding,
        PublicLaunchBindingV55,
    )
    eligibility_contract = _load_model(
        args.eligibility_contract,
        PublicEligibilityContractV54,
    )
    eligibility_input = _load_model(
        args.eligibility_input,
        PublicEligibilityInputV54,
    )
    eligibility_assessment = _load_model(
        args.eligibility_assessment,
        PublicEligibilityAssessmentV54,
    )
    eligibility_receipt = _load_model(
        args.eligibility_receipt,
        PublicEligibilityReceiptV54,
    )
    private_authorization = _load_model(
        args.private_authorization,
        PrivateEvaluationAuthorizationV54,
    )
    request = _load_model(args.request, PrivateEvaluationRequestV53)
    score_contract = _load_model(args.score_contract, PrivateScoreContractV53)
    prediction_bytes = Path(args.prediction).read_bytes()
    prediction = PredictionDocumentV50.model_validate_json(prediction_bytes)
    prediction_bytes_hash = hashlib.sha256(prediction_bytes).hexdigest()
    v53_custody_attestation = _load_model(
        args.v53_custody_attestation,
        ExternalCustodyAttestationV53,
    )
    private_target_envelope = _load_model(
        args.private_target_envelope,
        EncryptedCustodyEnvelopeV55,
    )
    source_provenance_envelope = _load_model(
        args.source_provenance_envelope,
        EncryptedCustodyEnvelopeV55,
    )
    split_custody_attestation = _load_model(
        args.split_custody_attestation,
        SplitCustodyAttestationV55,
    )
    legacy_custody_bridge = _load_model(
        args.legacy_custody_bridge,
        LegacyCustodyBridgeV55,
    )
    eligibility_authority_public_key_pem = Path(
        args.eligibility_authority_public_key
    ).read_bytes()
    custody_public_key_pem = Path(args.custody_public_key).read_bytes()

    assert_authorized_encrypted_private_preconditions_v55(
        protocol=protocol,
        candidate_policy=candidate_policy,
        public_launch_binding=public_launch_binding,
        eligibility_contract=eligibility_contract,
        eligibility_input=eligibility_input,
        eligibility_assessment=eligibility_assessment,
        eligibility_receipt=eligibility_receipt,
        private_authorization=private_authorization,
        eligibility_authority_public_key_pem=(
            eligibility_authority_public_key_pem
        ),
        request=request,
        score_contract=score_contract,
        prediction=prediction,
        prediction_bytes_hash=prediction_bytes_hash,
        v53_custody_attestation=v53_custody_attestation,
        private_target_envelope=private_target_envelope,
        source_provenance_envelope=source_provenance_envelope,
        split_custody_attestation=split_custody_attestation,
        legacy_custody_bridge=legacy_custody_bridge,
        custody_public_key_pem=custody_public_key_pem,
        expected_coordinator_host_id=args.expected_coordinator_host_id,
        expected_generator_host_id=args.expected_generator_host_id,
    )

    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise ValueError("authorized private output already exists")
    budget_claim, claim_path = claim_private_evaluation_budget_v55(
        ledger_root=Path(args.budget_ledger_root),
        budget_ledger_id=args.budget_ledger_id,
        request=request,
        private_authorization=private_authorization,
        private_target_envelope=private_target_envelope,
        split_custody_attestation=split_custody_attestation,
        fixture_only=args.fixture_only,
    )

    worker_private_key_pem = Path(args.worker_private_key).read_bytes()
    runner_path = Path(__file__).resolve()
    output = evaluate_authorized_encrypted_private_inputs_v55(
        protocol=protocol,
        candidate_policy=candidate_policy,
        public_launch_binding=public_launch_binding,
        eligibility_contract=eligibility_contract,
        eligibility_input=eligibility_input,
        eligibility_assessment=eligibility_assessment,
        eligibility_receipt=eligibility_receipt,
        private_authorization=private_authorization,
        eligibility_authority_public_key_pem=(
            eligibility_authority_public_key_pem
        ),
        request=request,
        score_contract=score_contract,
        prediction=prediction,
        prediction_bytes_hash=prediction_bytes_hash,
        v53_custody_attestation=v53_custody_attestation,
        private_target_envelope=private_target_envelope,
        private_target_key_path=Path(args.private_target_key),
        source_provenance_envelope=source_provenance_envelope,
        split_custody_attestation=split_custody_attestation,
        legacy_custody_bridge=legacy_custody_bridge,
        custody_public_key_pem=custody_public_key_pem,
        budget_claim=budget_claim,
        budget_claim_path=claim_path,
        expected_coordinator_host_id=args.expected_coordinator_host_id,
        expected_generator_host_id=args.expected_generator_host_id,
        worker_id=args.worker_id,
        worker_host_id=args.worker_host_id,
        worker_executable_hash=hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        runner_source_hash=hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        worker_key_id=args.worker_key_id,
        worker_private_key_pem=worker_private_key_pem,
        fixture_only=args.fixture_only,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(output) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.5-authorized-private-worker-output",
                "request_hash": request.request_hash,
                "budget_claim_hash": budget_claim.claim_hash,
                "budget_claim_path": str(claim_path),
                "output_hash": output.output_hash,
                "quality_score": output.worker_receipt_v53.quality_score,
                "threshold_passed": output.worker_receipt_v53.threshold_passed,
                "fixture_only": args.fixture_only,
                "private_values_disclosed": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
