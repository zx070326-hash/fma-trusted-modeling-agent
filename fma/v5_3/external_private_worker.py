"""Separate-host CLI for one V5.3 private evaluation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from fma.hashing import canonical_json
from fma.v5.external_harness import (
    PredictionDocumentV50,
    PrivateCaseCapsuleV50,
)

from .custody import PrivateScoreContractV53
from .external_private import (
    PrivateEvaluationRequestV53,
    evaluate_external_private_inputs_v53,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--score-contract", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--private-capsule", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker-host-id", required=True)
    parser.add_argument("--worker-key-id", required=True)
    parser.add_argument("--worker-private-key", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = PrivateEvaluationRequestV53.model_validate_json(
        Path(args.request).read_text(encoding="utf-8")
    )
    score_contract = PrivateScoreContractV53.model_validate_json(
        Path(args.score_contract).read_text(encoding="utf-8")
    )
    prediction_bytes = Path(args.prediction).read_bytes()
    prediction = PredictionDocumentV50.model_validate_json(prediction_bytes)
    capsule = PrivateCaseCapsuleV50.model_validate_json(
        Path(args.private_capsule).read_text(encoding="utf-8")
    )
    runner_path = Path(__file__).resolve()
    receipt = evaluate_external_private_inputs_v53(
        request=request,
        score_contract=score_contract,
        prediction=prediction,
        prediction_bytes_hash=hashlib.sha256(prediction_bytes).hexdigest(),
        capsule=capsule,
        worker_id=args.worker_id,
        worker_host_id=args.worker_host_id,
        worker_executable_hash=hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        runner_source_hash=hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        worker_key_id=args.worker_key_id,
        worker_private_key_pem=Path(args.worker_private_key).read_bytes(),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(receipt) + "\n")
    print(
        canonical_json(
            {
                "schema_version": "5.3-external-worker-output",
                "request_hash": request.request_hash,
                "worker_receipt_hash": receipt.receipt_hash,
                "quality_score": receipt.quality_score,
                "threshold_passed": receipt.threshold_passed,
                "private_values_disclosed": False,
                "per_target_feedback_disclosed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
