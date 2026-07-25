"""Fresh-process entry point for the V5.2 private evaluation protocol."""

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
from fma.v5_2.private_qualification import (
    PrivateEvaluationRequestV52,
    PrivateWorkerAuthorityV52,
    evaluate_private_inputs_v52,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--capsule", required=True)
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--worker-key-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker-host-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    request = PrivateEvaluationRequestV52.model_validate_json(
        Path(args.request).read_bytes()
    )
    prediction_bytes = Path(args.prediction).read_bytes()
    prediction = PredictionDocumentV50.model_validate_json(prediction_bytes)
    capsule = PrivateCaseCapsuleV50.model_validate_json(
        Path(args.capsule).read_bytes()
    )
    authority = PrivateWorkerAuthorityV52(
        key_id=args.worker_key_id,
        secret=Path(args.secret_file).read_bytes(),
    )
    runner_path = Path(__file__).resolve()
    receipt = evaluate_private_inputs_v52(
        request=request,
        prediction=prediction,
        prediction_bytes_hash=hashlib.sha256(prediction_bytes).hexdigest(),
        capsule=capsule,
        worker_authority=authority,
        worker_id=args.worker_id,
        worker_host_id=args.worker_host_id,
        worker_executable_hash=hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        runner_source_hash=hashlib.sha256(runner_path.read_bytes()).hexdigest(),
    )
    Path(args.output).write_text(
        canonical_json(receipt) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
