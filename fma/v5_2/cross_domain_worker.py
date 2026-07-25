"""Deterministic fresh-process worker for V5.2 ablation plumbing tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from fma.hashing import sha256_value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--mechanism-enabled", choices=["0", "1"], required=True)
    parser.add_argument("--nuisance-identity-hash", required=True)
    parser.add_argument("--fixture-seed", type=int, required=True)
    args = parser.parse_args()
    enabled = args.mechanism_enabled == "1"
    base_material = {
        "domain_id": args.domain_id,
        "case_id": args.case_id,
        "repetition": args.repetition,
        "nuisance_identity_hash": args.nuisance_identity_hash,
        "fixture_seed": args.fixture_seed,
    }
    stable = int(sha256_value(base_material)[:12], 16) / float(16**12)
    base_score = 0.35 + 0.25 * stable
    effect = 0.08 + 0.02 * (
        int(hashlib.sha256(args.domain_id.encode()).hexdigest()[:8], 16)
        / float(16**8)
    )
    score = base_score + (effect if enabled else 0.0)
    output = {
        "base_material": base_material,
        "mechanism_id": args.mechanism_id,
        "mechanism_enabled": enabled,
        "observed_mechanism_event": enabled,
        "score": score,
    }
    print(
        json.dumps(
            {
                "process_id": os.getpid(),
                "observed_mechanism_event": enabled,
                "score": score,
                "output_artifact_hash": sha256_value(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
