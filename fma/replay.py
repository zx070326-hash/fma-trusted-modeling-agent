from __future__ import annotations

import argparse
import json
from pathlib import Path

from .optimization import compile_optimization, solve_compiled
from .schemas import OptimizationModelIR, ProblemContract
from .validation import REQUIRED_HARD_CHECKS, validate_candidate


def replay_bundle(bundle: dict[str, object]) -> dict[str, object]:
    contract = ProblemContract.model_validate(bundle["contract"])
    ir = OptimizationModelIR.model_validate(bundle["model_ir"])
    compiled = compile_optimization(ir)
    solution = solve_compiled(compiled)
    validation = validate_candidate(contract, ir, compiled.certificate, solution)
    replay_checks = {
        name: validation.checks[name].status
        for name in sorted(REQUIRED_HARD_CHECKS - {"reproducibility"})
    }
    return {
        "replay_schema": "1.0",
        "compiler_certificate": compiled.certificate.model_dump(mode="json"),
        "solution": solution.model_dump(mode="json"),
        "validation_checks": replay_checks,
        "validation_violations": validation.violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-process deterministic replay")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    result = replay_bundle(bundle)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

