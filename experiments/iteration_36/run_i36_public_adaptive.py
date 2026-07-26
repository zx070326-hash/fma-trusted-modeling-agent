"""Run the single frozen I36 public V5.7 adaptive modelling attempt."""

from __future__ import annotations

import argparse
from pathlib import Path

from fma.hashing import canonical_json
from fma.v5_7.public_adaptive_campaign import (
    run_public_adaptive_campaign_v57,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseen-campaign", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--primary-thresholds", required=True)
    parser.add_argument("--adaptive-thresholds", required=True)
    parser.add_argument("--forecast-plan", required=True)
    parser.add_argument("--replay-secret", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_public_adaptive_campaign_v57(
        unseen_campaign_dir=Path(args.unseen_campaign),
        protocol_path=Path(args.protocol),
        primary_threshold_path=Path(args.primary_thresholds),
        adaptive_threshold_path=Path(args.adaptive_thresholds),
        forecast_plan_path=Path(args.forecast_plan),
        replay_secret_path=Path(args.replay_secret),
        output_dir=Path(args.output_dir),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
