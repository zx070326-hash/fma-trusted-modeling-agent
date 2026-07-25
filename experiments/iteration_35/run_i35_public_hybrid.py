"""Run the single frozen I35 public V5.6 hybrid modelling attempt."""

from __future__ import annotations

import argparse
from pathlib import Path

from fma.hashing import canonical_json
from fma.v5_6.public_hybrid_campaign import (
    run_public_hybrid_campaign_v56,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseen-campaign", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--forecast-plan", required=True)
    parser.add_argument("--replay-secret", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_public_hybrid_campaign_v56(
        unseen_campaign_dir=Path(args.unseen_campaign),
        protocol_path=Path(args.protocol),
        threshold_path=Path(args.thresholds),
        forecast_plan_path=Path(args.forecast_plan),
        replay_secret_path=Path(args.replay_secret),
        output_dir=Path(args.output_dir),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
