"""Materialize the frozen I36 V5.7 forecast plan before model execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from fma.hashing import canonical_json
from fma.v5_6.public_hybrid_campaign import load_hybrid_thresholds_v56
from fma.v5_7.public_adaptive_campaign import (
    AdaptiveCampaignProtocolV57,
    load_adaptive_thresholds_v57,
    materialize_adaptive_forecast_plan_v57,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unseen-campaign", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--primary-thresholds", required=True)
    parser.add_argument("--adaptive-thresholds", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"forecast plan already exists: {output}")
    protocol = AdaptiveCampaignProtocolV57.model_validate_json(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    primary = load_hybrid_thresholds_v56(Path(args.primary_thresholds))
    adaptive = load_adaptive_thresholds_v57(Path(args.adaptive_thresholds))
    plan = materialize_adaptive_forecast_plan_v57(
        unseen_campaign_dir=Path(args.unseen_campaign),
        protocol=protocol,
        primary_thresholds=primary,
        adaptive_thresholds=adaptive,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(plan) + "\n")
    print(canonical_json(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
