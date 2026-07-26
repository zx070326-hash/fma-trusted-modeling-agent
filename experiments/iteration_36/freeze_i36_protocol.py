"""Create the prospective I36 protocol before any seed or source access."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path

from fma.hashing import canonical_json, sha256_value
from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import (
    ProspectiveCampaignProtocolV55,
    PublicEligibilitySettingsV55,
)
from fma.v5_5.world_bank_custodian import WorldBankSelectionSpecV55
from fma.v5_6.public_hybrid_campaign import load_hybrid_thresholds_v56
from fma.v5_6.unseen_source import (
    PriorSourceExclusionV56,
    UnseenSourceRegistryV56,
)
from fma.v5_7.public_adaptive_campaign import (
    AdaptiveCampaignProtocolV57,
    load_adaptive_thresholds_v57,
)


ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--frozen-at", required=True)
    parser.add_argument("--regression-stdout", required=True)
    parser.add_argument("--regression-stderr", required=True)
    parser.add_argument("--regression-duration-seconds", type=float, required=True)
    parser.add_argument("--regression-passed", type=int, required=True)
    return parser


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _source_hash(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _assert_commit_source(
    implementation_commit: str,
    relative: str,
) -> str:
    current = _source_hash(relative)
    committed = subprocess.run(
        ["git", "show", f"{implementation_commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != current:
        raise ValueError(f"working source differs from commit: {relative}")
    return current


def _protocol_report(
    *,
    implementation_commit: str,
    frozen_at: datetime,
    adaptive_protocol: AdaptiveCampaignProtocolV57,
    registry: UnseenSourceRegistryV56,
) -> str:
    return "\n".join(
        [
            "# Iteration 36 prospective adaptive positive-series protocol",
            "",
            "## Question",
            "",
            (
                "On one newly selected complete, positive, annual World Bank "
                "scalar series, can the frozen V5.7 graph retain a passing "
                "autonomous-ODE branch or recover through a guarded stationary "
                "log-growth branch, pass public L0-L4, and register four future "
                "predictions without accessing their values?"
            ),
            "",
            "## Freeze boundary",
            "",
            f"- Implementation commit: `{implementation_commit}`",
            f"- Frozen at: `{frozen_at.isoformat()}`",
            f"- Adaptive protocol: `{adaptive_protocol.protocol_hash}`",
            f"- Source exclusion registry: `{registry.registry_hash}`",
            "- I34 and I35 are excluded by source identity, response bytes, and provenance-record hash.",
            "- The source-selection universe, V5.2 source thresholds, V5.6 primary thresholds, V5.7 adaptive thresholds, candidates, gate, and prediction rules are frozen.",
            "- The selection seed and all custody keys must be generated only after this directory is committed.",
            "- Source selection is a secret HMAC permutation followed only by frozen data-quality eligibility; no model score or source preference is used.",
            "",
            "## Frozen graph",
            "",
            "1. Evaluate the V5.6 four-family autonomous-ODE graph.",
            "2. Retain that branch only if its public L1-L4 all pass.",
            "3. Otherwise evaluate exactly `log_random_walk_drift` and `log_growth_ar1` on log increments.",
            "4. Admit a recovery only if all frozen fit, improvement, stationarity, stability, break, outlier, interval, and growth-plausibility guards pass.",
            "5. If no branch is admissible, emit diagnostic predictions but `ABSTAIN`; diagnostic output is never registered.",
            "",
            "## Gate and stop rules",
            "",
            "- Public `ELIGIBLE` requires a real non-fixture task and PASS at every L0-L4 level.",
            "- `ELIGIBLE` registers exactly four predictions, but private evaluation remains `BLOCKED_EXTERNAL_HOST_NOT_RUN` on this same host.",
            "- Any public failure yields `ABSTAIN`, provisional predictions, and `NOT_AUTHORIZED_NOT_RUN`.",
            "- One source draw, one public modelling attempt, and at most one private evaluation are allowed.",
            "- No threshold change, candidate addition, task replacement, or retry is allowed after seeing I36 public results.",
            "",
            "## Claim limits",
            "",
            "This run can establish same-host real-data public modelling evidence only. It cannot establish external-host independence, causal identification, general mathematical-modelling capability, scientific qualification, or real-world action authority.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"freeze output already exists: {output}")
    frozen_at = datetime.fromisoformat(args.frozen_at)
    if frozen_at.utcoffset() is None:
        raise ValueError("frozen-at must be timezone-aware")
    implementation_commit = args.implementation_commit
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if head != implementation_commit:
        raise ValueError("implementation commit must equal current HEAD")

    source_paths = {
        "world_bank_custodian": "fma/v5_5/world_bank_custodian.py",
        "unseen_source_core": "fma/v5_6/unseen_source.py",
        "primary_adapter": "fma/v5_6/hybrid_ode.py",
        "unseen_source_adapter": "fma/v5_7/unseen_source.py",
        "adaptive_adapter": "fma/v5_7/adaptive_positive_series.py",
        "public_runner": "fma/v5_7/public_adaptive_campaign.py",
    }
    source_hashes = {
        name: _assert_commit_source(implementation_commit, relative)
        for name, relative in source_paths.items()
    }
    v52 = ODEThresholdsV52.seal()
    primary = load_hybrid_thresholds_v56(
        ROOT / "V5_6_HYBRID_THRESHOLDS.json"
    )
    adaptive = load_adaptive_thresholds_v57(
        ROOT / "V5_7_ADAPTIVE_THRESHOLDS.json"
    )
    v55 = ProspectiveCampaignProtocolV55.seal(
        protocol_id="i36-real-world-bank-adaptive-series-protocol",
        baseline_id="persistence_last_value",
        candidate_families=[
            "constant",
            "exponential",
            "gompertz",
            "logistic",
        ],
        maximum_candidate_search_count=16,
        public_eligibility=PublicEligibilitySettingsV55(
            expected_horizons=[1, 2, 3, 4],
            minimum_origin_count=12,
            contiguous_time_block_count=3,
            recent_origin_count=4,
            bootstrap_replicates=8192,
            bootstrap_block_length=4,
            multiplicity_correction_count=16,
            bootstrap_seed=65537,
        ),
        frozen_at=frozen_at,
    )
    registry = UnseenSourceRegistryV56.seal(
        registry_id="i36-real-double-source-exclusion-registry",
        required_prior_campaign_ids=["i34", "i35"],
        exclusions=[
            PriorSourceExclusionV56(
                campaign_id="i34",
                source_identity_hash=(
                    "e4dfd716a3969b763e30471cd0f34233798c73401fca475989c4ac946582b2ea"
                ),
                source_artifact_sha256=(
                    "45be724f7ae6019c97793e4add15d5cb09f332a717ad499e0ca507e122fea75d"
                ),
                source_provenance_record_hash=(
                    "06f4c5c14d12667e3b35a4089551d618c7aa2c3e5af01f87f3bed44ed489630f"
                ),
            ),
            PriorSourceExclusionV56(
                campaign_id="i35",
                source_identity_hash=(
                    "8e19ecb393fb7f77403e5a0f6ee54243d72754a1fe201a78236d7ff5feda283c"
                ),
                source_artifact_sha256=(
                    "2b4af5adf57f1b80874c8290b7d48261cdc94f5af8c6f806847b34b4b805f987"
                ),
                source_provenance_record_hash=(
                    "4c66370bc440e290eaaf6e4e636918626696371b8f9d0861f404659fd9d67065"
                ),
            ),
        ],
        fixture_only=False,
        frozen_at=frozen_at,
    )
    selection = WorldBankSelectionSpecV55.seal(
        selection_spec_id="i36-world-bank-double-unseen-source-selection",
        protocol_hash=v55.protocol_hash,
        country_codes=[
            "ALB",
            "ARG",
            "BGD",
            "BRA",
            "DZA",
            "EGY",
            "IDN",
            "IND",
            "KEN",
            "MAR",
            "MEX",
            "MYS",
            "NGA",
            "PAK",
            "PHL",
            "THA",
            "TUR",
            "VNM",
            "ZAF",
        ],
        indicator_codes=[
            "AG.LND.FRST.K2",
            "NE.EXP.GNFS.KD",
            "NY.GDP.MKTP.KD",
            "SP.POP.1564.TO",
            "SP.URB.TOTL",
        ],
        public_start_year=1990,
        public_end_year=2017,
        private_end_year=2021,
        ode_threshold_hash=v52.threshold_hash,
        prior_campaign_exclusion_hashes=registry.exclusion_hashes(),
        fixture_only=False,
        frozen_at=frozen_at,
    )
    adaptive_protocol = AdaptiveCampaignProtocolV57.seal(
        protocol_id="i36-real-adaptive-positive-series-public-protocol",
        implementation_source_commit=implementation_commit,
        v55_protocol_hash=v55.protocol_hash,
        source_registry_hash=registry.registry_hash,
        source_selection_spec_hash=selection.selection_spec_hash,
        source_ode_threshold_hash=v52.threshold_hash,
        primary_threshold_hash=primary.threshold_hash,
        adaptive_threshold_hash=adaptive.threshold_hash,
        primary_adapter_source_sha256=source_hashes["primary_adapter"],
        adaptive_adapter_source_sha256=source_hashes["adaptive_adapter"],
        unseen_source_adapter_source_sha256=source_hashes[
            "unseen_source_adapter"
        ],
        unseen_source_core_source_sha256=source_hashes[
            "unseen_source_core"
        ],
        world_bank_custodian_source_sha256=source_hashes[
            "world_bank_custodian"
        ],
        public_runner_source_sha256=source_hashes["public_runner"],
        primary_candidate_families=[
            "constant",
            "exponential",
            "gompertz",
            "logistic",
        ],
        primary_residual_modes=["ar1_residual", "trend_only"],
        recovery_growth_modes=[
            "log_growth_ar1",
            "log_random_walk_drift",
        ],
        required_public_levels=["L0", "L1", "L2", "L3", "L4"],
        frozen_at=frozen_at,
    )

    stdout = Path(args.regression_stdout).read_bytes()
    stderr = Path(args.regression_stderr).read_bytes()
    regression = {
        "schema_version": "5.7-local-full-regression-receipt",
        "source_commit": implementation_commit,
        "command": "python -m pytest",
        "exit_code": 0,
        "passed": args.regression_passed,
        "duration_seconds": args.regression_duration_seconds,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_size_bytes": len(stdout),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_size_bytes": len(stderr),
        "same_host_local_only": True,
        "external_qualification_evidence": False,
        "scientific_qualification_granted": False,
    }
    regression["receipt_hash"] = sha256_value(regression)

    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "ADAPTIVE_CAMPAIGN_PROTOCOL_V57.json": _json_bytes(
            adaptive_protocol
        ),
        "ADAPTIVE_THRESHOLDS_V57.json": _json_bytes(adaptive),
        "FULL_REGRESSION_RECEIPT_V57.json": _json_bytes(regression),
        "FULL_REGRESSION_STDERR.log": stderr,
        "FULL_REGRESSION_STDOUT.log": stdout,
        "ODE_THRESHOLDS_V52.json": _json_bytes(v52),
        "PRIMARY_THRESHOLDS_V56.json": _json_bytes(primary),
        "PROSPECTIVE_CAMPAIGN_PROTOCOL_V55.json": _json_bytes(v55),
        "SCIENTIFIC_PROTOCOL.md": _protocol_report(
            implementation_commit=implementation_commit,
            frozen_at=frozen_at,
            adaptive_protocol=adaptive_protocol,
            registry=registry,
        ).encode("utf-8"),
        "SOURCE_SELECTION_SPEC_V55.json": _json_bytes(selection),
        "UNSEEN_SOURCE_REGISTRY_V56.json": _json_bytes(registry),
    }
    for name, payload in sorted(artifacts.items()):
        _write_new(output / name, payload)
    manifest = {
        "schema_version": "5.7-i36-protocol-freeze-manifest",
        "implementation_source_commit": implementation_commit,
        "frozen_at": frozen_at.isoformat(),
        "files": [
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(output.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ],
        "selection_seed_generated": False,
        "real_source_requested": False,
        "private_target_values_accessed": False,
        "external_host_established": False,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }
    manifest["manifest_hash"] = sha256_value(manifest)
    _write_new(
        output / "I36_FREEZE_MANIFEST_V57.json",
        _json_bytes(manifest),
    )
    print(
        canonical_json(
            {
                "protocol_hash": adaptive_protocol.protocol_hash,
                "registry_hash": registry.registry_hash,
                "selection_spec_hash": selection.selection_spec_hash,
                "manifest_hash": manifest["manifest_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
