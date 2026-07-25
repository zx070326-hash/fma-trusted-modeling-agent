from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import BenchmarkRunner
from .benchmark_cases import build_fma_bench_v0
from .codex_driver import (
    DEFAULT_EXPECTED_CLI_VERSION,
    CodexCLIConfig,
    CodexDrivenModelingAgent,
)
from .demo import run_demo
from .examples import resource_allocation_contract
from .schemas import ProblemContract


def _add_codex_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=Path("codex_agent_output"))
    parser.add_argument("--codex-bin", type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--expected-cli-version",
        default=DEFAULT_EXPECTED_CLI_VERSION,
        help="fail closed when the installed CLI has not been audited at this version",
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--max-rounds",
        type=int,
        choices=(1, 2),
        default=1,
        help="round two may receive public structural diagnostics only",
    )
    parser.add_argument(
        "--approve-high-risk",
        action="store_true",
        help="explicitly authorize A3/A4 local modeling runs; external actions remain unavailable",
    )


def _load_contract(path: Path) -> ProblemContract:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("kind") == "contract" and "payload" in raw:
        raw = raw["payload"]
    contract = ProblemContract.model_validate(raw)
    contract.assert_frozen()
    return contract


def _run_codex_agent(args: argparse.Namespace, contract: ProblemContract) -> int:
    config = CodexCLIConfig(
        executable=args.codex_bin,
        requested_model=args.model,
        expected_cli_version=args.expected_cli_version,
        timeout_seconds=args.timeout,
    )
    outcome = CodexDrivenModelingAgent(
        args.output,
        config,
        max_rounds=args.max_rounds,
        allow_high_risk=args.approve_high_risk,
    ).run(contract)
    print(json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if outcome.status == "validated" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="FMA trusted modeling kernel")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="run the synthetic trust-chain experiment")
    demo_parser.add_argument("--output", type=Path, default=Path("demo_output"))
    codex_demo_parser = subparsers.add_parser(
        "codex-demo", help="have Codex CLI propose the synthetic model, then verify it"
    )
    _add_codex_arguments(codex_demo_parser)
    codex_run_parser = subparsers.add_parser(
        "codex-run", help="propose and assess candidates for a frozen contract JSON"
    )
    codex_run_parser.add_argument("--contract", type=Path, required=True)
    _add_codex_arguments(codex_run_parser)
    v2_fixture_parser = subparsers.add_parser(
        "v2-capacity-fixture",
        help="run the experimental V2 public-proposal/private-test bridge fixture",
    )
    v2_fixture_parser.add_argument(
        "--output", type=Path, default=Path("fma_v2_capacity_output")
    )
    v2_intake_parser = subparsers.add_parser(
        "v2-ingest-brief",
        help="ingest one scoped local Markdown/text brief as untrusted V2 evidence",
    )
    v2_intake_parser.add_argument("--brief-file", type=Path, required=True)
    v2_intake_parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="brief file must resolve inside this existing directory",
    )
    v2_intake_parser.add_argument("--source-ref")
    v2_intake_parser.add_argument("--snapshot-id", default="brief_snapshot")
    v2_discovery_fixture_parser = subparsers.add_parser(
        "v2-discovery-fixture",
        help="run the experimental V2 append-only problem-discovery ledger fixture",
    )
    v2_discovery_fixture_parser.add_argument(
        "--output", type=Path, default=Path("fma_v2_discovery_output")
    )
    v2_codex_discovery_parser = subparsers.add_parser(
        "v2-codex-discovery-fixture",
        help="run one explicitly authorized Codex draft-only V2 discovery fixture",
    )
    v2_codex_discovery_parser.add_argument(
        "--output", type=Path, default=Path("fma_v2_codex_discovery_output")
    )
    v2_codex_discovery_parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize a real Codex CLI inference call",
    )
    v2_codex_discovery_parser.add_argument("--codex-bin", type=Path)
    v2_codex_discovery_parser.add_argument("--model")
    v2_codex_discovery_parser.add_argument(
        "--expected-cli-version", default=DEFAULT_EXPECTED_CLI_VERSION
    )
    v2_codex_discovery_parser.add_argument("--timeout", type=int, default=240)
    v2_empirical_parser = subparsers.add_parser(
        "v2-empirical-capacity-fixture",
        help="run frozen rolling-origin validation and a shadow capacity-decision gate",
    )
    v2_empirical_parser.add_argument(
        "--output", type=Path, default=Path("fma_v2_empirical_capacity_output")
    )
    v2_empirical_parser.add_argument(
        "--scenario", choices=("stable", "regime_shift"), default="stable"
    )
    v2_official_shadow_parser = subparsers.add_parser(
        "v2-official-shadow",
        help="run a read-only retrospective shadow benchmark on a frozen official API series",
    )
    v2_official_shadow_parser.add_argument(
        "--output", type=Path, default=Path("fma_v2_official_shadow_output")
    )
    v2_official_shadow_parser.add_argument(
        "--dataset",
        choices=(
            "bls_nonfarm_employment",
            "usgs_potomac_discharge",
            "bls_private_weekly_hours",
            "usgs_point_of_rocks_discharge",
        ),
        required=True,
    )
    v2_official_shadow_parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize one read-only request to the allowlisted official API",
    )
    v3_epistemic_parser = subparsers.add_parser(
        "v3-epistemic-fixture",
        help="run a local synthetic problem-reformulation epistemic-loop fixture",
    )
    v3_epistemic_parser.add_argument(
        "--output", type=Path, default=Path("fma_v3_epistemic_output")
    )
    v3_epistemic_parser.add_argument(
        "--phase",
        choices=("v30-exploratory", "v301-exploratory", "v301-confirmation"),
        required=True,
    )
    v3_epistemic_parser.add_argument(
        "--prior-failure-report-hash",
        help="required for V3.0.1; binds the evolved horizon to the V3.0 failure",
    )
    v3_epistemic_parser.add_argument("--run-id")
    subparsers.add_parser(
        "bench-list", help="print the public FMA-Bench v0 manifest"
    )
    bench_parser = subparsers.add_parser(
        "bench-run", help="run a fixture-control or explicitly authorized live benchmark arm"
    )
    bench_parser.add_argument("--output", type=Path, default=Path("fma_bench_output"))
    bench_parser.add_argument(
        "--arm",
        choices=("fixture_golden", "fixture_mutant", "live_single", "live_repair"),
        default="fixture_golden",
    )
    bench_parser.add_argument(
        "--cases",
        nargs="+",
        help="optional case IDs; omission runs all 24 cases",
    )
    bench_parser.add_argument("--repetitions", type=int, choices=range(1, 6), default=1)
    bench_parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize real Codex CLI inference for a live arm",
    )
    bench_parser.add_argument("--codex-bin", type=Path)
    bench_parser.add_argument("--model")
    bench_parser.add_argument(
        "--expected-cli-version", default=DEFAULT_EXPECTED_CLI_VERSION
    )
    bench_parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.command == "demo":
        summary = run_demo(args.output)
        print(
            json.dumps(
                {
                    "chain_established": summary["chain_established"],
                    "experiment_directory": summary["experiment_directory"],
                    "summary_path": summary["summary_path"],
                    "report_path": summary["report_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["chain_established"] else 1
    if args.command == "codex-demo":
        return _run_codex_agent(args, resource_allocation_contract())
    if args.command == "codex-run":
        return _run_codex_agent(args, _load_contract(args.contract))
    if args.command == "v2-capacity-fixture":
        from .v2.capacity_planning import (
            build_capacity_planning_fixture,
            run_capacity_planning_fixture,
        )

        binding = build_capacity_planning_fixture()
        outcome = run_capacity_planning_fixture(args.output)
        print(
            json.dumps(
                {
                    "v2_protocol": "experimental_schema_pack_v2",
                    "proposal_hash": binding.proposal_hash,
                    "acceptance_bundle_hash": binding.acceptance_bundle_hash,
                    "legacy_contract_hash": binding.contract.frozen_hash,
                    "decision_status": outcome.decision.status,
                    "validation_scope": outcome.decision.validation_scope,
                    "run_directory": outcome.run_directory,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if outcome.decision.status == "validated" else 1
    if args.command == "v2-ingest-brief":
        from .v2.intake import ingest_local_brief

        snapshot = ingest_local_brief(
            args.brief_file,
            workspace_root=args.workspace_root,
            source_ref=args.source_ref,
            snapshot_id=args.snapshot_id,
        )
        print(
            json.dumps(
                {
                    "v2_protocol": "experimental_evidence_intake_v2",
                    "snapshot_hash": snapshot.snapshot_hash,
                    "source_ref": snapshot.pedigree.source_ref,
                    "content_type": snapshot.content_type,
                    "trust_class": snapshot.trust_class,
                    "raw_text_chars": len(snapshot.raw_text),
                    "next_safe_action": "Build a draft-only ProblemHypothesis context under an approved MissionSpec.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "v2-discovery-fixture":
        from .v2.capacity_planning import run_capacity_discovery_fixture

        store, outcome = run_capacity_discovery_fixture(args.output)
        state = store.project_state()
        print(
            json.dumps(
                {
                    "v2_protocol": "experimental_discovery_ledger_v2",
                    "run_directory": str(store.run_directory),
                    "admission_status": outcome.status,
                    "mission_spec_hash": state.mission_spec_hash,
                    "evidence_snapshot_hashes": state.evidence_snapshot_hashes,
                    "admitted_hypothesis_hashes": state.admitted_hypothesis_hashes,
                    "event_count": state.event_count,
                    "event_chain_verified": store.verify(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if outcome.status == "admitted" and store.verify() else 1
    if args.command == "v2-codex-discovery-fixture":
        if not args.live:
            parser.error("v2-codex-discovery-fixture requires --live")
        from .v2.capacity_planning import run_codex_capacity_discovery_fixture

        store, proposal, outcome = run_codex_capacity_discovery_fixture(
            args.output,
            CodexCLIConfig(
                executable=args.codex_bin,
                requested_model=args.model,
                expected_cli_version=args.expected_cli_version,
                timeout_seconds=args.timeout,
                max_candidates=1,
            ),
        )
        state = store.project_state()
        print(
            json.dumps(
                {
                    "v2_protocol": "experimental_codex_problem_discovery_v2",
                    "run_directory": str(store.run_directory),
                    "provider_status": proposal.status,
                    "terminal_code": proposal.terminal_code or None,
                    "admission_status": outcome.status if outcome else None,
                    "provider_observation_hashes": state.provider_observation_hashes,
                    "admitted_hypothesis_hashes": state.admitted_hypothesis_hashes,
                    "event_count": state.event_count,
                    "event_chain_verified": store.verify(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if proposal.status == "error":
            return 1
        return 0 if store.verify() else 1
    if args.command == "v2-empirical-capacity-fixture":
        from .v2.empirical_capacity import run_empirical_capacity_fixture

        outcome = run_empirical_capacity_fixture(
            args.output,
            scenario=args.scenario,
        )
        print(
            json.dumps(
                {
                    "v2_protocol": "experimental_empirical_capacity_v2_1",
                    "scenario": args.scenario,
                    "run_directory": str(outcome.store.run_directory),
                    "manifest_hash": outcome.manifest.manifest_hash,
                    "forecast_validation_status": outcome.validation_report.status,
                    "passing_candidate_ids": outcome.validation_report.passing_candidate_ids,
                    "decision_status": outcome.decision_report.status,
                    "decision_reason_codes": outcome.decision_report.reason_codes,
                    "permissible_uses": outcome.decision_report.permissible_uses,
                    "real_world_action_authorized": (
                        outcome.decision_report.real_world_action_authorized
                    ),
                    "event_chain_verified": outcome.store.verify_event_chain(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "v2-official-shadow":
        if not args.live:
            parser.error("v2-official-shadow requires --live")
        from .v2.official_shadow import run_official_shadow_benchmark

        outcome = run_official_shadow_benchmark(
            args.output,
            dataset_name=args.dataset,
        )
        print(
            json.dumps(
                {
                    "v2_protocol": "official_retrospective_shadow_v2_1",
                    "dataset": args.dataset,
                    "run_directory": str(outcome.store.run_directory),
                    "manifest_hash": outcome.manifest.manifest_hash,
                    "forecast_validation_status": outcome.validation_report.status,
                    "passing_candidate_ids": (
                        outcome.validation_report.passing_candidate_ids
                    ),
                    "supported_challenger_ids": outcome.report.supported_challenger_ids,
                    "shift_status": outcome.shift_report.status,
                    "terminal_status": outcome.report.status,
                    "reason_codes": outcome.report.reason_codes,
                    "warnings": outcome.report.warnings,
                    "real_world_action_authorized": (
                        outcome.report.real_world_action_authorized
                    ),
                    "event_chain_verified": outcome.store.verify_event_chain(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "v3-epistemic-fixture":
        if args.phase == "v30-exploratory":
            from .v3 import (
                default_problem_reformulation_exploratory_spec_v30,
                run_problem_reformulation_worldpack_v30,
            )

            spec, baseline, candidate = (
                default_problem_reformulation_exploratory_spec_v30()
            )
            outcome = run_problem_reformulation_worldpack_v30(
                args.output,
                spec=spec,
                baseline_policy=baseline,
                candidate_policy=candidate,
                run_id=args.run_id,
            )
        else:
            if not args.prior_failure_report_hash:
                parser.error(
                    "V3.0.1 phases require --prior-failure-report-hash"
                )
            from .v3 import (
                default_problem_reformulation_confirmation_spec_v301,
                default_problem_reformulation_exploratory_spec_v301,
                run_problem_reformulation_worldpack_v301,
            )

            if args.phase == "v301-exploratory":
                spec, baseline, candidate = (
                    default_problem_reformulation_exploratory_spec_v301(
                        prior_failure_report_hash=args.prior_failure_report_hash
                    )
                )
            else:
                spec, baseline, candidate = (
                    default_problem_reformulation_confirmation_spec_v301(
                        prior_failure_report_hash=args.prior_failure_report_hash
                    )
                )
            outcome = run_problem_reformulation_worldpack_v301(
                args.output,
                spec=spec,
                baseline_policy=baseline,
                candidate_policy=candidate,
                run_id=args.run_id,
            )
        report = outcome.report
        print(
            json.dumps(
                {
                    "v3_protocol": report.schema_version,
                    "phase": args.phase,
                    "run_directory": str(outcome.store.run_directory),
                    "terminal_status": report.status,
                    "report_hash": report.report_hash,
                    "macro_regret_improvement": report.macro_regret_improvement,
                    "macro_regret_improvement_ci": [
                        report.macro_regret_improvement_ci_lower,
                        report.macro_regret_improvement_ci_upper,
                    ],
                    "material_negative_transfer_count": (
                        report.material_negative_transfer_count
                    ),
                    "negative_transfer_rate_upper": (
                        report.negative_transfer_rate_upper
                    ),
                    "same_epistemic_action_cost": (
                        report.same_epistemic_action_cost
                    ),
                    "qualification_scope": (
                        outcome.qualification.qualification_scope
                        if outcome.qualification
                        else None
                    ),
                    "real_world_action_authorized": False,
                    "event_chain_verified": outcome.store.verify_event_chain(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "bench-list":
        print(
            json.dumps(
                build_fma_bench_v0().public_manifest(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "bench-run":
        if args.arm.startswith("live_") and not args.live:
            parser.error("live benchmark arms require --live")
        summary = BenchmarkRunner(
            args.output,
            codex_config=CodexCLIConfig(
                executable=args.codex_bin,
                requested_model=args.model,
                expected_cli_version=args.expected_cli_version,
                timeout_seconds=args.timeout,
            ),
        ).run(
            args.arm,
            case_ids=args.cases,
            repetitions=args.repetitions,
            live_authorized=args.live,
        )
        print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0 if summary.aggregate.harness_integrity_passed else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
