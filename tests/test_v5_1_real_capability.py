from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from fma.codex_driver import CodexCLIConfig, ProcessResult
from fma.hashing import sha256_value
from fma.v4.unseen_event_process import EarthquakeEventV40
from fma.v5_1.codex_stage_driver import (
    CodexStageRoleTransportV51,
    RoleDraftV51,
    StageRoleDriverV51,
)
from fma.v5_1.evaluation_harness import (
    GoldAuthorityV51,
    GoldFileV51,
    MechanismProfileV51,
    MechanismRunReceiptV51,
    NuisanceIdentityV51,
    compare_ablation_runs_v51,
    inject_gold_stage_v51,
)
from fma.v5_1.event_process import (
    USGSGlobalQueryV51,
    USGSGlobalSnapshotV51,
    build_event_process_bundle_v51,
)
from fma.v5_1.executable_evaluation import (
    ExecutedMechanismRunV511,
    compare_executed_ablation_v511,
)


NOW = datetime(2026, 7, 24, tzinfo=timezone.utc)


class _RoleRunner:
    def __init__(self) -> None:
        self.exec_argv: list[list[str]] = []
        self.prompts: list[str] = []

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        input_text: str | None,
        timeout_seconds: int,
        env: dict[str, str],
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> ProcessResult:
        del cwd, timeout_seconds, env, max_stdout_bytes, max_stderr_bytes
        if argv[-1] == "--version":
            return ProcessResult(0, "codex-cli 0.144.6\n", "")
        if "login" in argv and "status" in argv:
            return ProcessResult(0, "Logged in using ChatGPT\n", "")
        if "mcp" in argv and "list" in argv:
            disabled = any("mcp_servers." in value for value in argv)
            return ProcessResult(
                0,
                json.dumps([{"name": "node_repl", "enabled": not disabled}]),
                "",
            )
        assert input_text is not None
        self.exec_argv.append(list(argv))
        self.prompts.append(input_text)
        request = json.loads(input_text.split("INPUT_JSON\n", 1)[1])
        draft = RoleDraftV51(
            request_hash=request["request_hash"],
            role_name=request["role_name"],
            selected_candidate_id=request["allowed_candidate_ids"][0],
            verdict="APPROVE",
            rationale="The public evidence supports this bounded draft only.",
            assumptions=["The public summary is content-addressed."],
            findings=["Candidate remains subject to independent checks."],
            uncertainties=["Private holdout performance is unknown."],
            proposed_artifacts=[
                {
                    "artifact_type": "candidate_selection",
                    "content": request["allowed_candidate_ids"][0],
                }
            ],
        )
        events = [
            {"type": "thread.started", "thread_id": "role-thread"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "message-1",
                    "type": "agent_message",
                    "text": draft.model_dump_json(),
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 40,
                    "reasoning_output_tokens": 5,
                },
            },
        ]
        return ProcessResult(
            0,
            "\n".join(json.dumps(item) for item in events) + "\n",
            "",
        )


def _protocol() -> dict:
    protocol = json.loads(
        Path("experiments/iteration_29/PROTOCOL.json").read_text(encoding="utf-8")
    )
    protocol["scientific_checks"]["L4"]["bootstrap_replicates"] = 20
    return protocol


def _development_snapshot() -> USGSGlobalSnapshotV51:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 7, 1, tzinfo=timezone.utc)
    query = USGSGlobalQueryV51.seal(
        phase="development",
        start=start,
        end_exclusive=end,
        min_latitude=50.0,
        max_latitude=72.0,
        min_longitude=-170.0,
        max_longitude=-130.0,
        min_magnitude=4.0,
    )
    rng = np.random.default_rng(51)
    gaps = rng.exponential(scale=1.0, size=160)
    times = np.cumsum(gaps)
    times *= 0.98 * ((end - start).total_seconds() / 86400.0) / times[-1]
    events = [
        EarthquakeEventV40(
            event_id=f"fixture-{index:04d}",
            origin_time=start + timedelta(days=float(value)),
            magnitude=4.2,
            latitude=60.0,
            longitude=-150.0,
            depth_km=20.0,
        )
        for index, value in enumerate(times)
    ]
    return USGSGlobalSnapshotV51.seal(
        query=query,
        response_sha256="1" * 64,
        events=events,
        retrieved_at=NOW,
    )


def test_real_codex_role_transport_uses_fresh_isolated_processes(tmp_path) -> None:
    fake_cli = tmp_path / "codex.exe"
    fake_cli.write_bytes(b"fake-codex")
    runner = _RoleRunner()
    driver = StageRoleDriverV51(
        CodexStageRoleTransportV51(
            tmp_path / "roles",
            CodexCLIConfig(executable=fake_cli, timeout_seconds=30),
            process_runner=runner,
            cli_locator=lambda explicit: fake_cli,
        )
    )
    first = driver.run(
        task_id="unseen_task",
        stage="S1",
        role_name="modeler_1",
        role_kind="generator",
        subject_id="candidate_set",
        objective="Choose one registered candidate from public evidence.",
        public_inputs={"summary_hash": "a" * 64},
        allowed_candidate_ids=["homogeneous_poisson"],
    )
    second = driver.run(
        task_id="unseen_task",
        stage="S1",
        role_name="formalization_referee",
        role_kind="reviewer",
        subject_id="candidate_set",
        objective="Independently challenge the public candidate draft.",
        public_inputs={"draft_hash": first.receipt.output_hash},
        allowed_candidate_ids=["homogeneous_poisson"],
    )

    assert first.request.run_id != second.request.run_id
    assert first.request.context_id != second.request.context_id
    assert len(runner.exec_argv) == 2
    assert all("--ephemeral" in argv for argv in runner.exec_argv)
    assert all("read-only" in argv and "never" in argv for argv in runner.exec_argv)
    assert all(receipt.tool_event_count == 0 for receipt in (first.receipt, second.receipt))
    assert all(receipt.scratch_unchanged for receipt in (first.receipt, second.receipt))


def test_event_process_bundle_runs_real_l0_to_l4_computations() -> None:
    development = _development_snapshot()
    bundle = build_event_process_bundle_v51(
        task_id="synthetic_adapter_contract",
        protocol=_protocol(),
        development=development,
        replay_output_hashes=["2" * 64, "2" * 64],
    )

    assert [item.level for item in bundle.levels] == ["L0", "L1", "L2", "L3", "L4"]
    assert bundle.levels[0].status == "PASS"
    assert bundle.levels[1].status == "PASS"
    assert bundle.levels[2].status == "PASS"
    assert bundle.selected_candidate_id in {
        "homogeneous_poisson",
        "weibull_renewal",
        "exponential_hawkes",
    }
    assert len(bundle.candidates) == 3
    assert bundle.scientific_qualification_granted is False
    assert bundle.real_world_action_authorized is False


def test_gold_injection_is_authenticated_and_ablation_requires_path_delta(
    tmp_path,
) -> None:
    authority = GoldAuthorityV51("gold-test-key", b"k" * 32)
    payload = b'{"stage":"S1","gold":true}\n'
    package = authority.seal_package(
        package_id="gold_s1",
        task_id="unseen_task",
        protocol_hash="3" * 64,
        through_stage="S1",
        predecessor_package_hash=None,
        files=[
            GoldFileV51(
                relative_path="docs/gold_s1.json",
                content_base64=base64.b64encode(payload).decode("ascii"),
                content_sha256=hashlib.sha256(payload).hexdigest(),
            )
        ],
    )
    receipt = inject_gold_stage_v51(
        package, authority=authority, target_root=tmp_path / "gold"
    )
    assert receipt.private_acceptance_data_injected is False
    assert (tmp_path / "gold" / "docs" / "gold_s1.json").read_bytes() == payload

    nuisance = NuisanceIdentityV51(
        task_hash="4" * 64,
        development_data_hash="5" * 64,
        candidate_registry_hash="6" * 64,
        role_prompt_pack_hash="7" * 64,
        requested_model=None,
        seed=29,
        maximum_role_calls=20,
        maximum_input_tokens=100000,
        wall_time_limit_seconds=3600,
    )
    control = MechanismRunReceiptV51.seal(
        run_id="control",
        nuisance_identity=nuisance,
        profile=MechanismProfileV51(competition=True),
        observed_mechanism_events=[
            "competition_executed",
            "independent_review_executed",
            "scientific_adapters_executed",
        ],
        terminal_state="SCIENTIFICALLY_REJECTED",
        development_score=-1.0,
        holdout_score=None,
        role_call_count=4,
        input_tokens=100,
        output_tokens=50,
        wall_time_seconds=2.0,
        output_artifact_hash=sha256_value({"winner": "a"}),
    )
    treatment = MechanismRunReceiptV51.seal(
        run_id="treatment",
        nuisance_identity=nuisance,
        profile=MechanismProfileV51(competition=False),
        observed_mechanism_events=[
            "independent_review_executed",
            "scientific_adapters_executed",
        ],
        terminal_state="SCIENTIFICALLY_REJECTED",
        development_score=-1.2,
        holdout_score=None,
        role_call_count=2,
        input_tokens=100,
        output_tokens=50,
        wall_time_seconds=2.0,
        output_artifact_hash=sha256_value({"winner": "b"}),
    )
    comparison = compare_ablation_runs_v51(
        control, treatment, mechanism_id="competition"
    )
    assert comparison.valid_ablation
    assert comparison.observed_execution_path_delta
    assert not comparison.no_op_detected


def test_executed_ablation_rejects_reused_process_receipts() -> None:
    nuisance = NuisanceIdentityV51(
        task_hash="4" * 64,
        development_data_hash="5" * 64,
        candidate_registry_hash="6" * 64,
        role_prompt_pack_hash="7" * 64,
        requested_model=None,
        seed=29,
        maximum_role_calls=20,
        maximum_input_tokens=100000,
        wall_time_limit_seconds=3600,
    )

    def arm(run_id: str, enabled: bool, receipt_hash: str):
        return ExecutedMechanismRunV511.seal(
            run_id=run_id,
            nuisance_identity=nuisance,
            profile=MechanismProfileV51(competition=enabled),
            process_receipt_hashes=[receipt_hash],
            process_run_ids=[f"{run_id}-process"],
            process_context_ids=[f"{run_id}-context"],
            observed_mechanism_events=(
                ["competition_executed"] if enabled else []
            ),
            selected_candidate_id="homogeneous_poisson",
            development_score=-1.0,
            output_artifact_hash=sha256_value({"run": run_id}),
            terminal_state="SCIENTIFICALLY_REJECTED",
        )

    control = arm("control", True, "8" * 64)
    reused = arm("reused", False, "8" * 64)
    invalid = compare_executed_ablation_v511(
        control, reused, mechanism_id="competition"
    )
    assert not invalid.valid_executed_ablation
    assert "process_receipts_reused_across_arms" in invalid.reasons

    fresh = arm("fresh", False, "9" * 64)
    valid = compare_executed_ablation_v511(
        control, fresh, mechanism_id="competition"
    )
    assert valid.valid_executed_ablation
    assert valid.process_receipts_disjoint
