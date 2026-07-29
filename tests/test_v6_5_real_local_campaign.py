from __future__ import annotations

import json
import time
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any, Literal

import pytest

from fma._file_lock import exclusive_file_lock
from fma.codex_driver import CodexCLIConfig
from fma.codex_wsl import WslCodexRuntimeV65
from fma.hashing import canonical_json, sha256_value
from fma.studio.backhalf_runtime import (
    RAW_RELATIVE_PATH,
    StudioODEDataRequestV59,
)
from fma.studio.service import StudioTaskService
from fma.v6 import real_local_campaign_cli
from fma.v6 import real_local_campaign as campaign_module
from fma.v6 import public_source
from fma.v6.provenance import MEASUREMENT_SCHEMA_PATH
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    SourceHTTPResponseV62,
    materialize_world_bank_series_v62,
)
from fma.v6.real_local_campaign import (
    CampaignConflictError,
    CampaignReconciliationRequired,
    CampaignRetryRequired,
    CodexRuntimeBudgetsV65,
    CodexRuntimeContractV65,
    LiveExecutionNotAuthorized,
    RealLocalCampaignError,
    RealLocalCampaignEventV65,
    RealLocalCampaignRunnerV65,
    RealLocalCampaignSpecV65,
    RealLocalCampaignTerminalReceiptV65,
    _assert_local_python_runtime_identity_current,
    _assert_runtime_code_manifest_current,
    _canonical_system_wsl_executable_v65,
    _local_python_runtime_identity,
    _runtime_code_manifest,
    _trusted_runtime_adapter,
)
from fma.v6.source_auth import (
    SOURCE_ACQUISITION_AUTH_PATH,
    SourceTransportAuthorityV62,
)


AUTHORITY_KEY = b"real-local-campaign-test-key-" + b"k" * 32


def _frozen_runtime_contract() -> CodexRuntimeContractV65:
    config = CodexCLIConfig(requested_model="gpt-test")
    manifest = _runtime_code_manifest("adaptive_positive_series_v57")
    identity = {
        "transport": "native_process",
        "process_runner": "fma.codex_driver._default_process_runner",
        "cli_locator": "fma.codex_driver.discover_codex_cli",
        "local_python_runtime": _local_python_runtime_identity(),
    }
    return CodexRuntimeContractV65.seal(
        provider="openai_codex_cli",
        runtime_adapter_id="native_codex_cli_v1",
        requested_model="gpt-test",
        expected_cli_version=config.expected_cli_version,
        observed_cli_version=config.expected_cli_version,
        executable_name="codex.exe",
        executable_sha256=(
            "4b76ded066d0239115ca97473d010c920"
            "72bc5c5550a45dd7cbebe1e9eb956a7"
        ),
        runtime_identity=identity,
        runtime_identity_hash=sha256_value(identity),
        budgets=CodexRuntimeBudgetsV65.from_config(config),
        source_adapter_id="adaptive_positive_series_v57",
        code_manifest=manifest,
        code_manifest_hash=sha256_value(
            [item.model_dump(mode="json") for item in manifest]
        ),
    )


def _spec(
    *,
    live_codex: bool = True,
    live_world_bank: bool = True,
    runtime_contract: CodexRuntimeContractV65 | None = None,
    execution_mode: Literal["real", "control"] = "control",
) -> RealLocalCampaignSpecV65:
    return RealLocalCampaignSpecV65.seal(
        campaign_id="campaign-new-public-series",
        task_id="task-new-public-series",
        objective=(
            "Forecast the next values of one frozen public annual indicator "
            "and report uncertainty without authorizing a decision."
        ),
        live_codex=live_codex,
        live_world_bank=live_world_bank,
        execution_mode=execution_mode,
        codex_runtime_contract=runtime_contract,
        world_bank_request={
            "adapter_id": "adaptive_positive_series_v57",
            "contract_id": "wb-life-expectancy-india",
            "country_code": "IND",
            "indicator_id": "SP.DYN.LE00.IN",
            "start_year": 1988,
            "end_year": 2023,
            "minimum_observations": 23,
            "state_unit": "years",
            "attribution": "World Bank World Development Indicators",
            "semantic_name": "life expectancy at birth",
            "operational_definition": (
                "Expected years a newborn would live under current mortality."
            ),
            "observation_time_basis": "annual calendar-year estimate",
            "aggregation_level": "national population",
            "fixture_only": False,
        },
    )


class _FakeStudio:
    def __init__(
        self,
        task_root: Path,
        *,
        fail_action: str | None = None,
        crash_source: bool = False,
        overclaim: bool = False,
        action_delay_seconds: float = 0,
    ) -> None:
        self.task_root = Path(task_root)
        self.fail_action = fail_action
        self.crash_source = crash_source
        self.overclaim = overclaim
        self.action_delay_seconds = action_delay_seconds
        self.calls: list[str] = []
        self.s0 = False
        self.s1 = False
        self.data = False
        self.complete = False
        self.objective: str | None = None

    def _maybe_fail(self, action: str) -> None:
        self.calls.append(action)
        if self.action_delay_seconds:
            time.sleep(self.action_delay_seconds)
        if self.fail_action == action:
            raise RuntimeError(f"{action} failed")

    def _snapshot(self, task_id: str) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "objective": self.objective,
            "workflow": {
                "stage_statuses": {
                    "S0": "gate_open" if self.s0 else "frontier",
                    "S1": "gate_open" if self.s1 else "pending",
                    "S2": "gate_open" if self.complete else "pending",
                    "S3": "gate_open" if self.complete else "pending",
                    "S4": "gate_open" if self.complete else "pending",
                    "S5": "gate_open" if self.complete else "pending",
                    "S6": "gate_open" if self.complete else "pending",
                }
            },
            "backhalf": {
                "data_received": self.data,
                "workflow_complete": self.complete,
                "fixture_only": False if self.complete else None,
            },
            # The runner must never propagate these attempted fake claims.
            "scientific_qualification_granted": self.overclaim,
            "real_world_action_authorized": self.overclaim,
        }

    def create_task(self, request: dict[str, Any]) -> dict[str, Any]:
        self._maybe_fail("create_task")
        self.objective = request["objective"]
        root = self.task_root / request["workspace_id"] / ".fma"
        root.mkdir(parents=True, exist_ok=True)
        (root / "workspace_spec.json").write_text("{}", encoding="utf-8")
        return self._snapshot(request["workspace_id"])

    def snapshot(self, task_id: str) -> dict[str, Any]:
        return self._snapshot(task_id)

    def run_s0(self, task_id: str) -> dict[str, Any]:
        self._maybe_fail("run_s0")
        self.s0 = True
        return self._snapshot(task_id)

    def run_s1(self, task_id: str) -> dict[str, Any]:
        self._maybe_fail("run_s1")
        self.s1 = True
        return self._snapshot(task_id)

    def ingest_world_bank_data(
        self,
        task_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append("ingest_world_bank_data")
        if self.crash_source:
            partial = self.task_root / task_id / SOURCE_CONTRACT_PATH
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text("{}", encoding="utf-8")
            raise KeyboardInterrupt("simulated hard interruption")
        if self.fail_action == "ingest_world_bank_data":
            raise RuntimeError("ingest_world_bank_data failed")
        self.data = True
        return self._snapshot(task_id)

    def run_backhalf(self, task_id: str) -> dict[str, Any]:
        self._maybe_fail("run_backhalf")
        self.complete = True
        return self._snapshot(task_id)


class _BlockedS0Studio(_FakeStudio):
    def run_s0(self, task_id: str) -> dict[str, Any]:
        self._maybe_fail("run_s0")
        snapshot = self._snapshot(task_id)
        snapshot["workflow"]["stage_statuses"]["S0"] = "blocked"
        snapshot["recovery"] = {
            "human_required": True,
            "stopped": False,
        }
        snapshot["events"] = [
            {
                "event_type": "s0_gate_evaluated",
                "details": {
                    "decision": "BLOCKED",
                    "review_verdict": "REJECT",
                    "finding_signature": "a" * 64,
                    "gate_outcome_hash": "b" * 64,
                },
            }
        ]
        return snapshot


def _factory_for(fake: _FakeStudio):
    def factory(**kwargs: Any) -> _FakeStudio:
        assert Path(kwargs["task_root"]) == fake.task_root
        return fake

    return factory


def _runner(
    root: Path,
    fake: _FakeStudio,
) -> RealLocalCampaignRunnerV65:
    return RealLocalCampaignRunnerV65(
        root,
        authority_key=AUTHORITY_KEY,
        service_factory=_factory_for(fake),
    )


def test_prepare_freezes_spec_without_constructing_service(
    tmp_path: Path,
) -> None:
    constructed = 0

    def forbidden_factory(**_: Any) -> Any:
        nonlocal constructed
        constructed += 1
        raise AssertionError("prepare must not construct Studio")

    runner = RealLocalCampaignRunnerV65(
        tmp_path,
        authority_key=AUTHORITY_KEY,
        service_factory=forbidden_factory,
    )
    prepared = runner.prepare(_spec(live_codex=False, live_world_bank=False))

    assert prepared.spec_hash
    assert constructed == 0
    assert runner.status()["event_count"] == 1
    assert runner.status()["live_execution_default"] is False


def test_prepare_requires_authority_and_creates_freeze_anchor(
    tmp_path: Path,
) -> None:
    unsigned_runner = RealLocalCampaignRunnerV65(tmp_path)
    with pytest.raises(
        RealLocalCampaignError,
        match="requires an authority key",
    ):
        unsigned_runner.prepare(_spec())
    assert not unsigned_runner.spec_path.exists()

    runner = RealLocalCampaignRunnerV65(
        tmp_path,
        authority_key=AUTHORITY_KEY,
    )
    spec = runner.prepare(_spec())

    assert runner.freeze_receipt_path.is_file()
    assert runner.load_spec() == spec


def test_live_real_campaign_without_runtime_contract_is_rejected_at_prepare(
    tmp_path: Path,
) -> None:
    runner = RealLocalCampaignRunnerV65(
        tmp_path,
        authority_key=AUTHORITY_KEY,
    )
    payload = _spec().model_dump(mode="json", exclude={"spec_hash"})
    payload["execution_mode"] = "real"
    payload["codex_runtime_contract"] = None

    with pytest.raises(
        ValueError,
        match="requires a frozen Codex runtime contract",
    ):
        runner.prepare(payload)

    assert not runner.spec_path.exists()
    assert not runner.events_path.exists()
    assert not runner.freeze_receipt_path.exists()
    assert not runner.terminal_receipts_path.exists()


def test_pre_anchor_campaign_cannot_be_silently_upgraded(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    spec = runner.prepare(_spec())
    runner.freeze_receipt_path.unlink()

    with pytest.raises(CampaignConflictError, match="cannot be silently"):
        runner.prepare(spec)
    status = runner.status()
    assert status["event_chain_verified"] is False
    assert status["reconciliation_required"] is True


def test_execute_requires_cli_and_both_frozen_live_opt_ins(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())

    with pytest.raises(LiveExecutionNotAuthorized):
        runner.execute(execute_live=False)
    assert fake.calls == []

    other_root = tmp_path / "other"
    other_fake = _FakeStudio(other_root / "workspace")
    other = _runner(other_root, other_fake)
    other.prepare(_spec(live_codex=True, live_world_bank=False))
    with pytest.raises(LiveExecutionNotAuthorized):
        other.execute(execute_live=True)
    assert other_fake.calls == []


def test_control_run_is_replay_stable_and_never_overclaims(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace", overclaim=True)
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())

    first = runner.execute(execute_live=True)
    second = runner.execute(execute_live=True)

    assert first == second
    assert first.terminal_status == "COMPLETED_CONTROL"
    assert first.fixture_or_control is True
    assert first.snapshot_fixture_only is True
    assert first.claim_ceiling == "control_protocol_only"
    assert first.external_scientific_qualification_status == "NOT_RUN"
    assert first.scientific_qualification_granted is False
    assert first.real_world_action_authorized is False
    assert fake.calls == [
        "create_task",
        "run_s0",
        "run_s1",
        "ingest_world_bank_data",
        "run_backhalf",
    ]
    assert runner.verify(require_real=True) is False
    assert runner.verify(require_real=False) is True


def test_cached_control_terminal_rejects_real_service_boundary(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    control_runner = _runner(tmp_path, fake)
    spec = control_runner.prepare(_spec())
    receipt = control_runner.execute(execute_live=True)
    real_runner = RealLocalCampaignRunnerV65(
        tmp_path,
        authority_key=AUTHORITY_KEY,
    )

    with pytest.raises(
        RealLocalCampaignError,
        match="execution_mode differs",
    ):
        real_runner.execute(execute_live=True)

    status = real_runner.status()
    assert status["journal_current"] is True
    assert status["verified_current"] is False
    assert status["reconciliation_required"] is True
    assert control_runner._read_receipts(spec) == [receipt]


def test_conflicting_spec_and_tampered_hash_chain_fail_closed(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    conflicting = _spec().model_copy(
        update={
            "objective": (
                "Forecast a different frozen public indicator and report only."
            ),
            "spec_hash": None,
        }
    )
    conflicting = RealLocalCampaignSpecV65.seal(
        **conflicting.model_dump(exclude={"spec_hash"})
    )

    with pytest.raises(CampaignConflictError):
        runner.prepare(conflicting)

    events_path = tmp_path / "campaign_events_v65.jsonl"
    original = events_path.read_text(encoding="utf-8")
    events_path.write_text(
        original.replace('"PREPARED"', '"SUCCEEDED"', 1),
        encoding="utf-8",
    )
    assert runner.status()["event_chain_verified"] is False


def test_exception_produces_claim_limited_failure_receipt(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace", fail_action="run_s1")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())

    receipt = runner.execute(execute_live=True)

    assert receipt.terminal_status == "FAILED"
    assert receipt.fixture_or_control is True
    assert receipt.workspace_verified is False
    assert receipt.external_scientific_qualification_status == "NOT_RUN"
    assert receipt.scientific_qualification_granted is False
    assert receipt.real_world_action_authorized is False
    assert runner.status()["terminal_status"] == "FAILED"


def test_interrupted_source_partial_requires_human_reconciliation(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace", crash_source=True)
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())

    with pytest.raises(KeyboardInterrupt):
        runner.execute(execute_live=True)
    fake.crash_source = False
    resumed = _runner(tmp_path, fake)
    receipt = resumed.execute(execute_live=True)

    assert receipt.terminal_status == "HUMAN_RECONCILIATION_REQUIRED"
    assert receipt.fixture_or_control is True
    assert receipt.scientific_qualification_granted is False
    assert fake.calls.count("ingest_world_bank_data") == 1


def test_self_consistent_spec_replacement_is_rejected_by_genesis(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    replacement = _spec().model_copy(
        update={
            "objective": (
                "Forecast a replacement public series under another frozen "
                "objective without authorizing a decision."
            ),
            "spec_hash": None,
        }
    )
    replacement = RealLocalCampaignSpecV65.seal(
        **replacement.model_dump(mode="json", exclude={"spec_hash"})
    )
    runner.spec_path.write_text(
        canonical_json(replacement.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    status = runner.status()

    assert status["event_chain_verified"] is False
    assert status["reconciliation_required"] is True
    with pytest.raises(CampaignReconciliationRequired):
        runner.execute(execute_live=True)


def test_synchronized_spec_and_genesis_replacement_fails_freeze_anchor(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    original = runner.prepare(_spec())
    original_genesis = runner._read_events(original)[0]
    replacement = RealLocalCampaignSpecV65.seal(
        **{
            **original.model_dump(mode="json", exclude={"spec_hash"}),
            "objective": (
                "Forecast a synchronized attacker replacement series while "
                "preserving a self-consistent local hash chain."
            ),
        }
    )
    replacement_genesis = RealLocalCampaignEventV65.seal(
        campaign_id=replacement.campaign_id,
        sequence=1,
        event_type="CAMPAIGN_PREPARED",
        status="PREPARED",
        details={
            "spec_hash": replacement.spec_hash,
            "execution_mode": replacement.execution_mode,
            "live_codex": replacement.live_codex,
            "live_world_bank": replacement.live_world_bank,
            "default_live_execution": False,
            "execution_semantics": replacement.execution_semantics,
            "max_execution_attempts": replacement.max_execution_attempts,
        },
        recorded_at=original_genesis.recorded_at,
        previous_event_hash=None,
    )
    runner.spec_path.write_text(
        canonical_json(replacement.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    runner.events_path.write_text(
        canonical_json(replacement_genesis.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    status = runner.status()

    assert status["event_chain_verified"] is False
    assert status["verified_current"] is False
    assert status["reconciliation_required"] is True
    with pytest.raises(CampaignReconciliationRequired):
        runner.execute(execute_live=True)


def test_injected_codex_runner_cannot_enter_real_mode(
    tmp_path: Path,
) -> None:
    class FakeRunner:
        def __call__(self, *_: Any, **__: Any) -> Any:
            raise AssertionError("injected runner must not be invoked")

    fake_runner = FakeRunner()
    runner = RealLocalCampaignRunnerV65(
        tmp_path,
        authority_key=AUTHORITY_KEY,
        codex_config=CodexCLIConfig(requested_model="gpt-test"),
        codex_process_runner=fake_runner,
        codex_cli_locator=lambda *_: tmp_path / "fake-codex.exe",
    )
    runner.prepare(
        _spec(
            runtime_contract=_frozen_runtime_contract(),
            execution_mode="real",
        )
    )

    receipt = runner.execute(execute_live=True)

    assert receipt.terminal_status == "FAILED"
    assert receipt.fixture_or_control is False
    assert receipt.completed_actions == []
    assert receipt.claim_ceiling == "no_scientific_claim"
    assert receipt.authority_auth_tag
    assert runner._read_receipts(runner.load_spec()) == [receipt]


def test_wsl_runtime_requires_exact_adapter_and_bound_locator() -> None:
    class WslSubclass(WslCodexRuntimeV65):
        pass

    subclass = object.__new__(WslSubclass)
    exact = object.__new__(WslCodexRuntimeV65)

    with pytest.raises(
        RealLocalCampaignError,
        match="rejects injected Codex runner or locator",
    ):
        _trusted_runtime_adapter(subclass, subclass.locate)
    with pytest.raises(
        RealLocalCampaignError,
        match="rejects injected Codex runner or locator",
    ):
        _trusted_runtime_adapter(exact, lambda *_: Path("codex"))
    with pytest.raises(
        RealLocalCampaignError,
        match="not fully initialized",
    ):
        _trusted_runtime_adapter(exact, exact.locate)


def test_fake_outer_wsl_wrapper_is_rejected(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text("{}", encoding="utf-8")
    fake_wsl = tmp_path / "fake-wsl.exe"
    fake_wsl.write_bytes(b"fake outer wrapper")

    with WslCodexRuntimeV65(
        source_codex_home=codex_home,
        wsl_executable=fake_wsl,
    ) as runtime:
        with pytest.raises(
            RealLocalCampaignError,
            match="canonical Windows system executable",
        ):
            _trusted_runtime_adapter(runtime, runtime.locate)


def test_canonical_system_wsl_transport_is_separately_frozen(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text("{}", encoding="utf-8")
    system_wsl = _canonical_system_wsl_executable_v65()

    with WslCodexRuntimeV65(
        source_codex_home=codex_home,
        wsl_executable=system_wsl,
    ) as runtime:
        provider, adapter, identity, outer = _trusted_runtime_adapter(
            runtime,
            runtime.locate,
        )

    assert provider == "openai_codex_cli_via_wsl"
    assert adapter == "wsl_codex_cli_v65"
    assert outer is not None
    assert outer.canonical_executable_path == str(system_wsl)
    assert outer.executable_sha256 == identity["wsl_executable_sha256"]
    assert (
        outer.release_fingerprint_policy
        == "code_owned_local_allowlist_not_windows_signature"
    )


def test_runtime_manifest_covers_every_fma_python_file_and_rejects_omission(
    tmp_path: Path,
) -> None:
    manifest = _runtime_code_manifest("adaptive_positive_series_v57")
    repository_root = Path(__file__).resolve().parents[1]
    expected = sorted(
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "fma").rglob("*.py")
        if path.is_file()
    )

    assert [item.relative_path for item in manifest] == expected

    contract = _frozen_runtime_contract()
    truncated = contract.model_dump(
        mode="json",
        exclude={"contract_hash"},
    )
    truncated["code_manifest"] = truncated["code_manifest"][:-1]
    truncated["code_manifest_hash"] = sha256_value(
        truncated["code_manifest"]
    )
    omitted = CodexRuntimeContractV65.seal(**truncated)
    with pytest.raises(
        RealLocalCampaignError,
        match="differs from the frozen runtime manifest",
    ):
        _assert_runtime_code_manifest_current(omitted)


def test_self_reported_codex_version_cannot_bypass_release_fingerprint() -> None:
    contract = _frozen_runtime_contract()
    payload = contract.model_dump(mode="json", exclude={"contract_hash"})
    payload["executable_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="local release allowlist"):
        CodexRuntimeContractV65.seal(**payload)


def test_local_runtime_identity_covers_declared_dependencies_and_numpy_build(
) -> None:
    identity = _local_python_runtime_identity()

    assert set(identity["declared_runtime_distributions"]) == {
        "cryptography",
        "numpy",
        "pydantic",
        "pydantic-core",
        "scipy",
    }
    assert identity["python_cache_tag"]
    assert identity["os_system"] == "Windows"
    assert identity["machine"]
    assert identity["numpy_build_identity"]["config_hash"]
    assert "not_complete_environment" in identity["claim_ceiling"]


def test_missing_declared_runtime_dependency_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_version = importlib.metadata.version

    def missing_cryptography(name: str) -> str:
        if name == "cryptography":
            raise importlib.metadata.PackageNotFoundError(name)
        return actual_version(name)

    monkeypatch.setattr(
        campaign_module.importlib.metadata,
        "version",
        missing_cryptography,
    )

    with pytest.raises(
        RealLocalCampaignError,
        match="required runtime dependency is unavailable",
    ):
        _local_python_runtime_identity()


def test_declared_dependency_version_drift_rejects_frozen_runtime() -> None:
    contract = _frozen_runtime_contract()
    payload = contract.model_dump(mode="json", exclude={"contract_hash"})
    identity = json.loads(json.dumps(payload["runtime_identity"]))
    identity["local_python_runtime"]["declared_runtime_distributions"][
        "pydantic"
    ] = "0.0-drifted"
    payload["runtime_identity"] = identity
    payload["runtime_identity_hash"] = sha256_value(identity)
    drifted = CodexRuntimeContractV65.seal(**payload)

    with pytest.raises(
        RealLocalCampaignError,
        match="declared dependency runtime differs",
    ):
        _assert_local_python_runtime_identity_current(drifted)


def test_wrong_partial_source_cannot_be_recorded_as_success(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    fake.objective = _spec().objective
    fake.s0 = True
    fake.s1 = True
    fake.data = True
    source_path = (
        fake.task_root / "task-new-public-series" / SOURCE_CONTRACT_PATH
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps({"country_code": "USA"}),
        encoding="utf-8",
    )

    receipt = runner.execute(execute_live=True)

    assert receipt.terminal_status == "HUMAN_RECONCILIATION_REQUIRED"
    assert "ingest_world_bank_data" not in fake.calls
    assert "run_backhalf" not in fake.calls


def test_real_stage_workspace_source_layout_proves_ingest_postcondition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = RealLocalCampaignRunnerV65(
        tmp_path,
        authority_key=AUTHORITY_KEY,
    )
    spec = runner.prepare(
        _spec(
            live_codex=False,
            live_world_bank=False,
            execution_mode="real",
        )
    )
    service = StudioTaskService(
        task_root=runner.workspace_root,
        authority_key=AUTHORITY_KEY,
        authority_key_id=runner.authority_key_id,
    )
    service.create_task(
        {
            "workspace_id": spec.task_id,
            "objective": spec.objective,
            "evidence_scope": "public_data",
        }
    )
    workspace = service._workspace(spec.task_id)
    contract = runner._expected_source_contract(spec)
    records = [
        {
            "indicator": {
                "id": contract.indicator_id,
                "value": "Life expectancy at birth, total (years)",
            },
            "countryiso3code": contract.country_code,
            "date": str(year),
            "value": 55.0 + (year - contract.start_year) * 0.2,
        }
        for year in range(contract.end_year, contract.start_year - 1, -1)
    ]
    body = json.dumps(
        [
            {"page": 1, "pages": 1, "total": len(records)},
            records,
        ]
    ).encode("utf-8")
    monkeypatch.setattr(
        public_source,
        "_default_fetcher",
        lambda frozen: SourceHTTPResponseV62(
            status=200,
            final_url=frozen.exact_url,
            content_type="application/json",
            body=body,
        ),
    )
    source_authority = SourceTransportAuthorityV62.from_stage_workspace(
        workspace
    )
    acquisition = source_authority.acquire_world_bank_series(
        workspace_spec=workspace.spec,
        task_id=spec.task_id,
        contract=contract,
    )
    materialize_world_bank_series_v62(
        workspace_root=workspace.root,
        fetched=acquisition.fetched,
    )
    (workspace.root / SOURCE_ACQUISITION_AUTH_PATH).write_text(
        canonical_json(
            acquisition.authority_receipt.model_dump(mode="json")
        )
        + "\n",
        encoding="utf-8",
    )
    measurement = runner._expected_measurement_schema(spec, contract)
    (workspace.root / MEASUREMENT_SCHEMA_PATH).write_text(
        canonical_json(measurement.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    fetched = acquisition.fetched
    raw_request = StudioODEDataRequestV59(
        adapter_id=spec.world_bank_request.adapter_id,
        time_unit=fetched.snapshot.time_unit,
        state_unit=fetched.snapshot.state_unit,
        times=fetched.snapshot.times,
        observations=fetched.snapshot.observations,
        source_id=fetched.receipt.source_id,
        license_status=(
            "world_bank_default_open_data_recorded;"
            "independent_license_review_absent"
        ),
        fixture_only=False,
    )
    raw_path = workspace.root / RAW_RELATIVE_PATH
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        canonical_json(raw_request.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    snapshot = service.snapshot(spec.task_id)
    proven, evidence_hash = runner._action_postcondition(
        spec,
        "ingest_world_bank_data",
        snapshot,
    )

    assert workspace.verify() is True
    assert snapshot["backhalf"]["data_received"] is True
    assert proven is True
    assert evidence_hash


def test_failed_terminal_requires_explicit_retry_and_binds_predecessor(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace", fail_action="run_s1")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    first = runner.execute(execute_live=True)
    assert first.terminal_status == "FAILED"

    with pytest.raises(CampaignRetryRequired):
        runner.execute(execute_live=True)

    fake.fail_action = None
    second = runner.execute(execute_live=True, retry_failed=True)

    assert second.terminal_status == "COMPLETED_CONTROL"
    assert second.attempt_index == 2
    assert second.previous_receipt_hash == first.receipt_hash
    assert second.execution_id != first.execution_id
    assert runner.verify(require_real=False) is True


def test_execution_attempt_budget_stops_after_three_failed_receipts(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace", fail_action="create_task")
    runner = _runner(tmp_path, fake)
    spec = runner.prepare(_spec())

    first = runner.execute(execute_live=True)
    second = runner.execute(execute_live=True, retry_failed=True)
    third = runner.execute(execute_live=True, retry_failed=True)

    assert [first.attempt_index, second.attempt_index, third.attempt_index] == [
        1,
        2,
        3,
    ]
    assert all(
        receipt.terminal_status == "FAILED"
        for receipt in (first, second, third)
    )
    with pytest.raises(CampaignRetryRequired, match="budget exhausted"):
        runner.execute(execute_live=True, retry_failed=True)
    assert len(runner._read_receipts(spec)) == 3
    assert fake.calls.count("create_task") == 3


def test_preexisting_partial_workspace_resolves_to_human(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    spec = runner.prepare(_spec())
    partial = fake.task_root / spec.task_id / ".fma"
    partial.mkdir(parents=True)
    (partial / "interrupted.tmp").write_text("partial", encoding="utf-8")

    receipt = runner.execute(execute_live=True)
    events = runner._read_events(spec)

    assert receipt.terminal_status == "HUMAN_RECONCILIATION_REQUIRED"
    assert fake.calls == []
    assert runner._pending_intent(events) is None
    assert events[-1].event_type == "ACTION_RESULT"
    assert events[-1].status == "AMBIGUOUS"


def test_service_construction_failure_gets_signed_failed_receipt(
    tmp_path: Path,
) -> None:
    def failing_factory(**_: Any) -> Any:
        raise RuntimeError("service construction failed")

    runner = RealLocalCampaignRunnerV65(
        tmp_path,
        authority_key=AUTHORITY_KEY,
        service_factory=failing_factory,
    )
    runner.prepare(_spec())

    receipt = runner.execute(execute_live=True)

    assert receipt.terminal_status == "FAILED"
    assert receipt.completed_actions == []
    assert receipt.authority_auth_tag
    assert runner.verify(require_real=False) is False


def test_stage_rejection_is_preserved_in_campaign_evidence(
    tmp_path: Path,
) -> None:
    fake = _BlockedS0Studio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    spec = runner.prepare(_spec())

    receipt = runner.execute(execute_live=True)
    events = runner._read_events(spec)
    result = next(
        event
        for event in events
        if event.event_type == "ACTION_RESULT"
        and event.action == "run_s0"
    )

    assert receipt.terminal_status == "FAILED"
    assert receipt.completed_actions == ["create_task"]
    assert "stage_blocked_S0" in receipt.reason_codes
    assert "gate_decision_BLOCKED" in receipt.reason_codes
    assert "review_verdict_REJECT" in receipt.reason_codes
    assert "human_review_required" in receipt.reason_codes
    assert receipt.snapshot_hash is not None
    assert result.status == "FAILED"
    assert result.details["failure_kind"] == "STAGE_BLOCKED"
    assert result.details["stage"] == "S0"
    assert result.details["finding_signature"] == "a" * 64
    assert result.details["gate_outcome_hash"] == "b" * 64
    assert runner._pending_intent(events) is None


def test_retry_reproves_historical_completed_postconditions(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace", fail_action="run_s1")
    runner = _runner(tmp_path, fake)
    spec = runner.prepare(_spec())
    first = runner.execute(execute_live=True)
    assert first.completed_actions == ["create_task", "run_s0"]

    fake.calls.clear()
    fake.fail_action = None
    fake.s0 = False
    second = runner.execute(execute_live=True, retry_failed=True)

    assert second.terminal_status == "HUMAN_RECONCILIATION_REQUIRED"
    assert fake.calls == []
    assert runner._pending_intent(runner._read_events(spec)) is None


def test_forged_self_consistent_terminal_receipt_fails_authority_replay(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    valid = runner.execute(execute_live=True)
    payload = valid.model_dump(mode="json")
    payload["authority_auth_tag"] = "f" * 64
    payload["receipt_hash"] = None
    draft = RealLocalCampaignTerminalReceiptV65.model_validate(payload)
    payload["receipt_hash"] = draft.content_hash()
    forged = RealLocalCampaignTerminalReceiptV65.model_validate(payload)
    runner.terminal_receipts_path.write_text(
        canonical_json(forged.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    status = runner.status()

    assert status["journal_current"] is True
    assert status["verified_current"] is False
    assert status["authority_receipts_verified"] is False
    assert status["reconciliation_required"] is True
    assert runner.verify(require_real=False) is False
    with pytest.raises(CampaignReconciliationRequired):
        runner.execute(execute_live=True)


def test_status_without_authority_never_claims_current_or_verified(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    runner.execute(execute_live=True)

    unsigned_status = RealLocalCampaignRunnerV65(tmp_path).status()

    assert unsigned_status["prepared"] is True
    assert unsigned_status["journal_current"] is False
    assert unsigned_status["verified_current"] is False
    assert unsigned_status["terminal_current"] is False
    assert unsigned_status["authority_verification_available"] is False
    assert unsigned_status["verification_unavailable"] is True
    assert unsigned_status["reconciliation_required"] is True


def test_current_success_with_corrupted_workspace_requires_reconciliation(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    spec = runner.prepare(_spec())
    runner.execute(execute_live=True)
    (
        fake.task_root
        / spec.task_id
        / ".fma"
        / "workspace_spec.json"
    ).unlink()

    status = runner.status()

    assert status["journal_current"] is True
    assert status["authority_receipts_verified"] is True
    assert status["verified_current"] is False
    assert status["reconciliation_required"] is True
    with pytest.raises(
        CampaignReconciliationRequired,
        match="failed full-chain replay",
    ):
        runner.execute(execute_live=True)


def test_post_terminal_event_invalidates_current_receipt(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    spec = runner.prepare(_spec())
    receipt = runner.execute(execute_live=True)
    events = runner._read_events(spec)
    request = {"task_id": spec.task_id, "action": "run_s0"}
    appended = RealLocalCampaignEventV65.seal(
        campaign_id=spec.campaign_id,
        sequence=len(events) + 1,
        event_type="ACTION_INTENT",
        status="INTENT_RECORDED",
        action_id="action-post-terminal",
        action="run_s0",
        request_hash=sha256_value(request),
        details={
            "execution_semantics": spec.execution_semantics,
            "execution_id": "exec-post-terminal",
            "attempt_index": receipt.attempt_index + 1,
            "predecessor_receipt_hash": receipt.receipt_hash,
        },
        recorded_at=receipt.finished_at,
        previous_event_hash=events[-1].event_hash,
    )
    with runner.events_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(appended.model_dump(mode="json")) + "\n")

    assert runner.verify(require_real=False) is False
    assert runner.status()["terminal_current"] is False
    with pytest.raises(CampaignReconciliationRequired):
        runner.execute(execute_live=True)


def test_execute_holds_single_writer_lock_for_entire_attempt(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(
        tmp_path / "workspace",
        action_delay_seconds=0.02,
    )
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(
            pool.map(
                lambda _: runner.execute(execute_live=True),
                range(2),
            )
        )

    assert receipts[0] == receipts[1]
    assert receipts[0].terminal_status == "COMPLETED_CONTROL"
    assert fake.calls == [
        "create_task",
        "run_s0",
        "run_s1",
        "ingest_world_bank_data",
        "run_backhalf",
    ]


def test_status_lock_timeout_reports_running_without_false_reconciliation(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    entered = Event()
    release = Event()

    def hold_campaign_lock() -> None:
        with exclusive_file_lock(runner.lock_path):
            entered.set()
            assert release.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold_campaign_lock)
        assert entered.wait(timeout=2)
        try:
            status = runner.status(lock_timeout_seconds=0.05)
        finally:
            release.set()
        future.result(timeout=2)

    assert status["execution_state"] == "RUNNING_OR_LOCKED"
    assert status["event_chain_verified"] is None
    assert status["journal_current"] is False
    assert status["verified_current"] is False
    assert status["terminal_current"] is False
    assert status["verification_unavailable"] is True
    assert status["reconciliation_required"] is None


def test_partial_jsonl_fails_closed_to_human_reconciliation(
    tmp_path: Path,
) -> None:
    fake = _FakeStudio(tmp_path / "workspace")
    runner = _runner(tmp_path, fake)
    runner.prepare(_spec())
    with runner.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial":')

    with pytest.raises(CampaignReconciliationRequired):
        runner.execute(execute_live=True)
    status = runner.status()
    assert status["event_chain_verified"] is False
    assert status["reconciliation_required"] is True


@pytest.mark.parametrize(
    ("terminal_status", "expected_exit"),
    [
        ("FAILED", 1),
        ("HUMAN_RECONCILIATION_REQUIRED", 3),
    ],
)
def test_cli_terminal_failures_have_stable_nonzero_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    terminal_status: str,
    expected_exit: int,
) -> None:
    class StubReceipt:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return {
                "terminal_status": terminal_status,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }

    class StubRunner:
        def execute(self, **_: Any) -> StubReceipt:
            return StubReceipt()

    monkeypatch.setattr(
        real_local_campaign_cli,
        "_runner_from_args",
        lambda _: StubRunner(),
    )
    code = real_local_campaign_cli.main(
        [
            "execute",
            "--campaign-root",
            str(tmp_path),
            "--authority-key-file",
            str(tmp_path / "unused.key"),
            "--execute-live",
        ]
    )

    assert code == expected_exit
    output = json.loads(capsys.readouterr().out)
    assert output["terminal_status"] == terminal_status


def test_cli_prepare_requires_and_writes_authority_anchor(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        canonical_json(
            _spec(
                live_codex=False,
                live_world_bank=False,
            ).model_dump(mode="json")
        )
        + "\n",
        encoding="utf-8",
    )
    key_path = tmp_path / "authority.key"
    key_path.write_bytes(AUTHORITY_KEY)
    campaign_root = tmp_path / "campaign"

    code = real_local_campaign_cli.main(
        [
            "prepare",
            "--campaign-root",
            str(campaign_root),
            "--spec-file",
            str(spec_path),
            "--authority-key-file",
            str(key_path),
        ]
    )

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["prepared"] is True
    assert output["event_chain_verified"] is True
    assert (campaign_root / "campaign_freeze_receipt_v65.json").is_file()

    wrong_key_path = tmp_path / "wrong-authority.key"
    wrong_key_path.write_bytes(b"wrong-authority-key-" + b"x" * 32)
    status_code = real_local_campaign_cli.main(
        [
            "status",
            "--campaign-root",
            str(campaign_root),
            "--authority-key-file",
            str(wrong_key_path),
        ]
    )

    assert status_code == 3
    wrong_status = json.loads(capsys.readouterr().out)
    assert wrong_status["verified_current"] is False
    assert wrong_status["reconciliation_required"] is True
