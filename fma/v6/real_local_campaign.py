"""Replay-aware V6.5 runner for one real local Studio campaign.

The runner is deliberately smaller than the Studio workflow it drives.  It
freezes one public-data campaign before constructing a service, journals an
intent before each stage call, and records a result afterwards.  The journal
does not claim exactly-once execution: an interrupted model or network action
that cannot be resolved from the authoritative workspace is sent to human
reconciliation.

This module grants neither scientific qualification nor real-world action
authority.  A completed receipt establishes, at most, a locally replayed
S0--S6 workflow over one public World Bank series.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator
import numpy as np

from fma._file_lock import exclusive_file_lock
from fma.codex_driver import (
    CodexCLIConfig,
    ProcessOutputLimitExceeded,
    ProcessResult,
    _clean_process_env,
    _default_process_runner,
    discover_codex_cli,
)
from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_1.codex_stage_driver import RoleProcessReceiptV51
from fma.v5.workspace_schemas import ProcessedManifestV50
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v6.provenance import (
    MEASUREMENT_SCHEMA_PATH,
    MeasurementSchemaV62,
    S2TransformReceiptV62,
)
from fma.v6.public_source import (
    SOURCE_CONTRACT_PATH,
    SOURCE_RAW_PATH,
    SOURCE_RECEIPT_PATH,
    SOURCE_VERIFICATION_PATH,
    SourceVerificationV62,
    WorldBankSourceContractV62,
    WorldBankSourceReceiptV62,
    verify_world_bank_source_v62,
)
from fma.v6.source_auth import (
    S2_SOURCE_REVERIFICATION_PATH,
    SOURCE_ACQUISITION_AUTH_PATH,
    S2SourceReverificationReceiptV62,
    SourceAcquisitionReceiptV62,
    SourceTransportAuthorityV62,
)
from fma.studio.backhalf_runtime import (
    PROCESSED_RELATIVE_PATH,
    RAW_RELATIVE_PATH,
    S2_TRANSFORM_RECEIPT_PATH,
    StudioODEDataRequestV59,
)
from fma.studio.service import (
    CreateTaskRequest,
    StudioTaskService,
    StudioWorldBankDataRequestV62,
)


SPEC_PATH_V65 = "campaign_spec_v65.json"
EVENTS_PATH_V65 = "campaign_events_v65.jsonl"
FREEZE_RECEIPT_PATH_V65 = "campaign_freeze_receipt_v65.json"
TERMINAL_RECEIPTS_PATH_V65 = "terminal_receipts_v65.jsonl"
WORKSPACE_ROOT_V65 = "workspace"
LOCK_PATH_V65 = ".real_local_campaign_v65.lock"

ActionV65 = Literal[
    "create_task",
    "run_s0",
    "run_s1",
    "ingest_world_bank_data",
    "run_backhalf",
]
EventTypeV65 = Literal[
    "CAMPAIGN_PREPARED",
    "ACTION_INTENT",
    "ACTION_RESULT",
]
EventStatusV65 = Literal[
    "PREPARED",
    "INTENT_RECORDED",
    "SUCCEEDED",
    "REPLAYED",
    "FAILED",
    "AMBIGUOUS",
]
TerminalStatusV65 = Literal[
    "COMPLETED_LOCAL",
    "COMPLETED_CONTROL",
    "FAILED",
    "HUMAN_RECONCILIATION_REQUIRED",
]

_ACTION_ORDER: tuple[ActionV65, ...] = (
    "create_task",
    "run_s0",
    "run_s1",
    "ingest_world_bank_data",
    "run_backhalf",
)
_SOURCE_PATHS: tuple[str, ...] = (
    RAW_RELATIVE_PATH,
    SOURCE_CONTRACT_PATH,
    SOURCE_RAW_PATH,
    SOURCE_RECEIPT_PATH,
    SOURCE_VERIFICATION_PATH,
    SOURCE_ACQUISITION_AUTH_PATH,
    S2_SOURCE_REVERIFICATION_PATH,
    MEASUREMENT_SCHEMA_PATH,
    PROCESSED_RELATIVE_PATH,
    "data/processed/manifest.json",
    S2_TRANSFORM_RECEIPT_PATH,
)
_SOURCE_INTAKE_PATHS: tuple[str, ...] = (
    RAW_RELATIVE_PATH,
    SOURCE_CONTRACT_PATH,
    SOURCE_RAW_PATH,
    SOURCE_RECEIPT_PATH,
    SOURCE_ACQUISITION_AUTH_PATH,
    MEASUREMENT_SCHEMA_PATH,
)
_SOURCE_S2_PATHS: tuple[str, ...] = (
    SOURCE_VERIFICATION_PATH,
    S2_SOURCE_REVERIFICATION_PATH,
    PROCESSED_RELATIVE_PATH,
    "data/processed/manifest.json",
    S2_TRANSFORM_RECEIPT_PATH,
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_CODE_PATHS: dict[str, str] = {
    "scalar_autonomous_ode_v52": "fma/v5_2/ode_system.py",
    "adaptive_positive_series_v57": (
        "fma/v5_7/adaptive_positive_series.py"
    ),
}
_APPROVED_CODEX_RELEASES_V65: dict[tuple[str, str], str] = {
    (
        "native_codex_cli_v1",
        "0.144.6",
    ): "4b76ded066d0239115ca97473d010c92072bc5c5550a45dd7cbebe1e9eb956a7",
    (
        "wsl_codex_cli_v65",
        "0.145.0-alpha.18",
    ): "16db86b6bf81cc426032fd42216dd97e60f97b149272f1f9963845a0675dae94",
}
_APPROVED_WSL_TRANSPORT_SHA256_V65: frozenset[str] = frozenset(
    {
        # Windows 11 24H2 build 26100.8737. This is a code-owned local
        # allowlist entry, not a Windows publisher-signature attestation.
        "e27cbfcbd61c44796e2cfdd031663245bda8d6e4a43c1451b1fc505333908126",
    }
)
_DECLARED_RUNTIME_DISTRIBUTIONS_V65: tuple[str, ...] = (
    "cryptography",
    "numpy",
    "pydantic",
    "pydantic-core",
    "scipy",
)
_SAFE_REASON = re.compile(r"[^A-Za-z0-9_.-]+")


class RealLocalCampaignError(RuntimeError):
    """The V6.5 campaign could not make a safe transition."""


class CampaignConflictError(RealLocalCampaignError):
    """A frozen campaign or journal conflicts with the requested operation."""


class LiveExecutionNotAuthorized(RealLocalCampaignError):
    """The command-line and frozen-spec live opt-ins are not all present."""


class CampaignReconciliationRequired(RealLocalCampaignError):
    """An interrupted external action cannot be resolved automatically."""


class CampaignRetryRequired(RealLocalCampaignError):
    """A failed or human terminal requires an explicit bounded retry."""


class CampaignStageBlocked(RealLocalCampaignError):
    """A workflow gate legitimately blocked an action's postcondition."""

    def __init__(
        self,
        *,
        action: str,
        stage: str,
        stage_status: str,
        decision: str | None,
        review_verdict: str | None,
        finding_signature: str | None,
        gate_outcome_hash: str | None,
        human_required: bool,
        snapshot: dict[str, Any],
    ) -> None:
        self.action = action
        self.stage = stage
        self.stage_status = stage_status
        self.decision = decision
        self.review_verdict = review_verdict
        self.finding_signature = finding_signature
        self.gate_outcome_hash = gate_outcome_hash
        self.human_required = human_required
        self.snapshot = snapshot
        super().__init__(
            f"{action} reached authenticated {stage} status "
            f"{stage_status}; decision={decision or 'unknown'}"
        )

    def evidence_details(self) -> dict[str, object]:
        return {
            "failure_kind": "STAGE_BLOCKED",
            "action": self.action,
            "stage": self.stage,
            "stage_status": self.stage_status,
            "decision": self.decision,
            "review_verdict": self.review_verdict,
            "finding_signature": self.finding_signature,
            "gate_outcome_hash": self.gate_outcome_hash,
            "human_required": self.human_required,
            "postcondition_verified": False,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_aware(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _reason_code(prefix: str, value: object) -> str:
    cleaned = _SAFE_REASON.sub("_", str(value)).strip("_.-")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"reason_{cleaned or 'unknown'}"
    return f"{prefix}_{cleaned}"[:120]


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_line(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _canonical_system_wsl_executable_v65() -> Path:
    """Resolve wsl.exe from the Windows system-directory API, not PATH."""

    if os.name != "nt":
        raise RealLocalCampaignError(
            "the WSL transport is available only on Windows"
        )
    buffer = ctypes.create_unicode_buffer(32768)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_system_directory = kernel32.GetSystemDirectoryW
    get_system_directory.argtypes = [ctypes.c_wchar_p, ctypes.c_uint]
    get_system_directory.restype = ctypes.c_uint
    length = get_system_directory(buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        raise RealLocalCampaignError(
            "Windows system-directory resolution failed"
        )
    system_directory = Path(buffer.value).resolve(strict=True)
    executable = (system_directory / "wsl.exe").resolve(strict=True)
    if (
        executable.parent != system_directory
        or executable.name.casefold() != "wsl.exe"
    ):
        raise RealLocalCampaignError(
            "canonical Windows WSL executable path is invalid"
        )
    return executable


class RuntimeCodeFileV65(StrictModel):
    relative_path: Annotated[str, Field(min_length=3, max_length=300)]
    sha256: Sha256

    @model_validator(mode="after")
    def validate_path(self) -> "RuntimeCodeFileV65":
        path = Path(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or self.relative_path != path.as_posix()
        ):
            raise ValueError("runtime code path must be safe and normalized")
        return self


class WslOuterTransportContractV65(StrictModel):
    """Frozen outer Windows transport, separate from the inner Codex ELF."""

    schema_version: Literal["6.5-wsl-outer-transport"] = (
        "6.5-wsl-outer-transport"
    )
    canonical_executable_path: Annotated[
        str, Field(min_length=8, max_length=500)
    ]
    executable_sha256: Sha256
    canonical_path_policy: Literal[
        "kernel32_get_system_directory_w_exact_wsl_exe"
    ] = "kernel32_get_system_directory_w_exact_wsl_exe"
    release_fingerprint_policy: Literal[
        "code_owned_local_allowlist_not_windows_signature"
    ] = "code_owned_local_allowlist_not_windows_signature"
    transport_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_transport(self) -> "WslOuterTransportContractV65":
        path = Path(self.canonical_executable_path)
        if not path.is_absolute() or path.name.casefold() != "wsl.exe":
            raise ValueError("outer WSL transport path is not canonical")
        if (
            self.executable_sha256
            not in _APPROVED_WSL_TRANSPORT_SHA256_V65
        ):
            raise ValueError(
                "outer WSL transport is absent from the code-owned "
                "local allowlist"
            )
        if self.transport_hash and self.transport_hash != self.content_hash():
            raise ValueError("outer WSL transport hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "transport_hash")

    def assert_sealed(self) -> None:
        if (
            not self.transport_hash
            or self.transport_hash != self.content_hash()
        ):
            raise ValueError("outer WSL transport is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "WslOuterTransportContractV65":
        draft = cls(**data)
        payload = draft.model_dump(mode="json", exclude={"transport_hash"})
        payload["transport_hash"] = draft.content_hash()
        return cls(**payload)


class CodexRuntimeBudgetsV65(StrictModel):
    timeout_seconds: Annotated[int, Field(ge=1)]
    max_candidates: Annotated[int, Field(ge=1, le=3)]
    max_input_bytes: Annotated[int, Field(ge=1)]
    max_schema_bytes: Annotated[int, Field(ge=1)]
    max_stdout_bytes: Annotated[int, Field(ge=1)]
    max_stderr_bytes: Annotated[int, Field(ge=1)]
    max_jsonl_line_bytes: Annotated[int, Field(ge=1)]
    max_events: Annotated[int, Field(ge=1)]
    max_oracle_assignments: Annotated[int, Field(ge=1)]
    s1_max_model_calls: Literal[16] = 16
    campaign_action_limit: Literal[5] = 5

    @classmethod
    def from_config(cls, config: CodexCLIConfig) -> "CodexRuntimeBudgetsV65":
        return cls(
            timeout_seconds=config.timeout_seconds,
            max_candidates=config.max_candidates,
            max_input_bytes=config.max_input_bytes,
            max_schema_bytes=config.max_schema_bytes,
            max_stdout_bytes=config.max_stdout_bytes,
            max_stderr_bytes=config.max_stderr_bytes,
            max_jsonl_line_bytes=config.max_jsonl_line_bytes,
            max_events=config.max_events,
            max_oracle_assignments=config.max_oracle_assignments,
        )


class CodexRuntimeContractV65(StrictModel):
    """Frozen identity and budgets for every Codex role process."""

    schema_version: Literal["6.5-codex-runtime-contract"] = (
        "6.5-codex-runtime-contract"
    )
    provider: Literal[
        "openai_codex_cli",
        "openai_codex_cli_via_wsl",
    ]
    runtime_adapter_id: Literal[
        "native_codex_cli_v1",
        "wsl_codex_cli_v65",
    ]
    requested_model: Annotated[str, Field(min_length=1, max_length=200)]
    expected_cli_version: Annotated[str, Field(min_length=1, max_length=100)]
    observed_cli_version: Annotated[str, Field(min_length=1, max_length=100)]
    executable_name: Annotated[str, Field(min_length=1, max_length=300)]
    executable_sha256: Sha256
    runtime_identity: dict[str, Any]
    runtime_identity_hash: Sha256
    outer_wsl_transport: WslOuterTransportContractV65 | None = None
    budgets: CodexRuntimeBudgetsV65
    source_adapter_id: Literal[
        "scalar_autonomous_ode_v52",
        "adaptive_positive_series_v57",
    ]
    code_manifest: Annotated[
        list[RuntimeCodeFileV65], Field(min_length=1, max_length=4096)
    ]
    code_manifest_hash: Sha256
    release_fingerprint_policy: Literal[
        "code_owned_local_allowlist_not_publisher_attestation"
    ] = "code_owned_local_allowlist_not_publisher_attestation"
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "CodexRuntimeContractV65":
        if self.requested_model == "cli_default":
            raise ValueError("real execution requires an explicit requested model")
        if self.observed_cli_version != self.expected_cli_version:
            raise ValueError("observed Codex CLI version differs from expectation")
        if self.runtime_identity_hash != sha256_value(self.runtime_identity):
            raise ValueError("Codex runtime identity hash differs")
        local_runtime = self.runtime_identity.get("local_python_runtime")
        if not isinstance(local_runtime, dict):
            raise ValueError("Codex runtime lacks local Python identity")
        dependencies = local_runtime.get("declared_runtime_distributions")
        if (
            not isinstance(dependencies, dict)
            or set(dependencies) != set(_DECLARED_RUNTIME_DISTRIBUTIONS_V65)
            or any(
                not isinstance(version, str) or not version
                for version in dependencies.values()
            )
        ):
            raise ValueError(
                "local Python identity lacks exact declared dependencies"
            )
        required_identity_fields = (
            "python_implementation",
            "python_version",
            "python_cache_tag",
            "python_executable_sha256",
            "os_system",
            "os_release",
            "os_version",
            "machine",
            "architecture",
            "numpy_build_identity",
            "claim_ceiling",
        )
        if any(
            field not in local_runtime
            for field in required_identity_fields
        ):
            raise ValueError("local Python runtime identity is incomplete")
        paths = [item.relative_path for item in self.code_manifest]
        if paths != sorted(set(paths)):
            raise ValueError("runtime code manifest must be sorted and unique")
        if self.code_manifest_hash != sha256_value(
            [item.model_dump(mode="json") for item in self.code_manifest]
        ):
            raise ValueError("runtime code manifest hash differs")
        expected_adapter = (
            "native_codex_cli_v1"
            if self.provider == "openai_codex_cli"
            else "wsl_codex_cli_v65"
        )
        if self.runtime_adapter_id != expected_adapter:
            raise ValueError("runtime adapter differs from provider")
        if self.provider == "openai_codex_cli_via_wsl":
            if self.outer_wsl_transport is None:
                raise ValueError(
                    "WSL runtime lacks its frozen outer transport"
                )
            self.outer_wsl_transport.assert_sealed()
        elif self.outer_wsl_transport is not None:
            raise ValueError(
                "native Codex runtime cannot contain an outer WSL transport"
            )
        approved_hash = _APPROVED_CODEX_RELEASES_V65.get(
            (self.runtime_adapter_id, self.observed_cli_version)
        )
        if (
            approved_hash is None
            or self.executable_sha256 != approved_hash
        ):
            raise ValueError(
                "Codex executable is absent from the code-owned "
                "local release allowlist"
            )
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("Codex runtime contract hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("Codex runtime contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "CodexRuntimeContractV65":
        draft = cls(**data)
        payload = draft.model_dump(mode="json", exclude={"contract_hash"})
        payload["contract_hash"] = draft.content_hash()
        return cls(**payload)


def _runtime_code_manifest(
    source_adapter_id: str,
    runtime_adapter_id: str = "native_codex_cli_v1",
) -> list[RuntimeCodeFileV65]:
    try:
        adapter_path = _ADAPTER_CODE_PATHS[source_adapter_id]
    except KeyError as exc:
        raise ValueError("unsupported source adapter for runtime contract") from exc
    del runtime_adapter_id
    paths = sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for path in (_REPO_ROOT / "fma").rglob("*.py")
        if path.is_file()
    )
    if adapter_path not in paths:
        raise ValueError("source adapter is absent from the FMA code tree")
    manifest: list[RuntimeCodeFileV65] = []
    for relative_path in paths:
        path = (_REPO_ROOT / relative_path).resolve(strict=True)
        path.relative_to(_REPO_ROOT)
        manifest.append(
            RuntimeCodeFileV65(
                relative_path=relative_path,
                sha256=_file_hash(path),
            )
        )
    return manifest


def _assert_runtime_code_manifest_current(
    contract: CodexRuntimeContractV65,
) -> list[RuntimeCodeFileV65]:
    current = _runtime_code_manifest(
        contract.source_adapter_id,
        contract.runtime_adapter_id,
    )
    if current != contract.code_manifest:
        raise RealLocalCampaignError(
            "current FMA Python code differs from the frozen runtime manifest"
        )
    return current


def _numpy_build_identity_v65() -> dict[str, Any]:
    try:
        config = np.__config__.show(mode="dicts")
    except TypeError:  # pragma: no cover - legacy NumPy fallback
        output = io.StringIO()
        with redirect_stdout(output):
            np.__config__.show()
        return {
            "capture_status": "TEXT_FINGERPRINT_ONLY",
            "config_hash": sha256_value(output.getvalue()),
            "blas": None,
            "lapack": None,
            "claim_ceiling": (
                "numpy_build_text_fingerprint_only_not_complete_"
                "environment"
            ),
        }
    if not isinstance(config, dict):
        raise RealLocalCampaignError(
            "NumPy build configuration is not machine-readable"
        )
    build_dependencies = config.get("Build Dependencies", {})
    if not isinstance(build_dependencies, dict):
        build_dependencies = {}

    def dependency_summary(name: str) -> dict[str, Any] | None:
        value = build_dependencies.get(name)
        if not isinstance(value, dict):
            return None
        return {
            key: value.get(key)
            for key in (
                "name",
                "found",
                "version",
                "detection method",
                "openblas configuration",
            )
        }

    return {
        "capture_status": "MACHINE_READABLE",
        "config_hash": sha256_value(config),
        "blas": dependency_summary("blas"),
        "lapack": dependency_summary("lapack"),
        "claim_ceiling": (
            "numpy_build_fingerprint_not_complete_environment_or_"
            "supply_chain_attestation"
        ),
    }


def _local_python_runtime_identity() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for distribution in _DECLARED_RUNTIME_DISTRIBUTIONS_V65:
        try:
            dependencies[distribution] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError as exc:
            raise RealLocalCampaignError(
                f"required runtime dependency is unavailable: {distribution}"
            ) from exc
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable_sha256": _file_hash(
            Path(sys.executable).resolve(strict=True)
        ),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "declared_runtime_distributions": dependencies,
        "numpy_build_identity": _numpy_build_identity_v65(),
        "claim_ceiling": (
            "process_and_declared_dependency_fingerprint_not_complete_"
            "environment_or_supply_chain_attestation"
        ),
    }


def _assert_local_python_runtime_identity_current(
    contract: CodexRuntimeContractV65,
) -> None:
    frozen = contract.runtime_identity.get("local_python_runtime")
    if frozen != _local_python_runtime_identity():
        raise RealLocalCampaignError(
            "current Python or declared dependency runtime differs from "
            "the frozen contract"
        )


def _trusted_runtime_adapter(
    process_runner: Any | None,
    cli_locator: Callable[..., Path] | None,
) -> tuple[
    Literal["openai_codex_cli", "openai_codex_cli_via_wsl"],
    Literal["native_codex_cli_v1", "wsl_codex_cli_v65"],
    dict[str, Any],
    WslOuterTransportContractV65 | None,
]:
    if process_runner is None and cli_locator is None:
        return (
            "openai_codex_cli",
            "native_codex_cli_v1",
            {
                "transport": "native_process",
                "process_runner": "fma.codex_driver._default_process_runner",
                "cli_locator": "fma.codex_driver.discover_codex_cli",
            },
            None,
        )
    try:
        from fma.codex_wsl import WslCodexRuntimeV65
    except ImportError as exc:  # pragma: no cover - installation corruption
        raise RealLocalCampaignError("WSL runtime adapter is unavailable") from exc
    locator_owner = getattr(cli_locator, "__self__", None)
    locator_function = getattr(cli_locator, "__func__", None)
    if (
        type(process_runner) is WslCodexRuntimeV65
        and locator_owner is process_runner
        and locator_function is WslCodexRuntimeV65.locate
    ):
        try:
            configured_outer = Path(
                process_runner._wsl_executable
            ).resolve(strict=True)
            canonical_outer = _canonical_system_wsl_executable_v65()
            initialized = bool(
                process_runner._temporary is not None
                and Path(process_runner._root).is_dir()
                and Path(process_runner._private_codex_home).is_dir()
                and isinstance(process_runner._staged_by_source, dict)
            )
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise RealLocalCampaignError(
                "WSL runtime adapter is not fully initialized"
            ) from exc
        if not initialized:
            raise RealLocalCampaignError(
                "WSL runtime adapter is not fully initialized"
            )
        if os.path.normcase(str(configured_outer)) != os.path.normcase(
            str(canonical_outer)
        ):
            raise RealLocalCampaignError(
                "WSL runtime must use the canonical Windows system executable"
            )
        outer_sha256 = _file_hash(canonical_outer)
        if outer_sha256 not in _APPROVED_WSL_TRANSPORT_SHA256_V65:
            raise RealLocalCampaignError(
                "outer WSL transport is absent from the code-owned "
                "local allowlist"
            )
        outer_contract = WslOuterTransportContractV65.seal(
            canonical_executable_path=str(canonical_outer),
            executable_sha256=outer_sha256,
        )
        runtime_identity = {
            "adapter_class": "fma.codex_wsl.WslCodexRuntimeV65",
            **process_runner.runtime_identity,
            "wsl_executable": str(canonical_outer),
            "wsl_executable_sha256": outer_sha256,
            "outer_transport_hash": outer_contract.transport_hash,
        }
        return (
            "openai_codex_cli_via_wsl",
            "wsl_codex_cli_v65",
            runtime_identity,
            outer_contract,
        )
    raise RealLocalCampaignError(
        "real mode rejects injected Codex runner or locator"
    )


def _locate_runtime_executable(
    *,
    config: CodexCLIConfig,
    process_runner: Any | None,
    cli_locator: Callable[..., Path] | None,
    runtime_adapter_id: str,
) -> Path:
    if runtime_adapter_id == "native_codex_cli_v1":
        return discover_codex_cli(config.executable)
    if process_runner is None or cli_locator is None:
        raise RealLocalCampaignError("WSL runtime adapter is incomplete")
    return Path(cli_locator(config.executable)).resolve(strict=True)


def _runtime_process_call(
    argv: list[str],
    *,
    cwd: Path,
    process_runner: Any | None,
) -> ProcessResult:
    runner = process_runner or _default_process_runner
    return cast(Any, runner)(
        argv,
        cwd=cwd,
        input_text=None,
        timeout_seconds=10,
        env=_clean_process_env(),
        max_stdout_bytes=64 * 1024,
        max_stderr_bytes=64 * 1024,
    )


def _build_codex_runtime_contract(
    *,
    config: CodexCLIConfig,
    source_adapter_id: str,
    budgets: CodexRuntimeBudgetsV65,
    contract_type: type[CodexRuntimeContractV65],
    process_runner: Any | None = None,
    cli_locator: Callable[..., Path] | None = None,
) -> CodexRuntimeContractV65:
    """Inspect and freeze one version-selected trusted Codex runtime."""

    if not config.requested_model:
        raise RealLocalCampaignError(
            "real Codex runtime requires an explicit requested model"
        )
    (
        provider,
        runtime_adapter_id,
        runtime_identity,
        outer_wsl_transport,
    ) = _trusted_runtime_adapter(process_runner, cli_locator)
    runtime_identity = {
        **runtime_identity,
        "local_python_runtime": _local_python_runtime_identity(),
    }
    executable = _locate_runtime_executable(
        config=config,
        process_runner=process_runner,
        cli_locator=cli_locator,
        runtime_adapter_id=runtime_adapter_id,
    )
    try:
        result = _runtime_process_call(
            [str(executable), "--version"],
            cwd=executable.parent,
            process_runner=process_runner,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
        ProcessOutputLimitExceeded,
    ) as exc:
        raise RealLocalCampaignError(
            f"Codex runtime version check failed: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        raise RealLocalCampaignError(
            "Codex runtime version check returned nonzero"
        )
    observed_version = result.stdout.strip().removeprefix("codex-cli ").strip()
    manifest = _runtime_code_manifest(
        source_adapter_id, runtime_adapter_id
    )
    return contract_type.seal(
        provider=provider,
        runtime_adapter_id=runtime_adapter_id,
        requested_model=config.requested_model,
        expected_cli_version=config.expected_cli_version,
        observed_cli_version=observed_version,
        executable_name=executable.name,
        executable_sha256=_file_hash(executable),
        runtime_identity=runtime_identity,
        runtime_identity_hash=sha256_value(runtime_identity),
        outer_wsl_transport=outer_wsl_transport,
        budgets=budgets.model_dump(mode="json"),
        source_adapter_id=source_adapter_id,
        code_manifest=manifest,
        code_manifest_hash=sha256_value(
            [item.model_dump(mode="json") for item in manifest]
        ),
    )


def build_codex_runtime_contract_v65(
    *,
    config: CodexCLIConfig,
    source_adapter_id: str,
    process_runner: Any | None = None,
    cli_locator: Callable[..., Path] | None = None,
) -> CodexRuntimeContractV65:
    """Inspect, version-check, and freeze one trusted V6.5 Codex runtime."""

    return _build_codex_runtime_contract(
        config=config,
        source_adapter_id=source_adapter_id,
        budgets=CodexRuntimeBudgetsV65.from_config(config),
        contract_type=CodexRuntimeContractV65,
        process_runner=process_runner,
        cli_locator=cli_locator,
    )


class RealLocalCampaignSpecV65(StrictModel):
    """Frozen public inputs and explicit permission bits for one campaign."""

    schema_version: Literal["6.5-real-local-campaign-spec"] = (
        "6.5-real-local-campaign-spec"
    )
    campaign_id: Identifier
    task_id: Identifier
    objective: Annotated[str, Field(min_length=12, max_length=4000)]
    world_bank_request: StudioWorldBankDataRequestV62
    evidence_scope: Literal["public_data"] = "public_data"
    execution_mode: Literal["real", "control"] = "real"
    live_codex: bool = False
    live_world_bank: bool = False
    codex_runtime_contract: CodexRuntimeContractV65 | None = None
    execution_semantics: Literal[
        "intent_logged_replay_checked_not_exactly_once"
    ] = "intent_logged_replay_checked_not_exactly_once"
    max_execution_attempts: Literal[3] = 3
    private_acceptance_data_permitted: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    spec_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_spec(self) -> "RealLocalCampaignSpecV65":
        if self.world_bank_request.fixture_only:
            raise ValueError(
                "a real-local campaign spec cannot request fixture source data"
            )
        if self.codex_runtime_contract is not None:
            self.codex_runtime_contract.assert_sealed()
            if not self.live_codex:
                raise ValueError(
                    "Codex runtime contract requires live_codex=true"
                )
            if (
                self.codex_runtime_contract.source_adapter_id
                != self.world_bank_request.adapter_id
            ):
                raise ValueError(
                    "Codex runtime source adapter differs from campaign request"
                )
        if (
            self.execution_mode == "real"
            and self.live_codex
            and self.codex_runtime_contract is None
        ):
            raise ValueError(
                "a live real campaign requires a frozen Codex runtime contract"
            )
        if (
            self.execution_mode == "control"
            and self.codex_runtime_contract is not None
        ):
            raise ValueError(
                "a control campaign cannot freeze a real Codex runtime contract"
            )
        if self.spec_hash and self.spec_hash != self.content_hash():
            raise ValueError("real-local campaign spec hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "spec_hash")

    def assert_sealed(self) -> None:
        if not self.spec_hash or self.spec_hash != self.content_hash():
            raise ValueError("real-local campaign spec is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RealLocalCampaignSpecV65":
        draft = cls(**data)
        payload = draft.model_dump(mode="json", exclude={"spec_hash"})
        payload["spec_hash"] = draft.content_hash()
        return cls(**payload)


class RealLocalCampaignEventV65(StrictModel):
    """One append-only campaign event linked to its predecessor."""

    schema_version: Literal["6.5-real-local-campaign-event"] = (
        "6.5-real-local-campaign-event"
    )
    campaign_id: Identifier
    sequence: Annotated[int, Field(ge=1)]
    event_type: EventTypeV65
    status: EventStatusV65
    action_id: Identifier | None = None
    action: ActionV65 | None = None
    request_hash: Sha256 | None = None
    intent_event_hash: Sha256 | None = None
    result_hash: Sha256 | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime
    previous_event_hash: Sha256 | None = None
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    event_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_event(self) -> "RealLocalCampaignEventV65":
        _assert_aware(self.recorded_at, "recorded_at")
        if self.event_type == "CAMPAIGN_PREPARED":
            if self.status != "PREPARED" or any(
                value is not None
                for value in (
                    self.action_id,
                    self.action,
                    self.request_hash,
                    self.intent_event_hash,
                    self.result_hash,
                )
            ):
                raise ValueError("campaign-prepared event has action fields")
        elif self.event_type == "ACTION_INTENT":
            if (
                self.status != "INTENT_RECORDED"
                or self.action_id is None
                or self.action is None
                or self.request_hash is None
                or self.intent_event_hash is not None
                or self.result_hash is not None
            ):
                raise ValueError("action intent fields are inconsistent")
        elif (
            self.action_id is None
            or self.action is None
            or self.request_hash is None
            or self.intent_event_hash is None
            or self.result_hash is None
            or self.status
            not in {"SUCCEEDED", "REPLAYED", "FAILED", "AMBIGUOUS"}
        ):
            raise ValueError("action result fields are inconsistent")
        if self.event_hash and self.event_hash != self.content_hash():
            raise ValueError("real-local campaign event hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "event_hash")

    def assert_sealed(self) -> None:
        if not self.event_hash or self.event_hash != self.content_hash():
            raise ValueError("real-local campaign event is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "RealLocalCampaignEventV65":
        draft = cls(**data)
        payload = draft.model_dump(mode="json", exclude={"event_hash"})
        payload["event_hash"] = draft.content_hash()
        return cls(**payload)


class RealLocalCampaignFreezeReceiptV65(StrictModel):
    """Authority anchor for the exact frozen spec and journal genesis."""

    schema_version: Literal["6.5-real-local-campaign-freeze"] = (
        "6.5-real-local-campaign-freeze"
    )
    audience: Literal["fma.real-local-campaign.v65.prepare"] = (
        "fma.real-local-campaign.v65.prepare"
    )
    campaign_id: Identifier
    task_id: Identifier
    spec_hash: Sha256
    genesis_event_hash: Sha256
    authority_key_id: Identifier
    frozen_at: datetime
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "RealLocalCampaignFreezeReceiptV65":
        _assert_aware(self.frozen_at, "frozen_at")
        if self.receipt_hash and not self.authority_auth_tag:
            raise ValueError(
                "campaign freeze receipt requires an authority auth tag"
            )
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("campaign freeze receipt hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self,
            "authority_auth_tag",
            "receipt_hash",
        )

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_sealed(
        self,
        *,
        authority_key: bytes,
        authority_key_id: str,
    ) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("campaign freeze receipt is not sealed")
        expected_tag = hmac.new(
            authority_key,
            f"real-local-freeze:{self.unsigned_hash()}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if (
            self.authority_key_id != authority_key_id
            or not self.authority_auth_tag
            or not hmac.compare_digest(
                self.authority_auth_tag,
                expected_tag,
            )
        ):
            raise ValueError("campaign freeze authority differs")

    @classmethod
    def seal(
        cls,
        *,
        authority_key: bytes,
        **data: object,
    ) -> "RealLocalCampaignFreezeReceiptV65":
        unsigned = cls(**data)
        payload = unsigned.model_dump(
            mode="json",
            exclude={"authority_auth_tag", "receipt_hash"},
        )
        payload["authority_auth_tag"] = hmac.new(
            authority_key,
            f"real-local-freeze:{unsigned.unsigned_hash()}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        tagged = cls(**payload)
        payload = tagged.model_dump(mode="json", exclude={"receipt_hash"})
        payload["receipt_hash"] = tagged.content_hash()
        return cls(**payload)


class RealLocalCampaignTerminalReceiptV65(StrictModel):
    """Claim-limited terminal record for one bounded execution attempt."""

    schema_version: Literal["6.5-real-local-campaign-terminal"] = (
        "6.5-real-local-campaign-terminal"
    )
    execution_id: Identifier
    attempt_index: Annotated[int, Field(ge=1, le=3)]
    previous_receipt_hash: Sha256 | None = None
    campaign_id: Identifier
    task_id: Identifier
    spec_hash: Sha256
    terminal_status: TerminalStatusV65
    reason_codes: list[Identifier] = Field(default_factory=list)
    completed_actions: list[ActionV65] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime
    last_event_hash: Sha256
    pending_intent_hash: Literal[None] = None
    snapshot_hash: Sha256 | None = None
    workspace_spec_hash: Sha256 | None = None
    runtime_contract_hash: Sha256 | None = None
    source_evidence_hash: Sha256 | None = None
    studio_event_tip_hash: Sha256 | None = None
    campaign_event_chain_verified: bool
    workspace_verified: bool
    studio_event_chain_verified: bool
    snapshot_fixture_only: bool | None
    workflow_complete: bool
    fixture_or_control: bool
    external_scientific_qualification_status: Literal["NOT_RUN"] = "NOT_RUN"
    claim_ceiling: Literal[
        "no_scientific_claim",
        "control_protocol_only",
        "local_workflow_evidence_only",
    ]
    receipt_is_authority: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authority_key_id: Identifier
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "RealLocalCampaignTerminalReceiptV65":
        _assert_aware(self.started_at, "started_at")
        _assert_aware(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("terminal receipt finishes before it starts")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("terminal reason codes must be sorted and unique")
        if len(self.completed_actions) != len(set(self.completed_actions)):
            raise ValueError("completed actions must be unique")
        indices = [_ACTION_ORDER.index(item) for item in self.completed_actions]
        if indices != sorted(indices):
            raise ValueError("completed actions are out of workflow order")
        if (self.attempt_index == 1) != (
            self.previous_receipt_hash is None
        ):
            raise ValueError(
                "terminal receipt predecessor differs from attempt index"
            )
        if self.fixture_or_control:
            if self.terminal_status == "COMPLETED_LOCAL":
                raise ValueError("control execution cannot be completed-local")
            if self.claim_ceiling != "control_protocol_only":
                raise ValueError("control execution has an invalid claim ceiling")
        elif self.terminal_status == "COMPLETED_CONTROL":
            raise ValueError("real execution cannot be completed-control")
        if self.terminal_status == "COMPLETED_LOCAL":
            if not (
                self.completed_actions == list(_ACTION_ORDER)
                and not self.reason_codes
                and self.campaign_event_chain_verified
                and self.workspace_verified
                and self.studio_event_chain_verified
                and self.snapshot_fixture_only is False
                and self.workflow_complete
                and self.claim_ceiling == "local_workflow_evidence_only"
                and self.runtime_contract_hash is not None
                and self.source_evidence_hash is not None
                and self.studio_event_tip_hash is not None
                and self.snapshot_hash is not None
                and self.workspace_spec_hash is not None
            ):
                raise ValueError(
                    "completed-local receipt lacks real local verification"
                )
        if self.receipt_hash and not self.authority_auth_tag:
            raise ValueError(
                "terminal receipt hash requires an authority auth tag"
            )
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("real-local terminal receipt hash differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(
            self, "authority_auth_tag", "receipt_hash"
        )

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    def assert_content_sealed(self) -> None:
        if not self.receipt_hash or self.receipt_hash != self.content_hash():
            raise ValueError("real-local terminal receipt is not sealed")

    def assert_sealed(
        self,
        *,
        authority_key: bytes,
        authority_key_id: str,
    ) -> None:
        self.assert_content_sealed()
        expected_tag = hmac.new(
            authority_key,
            f"real-local-terminal:{self.unsigned_hash()}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if (
            self.authority_key_id != authority_key_id
            or not self.authority_auth_tag
            or not hmac.compare_digest(
                self.authority_auth_tag, expected_tag
            )
        ):
            raise ValueError(
                "real-local terminal receipt authority differs"
            )

    @classmethod
    def seal(
        cls,
        *,
        authority_key: bytes,
        **data: object,
    ) -> "RealLocalCampaignTerminalReceiptV65":
        unsigned = cls(**data)
        payload = unsigned.model_dump(
            mode="json",
            exclude={"authority_auth_tag", "receipt_hash"},
        )
        payload["authority_auth_tag"] = hmac.new(
            authority_key,
            f"real-local-terminal:{unsigned.unsigned_hash()}".encode(
                "utf-8"
            ),
            hashlib.sha256,
        ).hexdigest()
        draft = cls(**payload)
        payload = draft.model_dump(mode="json", exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


# Short aliases make the public concepts explicit without hiding the version.
CampaignSpec = RealLocalCampaignSpecV65
CampaignEvent = RealLocalCampaignEventV65
FreezeReceipt = RealLocalCampaignFreezeReceiptV65
TerminalReceipt = RealLocalCampaignTerminalReceiptV65


ServiceFactoryV65 = Callable[..., Any]


class RealLocalCampaignRunnerV65:
    """Prepare, execute, replay, and verify one bounded Studio campaign."""

    def __init__(
        self,
        campaign_root: str | Path,
        *,
        authority_key: bytes | None = None,
        authority_key_id: str = "real-local-v65",
        codex_config: CodexCLIConfig | None = None,
        codex_process_runner: Any | None = None,
        codex_cli_locator: Callable[[], Path] | None = None,
        service_factory: ServiceFactoryV65 | None = None,
    ) -> None:
        self.root = Path(campaign_root).resolve()
        self.authority_key = (
            bytes(authority_key) if authority_key is not None else None
        )
        self.authority_key_id = authority_key_id
        self.codex_config = codex_config
        self.codex_process_runner = codex_process_runner
        self.codex_cli_locator = codex_cli_locator
        self.service_factory = service_factory
        self._service_instance: Any | None = None

    @property
    def spec_path(self) -> Path:
        return self.root / SPEC_PATH_V65

    @property
    def events_path(self) -> Path:
        return self.root / EVENTS_PATH_V65

    @property
    def freeze_receipt_path(self) -> Path:
        return self.root / FREEZE_RECEIPT_PATH_V65

    @property
    def terminal_receipts_path(self) -> Path:
        return self.root / TERMINAL_RECEIPTS_PATH_V65

    @property
    def workspace_root(self) -> Path:
        return self.root / WORKSPACE_ROOT_V65

    @property
    def lock_path(self) -> Path:
        return self.root / LOCK_PATH_V65

    @property
    def action_order(self) -> tuple[ActionV65, ...]:
        return _ACTION_ORDER

    @staticmethod
    def _runtime_budgets_from_config(
        config: CodexCLIConfig,
    ) -> CodexRuntimeBudgetsV65:
        return CodexRuntimeBudgetsV65.from_config(config)

    @staticmethod
    def _seal_runtime_contract(
        **data: object,
    ) -> CodexRuntimeContractV65:
        return CodexRuntimeContractV65.seal(**data)

    def prepare(
        self,
        spec: RealLocalCampaignSpecV65 | dict[str, Any],
    ) -> RealLocalCampaignSpecV65:
        """Authority-freeze the spec without service or live calls."""

        if isinstance(spec, RealLocalCampaignSpecV65):
            validated = spec
            validated.assert_sealed()
        else:
            payload = dict(spec)
            supplied_hash = payload.pop("spec_hash", None)
            validated = RealLocalCampaignSpecV65.seal(**payload)
            if supplied_hash is not None and supplied_hash != validated.spec_hash:
                raise CampaignConflictError("supplied campaign spec hash differs")
        authority_key = self._authority_key_required()
        self.root.mkdir(parents=True, exist_ok=True)
        with exclusive_file_lock(self.lock_path):
            core_paths = (
                self.spec_path,
                self.events_path,
                self.freeze_receipt_path,
            )
            material_state = any(
                path.exists()
                for path in (
                    *core_paths,
                    self.terminal_receipts_path,
                    self.workspace_root,
                )
            )
            if material_state and not all(path.is_file() for path in core_paths):
                raise CampaignConflictError(
                    "pre-anchor or partial campaign state cannot be silently "
                    "upgraded; create a new campaign root"
                )
            if material_state:
                existing = self.load_spec()
                if existing != validated:
                    raise CampaignConflictError(
                        "campaign root already contains another frozen spec"
                    )
            else:
                _write_new(
                    self.spec_path,
                    validated.model_dump(mode="json"),
                )
                self._append_event_locked(
                    validated,
                    event_type="CAMPAIGN_PREPARED",
                    status="PREPARED",
                    details={
                        "spec_hash": validated.spec_hash,
                        "execution_mode": validated.execution_mode,
                        "live_codex": validated.live_codex,
                        "live_world_bank": validated.live_world_bank,
                        "default_live_execution": False,
                        "execution_semantics": validated.execution_semantics,
                        "max_execution_attempts": (
                            validated.max_execution_attempts
                        ),
                    },
                )
                events = self._read_events(validated)
                if len(events) != 1 or events[0].event_hash is None:
                    raise CampaignConflictError(
                        "new campaign failed to create one exact genesis"
                    )
                freeze_receipt = RealLocalCampaignFreezeReceiptV65.seal(
                    authority_key=authority_key,
                    campaign_id=validated.campaign_id,
                    task_id=validated.task_id,
                    spec_hash=validated.spec_hash,
                    genesis_event_hash=events[0].event_hash,
                    authority_key_id=self.authority_key_id,
                    frozen_at=_utc_now(),
                )
                _write_new(
                    self.freeze_receipt_path,
                    freeze_receipt.model_dump(mode="json"),
                )
        return validated

    def load_spec(self) -> RealLocalCampaignSpecV65:
        try:
            spec = RealLocalCampaignSpecV65.model_validate_json(
                self.spec_path.read_text(encoding="utf-8")
            )
            spec.assert_sealed()
            events = self._read_events(spec)
            if not events or events[0].event_hash is None:
                raise CampaignConflictError(
                    "campaign lacks its exact prepared genesis"
                )
            self._read_freeze_receipt(spec, events[0])
            return spec
        except FileNotFoundError as exc:
            raise RealLocalCampaignError("campaign is not prepared") from exc

    def _read_freeze_receipt(
        self,
        spec: RealLocalCampaignSpecV65,
        genesis: RealLocalCampaignEventV65,
    ) -> RealLocalCampaignFreezeReceiptV65:
        receipt = RealLocalCampaignFreezeReceiptV65.model_validate(
            _read_json_object(self.freeze_receipt_path)
        )
        receipt.assert_sealed(
            authority_key=self._authority_key_required(),
            authority_key_id=self.authority_key_id,
        )
        if (
            genesis.event_type != "CAMPAIGN_PREPARED"
            or genesis.sequence != 1
            or genesis.previous_event_hash is not None
            or genesis.event_hash is None
            or receipt.campaign_id != spec.campaign_id
            or receipt.task_id != spec.task_id
            or receipt.spec_hash != spec.spec_hash
            or receipt.genesis_event_hash != genesis.event_hash
        ):
            raise CampaignConflictError(
                "campaign freeze receipt differs from the exact frozen "
                "spec or genesis"
            )
        return receipt

    def _read_events(
        self,
        spec: RealLocalCampaignSpecV65 | None = None,
    ) -> list[RealLocalCampaignEventV65]:
        if not self.events_path.is_file():
            return []
        expected_spec = spec or self.load_spec()
        events: list[RealLocalCampaignEventV65] = []
        previous: str | None = None
        intents: dict[str, RealLocalCampaignEventV65] = {}
        resolved: set[str] = set()
        for raw in self.events_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                raise CampaignConflictError("campaign journal contains blank line")
            event = RealLocalCampaignEventV65.model_validate_json(raw)
            event.assert_sealed()
            if event.campaign_id != expected_spec.campaign_id:
                raise CampaignConflictError("campaign journal ID differs")
            if event.sequence != len(events) + 1:
                raise CampaignConflictError("campaign journal sequence differs")
            if event.previous_event_hash != previous:
                raise CampaignConflictError("campaign journal predecessor differs")
            if event.event_type == "ACTION_INTENT":
                assert event.action_id is not None
                if event.action_id in intents:
                    raise CampaignConflictError("duplicate campaign action ID")
                intents[event.action_id] = event
            elif event.event_type == "ACTION_RESULT":
                assert event.action_id is not None
                intent = intents.get(event.action_id)
                if (
                    intent is None
                    or event.action_id in resolved
                    or event.intent_event_hash != intent.event_hash
                    or event.action != intent.action
                    or event.request_hash != intent.request_hash
                ):
                    raise CampaignConflictError(
                        "campaign action result does not bind one pending intent"
                    )
                resolved.add(event.action_id)
            events.append(event)
            previous = event.event_hash
        if events and events[0].event_type != "CAMPAIGN_PREPARED":
            raise CampaignConflictError(
                "campaign prepared event must be journal genesis"
            )
        prepared = [
            event
            for event in events
            if event.event_type == "CAMPAIGN_PREPARED"
        ]
        if events and len(prepared) != 1:
            raise CampaignConflictError(
                "campaign journal must contain exactly one prepared genesis"
            )
        if events:
            expected_genesis = {
                "spec_hash": expected_spec.spec_hash,
                "execution_mode": expected_spec.execution_mode,
                "live_codex": expected_spec.live_codex,
                "live_world_bank": expected_spec.live_world_bank,
                "default_live_execution": False,
                "execution_semantics": expected_spec.execution_semantics,
                "max_execution_attempts": (
                    expected_spec.max_execution_attempts
                ),
            }
            if events[0].details != expected_genesis:
                raise CampaignConflictError(
                    "campaign genesis differs from the exact frozen spec"
                )
        if len(set(intents) - resolved) > 1:
            raise CampaignConflictError(
                "campaign has more than one unresolved action intent"
            )
        return events

    def _append_event_locked(
        self,
        spec: RealLocalCampaignSpecV65,
        *,
        event_type: EventTypeV65,
        status: EventStatusV65,
        action_id: str | None = None,
        action: ActionV65 | None = None,
        request_hash: str | None = None,
        intent_event_hash: str | None = None,
        result_hash: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> RealLocalCampaignEventV65:
        events = self._read_events(spec)
        event = RealLocalCampaignEventV65.seal(
            campaign_id=spec.campaign_id,
            sequence=len(events) + 1,
            event_type=event_type,
            status=status,
            action_id=action_id,
            action=action,
            request_hash=request_hash,
            intent_event_hash=intent_event_hash,
            result_hash=result_hash,
            details=details or {},
            recorded_at=_utc_now(),
            previous_event_hash=events[-1].event_hash if events else None,
        )
        _append_line(self.events_path, event.model_dump(mode="json"))
        return event

    def _append_intent(
        self,
        spec: RealLocalCampaignSpecV65,
        *,
        action: ActionV65,
        request: object,
        execution_id: str,
        attempt_index: int,
        predecessor_receipt_hash: str | None,
    ) -> RealLocalCampaignEventV65:
        with exclusive_file_lock(self.lock_path):
            events = self._read_events(spec)
            if self._pending_intent(events) is not None:
                raise CampaignConflictError(
                    "cannot append an action while another intent is pending"
                )
            receipts = self._read_receipts(spec)
            if receipts and (
                receipts[-1].terminal_status
                in {"COMPLETED_LOCAL", "COMPLETED_CONTROL"}
                and receipts[-1].last_event_hash
                == (events[-1].event_hash if events else None)
            ):
                raise CampaignConflictError(
                    "cannot append an action after a current successful terminal"
                )
            return self._append_event_locked(
                spec,
                event_type="ACTION_INTENT",
                status="INTENT_RECORDED",
                action_id=f"action-{uuid4().hex}",
                action=action,
                request_hash=sha256_value(request),
                details={
                    "execution_semantics": spec.execution_semantics,
                    "execution_id": execution_id,
                    "attempt_index": attempt_index,
                    "predecessor_receipt_hash": predecessor_receipt_hash,
                },
            )

    def _append_result(
        self,
        spec: RealLocalCampaignSpecV65,
        intent: RealLocalCampaignEventV65,
        *,
        status: Literal["SUCCEEDED", "REPLAYED", "FAILED", "AMBIGUOUS"],
        result: object,
        details: dict[str, Any] | None = None,
    ) -> RealLocalCampaignEventV65:
        with exclusive_file_lock(self.lock_path):
            pending = self._pending_intent(self._read_events(spec))
            if pending is None or pending.event_hash != intent.event_hash:
                raise CampaignConflictError(
                    "action result no longer matches the pending intent"
                )
            return self._append_event_locked(
                spec,
                event_type="ACTION_RESULT",
                status=status,
                action_id=intent.action_id,
                action=intent.action,
                request_hash=intent.request_hash,
                intent_event_hash=intent.event_hash,
                result_hash=sha256_value(result),
                details=details or {},
            )

    @staticmethod
    def _pending_intent(
        events: list[RealLocalCampaignEventV65],
    ) -> RealLocalCampaignEventV65 | None:
        intents = {
            event.action_id: event
            for event in events
            if event.event_type == "ACTION_INTENT"
        }
        for event in events:
            if event.event_type == "ACTION_RESULT":
                intents.pop(event.action_id, None)
        return next(iter(intents.values()), None)

    def _completed_actions(
        self,
        events: list[RealLocalCampaignEventV65],
    ) -> list[ActionV65]:
        succeeded = {
            event.action
            for event in events
            if event.event_type == "ACTION_RESULT"
            and event.status in {"SUCCEEDED", "REPLAYED"}
            and event.action is not None
        }
        return [action for action in self.action_order if action in succeeded]

    def _read_receipts(
        self,
        spec: RealLocalCampaignSpecV65 | None = None,
        *,
        verify_authority: bool = True,
    ) -> list[RealLocalCampaignTerminalReceiptV65]:
        if not self.terminal_receipts_path.is_file():
            return []
        expected_spec = spec or self.load_spec()
        receipts: list[RealLocalCampaignTerminalReceiptV65] = []
        event_indices = {
            event.event_hash: event.sequence
            for event in self._read_events(expected_spec)
        }
        previous_receipt_hash: str | None = None
        previous_event_sequence = 0
        execution_ids: set[str] = set()
        for raw in self.terminal_receipts_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not raw.strip():
                raise CampaignConflictError(
                    "terminal receipt journal contains blank line"
                )
            receipt = RealLocalCampaignTerminalReceiptV65.model_validate_json(
                raw
            )
            receipt.assert_content_sealed()
            if verify_authority:
                receipt.assert_sealed(
                    authority_key=self._authority_key_required(),
                    authority_key_id=self.authority_key_id,
                )
            if (
                receipt.campaign_id != expected_spec.campaign_id
                or receipt.task_id != expected_spec.task_id
                or receipt.spec_hash != expected_spec.spec_hash
                or receipt.last_event_hash not in event_indices
            ):
                raise CampaignConflictError(
                    "terminal receipt differs from campaign evidence"
                )
            if (
                receipt.attempt_index != len(receipts) + 1
                or receipt.previous_receipt_hash != previous_receipt_hash
                or receipt.execution_id in execution_ids
            ):
                raise CampaignConflictError(
                    "terminal receipt attempt chain differs"
                )
            event_sequence = event_indices[receipt.last_event_hash]
            if event_sequence < previous_event_sequence:
                raise CampaignConflictError(
                    "terminal receipt event tips move backward"
                )
            receipts.append(receipt)
            execution_ids.add(receipt.execution_id)
            previous_receipt_hash = receipt.receipt_hash
            previous_event_sequence = event_sequence
        return receipts

    def _append_terminal(
        self,
        spec: RealLocalCampaignSpecV65,
        *,
        execution_id: str,
        started_at: datetime,
        terminal_status: TerminalStatusV65,
        reason_codes: list[str],
        snapshot: dict[str, Any] | None,
        workspace_spec_hash: str | None,
        campaign_event_chain_verified: bool,
        workspace_verified: bool,
        studio_event_chain_verified: bool,
        snapshot_fixture_only: bool | None,
        workflow_complete: bool,
        runtime_contract_hash: str | None = None,
        source_evidence_hash: str | None = None,
        studio_event_tip_hash: str | None = None,
    ) -> RealLocalCampaignTerminalReceiptV65:
        with exclusive_file_lock(self.lock_path):
            events = self._read_events(spec)
            if not events or events[-1].event_hash is None:
                raise CampaignConflictError(
                    "terminal receipt requires a verified campaign event"
                )
            if self._pending_intent(events) is not None:
                raise CampaignConflictError(
                    "terminal receipt cannot bind a pending action intent"
                )
            receipts = self._read_receipts(spec)
            control = self._is_control_execution()
            if control:
                claim_ceiling = "control_protocol_only"
            elif terminal_status == "COMPLETED_LOCAL":
                claim_ceiling = "local_workflow_evidence_only"
            else:
                claim_ceiling = "no_scientific_claim"
            receipt = RealLocalCampaignTerminalReceiptV65.seal(
                authority_key=self._authority_key_required(),
                execution_id=execution_id,
                attempt_index=len(receipts) + 1,
                previous_receipt_hash=(
                    receipts[-1].receipt_hash if receipts else None
                ),
                campaign_id=spec.campaign_id,
                task_id=spec.task_id,
                spec_hash=spec.spec_hash,
                terminal_status=terminal_status,
                reason_codes=sorted(set(reason_codes)),
                completed_actions=self._completed_actions(events),
                started_at=started_at,
                finished_at=_utc_now(),
                last_event_hash=events[-1].event_hash,
                pending_intent_hash=None,
                snapshot_hash=(
                    sha256_value(snapshot) if snapshot is not None else None
                ),
                workspace_spec_hash=workspace_spec_hash,
                runtime_contract_hash=runtime_contract_hash,
                source_evidence_hash=source_evidence_hash,
                studio_event_tip_hash=studio_event_tip_hash,
                campaign_event_chain_verified=campaign_event_chain_verified,
                workspace_verified=workspace_verified,
                studio_event_chain_verified=studio_event_chain_verified,
                snapshot_fixture_only=(
                    True if control else snapshot_fixture_only
                ),
                workflow_complete=workflow_complete,
                fixture_or_control=control,
                claim_ceiling=claim_ceiling,
                authority_key_id=self.authority_key_id,
            )
            _append_line(
                self.terminal_receipts_path,
                receipt.model_dump(mode="json"),
            )
            return receipt

    def _is_control_execution(self) -> bool:
        return self.service_factory is not None

    def _authority_key_required(self) -> bytes:
        if self.authority_key is None or len(self.authority_key) < 32:
            raise RealLocalCampaignError(
                "prepare/execute/verify requires an authority key of at "
                "least 32 bytes"
            )
        return self.authority_key

    def _assert_runtime_contract(
        self,
        spec: RealLocalCampaignSpecV65,
    ) -> CodexRuntimeContractV65:
        contract = spec.codex_runtime_contract
        if contract is None:
            raise RealLocalCampaignError(
                "real execution requires a frozen Codex runtime contract"
            )
        contract.assert_sealed()
        if contract.budgets.campaign_action_limit != len(self.action_order):
            raise RealLocalCampaignError(
                "frozen Codex runtime action budget differs from campaign "
                "action order"
            )
        _assert_local_python_runtime_identity_current(contract)
        current_manifest = _assert_runtime_code_manifest_current(contract)
        config = self.codex_config or CodexCLIConfig()
        if not config.requested_model:
            raise RealLocalCampaignError(
                "real execution requires an explicit requested model"
            )
        (
            provider,
            runtime_adapter_id,
            runtime_identity,
            outer_wsl_transport,
        ) = (
            _trusted_runtime_adapter(
                self.codex_process_runner,
                self.codex_cli_locator,
            )
        )
        runtime_identity = {
            **runtime_identity,
            "local_python_runtime": _local_python_runtime_identity(),
        }
        executable = _locate_runtime_executable(
            config=config,
            process_runner=self.codex_process_runner,
            cli_locator=self.codex_cli_locator,
            runtime_adapter_id=runtime_adapter_id,
        )
        try:
            version_result = _runtime_process_call(
                [str(executable), "--version"],
                cwd=executable.parent,
                process_runner=self.codex_process_runner,
            )
        except (
            OSError,
            subprocess.TimeoutExpired,
            ProcessOutputLimitExceeded,
        ) as exc:
            raise RealLocalCampaignError(
                f"Codex runtime verification failed: {type(exc).__name__}"
            ) from exc
        observed_version = (
            version_result.stdout.strip().removeprefix("codex-cli ").strip()
        )
        expected = {
            "provider": provider,
            "runtime_adapter_id": runtime_adapter_id,
            "requested_model": config.requested_model,
            "expected_cli_version": config.expected_cli_version,
            "observed_cli_version": observed_version,
            "executable_name": executable.name,
            "executable_sha256": _file_hash(executable),
            "runtime_identity": runtime_identity,
            "runtime_identity_hash": sha256_value(runtime_identity),
            "outer_wsl_transport": outer_wsl_transport,
            "budgets": self._runtime_budgets_from_config(config),
            "source_adapter_id": spec.world_bank_request.adapter_id,
            "code_manifest": current_manifest,
            "code_manifest_hash": sha256_value(
                [item.model_dump(mode="json") for item in current_manifest]
            ),
        }
        current = self._seal_runtime_contract(**expected)
        if version_result.returncode != 0 or current != contract:
            raise RealLocalCampaignError(
                "current Codex runtime differs from the frozen contract"
            )
        return contract

    def _make_service(self) -> Any:
        if self._service_instance is not None:
            return self._service_instance
        authority_key = self._authority_key_required()
        kwargs = {
            "task_root": self.workspace_root,
            "authority_key": authority_key,
            "authority_key_id": self.authority_key_id,
            "codex_config": self.codex_config,
            "codex_process_runner": self.codex_process_runner,
            "codex_cli_locator": self.codex_cli_locator,
        }
        if self.service_factory is None:
            _trusted_runtime_adapter(
                self.codex_process_runner,
                self.codex_cli_locator,
            )
            service = StudioTaskService(**kwargs)
            if (
                service.role_transport_factory is not None
                or service.world_bank_fetcher is not None
            ):
                raise RealLocalCampaignError(
                    "real mode rejects injected role or World Bank transports"
                )
        else:
            service = self.service_factory(**kwargs)
        self._service_instance = service
        return service

    def _workspace_path(self, spec: RealLocalCampaignSpecV65) -> Path:
        return self.workspace_root / spec.task_id

    def _workspace_state(
        self,
        spec: RealLocalCampaignSpecV65,
    ) -> Literal["ABSENT", "COMPLETE", "PARTIAL"]:
        root = self._workspace_path(spec)
        if not root.exists() and not root.is_symlink():
            return "ABSENT"
        if (
            root.is_dir()
            and (root / ".fma" / "workspace_spec.json").is_file()
        ):
            return "COMPLETE"
        return "PARTIAL"

    def _workspace_exists(self, spec: RealLocalCampaignSpecV65) -> bool:
        return self._workspace_state(spec) == "COMPLETE"

    def _expected_source_contract(
        self,
        spec: RealLocalCampaignSpecV65,
    ) -> WorldBankSourceContractV62:
        request = spec.world_bank_request
        return WorldBankSourceContractV62.seal(
            contract_id=request.contract_id,
            country_code=request.country_code,
            indicator_id=request.indicator_id,
            start_year=request.start_year,
            end_year=request.end_year,
            minimum_observations=request.minimum_observations,
            state_unit=request.state_unit,
            attribution=request.attribution,
            fixture_only=False,
        )

    def _expected_measurement_schema(
        self,
        spec: RealLocalCampaignSpecV65,
        contract: WorldBankSourceContractV62,
    ) -> MeasurementSchemaV62:
        request = spec.world_bank_request
        return MeasurementSchemaV62.seal(
            measurement_id=request.contract_id,
            source_contract_hash=contract.contract_hash,
            indicator_id=request.indicator_id,
            semantic_name=request.semantic_name,
            operational_definition=request.operational_definition,
            observation_time_basis=request.observation_time_basis,
            aggregation_level=request.aggregation_level,
            time_unit=contract.time_unit,
            state_unit=request.state_unit,
        )

    def _validate_source_workspace(
        self,
        spec: RealLocalCampaignSpecV65,
        snapshot: dict[str, Any],
        *,
        require_s2: bool = False,
    ) -> tuple[Literal["ABSENT", "COMPLETE", "PARTIAL"], str | None]:
        root = self._workspace_path(spec)
        data_received = bool(snapshot.get("backhalf", {}).get("data_received"))
        present = {
            relative: (root / relative).is_file()
            for relative in _SOURCE_PATHS
        }
        if not any(present.values()) and not data_received:
            return "ABSENT", None
        required = set(_SOURCE_INTAKE_PATHS)
        if require_s2:
            required.update(_SOURCE_S2_PATHS)
        if self._is_control_execution():
            if not data_received:
                return "PARTIAL", None
            if any(present.values()) and not all(
                present[item] for item in required
            ):
                return "PARTIAL", None
            return (
                "COMPLETE",
                sha256_value(
                    {
                        "scope": "control_snapshot_only",
                        "present": present,
                        "data_received": data_received,
                    }
                ),
            )
        if not data_received or not all(present[item] for item in required):
            return "PARTIAL", None
        try:
            workspace = StageWorkspaceV50.open_existing(
                root,
                authority_key=self.authority_key,
                authority_key_id=self.authority_key_id,
            )
            if (
                workspace.spec.workspace_id != spec.task_id
                or workspace.spec.objective != spec.objective
                or workspace.spec.evidence_scope != "public_data"
                or not workspace.verify()
            ):
                return "PARTIAL", None
            expected_contract = self._expected_source_contract(spec)
            contract = WorldBankSourceContractV62.model_validate(
                _read_json_object(root / SOURCE_CONTRACT_PATH)
            )
            contract.assert_sealed()
            if contract != expected_contract:
                return "PARTIAL", None
            source_receipt = WorldBankSourceReceiptV62.model_validate(
                _read_json_object(root / SOURCE_RECEIPT_PATH)
            )
            source_receipt.assert_sealed()
            raw_request = StudioODEDataRequestV59.model_validate(
                _read_json_object(root / RAW_RELATIVE_PATH)
            )
            source_snapshot = ODETimeSeriesSnapshotV52.seal(
                task_id=spec.task_id,
                time_unit=raw_request.time_unit,
                state_unit=raw_request.state_unit,
                times=raw_request.times,
                observations=raw_request.observations,
                source_id=raw_request.source_id,
                fixture_only=raw_request.fixture_only,
            )
            acquisition = SourceAcquisitionReceiptV62.model_validate(
                _read_json_object(root / SOURCE_ACQUISITION_AUTH_PATH)
            )
            measurement = MeasurementSchemaV62.model_validate(
                _read_json_object(root / MEASUREMENT_SCHEMA_PATH)
            )
            measurement.assert_sealed()
            if measurement != self._expected_measurement_schema(
                spec, contract
            ):
                return "PARTIAL", None
            raw_body = (root / SOURCE_RAW_PATH).read_bytes()
            source_authority = (
                SourceTransportAuthorityV62.from_stage_workspace(workspace)
            )
            if not source_authority.verify_acquisition(
                workspace_spec=workspace.spec,
                contract=contract,
                source_receipt=source_receipt,
                snapshot=source_snapshot,
                raw_body=raw_body,
                receipt=acquisition,
            ):
                return "PARTIAL", None
            if (
                raw_request.adapter_id
                != spec.world_bank_request.adapter_id
                or raw_request.time_unit != contract.time_unit
                or raw_request.state_unit
                != spec.world_bank_request.state_unit
                or raw_request.source_id != source_receipt.source_id
                or raw_request.fixture_only
                or raw_request.times
                != [
                    float(year)
                    for year in range(
                        spec.world_bank_request.start_year,
                        spec.world_bank_request.end_year + 1,
                    )
                ]
                or source_snapshot.task_id != spec.task_id
                or source_snapshot.snapshot_hash
                != source_receipt.snapshot_hash
                or source_receipt.observation_count
                != len(source_snapshot.observations)
                or source_receipt.observation_count
                < spec.world_bank_request.minimum_observations
                or source_receipt.first_year
                != spec.world_bank_request.start_year
                or source_receipt.last_year
                != spec.world_bank_request.end_year
                or source_receipt.fixture_only
                or source_snapshot.fixture_only
            ):
                return "PARTIAL", None
            evidence: dict[str, Any] = {
                "campaign_spec_hash": spec.spec_hash,
                "request_hash": sha256_value(
                    spec.world_bank_request.model_dump(mode="json")
                ),
                "workspace_spec_hash": workspace.spec.spec_hash,
                "contract_hash": contract.contract_hash,
                "source_receipt_hash": source_receipt.receipt_hash,
                "snapshot_hash": source_snapshot.snapshot_hash,
                "raw_request_hash": _file_hash(root / RAW_RELATIVE_PATH),
                "raw_response_hash": _file_hash(root / SOURCE_RAW_PATH),
                "acquisition_receipt_hash": acquisition.receipt_hash,
                "measurement_schema_hash": measurement.schema_hash,
            }
            if require_s2:
                processed_path = root / PROCESSED_RELATIVE_PATH
                processed_snapshot = ODETimeSeriesSnapshotV52.model_validate(
                    _read_json_object(processed_path)
                )
                processed_snapshot.assert_sealed()
                processed_hash = _file_hash(processed_path)
                processed_manifest = ProcessedManifestV50.model_validate(
                    _read_json_object(
                        root / "data/processed/manifest.json"
                    )
                )
                transform_receipt = S2TransformReceiptV62.model_validate(
                    _read_json_object(root / S2_TRANSFORM_RECEIPT_PATH)
                )
                transform_receipt.assert_sealed()
                verification = SourceVerificationV62.model_validate(
                    _read_json_object(root / SOURCE_VERIFICATION_PATH)
                )
                verification.assert_sealed()
                replay = verify_world_bank_source_v62(
                    workspace_root=root,
                    contract=contract,
                    receipt=source_receipt,
                    snapshot=source_snapshot,
                    verified_at=verification.verified_at,
                )
                s2_receipt = (
                    S2SourceReverificationReceiptV62.model_validate(
                        _read_json_object(root / S2_SOURCE_REVERIFICATION_PATH)
                    )
                )
                raw_baseline = workspace._raw_baseline_for_current_s2()
                if (
                    raw_baseline is None
                    or processed_snapshot != source_snapshot
                    or not processed_manifest.artifacts
                    or any(
                        artifact.relative_path != PROCESSED_RELATIVE_PATH
                        or artifact.artifact_hash != processed_hash
                        for artifact in processed_manifest.artifacts
                    )
                    or processed_manifest.raw_baseline_tree_hash
                    != raw_baseline.raw_tree_hash
                    or transform_receipt.workspace_spec_hash
                    != workspace.spec.spec_hash
                    or transform_receipt.raw_baseline_hash
                    != raw_baseline.baseline_hash
                    or transform_receipt.s2_attempt
                    != raw_baseline.s2_attempt
                    or transform_receipt.task_id != spec.task_id
                    or transform_receipt.input_relative_path
                    != RAW_RELATIVE_PATH
                    or transform_receipt.input_hash
                    != _file_hash(root / RAW_RELATIVE_PATH)
                    or transform_receipt.output_relative_path
                    != PROCESSED_RELATIVE_PATH
                    or transform_receipt.output_hash != processed_hash
                    or verification != replay
                    or verification.status != "PASS"
                    or not source_authority.is_s2_reverification_admissible(
                        workspace=workspace,
                        raw_baseline=raw_baseline,
                        contract=contract,
                        source_receipt=source_receipt,
                        snapshot=source_snapshot,
                        acquisition_receipt=acquisition,
                        verification=verification,
                        receipt=s2_receipt,
                    )
                ):
                    return "PARTIAL", None
                evidence.update(
                    {
                        "source_verification_hash": (
                            verification.verification_hash
                        ),
                        "s2_reverification_hash": s2_receipt.receipt_hash,
                        "raw_baseline_hash": raw_baseline.baseline_hash,
                        "processed_snapshot_hash": (
                            processed_snapshot.snapshot_hash
                        ),
                        "processed_file_hash": processed_hash,
                        "processed_manifest_hash": sha256_value(
                            processed_manifest.model_dump(mode="json")
                        ),
                        "transform_receipt_hash": (
                            transform_receipt.receipt_hash
                        ),
                    }
                )
            return "COMPLETE", sha256_value(evidence)
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            ValidationError,
        ):
            return "PARTIAL", None

    @staticmethod
    def _stage_open(snapshot: dict[str, Any], stage: str) -> bool:
        return (
            snapshot.get("workflow", {})
            .get("stage_statuses", {})
            .get(stage)
            == "gate_open"
        )

    @staticmethod
    def _stage_blocked_error(
        action: ActionV65,
        snapshot: dict[str, Any],
    ) -> CampaignStageBlocked | None:
        """Recover typed gate semantics instead of reporting a false conflict."""

        preferred_stage = {
            "run_s0": "S0",
            "run_s1": "S1",
        }.get(action)
        statuses = snapshot.get("workflow", {}).get("stage_statuses", {})
        if not isinstance(statuses, dict):
            return None
        candidate_stages = (
            [preferred_stage]
            if preferred_stage is not None
            else [
                stage
                for stage in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
                if statuses.get(stage) in {"blocked", "failed"}
            ]
        )
        recovery = snapshot.get("recovery", {})
        events = snapshot.get("events", [])
        for stage in candidate_stages:
            if stage is None:
                continue
            stage_status = statuses.get(stage)
            latest_gate: dict[str, Any] | None = None
            if isinstance(events, list):
                expected_event = f"{stage.lower()}_gate_evaluated"
                for event in reversed(events):
                    if (
                        isinstance(event, dict)
                        and event.get("event_type") == expected_event
                        and isinstance(event.get("details"), dict)
                    ):
                        latest_gate = event["details"]
                        break
            decision = (
                str(latest_gate.get("decision"))
                if latest_gate is not None
                and latest_gate.get("decision") is not None
                else None
            )
            human_required = bool(
                isinstance(recovery, dict)
                and (
                    recovery.get("human_required")
                    or recovery.get("stopped")
                )
            )
            recognized = stage_status in {"blocked", "failed"} or (
                stage_status == "awaiting_gate_evidence"
                and (
                    decision in {"BLOCKED", "NEEDS_EVIDENCE"}
                    or human_required
                )
            )
            if not recognized:
                continue
            return CampaignStageBlocked(
                action=action,
                stage=stage,
                stage_status=str(stage_status),
                decision=decision,
                review_verdict=(
                    str(latest_gate.get("review_verdict"))
                    if latest_gate is not None
                    and latest_gate.get("review_verdict") is not None
                    else None
                ),
                finding_signature=(
                    str(latest_gate.get("finding_signature"))
                    if latest_gate is not None
                    and latest_gate.get("finding_signature") is not None
                    else None
                ),
                gate_outcome_hash=(
                    str(latest_gate.get("gate_outcome_hash"))
                    if latest_gate is not None
                    and latest_gate.get("gate_outcome_hash") is not None
                    else None
                ),
                human_required=human_required,
                snapshot=snapshot,
            )
        return None

    def _action_postcondition(
        self,
        spec: RealLocalCampaignSpecV65,
        action: ActionV65,
        snapshot: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if (
            snapshot.get("task_id") != spec.task_id
            or snapshot.get("objective") != spec.objective
        ):
            return False, None
        if action == "create_task":
            proven = self._workspace_exists(spec)
        elif action == "run_s0":
            proven = self._stage_open(snapshot, "S0")
        elif action == "run_s1":
            proven = self._stage_open(snapshot, "S1")
        elif action == "ingest_world_bank_data":
            source_state, source_hash = self._validate_source_workspace(
                spec, snapshot, require_s2=False
            )
            return source_state == "COMPLETE", source_hash
        else:
            source_state, source_hash = self._validate_source_workspace(
                spec, snapshot, require_s2=True
            )
            proven = bool(
                source_state == "COMPLETE"
                and snapshot.get("backhalf", {}).get("workflow_complete")
                and all(
                    self._stage_open(snapshot, stage)
                    for stage in ("S2", "S3", "S4", "S5", "S6")
                )
            )
            return proven, source_hash if proven else None
        if not proven or self._is_control_execution():
            return proven, None
        try:
            workspace = StageWorkspaceV50.open_existing(
                self._workspace_path(spec),
                authority_key=self.authority_key,
                authority_key_id=self.authority_key_id,
            )
            return (
                bool(
                    workspace.spec.workspace_id == spec.task_id
                    and workspace.spec.objective == spec.objective
                    and workspace.spec.evidence_scope == "public_data"
                    and workspace.verify()
                ),
                None,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            return False, None

    def _action_request(
        self,
        spec: RealLocalCampaignSpecV65,
        action: ActionV65,
    ) -> object:
        if action == "create_task":
            return CreateTaskRequest(
                objective=spec.objective,
                workspace_id=spec.task_id,
                evidence_scope="public_data",
            ).model_dump(mode="json")
        if action == "ingest_world_bank_data":
            return spec.world_bank_request.model_dump(mode="json")
        return {"task_id": spec.task_id, "action": action}

    def _call_action(
        self,
        service: Any,
        spec: RealLocalCampaignSpecV65,
        action: ActionV65,
    ) -> dict[str, Any]:
        if action == "create_task":
            return service.create_task(self._action_request(spec, action))
        if action == "run_s0":
            return service.run_s0(spec.task_id)
        if action == "run_s1":
            return service.run_s1(spec.task_id)
        if action == "ingest_world_bank_data":
            return service.ingest_world_bank_data(
                spec.task_id,
                spec.world_bank_request.model_dump(mode="json"),
            )
        return service.run_backhalf(spec.task_id)

    def _record_or_replay_action(
        self,
        service: Any,
        spec: RealLocalCampaignSpecV65,
        action: ActionV65,
        *,
        execution_id: str,
        attempt_index: int,
        predecessor_receipt_hash: str | None,
    ) -> dict[str, Any]:
        request = self._action_request(spec, action)
        intent = self._append_intent(
            spec,
            action=action,
            request=request,
            execution_id=execution_id,
            attempt_index=attempt_index,
            predecessor_receipt_hash=predecessor_receipt_hash,
        )
        try:
            snapshot: dict[str, Any] | None = None
            replay_reason: str | None = None
            if (
                action == "create_task"
                and self._workspace_state(spec) == "PARTIAL"
            ):
                raise CampaignReconciliationRequired(
                    "task workspace exists without a complete workspace spec"
                )
            if self._workspace_exists(spec):
                snapshot = service.snapshot(spec.task_id)
                proven, _ = self._action_postcondition(
                    spec, action, snapshot
                )
                if proven:
                    replay_reason = f"{action}_postcondition_already_proven"
                elif action == "ingest_world_bank_data":
                    source_state, _ = self._validate_source_workspace(
                        spec, snapshot, require_s2=False
                    )
                    if source_state == "PARTIAL":
                        raise CampaignReconciliationRequired(
                            "official-source intake is partially or "
                            "inconsistently materialized"
                        )
            if replay_reason is None:
                snapshot = self._call_action(service, spec, action)
            assert snapshot is not None
            proven, postcondition_hash = self._action_postcondition(
                spec, action, snapshot
            )
            if not proven:
                stage_blocked = self._stage_blocked_error(action, snapshot)
                if stage_blocked is not None:
                    raise stage_blocked
                raise CampaignConflictError(
                    f"{action} returned without its concrete postcondition"
                )
        except Exception as exc:
            partial_create = bool(
                action == "create_task"
                and self._workspace_state(spec) == "PARTIAL"
            )
            error_details: dict[str, object] = {
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            if isinstance(exc, CampaignStageBlocked):
                error_details.update(exc.evidence_details())
            self._append_result(
                spec,
                intent,
                status=(
                    "AMBIGUOUS"
                    if (
                        isinstance(exc, CampaignReconciliationRequired)
                        or partial_create
                    )
                    else "FAILED"
                ),
                result=error_details,
                details=error_details,
            )
            if (
                partial_create
                and not isinstance(exc, CampaignReconciliationRequired)
            ):
                raise CampaignReconciliationRequired(
                    "create_task failed after partially materializing its "
                    "workspace"
                ) from exc
            raise
        self._append_result(
            spec,
            intent,
            status="REPLAYED" if replay_reason else "SUCCEEDED",
            result=snapshot,
            details=(
                {
                    "replay_reason": replay_reason,
                    "postcondition_verified": True,
                    "postcondition_hash": postcondition_hash,
                }
                if replay_reason
                else {
                    "call_completed": True,
                    "postcondition_verified": True,
                    "postcondition_hash": postcondition_hash,
                }
            ),
        )
        return snapshot

    def _reconcile_pending(
        self,
        service: Any,
        spec: RealLocalCampaignSpecV65,
        pending: RealLocalCampaignEventV65,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Resolve only outcomes proven by current workspace authority."""

        assert pending.action is not None
        action = pending.action
        if action == "create_task":
            workspace_state = self._workspace_state(spec)
            if workspace_state == "PARTIAL":
                self._append_result(
                    spec,
                    pending,
                    status="AMBIGUOUS",
                    result={"workspace_state": workspace_state},
                    details={
                        "human_reconciliation_required": True,
                        "postcondition_verified": False,
                    },
                )
                return None, False
            if workspace_state == "ABSENT":
                try:
                    snapshot = self._call_action(service, spec, action)
                except Exception as exc:
                    workspace_state = self._workspace_state(spec)
                    if workspace_state == "COMPLETE":
                        try:
                            snapshot = service.snapshot(spec.task_id)
                            proven, postcondition_hash = (
                                self._action_postcondition(
                                    spec, action, snapshot
                                )
                            )
                        except Exception:
                            proven = False
                            postcondition_hash = None
                        if proven:
                            self._append_result(
                                spec,
                                pending,
                                status="REPLAYED",
                                result=snapshot,
                                details={
                                    "replay_reason": (
                                        "workspace_proves_interrupted_"
                                        "create_task"
                                    ),
                                    "postcondition_verified": True,
                                    "postcondition_hash": (
                                        postcondition_hash
                                    ),
                                },
                            )
                            return snapshot, True
                    if workspace_state != "ABSENT":
                        self._append_result(
                            spec,
                            pending,
                            status="AMBIGUOUS",
                            result={"workspace_state": workspace_state},
                            details={
                                "human_reconciliation_required": True,
                                "postcondition_verified": False,
                                "error_type": type(exc).__name__,
                            },
                        )
                        return None, False
                    self._append_result(
                        spec,
                        pending,
                        status="FAILED",
                        result={
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        },
                        details={
                            "safe_local_resume": True,
                            "postcondition_verified": False,
                        },
                    )
                    raise
                proven, postcondition_hash = self._action_postcondition(
                    spec, action, snapshot
                )
                if not proven:
                    workspace_state = self._workspace_state(spec)
                    status: Literal["FAILED", "AMBIGUOUS"] = (
                        "FAILED"
                        if workspace_state == "ABSENT"
                        else "AMBIGUOUS"
                    )
                    self._append_result(
                        spec,
                        pending,
                        status=status,
                        result=snapshot,
                        details={
                            "safe_local_resume": True,
                            "human_reconciliation_required": (
                                status == "AMBIGUOUS"
                            ),
                            "postcondition_verified": False,
                        },
                    )
                    if status == "AMBIGUOUS":
                        return snapshot, False
                    raise CampaignConflictError(
                        "resumed create_task lacks its postcondition"
                    )
                self._append_result(
                    spec,
                    pending,
                    status="SUCCEEDED",
                    result=snapshot,
                    details={
                        "safe_local_resume": True,
                        "postcondition_verified": True,
                        "postcondition_hash": postcondition_hash,
                    },
                )
                return snapshot, True
            snapshot = service.snapshot(spec.task_id)
            proven, postcondition_hash = self._action_postcondition(
                spec, action, snapshot
            )
            if not proven:
                self._append_result(
                    spec,
                    pending,
                    status="AMBIGUOUS",
                    result=snapshot,
                    details={
                        "human_reconciliation_required": True,
                        "postcondition_verified": False,
                    },
                )
                return snapshot, False
            self._append_result(
                spec,
                pending,
                status="REPLAYED",
                result=snapshot,
                details={
                    "replay_reason": "workspace_proves_create_task",
                    "postcondition_verified": True,
                    "postcondition_hash": postcondition_hash,
                },
            )
            return snapshot, True

        if not self._workspace_exists(spec):
            self._append_result(
                spec,
                pending,
                status="AMBIGUOUS",
                result={"workspace_exists": False},
                details={
                    "human_reconciliation_required": True,
                    "postcondition_verified": False,
                },
            )
            return None, False
        snapshot = service.snapshot(spec.task_id)
        proven, postcondition_hash = self._action_postcondition(
            spec, action, snapshot
        )
        if not proven and action == "run_s0":
            # Studio V6.6 can replay only an authenticated interrupted S0
            # reject/recovery transition.  Let that inner authority reconcile
            # before declaring the outer action ambiguous.
            try:
                snapshot = self._call_action(service, spec, action)
            except Exception as exc:
                self._append_result(
                    spec,
                    pending,
                    status="AMBIGUOUS",
                    result={
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    },
                    details={
                        "human_reconciliation_required": True,
                        "postcondition_verified": False,
                        "nested_s0_recovery_attempted": True,
                    },
                )
                return None, False
            proven, postcondition_hash = self._action_postcondition(
                spec, action, snapshot
            )
            if not proven:
                stage_blocked = self._stage_blocked_error(action, snapshot)
                if stage_blocked is not None:
                    details = stage_blocked.evidence_details()
                    details["nested_s0_recovery_attempted"] = True
                    self._append_result(
                        spec,
                        pending,
                        status="FAILED",
                        result=details,
                        details=details,
                    )
                    raise stage_blocked
        if not proven:
            self._append_result(
                spec,
                pending,
                status="AMBIGUOUS",
                result=snapshot,
                details={
                    "human_reconciliation_required": True,
                    "postcondition_verified": False,
                },
            )
            return snapshot, False
        self._append_result(
            spec,
            pending,
            status="REPLAYED",
            result=snapshot,
            details={
                "replay_reason": (
                    "authoritative_workspace_proves_pending_action"
                ),
                "postcondition_verified": True,
                "postcondition_hash": postcondition_hash,
            },
        )
        return snapshot, True

    @staticmethod
    def _verify_codex_process_receipts(
        workspace: StageWorkspaceV50,
        runtime_contract: CodexRuntimeContractV65,
    ) -> bool:
        receipts: list[RoleProcessReceiptV51] = []
        try:
            for kind in (
                "codex_role_transport_trace_v51",
                "codex_role_transport_trace_v58",
            ):
                for _, payload in workspace._artifacts_of_kind(kind):
                    if not isinstance(payload, dict):
                        return False
                    receipt = RoleProcessReceiptV51.model_validate(
                        payload.get("process_receipt")
                    )
                    if (
                        not receipt.receipt_hash
                        or receipt.receipt_hash != receipt.content_hash()
                    ):
                        return False
                    receipts.append(receipt)
        except (OSError, RuntimeError, TypeError, ValueError, ValidationError):
            return False
        return bool(
            receipts
            and all(
                receipt.transport == "codex_cli"
                and receipt.provider == runtime_contract.provider
                and receipt.requested_model
                == runtime_contract.requested_model
                and receipt.cli_version
                == runtime_contract.observed_cli_version
                and receipt.executable_sha256
                == runtime_contract.executable_sha256
                and receipt.tool_event_count == 0
                and receipt.scratch_unchanged
                for receipt in receipts
            )
        )

    def _real_verification(
        self,
        service: Any,
        spec: RealLocalCampaignSpecV65,
        snapshot: dict[str, Any],
        runtime_contract: CodexRuntimeContractV65 | None,
    ) -> tuple[
        bool,
        bool,
        bool | None,
        str | None,
        str | None,
        str | None,
        bool,
    ]:
        if self._is_control_execution():
            return False, False, True, None, None, None, False
        if (
            getattr(service, "role_transport_factory", None) is not None
            or getattr(service, "world_bank_fetcher", None) is not None
        ):
            raise RealLocalCampaignError(
                "real verifier rejects injected role or source transports"
            )
        workspace = StageWorkspaceV50.open_existing(
            self._workspace_path(spec),
            authority_key=self.authority_key,
            authority_key_id=self.authority_key_id,
        )
        workspace_verified = workspace.verify()
        studio_events = service._events(spec.task_id)
        studio_event_chain_verified = bool(studio_events)
        studio_event_tip_hash = (
            studio_events[-1].event_hash if studio_events else None
        )
        fixture_value = snapshot.get("backhalf", {}).get("fixture_only")
        snapshot_fixture_only = (
            fixture_value if isinstance(fixture_value, bool) else None
        )
        source_state, source_evidence_hash = self._validate_source_workspace(
            spec, snapshot, require_s2=True
        )
        runtime_receipts_verified = bool(
            runtime_contract is not None
            and self._verify_codex_process_receipts(
                workspace, runtime_contract
            )
        )
        return (
            workspace_verified,
            studio_event_chain_verified,
            snapshot_fixture_only,
            workspace.spec.spec_hash,
            studio_event_tip_hash,
            source_evidence_hash if source_state == "COMPLETE" else None,
            runtime_receipts_verified,
        )

    def execute(
        self,
        *,
        execute_live: bool = False,
        retry_failed: bool = False,
        reconcile_human: bool = False,
    ) -> RealLocalCampaignTerminalReceiptV65:
        """Run or replay S0--S6 after all three live permissions are present."""

        with exclusive_file_lock(self.lock_path):
            return self._execute_locked(
                execute_live=execute_live,
                retry_failed=retry_failed,
                reconcile_human=reconcile_human,
            )

    def _execute_locked(
        self,
        *,
        execute_live: bool,
        retry_failed: bool,
        reconcile_human: bool,
    ) -> RealLocalCampaignTerminalReceiptV65:
        try:
            spec = self.load_spec()
            events = self._read_events(spec)
            receipts = self._read_receipts(spec)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CampaignReconciliationRequired(
                "campaign JSONL or frozen state is not safely replayable"
            ) from exc
        pending = self._pending_intent(events)
        current_tip = events[-1].event_hash if events else None
        if (spec.execution_mode == "control") != self._is_control_execution():
            raise RealLocalCampaignError(
                "campaign execution_mode differs from the configured "
                "service boundary"
            )
        if receipts:
            latest = receipts[-1]
            latest_is_current = bool(
                current_tip is not None
                and latest.last_event_hash == current_tip
                and pending is None
            )
            if latest.terminal_status in {
                "COMPLETED_LOCAL",
                "COMPLETED_CONTROL",
            }:
                if latest_is_current:
                    require_real = (
                        latest.terminal_status == "COMPLETED_LOCAL"
                    )
                    if not self.verify(require_real=require_real):
                        raise CampaignReconciliationRequired(
                            "current successful terminal failed full-chain "
                            "replay"
                        )
                    return latest
                raise CampaignReconciliationRequired(
                    "events exist after the successful terminal receipt"
                )
            if latest.terminal_status == "FAILED" and not retry_failed:
                raise CampaignRetryRequired(
                    "latest attempt failed; pass retry_failed=true explicitly"
                )
            if (
                latest.terminal_status
                == "HUMAN_RECONCILIATION_REQUIRED"
                and not reconcile_human
            ):
                raise CampaignRetryRequired(
                    "latest attempt requires human reconciliation; "
                    "pass reconcile_human=true explicitly"
                )
        if len(receipts) >= spec.max_execution_attempts:
            raise CampaignRetryRequired(
                "execution attempt budget exhausted"
            )
        if not (
            execute_live and spec.live_codex and spec.live_world_bank
        ):
            raise LiveExecutionNotAuthorized(
                "execution requires --execute-live plus frozen "
                "live_codex=true and live_world_bank=true"
            )

        execution_id = f"exec-{uuid4().hex}"
        started_at = _utc_now()
        attempt_index = len(receipts) + 1
        predecessor_receipt_hash = (
            receipts[-1].receipt_hash if receipts else None
        )
        runtime_contract: CodexRuntimeContractV65 | None = None
        service: Any | None = None
        snapshot: dict[str, Any] | None = None
        source_evidence_hash: str | None = None
        studio_event_tip_hash: str | None = None
        try:
            if not self._is_control_execution():
                runtime_contract = self._assert_runtime_contract(spec)
            service = self._make_service()
            if pending is not None:
                snapshot, reconciled = self._reconcile_pending(
                    service, spec, pending
                )
                if not reconciled:
                    return self._append_terminal(
                        spec,
                        execution_id=execution_id,
                        started_at=started_at,
                        terminal_status="HUMAN_RECONCILIATION_REQUIRED",
                        reason_codes=[
                            _reason_code(
                                "pending_action_ambiguous",
                                pending.action or "unknown",
                            )
                        ],
                        snapshot=snapshot,
                        workspace_spec_hash=None,
                        campaign_event_chain_verified=True,
                        workspace_verified=False,
                        studio_event_chain_verified=False,
                        snapshot_fixture_only=None,
                        workflow_complete=False,
                        runtime_contract_hash=(
                            runtime_contract.contract_hash
                            if runtime_contract is not None
                            else None
                        ),
                    )

            events = self._read_events(spec)
            completed = set(self._completed_actions(events))
            if completed:
                if self._workspace_state(spec) != "COMPLETE":
                    raise CampaignReconciliationRequired(
                        "historical actions lack a complete task workspace"
                    )
                try:
                    snapshot = service.snapshot(spec.task_id)
                except Exception as exc:
                    raise CampaignReconciliationRequired(
                        "historical actions cannot be replay-verified"
                    ) from exc
                for action in self.action_order:
                    if action not in completed:
                        continue
                    proven, _ = self._action_postcondition(
                        spec, action, snapshot
                    )
                    if not proven:
                        raise CampaignReconciliationRequired(
                            f"historical {action} postcondition is no longer "
                            "proven"
                        )
            for action in self.action_order:
                if action in completed:
                    continue
                snapshot = self._record_or_replay_action(
                    service,
                    spec,
                    action,
                    execution_id=execution_id,
                    attempt_index=attempt_index,
                    predecessor_receipt_hash=predecessor_receipt_hash,
                )
                completed.add(action)
            if snapshot is None:
                snapshot = service.snapshot(spec.task_id)
            workflow_complete = bool(
                snapshot.get("backhalf", {}).get("workflow_complete")
            )
            if not self._is_control_execution():
                runtime_contract = self._assert_runtime_contract(spec)
            (
                workspace_verified,
                studio_chain_verified,
                snapshot_fixture_only,
                workspace_spec_hash,
                studio_event_tip_hash,
                source_evidence_hash,
                runtime_receipts_verified,
            ) = self._real_verification(
                service, spec, snapshot, runtime_contract
            )
            if not self._is_control_execution():
                completed_ok = (
                    workflow_complete
                    and workspace_verified
                    and studio_chain_verified
                    and snapshot_fixture_only is False
                    and source_evidence_hash is not None
                    and runtime_receipts_verified
                    and snapshot.get("scientific_qualification_granted")
                    is False
                    and snapshot.get("real_world_action_authorized") is False
                )
                terminal_status: TerminalStatusV65 = (
                    "COMPLETED_LOCAL" if completed_ok else "FAILED"
                )
                reasons = (
                    []
                    if completed_ok
                    else ["real_local_verification_incomplete"]
                )
            else:
                terminal_status = (
                    "COMPLETED_CONTROL" if workflow_complete else "FAILED"
                )
                reasons = ["injected_service_control_run"]
                if not workflow_complete:
                    reasons.append("workflow_incomplete")
            return self._append_terminal(
                spec,
                execution_id=execution_id,
                started_at=started_at,
                terminal_status=terminal_status,
                reason_codes=reasons,
                snapshot=snapshot,
                workspace_spec_hash=workspace_spec_hash,
                campaign_event_chain_verified=True,
                workspace_verified=workspace_verified,
                studio_event_chain_verified=studio_chain_verified,
                snapshot_fixture_only=snapshot_fixture_only,
                workflow_complete=workflow_complete,
                runtime_contract_hash=(
                    runtime_contract.contract_hash
                    if runtime_contract is not None
                    else None
                ),
                source_evidence_hash=source_evidence_hash,
                studio_event_tip_hash=studio_event_tip_hash,
            )
        except CampaignStageBlocked as exc:
            if self._pending_intent(self._read_events(spec)) is not None:
                raise CampaignReconciliationRequired(
                    "stage blocked while an action intent remains pending"
                ) from exc
            snapshot = exc.snapshot
            workspace_verified = False
            workspace_spec_hash = None
            studio_chain_verified = False
            studio_event_tip_hash = None
            try:
                workspace = StageWorkspaceV50.open_existing(
                    self._workspace_path(spec),
                    authority_key=self.authority_key,
                    authority_key_id=self.authority_key_id,
                )
                workspace_verified = workspace.verify()
                workspace_spec_hash = workspace.spec.spec_hash
                if service is not None and hasattr(service, "_events"):
                    studio_events = service._events(spec.task_id)
                    studio_chain_verified = bool(studio_events)
                    if studio_events:
                        studio_event_tip_hash = studio_events[-1].event_hash
            except (OSError, RuntimeError, TypeError, ValueError):
                workspace_verified = False
                workspace_spec_hash = None
                studio_chain_verified = False
                studio_event_tip_hash = None
            reasons = [
                _reason_code("stage_blocked", exc.stage),
                _reason_code(
                    "gate_decision", exc.decision or exc.stage_status
                ),
            ]
            if exc.review_verdict is not None:
                reasons.append(
                    _reason_code("review_verdict", exc.review_verdict)
                )
            if exc.finding_signature is not None:
                reasons.append(
                    _reason_code(
                        "finding_signature",
                        exc.finding_signature[:24],
                    )
                )
            if exc.human_required:
                reasons.append("human_review_required")
            return self._append_terminal(
                spec,
                execution_id=execution_id,
                started_at=started_at,
                terminal_status="FAILED",
                reason_codes=reasons,
                snapshot=snapshot,
                workspace_spec_hash=workspace_spec_hash,
                campaign_event_chain_verified=True,
                workspace_verified=workspace_verified,
                studio_event_chain_verified=studio_chain_verified,
                snapshot_fixture_only=None,
                workflow_complete=False,
                runtime_contract_hash=(
                    runtime_contract.contract_hash
                    if runtime_contract is not None
                    else None
                ),
                studio_event_tip_hash=studio_event_tip_hash,
            )
        except CampaignReconciliationRequired as exc:
            if self._pending_intent(self._read_events(spec)) is not None:
                raise CampaignReconciliationRequired(
                    "execution preflight failed while an action intent "
                    "remains pending"
                ) from exc
            return self._append_terminal(
                spec,
                execution_id=execution_id,
                started_at=started_at,
                terminal_status="HUMAN_RECONCILIATION_REQUIRED",
                reason_codes=[
                    _reason_code("reconciliation", type(exc).__name__)
                ],
                snapshot=snapshot,
                workspace_spec_hash=None,
                campaign_event_chain_verified=True,
                workspace_verified=False,
                studio_event_chain_verified=False,
                snapshot_fixture_only=None,
                workflow_complete=False,
                runtime_contract_hash=(
                    runtime_contract.contract_hash
                    if runtime_contract is not None
                    else None
                ),
            )
        except Exception as exc:
            if self._pending_intent(self._read_events(spec)) is not None:
                raise CampaignReconciliationRequired(
                    "execution failed while an action intent remains pending"
                ) from exc
            return self._append_terminal(
                spec,
                execution_id=execution_id,
                started_at=started_at,
                terminal_status="FAILED",
                reason_codes=[
                    _reason_code("execution", type(exc).__name__)
                ],
                snapshot=snapshot,
                workspace_spec_hash=None,
                campaign_event_chain_verified=True,
                workspace_verified=False,
                studio_event_chain_verified=False,
                snapshot_fixture_only=None,
                workflow_complete=False,
                runtime_contract_hash=(
                    runtime_contract.contract_hash
                    if runtime_contract is not None
                    else None
                ),
            )

    def verify(self, *, require_real: bool = True) -> bool:
        """Replay campaign, workspace, Studio event, and fixture boundaries."""

        try:
            with exclusive_file_lock(self.lock_path):
                spec = self.load_spec()
                events = self._read_events(spec)
                receipts = self._read_receipts(spec)
                if (
                    not events
                    or events[0].event_type != "CAMPAIGN_PREPARED"
                    or not receipts
                    or self._pending_intent(events) is not None
                ):
                    return False
                receipt = receipts[-1]
                if (
                    receipt.last_event_hash != events[-1].event_hash
                    or receipt.completed_actions
                    != self._completed_actions(events)
                ):
                    return False
                if require_real and (
                    self._is_control_execution()
                    or receipt.fixture_or_control
                    or receipt.terminal_status != "COMPLETED_LOCAL"
                ):
                    return False
                if self._is_control_execution():
                    if (
                        require_real
                        or receipt.terminal_status != "COMPLETED_CONTROL"
                        or not receipt.fixture_or_control
                        or receipt.completed_actions != list(self.action_order)
                    ):
                        return False
                    service = self._make_service()
                    snapshot = service.snapshot(spec.task_id)
                    if receipt.snapshot_hash != sha256_value(snapshot):
                        return False
                    for action in self.action_order:
                        proven, _ = self._action_postcondition(
                            spec,
                            action,
                            snapshot,
                        )
                        if not proven:
                            return False
                    return bool(
                        snapshot.get("backhalf", {}).get(
                            "workflow_complete"
                        )
                        and not receipt.scientific_qualification_granted
                        and not receipt.real_world_action_authorized
                    )
                runtime_contract = self._assert_runtime_contract(spec)
                service = self._make_service()
                snapshot = service.snapshot(spec.task_id)
                (
                    workspace_verified,
                    studio_chain_verified,
                    fixture_only,
                    workspace_spec_hash,
                    studio_event_tip_hash,
                    source_evidence_hash,
                    runtime_receipts_verified,
                ) = self._real_verification(
                    service, spec, snapshot, runtime_contract
                )
                return bool(
                    receipt.terminal_status == "COMPLETED_LOCAL"
                    and receipt.campaign_event_chain_verified
                    and workspace_verified
                    and studio_chain_verified
                    and fixture_only is False
                    and source_evidence_hash is not None
                    and runtime_receipts_verified
                    and snapshot.get("backhalf", {}).get(
                        "workflow_complete"
                    )
                    and receipt.workspace_spec_hash == workspace_spec_hash
                    and receipt.runtime_contract_hash
                    == runtime_contract.contract_hash
                    and receipt.source_evidence_hash
                    == source_evidence_hash
                    and receipt.studio_event_tip_hash
                    == studio_event_tip_hash
                    and receipt.snapshot_hash == sha256_value(snapshot)
                    and snapshot.get("scientific_qualification_granted")
                    is False
                    and snapshot.get("real_world_action_authorized")
                    is False
                    and not receipt.scientific_qualification_granted
                    and not receipt.real_world_action_authorized
                )
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            return False

    def status(
        self,
        *,
        lock_timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Return a read-only campaign projection without constructing a service."""

        if lock_timeout_seconds <= 0:
            raise ValueError("status lock timeout must be positive")
        prepared = any(
            path.exists()
            for path in (
                self.spec_path,
                self.events_path,
                self.freeze_receipt_path,
                self.terminal_receipts_path,
                self.workspace_root,
            )
        )
        if not prepared:
            authority_available = bool(
                self.authority_key is not None
                and len(self.authority_key) >= 32
            )
            return {
                "schema_version": "6.5-real-local-campaign-status",
                "prepared": False,
                "event_chain_verified": False,
                "terminal": False,
                "journal_current": False,
                "verified_current": False,
                "terminal_current": False,
                "authority_verification_available": authority_available,
                "verification_unavailable": not authority_available,
                "live_execution_default": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
        try:
            with exclusive_file_lock(
                self.lock_path,
                timeout_seconds=lock_timeout_seconds,
            ):
                spec = self.load_spec()
                events = self._read_events(spec)
                receipts = self._read_receipts(
                    spec, verify_authority=False
                )
                pending = self._pending_intent(events)
                current_tip = events[-1].event_hash if events else None
                journal_current = bool(
                    receipts
                    and pending is None
                    and receipts[-1].last_event_hash == current_tip
                )
                authority_available = bool(
                    self.authority_key is not None
                    and len(self.authority_key) >= 32
                )
                authority_receipts_verified = False
                if receipts and authority_available:
                    try:
                        self._read_receipts(
                            spec,
                            verify_authority=True,
                        )
                        authority_receipts_verified = True
                    except (
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        authority_receipts_verified = False
                verified_current = False
                if journal_current and authority_available:
                    latest = receipts[-1]
                    if latest.terminal_status == "COMPLETED_LOCAL":
                        verified_current = self.verify(require_real=True)
                    elif (
                        latest.terminal_status == "COMPLETED_CONTROL"
                        and self._is_control_execution()
                    ):
                        verified_current = self.verify(require_real=False)
                return {
                    "schema_version": "6.5-real-local-campaign-status",
                    "execution_state": "IDLE",
                    "prepared": True,
                    "campaign_id": spec.campaign_id,
                    "task_id": spec.task_id,
                    "spec_hash": spec.spec_hash,
                    "event_count": len(events),
                    "event_tip": current_tip,
                    "event_chain_verified": True,
                    "pending_action": pending.action if pending else None,
                    "completed_actions": self._completed_actions(events),
                    "terminal": bool(receipts),
                    "journal_current": journal_current,
                    "verified_current": verified_current,
                    "terminal_current": verified_current,
                    "authority_verification_available": (
                        authority_available
                    ),
                    "authority_receipts_verified": (
                        authority_receipts_verified
                    ),
                    "verification_unavailable": not authority_available,
                    "terminal_status": (
                        receipts[-1].terminal_status if receipts else None
                    ),
                    "fixture_or_control": (
                        receipts[-1].fixture_or_control
                        if receipts
                        else None
                    ),
                    "reconciliation_required": bool(
                        pending is not None
                        or (
                            receipts
                            and receipts[-1].terminal_status
                            == "HUMAN_RECONCILIATION_REQUIRED"
                        )
                        or (receipts and not journal_current)
                        or (
                            journal_current
                            and authority_available
                            and (
                                not authority_receipts_verified
                                or (
                                    receipts[-1].terminal_status
                                    in {
                                        "COMPLETED_LOCAL",
                                        "COMPLETED_CONTROL",
                                    }
                                    and not verified_current
                                )
                            )
                        )
                    ),
                    "live_execution_default": False,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
        except TimeoutError:
            authority_available = bool(
                self.authority_key is not None
                and len(self.authority_key) >= 32
            )
            return {
                "schema_version": "6.5-real-local-campaign-status",
                "execution_state": "RUNNING_OR_LOCKED",
                "prepared": True,
                "event_chain_verified": None,
                "terminal": None,
                "journal_current": False,
                "verified_current": False,
                "terminal_current": False,
                "authority_verification_available": authority_available,
                "authority_receipts_verified": None,
                "verification_unavailable": True,
                "reconciliation_required": None,
                "live_execution_default": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            authority_available = bool(
                self.authority_key is not None
                and len(self.authority_key) >= 32
            )
            return {
                "schema_version": "6.5-real-local-campaign-status",
                "execution_state": "INVALID_OR_UNREADABLE",
                "prepared": True,
                "event_chain_verified": False,
                "terminal": False,
                "journal_current": False,
                "verified_current": False,
                "terminal_current": False,
                "authority_verification_available": authority_available,
                "authority_receipts_verified": False,
                "verification_unavailable": not authority_available,
                "reconciliation_required": True,
                "error_type": type(exc).__name__,
                "live_execution_default": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }


# Concise alias for callers that do not need the version in every expression.
RealLocalCampaign = RealLocalCampaignRunnerV65


__all__ = [
    "CampaignConflictError",
    "CampaignEvent",
    "CampaignReconciliationRequired",
    "CampaignRetryRequired",
    "CampaignStageBlocked",
    "CampaignSpec",
    "CodexRuntimeBudgetsV65",
    "CodexRuntimeContractV65",
    "EVENTS_PATH_V65",
    "FREEZE_RECEIPT_PATH_V65",
    "FreezeReceipt",
    "LiveExecutionNotAuthorized",
    "RealLocalCampaign",
    "RealLocalCampaignError",
    "RealLocalCampaignEventV65",
    "RealLocalCampaignFreezeReceiptV65",
    "RealLocalCampaignRunnerV65",
    "RealLocalCampaignSpecV65",
    "RealLocalCampaignTerminalReceiptV65",
    "SPEC_PATH_V65",
    "TERMINAL_RECEIPTS_PATH_V65",
    "TerminalReceipt",
    "WslOuterTransportContractV65",
    "WORKSPACE_ROOT_V65",
    "build_codex_runtime_contract_v65",
]
