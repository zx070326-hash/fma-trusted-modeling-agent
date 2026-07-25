from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, Literal, Protocol
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from .controller import CandidatePreflightError, ModelingAgent
from .hashing import canonical_json, sha256_value
from .schemas import (
    ArtifactRef,
    CandidateRunOutcome,
    FiniteNumber,
    Identifier,
    OptimizationModelIR,
    ProblemContract,
    StrictModel,
)
from .storage import RunStore
from .validation import public_preflight_candidate


PROTOCOL_VERSION = "fma.codex_cli_explorer.v1"
DEFAULT_EXPECTED_CLI_VERSION = "0.144.6"
MAX_EXACT_ORACLE_ASSIGNMENTS = 100_000
MAX_ABS_MODEL_SCALAR = 1e12
MAX_ABS_DERIVED_VALUE = 1e15

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "codex_explorer.md"
_ALLOWED_EVENT_TYPES = {
    "thread.started",
    "turn.started",
    "turn.completed",
    "turn.failed",
    "item.started",
    "item.updated",
    "item.completed",
    "error",
}
_ALLOWED_ITEM_TYPES = {"reasoning", "agent_message"}
_DISABLED_FEATURES = (
    "shell_tool",
    "shell_snapshot",
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "image_generation",
    "multi_agent",
    "goals",
    "memories",
    "hooks",
    "workspace_dependencies",
    "skill_mcp_dependency_install",
    "tool_suggest",
    "auth_elicitation",
    "tool_call_mcp_elicitation",
)

BoundedAction = Annotated[str, Field(min_length=1, max_length=256)]


class ExplorerClauseView(StrictModel):
    clause_id: Identifier
    kind: Literal["objective", "hard_constraint", "assumption"]
    statement: Annotated[str, Field(min_length=3, max_length=4_000)]
    unit: Annotated[str, Field(min_length=1, max_length=128)]
    acceptance_criterion: Annotated[str, Field(min_length=3, max_length=4_000)]


class ExplorerFactView(StrictModel):
    fact_id: Identifier
    statement: Annotated[str, Field(min_length=3, max_length=2_000)]
    value: FiniteNumber
    unit: Annotated[str, Field(min_length=1, max_length=128)]


class ExplorerDecisionView(StrictModel):
    decision_id: Identifier
    statement: Annotated[str, Field(min_length=3, max_length=2_000)]
    kind: Literal["continuous", "integer", "binary"]
    unit: Annotated[str, Field(min_length=1, max_length=128)]
    lower_bound: FiniteNumber | None
    upper_bound: FiniteNumber | None


class ExplorerProblemView(StrictModel):
    """The only contract fields an untrusted Explorer is allowed to see."""

    contract_id: Identifier
    version: Annotated[int, Field(ge=1)]
    question: Annotated[str, Field(min_length=5, max_length=8_000)]
    system_boundary: Annotated[str, Field(min_length=3, max_length=8_000)]
    decision_horizon: Annotated[str, Field(min_length=3, max_length=2_000)]
    public_decisions: Annotated[list[ExplorerDecisionView], Field(max_length=128)]
    public_clauses: Annotated[list[ExplorerClauseView], Field(min_length=1, max_length=128)]
    public_facts: Annotated[list[ExplorerFactView], Field(max_length=512)]
    permitted_actions: Annotated[list[BoundedAction], Field(max_length=32)]
    forbidden_actions: Annotated[list[BoundedAction], Field(max_length=32)]
    risk_level: Literal["A0", "A1", "A2", "A3", "A4"]

    @classmethod
    def from_contract(cls, contract: ProblemContract) -> "ExplorerProblemView":
        contract.assert_frozen()
        return cls(
            contract_id=contract.contract_id,
            version=contract.version,
            question=contract.question,
            system_boundary=contract.system_boundary,
            decision_horizon=contract.decision_horizon,
            public_decisions=[
                ExplorerDecisionView(
                    decision_id=decision.decision_id,
                    statement=decision.statement,
                    kind=decision.kind,
                    unit=decision.unit,
                    lower_bound=decision.lower_bound,
                    upper_bound=decision.upper_bound,
                )
                for decision in contract.decisions
            ],
            public_clauses=[
                ExplorerClauseView(
                    clause_id=clause.clause_id,
                    kind=clause.kind.value,
                    statement=clause.statement,
                    unit=clause.unit,
                    acceptance_criterion=clause.acceptance_criterion,
                )
                for clause in contract.clauses
            ],
            public_facts=[
                ExplorerFactView(
                    fact_id=fact.fact_id,
                    statement=fact.statement,
                    value=fact.value,
                    unit=fact.unit,
                )
                for fact in contract.public_facts
            ],
            permitted_actions=contract.permitted_actions,
            forbidden_actions=contract.forbidden_actions,
            risk_level=contract.risk_level,
        )


class CodexLinearTerm(StrictModel):
    variable: Identifier
    coefficient: FiniteNumber


class CodexVariableDraft(StrictModel):
    name: Identifier
    kind: Literal["integer", "binary"]
    lower_bound: FiniteNumber
    upper_bound: FiniteNumber
    unit: Annotated[str, Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def validate_bounds(self) -> "CodexVariableDraft":
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"invalid bounds for {self.name}")
        if self.kind == "binary" and (
            self.lower_bound != 0 or self.upper_bound != 1
        ):
            raise ValueError("binary variables must use bounds [0, 1]")
        return self


class CodexObjectiveDraft(StrictModel):
    sense: Literal["minimize", "maximize"]
    coefficients: Annotated[list[CodexLinearTerm], Field(min_length=1, max_length=64)]
    constant: FiniteNumber
    unit: Annotated[str, Field(min_length=1, max_length=128)]
    contract_clause_ids: Annotated[list[Identifier], Field(min_length=1, max_length=32)]


class CodexConstraintDraft(StrictModel):
    constraint_id: Identifier
    coefficients: Annotated[list[CodexLinearTerm], Field(min_length=1, max_length=64)]
    sense: Literal["<=", ">=", "=="]
    rhs: FiniteNumber
    lhs_unit: Annotated[str, Field(min_length=1, max_length=128)]
    rhs_unit: Annotated[str, Field(min_length=1, max_length=128)]
    contract_clause_ids: Annotated[list[Identifier], Field(min_length=1, max_length=32)]


class CodexCandidateDraft(StrictModel):
    skeleton_id: Identifier
    evolution_operator: Annotated[str, Field(min_length=1, max_length=256)]
    rationale: Annotated[str, Field(min_length=3, max_length=2_000)]
    variables: Annotated[list[CodexVariableDraft], Field(min_length=1, max_length=32)]
    objective: CodexObjectiveDraft
    constraints: Annotated[list[CodexConstraintDraft], Field(min_length=1, max_length=128)]
    validation_obligations: Annotated[list[str], Field(min_length=1, max_length=16)]
    unresolved_assumptions: Annotated[list[str], Field(max_length=16)]


class CodexExplorerResponse(StrictModel):
    protocol_version: Literal[PROTOCOL_VERSION]
    request_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    status: Literal["proposed", "no_result"]
    candidates: Annotated[list[CodexCandidateDraft], Field(max_length=3)]
    no_result_reason: Annotated[str, Field(max_length=2_000)]
    notes: Annotated[list[str], Field(max_length=16)]

    @model_validator(mode="after")
    def validate_status(self) -> "CodexExplorerResponse":
        if self.status == "proposed" and not self.candidates:
            raise ValueError("proposed status requires at least one candidate")
        if self.status == "no_result" and self.candidates:
            raise ValueError("no_result status cannot contain candidates")
        if self.status == "no_result" and not self.no_result_reason.strip():
            raise ValueError("no_result status requires a reason")
        return self


class ExplorerErrorCode(str, Enum):
    CONTRACT_PROJECTION_INVALID = "contract_projection_invalid"
    CLI_NOT_FOUND = "cli_not_found"
    CLI_VERSION_MISMATCH = "cli_version_mismatch"
    AUTHENTICATION_ERROR = "authentication_error"
    MCP_PREFLIGHT_FAILED = "mcp_preflight_failed"
    TOOL_SURFACE_ACTIVE = "tool_surface_active"
    SCRATCH_GIT_ERROR = "scratch_git_error"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT_EXCEEDED = "output_limit_exceeded"
    NONZERO_EXIT = "nonzero_exit"
    MALFORMED_JSONL = "malformed_jsonl"
    TURN_FAILED = "turn_failed"
    POLICY_VIOLATION = "policy_violation"
    FINAL_MESSAGE_MISSING = "final_message_missing"
    OUTPUT_SCHEMA_INVALID = "output_schema_invalid"
    REQUEST_BINDING_FAILED = "request_binding_failed"
    SCRATCH_CHANGED = "scratch_changed"


class DraftRejection(StrictModel):
    draft_index: Annotated[int, Field(ge=0)]
    code: str
    detail: str


class ExplorationReceipt(StrictModel):
    receipt_version: Literal["1.0"] = "1.0"
    invocation_id: str
    request_id: str
    status: Literal["success", "no_result", "error"]
    error_code: str | None = None
    executable_path: str
    executable_sha256: str
    cli_version: str
    requested_model: str
    served_model_attested: Literal[False] = False
    public_view_hash: str
    prompt_hash: str
    output_schema_hash: str
    response_hash: str | None = None
    jsonl_hash: str | None = None
    observed_event_types: dict[str, int]
    observed_item_types: dict[str, int]
    tool_events_observed: Annotated[int, Field(ge=0)]
    thread_id: str | None = None
    usage: dict[str, int]
    scratch_unchanged: bool
    sandbox: Literal["read-only"] = "read-only"
    approval_policy: Literal["never"] = "never"
    user_config_ignored: Literal[True] = True
    session_persistence: Literal["ephemeral"] = "ephemeral"
    isolation_claim: Literal[
        "auth_only_codex_home_known_tools_disabled_zero_observed_tool_events"
    ] = "auth_only_codex_home_known_tools_disabled_zero_observed_tool_events"
    elapsed_ms: Annotated[int, Field(ge=0)]
    exit_code: int | None = None
    stdout_bytes: Annotated[int, Field(ge=0)]
    stderr_bytes: Annotated[int, Field(ge=0)]
    stderr_sha256: str
    attempt_index: Annotated[int, Field(ge=1)]
    invocation_manifest_artifact: ArtifactRef
    event_ledger_artifact: ArtifactRef
    stream_manifest_artifact: ArtifactRef


class ExplorationFailureReceipt(StrictModel):
    receipt_version: Literal["1.0"] = "1.0"
    status: Literal["error"] = "error"
    error_code: str
    message: str
    evidence: dict[str, object]


class ProposalBatch(StrictModel):
    status: Literal["success", "no_result"]
    candidates: list[OptimizationModelIR]
    rejections: list[DraftRejection]
    receipt: ExplorationReceipt
    receipt_artifact: ArtifactRef


class ExplorationRound(StrictModel):
    attempt_index: Annotated[int, Field(ge=1)]
    driver_status: Literal["success", "no_result", "error"]
    proposed_candidate_ids: list[str]
    public_rejections: list[DraftRejection]
    private_rejections: list[DraftRejection]
    assessed_candidate_ids: list[str]
    decision_statuses: list[str]
    feedback_disclosure: Literal["none", "public_diagnostics_only"]


class CodexAgentOutcome(StrictModel):
    outcome_version: Literal["1.0"] = "1.0"
    status: Literal[
        "validated",
        "run_invalid",
        "needs_evidence",
        "no_result",
        "driver_error",
        "permission_denied",
        "needs_approval",
    ]
    stop_reason: str
    driver_error_code: str | None = None
    exploration_directory: str
    rounds: list[ExplorationRound]
    candidate_outcomes: list[CandidateRunOutcome]


@dataclass(frozen=True)
class CodexCLIConfig:
    executable: Path | None = None
    requested_model: str | None = None
    expected_cli_version: str = DEFAULT_EXPECTED_CLI_VERSION
    timeout_seconds: int = 240
    max_candidates: int = 3
    max_input_bytes: int = 128 * 1024
    max_schema_bytes: int = 128 * 1024
    max_stdout_bytes: int = 4 * 1024 * 1024
    max_stderr_bytes: int = 1024 * 1024
    max_jsonl_line_bytes: int = 1024 * 1024
    max_events: int = 2048
    max_oracle_assignments: int = MAX_EXACT_ORACLE_ASSIGNMENTS

    def __post_init__(self) -> None:
        if not 1 <= self.max_candidates <= 3:
            raise ValueError("max_candidates must be between 1 and 3")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name in (
            "max_input_bytes",
            "max_schema_bytes",
            "max_stdout_bytes",
            "max_stderr_bytes",
            "max_jsonl_line_bytes",
            "max_events",
            "max_oracle_assignments",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


class ProcessRunner(Protocol):
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
    ) -> ProcessResult: ...


class CliLocator(Protocol):
    def __call__(self, explicit: str | Path | None = None) -> Path: ...


class PromptGuard(Protocol):
    def __call__(self, prompt: str) -> None: ...


class CodexCLIError(RuntimeError):
    def __init__(
        self,
        code: ExplorerErrorCode,
        message: str,
        *,
        trace_directory: Path,
        receipt_artifact: ArtifactRef | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.trace_directory = trace_directory
        self.receipt_artifact = receipt_artifact


class ProcessOutputLimitExceeded(RuntimeError):
    pass


def _terminate_owned_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                shell=False,
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()


def _default_process_runner(
    argv: list[str],
    *,
    cwd: Path,
    input_text: str | None,
    timeout_seconds: int,
    env: dict[str, str],
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> ProcessResult:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=env,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    overflow = threading.Event()
    writer_errors: list[BaseException] = []

    def read_bounded(
        stream: object,
        parts: list[bytes],
        limit: int,
    ) -> None:
        total = 0
        while True:
            chunk = stream.read(64 * 1024)  # type: ignore[attr-defined]
            if not chunk:
                return
            total += len(chunk)
            if total > limit:
                overflow.set()
                return
            parts.append(chunk)

    if process.stdout is None or process.stderr is None or process.stdin is None:
        _terminate_owned_process_tree(process)
        raise RuntimeError("failed to create Codex subprocess pipes")
    stdout_thread = threading.Thread(
        target=read_bounded,
        args=(process.stdout, stdout_parts, max_stdout_bytes),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_bounded,
        args=(process.stderr, stderr_parts, max_stderr_bytes),
        daemon=True,
    )
    input_bytes = input_text.encode("utf-8") if input_text is not None else b""

    def write_input() -> None:
        try:
            if input_bytes:
                process.stdin.write(input_bytes)  # type: ignore[union-attr]
            process.stdin.close()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as exc:
            writer_errors.append(exc)

    stdin_thread = threading.Thread(target=write_input, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    stdin_thread.start()
    try:
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            if overflow.is_set():
                _terminate_owned_process_tree(process)
                break
            if time.monotonic() >= deadline:
                _terminate_owned_process_tree(process)
                raise subprocess.TimeoutExpired(argv, timeout_seconds)
            time.sleep(0.01)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            _terminate_owned_process_tree(process)
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
        stdin_thread.join(timeout=10)
    if overflow.is_set():
        raise ProcessOutputLimitExceeded("Codex subprocess exceeded a stream byte limit")
    if writer_errors and process.returncode == 0:
        raise OSError(f"failed to stream prompt to Codex: {writer_errors[0]}")
    return ProcessResult(
        int(process.returncode),
        b"".join(stdout_parts).decode("utf-8", errors="replace"),
        b"".join(stderr_parts).decode("utf-8", errors="replace"),
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_process_env() -> dict[str, str]:
    allowed = {
        "APPDATA",
        "COMSPEC",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
        "CODEX_HOME",
    }
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def discover_codex_cli(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    elif os.environ.get("FMA_CODEX_BIN"):
        candidates.append(Path(os.environ["FMA_CODEX_BIN"]).expanduser())
    else:
        local_appdata = Path(os.environ.get("LOCALAPPDATA", ""))
        if str(local_appdata):
            candidates.append(
                local_appdata / "Programs" / "OpenAI" / "Codex" / "bin" / "codex.exe"
            )

        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        release_root = codex_home / "packages" / "standalone" / "releases"
        if release_root.is_dir():
            candidates.extend(
                sorted(
                    release_root.glob("*/bin/codex.exe"),
                    key=lambda path: path.parent.parent.name,
                    reverse=True,
                )
            )

        located = shutil.which("codex")
        if located:
            candidates.append(Path(located))
        candidates.append(codex_home / ".sandbox-bin" / "codex.exe")

    seen: set[str] = set()
    failures: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        if not resolved.is_file():
            continue
        try:
            completed = _default_process_runner(
                [str(resolved), "--version"],
                cwd=resolved.parent,
                input_text=None,
                timeout_seconds=10,
                env=_clean_process_env(),
                max_stdout_bytes=64 * 1024,
                max_stderr_bytes=64 * 1024,
            )
        except (OSError, subprocess.TimeoutExpired, ProcessOutputLimitExceeded) as exc:
            failures.append(f"{resolved}: {type(exc).__name__}")
            continue
        if completed.returncode == 0 and "codex-cli" in completed.stdout:
            return resolved
        failures.append(f"{resolved}: exit {completed.returncode}")
    detail = "; ".join(failures[-4:]) or "no candidate executable exists"
    raise FileNotFoundError(f"no runnable Codex CLI found ({detail})")


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    return json.loads(text, parse_constant=reject_constant)


def _tree_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        snapshot[path.relative_to(root).as_posix()] = _sha256_file(path)
    return snapshot


def _redacted_event_ledger(stdout: str) -> list[dict[str, object]]:
    """Keep protocol evidence while discarding reasoning, messages, and commands."""
    ledger: list[dict[str, object]] = []
    for sequence, line in enumerate(stdout.splitlines(), start=1):
        record: dict[str, object] = {
            "sequence": sequence,
            "line_bytes": len(line.encode("utf-8")),
            "line_sha256": _sha256_bytes(line.encode("utf-8")),
        }
        try:
            parsed = _strict_json_loads(line)
        except (ValueError, json.JSONDecodeError):
            record["valid_json"] = False
            ledger.append(record)
            continue
        record["valid_json"] = True
        if not isinstance(parsed, dict):
            record["json_kind"] = type(parsed).__name__
            ledger.append(record)
            continue
        event_type = parsed.get("type")
        record["event_type"] = event_type if isinstance(event_type, str) else None
        if event_type == "thread.started" and isinstance(parsed.get("thread_id"), str):
            record["thread_id"] = parsed["thread_id"]
        item = parsed.get("item")
        if isinstance(item, dict):
            for key in ("id", "type", "status"):
                if isinstance(item.get(key), str):
                    record[f"item_{key}"] = item[key]
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            record["usage"] = {
                str(key): int(value)
                for key, value in usage.items()
                if isinstance(value, int) and value >= 0
            }
        ledger.append(record)
    return ledger


def semantic_model_hash(ir: OptimizationModelIR) -> str:
    """Hash executable model semantics, excluding identity and prose lineage."""
    variables = sorted(
        (variable.model_dump(mode="json") for variable in ir.variables),
        key=lambda value: str(value["name"]),
    )
    objective = ir.objective.model_dump(mode="json")
    objective["contract_clause_ids"] = sorted(objective["contract_clause_ids"])
    constraints: list[dict[str, object]] = []
    for constraint in ir.constraints:
        value = constraint.model_dump(mode="json", exclude={"constraint_id"})
        value["contract_clause_ids"] = sorted(value["contract_clause_ids"])
        constraints.append(value)
    constraints.sort(key=canonical_json)
    return sha256_value(
        {
            "contract_hash": ir.contract_hash,
            "compiler_target": ir.compiler_target,
            "variables": variables,
            "objective": objective,
            "constraints": constraints,
        }
    )


def _mcp_disable_override(name: str) -> str:
    # Codex's CLI override parser treats quotes in a dotted key literally on
    # Windows.  Reject names that cannot be represented safely instead of
    # interpolating an ambiguous config path.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"MCP server id cannot be safely disabled: {name!r}")
    return f"mcp_servers.{name}.enabled=false"


def _terms_to_mapping(terms: list[CodexLinearTerm]) -> dict[str, float]:
    result: dict[str, float] = {}
    for term in terms:
        if term.variable in result:
            raise ValueError(f"duplicate coefficient term for {term.variable}")
        result[term.variable] = term.coefficient
    return result


def _oracle_assignment_count(draft: CodexCandidateDraft) -> int:
    count = 1
    for variable in draft.variables:
        lower = math.ceil(variable.lower_bound)
        upper = math.floor(variable.upper_bound)
        count *= max(0, upper - lower + 1)
    return count


def _enforce_numeric_budget(
    draft: CodexCandidateDraft,
    objective_coefficients: dict[str, float],
    constraints: list[dict[str, object]],
) -> None:
    bounds = {
        variable.name: max(abs(variable.lower_bound), abs(variable.upper_bound))
        for variable in draft.variables
    }
    scalars = [draft.objective.constant]
    scalars.extend(objective_coefficients.values())
    for variable in draft.variables:
        scalars.extend([variable.lower_bound, variable.upper_bound])
    for constraint in constraints:
        scalars.append(float(constraint["rhs"]))
        scalars.extend(float(value) for value in constraint["coefficients"].values())
    if any(abs(value) > MAX_ABS_MODEL_SCALAR for value in scalars):
        raise ValueError(
            f"a model scalar exceeds the absolute budget {MAX_ABS_MODEL_SCALAR:g}"
        )

    def exposure(coefficients: dict[str, float], constant: float = 0.0) -> float:
        values = [abs(constant)]
        for name, coefficient in coefficients.items():
            if name in bounds:
                values.append(abs(coefficient) * bounds[name])
        total = math.fsum(values)
        if not math.isfinite(total) or total > MAX_ABS_DERIVED_VALUE:
            raise ValueError(
                f"a derived linear expression exceeds {MAX_ABS_DERIVED_VALUE:g}"
            )
        return total

    exposure(objective_coefficients, draft.objective.constant)
    for constraint in constraints:
        exposure(constraint["coefficients"])  # type: ignore[arg-type]


def _draft_to_ir(
    draft: CodexCandidateDraft,
    contract: ProblemContract,
    *,
    attempt_index: int,
    draft_index: int,
    max_oracle_assignments: int,
) -> OptimizationModelIR:
    if draft.unresolved_assumptions:
        raise ValueError(
            "draft has unresolved assumptions: " + "; ".join(draft.unresolved_assumptions)
        )
    assignment_count = _oracle_assignment_count(draft)
    if assignment_count == 0:
        raise ValueError("draft has an empty integer domain")
    if assignment_count > max_oracle_assignments:
        raise ValueError(
            f"integer oracle space {assignment_count} exceeds {max_oracle_assignments}"
        )
    objective_coefficients = _terms_to_mapping(draft.objective.coefficients)
    constraints = []
    for constraint in draft.constraints:
        constraints.append(
            {
                "constraint_id": constraint.constraint_id,
                "coefficients": _terms_to_mapping(constraint.coefficients),
                "sense": constraint.sense,
                "rhs": constraint.rhs,
                "lhs_unit": constraint.lhs_unit,
                "rhs_unit": constraint.rhs_unit,
                "contract_clause_ids": constraint.contract_clause_ids,
            }
        )
    _enforce_numeric_budget(draft, objective_coefficients, constraints)
    digest = sha256_value(draft.model_dump(mode="json"))[:12]
    evolution_operator = draft.evolution_operator.strip()
    if evolution_operator.casefold() in {"none", "n/a", "na"}:
        evolution_operator = ""
    return OptimizationModelIR.seal(
        candidate_id=f"codex_r{attempt_index}_d{draft_index + 1}_{digest}",
        contract_hash=contract.frozen_hash,
        skeleton_id=draft.skeleton_id,
        lineage={
            "root_kind": "fresh",
            "parent_candidate_ids": [],
            "evolution_operator": evolution_operator or None,
            "rationale": draft.rationale,
        },
        variables=[variable.model_dump(mode="json") for variable in draft.variables],
        objective={
            "sense": draft.objective.sense,
            "coefficients": objective_coefficients,
            "constant": draft.objective.constant,
            "unit": draft.objective.unit,
            "contract_clause_ids": draft.objective.contract_clause_ids,
        },
        constraints=constraints,
        validation_obligations=draft.validation_obligations,
    )


@dataclass(frozen=True)
class _EventAudit:
    thread_id: str
    final_message: str
    event_counts: dict[str, int]
    item_counts: dict[str, int]
    usage: dict[str, int]
    tool_events: int


def _audit_jsonl(stdout: str, config: CodexCLIConfig) -> _EventAudit:
    encoded = stdout.encode("utf-8")
    if len(encoded) > config.max_stdout_bytes:
        raise ValueError("stdout exceeds the configured byte limit")
    lines = stdout.splitlines()
    if len(lines) > config.max_events:
        raise ValueError("JSONL event count exceeds the configured limit")

    event_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    thread_ids: list[str] = []
    completed_turns = 0
    started_turns = 0
    failed = False
    final_messages: list[str] = []
    usage: dict[str, int] = {}
    tool_events = 0
    for line in lines:
        if len(line.encode("utf-8")) > config.max_jsonl_line_bytes:
            raise ValueError("a JSONL event exceeds the configured line limit")
        parsed = _strict_json_loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("each JSONL event must be an object")
        event_type = parsed.get("type")
        if not isinstance(event_type, str) or event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError(f"unknown JSONL event type: {event_type!r}")
        event_counts[event_type] += 1
        if event_type == "thread.started":
            thread_id = parsed.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                raise ValueError("thread.started is missing thread_id")
            thread_ids.append(thread_id)
        elif event_type == "turn.started":
            started_turns += 1
        elif event_type == "turn.completed":
            completed_turns += 1
            raw_usage = parsed.get("usage", {})
            if isinstance(raw_usage, dict):
                usage = {
                    str(key): int(value)
                    for key, value in raw_usage.items()
                    if isinstance(value, int) and value >= 0
                }
        elif event_type in {"turn.failed", "error"}:
            failed = True
        elif event_type.startswith("item."):
            item = parsed.get("item")
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                raise ValueError("item event is missing a typed item")
            item_type = str(item["type"])
            item_counts[item_type] += 1
            if item_type not in _ALLOWED_ITEM_TYPES:
                tool_events += 1
            elif event_type == "item.completed" and item_type == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    final_messages.append(text)

    if failed:
        raise RuntimeError("Codex emitted an error or turn.failed event")
    if len(thread_ids) != 1 or started_turns != 1 or completed_turns != 1:
        raise ValueError("JSONL must contain exactly one thread and one completed turn")
    if tool_events:
        forbidden = sorted(item_type for item_type in item_counts if item_type not in _ALLOWED_ITEM_TYPES)
        raise PermissionError("forbidden Codex item types observed: " + ", ".join(forbidden))
    if not final_messages:
        raise LookupError("no completed agent_message was emitted")
    return _EventAudit(
        thread_id=thread_ids[0],
        final_message=final_messages[-1],
        event_counts=dict(event_counts),
        item_counts=dict(item_counts),
        usage=usage,
        tool_events=tool_events,
    )


class CodexCLIExplorer:
    """Untrusted Codex CLI proposal generator with an auditable outer harness."""

    def __init__(
        self,
        output_root: str | Path,
        config: CodexCLIConfig | None = None,
        *,
        process_runner: ProcessRunner | None = None,
        cli_locator: CliLocator | None = None,
        prompt_guard: PromptGuard | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config or CodexCLIConfig()
        self._uses_default_runner = process_runner is None
        self._process_runner = process_runner or _default_process_runner
        self._cli_locator = cli_locator or discover_codex_cli
        self._prompt_guard = prompt_guard
        self.store = RunStore(output_root, run_id=run_id)
        self._readiness: tuple[Path, str, str, list[str]] | None = None
        self._private_home: tempfile.TemporaryDirectory[str] | None = None

    @property
    def trace_directory(self) -> Path:
        return self.store.run_directory

    def _run_process(
        self,
        argv: list[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        timeout_seconds: int = 30,
    ) -> ProcessResult:
        env = _clean_process_env()
        if self._uses_default_runner:
            env["CODEX_HOME"] = str(self._ensure_private_codex_home())
        return self._process_runner(
            argv,
            cwd=cwd,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
            env=env,
            max_stdout_bytes=self.config.max_stdout_bytes,
            max_stderr_bytes=self.config.max_stderr_bytes,
        )

    def _ensure_private_codex_home(self) -> Path:
        if self._private_home is not None:
            return Path(self._private_home.name)
        source_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        auth_path = source_home / "auth.json"
        if not auth_path.is_file():
            self._raise(
                ExplorerErrorCode.AUTHENTICATION_ERROR,
                "saved Codex auth.json is unavailable for an isolated CLI home",
            )
        self._private_home = tempfile.TemporaryDirectory(prefix="fma-codex-auth-")
        private_home = Path(self._private_home.name)
        shutil.copy2(auth_path, private_home / "auth.json")
        return private_home

    def close(self) -> None:
        if self._private_home is not None:
            self._private_home.cleanup()
            self._private_home = None

    def _raise(
        self,
        code: ExplorerErrorCode,
        message: str,
        *,
        evidence: dict[str, object] | None = None,
    ) -> None:
        receipt = ExplorationFailureReceipt(
            error_code=code.value,
            message=message,
            evidence=evidence or {},
        )
        receipt_ref = self.store.put_artifact("exploration_failure_receipt", receipt)
        self.store.emit(
            "explorer_error",
            {
                "code": code.value,
                "message": message,
                "receipt": receipt_ref.model_dump(mode="json"),
            },
        )
        raise CodexCLIError(
            code,
            message,
            trace_directory=self.trace_directory,
            receipt_artifact=receipt_ref,
        )

    def _ensure_readiness(self) -> tuple[Path, str, str, list[str]]:
        if self._readiness is not None:
            return self._readiness
        try:
            executable = self._cli_locator(self.config.executable)
        except FileNotFoundError as exc:
            self._raise(ExplorerErrorCode.CLI_NOT_FOUND, str(exc))

        try:
            version_result = self._run_process(
                [str(executable), "--version"], cwd=self.trace_directory, timeout_seconds=10
            )
        except ProcessOutputLimitExceeded as exc:
            self._raise(ExplorerErrorCode.OUTPUT_LIMIT_EXCEEDED, str(exc))
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._raise(ExplorerErrorCode.CLI_NOT_FOUND, f"Codex version check failed: {exc}")
        if version_result.returncode != 0:
            self._raise(ExplorerErrorCode.CLI_NOT_FOUND, "Codex version check returned nonzero")
        cli_version = version_result.stdout.strip().removeprefix("codex-cli ").strip()
        if cli_version != self.config.expected_cli_version:
            self._raise(
                ExplorerErrorCode.CLI_VERSION_MISMATCH,
                f"expected Codex CLI {self.config.expected_cli_version}, got {cli_version}",
            )

        try:
            auth_result = self._run_process(
                [str(executable), "login", "status"],
                cwd=self.trace_directory,
                timeout_seconds=15,
            )
        except ProcessOutputLimitExceeded as exc:
            self._raise(ExplorerErrorCode.OUTPUT_LIMIT_EXCEEDED, str(exc))
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._raise(ExplorerErrorCode.AUTHENTICATION_ERROR, f"auth check failed: {exc}")
        auth_text = f"{auth_result.stdout}\n{auth_result.stderr}"
        if auth_result.returncode != 0 or "Logged in" not in auth_text:
            self._raise(ExplorerErrorCode.AUTHENTICATION_ERROR, "Codex CLI is not logged in")

        try:
            mcp_result = self._run_process(
                [str(executable), "mcp", "list", "--json"],
                cwd=self.trace_directory,
                timeout_seconds=20,
            )
            if mcp_result.returncode != 0:
                raise ValueError("mcp list returned nonzero")
            raw_servers = _strict_json_loads(mcp_result.stdout)
            if not isinstance(raw_servers, list):
                raise ValueError("mcp list did not return an array")
            server_names = sorted(
                {
                    str(server["name"])
                    for server in raw_servers
                    if isinstance(server, dict) and isinstance(server.get("name"), str)
                }
            )
            overrides = [_mcp_disable_override(name) for name in server_names]
            verify_argv = [str(executable), "mcp", "list", "--json"]
            for override in overrides:
                verify_argv.extend(["-c", override])
            verified = self._run_process(
                verify_argv, cwd=self.trace_directory, timeout_seconds=20
            )
            if verified.returncode != 0:
                raise ValueError("disabled MCP verification returned nonzero")
            verified_servers = _strict_json_loads(verified.stdout)
            if not isinstance(verified_servers, list):
                raise ValueError("disabled MCP verification did not return an array")
            enabled = [
                str(server.get("name"))
                for server in verified_servers
                if isinstance(server, dict) and server.get("enabled") is True
            ]
            if enabled:
                self._raise(
                    ExplorerErrorCode.TOOL_SURFACE_ACTIVE,
                    "MCP servers remain enabled after overrides: " + ", ".join(enabled),
                )
        except CodexCLIError:
            raise
        except ProcessOutputLimitExceeded as exc:
            self._raise(ExplorerErrorCode.OUTPUT_LIMIT_EXCEEDED, str(exc))
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            self._raise(ExplorerErrorCode.MCP_PREFLIGHT_FAILED, f"MCP preflight failed: {exc}")

        executable_hash = _sha256_file(executable)
        self.store.emit(
            "codex_readiness_verified",
            {
                "cli_version": cli_version,
                "executable_sha256": executable_hash,
                "disabled_mcp_ids": server_names,
                "served_model_attested": False,
            },
        )
        self._readiness = executable, cli_version, executable_hash, server_names
        return self._readiness

    def _initialize_scratch(self, invocation_id: str) -> Path:
        scratch = self.trace_directory / "cli_calls" / invocation_id
        scratch.mkdir(parents=True, exist_ok=False)
        try:
            completed = subprocess.run(
                ["git", "init", "--quiet"],
                cwd=scratch,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._raise(ExplorerErrorCode.SCRATCH_GIT_ERROR, f"git init failed: {exc}")
        if completed.returncode != 0:
            self._raise(ExplorerErrorCode.SCRATCH_GIT_ERROR, "git init returned nonzero")
        return scratch

    def _build_argv(
        self,
        executable: Path,
        scratch: Path,
        schema_path: Path,
        server_names: list[str],
    ) -> list[str]:
        argv = [
            str(executable),
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "-C",
            str(scratch),
            "--strict-config",
        ]
        if self.config.requested_model:
            argv.extend(["--model", self.config.requested_model])
        for feature in _DISABLED_FEATURES:
            argv.extend(["--disable", feature])
        for override in (
            'web_search="disabled"',
            'shell_environment_policy.inherit="none"',
            'history.persistence="none"',
            "project_doc_max_bytes=0",
            "hide_agent_reasoning=true",
            "check_for_update_on_startup=false",
        ):
            argv.extend(["-c", override])
        for name in server_names:
            argv.extend(["-c", f'mcp_servers.{name}.command="disabled"'])
            argv.extend(["-c", _mcp_disable_override(name)])
        argv.extend(
            [
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--json",
                "--color",
                "never",
                "--output-schema",
                str(schema_path),
                "-",
            ]
        )
        return argv

    def propose_batch(
        self,
        contract: ProblemContract,
        *,
        public_feedback: list[str] | None = None,
        attempt_index: int = 1,
    ) -> ProposalBatch:
        contract.assert_frozen()
        invocation_id = f"call-{attempt_index:02d}-{uuid4().hex[:8]}"
        request_id = uuid4().hex
        schema = CodexExplorerResponse.model_json_schema()
        schema_text = canonical_json(schema)
        if len(schema_text.encode("utf-8")) > self.config.max_schema_bytes:
            self._raise(ExplorerErrorCode.OUTPUT_LIMIT_EXCEEDED, "output schema is too large")
        try:
            public_view = ExplorerProblemView.from_contract(contract)
        except ValidationError as exc:
            self._raise(
                ExplorerErrorCode.CONTRACT_PROJECTION_INVALID,
                f"contract cannot fit the bounded Explorer view: {str(exc)[:1800]}",
            )
        input_payload = {
            "protocol_version": PROTOCOL_VERSION,
            "operation": "propose_bounded_integer_linear_models",
            "request_id": request_id,
            "public_problem": public_view.model_dump(mode="json"),
            "public_feedback": list(public_feedback or []),
            "limits": {
                "max_candidates": self.config.max_candidates,
                "allowed_variable_kinds": ["integer", "binary"],
                "max_oracle_assignments": self.config.max_oracle_assignments,
                "compiler_target": "scipy_milp_v1",
                "tools_allowed": False,
                "execution_allowed": False,
            },
        }
        prompt = _PROMPT_PATH.read_text(encoding="utf-8").rstrip() + "\n\nINPUT_JSON\n" + canonical_json(input_payload) + "\n"
        if self._prompt_guard is not None:
            self._prompt_guard(prompt)
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > self.config.max_input_bytes:
            self._raise(ExplorerErrorCode.OUTPUT_LIMIT_EXCEEDED, "Explorer input is too large")

        executable, cli_version, executable_hash, server_names = self._ensure_readiness()
        scratch = self._initialize_scratch(invocation_id)
        schema_path = scratch / "explorer-output.schema.json"
        schema_path.write_text(schema_text + "\n", encoding="utf-8")

        prompt_ref = self.store.put_artifact(
            "explorer_public_prompt",
            {
                "prompt": prompt,
                "excluded_private_fields": [
                    "acceptance tests",
                    "frozen hash",
                    "source references",
                    "prior solutions",
                    "validation results",
                ],
            },
        )
        schema_ref = self.store.put_artifact("explorer_output_schema", schema)
        before = _tree_snapshot(scratch)
        argv = self._build_argv(executable, scratch, schema_path, server_names)
        invocation_manifest = {
            "manifest_version": "1.0",
            "invocation_id": invocation_id,
            "request_id": request_id,
            "attempt_index": attempt_index,
            "argv": argv,
            "argv_hash": sha256_value(argv),
            "cwd": str(scratch),
            "prompt_via_stdin": True,
            "prompt": prompt_ref.model_dump(mode="json"),
            "schema": schema_ref.model_dump(mode="json"),
            "scratch_before_hash": sha256_value(before),
            "environment_policy": {
                "codex_home": "temporary_auth_only",
                "child_shell_inherit": "none",
                "secret_environment_allowlist": False,
            },
            "limits": {
                "timeout_seconds": self.config.timeout_seconds,
                "max_stdout_bytes": self.config.max_stdout_bytes,
                "max_stderr_bytes": self.config.max_stderr_bytes,
                "max_events": self.config.max_events,
            },
        }
        invocation_manifest_ref = self.store.put_artifact(
            "explorer_invocation_manifest", invocation_manifest
        )
        self.store.emit(
            "explorer_invocation_started",
            {
                "invocation_id": invocation_id,
                "request_id": request_id,
                "attempt_index": attempt_index,
                "manifest": invocation_manifest_ref.model_dump(mode="json"),
            },
        )
        invocation_evidence = {
            "invocation_manifest": invocation_manifest_ref.model_dump(mode="json")
        }
        started = time.monotonic()
        try:
            result = self._run_process(
                argv,
                cwd=scratch,
                input_text=prompt,
                timeout_seconds=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self._raise(
                ExplorerErrorCode.TIMEOUT,
                f"Codex invocation exceeded {self.config.timeout_seconds} seconds",
                evidence=invocation_evidence,
            )
        except ProcessOutputLimitExceeded as exc:
            self._raise(
                ExplorerErrorCode.OUTPUT_LIMIT_EXCEEDED,
                str(exc),
                evidence=invocation_evidence,
            )
        except OSError as exc:
            self._raise(
                ExplorerErrorCode.NONZERO_EXIT,
                f"Codex process failed: {exc}",
                evidence=invocation_evidence,
            )
        elapsed_ms = max(0, round((time.monotonic() - started) * 1000))

        stdout_bytes = len(result.stdout.encode("utf-8"))
        stderr_bytes = len(result.stderr.encode("utf-8"))
        after = _tree_snapshot(scratch)
        scratch_unchanged = before == after
        changed_paths = sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        )
        event_ledger_ref = self.store.put_artifact(
            "codex_event_ledger", _redacted_event_ledger(result.stdout)
        )
        stream_manifest_ref = self.store.put_artifact(
            "codex_stream_manifest",
            {
                "manifest_version": "1.0",
                "invocation_id": invocation_id,
                "returncode": result.returncode,
                "elapsed_ms": elapsed_ms,
                "stdout_bytes": stdout_bytes,
                "stdout_sha256": _sha256_bytes(result.stdout.encode("utf-8")),
                "stderr_bytes": stderr_bytes,
                "stderr_sha256": _sha256_bytes(result.stderr.encode("utf-8")),
                "event_ledger": event_ledger_ref.model_dump(mode="json"),
                "scratch_before_hash": sha256_value(before),
                "scratch_after_hash": sha256_value(after),
                "scratch_changed_paths": changed_paths,
            },
        )
        result_evidence = {
            **invocation_evidence,
            "event_ledger": event_ledger_ref.model_dump(mode="json"),
            "stream_manifest": stream_manifest_ref.model_dump(mode="json"),
        }
        if stdout_bytes > self.config.max_stdout_bytes or stderr_bytes > self.config.max_stderr_bytes:
            self._raise(
                ExplorerErrorCode.OUTPUT_LIMIT_EXCEEDED,
                "Codex output exceeded a byte limit",
                evidence=result_evidence,
            )
        if result.returncode != 0:
            self._raise(
                ExplorerErrorCode.NONZERO_EXIT,
                f"Codex returned exit code {result.returncode}; stderr_sha256={_sha256_bytes(result.stderr.encode('utf-8'))}",
                evidence=result_evidence,
            )

        try:
            audit = _audit_jsonl(result.stdout, self.config)
        except PermissionError as exc:
            self._raise(
                ExplorerErrorCode.POLICY_VIOLATION, str(exc), evidence=result_evidence
            )
        except RuntimeError as exc:
            self._raise(ExplorerErrorCode.TURN_FAILED, str(exc), evidence=result_evidence)
        except LookupError as exc:
            self._raise(
                ExplorerErrorCode.FINAL_MESSAGE_MISSING,
                str(exc),
                evidence=result_evidence,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._raise(
                ExplorerErrorCode.MALFORMED_JSONL, str(exc), evidence=result_evidence
            )

        if not scratch_unchanged:
            self._raise(
                ExplorerErrorCode.SCRATCH_CHANGED,
                "Codex invocation changed the isolated scratch repository",
                evidence=result_evidence,
            )
        try:
            parsed_response = _strict_json_loads(audit.final_message)
            response = CodexExplorerResponse.model_validate(parsed_response)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            self._raise(
                ExplorerErrorCode.OUTPUT_SCHEMA_INVALID,
                f"invalid structured output: {exc}",
                evidence=result_evidence,
            )
        if response.request_id != request_id:
            self._raise(
                ExplorerErrorCode.REQUEST_BINDING_FAILED,
                "Codex response request_id does not match the invocation nonce",
                evidence=result_evidence,
            )
        if len(response.candidates) > self.config.max_candidates:
            self._raise(
                ExplorerErrorCode.OUTPUT_SCHEMA_INVALID,
                "Codex response exceeds the configured candidate budget",
                evidence=result_evidence,
            )

        response_ref = self.store.put_artifact("explorer_response", response)
        rejections: list[DraftRejection] = []
        candidates: list[OptimizationModelIR] = []
        if response.status == "proposed":
            for draft_index, draft in enumerate(response.candidates):
                try:
                    candidates.append(
                        _draft_to_ir(
                            draft,
                            contract,
                            attempt_index=attempt_index,
                            draft_index=draft_index,
                            max_oracle_assignments=self.config.max_oracle_assignments,
                        )
                    )
                except (ValueError, ValidationError) as exc:
                    rejections.append(
                        DraftRejection(
                            draft_index=draft_index,
                            code="draft_conversion_rejected",
                            detail=str(exc),
                        )
                    )

        receipt = ExplorationReceipt(
            invocation_id=invocation_id,
            request_id=request_id,
            status="no_result" if response.status == "no_result" else "success",
            executable_path=str(executable),
            executable_sha256=executable_hash,
            cli_version=cli_version,
            requested_model=self.config.requested_model or "cli_default",
            public_view_hash=sha256_value(public_view.model_dump(mode="json")),
            prompt_hash=_sha256_bytes(prompt_bytes),
            output_schema_hash=_sha256_bytes(schema_text.encode("utf-8")),
            response_hash=response_ref.sha256,
            jsonl_hash=_sha256_bytes(result.stdout.encode("utf-8")),
            observed_event_types=audit.event_counts,
            observed_item_types=audit.item_counts,
            tool_events_observed=audit.tool_events,
            thread_id=audit.thread_id,
            usage=audit.usage,
            scratch_unchanged=scratch_unchanged,
            elapsed_ms=elapsed_ms,
            exit_code=result.returncode,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            stderr_sha256=_sha256_bytes(result.stderr.encode("utf-8")),
            attempt_index=attempt_index,
            invocation_manifest_artifact=invocation_manifest_ref,
            event_ledger_artifact=event_ledger_ref,
            stream_manifest_artifact=stream_manifest_ref,
        )
        receipt_ref = self.store.put_artifact("exploration_receipt", receipt)
        self.store.emit(
            "explorer_invocation_completed",
            {
                "invocation_id": invocation_id,
                "status": receipt.status,
                "candidate_count": len(candidates),
                "rejection_count": len(rejections),
                "response": response_ref.model_dump(mode="json"),
                "receipt": receipt_ref.model_dump(mode="json"),
            },
        )
        if not self.store.verify_event_chain():
            raise RuntimeError("exploration event hash chain failed")
        return ProposalBatch(
            status="no_result" if response.status == "no_result" else "success",
            candidates=candidates,
            rejections=rejections,
            receipt=receipt,
            receipt_artifact=receipt_ref,
        )

    def propose(self, contract: ProblemContract) -> list[OptimizationModelIR]:
        return self.propose_batch(contract).candidates


class CodexDrivenModelingAgent:
    """Bounded Explorer loop; the deterministic FMA kernel remains authoritative."""

    def __init__(
        self,
        output_root: str | Path,
        config: CodexCLIConfig | None = None,
        *,
        max_rounds: int = 1,
        process_runner: ProcessRunner | None = None,
        cli_locator: CliLocator | None = None,
        prompt_guard: PromptGuard | None = None,
        allow_high_risk: bool = False,
    ) -> None:
        if not 1 <= max_rounds <= 2:
            raise ValueError("max_rounds must be one or two")
        self.output_root = Path(output_root).resolve()
        self.config = config or CodexCLIConfig()
        self.max_rounds = max_rounds
        self.allow_high_risk = allow_high_risk
        self._process_runner = process_runner
        self._cli_locator = cli_locator
        self._prompt_guard = prompt_guard
        self.explorer: CodexCLIExplorer | None = None
        self.kernel: ModelingAgent | None = None

    def _initialize_components(self) -> None:
        if self.explorer is not None:
            return
        self.explorer = CodexCLIExplorer(
            self.output_root / "explorations",
            self.config,
            process_runner=self._process_runner,
            cli_locator=self._cli_locator,
            prompt_guard=self._prompt_guard,
        )
        self.kernel = ModelingAgent(
            self.output_root / "candidate_runs",
            max_candidates=self.config.max_candidates * self.max_rounds,
        )

    def _permission_outcome(self, contract: ProblemContract) -> CodexAgentOutcome | None:
        required = {
            "codex_cli_inference",
            "local_compute",
            "write_local_run_artifacts",
        }
        permitted = set(contract.permitted_actions)
        forbidden = set(contract.forbidden_actions)
        missing = sorted(required - permitted)
        conflicts = sorted(required & forbidden)
        if "network_access" in forbidden:
            conflicts.append("network_access blocks codex_cli_inference")
        if missing or conflicts:
            reasons = []
            if missing:
                reasons.append("missing permitted actions: " + ", ".join(missing))
            if conflicts:
                reasons.append("forbidden action conflicts: " + ", ".join(conflicts))
            return CodexAgentOutcome(
                status="permission_denied",
                stop_reason="; ".join(reasons),
                exploration_directory="",
                rounds=[],
                candidate_outcomes=[],
            )
        if contract.risk_level in {"A3", "A4"} and not self.allow_high_risk:
            return CodexAgentOutcome(
                status="needs_approval",
                stop_reason=(
                    f"risk level {contract.risk_level} requires an explicit high-risk approval"
                ),
                exploration_directory="",
                rounds=[],
                candidate_outcomes=[],
            )
        return None

    def _finish(
        self,
        *,
        status: Literal[
            "validated", "run_invalid", "needs_evidence", "no_result", "driver_error"
        ],
        stop_reason: str,
        rounds: list[ExplorationRound],
        outcomes: list[CandidateRunOutcome],
        driver_error_code: str | None = None,
    ) -> CodexAgentOutcome:
        if self.explorer is None:
            raise RuntimeError("Explorer components were not initialized")
        result = CodexAgentOutcome(
            status=status,
            stop_reason=stop_reason,
            driver_error_code=driver_error_code,
            exploration_directory=str(self.explorer.trace_directory),
            rounds=rounds,
            candidate_outcomes=outcomes,
        )
        result_ref = self.explorer.store.put_artifact("codex_agent_outcome", result)
        self.explorer.store.emit(
            "codex_agent_stopped",
            {
                "status": status,
                "stop_reason": stop_reason,
                "result": result_ref.model_dump(mode="json"),
            },
        )
        if not self.explorer.store.verify_event_chain():
            raise RuntimeError("exploration event hash chain failed at controller stop")
        self.explorer.close()
        return result

    def run(self, contract: ProblemContract) -> CodexAgentOutcome:
        contract.assert_frozen()
        permission_outcome = self._permission_outcome(contract)
        if permission_outcome is not None:
            return permission_outcome
        try:
            ExplorerProblemView.from_contract(contract)
        except ValidationError as exc:
            return CodexAgentOutcome(
                status="driver_error",
                stop_reason=f"contract cannot fit the bounded Explorer view: {str(exc)[:1800]}",
                driver_error_code=ExplorerErrorCode.CONTRACT_PROJECTION_INVALID.value,
                exploration_directory="",
                rounds=[],
                candidate_outcomes=[],
            )
        self._initialize_components()
        try:
            return self._run_authorized(contract)
        finally:
            if self.explorer is not None:
                self.explorer.close()

    def _run_authorized(self, contract: ProblemContract) -> CodexAgentOutcome:
        if self.explorer is None or self.kernel is None:
            raise RuntimeError("failed to initialize Codex agent components")
        rounds: list[ExplorationRound] = []
        outcomes: list[CandidateRunOutcome] = []
        feedback: list[str] = []
        seen_model_hashes: set[str] = set()

        for attempt_index in range(1, self.max_rounds + 1):
            try:
                batch = self.explorer.propose_batch(
                    contract,
                    public_feedback=feedback,
                    attempt_index=attempt_index,
                )
            except CodexCLIError as exc:
                rounds.append(
                    ExplorationRound(
                        attempt_index=attempt_index,
                        driver_status="error",
                        proposed_candidate_ids=[],
                        public_rejections=[],
                        private_rejections=[],
                        assessed_candidate_ids=[],
                        decision_statuses=[],
                        feedback_disclosure="public_diagnostics_only" if feedback else "none",
                    )
                )
                return self._finish(
                    status="driver_error",
                    stop_reason=str(exc),
                    driver_error_code=exc.code.value,
                    rounds=rounds,
                    outcomes=outcomes,
                )

            if batch.status == "no_result":
                rounds.append(
                    ExplorationRound(
                        attempt_index=attempt_index,
                        driver_status="no_result",
                        proposed_candidate_ids=[],
                        public_rejections=batch.rejections,
                        private_rejections=[],
                        assessed_candidate_ids=[],
                        decision_statuses=[],
                        feedback_disclosure="public_diagnostics_only" if feedback else "none",
                    )
                )
                return self._finish(
                    status="no_result",
                    stop_reason="Codex Explorer abstained because the public problem was insufficient",
                    rounds=rounds,
                    outcomes=outcomes,
                )

            public_rejections = list(batch.rejections)
            private_rejections: list[DraftRejection] = []
            public_valid: list[OptimizationModelIR] = []
            for candidate_index, candidate in enumerate(batch.candidates):
                model_hash = semantic_model_hash(candidate)
                if model_hash in seen_model_hashes:
                    public_rejections.append(
                        DraftRejection(
                            draft_index=candidate_index,
                            code="duplicate_candidate",
                            detail="candidate duplicates a prior round",
                        )
                    )
                    continue
                seen_model_hashes.add(model_hash)
                failures = public_preflight_candidate(contract, candidate)
                if failures:
                    public_rejections.append(
                        DraftRejection(
                            draft_index=candidate_index,
                            code="public_preflight_rejected",
                            detail="; ".join(failures),
                        )
                    )
                    continue
                public_valid.append(candidate)

            assessed: list[str] = []
            decisions: list[str] = []
            validated = False
            for candidate_index, candidate in enumerate(public_valid):
                try:
                    outcome = self.kernel.assess_candidate(contract, candidate)
                except CandidatePreflightError as exc:
                    private_rejections.append(
                        DraftRejection(
                            draft_index=candidate_index,
                            code="private_preflight_rejected",
                            detail=str(exc),
                        )
                    )
                    continue
                except Exception as exc:
                    private_rejections.append(
                        DraftRejection(
                            draft_index=candidate_index,
                            code="candidate_execution_error",
                            detail=f"{type(exc).__name__}: {str(exc)[:1800]}",
                        )
                    )
                    continue
                outcomes.append(outcome)
                assessed.append(candidate.candidate_id)
                decisions.append(outcome.decision.status)
                if outcome.decision.status == "validated":
                    validated = True
                    break

            rounds.append(
                ExplorationRound(
                    attempt_index=attempt_index,
                    driver_status="success",
                    proposed_candidate_ids=[candidate.candidate_id for candidate in batch.candidates],
                    public_rejections=public_rejections,
                    private_rejections=private_rejections,
                    assessed_candidate_ids=assessed,
                    decision_statuses=decisions,
                    feedback_disclosure="public_diagnostics_only" if feedback else "none",
                )
            )
            if validated:
                return self._finish(
                    status="validated",
                    stop_reason="a Codex-proposed IR passed the deterministic trusted kernel",
                    rounds=rounds,
                    outcomes=outcomes,
                )

            # Private evaluator outputs never enter another model prompt.  A
            # repair round is permitted only when every rejection is derivable
            # from the public contract view or the wire schema.
            if public_valid:
                final_status: Literal["run_invalid", "needs_evidence"] = (
                    "needs_evidence"
                    if any(outcome.decision.status == "needs_evidence" for outcome in outcomes)
                    else "run_invalid"
                )
                return self._finish(
                    status=final_status,
                    stop_reason="publicly valid candidates did not pass the private trusted kernel",
                    rounds=rounds,
                    outcomes=outcomes,
                )
            if attempt_index == self.max_rounds:
                return self._finish(
                    status="no_result",
                    stop_reason="all drafts failed public checks within the bounded repair budget",
                    rounds=rounds,
                    outcomes=outcomes,
                )
            feedback = [
                f"draft {rejection.draft_index + 1}: {rejection.detail}"
                for rejection in public_rejections[:8]
            ]
            if not feedback:
                return self._finish(
                    status="no_result",
                    stop_reason="the Explorer made no auditable progress",
                    rounds=rounds,
                    outcomes=outcomes,
                )

        raise AssertionError("bounded loop terminated without a result")
