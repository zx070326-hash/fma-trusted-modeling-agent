"""Codex CLI adapters with explicit sandbox, network, and trace contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Protocol

from .storage import atomic_write_json, file_hash, now


NETWORK_MODES = {
    "research-search",
    "source-review",
    "offline-compute",
    "delivery",
}


def _validate_network_mode(value: str) -> str:
    if value not in NETWORK_MODES:
        raise ValueError(f"invalid network mode: {value}")
    return value


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the JSON-Schema subset used by this package after CLI return."""

    if "anyOf" in schema:
        failures = []
        for alternative in schema["anyOf"]:
            try:
                _validate_schema(value, alternative, path)
                return
            except ValueError as exc:
                failures.append(str(exc))
        raise ValueError(f"{path} does not match any allowed schema: {failures}")
    expected = schema.get("type")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }
    if expected in valid and not valid[expected]:
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed enum value")
    if expected == "object":
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path} has unexpected fields: {sorted(extra)}")
        for key, child in value.items():
            if key in properties:
                _validate_schema(child, properties[key], f"{path}.{key}")
    if expected == "array":
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has too many items")
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} has too few items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")


class ModelAdapter(Protocol):
    def complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: str,
        workspace: Path,
        network_mode: str = "offline-compute",
        trace_path: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]: ...


class NativeResearcherAdapter(Protocol):
    def run(
        self,
        prompt: str,
        *,
        role: str,
        workspace: Path,
        trace_path: Path,
        timeout_seconds: float,
        network_mode: str = "offline-compute",
    ) -> dict[str, Any]: ...


class ScriptedModel:
    """Deterministic structured adapter for contract tests."""

    def __init__(
        self,
        responses: list[dict[str, Any]],
        *,
        receipts: list[dict[str, Any]] | None = None,
    ):
        self.responses = deque(responses)
        self.receipts = deque(receipts or [])
        self.calls: list[dict[str, Any]] = []
        self.last_receipt: dict[str, Any] | None = None

    def complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: str,
        workspace: Path,
        network_mode: str = "offline-compute",
        trace_path: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        _validate_network_mode(network_mode)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise TimeoutError(f"no time remains for role={role}")
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "role": role,
                "workspace": workspace,
                "network_mode": network_mode,
                "trace_path": trace_path,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.responses:
            raise RuntimeError(f"no scripted response remains for role={role}")
        value = self.responses.popleft()
        _validate_schema(value, schema)
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps({"type": "scripted", "role": role}) + "\n",
                encoding="utf-8",
            )
        self.last_receipt = {
            "role": role,
            "network_mode": network_mode,
            "observable_web_calls": 0,
            "observable_web_queries": [],
            "observable_tool_calls": 0,
            "sandbox": "read-only",
            "tool_free": True,
            "ephemeral": True,
            "trace_sha256": file_hash(trace_path) if trace_path is not None else "scripted",
            **(self.receipts.popleft() if self.receipts else {}),
        }
        return value


def discover_codex_cli(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    configured = os.environ.get("MODELING_AGENT_CODEX_BIN")
    if configured:
        candidates.append(Path(configured).expanduser())
    codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    releases = codex_root / "packages" / "standalone" / "releases"
    if releases.is_dir():
        candidates.extend(
            sorted(
                releases.glob("*/bin/codex.exe"),
                key=lambda item: item.parent.parent.name,
                reverse=True,
            )
        )
    located = shutil.which("codex")
    if located:
        candidates.append(Path(located))
    failures: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key in seen or not resolved.is_file():
            continue
        seen.add(key)
        try:
            result = subprocess.run(
                [str(resolved), "--version"],
                cwd=resolved.parent,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                shell=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{resolved}: {type(exc).__name__}")
            continue
        if result.returncode == 0 and "codex-cli" in result.stdout:
            return resolved
        failures.append(f"{resolved}: exit={result.returncode}")
    detail = "; ".join(failures[-3:]) or "no candidate found"
    raise FileNotFoundError(f"no runnable Codex CLI ({detail})")


def _clean_environment() -> dict[str, str]:
    allowed = {
        "APPDATA",
        "CODEX_HOME",
        "COMSPEC",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
    return {key: value for key, value in os.environ.items() if key.upper() in allowed}


def _trace_counts(path: Path | None) -> dict[str, Any]:
    tool_ids: set[str] = set()
    web_ids: set[str] = set()
    web_queries: set[str] = set()
    commands: set[str] = set()
    messages = 0
    if path is None or not path.is_file():
        return {
            "observable_tool_calls": 0,
            "observable_web_calls": 0,
            "observable_web_queries": [],
            "observable_commands": [],
            "agent_messages": 0,
        }
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "agent_message" and event.get("type") == "item.completed":
            messages += 1
        if kind in {"command_execution", "file_change", "mcp_tool_call", "tool_call"}:
            tool_ids.add(str(item.get("id") or event.get("id") or number))
        if isinstance(kind, str) and ("web" in kind or "search" in kind):
            web_ids.add(str(item.get("id") or event.get("id") or number))
            query = item.get("query")
            if isinstance(query, str) and query.strip():
                web_queries.add(query.strip())
        if kind == "command_execution":
            command = item.get("command")
            if isinstance(command, str) and command.strip():
                commands.add(command.strip())
    return {
        "observable_tool_calls": len(tool_ids),
        "observable_web_calls": len(web_ids),
        "observable_web_queries": sorted(web_queries),
        "observable_commands": sorted(commands),
        "agent_messages": messages,
    }


_DISABLED_SHARED = (
    "apps",
    "plugins",
    "browser_use",
    "computer_use",
    "multi_agent",
    "goals",
    "memories",
    "hooks",
)

_WORKSPACE_ONLY_PROFILE = "modeling-workspace-only"


def _workspace_only_profile_args() -> list[str]:
    prefix = f"permissions.{_WORKSPACE_ONLY_PROFILE}"
    return [
        "-c",
        f'{prefix}.extends=":workspace"',
        "-c",
        f'{prefix}.filesystem.:root="deny"',
        "-c",
        f'{prefix}.filesystem.:minimal="read"',
        "-c",
        f'{prefix}.filesystem.:tmpdir="deny"',
        "-c",
        f'{prefix}.filesystem.:slash_tmp="deny"',
    ]


def ensure_workspace_only_sandbox(
    executable: Path, workspace: Path, *, timeout_seconds: float
) -> None:
    """Prove allowed workspace reads and denied parent reads with real commands."""

    token = f"THIN_SANDBOX_{uuid.uuid4().hex}"
    allowed = workspace / f".sandbox-allowed-{uuid.uuid4().hex}.txt"
    denied = workspace.parent / f".sandbox-denied-{uuid.uuid4().hex}.txt"
    allowed.write_text(token, encoding="utf-8")
    denied.write_text(token, encoding="utf-8")
    base = [
        str(executable),
        "sandbox",
        "-C",
        str(workspace),
        *_workspace_only_profile_args(),
        "-P",
        _WORKSPACE_ONLY_PROFILE,
        "--",
    ]

    def read_command(path: Path) -> list[str]:
        if os.name == "nt":
            return [
                os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"),
                "/d",
                "/c",
                "type",
                str(path),
            ]
        return [shutil.which("cat") or "/bin/cat", str(path)]

    probe_timeout = max(1.0, min(20.0, timeout_seconds / 3))
    try:
        allowed_result = subprocess.run(
            [*base, *read_command(allowed)],
            cwd=workspace,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            shell=False,
            timeout=probe_timeout,
            env=_clean_environment(),
        )
        if allowed_result.returncode != 0 or token not in allowed_result.stdout:
            detail = (allowed_result.stderr or allowed_result.stdout)[-1200:]
            raise RuntimeError(
                "workspace-only sandbox failed its allowed-read probe: " + detail
            )
        denied_result = subprocess.run(
            [*base, *read_command(denied)],
            cwd=workspace,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            shell=False,
            timeout=probe_timeout,
            env=_clean_environment(),
        )
        observed = (denied_result.stdout or "") + (denied_result.stderr or "")
        if token in observed:
            raise RuntimeError("workspace-only sandbox leaked a parent-directory canary")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("workspace-only sandbox preflight timed out") from exc
    finally:
        allowed.unlink(missing_ok=True)
        denied.unlink(missing_ok=True)


def _base_argv(
    executable: Path,
    *,
    workspace: Path,
    model: str,
    sandbox: str,
    network_mode: str,
    tool_free: bool,
) -> list[str]:
    permission_profile = (
        _WORKSPACE_ONLY_PROFILE if sandbox == "workspace-write" else ":read-only"
    )
    argv = [
        str(executable),
        *(_workspace_only_profile_args() if sandbox == "workspace-write" else []),
        "-c",
        f'default_permissions="{permission_profile}"',
        "--ask-for-approval",
        "never",
        "-C",
        str(workspace),
        "--model",
        model,
    ]
    if network_mode in {"research-search", "source-review"}:
        argv.append("--search")
    disabled = [*_DISABLED_SHARED]
    if tool_free:
        disabled.extend(["shell_tool", "shell_snapshot"])
    for feature in disabled:
        argv.extend(["--disable", feature])
    argv.extend(
        [
            "-c",
            (
                'web_search="live"'
                if network_mode in {"research-search", "source-review"}
                else 'web_search="disabled"'
            ),
            "-c",
            'history.persistence="none"',
            "-c",
            "hide_agent_reasoning=true",
        ]
    )
    return argv


class CodexCLIModel:
    """One fresh, tool-free, read-only Codex context per structured review."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        executable: str | Path | None = None,
        timeout_seconds: int = 300,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.executable = discover_codex_cli(executable)
        self.timeout_seconds = timeout_seconds
        self.last_receipt: dict[str, Any] | None = None

    def complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: str,
        workspace: Path,
        network_mode: str = "offline-compute",
        trace_path: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        network_mode = _validate_network_mode(network_mode)
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise TimeoutError(f"no time remains for role={role}")
        timeout = self.timeout_seconds
        if timeout_seconds is not None:
            timeout = min(timeout, timeout_seconds)
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"modeling-{role}-") as directory:
            scratch = Path(directory)
            schema_path = scratch / "schema.json"
            output_path = scratch / "last-message.json"
            atomic_write_json(schema_path, schema)
            argv = _base_argv(
                self.executable,
                workspace=workspace,
                model=self.model,
                sandbox="read-only",
                network_mode=network_mode,
                tool_free=True,
            )
            argv.extend(
                [
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--json",
                    "--color",
                    "never",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
            )
            started = time.monotonic()
            try:
                result = subprocess.run(
                    argv,
                    cwd=workspace,
                    input=prompt,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    shell=False,
                    timeout=timeout,
                    env=_clean_environment(),
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Codex {role} step timed out") from exc
            if trace_path is not None:
                trace_path = trace_path.resolve()
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(result.stdout or "", encoding="utf-8", newline="\n")
            self.last_receipt = {
                "time": now(),
                "role": role,
                "model_requested": self.model,
                "returncode": result.returncode,
                "duration_seconds": round(time.monotonic() - started, 6),
                "network_mode": network_mode,
                "sandbox": "read-only",
                "sandbox_profile": ":read-only",
                "tool_free": True,
                "ephemeral": True,
                "trace": str(trace_path) if trace_path is not None else None,
                "trace_sha256": file_hash(trace_path) if trace_path is not None and trace_path.is_file() else None,
                "stderr_tail": (result.stderr or "")[-2000:],
                **_trace_counts(trace_path),
            }
            if result.returncode != 0:
                raise RuntimeError(f"Codex {role} failed: {(result.stderr or result.stdout)[-2000:]}")
            try:
                value = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Codex {role} returned invalid JSON") from exc
            _validate_schema(value, schema)
            return value


class NativeCodexResearcher:
    """One bounded Codex research session rooted only at its writable work tree."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        executable: str | Path | None = None,
    ):
        self.model = model
        self.executable = discover_codex_cli(executable)
        self.last_receipt: dict[str, Any] | None = None
        self._qualified_workspaces: set[Path] = set()

    def _ensure_workspace_sandbox(
        self, workspace: Path, *, timeout_seconds: float
    ) -> None:
        """Prove allowed workspace reads and denied parent reads before starting Lead."""

        if workspace in self._qualified_workspaces:
            return
        ensure_workspace_only_sandbox(
            self.executable, workspace, timeout_seconds=timeout_seconds
        )
        self._qualified_workspaces.add(workspace)

    def run(
        self,
        prompt: str,
        *,
        role: str,
        workspace: Path,
        trace_path: Path,
        timeout_seconds: float,
        network_mode: str = "offline-compute",
    ) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        network_mode = _validate_network_mode(network_mode)
        if network_mode not in {"research-search", "offline-compute"}:
            raise ValueError("researcher requires research-search or offline-compute")
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        trace_path = trace_path.resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        self._ensure_workspace_sandbox(workspace, timeout_seconds=timeout_seconds)
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"Codex {role} budget exhausted during sandbox preflight")
        with tempfile.TemporaryDirectory(prefix=f"modeling-{role}-") as directory:
            output_path = Path(directory) / "last-message.txt"
            argv = _base_argv(
                self.executable,
                workspace=workspace,
                model=self.model,
                sandbox="workspace-write",
                network_mode=network_mode,
                tool_free=False,
            )
            argv.extend(
                [
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--json",
                    "--color",
                    "never",
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
            )
            try:
                with trace_path.open("w", encoding="utf-8", newline="\n") as trace:
                    result = subprocess.run(
                        argv,
                        cwd=workspace,
                        input=prompt,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        stdout=trace,
                        stderr=subprocess.PIPE,
                        shell=False,
                        timeout=remaining,
                        env=_clean_environment(),
                    )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Codex {role} session timed out") from exc
            last_message = (
                output_path.read_text(encoding="utf-8", errors="replace")
                if output_path.is_file()
                else ""
            )
            receipt = {
                "time": now(),
                "role": role,
                "model_requested": self.model,
                "sandbox": "workspace-write",
                "sandbox_profile": _WORKSPACE_ONLY_PROFILE,
                "sandbox_implementation": "platform-native-permission-profile",
                "sandbox_preflight": "allowed-workspace-and-denied-parent",
                "network_mode": network_mode,
                "returncode": result.returncode,
                "duration_seconds": round(time.monotonic() - started, 6),
                "trace": str(trace_path),
                "trace_sha256": file_hash(trace_path),
                "stderr_tail": (result.stderr or "")[-2000:],
                "last_message_tail": last_message[-4000:],
                **_trace_counts(trace_path),
            }
            self.last_receipt = receipt
            if result.returncode != 0:
                raise RuntimeError(f"Codex {role} failed: {(result.stderr or last_message)[-2000:]}")
            return receipt
