"""Provider adapter used by the thin modeling loop."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from collections import deque
from pathlib import Path
from typing import Any, Protocol

from .core import atomic_write_json, file_hash, now


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "upsert_nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "question": {"type": "string"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "priority": {"type": "number"},
                },
                "required": ["id", "question", "depends_on", "priority"],
            },
        },
        "focus_node": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "call_id": {"type": "string"},
                    "name": {
                        "type": "string",
                        "enum": [
                            "read_text",
                            "read_files",
                            "write_text",
                            "write_files",
                            "python_compute",
                        ],
                    },
                    "arguments_json": {"type": "string"},
                },
                "required": ["call_id", "name", "arguments_json"],
            },
        },
        "candidate_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "node_id": {"type": "string"},
                    "admission": {
                        "type": "string",
                        "enum": ["working", "claim"],
                    },
                    "statement": {"type": "string"},
                    "artifact": {"type": "string"},
                    "supporting_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                    },
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        "file_nonempty",
                                        "json_finite",
                                        "numeric_assertion",
                                        "python_check",
                                    ],
                                },
                                "arguments_json": {"type": "string"},
                            },
                            "required": ["kind", "arguments_json"],
                        },
                    },
                },
                "required": [
                    "id",
                    "node_id",
                    "admission",
                    "statement",
                    "artifact",
                    "supporting_artifacts",
                    "checks",
                ],
            },
        },
        "final": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "answer": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "limitations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["answer", "evidence_ids", "limitations"],
                },
            ]
        },
    },
    "required": [
        "summary",
        "upsert_nodes",
        "focus_node",
        "tool_calls",
        "candidate_claims",
        "final",
    ],
}


SYNTHESIS_ACTION_SCHEMA: dict[str, Any] = deepcopy(ACTION_SCHEMA)
SYNTHESIS_ACTION_SCHEMA["properties"]["final"] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "evidence_ids", "limitations"],
}


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["APPROVE", "REJECT"]},
        "findings": {"type": "array", "items": {"type": "string"}},
        "claim_strength": {
            "type": "string",
            "enum": ["exploratory", "locally_supported", "unsupported"],
        },
    },
    "required": ["verdict", "findings", "claim_strength"],
}


RAW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "assumptions", "limitations"],
}


class ModelAdapter(Protocol):
    def complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: str,
        workspace: Path,
    ) -> dict[str, Any]: ...


class NativeResearcherAdapter(Protocol):
    def run(
        self,
        prompt: str,
        *,
        role: str,
        workspace: Path,
        trace_path: Path,
        timeout_seconds: int,
    ) -> dict[str, Any]: ...


class ScriptedModel:
    """Deterministic adapter for loop and ablation tests."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        role: str,
        workspace: Path,
    ) -> dict[str, Any]:
        self.calls.append(
            {"prompt": prompt, "schema": schema, "role": role, "workspace": workspace}
        )
        if not self.responses:
            raise RuntimeError(f"no scripted response remains for role={role}")
        return self.responses.popleft()


def discover_codex_cli(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("MODELING_AGENT_CODEX_BIN"):
        candidates.append(Path(os.environ["MODELING_AGENT_CODEX_BIN"]).expanduser())
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    release_root = codex_home / "packages" / "standalone" / "releases"
    if release_root.is_dir():
        candidates.extend(
            sorted(
                release_root.glob("*/bin/codex.exe"),
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


class CodexCLIModel:
    """One fresh, tool-free Codex process per modeling or review step."""

    _DISABLED_FEATURES = (
        "shell_tool",
        "shell_snapshot",
        "apps",
        "plugins",
        "browser_use",
        "computer_use",
        "multi_agent",
        "goals",
        "memories",
        "hooks",
    )

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
    ) -> dict[str, Any]:
        workspace = workspace.resolve()
        runtime = workspace / ".modeling-agent" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime, prefix=f"{role}-") as directory:
            scratch = Path(directory)
            schema_path = scratch / "schema.json"
            output_path = scratch / "last-message.json"
            atomic_write_json(schema_path, schema)
            argv = [
                str(self.executable),
                "--sandbox",
                "read-only",
                "--ask-for-approval",
                "never",
                "-C",
                str(workspace),
                "--model",
                self.model,
            ]
            for feature in self._DISABLED_FEATURES:
                argv.extend(["--disable", feature])
            argv.extend(
                [
                    "-c",
                    'web_search="disabled"',
                    "-c",
                    'history.persistence="none"',
                    "-c",
                    "hide_agent_reasoning=true",
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
                    timeout=self.timeout_seconds,
                    env=_clean_environment(),
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Codex {role} step timed out") from exc
            duration = round(time.monotonic() - started, 6)
            self.last_receipt = {
                "time": now(),
                "role": role,
                "model_requested": self.model,
                "returncode": result.returncode,
                "duration_seconds": duration,
                "stderr_tail": (result.stderr or "")[-2000:],
            }
            if result.returncode != 0:
                raise RuntimeError(
                    f"Codex {role} failed: {(result.stderr or result.stdout)[-2000:]}"
                )
            try:
                value = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Codex {role} returned invalid JSON") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Codex {role} response must be an object")
            return value


def _native_trace_counts(path: Path) -> dict[str, int]:
    tool_ids: set[str] = set()
    messages = 0
    if not path.is_file():
        return {"observable_tool_calls": 0, "agent_messages": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
        if kind in {
            "command_execution",
            "file_change",
            "mcp_tool_call",
            "tool_call",
        }:
            identifier = item.get("id") or event.get("id")
            if isinstance(identifier, str):
                tool_ids.add(identifier)
    return {
        "observable_tool_calls": len(tool_ids),
        "agent_messages": messages,
    }


class NativeCodexResearcher:
    """One bounded native Codex research session with project-local tools."""

    _DISABLED_FEATURES = (
        "apps",
        "plugins",
        "browser_use",
        "computer_use",
        "multi_agent",
        "goals",
        "memories",
        "hooks",
    )

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        executable: str | Path | None = None,
    ):
        self.model = model
        self.executable = discover_codex_cli(executable)
        self.last_receipt: dict[str, Any] | None = None

    def run(
        self,
        prompt: str,
        *,
        role: str,
        workspace: Path,
        trace_path: Path,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        workspace = workspace.resolve()
        trace_path = trace_path.resolve()
        if not trace_path.is_relative_to(workspace):
            raise ValueError("native trace must stay inside the workspace")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        runtime = workspace / ".modeling-agent" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime, prefix=f"{role}-") as directory:
            output_path = Path(directory) / "last-message.txt"
            argv = [
                str(self.executable),
                "--sandbox",
                "workspace-write",
                "--ask-for-approval",
                "never",
                "-C",
                str(workspace),
                "--model",
                self.model,
            ]
            sandbox_implementation = "platform-default"
            if os.name == "nt":
                # Nested CLI runs cannot refresh the elevated Windows helper in
                # some desktop sessions. Keep the workspace boundary by using
                # Codex's documented restricted-token fallback.
                sandbox_implementation = "windows-unelevated"
                argv.extend(["-c", 'windows.sandbox="unelevated"'])
            for feature in self._DISABLED_FEATURES:
                argv.extend(["--disable", feature])
            argv.extend(
                [
                    "-c",
                    'web_search="disabled"',
                    "-c",
                    'history.persistence="none"',
                    "-c",
                    "hide_agent_reasoning=true",
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
            started = time.monotonic()
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
                        timeout=timeout_seconds,
                        env=_clean_environment(),
                    )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Codex {role} session timed out") from exc
            duration = round(time.monotonic() - started, 6)
            last_message = (
                output_path.read_text(encoding="utf-8", errors="replace")
                if output_path.is_file()
                else ""
            )
            counts = _native_trace_counts(trace_path)
            receipt = {
                "time": now(),
                "role": role,
                "model_requested": self.model,
                "sandbox": "workspace-write",
                "sandbox_implementation": sandbox_implementation,
                "returncode": result.returncode,
                "duration_seconds": duration,
                "trace": str(trace_path.relative_to(workspace).as_posix()),
                "trace_sha256": file_hash(trace_path),
                "stderr_tail": (result.stderr or "")[-2000:],
                "last_message_tail": last_message[-4000:],
                **counts,
            }
            self.last_receipt = receipt
            if result.returncode != 0:
                raise RuntimeError(
                    f"Codex {role} failed: {(result.stderr or last_message)[-2000:]}"
                )
            return receipt
