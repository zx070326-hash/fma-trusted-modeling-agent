"""Narrow local tools and fail-closed mechanical evidence checks."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .core import file_hash, safe_path


WRITE_ROOTS = {"artifacts", "checks", "data", "notes", "paper", "results", "src"}
ROOT_WRITE_FILES = {"submission.json"}
SCRIPT_ROOTS = {"checks", "src"}
DENIED_IMPORT_ROOTS = {
    "ctypes",
    "ftplib",
    "http",
    "requests",
    "smtplib",
    "socket",
    "subprocess",
    "urllib",
}
DENIED_CALLS = {"__import__", "compile", "eval", "exec"}

CHECK_DESCRIPTIONS = [
    {
        "kind": "file_nonempty",
        "arguments": {"path": "workspace-relative artifact path"},
    },
    {
        "kind": "json_finite",
        "arguments": {"path": "workspace-relative JSON artifact path"},
    },
    {
        "kind": "numeric_assertion",
        "arguments": {
            "path": "workspace-relative JSON artifact path",
            "field": "required dotted JSON field, for example metrics.rmse",
            "operator": "one of <, <=, ==, !=, >=, >",
            "value": "finite number",
            "tolerance": "non-negative number; default 0",
        },
    },
    {
        "kind": "python_check",
        "arguments": {"script": "workspace-relative .py path below checks/"},
    },
]


def _result(
    status: str,
    summary: str,
    *,
    data: dict[str, Any] | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "data": data or {},
        "error_type": error_type,
    }


def _bounded(value: str, limit: int = 12_000) -> str:
    return value if len(value) <= limit else value[-limit:]


def _write_target(workspace: Path, relative: str) -> Path:
    path = safe_path(workspace, relative)
    parts = Path(relative.replace("\\", "/")).parts if relative else ()
    first = parts[0] if parts else ""
    root_delivery = len(parts) == 1 and first.casefold() in ROOT_WRITE_FILES
    if first not in WRITE_ROOTS and not root_delivery:
        raise ValueError(
            "writes are limited to project artifact directories and "
            f"{sorted(ROOT_WRITE_FILES)}; received {relative}"
        )
    return path


def _script_target(workspace: Path, relative: str) -> Path:
    path = safe_path(workspace, relative)
    parts = Path(relative.replace("\\", "/")).parts
    if not parts or parts[0] not in SCRIPT_ROOTS or path.suffix.casefold() != ".py":
        raise ValueError("Python scripts must be .py files below src/ or checks/")
    return path


def _audit_python_source(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in DENIED_IMPORT_ROOTS:
                    errors.append(f"denied import: {root}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in DENIED_IMPORT_ROOTS:
                errors.append(f"denied import: {root}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
                errors.append(f"denied call: {node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr in {"popen", "spawnl", "spawnv", "system"}
            ):
                errors.append(f"denied call: os.{node.func.attr}")
    return sorted(set(errors))


def _compute_environment(workspace: Path) -> dict[str, str]:
    allowed = {
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    environment["MODELING_AGENT_WORKSPACE"] = str(workspace)
    environment["PYTHONHASHSEED"] = "0"
    return environment


class ToolRegistry:
    """Only project-local read, write, and bounded Python computation."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    @property
    def descriptions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "read_text",
                "arguments": {"path": "str", "max_chars": "int <= 20000"},
                "effect": "read one workspace text file",
            },
            {
                "name": "read_files",
                "arguments": {
                    "paths": "1..12 workspace text paths",
                    "max_chars": "int <= 20000 per file",
                },
                "effect": "read a coherent file bundle; <=60000 chars total",
            },
            {
                "name": "write_text",
                "arguments": {"path": "str", "content": "str"},
                "effect": (
                    f"write below {sorted(WRITE_ROOTS)} or standard root "
                    f"deliveries {sorted(ROOT_WRITE_FILES)}"
                ),
            },
            {
                "name": "write_files",
                "arguments": {
                    "files": "1..12 objects with path and content; <=512 KiB total"
                },
                "effect": "validate, then write a small coherent file bundle",
            },
            {
                "name": "python_compute",
                "arguments": {
                    "script": "str under src/ or checks/",
                    "args": "list[str]",
                    "timeout": "1..120",
                    "expected_outputs": "list[str]",
                },
                "effect": "bounded local computation; network/process imports denied",
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "read_text":
                return self._read_text(arguments)
            if name == "read_files":
                return self._read_files(arguments)
            if name == "write_text":
                return self._write_text(arguments)
            if name == "write_files":
                return self._write_files(arguments)
            if name == "python_compute":
                return self._python_compute(arguments)
            return _result(
                "error", f"unknown tool: {name}", error_type="unknown_tool"
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return _result(
                "error",
                str(exc),
                error_type=type(exc).__name__,
            )

    def _read_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = arguments.get("path")
        limit = arguments.get("max_chars", 12_000)
        if not isinstance(limit, int) or not 1 <= limit <= 20_000:
            raise ValueError("max_chars must be an integer in 1..20000")
        path = safe_path(self.workspace, relative)
        if not path.is_file():
            return _result("error", f"file not found: {relative}", error_type="not_found")
        text = path.read_text(encoding="utf-8")
        return _result(
            "success",
            f"read {relative}",
            data={"path": relative, "content": text[:limit], "truncated": len(text) > limit},
        )

    def _read_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paths = arguments.get("paths")
        limit = arguments.get("max_chars", 12_000)
        if not isinstance(paths, list) or not 1 <= len(paths) <= 12:
            raise ValueError("read_files.paths must contain 1..12 paths")
        if not all(isinstance(item, str) for item in paths):
            raise ValueError("read_files.paths must be a string array")
        if len({item.casefold() for item in paths}) != len(paths):
            raise ValueError("read_files.paths contains duplicates")
        if not isinstance(limit, int) or not 1 <= limit <= 20_000:
            raise ValueError("max_chars must be an integer in 1..20000")

        prepared = []
        for relative in paths:
            path = safe_path(self.workspace, relative)
            if not path.is_file():
                return _result(
                    "error", f"file not found: {relative}", error_type="not_found"
                )
            prepared.append((relative, path))

        remaining = 60_000
        records = []
        for relative, path in prepared:
            text = path.read_text(encoding="utf-8")
            used = min(limit, remaining)
            content = text[:used]
            records.append(
                {
                    "path": relative,
                    "content": content,
                    "truncated": len(text) > used,
                }
            )
            remaining -= len(content)
        return _result(
            "success",
            f"read {len(records)} files",
            data={"files": records, "total_chars": 60_000 - remaining},
        )

    def _write_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("write_text.content must be a string")
        if len(content.encode("utf-8")) > 256 * 1024:
            raise ValueError("write_text content exceeds 256 KiB")
        path = _write_target(self.workspace, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return _result(
            "success",
            f"wrote {relative}",
            data={"path": relative, "sha256": file_hash(path), "bytes": path.stat().st_size},
        )

    def _write_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        files = arguments.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= 12:
            raise ValueError("write_files.files must contain 1..12 files")
        prepared = []
        total_bytes = 0
        seen = set()
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("each write_files item must be an object")
            relative = item.get("path")
            content = item.get("content")
            if not isinstance(content, str):
                raise ValueError("each write_files content must be a string")
            path = _write_target(self.workspace, relative)
            key = str(path).casefold()
            if key in seen:
                raise ValueError(f"duplicate write_files path: {relative}")
            seen.add(key)
            size = len(content.encode("utf-8"))
            total_bytes += size
            prepared.append((relative, path, content, size))
        if total_bytes > 512 * 1024:
            raise ValueError("write_files content exceeds 512 KiB total")
        records = []
        for relative, path, content, size in prepared:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            records.append(
                {
                    "path": relative,
                    "sha256": file_hash(path),
                    "bytes": size,
                }
            )
        return _result(
            "success",
            f"wrote {len(records)} files",
            data={"files": records, "total_bytes": total_bytes},
        )

    def _python_compute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = arguments.get("script")
        argv = arguments.get("args", [])
        timeout = arguments.get("timeout", 60)
        outputs = arguments.get("expected_outputs", [])
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError("python_compute.args must be a string array")
        if not isinstance(outputs, list) or not all(
            isinstance(item, str) for item in outputs
        ):
            raise ValueError("expected_outputs must be a string array")
        if not isinstance(timeout, int) or not 1 <= timeout <= 120:
            raise ValueError("python_compute.timeout must be in 1..120")
        script = _script_target(self.workspace, relative)
        if not script.is_file():
            raise ValueError(f"script not found: {relative}")
        output_paths = []
        for output in outputs:
            output_path = _write_target(self.workspace, output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_paths.append((output, output_path))
        audit_errors = _audit_python_source(script.read_text(encoding="utf-8"))
        if audit_errors:
            return _result(
                "denied",
                "Python policy denied the script",
                data={"findings": audit_errors},
                error_type="permission_denied",
            )
        started = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, "-I", str(script), *argv],
                cwd=self.workspace,
                env=_compute_environment(self.workspace),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                shell=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return _result(
                "error",
                f"Python computation timed out after {timeout}s",
                data={
                    "stdout": _bounded(str(exc.stdout or "")),
                    "stderr": _bounded(str(exc.stderr or "")),
                },
                error_type="timeout",
            )
        output_records = []
        missing = []
        for output, path in output_paths:
            exists = path.is_file()
            output_records.append(
                {
                    "path": output,
                    "exists": exists,
                    "sha256": file_hash(path) if exists else None,
                }
            )
            if not exists:
                missing.append(output)
        ok = process.returncode == 0 and not missing
        return _result(
            "success" if ok else "error",
            "Python computation completed" if ok else "Python computation failed",
            data={
                "script": relative,
                "script_sha256": file_hash(script),
                "returncode": process.returncode,
                "duration_seconds": round(time.monotonic() - started, 6),
                "stdout": _bounded(process.stdout or ""),
                "stderr": _bounded(process.stderr or ""),
                "outputs": output_records,
                "missing_outputs": missing,
            },
            error_type=None if ok else "compute_failed",
        )


def _all_finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def _json_field(value: Any, dotted: str) -> tuple[bool, Any]:
    if not isinstance(dotted, str) or not dotted:
        return False, None
    current = value
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def run_check(
    workspace: Path, kind: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Run one check. Unknown or malformed checks fail closed."""

    try:
        if kind == "file_nonempty":
            relative = arguments.get("path")
            path = safe_path(workspace, relative)
            ok = path.is_file() and path.stat().st_size > 0
            return {"kind": kind, "path": relative, "ok": ok}
        if kind in {"json_finite", "numeric_assertion"}:
            relative = arguments.get("path")
            path = safe_path(workspace, relative)
            data = json.loads(path.read_text(encoding="utf-8"))
            if kind == "json_finite":
                return {"kind": kind, "path": relative, "ok": _all_finite(data)}
            field = arguments.get("field")
            operator = arguments.get("operator", "==")
            target = float(arguments.get("value"))
            tolerance = float(arguments.get("tolerance", 0))
            exists, raw = _json_field(data, field)
            actual = float(raw) if exists else None
            finite = actual is not None and math.isfinite(actual)
            operations = (
                {
                    "<": actual < target + tolerance,
                    "<=": actual <= target + tolerance,
                    "==": abs(actual - target) <= tolerance,
                    "!=": abs(actual - target) > tolerance,
                    ">=": actual + tolerance >= target,
                    ">": actual + tolerance > target,
                }
                if finite
                else {}
            )
            return {
                "kind": kind,
                "path": relative,
                "field": field,
                "actual": actual,
                "operator": operator,
                "target": target,
                "tolerance": tolerance,
                "ok": exists and finite and operations.get(operator, False),
            }
        if kind == "python_check":
            relative = arguments.get("script")
            script = _script_target(workspace, relative)
            if not str(Path(relative).as_posix()).startswith("checks/"):
                raise ValueError("python_check must use a script below checks/")
            audit_errors = _audit_python_source(script.read_text(encoding="utf-8"))
            if audit_errors:
                return {"kind": kind, "script": relative, "ok": False, "errors": audit_errors}
            process = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=workspace,
                env=_compute_environment(workspace),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                shell=False,
                timeout=60,
            )
            return {
                "kind": kind,
                "script": relative,
                "ok": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": _bounded(process.stdout or "", 4000),
                "stderr": _bounded(process.stderr or "", 4000),
            }
        return {"kind": kind, "ok": False, "error": "unknown check kind"}
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        return {"kind": kind, "ok": False, "error": str(exc)}
