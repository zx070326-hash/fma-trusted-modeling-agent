"""Small durable primitives and the authority-separated run layout."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


_APPEND_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_json(value).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("path must be a non-empty string")
    value = Path(relative.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"path escapes workspace: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / value).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"path escapes workspace: {relative}")
    return resolved


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{number}")
        records.append(value)
    return records


def tree_hashes(root: Path, *, excluded: Iterable[Path] = ()) -> dict[str, str]:
    root = root.resolve()
    excluded_roots = [item.resolve() for item in excluded]
    result: dict[str, str] = {}
    if not root.is_dir():
        return result
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        resolved = path.resolve()
        if any(resolved.is_relative_to(item) for item in excluded_roots):
            continue
        result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


@dataclass(frozen=True)
class RunLayout:
    """The researcher can write only ``work``; the harness owns ``control``."""

    root: Path
    control: Path
    work: Path
    sources: Path
    branches: Path
    verdicts: Path

    @classmethod
    def open(cls, root: Path) -> "RunLayout":
        resolved = root.resolve()
        control = resolved / ".modeling-agent"
        return cls(
            root=resolved,
            control=control,
            work=resolved / "work",
            sources=resolved / "sources",
            branches=resolved / "branches",
            verdicts=control / "verdicts",
        )

    @property
    def contract_path(self) -> Path:
        return self.control / "task-contract.json"

    @property
    def work_contract_path(self) -> Path:
        return self.work / "task-contract.json"

    @property
    def state_path(self) -> Path:
        return self.control / "run-state.json"

    @property
    def events_path(self) -> Path:
        return self.control / "events.jsonl"

    @property
    def evidence_path(self) -> Path:
        return self.control / "evidence.jsonl"

    @property
    def research_path(self) -> Path:
        return self.work / "research" / "records.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.work / "submission_manifest.json"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.control,
            self.work,
            self.sources,
            self.branches,
            self.verdicts,
            self.control / "runtime",
            self.control / "traces",
            self.work / "research",
        ):
            path.mkdir(parents=True, exist_ok=True)


@contextmanager
def run_lock(layout: RunLayout):
    """One non-blocking OS lock per run; this is not a workflow state plane."""

    layout.ensure()
    path = layout.control / "run.lock"
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("another modeling process already owns this run") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class RunStore:
    """Operational state and append-only events/evidence owned by the harness."""

    def __init__(self, layout: RunLayout):
        self.layout = layout
        self.layout.ensure()

    def exists(self) -> bool:
        return self.layout.state_path.is_file()

    def load(self) -> dict[str, Any]:
        if not self.exists():
            raise FileNotFoundError(f"run state not found: {self.layout.state_path}")
        value = json.loads(self.layout.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != 1:
            raise ValueError("run state schema must be 1")
        return value

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        atomic_write_json(self.layout.state_path, state)

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.layout.events_path,
            {"time": now(), "kind": kind, "payload": payload},
        )

    def admit(self, record: dict[str, Any]) -> dict[str, Any]:
        existing = self.evidence()
        admitted = {
            **record,
            "admitted_at": now(),
            "sequence": len(existing) + 1,
            "previous_record_hash": existing[-1]["record_hash"] if existing else None,
        }
        admitted["record_hash"] = content_hash(admitted)
        append_jsonl(self.layout.evidence_path, admitted)
        return admitted

    def evidence(self) -> list[dict[str, Any]]:
        records = read_jsonl(self.layout.evidence_path)
        previous: str | None = None
        for sequence, record in enumerate(records, 1):
            record_hash = record.get("record_hash")
            payload = {key: value for key, value in record.items() if key != "record_hash"}
            if (
                record.get("sequence") != sequence
                or record.get("previous_record_hash") != previous
                or not isinstance(record_hash, str)
                or content_hash(payload) != record_hash
            ):
                raise ValueError(f"evidence log integrity failure at sequence {sequence}")
            previous = record_hash
        return records
