from __future__ import annotations

import json
import os
import re
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ._file_lock import exclusive_file_lock
from .hashing import canonical_json, jsonable, sha256_value
from .schemas import ArtifactRef


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Content-addressed artifacts plus a tamper-evident append-only event log."""

    def __init__(self, output_root: str | Path, run_id: str | None = None) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run-{uuid4().hex[:12]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.run_id):
            raise ValueError("run_id contains unsafe path characters")
        self.run_directory = (self.output_root / self.run_id).resolve()
        if not self.run_directory.is_relative_to(self.output_root):
            raise ValueError("run directory escapes the configured output root")
        self.artifact_directory = self.run_directory / "artifacts"
        self.artifact_directory.mkdir(parents=True, exist_ok=False)
        self.event_path = self.run_directory / "events.jsonl"
        self._writer_lock_path = self.run_directory / ".writer.lock"
        self._sequence = 0
        self._previous_event_hash: str | None = None
        with exclusive_file_lock(self._writer_lock_path):
            self._refresh_tip_locked(require_nonempty=False)
            self._emit_locked("run_created", {"run_id": self.run_id})

    @classmethod
    def open_existing(cls, run_directory: str | Path) -> "RunStore":
        directory = Path(run_directory).resolve()
        instance = cls.__new__(cls)
        instance.output_root = directory.parent
        instance.run_id = directory.name
        instance.run_directory = directory
        instance.artifact_directory = directory / "artifacts"
        instance.event_path = directory / "events.jsonl"
        instance._writer_lock_path = directory / ".writer.lock"
        if not instance.artifact_directory.is_dir() or not instance.event_path.is_file():
            raise FileNotFoundError(f"not an FMA run directory: {directory}")
        try:
            with exclusive_file_lock(instance._writer_lock_path):
                sequence, event_hash = instance._validated_tip()
        except RuntimeError as exc:
            raise RuntimeError(
                "cannot reopen a run with an invalid event hash chain"
            ) from exc
        if sequence < 1 or event_hash is None:
            raise RuntimeError(
                "cannot reopen a run with an invalid event hash chain"
            )
        instance._sequence = sequence
        instance._previous_event_hash = event_hash
        return instance

    def put_artifact(self, kind: str, payload: object) -> ArtifactRef:
        envelope = {
            "artifact_schema": "1.0",
            "kind": kind,
            "payload": jsonable(payload),
        }
        digest = sha256_value(envelope)
        # Keep the full content hash while avoiding Windows MAX_PATH failures in
        # deeply nested experiment directories.
        filename = f"{digest}.json"
        path = self.artifact_directory / filename
        serialized = json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        with exclusive_file_lock(self._writer_lock_path):
            self._refresh_tip_locked(require_nonempty=True)
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if sha256_value(existing) != digest:
                    raise RuntimeError(
                        f"existing artifact failed integrity check: {path}"
                    )
            else:
                temporary = self.artifact_directory / f".tmp-{uuid4().hex[:12]}"
                try:
                    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                        handle.write(serialized + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            relative_path = path.relative_to(self.run_directory).as_posix()
            reference = ArtifactRef(
                kind=kind, sha256=digest, relative_path=relative_path
            )
            self._emit_locked(
                "artifact_committed", reference.model_dump(mode="json")
            )
        return reference

    def writer_transaction(self) -> AbstractContextManager[None]:
        """Serialize a multi-event authority transition with nested store writes."""

        return exclusive_file_lock(self._writer_lock_path)

    def load_artifact(self, reference: ArtifactRef) -> object:
        path = (self.run_directory / reference.relative_path).resolve()
        if not path.is_relative_to(self.artifact_directory):
            raise RuntimeError("artifact reference escapes this run's artifact directory")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        actual = sha256_value(envelope)
        if actual != reference.sha256:
            raise RuntimeError(
                f"artifact integrity failure for {reference.relative_path}: "
                f"expected {reference.sha256}, got {actual}"
            )
        if envelope.get("kind") != reference.kind:
            raise RuntimeError("artifact kind does not match its reference")
        return envelope["payload"]

    def emit(self, event_type: str, payload: object) -> str:
        with exclusive_file_lock(self._writer_lock_path):
            self._refresh_tip_locked(require_nonempty=True)
            return self._emit_locked(event_type, payload)

    def _emit_locked(self, event_type: str, payload: object) -> str:
        self._sequence += 1
        event = {
            "sequence": self._sequence,
            "timestamp": utc_now(),
            "event_type": event_type,
            "payload": jsonable(payload),
            "previous_event_hash": self._previous_event_hash,
        }
        event_hash = sha256_value(event)
        record = {**event, "event_hash": event_hash}
        with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._previous_event_hash = event_hash
        return event_hash

    def verify_event_chain(self) -> bool:
        try:
            with exclusive_file_lock(self._writer_lock_path):
                sequence, _ = self._validated_tip()
        except RuntimeError:
            return False
        return sequence > 0

    def _refresh_tip_locked(self, *, require_nonempty: bool) -> None:
        sequence, event_hash = self._validated_tip()
        if require_nonempty and (sequence < 1 or event_hash is None):
            raise RuntimeError("run event chain is empty")
        self._sequence = sequence
        self._previous_event_hash = event_hash

    def _validated_tip(self) -> tuple[int, str | None]:
        if not self.event_path.exists():
            return 0, None
        try:
            serialized = self.event_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError("cannot read run event chain") from exc
        if serialized and not serialized.endswith("\n"):
            raise RuntimeError("run event chain ends with a partial record")
        previous: str | None = None
        expected_sequence = 1
        try:
            for line in serialized.splitlines():
                if not line.strip():
                    raise RuntimeError("run event chain contains a blank record")
                record = json.loads(line)
                event_hash = record.pop("event_hash")
                if record["sequence"] != expected_sequence:
                    raise RuntimeError("run event sequence is not contiguous")
                if record["previous_event_hash"] != previous:
                    raise RuntimeError("run event predecessor hash does not match")
                if sha256_value(record) != event_hash:
                    raise RuntimeError("run event content hash does not match")
                previous = str(event_hash)
                expected_sequence += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("run event chain is malformed") from exc
        return expected_sequence - 1, previous
