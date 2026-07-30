"""Thin operational control plane for the FMA Studio.

The operator plane owns resumable work coordination and immutable intake
publication.  It is deliberately not an authority plane: no row in this
database can open a stage gate, establish scientific evidence, grant
qualification, or authorize a real-world action.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from fma._file_lock import exclusive_file_lock
from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel


OPERATOR_SCHEMA_VERSION_V70 = "7.0"
OPERATOR_ROOT_NAME_V70 = ".fma-op-v70"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_WORKSPACE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$"
_INTAKE_PATTERN = r"^intake-[0-9a-f]{24}$"
_WORK_STATES = {
    "PENDING",
    "LEASED",
    "RECOVERY_PENDING",
    "SUBMITTED",
    "ACCEPTED",
    "REJECTED",
    "FAILED",
    "BLOCKED",
}
_ACTIVE_WORK_STATES = {"LEASED", "RECOVERY_PENDING"}
_MAX_ATTACHMENT_BYTES = 64 * 1024 * 1024
_MAX_INTAKE_BYTES = 256 * 1024 * 1024
_MAX_ATTACHMENTS = 128
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class OperatorPlaneError(RuntimeError):
    """Base error for fail-closed operational state transitions."""


class OperatorConflictError(OperatorPlaneError):
    """A semantic idempotency, ownership, or state conflict."""


class OperatorLeaseError(OperatorPlaneError):
    """A lease is missing, expired, or fenced by a newer attempt."""


class OperatorIntegrityError(OperatorPlaneError):
    """Persisted operational state failed deterministic verification."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def normalize_owned_path(path: str) -> str:
    """Return a portable, workspace-relative ownership path."""

    candidate = path.replace("\\", "/").strip()
    if candidate in {"", "."}:
        return "."
    pure = PurePosixPath(candidate)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"owned path is not workspace-relative: {path}")
    for part in pure.parts:
        if (
            not part
            or part.endswith((".", " "))
            or ":" in part
            or any(ord(character) < 32 for character in part)
            or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        ):
            raise ValueError(f"owned path contains an unsafe component: {path}")
    normalized = pure.as_posix().strip("/")
    if not normalized or normalized == ".":
        return "."
    if normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"owned path escapes workspace: {path}")
    return normalized.casefold()


def _validate_logical_name(name: str) -> None:
    if (
        name in {"", ".", ".."}
        or "/" in name
        or "\\" in name
        or name.endswith((".", " "))
        or ":" in name
        or any(ord(character) < 32 for character in name)
        or name.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(f"attachment name is unsafe: {name}")


def _validate_intake_id(intake_id: str) -> None:
    if re.fullmatch(_INTAKE_PATTERN, intake_id) is None:
        raise ValueError("intake id is unsafe")


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _assert_safe_descendant(root: Path, target: Path) -> None:
    """Reject existing symlink/junction components below a trusted root."""

    resolved_root = root.resolve(strict=True)
    try:
        relative = target.relative_to(resolved_root)
    except ValueError as exc:
        raise OperatorIntegrityError("path escapes its trusted root") from exc
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if not current.exists():
            continue
        if _is_link_like(current):
            raise OperatorIntegrityError(
                f"path contains a symbolic link or junction: {current}"
            )
        try:
            current.resolve(strict=True).relative_to(resolved_root)
        except ValueError as exc:
            raise OperatorIntegrityError(
                f"path resolves outside its trusted root: {current}"
            ) from exc


def owned_paths_overlap(left: str, right: str) -> bool:
    left_normalized = normalize_owned_path(left)
    right_normalized = normalize_owned_path(right)
    if "." in {left_normalized, right_normalized}:
        return True
    return (
        left_normalized == right_normalized
        or left_normalized.startswith(right_normalized + "/")
        or right_normalized.startswith(left_normalized + "/")
    )


def capture_file_manifest(root: str | Path) -> dict[str, str]:
    """Capture a deterministic file manifest below one task workspace."""

    resolved = Path(root).resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("manifest root must be a directory")
    manifest: dict[str, str] = {}
    for path in sorted(resolved.rglob("*")):
        if _is_link_like(path):
            raise OperatorIntegrityError(
                f"workspace manifest refuses symbolic link or junction: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        manifest[relative] = _sha256_bytes(path.read_bytes())
    return manifest


def changed_manifest_paths(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[str, ...]:
    paths = set(before) | set(after)
    return tuple(sorted(path for path in paths if before.get(path) != after.get(path)))


def assert_changed_paths_owned(
    changed_paths: Sequence[str],
    owned_paths: Sequence[str],
) -> None:
    normalized_owners = tuple(normalize_owned_path(path) for path in owned_paths)
    if changed_paths and not normalized_owners:
        raise OperatorConflictError("read-only packet changed workspace files")
    for changed in changed_paths:
        normalized_changed = normalize_owned_path(changed)
        if not any(
            owned_paths_overlap(normalized_changed, owner)
            and (
                owner == "."
                or normalized_changed == owner
                or normalized_changed.startswith(owner + "/")
            )
            for owner in normalized_owners
        ):
            raise OperatorConflictError(
                f"worker changed path outside declared ownership: {changed}"
            )


class OperatorAuthorityBindingV70(StrictModel):
    """Read-only binding to the code-owned FMA authority state."""

    schema_version: Literal["7.0-authority-binding"] = "7.0-authority-binding"
    workspace_id: str = Field(pattern=_WORKSPACE_PATTERN)
    graph_id: str = Field(min_length=1, max_length=160)
    workspace_spec_hash: str = Field(pattern=_SHA256_PATTERN)
    graph_snapshot_hash: str = Field(pattern=_SHA256_PATTERN)
    frontier_node_hashes: tuple[str, ...] = ()
    stage_statuses: dict[str, str] = Field(default_factory=dict)
    current_gate_hashes: dict[str, str] = Field(default_factory=dict)
    frontier_stages: tuple[str, ...] = ()
    operator_policy_hash: str = Field(pattern=_SHA256_PATTERN)
    claim_scope: Literal["workflow_control_only"] = "workflow_control_only"
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    binding_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_binding(self) -> "OperatorAuthorityBindingV70":
        expected = self.content_hash()
        if self.binding_hash is not None and self.binding_hash != expected:
            raise ValueError("operator authority binding hash differs")
        for gate_hash in self.current_gate_hashes.values():
            if len(gate_hash) != 64:
                raise ValueError("current gate hash is not SHA-256")
        if any(len(node_hash) != 64 for node_hash in self.frontier_node_hashes):
            raise ValueError("frontier node hash is not SHA-256")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"binding_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "OperatorAuthorityBindingV70":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"binding_hash"}),
            binding_hash=draft.content_hash(),
        )


class OperatorPacketV70(StrictModel):
    """Bounded, deterministic work projected from one authority snapshot."""

    schema_version: Literal["7.0-operator-packet"] = "7.0-operator-packet"
    workspace_id: str = Field(pattern=_WORKSPACE_PATTERN)
    action: str = Field(min_length=2, max_length=80)
    purpose: str = Field(min_length=3, max_length=500)
    authority_binding: OperatorAuthorityBindingV70
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    allowed_tool_profile: str = Field(min_length=2, max_length=80)
    expected_outputs: tuple[str, ...] = ()
    max_attempts: int = Field(default=3, ge=1, le=10)
    lease_seconds: int = Field(default=300, ge=30, le=3600)
    max_wall_seconds: int = Field(default=1800, ge=30, le=21600)
    idempotency_key: str = Field(min_length=16, max_length=200)
    claim_scope: Literal["workflow_control_only"] = "workflow_control_only"
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    packet_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_packet(self) -> "OperatorPacketV70":
        if self.workspace_id != self.authority_binding.workspace_id:
            raise ValueError("packet workspace differs from authority binding")
        normalized_reads = tuple(normalize_owned_path(path) for path in self.read_paths)
        normalized_writes = tuple(
            normalize_owned_path(path) for path in self.write_paths
        )
        if len(normalized_reads) != len(set(normalized_reads)):
            raise ValueError("packet read paths contain duplicates")
        if len(normalized_writes) != len(set(normalized_writes)):
            raise ValueError("packet write paths contain duplicates")
        expected = self.content_hash()
        if self.packet_hash is not None and self.packet_hash != expected:
            raise ValueError("operator packet hash differs")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"packet_hash"}))

    @classmethod
    def seal(cls, **data: object) -> "OperatorPacketV70":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"packet_hash"}),
            packet_hash=draft.content_hash(),
        )


class OperatorLeaseV70(StrictModel):
    schema_version: Literal["7.0-operator-lease"] = "7.0-operator-lease"
    work_id: str
    workspace_id: str
    worker_id: str
    attempt_epoch: int = Field(ge=1)
    fencing_token: int = Field(ge=1)
    lease_until_epoch: float
    packet_hash: str = Field(pattern=_SHA256_PATTERN)


class OperatorSubmissionV70(StrictModel):
    """Worker submission; never a scientific or stage-gate decision."""

    schema_version: Literal["7.0-operator-submission"] = "7.0-operator-submission"
    work_id: str
    packet_hash: str = Field(pattern=_SHA256_PATTERN)
    input_binding_hash: str = Field(pattern=_SHA256_PATTERN)
    output_binding: OperatorAuthorityBindingV70
    before_manifest_hash: str = Field(pattern=_SHA256_PATTERN)
    after_manifest_hash: str = Field(pattern=_SHA256_PATTERN)
    changed_paths: tuple[str, ...]
    result_summary: dict[str, Any] = Field(default_factory=dict)
    submitted_at: str
    claim_scope: Literal["workflow_control_only"] = "workflow_control_only"
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    submission_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_submission(self) -> "OperatorSubmissionV70":
        expected = self.content_hash()
        if self.submission_hash is not None and self.submission_hash != expected:
            raise ValueError("operator submission hash differs")
        return self

    def content_hash(self) -> str:
        return sha256_value(
            self.model_dump(mode="json", exclude={"submission_hash"})
        )

    @classmethod
    def seal(cls, **data: object) -> "OperatorSubmissionV70":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"submission_hash"}),
            submission_hash=draft.content_hash(),
        )


class IntakeAttachmentV70(StrictModel):
    logical_name: str = Field(min_length=1, max_length=160)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=160)
    blob_ref: str = Field(min_length=8, max_length=100)

    @model_validator(mode="after")
    def validate_attachment(self) -> "IntakeAttachmentV70":
        _validate_logical_name(self.logical_name)
        if self.size_bytes > _MAX_ATTACHMENT_BYTES:
            raise ValueError("intake attachment exceeds size policy")
        if self.blob_ref != f"sha256/{self.sha256}":
            raise ValueError("intake blob reference differs from content hash")
        return self


class IntakeManifestV70(StrictModel):
    schema_version: Literal["7.0-intake-manifest"] = "7.0-intake-manifest"
    intake_id: str = Field(pattern=_INTAKE_PATTERN)
    idempotency_key: str = Field(min_length=8, max_length=200)
    objective: str = Field(min_length=12, max_length=4000)
    objective_hash: str = Field(pattern=_SHA256_PATTERN)
    requested_workspace_id: str | None = Field(
        default=None, pattern=_WORKSPACE_PATTERN
    )
    evidence_scope: Literal["development", "public_data"]
    workflow_mode: Literal["legacy", "v67"]
    attachments: tuple[IntakeAttachmentV70, ...]
    total_bytes: int = Field(ge=0)
    source_trust: Literal["user_supplied_untrusted"] = "user_supplied_untrusted"
    received_at: str
    request_hash: str = Field(pattern=_SHA256_PATTERN)
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    manifest_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_manifest(self) -> "IntakeManifestV70":
        _validate_intake_id(self.intake_id)
        if self.objective_hash != _sha256_bytes(self.objective.encode("utf-8")):
            raise ValueError("intake objective hash differs")
        if len(self.attachments) > _MAX_ATTACHMENTS:
            raise ValueError("intake has too many attachments")
        if self.total_bytes != sum(item.size_bytes for item in self.attachments):
            raise ValueError("intake total byte count differs")
        if self.total_bytes > _MAX_INTAKE_BYTES:
            raise ValueError("intake exceeds aggregate size policy")
        names = [item.logical_name.casefold() for item in self.attachments]
        if len(names) != len(set(names)):
            raise ValueError("intake attachment names are not unique")
        expected_request_hash = sha256_value(
            {
                "objective": self.objective,
                "objective_hash": self.objective_hash,
                "requested_workspace_id": self.requested_workspace_id,
                "evidence_scope": self.evidence_scope,
                "workflow_mode": self.workflow_mode,
                "attachments": [
                    item.model_dump(mode="json") for item in self.attachments
                ],
                "total_bytes": self.total_bytes,
                "source_trust": self.source_trust,
            }
        )
        if self.request_hash != expected_request_hash:
            raise ValueError("intake request hash differs")
        expected = self.content_hash()
        if self.manifest_hash is not None and self.manifest_hash != expected:
            raise ValueError("intake manifest hash differs")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"manifest_hash"}))

    @classmethod
    def seal(cls, **data: object) -> "IntakeManifestV70":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


class OperatorStoreV70:
    """SQLite WAL task ledger plus immutable content-addressed intake store."""

    def __init__(self, task_root: str | Path) -> None:
        self.task_root = Path(task_root).resolve()
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.root = self.task_root / OPERATOR_ROOT_NAME_V70
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "operator.sqlite3"
        self.intakes_root = self.root / "intakes"
        self.blobs_root = self.root / "blobs"
        self.staging_root = self.root / "staging"
        for directory in (self.intakes_root, self.blobs_root, self.staging_root):
            directory.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA trusted_schema=OFF")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS work_items(
                    work_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    workspace_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    packet_hash TEXT NOT NULL,
                    packet_json TEXT NOT NULL,
                    write_paths_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    attempt_epoch INTEGER NOT NULL DEFAULT 0,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    lease_until_epoch REAL,
                    submission_json TEXT,
                    authority_projection_json TEXT,
                    last_error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_operator_work_status
                    ON work_items(status, workspace_id);
                CREATE TABLE IF NOT EXISTS operator_events(
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    work_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT UNIQUE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS intakes(
                    intake_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    request_hash TEXT NOT NULL,
                    manifest_hash TEXT,
                    manifest_json TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_error_json TEXT
                );
                CREATE TABLE IF NOT EXISTS intake_bindings(
                    intake_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    binding_hash TEXT NOT NULL,
                    bound_at TEXT NOT NULL,
                    PRIMARY KEY(intake_id, workspace_id),
                    FOREIGN KEY(intake_id) REFERENCES intakes(intake_id)
                );
                """
            )
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('schema_version',?)",
                    (OPERATOR_SCHEMA_VERSION_V70,),
                )
            elif existing["value"] != OPERATOR_SCHEMA_VERSION_V70:
                raise OperatorIntegrityError(
                    "operator database schema version is unsupported"
                )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        *,
        event_type: str,
        payload: dict[str, Any],
        work_id: str | None = None,
    ) -> str:
        previous = connection.execute(
            "SELECT event_hash FROM operator_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        recorded_at = _utc_now()
        event_payload = {
            "work_id": work_id,
            "event_type": event_type,
            "payload": payload,
            "recorded_at": recorded_at,
            "previous_event_hash": previous["event_hash"] if previous else None,
        }
        event_hash = sha256_value(event_payload)
        connection.execute(
            """
            INSERT INTO operator_events(
                work_id,event_type,payload_json,recorded_at,
                previous_event_hash,event_hash
            ) VALUES(?,?,?,?,?,?)
            """,
            (
                work_id,
                event_type,
                canonical_json(payload),
                recorded_at,
                event_payload["previous_event_hash"],
                event_hash,
            ),
        )
        return event_hash

    @staticmethod
    def _work_row(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["packet"] = json.loads(payload.pop("packet_json"))
        payload["write_paths"] = json.loads(payload.pop("write_paths_json"))
        for field in (
            "submission_json",
            "authority_projection_json",
            "last_error_json",
        ):
            value = payload.pop(field)
            payload[field.removesuffix("_json")] = (
                json.loads(value) if value is not None else None
            )
        return payload

    def ensure_work(self, packet: OperatorPacketV70) -> dict[str, Any]:
        packet = OperatorPacketV70.model_validate(packet.model_dump(mode="json"))
        if (
            packet.packet_hash is None
            or packet.authority_binding.binding_hash is None
        ):
            raise ValueError("operator packet must be sealed")
        work_id = f"work-{packet.packet_hash[:24]}"
        now = _utc_now()
        packet_json = canonical_json(packet.model_dump(mode="json"))
        write_paths = tuple(normalize_owned_path(path) for path in packet.write_paths)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM work_items WHERE idempotency_key=?",
                (packet.idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["packet_hash"] != packet.packet_hash:
                    connection.execute("ROLLBACK")
                    raise OperatorConflictError(
                        "idempotency key was reused for a different operator packet"
                    )
                connection.execute("COMMIT")
                return self._work_row(existing)
            connection.execute(
                """
                INSERT INTO work_items(
                    work_id,idempotency_key,workspace_id,action,packet_hash,
                    packet_json,write_paths_json,status,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,'PENDING',?,?)
                """,
                (
                    work_id,
                    packet.idempotency_key,
                    packet.workspace_id,
                    packet.action,
                    packet.packet_hash,
                    packet_json,
                    canonical_json(write_paths),
                    now,
                    now,
                ),
            )
            self._event(
                connection,
                event_type="work.created",
                work_id=work_id,
                payload={
                    "workspace_id": packet.workspace_id,
                    "action": packet.action,
                    "packet_hash": packet.packet_hash,
                    "claim_scope": "workflow_control_only",
                },
            )
            connection.execute("COMMIT")
        return self.get_work(work_id)

    def get_work(self, work_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM work_items WHERE work_id=?", (work_id,)
            ).fetchone()
        if row is None:
            raise OperatorPlaneError(f"operator work item not found: {work_id}")
        return self._work_row(row)

    def list_work(self, workspace_id: str | None = None) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE (? IS NULL OR workspace_id=?)
                ORDER BY created_at, work_id
                """,
                (workspace_id, workspace_id),
            ).fetchall()
        return [self._work_row(row) for row in rows]

    def _live_conflicts(
        self,
        connection: sqlite3.Connection,
        *,
        work_id: str,
        workspace_id: str,
        write_paths: Sequence[str],
        now_epoch: float,
    ) -> list[str]:
        rows = connection.execute(
            """
            SELECT work_id,write_paths_json FROM work_items
            WHERE workspace_id=? AND work_id<>?
              AND (
                status='RECOVERY_PENDING'
                OR (
                  status='LEASED' AND lease_until_epoch IS NOT NULL
                  AND lease_until_epoch>=?
                )
              )
            """,
            (workspace_id, work_id, now_epoch),
        ).fetchall()
        conflicts: list[str] = []
        for row in rows:
            other_paths = json.loads(row["write_paths_json"])
            if any(
                owned_paths_overlap(left, right)
                for left in write_paths
                for right in other_paths
            ):
                conflicts.append(row["work_id"])
        return conflicts

    def claim(
        self,
        work_id: str,
        *,
        worker_id: str,
        lease_seconds: int | None = None,
    ) -> OperatorLeaseV70:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        now_epoch = time.time()
        now = _utc_now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_items WHERE work_id=?", (work_id,)
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise OperatorPlaneError(f"operator work item not found: {work_id}")
            packet = OperatorPacketV70.model_validate_json(row["packet_json"])
            if row["status"] not in {"PENDING", "FAILED"}:
                connection.execute("ROLLBACK")
                raise OperatorConflictError(
                    f"work item cannot be claimed from state {row['status']}"
                )
            if row["attempt_epoch"] >= packet.max_attempts:
                connection.execute(
                    """
                    UPDATE work_items SET status='BLOCKED',updated_at=?
                    WHERE work_id=?
                    """,
                    (now, work_id),
                )
                self._event(
                    connection,
                    event_type="work.blocked",
                    work_id=work_id,
                    payload={"reason": "attempt_budget_exhausted"},
                )
                connection.execute("COMMIT")
                raise OperatorConflictError("operator attempt budget is exhausted")
            write_paths = json.loads(row["write_paths_json"])
            conflicts = self._live_conflicts(
                connection,
                work_id=work_id,
                workspace_id=row["workspace_id"],
                write_paths=write_paths,
                now_epoch=now_epoch,
            )
            if conflicts:
                connection.execute("ROLLBACK")
                raise OperatorConflictError(
                    f"write ownership conflicts with active work: {conflicts}"
                )
            duration = lease_seconds or packet.lease_seconds
            duration = max(30, min(duration, 3600))
            attempt_epoch = int(row["attempt_epoch"]) + 1
            fencing_token = int(row["fencing_token"]) + 1
            lease_until = now_epoch + duration
            connection.execute(
                """
                UPDATE work_items SET
                    status='LEASED',worker_id=?,attempt_epoch=?,
                    fencing_token=?,lease_until_epoch=?,
                    submission_json=NULL,authority_projection_json=NULL,
                    last_error_json=NULL,updated_at=?
                WHERE work_id=?
                """,
                (
                    worker_id,
                    attempt_epoch,
                    fencing_token,
                    lease_until,
                    now,
                    work_id,
                ),
            )
            self._event(
                connection,
                event_type="work.claimed",
                work_id=work_id,
                payload={
                    "worker_id": worker_id,
                    "attempt_epoch": attempt_epoch,
                    "fencing_token": fencing_token,
                    "lease_until_epoch": lease_until,
                },
            )
            connection.execute("COMMIT")
        return OperatorLeaseV70(
            work_id=work_id,
            workspace_id=packet.workspace_id,
            worker_id=worker_id,
            attempt_epoch=attempt_epoch,
            fencing_token=fencing_token,
            lease_until_epoch=lease_until,
            packet_hash=packet.packet_hash,
        )

    @staticmethod
    def _assert_live_lease(
        row: sqlite3.Row,
        *,
        lease: OperatorLeaseV70,
        now_epoch: float,
    ) -> None:
        if (
            row["status"] != "LEASED"
            or row["worker_id"] != lease.worker_id
            or row["attempt_epoch"] != lease.attempt_epoch
            or row["fencing_token"] != lease.fencing_token
        ):
            raise OperatorLeaseError("operator lease was fenced or is not active")
        if (
            row["lease_until_epoch"] is None
            or float(row["lease_until_epoch"]) < now_epoch
        ):
            raise OperatorLeaseError("operator lease expired")

    def heartbeat(
        self,
        lease: OperatorLeaseV70,
        *,
        lease_seconds: int = 300,
    ) -> OperatorLeaseV70:
        now_epoch = time.time()
        lease_until = now_epoch + max(30, min(lease_seconds, 3600))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_items WHERE work_id=?", (lease.work_id,)
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise OperatorLeaseError("operator work item disappeared")
            try:
                self._assert_live_lease(row, lease=lease, now_epoch=now_epoch)
            except Exception:
                connection.execute("ROLLBACK")
                raise
            connection.execute(
                """
                UPDATE work_items SET lease_until_epoch=?,updated_at=?
                WHERE work_id=?
                """,
                (lease_until, _utc_now(), lease.work_id),
            )
            self._event(
                connection,
                event_type="work.heartbeat",
                work_id=lease.work_id,
                payload={
                    "attempt_epoch": lease.attempt_epoch,
                    "fencing_token": lease.fencing_token,
                    "lease_until_epoch": lease_until,
                },
            )
            connection.execute("COMMIT")
        return lease.model_copy(update={"lease_until_epoch": lease_until})

    def submit(
        self,
        lease: OperatorLeaseV70,
        submission: OperatorSubmissionV70,
    ) -> dict[str, Any]:
        now_epoch = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_items WHERE work_id=?", (lease.work_id,)
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise OperatorLeaseError("operator work item disappeared")
            try:
                self._assert_live_lease(row, lease=lease, now_epoch=now_epoch)
            except Exception:
                connection.execute("ROLLBACK")
                raise
            packet = OperatorPacketV70.model_validate_json(row["packet_json"])
            if (
                submission.submission_hash is None
                or submission.output_binding.binding_hash is None
                or
                submission.work_id != lease.work_id
                or submission.packet_hash != packet.packet_hash
                or submission.input_binding_hash
                != packet.authority_binding.binding_hash
            ):
                connection.execute("ROLLBACK")
                raise OperatorConflictError(
                    "submission is not bound to the claimed operator packet"
                )
            assert_changed_paths_owned(
                submission.changed_paths,
                packet.write_paths,
            )
            connection.execute(
                """
                UPDATE work_items SET
                    status='SUBMITTED',submission_json=?,
                    lease_until_epoch=NULL,updated_at=?
                WHERE work_id=?
                """,
                (
                    canonical_json(submission.model_dump(mode="json")),
                    _utc_now(),
                    lease.work_id,
                ),
            )
            self._event(
                connection,
                event_type="work.submitted",
                work_id=lease.work_id,
                payload={
                    "submission_hash": submission.submission_hash,
                    "output_binding_hash": submission.output_binding.binding_hash,
                    "changed_path_count": len(submission.changed_paths),
                    "claim_scope": "workflow_control_only",
                },
            )
            connection.execute("COMMIT")
        return self.get_work(lease.work_id)

    def project_authority_decision(
        self,
        work_id: str,
        *,
        accepted: bool,
        authority_receipt_hash: str,
        reason_codes: Sequence[str] = (),
    ) -> dict[str, Any]:
        if len(authority_receipt_hash) != 64:
            raise ValueError("authority receipt hash must be SHA-256")
        projection = {
            "schema_version": "7.0-authority-projection",
            "status": "ACCEPTED" if accepted else "REJECTED",
            "authority_receipt_hash": authority_receipt_hash,
            "reason_codes": list(reason_codes),
            "claim_scope": "workflow_control_only",
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM work_items WHERE work_id=?", (work_id,)
            ).fetchone()
            if row is None or row["status"] != "SUBMITTED":
                connection.execute("ROLLBACK")
                raise OperatorConflictError(
                    "authority projection requires a submitted work item"
                )
            status = projection["status"]
            connection.execute(
                """
                UPDATE work_items SET status=?,authority_projection_json=?,
                    updated_at=? WHERE work_id=?
                """,
                (status, canonical_json(projection), _utc_now(), work_id),
            )
            self._event(
                connection,
                event_type=f"work.{status.lower()}",
                work_id=work_id,
                payload=projection,
            )
            connection.execute("COMMIT")
        return self.get_work(work_id)

    def reconcile_authority_effect(
        self,
        work_id: str,
        *,
        output_binding: OperatorAuthorityBindingV70,
        authority_receipt_hash: str,
        reason_code: str,
        accepted: bool,
    ) -> dict[str, Any]:
        """Project an already committed authority effect without rerunning work.

        This transition repairs only the operational ledger.  The caller must
        have independently verified the authority workspace; the operator
        database is never consulted by a gate.
        """

        if len(authority_receipt_hash) != 64:
            raise ValueError("authority receipt hash must be SHA-256")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_items WHERE work_id=?", (work_id,)
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise OperatorPlaneError(f"operator work item not found: {work_id}")
            if row["status"] != "SUBMITTED" or row["submission_json"] is None:
                connection.execute("ROLLBACK")
                raise OperatorConflictError(
                    "authority reconciliation requires an exact worker submission"
                )
            packet = OperatorPacketV70.model_validate_json(row["packet_json"])
            submission = OperatorSubmissionV70.model_validate_json(
                row["submission_json"]
            )
            if (
                output_binding.workspace_id != packet.workspace_id
                or submission.packet_hash != packet.packet_hash
                or submission.output_binding.binding_hash
                != output_binding.binding_hash
            ):
                connection.execute("ROLLBACK")
                raise OperatorConflictError(
                    "reconciled authority differs from the exact worker submission"
                )
            status = "ACCEPTED" if accepted else "REJECTED"
            projection = {
                "schema_version": "7.0-authority-reconciliation",
                "status": status,
                "authority_receipt_hash": authority_receipt_hash,
                "input_binding_hash": packet.authority_binding.binding_hash,
                "output_binding_hash": output_binding.binding_hash,
                "reason_codes": [reason_code],
                "worker_submission_recovered": True,
                "claim_scope": "workflow_control_only",
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
            connection.execute(
                """
                UPDATE work_items SET status=?,
                    authority_projection_json=?,lease_until_epoch=NULL,
                    updated_at=? WHERE work_id=?
                """,
                (status, canonical_json(projection), _utc_now(), work_id),
            )
            self._event(
                connection,
                event_type="work.reconciled_from_authority",
                work_id=work_id,
                payload=projection,
            )
            connection.execute("COMMIT")
        return self.get_work(work_id)

    def fail(
        self,
        lease: OperatorLeaseV70,
        *,
        error_type: str,
        message: str,
    ) -> dict[str, Any]:
        error = {
            "error_type": error_type[:120],
            "message": message[:1000],
            "attempt_epoch": lease.attempt_epoch,
            "fencing_token": lease.fencing_token,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        now_epoch = time.time()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM work_items WHERE work_id=?", (lease.work_id,)
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise OperatorLeaseError("operator work item disappeared")
            try:
                self._assert_live_lease(row, lease=lease, now_epoch=now_epoch)
            except Exception:
                connection.execute("ROLLBACK")
                raise
            packet = OperatorPacketV70.model_validate_json(row["packet_json"])
            status = (
                "BLOCKED"
                if int(row["attempt_epoch"]) >= packet.max_attempts
                else "FAILED"
            )
            connection.execute(
                """
                UPDATE work_items SET status=?,lease_until_epoch=NULL,
                    last_error_json=?,updated_at=? WHERE work_id=?
                """,
                (
                    status,
                    canonical_json(error),
                    _utc_now(),
                    lease.work_id,
                ),
            )
            self._event(
                connection,
                event_type=f"work.{status.lower()}",
                work_id=lease.work_id,
                payload=error,
            )
            connection.execute("COMMIT")
        return self.get_work(lease.work_id)

    def reconcile_expired(self, *, now_epoch: float | None = None) -> list[str]:
        threshold = time.time() if now_epoch is None else now_epoch
        recovered: list[str] = []
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE status='LEASED' AND lease_until_epoch IS NOT NULL
                  AND lease_until_epoch<?
                ORDER BY work_id
                """,
                (threshold,),
            ).fetchall()
            for row in rows:
                status = "RECOVERY_PENDING"
                connection.execute(
                    """
                    UPDATE work_items SET status=?,worker_id=NULL,
                        lease_until_epoch=NULL,updated_at=?
                    WHERE work_id=?
                    """,
                    (status, _utc_now(), row["work_id"]),
                )
                self._event(
                    connection,
                    event_type="work.lease_expired",
                    work_id=row["work_id"],
                    payload={
                        "attempt_epoch": row["attempt_epoch"],
                        "fencing_token": row["fencing_token"],
                        "next_status": status,
                    },
                )
                recovered.append(row["work_id"])
            connection.execute("COMMIT")
        return recovered

    def has_live_lease(self, workspace_id: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM work_items
                WHERE workspace_id=? AND status='LEASED'
                  AND lease_until_epoch IS NOT NULL AND lease_until_epoch>=?
                LIMIT 1
                """,
                (workspace_id, time.time()),
            ).fetchone()
        return row is not None

    def operational_summary(self, workspace_id: str) -> dict[str, Any]:
        items = self.list_work(workspace_id)
        counts = {state: 0 for state in sorted(_WORK_STATES)}
        for item in items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        current = items[-1] if items else None
        return {
            "schema_version": "7.0-operator-summary",
            "workspace_id": workspace_id,
            "counts": counts,
            "live_lease": self.has_live_lease(workspace_id),
            "latest_work": (
                {
                    "work_id": current["work_id"],
                    "action": current["action"],
                    "status": current["status"],
                    "attempt_epoch": current["attempt_epoch"],
                    "packet_hash": current["packet_hash"],
                }
                if current
                else None
            ),
            "claim_scope": "workflow_control_only",
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }

    def _intake_request(
        self,
        *,
        objective: str,
        attachment_paths: Sequence[str | Path],
        requested_workspace_id: str | None,
        evidence_scope: Literal["development", "public_data"],
        workflow_mode: Literal["legacy", "v67"],
    ) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
        objective = objective.strip()
        if len(objective) < 12 or len(objective) > 4000:
            raise ValueError("intake objective must contain 12 to 4000 characters")
        if requested_workspace_id is not None:
            if re.fullmatch(_WORKSPACE_PATTERN, requested_workspace_id) is None:
                raise ValueError("requested workspace id is unsafe")
        if len(attachment_paths) > _MAX_ATTACHMENTS:
            raise ValueError("intake has too many attachments")
        attachments: list[tuple[Path, dict[str, Any]]] = []
        names: set[str] = set()
        total = 0
        for source in attachment_paths:
            unresolved = Path(source).expanduser()
            if _is_link_like(unresolved):
                raise ValueError(
                    "attachment cannot be a symbolic link or junction: "
                    f"{unresolved}"
                )
            path = unresolved.resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"attachment is not a regular file: {path}")
            name = path.name
            _validate_logical_name(name)
            if name.casefold() in names:
                raise ValueError(f"attachment name is duplicated: {name}")
            names.add(name.casefold())
            size = path.stat().st_size
            if size > _MAX_ATTACHMENT_BYTES:
                raise ValueError(f"attachment exceeds size policy: {name}")
            total += size
            if total > _MAX_INTAKE_BYTES:
                raise ValueError("intake exceeds aggregate size policy")
            payload = path.read_bytes()
            if len(payload) != size:
                raise OperatorIntegrityError(
                    f"attachment changed while being read: {name}"
                )
            digest = _sha256_bytes(payload)
            media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            metadata = {
                "logical_name": name,
                "sha256": digest,
                "size_bytes": size,
                "media_type": media_type,
                "blob_ref": f"sha256/{digest}",
            }
            attachments.append((path, metadata))
        request = {
            "objective": objective,
            "objective_hash": _sha256_bytes(objective.encode("utf-8")),
            "requested_workspace_id": requested_workspace_id,
            "evidence_scope": evidence_scope,
            "workflow_mode": workflow_mode,
            "attachments": [metadata for _, metadata in attachments],
            "total_bytes": total,
            "source_trust": "user_supplied_untrusted",
        }
        request["request_hash"] = sha256_value(request)
        return request, attachments

    def _write_current_intake_projection(
        self,
        manifest: IntakeManifestV70,
    ) -> None:
        _atomic_write_bytes(
            self.root / "current_intake.json",
            _json_bytes(
                {
                    "schema_version": "7.0-current-intake-projection",
                    "intake_id": manifest.intake_id,
                    "manifest_hash": manifest.manifest_hash,
                    "claim_scope": "workflow_control_only",
                }
            ),
        )

    def publish_intake(
        self,
        *,
        idempotency_key: str,
        objective: str,
        attachment_paths: Sequence[str | Path] = (),
        requested_workspace_id: str | None = None,
        evidence_scope: Literal["development", "public_data"] = "development",
        workflow_mode: Literal["legacy", "v67"] = "legacy",
    ) -> IntakeManifestV70:
        if len(idempotency_key) < 8 or len(idempotency_key) > 200:
            raise ValueError("intake idempotency key must contain 8 to 200 characters")
        request, attachments = self._intake_request(
            objective=objective,
            attachment_paths=attachment_paths,
            requested_workspace_id=requested_workspace_id,
            evidence_scope=evidence_scope,
            workflow_mode=workflow_mode,
        )
        request_hash = request["request_hash"]
        intake_id = f"intake-{sha256_value({'key': idempotency_key, 'request': request_hash})[:24]}"
        created_at = _utc_now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM intakes WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    connection.execute("ROLLBACK")
                    raise OperatorConflictError(
                        "intake idempotency key was reused with different content"
                    )
                if existing["status"] in {"PUBLISHED", "BOUND"}:
                    manifest = IntakeManifestV70.model_validate_json(
                        existing["manifest_json"]
                    )
                    connection.execute("COMMIT")
                    persisted = self.get_intake(manifest.intake_id)
                    current_projection = self.root / "current_intake.json"
                    if not current_projection.exists():
                        self._write_current_intake_projection(persisted)
                    return persisted
                connection.execute("COMMIT")
            else:
                connection.execute(
                    """
                    INSERT INTO intakes(
                        intake_id,idempotency_key,request_hash,status,
                        created_at,updated_at
                    ) VALUES(?,?,?,'STAGING',?,?)
                    """,
                    (
                        intake_id,
                        idempotency_key,
                        request_hash,
                        created_at,
                        created_at,
                    ),
                )
                self._event(
                    connection,
                    event_type="intake.staging",
                    payload={
                        "intake_id": intake_id,
                        "request_hash": request_hash,
                    },
                )
                connection.execute("COMMIT")

        staging = self.staging_root / f"i-{uuid.uuid4().hex[:12]}"
        final = self.intakes_root / intake_id
        lock_path = self.root / ".intake-publisher.lock"
        try:
            with exclusive_file_lock(lock_path):
                if not final.exists():
                    staging.mkdir(parents=False, exist_ok=False)
                    blobs_dir = staging / "attachments"
                    blobs_dir.mkdir()
                    for source, metadata in attachments:
                        payload = source.read_bytes()
                        if _sha256_bytes(payload) != metadata["sha256"]:
                            raise OperatorIntegrityError(
                                f"attachment changed during publication: {source.name}"
                            )
                        blob = self.blobs_root / metadata["sha256"]
                        if blob.exists():
                            if _sha256_bytes(blob.read_bytes()) != metadata["sha256"]:
                                raise OperatorIntegrityError(
                                    "content-addressed blob hash differs"
                                )
                        else:
                            _write_new_bytes(blob, payload)
                        _write_new_bytes(blobs_dir / source.name, payload)
                    manifest = IntakeManifestV70.seal(
                        intake_id=intake_id,
                        idempotency_key=idempotency_key,
                        objective=request["objective"],
                        objective_hash=request["objective_hash"],
                        requested_workspace_id=requested_workspace_id,
                        evidence_scope=evidence_scope,
                        workflow_mode=workflow_mode,
                        attachments=tuple(
                            IntakeAttachmentV70(**metadata)
                            for _, metadata in attachments
                        ),
                        total_bytes=request["total_bytes"],
                        received_at=created_at,
                        request_hash=request_hash,
                    )
                    _write_new_bytes(
                        staging / "manifest.json",
                        _json_bytes(manifest.model_dump(mode="json")),
                    )
                    _fsync_directory(staging)
                    os.replace(staging, final)
                    _fsync_directory(self.intakes_root)
                else:
                    manifest = IntakeManifestV70.model_validate_json(
                        (final / "manifest.json").read_text(encoding="utf-8")
                    )
                    if (
                        manifest.request_hash != request_hash
                        or manifest.idempotency_key != idempotency_key
                    ):
                        raise OperatorConflictError(
                            "published intake directory has different content"
                        )
                self.verify_intake(intake_id)
                with closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        """
                        UPDATE intakes SET status='PUBLISHED',manifest_hash=?,
                            manifest_json=?,updated_at=?,last_error_json=NULL
                        WHERE intake_id=? AND request_hash=?
                        """,
                        (
                            manifest.manifest_hash,
                            canonical_json(manifest.model_dump(mode="json")),
                            _utc_now(),
                            intake_id,
                            request_hash,
                        ),
                    )
                    self._event(
                        connection,
                        event_type="intake.published",
                        payload={
                            "intake_id": intake_id,
                            "manifest_hash": manifest.manifest_hash,
                        },
                    )
                    connection.execute("COMMIT")
                self._write_current_intake_projection(manifest)
                return manifest
        except Exception as exc:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE intakes SET last_error_json=?,updated_at=?
                    WHERE intake_id=?
                    """,
                    (
                        canonical_json(
                            {
                                "error_type": type(exc).__name__,
                                "message": str(exc)[:1000],
                            }
                        ),
                        _utc_now(),
                        intake_id,
                    ),
                )
                self._event(
                    connection,
                    event_type="intake.publication_failed",
                    payload={
                        "intake_id": intake_id,
                        "error_type": type(exc).__name__,
                    },
                )
                connection.execute("COMMIT")
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def get_intake(self, intake_id: str) -> IntakeManifestV70:
        _validate_intake_id(intake_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM intakes WHERE intake_id=?", (intake_id,)
            ).fetchone()
        if (
            row is None
            or row["status"] not in {"PUBLISHED", "BOUND"}
            or row["manifest_json"] is None
        ):
            raise OperatorPlaneError(f"published intake not found: {intake_id}")
        database_manifest = IntakeManifestV70.model_validate_json(
            row["manifest_json"]
        )
        disk_manifest = self._verified_intake_manifest(intake_id)
        if (
            row["manifest_hash"] != database_manifest.manifest_hash
            or database_manifest.manifest_hash != disk_manifest.manifest_hash
            or database_manifest.model_dump(mode="json")
            != disk_manifest.model_dump(mode="json")
        ):
            raise OperatorIntegrityError(
                "database and committed intake manifests differ"
            )
        return disk_manifest

    def _verified_intake_manifest(
        self,
        intake_id: str,
    ) -> IntakeManifestV70:
        _validate_intake_id(intake_id)
        if _is_link_like(self.intakes_root) or _is_link_like(self.blobs_root):
            raise OperatorIntegrityError(
                "operator intake storage cannot be a link or junction"
            )
        final = self.intakes_root / intake_id
        _assert_safe_descendant(self.intakes_root, final)
        manifest_path = final / "manifest.json"
        if not manifest_path.is_file():
            raise OperatorIntegrityError("published intake manifest is missing")
        manifest = IntakeManifestV70.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.intake_id != intake_id or manifest.manifest_hash is None:
            raise OperatorIntegrityError("published intake identity differs")
        for attachment in manifest.attachments:
            published = final / "attachments" / attachment.logical_name
            blob = self.blobs_root / attachment.sha256
            for path in (published, blob):
                if (
                    _is_link_like(path)
                    or
                    not path.is_file()
                    or path.stat().st_size != attachment.size_bytes
                    or _sha256_bytes(path.read_bytes()) != attachment.sha256
                ):
                    raise OperatorIntegrityError(
                        f"intake attachment verification failed: {attachment.logical_name}"
                    )
        expected_files = {
            "manifest.json",
            *(
                f"attachments/{attachment.logical_name}"
                for attachment in manifest.attachments
            ),
        }
        actual_files: set[str] = set()
        for path in final.rglob("*"):
            if _is_link_like(path):
                raise OperatorIntegrityError(
                    "committed intake contains a symbolic link or junction"
                )
            if path.is_file():
                actual_files.add(path.relative_to(final).as_posix())
        if actual_files != expected_files:
            raise OperatorIntegrityError(
                "committed intake contains missing or unexpected files"
            )
        return manifest

    def verify_intake(self, intake_id: str) -> bool:
        self._verified_intake_manifest(intake_id)
        return True

    def verify_materialized_intake(
        self,
        intake_id: str,
        workspace_root: str | Path,
    ) -> bool:
        manifest = self.get_intake(intake_id)
        unresolved_root = Path(workspace_root)
        if _is_link_like(unresolved_root):
            raise OperatorIntegrityError(
                "workspace root cannot be a symbolic link or junction"
            )
        root = unresolved_root.resolve(strict=True)
        target = root / "problem" / "intake"
        _assert_safe_descendant(root, target)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file() or _is_link_like(manifest_path):
            raise OperatorIntegrityError(
                "workspace intake manifest is missing or unsafe"
            )
        installed = IntakeManifestV70.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            installed.manifest_hash != manifest.manifest_hash
            or installed.model_dump(mode="json")
            != manifest.model_dump(mode="json")
        ):
            raise OperatorIntegrityError(
                "workspace intake manifest differs from committed intake"
            )
        expected_files = {
            "manifest.json",
            *(
                f"attachments/{attachment.logical_name}"
                for attachment in manifest.attachments
            ),
        }
        actual_files: set[str] = set()
        for path in target.rglob("*"):
            if _is_link_like(path):
                raise OperatorIntegrityError(
                    "workspace intake contains a symbolic link or junction"
                )
            if not path.is_file():
                continue
            relative = path.relative_to(target).as_posix()
            actual_files.add(relative)
            if relative == "manifest.json":
                continue
            attachment = next(
                (
                    item
                    for item in manifest.attachments
                    if relative == f"attachments/{item.logical_name}"
                ),
                None,
            )
            if (
                attachment is None
                or path.stat().st_size != attachment.size_bytes
                or _sha256_bytes(path.read_bytes()) != attachment.sha256
            ):
                raise OperatorIntegrityError(
                    f"workspace intake attachment differs: {relative}"
                )
        if actual_files != expected_files:
            raise OperatorIntegrityError(
                "workspace intake contains missing or unexpected files"
            )
        return True

    def materialize_intake(self, intake_id: str, workspace_root: str | Path) -> Path:
        manifest = self.get_intake(intake_id)
        unresolved_root = Path(workspace_root)
        unresolved_root.mkdir(parents=True, exist_ok=True)
        if _is_link_like(unresolved_root):
            raise OperatorIntegrityError(
                "workspace root cannot be a symbolic link or junction"
            )
        root = unresolved_root.resolve(strict=True)
        target = root / "problem" / "intake"
        _assert_safe_descendant(root, target)
        if target.exists():
            existing = target / "manifest.json"
            if not existing.is_file():
                raise OperatorConflictError(
                    "workspace intake directory exists without a manifest"
                )
            installed = IntakeManifestV70.model_validate_json(
                existing.read_text(encoding="utf-8")
            )
            if installed.manifest_hash != manifest.manifest_hash:
                raise OperatorConflictError(
                    "workspace already contains another intake manifest"
                )
            self.verify_materialized_intake(intake_id, root)
            return target
        staging = target.with_name(f".i-{uuid.uuid4().hex[:12]}")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            attachments = staging / "attachments"
            attachments.mkdir()
            for item in manifest.attachments:
                payload = (self.blobs_root / item.sha256).read_bytes()
                if _sha256_bytes(payload) != item.sha256:
                    raise OperatorIntegrityError(
                        f"intake blob changed before materialization: {item.logical_name}"
                    )
                _write_new_bytes(attachments / item.logical_name, payload)
            _write_new_bytes(
                staging / "manifest.json",
                _json_bytes(manifest.model_dump(mode="json")),
            )
            _fsync_directory(staging)
            os.replace(staging, target)
            _fsync_directory(target.parent)
            self.verify_materialized_intake(intake_id, root)
            return target
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def bind_intake(
        self,
        intake_id: str,
        *,
        workspace_id: str,
        authority_binding_hash: str,
    ) -> None:
        manifest = self.get_intake(intake_id)
        if len(authority_binding_hash) != 64:
            raise ValueError("authority binding hash must be SHA-256")
        binding_hash = sha256_value(
            {
                "intake_manifest_hash": manifest.manifest_hash,
                "workspace_id": workspace_id,
                "authority_binding_hash": authority_binding_hash,
            }
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT binding_hash FROM intake_bindings
                WHERE intake_id=? AND workspace_id=?
                """,
                (intake_id, workspace_id),
            ).fetchone()
            if existing is not None and existing["binding_hash"] != binding_hash:
                connection.execute("ROLLBACK")
                raise OperatorConflictError("intake binding differs on replay")
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO intake_bindings(
                        intake_id,workspace_id,binding_hash,bound_at
                    ) VALUES(?,?,?,?)
                    """,
                    (intake_id, workspace_id, binding_hash, _utc_now()),
                )
                self._event(
                    connection,
                    event_type="intake.bound",
                    payload={
                        "intake_id": intake_id,
                        "workspace_id": workspace_id,
                        "binding_hash": binding_hash,
                    },
                )
            connection.execute(
                "UPDATE intakes SET status='BOUND',updated_at=? WHERE intake_id=?",
                (_utc_now(), intake_id),
            )
            connection.execute("COMMIT")
        _atomic_write_bytes(
            self.root / "current_task.json",
            _json_bytes(
                {
                    "schema_version": "7.0-current-task-projection",
                    "workspace_id": workspace_id,
                    "intake_id": intake_id,
                    "binding_hash": binding_hash,
                    "claim_scope": "workflow_control_only",
                }
            ),
        )

    def get_intake_binding(
        self,
        intake_id: str,
        *,
        workspace_id: str,
    ) -> dict[str, str] | None:
        _validate_intake_id(intake_id)
        if re.fullmatch(_WORKSPACE_PATTERN, workspace_id) is None:
            raise ValueError("workspace id is unsafe")
        self.get_intake(intake_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT binding_hash,bound_at FROM intake_bindings
                WHERE intake_id=? AND workspace_id=?
                """,
                (intake_id, workspace_id),
            ).fetchone()
        if row is None:
            return None
        if re.fullmatch(_SHA256_PATTERN, row["binding_hash"]) is None:
            raise OperatorIntegrityError("intake binding hash is invalid")
        return {
            "intake_id": intake_id,
            "workspace_id": workspace_id,
            "binding_hash": row["binding_hash"],
            "bound_at": row["bound_at"],
        }

    def doctor(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        expired: list[str] = []
        with closing(self._connect()) as connection:
            quick = connection.execute("PRAGMA quick_check").fetchone()[0]
            if quick != "ok":
                errors.append(f"sqlite quick_check: {quick}")
            metadata = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if (
                metadata is None
                or metadata["value"] != OPERATOR_SCHEMA_VERSION_V70
            ):
                errors.append("operator schema version differs")
            work_rows = connection.execute(
                "SELECT * FROM work_items ORDER BY created_at,work_id"
            ).fetchall()
            work_rows_by_id = {row["work_id"]: row for row in work_rows}
            parsed_submissions: dict[str, OperatorSubmissionV70] = {}
            parsed_projections: dict[str, dict[str, Any]] = {}
            recovery_pending: list[str] = []
            for row in work_rows:
                try:
                    packet = OperatorPacketV70.model_validate_json(row["packet_json"])
                    if (
                        packet.packet_hash is None
                        or packet.authority_binding.binding_hash is None
                    ):
                        errors.append(f"{row['work_id']}: packet is not sealed")
                    if packet.packet_hash != row["packet_hash"]:
                        errors.append(f"{row['work_id']}: packet hash differs")
                    if packet.packet_hash is not None and (
                        row["work_id"] != f"work-{packet.packet_hash[:24]}"
                        or row["workspace_id"] != packet.workspace_id
                        or row["action"] != packet.action
                        or row["idempotency_key"] != packet.idempotency_key
                    ):
                        errors.append(
                            f"{row['work_id']}: routing fields differ from packet"
                        )
                    persisted_paths = tuple(json.loads(row["write_paths_json"]))
                    expected_paths = tuple(
                        normalize_owned_path(path) for path in packet.write_paths
                    )
                    if persisted_paths != expected_paths:
                        errors.append(f"{row['work_id']}: write paths differ")
                    if row["status"] not in _WORK_STATES:
                        errors.append(f"{row['work_id']}: invalid state")
                    if row["status"] == "RECOVERY_PENDING":
                        recovery_pending.append(row["work_id"])
                    if row["status"] == "LEASED" and (
                        not row["worker_id"]
                        or int(row["attempt_epoch"]) < 1
                        or int(row["fencing_token"]) < 1
                        or row["lease_until_epoch"] is None
                    ):
                        errors.append(f"{row['work_id']}: leased state is incomplete")
                    if row["status"] in {"SUBMITTED", "ACCEPTED", "REJECTED"}:
                        if row["submission_json"] is None:
                            errors.append(
                                f"{row['work_id']}: worker submission is missing"
                            )
                        else:
                            submission = OperatorSubmissionV70.model_validate_json(
                                row["submission_json"]
                            )
                            parsed_submissions[row["work_id"]] = submission
                            if (
                                submission.submission_hash is None
                                or submission.output_binding.binding_hash is None
                            ):
                                errors.append(
                                    f"{row['work_id']}: submission is not sealed"
                                )
                            if submission.packet_hash != packet.packet_hash:
                                errors.append(
                                    f"{row['work_id']}: submission packet differs"
                                )
                            if (
                                submission.work_id != row["work_id"]
                                or submission.input_binding_hash
                                != packet.authority_binding.binding_hash
                                or submission.output_binding.workspace_id
                                != packet.workspace_id
                            ):
                                errors.append(
                                    f"{row['work_id']}: submission binding differs"
                                )
                    if row["status"] in {"ACCEPTED", "REJECTED"}:
                        if row["authority_projection_json"] is None:
                            errors.append(
                                f"{row['work_id']}: authority projection is missing"
                            )
                        else:
                            projection = json.loads(
                                row["authority_projection_json"]
                            )
                            parsed_projections[row["work_id"]] = projection
                            if (
                                projection.get("schema_version")
                                not in {
                                    "7.0-authority-projection",
                                    "7.0-authority-reconciliation",
                                }
                                or projection.get("claim_scope")
                                != "workflow_control_only"
                                or projection.get(
                                    "scientific_qualification_granted"
                                )
                                is not False
                                or projection.get("real_world_action_authorized")
                                is not False
                            ):
                                errors.append(
                                    f"{row['work_id']}: authority projection is invalid"
                                )
                            if projection.get("status") != row["status"]:
                                errors.append(
                                    f"{row['work_id']}: authority projection status differs"
                                )
                    if (
                        row["status"] == "LEASED"
                        and row["lease_until_epoch"] is not None
                        and float(row["lease_until_epoch"]) < time.time()
                    ):
                        expired.append(row["work_id"])
                except Exception as exc:
                    errors.append(f"{row['work_id']}: {type(exc).__name__}")
            event_rows = connection.execute(
                "SELECT * FROM operator_events ORDER BY sequence"
            ).fetchall()
            previous: str | None = None
            replayed_states: dict[str, str] = {}
            replayed_packet_hashes: dict[str, str] = {}
            replayed_submission_receipts: dict[str, tuple[str, str]] = {}
            replayed_projections: dict[str, dict[str, Any]] = {}
            for row in event_rows:
                try:
                    payload = json.loads(row["payload_json"])
                    expected = sha256_value(
                        {
                            "work_id": row["work_id"],
                            "event_type": row["event_type"],
                            "payload": payload,
                            "recorded_at": row["recorded_at"],
                            "previous_event_hash": previous,
                        }
                    )
                except Exception as exc:
                    errors.append(
                        "operator event payload is unreadable at sequence "
                        f"{row['sequence']}: {type(exc).__name__}"
                    )
                    break
                if (
                    row["previous_event_hash"] != previous
                    or row["event_hash"] != expected
                ):
                    errors.append(
                        f"operator event chain differs at sequence {row['sequence']}"
                    )
                    break
                previous = row["event_hash"]
                work_id = row["work_id"]
                event_type = row["event_type"]
                if work_id is None:
                    continue
                if work_id not in work_rows_by_id:
                    errors.append(
                        f"operator event references missing work item: {work_id}"
                    )
                    continue
                if event_type == "work.created":
                    packet_hash = payload.get("packet_hash")
                    if isinstance(packet_hash, str):
                        replayed_packet_hashes[work_id] = packet_hash
                elif event_type == "work.submitted":
                    submission_hash = payload.get("submission_hash")
                    output_binding_hash = payload.get("output_binding_hash")
                    if isinstance(submission_hash, str) and isinstance(
                        output_binding_hash, str
                    ):
                        replayed_submission_receipts[work_id] = (
                            submission_hash,
                            output_binding_hash,
                        )
                elif event_type in {
                    "work.accepted",
                    "work.rejected",
                    "work.reconciled_from_authority",
                }:
                    replayed_projections[work_id] = payload
                next_state = {
                    "work.created": "PENDING",
                    "work.claimed": "LEASED",
                    "work.submitted": "SUBMITTED",
                    "work.accepted": "ACCEPTED",
                    "work.rejected": "REJECTED",
                    "work.failed": "FAILED",
                    "work.blocked": "BLOCKED",
                }.get(event_type)
                if event_type == "work.lease_expired":
                    candidate = payload.get("next_status")
                    next_state = candidate if candidate in _WORK_STATES else None
                elif event_type == "work.reconciled_from_authority":
                    candidate = payload.get("status")
                    next_state = candidate if candidate in {"ACCEPTED", "REJECTED"} else None
                elif event_type == "work.heartbeat":
                    next_state = replayed_states.get(work_id)
                if next_state is not None:
                    replayed_states[work_id] = next_state
            for work_id, row in work_rows_by_id.items():
                if replayed_packet_hashes.get(work_id) != row["packet_hash"]:
                    errors.append(
                        f"{work_id}: packet differs from event-derived packet"
                    )
                replayed = replayed_states.get(work_id)
                if replayed is None:
                    errors.append(f"{work_id}: event-derived state is missing")
                elif replayed != row["status"]:
                    errors.append(
                        f"{work_id}: row state differs from event-derived state"
                    )
                submission = parsed_submissions.get(work_id)
                if submission is not None:
                    receipt = replayed_submission_receipts.get(work_id)
                    expected_receipt = (
                        submission.submission_hash,
                        submission.output_binding.binding_hash,
                    )
                    if receipt != expected_receipt:
                        errors.append(
                            f"{work_id}: submission differs from event-derived receipt"
                        )
                projection = parsed_projections.get(work_id)
                if projection is not None and canonical_json(
                    replayed_projections.get(work_id)
                ) != canonical_json(projection):
                    errors.append(
                        f"{work_id}: authority projection differs from event"
                    )
            intake_rows = connection.execute(
                "SELECT * FROM intakes ORDER BY intake_id"
            ).fetchall()
            binding_rows = connection.execute(
                """
                SELECT intake_id,workspace_id,binding_hash,bound_at
                FROM intake_bindings ORDER BY intake_id,workspace_id
                """
            ).fetchall()
        for row in intake_rows:
            if row["status"] in {"PUBLISHED", "BOUND"}:
                try:
                    self.get_intake(row["intake_id"])
                except Exception as exc:
                    errors.append(
                        f"{row['intake_id']}: {type(exc).__name__}"
                    )
            elif row["status"] == "STAGING":
                warnings.append(f"{row['intake_id']}: publication incomplete")
        known_intakes = {row["intake_id"] for row in intake_rows}
        intake_statuses = {
            row["intake_id"]: row["status"] for row in intake_rows
        }
        bound_intakes = {row["intake_id"] for row in binding_rows}
        for row in binding_rows:
            if (
                row["intake_id"] not in known_intakes
                or intake_statuses.get(row["intake_id"]) != "BOUND"
                or re.fullmatch(_WORKSPACE_PATTERN, row["workspace_id"]) is None
                or re.fullmatch(_SHA256_PATTERN, row["binding_hash"]) is None
            ):
                errors.append(
                    f"{row['intake_id']}: intake binding is invalid"
                )
        for row in intake_rows:
            if (
                row["status"] == "BOUND"
                and row["intake_id"] not in bound_intakes
            ):
                errors.append(
                    f"{row['intake_id']}: bound intake has no binding record"
                )
        for path in sorted(self.intakes_root.iterdir()):
            if path.is_dir() and path.name not in known_intakes:
                errors.append(f"{path.name}: published intake is not registered")
        current_intake_path = self.root / "current_intake.json"
        committed_intakes = [
            row
            for row in intake_rows
            if row["status"] in {"PUBLISHED", "BOUND"}
        ]
        if current_intake_path.exists():
            try:
                projection = json.loads(
                    current_intake_path.read_text(encoding="utf-8")
                )
                intake_id = projection["intake_id"]
                manifest = self.get_intake(intake_id)
                if (
                    projection.get("schema_version")
                    != "7.0-current-intake-projection"
                    or projection.get("manifest_hash") != manifest.manifest_hash
                    or projection.get("claim_scope") != "workflow_control_only"
                ):
                    raise OperatorIntegrityError(
                        "current intake projection differs"
                    )
            except Exception as exc:
                errors.append(
                    f"current intake projection: {type(exc).__name__}"
                )
        elif committed_intakes:
            warnings.append(
                "current intake projection is missing and requires exact replay"
            )
        current_task_path = self.root / "current_task.json"
        if current_task_path.exists():
            try:
                projection = json.loads(
                    current_task_path.read_text(encoding="utf-8")
                )
                binding = self.get_intake_binding(
                    projection["intake_id"],
                    workspace_id=projection["workspace_id"],
                )
                if (
                    binding is None
                    or projection.get("schema_version")
                    != "7.0-current-task-projection"
                    or projection.get("binding_hash")
                    != binding["binding_hash"]
                    or projection.get("claim_scope")
                    != "workflow_control_only"
                ):
                    raise OperatorIntegrityError(
                        "current task projection differs"
                    )
            except Exception as exc:
                errors.append(
                    f"current task projection: {type(exc).__name__}"
                )
        elif binding_rows:
            warnings.append(
                "current task projection is missing and requires exact replay"
            )
        staged_paths = [path for path in self.staging_root.iterdir() if path.is_dir()]
        if staged_paths:
            warnings.append(
                f"{len(staged_paths)} incomplete staging directorie(s) require inspection"
            )
        if expired:
            warnings.append(
                f"{len(expired)} expired lease(s) require explicit reconcile"
            )
        if recovery_pending:
            warnings.append(
                f"{len(recovery_pending)} ambiguous work item(s) require "
                "explicit recovery"
            )
        status = "FAIL" if errors else ("RECOVERY_PENDING" if warnings else "PASS")
        return {
            "schema_version": "7.0-operator-doctor",
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "expired_work_ids": expired,
            "work_item_count": len(work_rows),
            "intake_count": len(intake_rows),
            "intake_binding_count": len(binding_rows),
            "event_count": len(event_rows),
            "claim_scope": "workflow_control_only",
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }


__all__ = [
    "IntakeAttachmentV70",
    "IntakeManifestV70",
    "OPERATOR_ROOT_NAME_V70",
    "OPERATOR_SCHEMA_VERSION_V70",
    "OperatorAuthorityBindingV70",
    "OperatorConflictError",
    "OperatorIntegrityError",
    "OperatorLeaseError",
    "OperatorLeaseV70",
    "OperatorPacketV70",
    "OperatorPlaneError",
    "OperatorStoreV70",
    "OperatorSubmissionV70",
    "assert_changed_paths_owned",
    "capture_file_manifest",
    "changed_manifest_paths",
    "normalize_owned_path",
    "owned_paths_overlap",
]
