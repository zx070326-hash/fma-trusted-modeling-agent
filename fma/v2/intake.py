from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fma.hashing import sha256_value

from .schemas import EvidencePedigree, EvidenceSnapshot


MAX_BRIEF_BYTES = 65_536
ALLOWED_BRIEF_SUFFIXES = {".md", ".txt"}
SENSITIVE_FILE_NAMES = {".env", "id_rsa", "id_ed25519", "credentials.json"}
SENSITIVE_FILE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def ingest_local_brief(
    brief_file: str | Path,
    *,
    workspace_root: str | Path,
    source_ref: str | None = None,
    snapshot_id: str = "brief_snapshot",
    captured_at: datetime | None = None,
    max_bytes: int = MAX_BRIEF_BYTES,
) -> EvidenceSnapshot:
    """Read one scoped UTF-8 text brief as untrusted evidence.

    This is intentionally not a generic file-reader.  It allows only small
    Markdown/text files under an explicit workspace root and rejects common
    secret-bearing filenames and extensions.  The result carries no authority
    beyond a content-addressed evidence reference.
    """

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError("workspace_root must be an existing directory")
    path = Path(brief_file).resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("brief file escapes the approved workspace root") from exc
    if not path.is_file():
        raise ValueError("brief file must be an existing regular file")
    _assert_safe_brief_path(path)
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > max_bytes:
        raise ValueError(f"brief exceeds the {max_bytes}-byte intake limit")
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("brief must be UTF-8 text") from exc
    if "\x00" in raw_text:
        raise ValueError("brief contains a NUL byte")
    content_type = "text/markdown" if path.suffix.lower() == ".md" else "text/plain"
    effective_source_ref = source_ref or f"local_file:{relative_path.as_posix()}"
    collected_at = captured_at or datetime.now(timezone.utc)
    return EvidenceSnapshot.seal(
        snapshot_id=snapshot_id,
        pedigree=EvidencePedigree(
            source_kind="local_file",
            source_ref=effective_source_ref,
            collector="harness",
            collected_at=collected_at,
            source_content_hash=sha256_value({"raw_text": raw_text}),
        ),
        content_type=content_type,
        raw_text=raw_text,
    )


def _assert_safe_brief_path(path: Path) -> None:
    name = path.name.lower()
    if name in SENSITIVE_FILE_NAMES or path.suffix.lower() in SENSITIVE_FILE_SUFFIXES:
        raise ValueError("brief path appears to contain sensitive credentials")
    if path.suffix.lower() not in ALLOWED_BRIEF_SUFFIXES:
        raise ValueError("brief must use a .md or .txt extension")
