"""Create an empty, fail-closed FMA V5 task workspace.

The scaffold is deliberately boring: it creates places for evidence and
instructions for producing it, but it does not create reviews, gate
certificates, registered predictions, results, or scientific claims.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Final


class WorkspaceScaffoldError(ValueError):
    """Raised when a workspace cannot be created without overwriting data."""


_WORKSPACE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,58}[A-Za-z0-9])?"
)
_REQUIRED_DIRECTORIES: Final[tuple[str, ...]] = (
    "problem",
    "docs/candidates",
    "data/raw",
    "data/processed",
    "src/models",
    "src/solvers",
    "src/figures",
    "checks",
    "results",
    "predictions",
    "gates",
    "paper/build",
    ".fma",
    "skills",
)
_STAGES: Final[tuple[str, ...]] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")


def _validate_workspace_id(workspace_id: str) -> str:
    if not isinstance(workspace_id, str) or not _WORKSPACE_ID_PATTERN.fullmatch(
        workspace_id
    ):
        raise WorkspaceScaffoldError(
            "workspace_id must be a 1-60 character identifier containing only "
            "letters, numbers, '.', '_', or '-' and may not contain a path"
        )
    if workspace_id in {".", ".."}:
        raise WorkspaceScaffoldError("workspace_id may not be '.' or '..'")
    return workspace_id


def _validate_objective(objective: str) -> str:
    if not isinstance(objective, str) or not objective.strip():
        raise WorkspaceScaffoldError("objective must be a non-empty string")
    if len(objective) > 4_000:
        raise WorkspaceScaffoldError("objective must not exceed 4000 characters")
    return objective.strip()


def _resolve_target(root: os.PathLike[str] | str) -> Path:
    if isinstance(root, str) and not root.strip():
        raise WorkspaceScaffoldError("root must identify a task workspace directory")
    unresolved = Path(root).expanduser()
    if unresolved.is_symlink():
        raise WorkspaceScaffoldError("root may not be a symbolic link")
    target = unresolved.resolve(strict=False)
    if target.exists():
        if not target.is_dir():
            raise WorkspaceScaffoldError(f"root is not a directory: {target}")
        if next(target.iterdir(), None) is not None:
            raise WorkspaceScaffoldError(
                f"refusing to overwrite non-empty workspace: {target}"
            )
    return target


def _safe_destination(root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise WorkspaceScaffoldError(
            f"template path escapes the workspace: {relative_path}"
        )
    destination = (root / relative_path).resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise WorkspaceScaffoldError(
            f"template path escapes the workspace: {relative_path}"
        ) from exc
    return destination


def _write_new_text(path: Path, content: str) -> None:
    if path.exists():
        raise WorkspaceScaffoldError(f"refusing to overwrite scaffold file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _scaffold_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scaffold_task_workspace(
    root: os.PathLike[str] | str,
    workspace_id: str,
    objective: str,
) -> Path:
    """Create and return an empty task workspace.

    ``root`` must be absent or an empty directory.  The function never
    overwrites an existing file and rejects identifiers that could be treated
    as paths.  All generated gate material is documentation only; trusted gate
    certificates must be issued later by the external harness.
    """

    validated_workspace_id = _validate_workspace_id(workspace_id)
    validated_objective = _validate_objective(objective)
    target = _resolve_target(root)
    template_root = Path(__file__).with_name("task_template").resolve(strict=True)

    template_files: list[tuple[Path, Path]] = []
    for source in sorted(template_root.rglob("*")):
        if source.is_symlink():
            raise WorkspaceScaffoldError(f"template may not contain symlinks: {source}")
        if source.is_file():
            relative = source.relative_to(template_root)
            template_files.append((source, _safe_destination(target, relative)))

    target.mkdir(parents=True, exist_ok=True)
    for relative_directory in _REQUIRED_DIRECTORIES:
        _safe_destination(target, Path(relative_directory)).mkdir(
            parents=True, exist_ok=True
        )

    replacements = {
        "{{WORKSPACE_ID}}": validated_workspace_id,
        "{{OBJECTIVE}}": validated_objective,
    }
    for source, destination in template_files:
        content = source.read_text(encoding="utf-8")
        for marker, value in replacements.items():
            content = content.replace(marker, value)
        _write_new_text(destination, content)

    profile = {
        "schema_version": "5.0",
        "workspace_id": validated_workspace_id,
        "objective": validated_objective,
        "workflow_stages": list(_STAGES),
        "gate_authority": "external_harness_only",
        "template_status": "empty_scaffold_no_results",
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
        "raw_data_path": "data/raw",
        "advisory_journal_paths_not_authority": [
            "docs/notebook.md",
            "docs/decisions.log",
            "docs/model_genealogy.md",
        ],
    }
    _write_new_text(
        _safe_destination(target, Path("workflow_profile.json")),
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    scaffold_marker = {
        "schema_version": "5.0",
        "workspace_id": validated_workspace_id,
        "objective": validated_objective,
        "template_status": "recognized_empty_scaffold",
    }
    scaffold_marker["scaffold_hash"] = _scaffold_hash(scaffold_marker)
    _write_new_text(
        _safe_destination(target, Path(".fma/scaffold.json")),
        json.dumps(
            scaffold_marker,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return target


def validate_task_scaffold(
    root: os.PathLike[str] | str,
    workspace_id: str,
    objective: str,
) -> Path:
    """Verify that a non-empty directory is an untouched V5 scaffold."""

    validated_workspace_id = _validate_workspace_id(workspace_id)
    validated_objective = _validate_objective(objective)
    target = Path(root).expanduser().resolve(strict=True)
    marker_path = target / ".fma" / "scaffold.json"
    profile_path = target / "workflow_profile.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceScaffoldError(
            "non-empty target is not a recognized V5 scaffold"
        ) from exc
    marker_hash = marker.pop("scaffold_hash", None)
    expected_marker = {
        "schema_version": "5.0",
        "workspace_id": validated_workspace_id,
        "objective": validated_objective,
        "template_status": "recognized_empty_scaffold",
    }
    if marker != expected_marker or marker_hash != _scaffold_hash(marker):
        raise WorkspaceScaffoldError(
            "scaffold marker does not match workspace identity"
        )
    if (
        profile.get("workspace_id") != validated_workspace_id
        or profile.get("objective") != validated_objective
        or profile.get("template_status") != "empty_scaffold_no_results"
        or profile.get("scientific_qualification_granted") is not False
        or profile.get("real_world_action_authorized") is not False
    ):
        raise WorkspaceScaffoldError(
            "workflow profile does not match scaffold identity"
        )
    required_files = (
        "AGENTS.md",
        "Makefile",
        "gates/README.md",
        "skills/methods-mechanistic/SKILL.md",
        "skills/regime-diagnosis/SKILL.md",
        "skills/verification-manual/SKILL.md",
    )
    for relative in _REQUIRED_DIRECTORIES:
        path = target / relative
        if not path.is_dir() or path.is_symlink():
            raise WorkspaceScaffoldError(
                f"recognized scaffold directory is missing or unsafe: {relative}"
            )
    for relative in required_files:
        path = target / relative
        if not path.is_file() or path.is_symlink():
            raise WorkspaceScaffoldError(
                f"recognized scaffold file is missing or unsafe: {relative}"
            )
    control_entries = sorted(
        item.name for item in (target / ".fma").iterdir()
    )
    if control_entries != ["scaffold.json"]:
        raise WorkspaceScaffoldError(
            "scaffold control directory is not in its pre-initialization state"
        )
    return target


__all__ = [
    "WorkspaceScaffoldError",
    "scaffold_task_workspace",
    "validate_task_scaffold",
]
