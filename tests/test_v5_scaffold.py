from __future__ import annotations

import json
from pathlib import Path

import pytest

from fma.v5.scaffold import (
    WorkspaceScaffoldError,
    scaffold_task_workspace,
    validate_task_scaffold,
)


REQUIRED_DIRECTORIES = {
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
}


def test_scaffold_creates_empty_fail_closed_workspace(tmp_path: Path) -> None:
    target = tmp_path / "task"

    created = scaffold_task_workspace(
        target,
        workspace_id="frontier-model-001",
        objective="Distinguish competing dynamical explanations.",
    )

    assert created == target.resolve()
    assert all((created / relative).is_dir() for relative in REQUIRED_DIRECTORIES)
    assert (created / "AGENTS.md").is_file()
    assert (created / "Makefile").is_file()
    assert {
        path.relative_to(created / "skills").as_posix()
        for path in (created / "skills").rglob("SKILL.md")
    } == {
        "methods-mechanistic/SKILL.md",
        "regime-diagnosis/SKILL.md",
        "verification-manual/SKILL.md",
    }

    profile = json.loads((created / "workflow_profile.json").read_text("utf-8"))
    assert profile["workspace_id"] == "frontier-model-001"
    assert profile["objective"] == "Distinguish competing dynamical explanations."
    assert profile["workflow_stages"] == [f"S{index}" for index in range(7)]
    assert profile["gate_authority"] == "external_harness_only"
    assert profile["template_status"] == "empty_scaffold_no_results"
    assert profile["scientific_qualification_granted"] is False
    assert profile["real_world_action_authorized"] is False
    assert validate_task_scaffold(
        created,
        "frontier-model-001",
        "Distinguish competing dynamical explanations.",
    ) == created

    agents = (created / "AGENTS.md").read_text("utf-8")
    assert "frontier-model-001" in agents
    assert "Distinguish competing dynamical explanations." in agents
    assert "A file named `*.stamp` is not a" in agents
    assert "certificate." in agents
    assert "Missing domain checks are `NOT_RUN`" in agents

    assert [path.name for path in (created / "gates").iterdir()] == ["README.md"]
    assert list((created / "results").iterdir()) == []
    assert list((created / "predictions").iterdir()) == []
    assert not list(created.rglob("*.stamp"))


def test_scaffold_accepts_existing_empty_target(tmp_path: Path) -> None:
    target = tmp_path / "empty"
    target.mkdir()

    assert scaffold_task_workspace(target, "task-2", "Test a bounded hypothesis.") == (
        target.resolve()
    )


def test_scaffold_refuses_non_empty_target_without_changing_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(WorkspaceScaffoldError, match="non-empty"):
        scaffold_task_workspace(target, "task-3", "Do not overwrite.")

    assert marker.read_text("utf-8") == "user data"
    assert list(target.iterdir()) == [marker]


def test_unrecognized_non_empty_directory_cannot_be_adopted(
    tmp_path: Path,
) -> None:
    target = tmp_path / "unrelated"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(WorkspaceScaffoldError, match="recognized V5 scaffold"):
        validate_task_scaffold(target, "task-3", "Do not adopt this folder.")

    assert marker.read_text("utf-8") == "user data"
    assert not (target / ".fma").exists()


@pytest.mark.parametrize(
    "workspace_id",
    ["../escape", "..", ".", "nested/task", r"nested\\task", "/absolute"],
)
def test_scaffold_rejects_path_like_workspace_ids(
    tmp_path: Path, workspace_id: str
) -> None:
    target = tmp_path / "task"

    with pytest.raises(WorkspaceScaffoldError, match="workspace_id"):
        scaffold_task_workspace(target, workspace_id, "Bounded objective.")

    assert not target.exists()


@pytest.mark.parametrize("objective", ["", " ", "\n\t"])
def test_scaffold_rejects_blank_objective(
    tmp_path: Path, objective: str
) -> None:
    target = tmp_path / "task"

    with pytest.raises(WorkspaceScaffoldError, match="objective"):
        scaffold_task_workspace(target, "valid-id", objective)

    assert not target.exists()


def test_makefile_is_a_facade_and_does_not_mint_stamps(tmp_path: Path) -> None:
    created = scaffold_task_workspace(
        tmp_path / "task", "task-4", "Exercise the stage protocol."
    )
    makefile = (created / "Makefile").read_text("utf-8")

    assert "-m fma.v5" in makefile
    assert "gate-s0:" in makefile
    assert "gate-s6:" in makefile
    assert "external-harness certificate" in makefile
    assert "touch gates/" not in makefile
    assert ".stamp" not in makefile
