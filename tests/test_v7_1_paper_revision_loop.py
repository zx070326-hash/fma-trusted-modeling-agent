from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fma.v7_1 import paper_cli


class _Result:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        assert mode in {"python", "json"}
        return dict(self.__dict__)


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root

    def current_gate(self, stage: str) -> str | None:
        assert stage == "S6"
        return "6" * 64


def _args(*, max_revision_rounds: int) -> Namespace:
    return Namespace(
        command="run",
        workspace="unused",
        title="Bounded fixture paper",
        author=["Fixture Author"],
        language="en",
        venue_profile="academic_article",
        model="gpt-5.6-sol",
        max_pages=12,
        max_revision_rounds=max_revision_rounds,
        codex_bin=None,
        expected_codex_cli_version="test",
        codex_timeout_seconds=1,
        xelatex="xelatex",
        pdfinfo="pdfinfo",
        pdftoppm="pdftoppm",
        build_timeout_seconds=1.0,
    )


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    audits: list[_Result],
    semantics: list[_Result],
    layouts: list[_Result],
) -> tuple[dict[str, list[Any]], list[object]]:
    calls: dict[str, list[Any]] = {
        "author": [],
        "audit": [],
        "semantic": [],
        "build": [],
        "layout": [],
        "finalize": [],
    }
    outputs: list[object] = []
    workspace = _Workspace(tmp_path)
    paths = SimpleNamespace(
        attempt_id="paper-" + "a" * 16 + "-" + "b" * 12,
        attempt_root=tmp_path / "attempt",
    )

    monkeypatch.setattr(paper_cli, "_workspace", lambda args: workspace)
    monkeypatch.setattr(paper_cli, "_codex_config", lambda args: object())
    monkeypatch.setattr(paper_cli, "_json", outputs.append)
    monkeypatch.setattr(
        paper_cli,
        "prepare_paper_delivery_v71",
        lambda *args, **kwargs: paths,
    )

    def author(*args: object, **kwargs: object) -> _Result:
        calls["author"].append(
            {
                "revision_round": kwargs.get("revision_round"),
                "revision_feedback": kwargs.get("revision_feedback"),
            }
        )
        return _Result(
            schema_version="7.1-native-paper-draft",
            generation=len(calls["author"]),
        )

    def audit(*args: object, **kwargs: object) -> _Result:
        calls["audit"].append(None)
        return audits.pop(0)

    def semantic(*args: object, **kwargs: object) -> _Result:
        calls["semantic"].append(None)
        return semantics.pop(0)

    def build(*args: object, **kwargs: object) -> _Result:
        calls["build"].append(None)
        return _Result(build_hash="7" * 64)

    def layout(*args: object, **kwargs: object) -> _Result:
        calls["layout"].append(None)
        return layouts.pop(0)

    def finalize(*args: object, **kwargs: object) -> _Result:
        calls["finalize"].append(None)
        return _Result(status="DRAFT_READY", delivery_hash="8" * 64)

    monkeypatch.setattr(paper_cli, "run_native_paper_author_v71", author)
    monkeypatch.setattr(paper_cli, "audit_paper_content_v71", audit)
    monkeypatch.setattr(
        paper_cli, "run_native_semantic_review_v71", semantic
    )
    monkeypatch.setattr(paper_cli, "build_paper_v71", build)
    monkeypatch.setattr(paper_cli, "run_native_layout_review_v71", layout)
    monkeypatch.setattr(paper_cli, "finalize_paper_delivery_v71", finalize)
    monkeypatch.setattr(
        paper_cli,
        "project_paper_status_v71",
        lambda *args, **kwargs: None,
    )
    return calls, outputs


def test_run_reauthors_fresh_after_mechanical_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, outputs = _patch_common(
        monkeypatch,
        tmp_path,
        audits=[
            _Result(
                status="FAIL",
                errors=["raw numeric literal"],
                warnings=[],
            ),
            _Result(status="PASS", errors=[], warnings=[]),
        ],
        semantics=[
            _Result(verdict="APPROVE", findings=[]),
        ],
        layouts=[
            _Result(verdict="APPROVE", findings=[]),
        ],
    )

    exit_code = paper_cli._run(_args(max_revision_rounds=1))

    assert exit_code == 0
    assert calls["author"] == [
        {"revision_round": 0, "revision_feedback": []},
        {
            "revision_round": 1,
            "revision_feedback": ["content_audit: raw numeric literal"],
        },
    ]
    assert len(calls["audit"]) == 2
    assert len(calls["semantic"]) == 1
    assert len(calls["build"]) == 1
    assert len(calls["layout"]) == 1
    assert len(calls["finalize"]) == 1
    assert outputs[-1]["status"] == "DRAFT_READY"  # type: ignore[index]


def test_run_reuses_verified_finalized_attempt_without_reopening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, outputs = _patch_common(
        monkeypatch,
        tmp_path,
        audits=[],
        semantics=[],
        layouts=[],
    )
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "delivery_receipt.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        paper_cli,
        "verify_paper_delivery_v71",
        lambda workspace: _Result(
            ok=True,
            status="DRAFT_READY",
            mismatches=[],
        ),
    )

    exit_code = paper_cli._run(_args(max_revision_rounds=2))

    assert exit_code == 0
    assert all(not value for value in calls.values())
    assert outputs[-1]["idempotent_reuse"] is True  # type: ignore[index]
    assert outputs[-1]["status"] == "DRAFT_READY"  # type: ignore[index]


def test_run_stops_immediately_on_semantic_human(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, outputs = _patch_common(
        monkeypatch,
        tmp_path,
        audits=[_Result(status="PASS", errors=[], warnings=[])],
        semantics=[
            _Result(verdict="HUMAN", findings=["venue judgment required"]),
        ],
        layouts=[],
    )

    exit_code = paper_cli._run(_args(max_revision_rounds=2))

    assert exit_code == 3
    assert calls["author"] == [
        {"revision_round": 0, "revision_feedback": []}
    ]
    assert len(calls["semantic"]) == 1
    assert calls["build"] == []
    assert calls["layout"] == []
    assert calls["finalize"] == []
    assert outputs[-1]["status"] == "HUMAN"  # type: ignore[index]
    assert outputs[-1]["stopped_at"] == "semantic_review"  # type: ignore[index]


def test_run_returns_needs_revision_when_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, outputs = _patch_common(
        monkeypatch,
        tmp_path,
        audits=[
            _Result(status="FAIL", errors=["first failure"], warnings=[]),
            _Result(status="FAIL", errors=["second failure"], warnings=[]),
        ],
        semantics=[],
        layouts=[],
    )

    exit_code = paper_cli._run(_args(max_revision_rounds=1))

    assert exit_code == 3
    assert calls["author"] == [
        {"revision_round": 0, "revision_feedback": []},
        {
            "revision_round": 1,
            "revision_feedback": ["content_audit: first failure"],
        },
    ]
    assert len(calls["audit"]) == 2
    assert calls["semantic"] == []
    assert calls["build"] == []
    assert calls["layout"] == []
    assert calls["finalize"] == []
    assert outputs[-1]["status"] == "NEEDS_REVISION"  # type: ignore[index]
    assert outputs[-1]["stopped_at"] == "content_audit"  # type: ignore[index]
