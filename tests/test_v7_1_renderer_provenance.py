from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v7_1.paper_renderer import (
    _executable,
    _minimal_environment,
    _run,
    _tool_identity,
    _tool_version,
)
from fma.v7_1.paper_runtime import PaperDeliveryError
from fma.v7_1.paper_schemas import (
    PaperBuildReceiptV71,
    PaperToolIdentityV71,
)


def _hash(character: str) -> str:
    return character * 64


def _identity(tool: str, marker: str) -> PaperToolIdentityV71:
    return PaperToolIdentityV71(
        tool=tool,
        resolved_path=f"C:\\tools\\{tool}.exe",
        binary_sha256=_hash(marker),
        version=f"{tool} version 1",
        argv_hash=_hash(marker),
    )


def _receipt() -> PaperBuildReceiptV71:
    xelatex = _identity("xelatex", "1")
    return PaperBuildReceiptV71.seal(
        bundle_hash=_hash("a"),
        content_audit_hash=_hash("b"),
        semantic_review_sha256=_hash("c"),
        template_sha256=_hash("d"),
        author_request_sha256=_hash("e"),
        writer_packet_sha256=_hash("f"),
        metadata_sha256=_hash("1"),
        abstract_sha256=_hash("2"),
        body_sha256=_hash("3"),
        claim_ledger_sha256=_hash("4"),
        citation_manifest_sha256=_hash("5"),
        figure_manifest_sha256=_hash("6"),
        table_manifest_sha256=_hash("7"),
        writer_role_request_sha256=_hash("8"),
        writer_transport_receipt_sha256=_hash("9"),
        writer_output_sha256=_hash("a"),
        generated_tex_path="delivery/paper/main.tex",
        generated_tex_sha256=_hash("b"),
        pdf_path="delivery/paper/main.pdf",
        pdf_sha256=_hash("c"),
        compiler_command=[xelatex.resolved_path, "main.tex"],
        compiler_version=xelatex.version,
        xelatex_identity=xelatex,
        pdfinfo_identity=_identity("pdfinfo", "2"),
        pdftoppm_identity=_identity("pdftoppm", "3"),
        environment_hash=_hash("d"),
        compiler_log_sha256=_hash("e"),
        page_images={"delivery/paper/pages/page-001.png": _hash("f")},
        layout_lint=[],
        built_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )


def test_build_receipt_seals_toolchain_and_environment() -> None:
    receipt = _receipt()

    assert receipt.build_hash is not None
    assert receipt.xelatex_identity.tool == "xelatex"
    assert receipt.pdfinfo_identity.tool == "pdfinfo"
    assert receipt.pdftoppm_identity.tool == "pdftoppm"

    payload = receipt.model_dump(mode="json")
    payload["environment_hash"] = _hash("0")
    with pytest.raises(ValidationError, match="build_hash"):
        PaperBuildReceiptV71.model_validate(payload)


def test_build_receipt_rejects_tool_identity_swaps() -> None:
    payload = _receipt().model_dump(mode="json")
    payload["pdfinfo_identity"]["tool"] = "pdftoppm"
    payload["build_hash"] = None

    with pytest.raises(ValidationError, match="assigned incorrectly"):
        PaperBuildReceiptV71.model_validate(payload)


def test_minimal_environment_excludes_unrelated_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FMA_RENDER_SECRET", "must-not-propagate")

    environment = _minimal_environment()

    assert "FMA_RENDER_SECRET" not in environment
    assert environment["openin_any"] == "p"
    assert environment["openout_any"] == "p"
    assert environment["shell_escape"] == "0"
    assert len(sha256_value(environment)) == 64


def test_bounded_runner_rejects_excessive_stdout(tmp_path: Path) -> None:
    with pytest.raises(PaperDeliveryError, match="output limit"):
        _run(
            [
                sys.executable,
                "-c",
                "import os; os.write(1, b'x' * 4096)",
            ],
            cwd=tmp_path,
            timeout_seconds=10,
            environment=_minimal_environment(),
            max_stdout_bytes=128,
            max_stderr_bytes=128,
        )


def test_tool_probe_records_resolved_binary_version_and_argv(
    tmp_path: Path,
) -> None:
    executable = str(Path(sys.executable).resolve())
    command = [executable, "--version"]
    environment = _minimal_environment()

    version = _tool_version(
        command,
        cwd=tmp_path,
        timeout_seconds=10,
        environment=environment,
    )
    identity = _tool_identity("xelatex", executable, version, [command])

    assert version.startswith("Python ")
    assert identity.resolved_path == executable
    assert identity.binary_sha256
    assert identity.argv_hash == sha256_value([command])


@pytest.mark.skipif(os.name != "nt", reason="Windows executable precedence")
def test_windows_bare_command_prefers_exe_over_cmd_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "paper-tool.cmd").write_text("@exit /b 0\n", encoding="utf-8")
    executable = second / "paper-tool.exe"
    executable.write_bytes(b"fixture")
    monkeypatch.setenv("PATH", os.pathsep.join((str(first), str(second))))

    assert _executable("paper-tool") == str(executable.resolve())
