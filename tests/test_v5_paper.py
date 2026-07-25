from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from fma.v5.paper import (
    EVIDENCE_SCOPE,
    PaperBuildError,
    assert_paper_build_consistent,
    build_paper,
    render_paper,
    verify_paper_build,
)


def _workspace(
    root: Path,
    *,
    template: str = "Estimate={{result.estimate}}.",
    values: str = '{"estimate": 1.25, "nested": {"lower": 0.75}}',
) -> Path:
    (root / "paper").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / "paper" / "main.template.tex").write_text(
        template, encoding="utf-8"
    )
    (root / "results" / "values.json").write_text(values, encoding="utf-8")
    return root


def test_render_injects_only_finite_numeric_results(tmp_path: Path) -> None:
    root = _workspace(
        tmp_path,
        template=(
            "Estimate={{result.estimate}}; "
            "lower={{result.nested.lower}}."
        ),
    )

    output = render_paper(root)

    assert output == root / "paper" / "build" / "main.tex"
    assert output.read_text(encoding="utf-8") == "Estimate=1.25; lower=0.75."


@pytest.mark.parametrize(
    ("template", "values", "message"),
    [
        (
            "Estimate={{result.missing}}.",
            '{"estimate": 1.25}',
            "missing result value",
        ),
        (
            "Estimate={{result.estimate}}.",
            '{"estimate": NaN}',
            "non-finite JSON number",
        ),
        (
            "Estimate={{result.estimate}}; {{author.name}}.",
            '{"estimate": 1.25}',
            "unresolved template placeholder",
        ),
        (
            "Estimate={{result.estimate}}.",
            '{"estimate": "1.25"}',
            "must resolve to a number",
        ),
        (
            "No machine result is referenced.",
            '{"estimate": 1.25}',
            "must contain at least one",
        ),
        (
            "Estimate={{result.estimate}}; hand-written=42.",
            '{"estimate": 1.25}',
            "hard-coded numeric literal",
        ),
        (
            r"Estimate={{result.estimate}}; \includegraphics{plot.pdf}.",
            '{"estimate": 1.25}',
            "figure manifest",
        ),
    ],
)
def test_render_fails_closed(
    tmp_path: Path,
    template: str,
    values: str,
    message: str,
) -> None:
    root = _workspace(tmp_path, template=template, values=values)

    with pytest.raises(PaperBuildError, match=message):
        render_paper(root)


def test_failed_compiler_cannot_leave_a_success_receipt(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    build_dir = root / "paper" / "build"
    build_dir.mkdir()
    (build_dir / "stale.pdf").write_bytes(b"stale")

    with pytest.raises(PaperBuildError, match="return code 7"):
        build_paper(
            root,
            pdflatex_command=[
                sys.executable,
                "-c",
                "import sys; sys.exit(7)",
            ],
        )

    assert not (build_dir / "stale.pdf").exists()
    assert not (build_dir / "build_receipt.json").exists()


def test_clean_pdflatex_build_and_consistency_detection(
    tmp_path: Path,
) -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        pytest.skip("pdflatex is not installed")
    root = _workspace(
        tmp_path,
        template=r"""\documentclass{article}
\begin{document}
Machine-injected estimate: {{result.estimate}}.
\end{document}
""",
    )
    build_dir = root / "paper" / "build"
    build_dir.mkdir()
    (build_dir / "stale.txt").write_text("stale", encoding="utf-8")

    receipt = build_paper(root, pdflatex_command=pdflatex)

    assert receipt.returncode == 0
    assert receipt.evidence_scope == EVIDENCE_SCOPE
    assert "-halt-on-error" in receipt.command
    assert receipt.scientific_correctness_established is False
    assert receipt.real_world_action_authorized is False
    assert not (build_dir / "stale.txt").exists()
    assert (build_dir / "main.pdf").read_bytes().startswith(b"%PDF-")
    assert json.loads(
        (build_dir / "build_receipt.json").read_text(encoding="utf-8")
    )["pdf_sha256"] == receipt.pdf_sha256

    verification = verify_paper_build(root)
    assert verification.ok
    assert verification.mismatches == []
    assert verification.scientific_correctness_established is False
    assert verification.real_world_action_authorized is False
    assert assert_paper_build_consistent(root) == receipt

    paths_and_expected_mismatches = [
        (
            root / "paper" / "main.template.tex",
            b"\n% template mutation",
            "template_sha256",
        ),
        (
            root / "results" / "values.json",
            b" ",
            "results_sha256",
        ),
        (
            build_dir / "main.tex",
            b"\n% generated source mutation",
            "generated_tex_sha256",
        ),
        (
            build_dir / "main.pdf",
            b"\n% pdf mutation",
            "pdf_sha256",
        ),
    ]
    for path, mutation, expected_mismatch in paths_and_expected_mismatches:
        original = path.read_bytes()
        path.write_bytes(original + mutation)
        changed = verify_paper_build(root)
        assert not changed.ok
        assert any(
            expected_mismatch in mismatch
            for mismatch in changed.mismatches
        )
        with pytest.raises(PaperBuildError):
            assert_paper_build_consistent(root)
        path.write_bytes(original)
        assert verify_paper_build(root).ok


def test_receipt_cannot_claim_scientific_correctness(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    # A success-like fake compiler is sufficient here because this test only
    # challenges receipt schema boundaries, not TeX/PDF tool integration.
    script = (
        "from pathlib import Path; "
        "Path('main.pdf').write_bytes(b'%PDF-1.4\\n%%EOF\\n')"
    )
    receipt = build_paper(
        root,
        pdflatex_command=[sys.executable, "-c", script],
    )
    receipt_path = root / "paper" / "build" / "build_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt.scientific_correctness_established is False

    payload["scientific_correctness_established"] = True
    receipt_path.write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )

    verification = verify_paper_build(root)
    assert not verification.ok
    assert verification.receipt_hash is None
    assert "invalid build receipt" in verification.mismatches[0]
