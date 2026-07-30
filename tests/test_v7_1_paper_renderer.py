from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from fma.hashing import sha256_value
from fma.v5_1.codex_stage_driver import RoleProcessReceiptV51, RoleRequestV51
from fma.v7_1 import paper_runtime
from fma.v7_1.paper_role_driver import (
    NativeLayoutReviewDraftV71,
    NativePaperDraftV71,
    NativeSemanticReviewDraftV71,
)
from fma.v7_1.paper_renderer import (
    _render_sources,
    build_paper_v71,
    verify_paper_build_v71,
)
from fma.v7_1.paper_runtime import (
    PaperDeliveryError,
    record_layout_review_v71,
)
from fma.v7_1.paper_schemas import (
    CitationManifestV71,
    CitationRecordV71,
    FigureBindingV71,
    FigureManifestV71,
    PaperAuthoringRequestV71,
    PaperBuildReceiptV71,
    PaperClaimLedgerV71,
    PaperClaimV71,
    PaperContentAuditV71,
    PaperCurrentProjectionV71,
    PaperEvidenceBundleV71,
    PaperEvidenceItemV71,
    PaperLayoutReviewV71,
    PaperMetadataV71,
    PaperNumericTokenV71,
    PaperSemanticReviewV71,
    TableBindingV71,
    TableManifestV71,
)


_ZERO_HASH = "0" * 64
_GATE_HASHES = {f"S{index}": str(index) * 64 for index in range(7)}
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_test_png(path: Path, *, width: int = 96, height: int = 48) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((36, 90, 141) if x < width // 2 else (238, 243, 247))
        rows.append(bytes(row))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _bundle(root: Path) -> PaperEvidenceBundleV71:
    evidence_path = root / "results" / "metric.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text('{"score": 42.5}\n', encoding="utf-8")
    evidence = PaperEvidenceItemV71(
        evidence_id="E.s6.metric",
        stage="S6",
        relative_path="results/metric.json",
        sha256=_sha256(evidence_path),
        size_bytes=evidence_path.stat().st_size,
        manifest_hash="a" * 64,
        gate_hash=_GATE_HASHES["S6"],
        kind="result",
    )
    numeric = PaperNumericTokenV71(
        token_id="N.s6.metric",
        evidence_id=evidence.evidence_id,
        json_pointer="/score",
        value=42.5,
        display_value="42.5",
    )
    return PaperEvidenceBundleV71.seal(
        workspace_id="workspace.test",
        workspace_spec_hash="b" * 64,
        objective="Test the evidence-bound paper renderer.",
        s6_gate_hash=_GATE_HASHES["S6"],
        current_gate_hashes=_GATE_HASHES,
        evidence_items=[evidence],
        numeric_tokens=[numeric],
        allowed_claim_types=[
            "comparison",
            "decision",
            "limitation",
            "method",
            "model_structure",
            "problem",
            "quantitative",
            "robustness",
        ],
        forbidden_claim_types=[
            "causal",
            "global_optimality",
            "mechanistic_truth",
            "unsupported_extrapolation",
        ],
        requested_model="gpt-5.6-sol",
    )


def _publication_models(
    root: Path,
    bundle: PaperEvidenceBundleV71,
    *,
    include_figure: bool = True,
):
    figure_path = root / "assets" / "figure.png"
    _write_test_png(figure_path)
    table_path = root / "assets" / "table.csv"
    table_path.write_text("Metric,Value\nscore,bound\n", encoding="utf-8")
    citation_snapshot = root / "assets" / "citation.json"
    _write_json(
        citation_snapshot,
        {
            "schema_version": "7.1-citation-source-snapshot",
            "title": "A reproducible modelling reference",
            "authors": ["A. Researcher"],
            "year": 2024,
            "venue": "Test Journal",
            "doi": "10.1234/fma.test",
            "url": None,
        },
    )
    metadata = PaperMetadataV71(
        bundle_hash=bundle.bundle_hash,
        title="Evidence-bound renderer test",
        authors=["FMA Test"],
        language="en",
        venue_profile="academic_article",
        requested_model="gpt-5.6-sol",
        writer_context_ids=["writer.context"],
    )
    ledger = PaperClaimLedgerV71(
        bundle_hash=bundle.bundle_hash,
        claims=[
            PaperClaimV71(
                claim_id="claim.result",
                claim_type="quantitative",
                statement="The bound score is reported from frozen evidence.",
                scope_qualifier="Only for this renderer fixture.",
                evidence_ids=["E.s6.metric"],
                numeric_token_ids=["N.s6.metric"],
                citation_ids=["cite.method"],
            )
        ],
    )
    citations = CitationManifestV71(
        bundle_hash=bundle.bundle_hash,
        citations=[
            CitationRecordV71(
                citation_id="cite.method",
                title="A reproducible modelling reference",
                authors=["A. Researcher"],
                year=2024,
                venue="Test Journal",
                doi="10.1234/fma.test",
                source_snapshot_path="assets/citation.json",
                source_snapshot_sha256=_sha256(citation_snapshot),
                supports_claim_ids=["claim.result"],
                verification_status="SNAPSHOT_BOUND",
            )
        ],
    )
    figures = FigureManifestV71(
        bundle_hash=bundle.bundle_hash,
        figures=(
            [
                FigureBindingV71(
                    figure_id="fig.result",
                    artifact_path="assets/figure.png",
                    artifact_sha256=_sha256(figure_path),
                    caption="Bound result graphic",
                    alt_text="Two-colour renderer test image",
                    evidence_ids=["E.s6.metric"],
                    claim_ids=["claim.result"],
                )
            ]
            if include_figure
            else []
        ),
    )
    tables = TableManifestV71(
        bundle_hash=bundle.bundle_hash,
        tables=[
            TableBindingV71(
                table_id="tab.result",
                csv_path="assets/table.csv",
                csv_sha256=_sha256(table_path),
                caption="Bound result table",
                evidence_ids=["E.s6.metric"],
                claim_ids=["claim.result"],
            )
        ],
    )
    return metadata, ledger, citations, figures, tables


def _writer_receipt(
    request: RoleRequestV51, output_hash: str
) -> RoleProcessReceiptV51:
    return RoleProcessReceiptV51.seal(
        request_hash=request.request_hash,
        run_id=request.run_id,
        context_id=request.context_id,
        role_name=request.role_name,
        role_kind="generator",
        transport="fixture",
        provider="test",
        requested_model="gpt-5.6-sol",
        cli_version="fixture",
        executable_sha256="2" * 64,
        prompt_hash="3" * 64,
        output_schema_hash="4" * 64,
        argv_hash="5" * 64,
        stdout_sha256="6" * 64,
        stderr_sha256="7" * 64,
        output_hash=output_hash,
        event_counts={},
        item_counts={},
        usage={},
        tool_event_count=0,
        scratch_unchanged=True,
        completed_at=datetime.now(timezone.utc),
    )


def _reviewer_receipt(
    request: RoleRequestV51, output_hash: str
) -> RoleProcessReceiptV51:
    return RoleProcessReceiptV51.seal(
        request_hash=request.request_hash,
        run_id=request.run_id,
        context_id=request.context_id,
        role_name=request.role_name,
        role_kind="reviewer",
        transport="fixture",
        provider="test",
        requested_model="gpt-5.6-sol",
        cli_version="fixture",
        executable_sha256="b" * 64,
        prompt_hash="c" * 64,
        output_schema_hash="d" * 64,
        argv_hash="e" * 64,
        stdout_sha256="f" * 64,
        stderr_sha256="0" * 64,
        output_hash=output_hash,
        event_counts={},
        item_counts={},
        usage={},
        tool_event_count=0,
        scratch_unchanged=True,
        completed_at=datetime.now(timezone.utc),
    )


def _make_attempt(
    root: Path, *, include_figure: bool = True
) -> tuple[Path, PaperEvidenceBundleV71]:
    bundle = _bundle(root)
    metadata, ledger, citations, figures, tables = _publication_models(
        root, bundle, include_figure=include_figure
    )
    request = PaperAuthoringRequestV71.seal(
        bundle_hash=bundle.bundle_hash,
        language="en",
        venue_profile="academic_article",
        requested_model="gpt-5.6-sol",
        title_hint=metadata.title,
        authors=metadata.authors,
        max_pages=12,
        max_revision_rounds=1,
    )
    attempt_id = (
        f"paper-{bundle.bundle_hash[:16]}-{request.request_hash[:12]}"
    )
    attempt = root / "delivery" / "paper" / "v71" / "attempts" / attempt_id
    source = attempt / "source"
    manifests = attempt / "manifests"
    reviews = attempt / "reviews"
    for directory in (source, manifests, reviews, attempt / "builds"):
        directory.mkdir(parents=True, exist_ok=True)
    abstract = (
        r"\FMAClaim{claim.result} "
        r"The bound score is \FMAValue{N.s6.metric}. "
        r"This evidence-bound renderer fixture tests a publication projection "
        r"without claiming scientific qualification or operational authority."
    )
    body = "\n\n".join(
        [
            r"\FMASection{introduction}{Introduction}",
            (
                r"\FMAClaim{claim.result} "
                r"The bound score is reported from frozen evidence. "
                r"\FMACite{cite.method}"
            ),
            r"\FMASection{methods}{Methods}",
            (
                r"\FMAClaim{claim.result} The renderer substitutes typed macros "
                r"from a frozen bundle and preserves provenance across the "
                r"generated document, its build receipt, and rendered pages."
            ),
            r"\FMASection{results}{Results}",
            (
                r"\FMAClaim{claim.result} "
                + (
                    r"\FMAFigure{fig.result}"
                    if include_figure
                    else "The bound renderer result is retained. "
                )
                + r"\FMATable{tab.result}"
            ),
            r"\FMASection{discussion}{Discussion}",
            (
                r"\FMAClaim{claim.result} Interpretation remains scoped to the "
                r"fixture because publication quality does not establish the "
                r"scientific validity of an underlying mathematical model."
            ),
            r"\FMASection{conclusion}{Conclusion}",
            (
                r"\FMAClaim{claim.result} The publication projection grants no "
                r"authority and is useful only when all upstream evidence and "
                r"independent review bindings remain unchanged."
            ),
        ]
    )
    _write_json(attempt / "evidence_bundle.json", bundle)
    _write_json(attempt / "author_request.json", request)
    packet = paper_runtime._writer_packet(root, bundle, request)
    _write_json(attempt / "writer_packet.json", packet)
    writer_role_request = RoleRequestV51.seal(
        request_id="paper.writer.request",
        task_id="paper.renderer.test",
        stage="S6",
        role_name="paper_writer_v71",
        role_kind="generator",
        subject_id=attempt_id,
        objective="Write an evidence-bound publication draft for renderer testing.",
        public_inputs={
            "author_request": request.model_dump(mode="json"),
            "writer_packet": packet.model_dump(mode="json"),
            "revision_context": {
                "revision_round": 0,
                "feedback": [],
                "previous_output_sha256": None,
                "previous_output": None,
            },
            "format_retry": {
                "format_attempt": 0,
                "previous_contract_error": None,
            },
        },
        allowed_candidate_ids=[],
        authority_denials=[
            "cannot authorize real-world action",
            "cannot grant scientific qualification",
            "cannot review own manuscript",
        ],
        run_id="paper.writer.run",
        context_id="writer.context",
    )
    writer_output = NativePaperDraftV71(
        request_hash=writer_role_request.request_hash,
        role_name="paper_writer_v71",
        metadata=metadata,
        abstract_tex=abstract,
        body_tex=body,
        claim_ledger=ledger,
        citations=citations,
        figures=figures,
        tables=tables,
    ).model_dump(mode="json")
    writer_receipt = _writer_receipt(
        writer_role_request, sha256_value(writer_output)
    )
    _write_json(attempt / "writer_role_request.json", writer_role_request)
    _write_json(attempt / "metadata.json", metadata)
    (source / "abstract.tex").write_text(abstract, encoding="utf-8", newline="\n")
    (source / "body.tex").write_text(body, encoding="utf-8", newline="\n")
    _write_json(manifests / "claim_ledger.json", ledger)
    _write_json(manifests / "citations.json", citations)
    _write_json(manifests / "figures.json", figures)
    _write_json(manifests / "tables.json", tables)
    _write_json(attempt / "writer_transport_receipt.json", writer_receipt)
    _write_json(attempt / "writer_output.json", writer_output)
    audit = _seal_audit(attempt, bundle, attempt_id)
    semantic_request = RoleRequestV51.seal(
        request_id="paper.semantic.request",
        task_id="paper.renderer.test",
        stage="S6",
        role_name="paper_semantic_reviewer_v71",
        role_kind="reviewer",
        subject_id=attempt_id,
        objective="Independently review claim support in the paper fixture.",
        public_inputs={
            "content_audit": audit.model_dump(mode="json"),
            "format_retry": {
                "format_attempt": 0,
                "previous_contract_error": None,
            },
        },
        allowed_candidate_ids=[],
        authority_denials=[
            "cannot authorize real-world action",
            "cannot grant scientific qualification",
            "cannot modify the manuscript",
        ],
        run_id="paper.semantic.run",
        context_id="reviewer.context",
    )
    semantic_draft = NativeSemanticReviewDraftV71(
        request_hash=semantic_request.request_hash,
        role_name="paper_semantic_reviewer_v71",
        reviewed_claim_ids=["claim.result"],
        verdict="APPROVE",
        findings=[],
    ).model_dump(mode="json")
    semantic_request_path = reviews / "semantic_role_request.json"
    semantic_draft_path = reviews / "semantic_role_draft.json"
    reviewer_receipt_path = reviews / "semantic_transport_receipt.json"
    _write_json(semantic_request_path, semantic_request)
    _write_json(semantic_draft_path, semantic_draft)
    _write_json(
        reviewer_receipt_path,
        _reviewer_receipt(semantic_request, sha256_value(semantic_draft)),
    )
    semantic = PaperSemanticReviewV71(
        bundle_hash=bundle.bundle_hash,
        content_audit_hash=audit.audit_hash,
        writer_context_ids=["writer.context"],
        reviewer_context_id="reviewer.context",
        context_isolated=True,
        reviewed_claim_ids=["claim.result"],
        verdict="APPROVE",
        findings=[],
        reviewer_request_sha256=_sha256(semantic_request_path),
        reviewer_draft_sha256=_sha256(semantic_draft_path),
        reviewer_transport_receipt_sha256=_sha256(reviewer_receipt_path),
        requested_model="gpt-5.6-sol",
    )
    _write_json(reviews / "semantic_review.json", semantic)
    _write_json(
        root / "delivery" / "paper" / "v71" / "current.json",
        PaperCurrentProjectionV71(
            attempt_id=attempt_id,
            bundle_hash=bundle.bundle_hash,
            request_hash=request.request_hash,
            status="NEEDS_REVISION",
        ),
    )
    return attempt, bundle


def _seal_audit(
    attempt: Path, bundle: PaperEvidenceBundleV71, attempt_id: str
) -> PaperContentAuditV71:
    audit = PaperContentAuditV71.seal(
        bundle_hash=bundle.bundle_hash,
        attempt_id=attempt_id,
        status="PASS",
        errors=[],
        warnings=[],
        metadata_sha256=_sha256(attempt / "metadata.json"),
        author_request_sha256=_sha256(attempt / "author_request.json"),
        writer_packet_sha256=_sha256(attempt / "writer_packet.json"),
        abstract_sha256=_sha256(attempt / "source" / "abstract.tex"),
        body_sha256=_sha256(attempt / "source" / "body.tex"),
        claim_ledger_sha256=_sha256(attempt / "manifests" / "claim_ledger.json"),
        citation_manifest_sha256=_sha256(attempt / "manifests" / "citations.json"),
        figure_manifest_sha256=_sha256(attempt / "manifests" / "figures.json"),
        table_manifest_sha256=_sha256(attempt / "manifests" / "tables.json"),
        writer_role_request_sha256=_sha256(
            attempt / "writer_role_request.json"
        ),
        writer_transport_receipt_sha256=_sha256(
            attempt / "writer_transport_receipt.json"
        ),
        writer_output_sha256=_sha256(attempt / "writer_output.json"),
        checked_at=datetime.now(timezone.utc),
    )
    _write_json(attempt / "reviews" / "content_audit.json", audit)
    return audit


def _render_tools() -> dict[str, str]:
    tools: dict[str, str] = {}
    missing: list[str] = []
    for executable in ("xelatex", "pdfinfo", "pdftoppm"):
        resolved = shutil.which(executable)
        if resolved is None:
            missing.append(executable)
            continue
        path = Path(resolved)
        if executable != "xelatex" and path.suffix.lower() in {".cmd", ".bat"}:
            native = (
                path.parents[2]
                / "native"
                / "poppler"
                / "Library"
                / "bin"
                / f"{executable}.exe"
            )
            if native.is_file():
                path = native
        tools[executable] = str(path)
    if missing:
        pytest.skip("paper rendering tools are missing: " + ", ".join(missing))
    return tools


@pytest.fixture
def short_workspace() -> Path:
    parent = Path.cwd() / ".pytest-v71-paper"
    root = parent / uuid4().hex[:8]
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def test_render_sources_expands_registered_macros(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    metadata, _, citations, figures, tables = _publication_models(tmp_path, bundle)
    build_root = tmp_path / "build"
    build_root.mkdir()

    rendered = _render_sources(
        tmp_path,
        build_root,
        bundle,
        metadata,
        r"\FMAClaim{claim.result} Value: \FMAValue{N.s6.metric}.",
        (
            r"\FMASection{results}{Results}"
            r"\FMACite{cite.method}\FMAFigure{fig.result}\FMATable{tab.result}"
        ),
        citations,
        figures,
        tables,
    )

    assert r"\FMAValue{" not in rendered
    assert r"\FMACite{" not in rendered
    assert r"\FMAFigure{" not in rendered
    assert r"\FMATable{" not in rendered
    assert r"\FMASection{" not in rendered
    assert r"\FMAClaim{claim.result}" not in rendered
    assert "42.5" in rendered
    assert r"\section{Results}\label{sec:results}" in rendered
    assert r"\includegraphics" in rendered
    assert r"\begin{tabularx}" in rendered
    assert (build_root / "assets" / "figures").is_dir()


def test_real_build_renders_every_page_and_detects_tampering(
    short_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _render_tools()
    if "HOME" not in os.environ and "USERPROFILE" in os.environ:
        monkeypatch.setenv("HOME", os.environ["USERPROFILE"])
    _, _ = _make_attempt(short_workspace, include_figure=False)

    receipt = build_paper_v71(
        short_workspace,
        xelatex_command=tools["xelatex"],
        pdfinfo_command=tools["pdfinfo"],
        pdftoppm_command=tools["pdftoppm"],
        timeout_seconds=120,
    )

    assert receipt.page_images
    assert Path(
        short_workspace, *Path(receipt.pdf_path).parts
    ).read_bytes().startswith(b"%PDF-")
    for relative, expected in receipt.page_images.items():
        page = Path(short_workspace, *Path(relative).parts)
        assert page.is_file()
        assert page.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert _sha256(page) == expected
    assert verify_paper_build_v71(short_workspace).ok

    first_page = Path(
        short_workspace, *Path(next(iter(receipt.page_images))).parts
    )
    original_page = first_page.read_bytes()
    first_page.write_bytes(original_page + b"tamper")
    page_result = verify_paper_build_v71(short_workspace)
    assert not page_result.ok
    assert any("rendered page hash mismatch" in item for item in page_result.mismatches)

    first_page.write_bytes(original_page)
    pdf_path = Path(short_workspace, *Path(receipt.pdf_path).parts)
    original_pdf = pdf_path.read_bytes()
    pdf_path.write_bytes(original_pdf + b"tamper")
    pdf_result = verify_paper_build_v71(short_workspace)
    assert not pdf_result.ok
    assert any("main.pdf" in item and "hash mismatch" in item for item in pdf_result.mismatches)

    pdf_path.write_bytes(original_pdf)
    current_build = pdf_path.parent.parent.parent / "current_build.json"
    current_build.unlink()
    recovered = build_paper_v71(
        short_workspace,
        xelatex_command=tools["xelatex"],
        pdfinfo_command=tools["pdfinfo"],
        pdftoppm_command=tools["pdftoppm"],
        timeout_seconds=120,
    )
    assert recovered == receipt
    assert current_build.is_file()

    def make_layout_review(
        page_images: dict[str, str],
    ) -> PaperLayoutReviewV71:
        role_request = RoleRequestV51.seal(
            request_id=f"paper.layout.{uuid4().hex[:8]}",
            task_id="paper.renderer.test",
            stage="S6",
            role_name="paper_layout_reviewer_v71",
            role_kind="reviewer",
            subject_id=pdf_path.parents[2].name,
            objective="Inspect every rendered paper page.",
            public_inputs={
                "build_hash": receipt.build_hash,
                "page_images": page_images,
                "expected_page_count": len(receipt.page_images),
                "venue_profile": "academic_article",
                "max_pages": 12,
                "format_retry": {
                    "format_attempt": 0,
                    "previous_contract_error": None,
                },
            },
            allowed_candidate_ids=[],
            authority_denials=[
                "cannot grant scientific qualification",
                "cannot authorize real-world action",
                "cannot edit the paper",
            ],
            run_id=f"paper.layout.run.{uuid4().hex[:8]}",
            context_id=f"layout.context.{uuid4().hex[:8]}",
        )
        draft = NativeLayoutReviewDraftV71(
            request_hash=role_request.request_hash,
            role_name="paper_layout_reviewer_v71",
            pages_reviewed=list(range(1, len(receipt.page_images) + 1)),
            verdict="APPROVE",
            findings=[],
        ).model_dump(mode="json")
        reviews = pdf_path.parent.parent.parent / "reviews"
        request_path = reviews / "layout_role_request.json"
        draft_path = reviews / "layout_role_draft.json"
        transport_path = reviews / "layout_transport_receipt.json"
        _write_json(request_path, role_request)
        _write_json(draft_path, draft)
        _write_json(
            transport_path,
            _reviewer_receipt(role_request, sha256_value(draft)),
        )
        return PaperLayoutReviewV71(
            build_hash=receipt.build_hash,
            writer_context_ids=["writer.context"],
            reviewer_context_id=role_request.context_id,
            context_isolated=True,
            pages_reviewed=list(range(1, len(receipt.page_images) + 1)),
            verdict="APPROVE",
            findings=[],
            reviewer_request_sha256=_sha256(request_path),
            reviewer_draft_sha256=_sha256(draft_path),
            reviewer_transport_receipt_sha256=_sha256(transport_path),
            requested_model="gpt-5.6-sol",
        )

    record_layout_review_v71(
        short_workspace, make_layout_review(receipt.page_images)
    )
    tampered_images = dict(receipt.page_images)
    tampered_images[next(iter(tampered_images))] = "0" * 64
    with pytest.raises(PaperDeliveryError, match="reviewed build projection"):
        record_layout_review_v71(
            short_workspace, make_layout_review(tampered_images)
        )

    receipt_path = pdf_path.parent / "build_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    pdf_path.write_bytes(b"not a pdf")
    first_page.write_bytes(b"not a png")
    payload["pdf_sha256"] = _sha256(pdf_path)
    payload["page_images"][next(iter(payload["page_images"]))] = _sha256(
        first_page
    )
    payload.pop("build_hash")
    forged = PaperBuildReceiptV71.seal(**payload)
    _write_json(receipt_path, forged)
    current_payload = json.loads(current_build.read_text(encoding="utf-8"))
    current_payload["build_hash"] = forged.build_hash
    _write_json(current_build, current_payload)

    forged_result = verify_paper_build_v71(short_workspace)
    assert not forged_result.ok
    assert "build-bound PDF does not have a PDF signature" in forged_result.mismatches
    assert any("rendered page is not a PNG" in item for item in forged_result.mismatches)


def test_build_rejects_symlinked_figure_before_running_tex(tmp_path: Path) -> None:
    attempt, bundle = _make_attempt(tmp_path)
    target = tmp_path / "assets" / "figure.png"
    link = tmp_path / "assets" / "linked-figure.png"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable on this host: {exc}")
    figures = FigureManifestV71(
        bundle_hash=bundle.bundle_hash,
        figures=[
            FigureBindingV71(
                figure_id="fig.result",
                artifact_path="assets/linked-figure.png",
                artifact_sha256=_sha256(target),
                caption="Bound result graphic",
                alt_text="Two-colour renderer test image",
                evidence_ids=["E.s6.metric"],
                claim_ids=["claim.result"],
            )
        ],
    )
    _write_json(attempt / "manifests" / "figures.json", figures)
    audit = _seal_audit(attempt, bundle, attempt.name)
    semantic_path = attempt / "reviews" / "semantic_review.json"
    semantic = PaperSemanticReviewV71.model_validate(
        json.loads(semantic_path.read_text(encoding="utf-8"))
    )
    _write_json(
        semantic_path,
        semantic.model_copy(update={"content_audit_hash": audit.audit_hash}),
    )

    with pytest.raises(PaperDeliveryError, match="may not use symlinks"):
        build_paper_v71(
            tmp_path,
            xelatex_command="definitely-not-invoked",
            pdfinfo_command="definitely-not-invoked",
            pdftoppm_command="definitely-not-invoked",
        )


def test_path_traversal_is_rejected_by_manifest_schema(tmp_path: Path) -> None:
    artifact = tmp_path / "figure.png"
    _write_test_png(artifact)
    with pytest.raises(ValueError, match="unsafe"):
        FigureBindingV71(
            figure_id="fig.unsafe",
            artifact_path="../figure.png",
            artifact_sha256=_sha256(artifact),
            caption="Unsafe path fixture",
            alt_text="Unsafe path fixture",
            evidence_ids=["E.s6.metric"],
            claim_ids=["claim.result"],
        )


def test_content_audit_rejects_unsafe_authored_tex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt, bundle = _make_attempt(tmp_path)
    body_path = attempt / "source" / "body.tex"
    body_path.write_text(
        body_path.read_text(encoding="utf-8")
        + "\n\n"
        + r"\FMAClaim{claim.result}\write18{touch should-not-run}",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(
        paper_runtime,
        "build_evidence_bundle_v71",
        lambda workspace, requested_model: bundle,
    )

    audit = paper_runtime.audit_paper_content_v71(
        SimpleNamespace(root=tmp_path)
    )

    assert audit.status == "FAIL"
    assert "authored TeX contains a forbidden control sequence" in audit.errors
