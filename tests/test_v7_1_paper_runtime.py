from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fma.hashing import sha256_value
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5_1.codex_stage_driver import RoleProcessReceiptV51, RoleRequestV51
from fma.v7_1 import paper_runtime
from fma.v7_1.paper_runtime import (
    PaperAttemptPathsV71,
    PaperDeliveryError,
    assert_paper_attempt_open_v71,
    audit_paper_content_v71,
    current_paper_attempt_v71,
    load_validated_writer_packet_v71,
    prepare_paper_delivery_v71,
)
from fma.v7_1.paper_schemas import (
    CitationManifestV71,
    FigureManifestV71,
    PaperAuthoringRequestV71,
    PaperClaimLedgerV71,
    PaperClaimV71,
    PaperEvidenceBundleV71,
    PaperMetadataV71,
    PaperWriterPacketV71,
    TableManifestV71,
)
from tests.test_v5_stage_workspace import (
    _open_stage,
    _open_through_s2,
    _write_s3,
    _write_s4,
    _write_s5,
    _write_s6,
)


HASH = "4" * 64


def _write_json(path: Path, payload: object) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")  # type: ignore[union-attr]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _workspace_through_s5(
    tmp_path: Path,
) -> tuple[Path, StageWorkspaceV50]:
    root, workspace, plan = _open_through_s2(tmp_path)
    _write_s3(root)
    _open_stage(
        workspace,
        "S3",
        actor="harness",
        scientific_checks=[
            item for item in plan.obligations if item.stage == "S3"
        ],
    )
    _write_s4(root, plan)
    _open_stage(
        workspace,
        "S4",
        actor="harness",
        scientific_checks=[
            item for item in plan.obligations if item.stage == "S4"
        ],
    )
    _write_s5(root)
    _open_stage(workspace, "S5", actor="model")
    return root, workspace


def test_prepare_requires_current_authenticated_s6(tmp_path: Path) -> None:
    _, workspace = _workspace_through_s5(tmp_path)

    with pytest.raises(
        PaperDeliveryError,
        match="requires a current authenticated S6 gate",
    ):
        prepare_paper_delivery_v71(
            workspace,
            title_hint="Synthetic fixture paper",
            authors=["Fixture Author"],
        )


def test_finalized_attempt_rejects_further_mutation(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    paths = PaperAttemptPathsV71(tmp_path, "paper-test", attempt)
    (attempt / "delivery_receipt.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PaperDeliveryError, match="immutable"):
        assert_paper_attempt_open_v71(paths)


def test_fixture_role_receipt_cannot_qualify_for_final_delivery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture-receipt.json"
    receipt = RoleProcessReceiptV51.seal(
        request_hash=HASH,
        run_id="fixture.run",
        context_id="fixture.context",
        role_name="paper_writer_v71",
        role_kind="generator",
        transport="fixture",
        provider="fixture",
        requested_model="gpt-5.6-sol",
        cli_version="fixture",
        executable_sha256=HASH,
        prompt_hash=HASH,
        output_schema_hash=HASH,
        argv_hash=HASH,
        stdout_sha256=HASH,
        stderr_sha256=HASH,
        output_hash=HASH,
        event_counts={"fixture": 1},
        item_counts={"agent_message": 1},
        usage={},
        tool_event_count=0,
        scratch_unchanged=True,
        completed_at=datetime.now(timezone.utc),
    )
    _write_json(path, receipt)

    with pytest.raises(PaperDeliveryError, match="not a qualifying"):
        paper_runtime._require_native_codex_receipt(
            path,
            label="fixture author",
            requested_model="gpt-5.6-sol",
        )


@pytest.fixture(scope="module")
def prepared_s6(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, StageWorkspaceV50, PaperAttemptPathsV71]:
    root, workspace = _workspace_through_s5(
        tmp_path_factory.mktemp("v71-paper-s6")
    )
    _write_s6(root)
    _open_stage(workspace, "S6", actor="harness")
    paths = prepare_paper_delivery_v71(
        workspace,
        title_hint="Synthetic fixture paper",
        authors=["Fixture Author"],
    )
    return root, workspace, paths


def test_prepared_projection_rejects_packet_and_current_tampering(
    prepared_s6: tuple[Path, StageWorkspaceV50, PaperAttemptPathsV71],
) -> None:
    root, _, paths = prepared_s6
    packet_path = paths.attempt_root / "writer_packet.json"
    original_packet = json.loads(packet_path.read_text(encoding="utf-8"))
    tampered_packet = dict(original_packet)
    tampered_packet["objective"] = str(tampered_packet["objective"]) + " tampered"
    tampered_packet["packet_hash"] = sha256_value(
        {
            key: value
            for key, value in tampered_packet.items()
            if key != "packet_hash"
        }
    )
    _write_json(packet_path, tampered_packet)
    with pytest.raises(PaperDeliveryError, match="writer packet differs"):
        load_validated_writer_packet_v71(paths)
    _write_json(packet_path, original_packet)
    assert isinstance(
        load_validated_writer_packet_v71(paths), PaperWriterPacketV71
    )

    current_path = root / "delivery" / "paper" / "v71" / "current.json"
    original_current = json.loads(current_path.read_text(encoding="utf-8"))
    tampered_current = dict(original_current)
    tampered_current["bundle_hash"] = "f" * 64
    _write_json(current_path, tampered_current)
    with pytest.raises(PaperDeliveryError, match="exact attempt inputs"):
        current_paper_attempt_v71(root)
    _write_json(current_path, original_current)
    assert current_paper_attempt_v71(root) == paths


def test_content_audit_rejects_raw_number_unknown_claim_and_dangerous_tex(
    prepared_s6: tuple[Path, StageWorkspaceV50, PaperAttemptPathsV71],
) -> None:
    _, workspace, paths = prepared_s6
    bundle = PaperEvidenceBundleV71.model_validate_json(
        (paths.attempt_root / "evidence_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    evidence_id = bundle.evidence_items[0].evidence_id
    context_id = "paper-writer-context"
    statement = "This fixture remains a bounded workflow demonstration."

    abstract = (
        f"{statement} The unsupported success rate is 99%, establishes a "
        "causal relationship, and guarantees all deployments. "
        r"\FMAClaim{claim.valid}"
        "\n"
    )
    body = (
        "\n".join(
            [
                r"\FMASection{introduction}{Introduction}",
                r"\FMASection{methods}{Methods}",
                r"\FMASection{results}{Results}",
                r"\FMAClaim{claim.unknown}",
                r"\input{outside.tex}",
                r"\FMASection{discussion}{Discussion}",
                r"\FMASection{conclusion}{Conclusion}",
            ]
        )
        + "\n"
    )
    (paths.source_root / "abstract.tex").write_text(abstract, encoding="utf-8")
    (paths.source_root / "body.tex").write_text(body, encoding="utf-8")
    metadata = PaperMetadataV71(
        bundle_hash=bundle.bundle_hash,
        title="Synthetic fixture paper",
        authors=["Fixture Author"],
        language="zh",
        venue_profile="academic_article",
        requested_model="gpt-5.6-sol",
        writer_context_ids=[context_id],
    )
    ledger = PaperClaimLedgerV71(
        bundle_hash=bundle.bundle_hash,
        claims=[
            PaperClaimV71(
                claim_id="claim.valid",
                claim_type="problem",
                statement=statement,
                scope_qualifier="Synthetic fixture only, with threshold 1.",
                evidence_ids=[evidence_id],
            )
        ],
    )
    citations = CitationManifestV71(bundle_hash=bundle.bundle_hash)
    figures = FigureManifestV71(bundle_hash=bundle.bundle_hash)
    tables = TableManifestV71(bundle_hash=bundle.bundle_hash)
    _write_json(paths.attempt_root / "metadata.json", metadata)
    _write_json(paths.manifests_root / "claim_ledger.json", ledger)
    _write_json(paths.manifests_root / "citations.json", citations)
    _write_json(paths.manifests_root / "figures.json", figures)
    _write_json(paths.manifests_root / "tables.json", tables)
    author_request = PaperAuthoringRequestV71.model_validate_json(
        (paths.attempt_root / "author_request.json").read_text(
            encoding="utf-8"
        )
    )
    writer_packet = load_validated_writer_packet_v71(paths)
    run_id = "paper-writer-run"
    role_request = RoleRequestV51.seal(
        request_id="paper-writer-request",
        task_id="paper-writer-task",
        stage="S6",
        role_name="paper_writer_v71",
        role_kind="generator",
        subject_id=paths.attempt_id,
        objective=(
            "Write the evidence-bound fixture manuscript without authority."
        ),
        public_inputs={
            "author_request": author_request.model_dump(mode="json"),
            "writer_packet": writer_packet.model_dump(mode="json"),
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
            "cannot grant scientific qualification",
            "cannot authorize real-world action",
        ],
        run_id=run_id,
        context_id=context_id,
    )
    _write_json(
        paths.attempt_root / "writer_role_request.json", role_request
    )
    request_hash = role_request.request_hash
    writer_output = {
        "request_hash": request_hash,
        "role_name": "paper_writer_v71",
        "selected_candidate_id": None,
        "metadata": metadata.model_dump(mode="json"),
        "abstract_tex": abstract,
        "body_tex": body,
        "claim_ledger": ledger.model_dump(mode="json"),
        "citations": citations.model_dump(mode="json"),
        "figures": figures.model_dump(mode="json"),
        "tables": tables.model_dump(mode="json"),
    }
    _write_json(paths.attempt_root / "writer_output.json", writer_output)
    _write_json(
        paths.attempt_root / "writer_transport_receipt.json",
        RoleProcessReceiptV51.seal(
            request_hash=request_hash,
            run_id=run_id,
            context_id=context_id,
            role_name="paper_writer_v71",
            role_kind="generator",
            transport="fixture",
            provider="fixture",
            requested_model="gpt-5.6-sol",
            cli_version="fixture",
            executable_sha256=HASH,
            prompt_hash=HASH,
            output_schema_hash=HASH,
            argv_hash=HASH,
            stdout_sha256=HASH,
            stderr_sha256=HASH,
            output_hash=sha256_value(writer_output),
            event_counts={"fixture": 1},
            item_counts={"agent_message": 1},
            usage={},
            tool_event_count=0,
            scratch_unchanged=True,
            completed_at=datetime.now(timezone.utc),
        ),
    )

    audit = audit_paper_content_v71(workspace)

    assert audit.status == "FAIL"
    assert any(
        "raw multi-digit/decimal numeric literal: 99" in error
        for error in audit.errors
    )
    assert "unknown claim referenced by manuscript: claim.unknown" in audit.errors
    assert "authored TeX contains a forbidden control sequence" in audit.errors
    assert "manuscript contains forbidden claim-widening language" in audit.errors
    assert "writer transport is fixture-only, not native Codex CLI" in audit.errors
    assert (
        "claim claim.valid scope contains a raw numeric literal; "
        "claim-ledger prose must be number-free and bind values only through "
        "numeric_token_ids"
        in audit.errors
    )
