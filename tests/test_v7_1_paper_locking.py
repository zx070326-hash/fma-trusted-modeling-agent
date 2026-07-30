from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from fma.v7_1 import paper_renderer, paper_runtime
from fma.v7_1.paper_runtime import PaperAttemptPathsV71
from fma.v7_1.paper_schemas import (
    PaperAuthoringRequestV71,
    PaperCurrentProjectionV71,
    PaperEvidenceBundleV71,
    PaperEvidenceItemV71,
    PaperNumericTokenV71,
)


_PREPARE_WORKER = r"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from fma.v7_1 import paper_runtime
from fma.v7_1.paper_schemas import PaperEvidenceBundleV71

root = Path(sys.argv[1]).resolve()
bundle = PaperEvidenceBundleV71.model_validate(
    json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
)
start = Path(sys.argv[3])
deadline = time.monotonic() + 30
while not start.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("prepare barrier was not released")
    time.sleep(0.005)
paper_runtime.build_evidence_bundle_v71 = (
    lambda workspace, requested_model: bundle
)
paths = paper_runtime.prepare_paper_delivery_v71(
    SimpleNamespace(root=root),
    title_hint="Concurrent paper preparation",
    authors=["FMA Test"],
    language="en",
    requested_model="gpt-5.6-sol",
)
print(json.dumps({"attempt_id": paths.attempt_id}, sort_keys=True))
"""


_WRITE_WORKER = r"""
import sys
import time
from pathlib import Path

from fma.v7_1 import paper_runtime

target = Path(sys.argv[1]).resolve()
start = Path(sys.argv[2])
workspace_root = Path(sys.argv[3]).resolve()
worker = int(sys.argv[4])
deadline = time.monotonic() + 30
while not start.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("write barrier was not released")
    time.sleep(0.005)
with paper_runtime.paper_writer_lock_v71(workspace_root):
    for sequence in range(1):
        paper_runtime._write_json(
            target,
            {
                "schema_version": "concurrent-write-fixture",
                "worker": worker,
                "sequence": sequence,
                "payload": "x" * 256,
            },
        )
"""


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


def _bundle(root: Path) -> PaperEvidenceBundleV71:
    evidence_path = root / "results" / "metric.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text('{"score": 7.5}\n', encoding="utf-8")
    evidence = PaperEvidenceItemV71(
        evidence_id="E.s6.metric",
        stage="S6",
        relative_path="results/metric.json",
        sha256=_sha256(evidence_path),
        size_bytes=evidence_path.stat().st_size,
        manifest_hash="a" * 64,
        gate_hash="6" * 64,
        kind="result",
    )
    return PaperEvidenceBundleV71.seal(
        workspace_id="workspace.concurrent",
        workspace_spec_hash="b" * 64,
        objective="Exercise deterministic V7.1 cross-process publication writes.",
        s6_gate_hash="6" * 64,
        current_gate_hashes={
            f"S{index}": str(index) * 64 for index in range(7)
        },
        evidence_items=[evidence],
        numeric_tokens=[
            PaperNumericTokenV71(
                token_id="N.s6.metric",
                evidence_id=evidence.evidence_id,
                json_pointer="/score",
                value=7.5,
                display_value="7.5",
            )
        ],
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


def _launch_workers(
    script: str, arguments: list[str], *, count: int
) -> list[subprocess.Popen[str]]:
    return [
        subprocess.Popen(
            [sys.executable, "-c", script, *arguments, str(index)],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index in range(count)
    ]


def test_concurrent_prepare_is_idempotent_and_leaves_valid_projection(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle_path = tmp_path / "input-bundle.json"
    _write_json(bundle_path, bundle)
    start = tmp_path / "prepare.start"
    workers = _launch_workers(
        _PREPARE_WORKER,
        [str(tmp_path), str(bundle_path), str(start)],
        count=4,
    )
    start.write_text("go\n", encoding="utf-8")
    results = [worker.communicate(timeout=60) for worker in workers]

    failures = [
        f"worker {index}: rc={workers[index].returncode}\n{stderr}"
        for index, (_, stderr) in enumerate(results)
        if workers[index].returncode != 0
    ]
    assert not failures, "\n".join(failures)
    attempt_ids = {
        json.loads(stdout.strip().splitlines()[-1])["attempt_id"]
        for stdout, _ in results
    }
    assert len(attempt_ids) == 1

    current_path = tmp_path / "delivery" / "paper" / "v71" / "current.json"
    current = PaperCurrentProjectionV71.model_validate(
        json.loads(current_path.read_text(encoding="utf-8"))
    )
    assert current.attempt_id == next(iter(attempt_ids))
    attempt = (
        tmp_path
        / "delivery"
        / "paper"
        / "v71"
        / "attempts"
        / current.attempt_id
    )
    assert PaperEvidenceBundleV71.model_validate(
        json.loads((attempt / "evidence_bundle.json").read_text(encoding="utf-8"))
    ) == bundle
    request = PaperAuthoringRequestV71.model_validate(
        json.loads((attempt / "author_request.json").read_text(encoding="utf-8"))
    )
    assert request.request_hash == current.request_hash
    paper_runtime.load_validated_writer_packet_v71(
        PaperAttemptPathsV71(
            workspace_root=tmp_path,
            attempt_id=current.attempt_id,
            attempt_root=attempt,
        )
    )
    assert not list((tmp_path / "delivery" / "paper" / "v71").rglob("*.tmp"))


def test_unique_atomic_temp_files_survive_cross_process_writers(
    tmp_path: Path,
) -> None:
    target = tmp_path / "shared" / "projection.json"
    start = tmp_path / "write.start"
    workers = _launch_workers(
        _WRITE_WORKER,
        [str(target), str(start), str(tmp_path)],
        count=6,
    )
    start.write_text("go\n", encoding="utf-8")
    results = [worker.communicate(timeout=60) for worker in workers]

    failures = [
        f"worker {index}: rc={workers[index].returncode}\n{stderr}"
        for index, (_, stderr) in enumerate(results)
        if workers[index].returncode != 0
    ]
    assert not failures, "\n".join(failures)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "concurrent-write-fixture"
    assert payload["worker"] in range(6)
    assert payload["sequence"] == 0
    assert payload["payload"] == "x" * 256
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_stale_finalize_cannot_roll_current_projection_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    request_a = PaperAuthoringRequestV71.seal(
        bundle_hash=bundle.bundle_hash,
        language="en",
        venue_profile="academic_article",
        requested_model="gpt-5.6-sol",
        title_hint="Older paper attempt",
        authors=["FMA Test"],
        max_pages=12,
        max_revision_rounds=1,
    )
    request_b = PaperAuthoringRequestV71.seal(
        bundle_hash=bundle.bundle_hash,
        language="en",
        venue_profile="academic_article",
        requested_model="gpt-5.6-sol",
        title_hint="Newer paper attempt",
        authors=["FMA Test"],
        max_pages=12,
        max_revision_rounds=1,
    )
    attempt_a_id = (
        f"paper-{bundle.bundle_hash[:16]}-{request_a.request_hash[:12]}"
    )
    attempt_b_id = (
        f"paper-{bundle.bundle_hash[:16]}-{request_b.request_hash[:12]}"
    )
    attempt_a = (
        tmp_path
        / "delivery"
        / "paper"
        / "v71"
        / "attempts"
        / attempt_a_id
    )
    reviews = attempt_a / "reviews"
    manifests = attempt_a / "manifests"
    reviews.mkdir(parents=True)
    manifests.mkdir()
    for name in (
        "semantic_transport_receipt.json",
        "layout_transport_receipt.json",
    ):
        (reviews / name).write_text("{}\n", encoding="utf-8")
    paths_a = PaperAttemptPathsV71(tmp_path, attempt_a_id, attempt_a)
    audit = SimpleNamespace(status="PASS", audit_hash="a" * 64)
    semantic = SimpleNamespace(
        verdict="APPROVE",
        bundle_hash=bundle.bundle_hash,
        content_audit_hash=audit.audit_hash,
        reviewed_claim_ids=["claim.result"],
        reviewer_transport_receipt_sha256="f" * 64,
    )
    layout = SimpleNamespace(
        verdict="APPROVE",
        build_hash="c" * 64,
        pages_reviewed=[1],
        reviewer_transport_receipt_sha256="f" * 64,
    )
    ledger = SimpleNamespace(
        claims=[SimpleNamespace(claim_id="claim.result")]
    )
    build = SimpleNamespace(
        build_hash="c" * 64,
        page_images={"page-001.png": "d" * 64},
        pdf_path="delivery/paper/v71/final.pdf",
        pdf_sha256="e" * 64,
    )
    model_values = {
        "PaperEvidenceBundleV71": bundle,
        "PaperContentAuditV71": audit,
        "PaperSemanticReviewV71": semantic,
        "PaperLayoutReviewV71": layout,
        "PaperClaimLedgerV71": ledger,
        "PaperAuthoringRequestV71": request_a,
    }
    current_path = tmp_path / "delivery" / "paper" / "v71" / "current.json"
    paper_runtime._write_json(
        current_path,
        PaperCurrentProjectionV71(
            attempt_id=attempt_a_id,
            bundle_hash=bundle.bundle_hash,
            request_hash=request_a.request_hash,
            status="NEEDS_REVISION",
        ).model_dump(mode="json"),
    )
    entered = threading.Event()
    release = threading.Event()
    projection_reads = 0

    def fake_read_model(path: Path, model_type: type[object]) -> object:
        nonlocal projection_reads
        if model_type is not PaperCurrentProjectionV71:
            return model_values[model_type.__name__]
        projection = PaperCurrentProjectionV71.model_validate(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
        projection_reads += 1
        if projection_reads == 1:
            entered.set()
            assert release.wait(timeout=10)
        return projection

    monkeypatch.setattr(
        paper_runtime,
        "current_paper_attempt_v71",
        lambda root: paths_a,
    )
    monkeypatch.setattr(
        paper_runtime,
        "_read_model",
        fake_read_model,
    )
    monkeypatch.setattr(
        paper_runtime, "_verify_review_role_artifacts", lambda *args: None
    )
    monkeypatch.setattr(
        paper_runtime, "_require_native_codex_receipt", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(paper_runtime, "_sha256_file", lambda path: "f" * 64)
    monkeypatch.setattr(
        paper_renderer, "load_current_build_v71", lambda paths: build
    )
    monkeypatch.setattr(
        paper_renderer,
        "verify_paper_build_v71",
        lambda root: SimpleNamespace(ok=True, mismatches=[]),
    )
    workspace = SimpleNamespace(
        root=tmp_path, current_gate=lambda stage: bundle.s6_gate_hash
    )
    errors: list[BaseException] = []

    def finalize() -> None:
        try:
            paper_runtime.finalize_paper_delivery_v71(workspace)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=finalize, daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    paper_runtime._write_json(
        current_path,
        PaperCurrentProjectionV71(
            attempt_id=attempt_b_id,
            bundle_hash=bundle.bundle_hash,
            request_hash=request_b.request_hash,
            status="NEEDS_REVISION",
        ).model_dump(mode="json"),
    )
    release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], paper_runtime.PaperDeliveryError)
    assert "changed during finalization" in str(errors[0])

    current = PaperCurrentProjectionV71.model_validate(
        json.loads(current_path.read_text(encoding="utf-8"))
    )
    assert current.attempt_id == attempt_b_id
    assert current.request_hash == request_b.request_hash
