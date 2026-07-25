from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fma.v2.method_knowledge import (
    MethodFetchResponse,
    MethodSourceContractV22,
    capture_method_candidate,
    fetch_method_source,
    verify_method_learning_run,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
URL = "https://otexts.com/fpp3/ses.html"
BODY = b"<html><body>Simple exponential smoothing uses a level.</body></html>"


def _contract() -> MethodSourceContractV22:
    return MethodSourceContractV22.seal(
        source_id="fpp3_ses",
        source_url=URL,
        max_bytes=1024,
        approved_claim_scope=["simple exponential smoothing recurrence and limits"],
        created_at=NOW,
    )


def _fetcher(url: str, max_bytes: int) -> MethodFetchResponse:
    assert url == URL
    assert max_bytes == 1024
    return MethodFetchResponse(
        status=200,
        final_url=URL,
        headers={"content-type": "text/html; charset=utf-8", "etag": "fixture"},
        body=BODY,
    )


def test_web_material_can_only_create_a_candidate_knowledge_record(tmp_path) -> None:
    outcome = capture_method_candidate(
        tmp_path,
        contract=_contract(),
        claim_id="ses_recent_weighting",
        statement="Recent observations receive geometrically greater weight in SES.",
        applicability_conditions=["series is dominated by a changing level"],
        exclusions=["unmodeled trend or seasonality dominates"],
        proposed_operator="exponential_smoothing",
        frozen_parameters={"alpha": 0.3},
        fetcher=_fetcher,
        captured_at=NOW,
        run_id="method_fixture",
    )
    assert outcome.snapshot.trust_class == "untrusted_web_data"
    assert outcome.knowledge.status == "candidate_requires_hidden_validation"
    assert outcome.knowledge.validation_requirement == "paired_hidden_worldpack_ablation"
    assert verify_method_learning_run(outcome.store.run_directory)


def test_fetcher_rejects_redirect_size_and_media_type() -> None:
    contract = _contract()

    def redirected(url: str, max_bytes: int) -> MethodFetchResponse:
        return MethodFetchResponse(200, "https://example.com/injected", {"content-type": "text/html"}, BODY)

    with pytest.raises(RuntimeError, match="redirected"):
        fetch_method_source(contract, fetcher=redirected, retrieved_at=NOW)

    def oversized(url: str, max_bytes: int) -> MethodFetchResponse:
        return MethodFetchResponse(200, URL, {"content-type": "text/html"}, b"x" * 1025)

    with pytest.raises(RuntimeError, match="max_bytes"):
        fetch_method_source(contract, fetcher=oversized, retrieved_at=NOW)

    def wrong_media(url: str, max_bytes: int) -> MethodFetchResponse:
        return MethodFetchResponse(200, URL, {"content-type": "application/json"}, BODY)

    with pytest.raises(RuntimeError, match="content type"):
        fetch_method_source(contract, fetcher=wrong_media, retrieved_at=NOW)


def test_tampered_method_evidence_breaks_replay(tmp_path) -> None:
    outcome = capture_method_candidate(
        tmp_path,
        contract=_contract(),
        claim_id="ses_tamper_case",
        statement="Simple exponential smoothing estimates a changing level.",
        applicability_conditions=["level-only forecasting is plausible"],
        exclusions=["trend or seasonality requires another skeleton"],
        proposed_operator="exponential_smoothing",
        frozen_parameters={"alpha": 0.3},
        fetcher=_fetcher,
        captured_at=NOW,
        run_id="method_tamper",
    )
    ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "method_evidence_snapshot_v22"
    )
    path = outcome.store.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["raw_text"] += " tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_method_learning_run(outcome.store.run_directory)


def test_non_allowlisted_method_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        MethodSourceContractV22.seal(
            source_id="untrusted_blog",
            source_url="https://example.com/method.html",
            approved_claim_scope=["anything"],
            created_at=NOW,
        )
