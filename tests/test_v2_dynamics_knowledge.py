from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from fma.v2.dynamics_knowledge import (
    APPROVED_DYNAMICS_SOURCES,
    LiteratureFetchResponse,
    capture_dynamics_knowledge,
    default_dynamics_source_contracts,
    fetch_literature_source,
    verify_dynamics_knowledge_run,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def _crossref_body(url: str) -> bytes:
    source_id = next(
        source_id
        for source_id, (_doi, approved_url) in APPROVED_DYNAMICS_SOURCES.items()
        if approved_url == url
    )
    doi = APPROVED_DYNAMICS_SOURCES[source_id][0]
    payload = {
        "status": "ok",
        "message-type": "work",
        "message-version": "1.0.0",
        "message": {
            "DOI": doi,
            "title": [f"Fixture title for {source_id}"],
            "publisher": "Fixture Publisher",
        },
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _fetcher(url: str, max_bytes: int) -> LiteratureFetchResponse:
    body = _crossref_body(url)
    assert len(body) <= max_bytes
    return LiteratureFetchResponse(
        status=200,
        final_url=url,
        headers={"content-type": "application/json; charset=utf-8"},
        body=body,
    )


def test_online_literature_can_only_create_candidate_dynamics_knowledge(tmp_path) -> None:
    outcome = capture_dynamics_knowledge(
        tmp_path,
        fetcher=_fetcher,
        captured_at=NOW,
        run_id="dynamics_knowledge_fixture",
    )
    assert len(outcome.snapshots) == 3
    assert all(snapshot.trust_class == "untrusted_bibliographic_data" for snapshot in outcome.snapshots)
    assert outcome.bundle.status == "candidate_requires_hidden_dynamics_validation"
    assert "identifiability_before_parameter_claim" in outcome.bundle.exact_design_rules
    assert verify_dynamics_knowledge_run(outcome.store.run_directory)


def test_literature_fetch_rejects_redirect_and_wrong_doi() -> None:
    contract = default_dynamics_source_contracts(created_at=NOW)[0]

    def redirected(url: str, max_bytes: int) -> LiteratureFetchResponse:
        return LiteratureFetchResponse(
            status=200,
            final_url="https://example.com/injected",
            headers={"content-type": "application/json"},
            body=_crossref_body(url),
        )

    with pytest.raises(RuntimeError, match="redirected"):
        fetch_literature_source(contract, fetcher=redirected, retrieved_at=NOW)

    def wrong_doi(url: str, max_bytes: int) -> LiteratureFetchResponse:
        payload = json.loads(_crossref_body(url))
        payload["message"]["DOI"] = "10.0000/not-the-contracted-paper"
        return LiteratureFetchResponse(
            status=200,
            final_url=url,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
        )

    with pytest.raises(ValueError, match="DOI"):
        fetch_literature_source(contract, fetcher=wrong_doi, retrieved_at=NOW)


def test_tampered_dynamics_literature_breaks_replay(tmp_path) -> None:
    outcome = capture_dynamics_knowledge(
        tmp_path,
        fetcher=_fetcher,
        captured_at=NOW,
        run_id="dynamics_knowledge_tamper",
    )
    ref = next(
        ref
        for ref in outcome.manifest.artifact_refs
        if ref.kind == "literature_evidence_snapshot_v24"
    )
    path = outcome.store.run_directory / ref.relative_path
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["title"] += " tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    assert not verify_dynamics_knowledge_run(outcome.store.run_directory)
