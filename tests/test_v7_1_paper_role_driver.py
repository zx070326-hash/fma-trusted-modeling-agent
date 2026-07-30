from __future__ import annotations

from fma.v7_1.paper_role_driver import _canonicalize_native_draft


def test_native_writer_normalizes_only_set_like_identifier_fields() -> None:
    raw = {
        "metadata": {
            "writer_context_ids": ["writer-b", "writer-a", "writer-a"],
            "authors": ["Second Author", "First Author"],
        },
        "claim_ledger": {
            "claims": [
                {
                    "claim_id": "claim.b",
                    "evidence_ids": ["ev.2", "ev.1", "ev.1"],
                    "numeric_token_ids": ["value.2", "value.1"],
                    "citation_ids": [],
                },
                {
                    "claim_id": "claim.a",
                    "evidence_ids": ["ev.1"],
                    "numeric_token_ids": [],
                    "citation_ids": [],
                },
            ]
        },
        "citations": {"citations": []},
        "figures": {"figures": []},
        "tables": {"tables": []},
    }

    normalized = _canonicalize_native_draft(raw, "writer")

    assert normalized["metadata"]["writer_context_ids"] == [
        "writer-a",
        "writer-b",
    ]
    assert normalized["metadata"]["authors"] == [
        "Second Author",
        "First Author",
    ]
    assert [
        claim["claim_id"] for claim in normalized["claim_ledger"]["claims"]
    ] == ["claim.a", "claim.b"]
    assert normalized["claim_ledger"]["claims"][1]["evidence_ids"] == [
        "ev.1",
        "ev.2",
    ]
    assert raw["claim_ledger"]["claims"][0]["evidence_ids"] == [
        "ev.2",
        "ev.1",
        "ev.1",
    ]


def test_native_reviewer_normalization_is_deterministic() -> None:
    semantic = _canonicalize_native_draft(
        {
            "reviewed_claim_ids": ["claim.b", "claim.a", "claim.a"],
            "findings": [
                {
                    "finding_id": "finding.b",
                    "claim_ids": ["claim.b", "claim.a"],
                    "evidence_ids": ["ev.b", "ev.a"],
                },
                {
                    "finding_id": "finding.a",
                    "claim_ids": [],
                    "evidence_ids": [],
                },
            ],
        },
        "semantic",
    )
    layout = _canonicalize_native_draft(
        {
            "pages_reviewed": [3, 1, 2, 2],
            "findings": ["widow", "overflow", "widow"],
        },
        "layout",
    )

    assert semantic["reviewed_claim_ids"] == ["claim.a", "claim.b"]
    assert [
        finding["finding_id"] for finding in semantic["findings"]
    ] == ["finding.a", "finding.b"]
    assert semantic["findings"][1]["evidence_ids"] == ["ev.a", "ev.b"]
    assert layout == {
        "pages_reviewed": [1, 2, 3],
        "findings": ["overflow", "widow"],
    }
