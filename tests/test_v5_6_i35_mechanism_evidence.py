from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fma.hashing import sha256_value
from fma.v5_6.hybrid_ode import (
    HybridReplayReceiptV56,
    HybridScientificBundleV56,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = (
    ROOT
    / "experiments"
    / "iteration_35"
    / "mechanism_suite_v56"
)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v56_mechanism_suite_manifest_binds_every_artifact() -> None:
    manifest = _load_json(SUITE / "MANIFEST.json")
    assert isinstance(manifest, dict)
    assert manifest["schema_version"] == "5.6-mechanism-suite-manifest"
    assert manifest["suite_passed"] is True
    assert manifest["scientific_qualification_granted"] is False
    assert manifest["real_world_action_authorized"] is False
    assert manifest["manifest_hash"] == sha256_value(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_hash"
        }
    )
    assert manifest["runner_sha256"] == hashlib.sha256(
        (
            ROOT
            / "experiments"
            / "iteration_35"
            / "run_v56_mechanism_suite.py"
        ).read_bytes()
    ).hexdigest()
    assert manifest["adapter_sha256"] == hashlib.sha256(
        (ROOT / "fma" / "v5_6" / "hybrid_ode.py").read_bytes()
    ).hexdigest()

    declared = {
        str(item["path"]): item
        for item in manifest["files"]
    }
    actual = {
        path.name
        for path in SUITE.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    }
    assert set(declared) == actual
    for name, entry in declared.items():
        payload = (SUITE / name).read_bytes()
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_v56_mechanism_suite_preserves_success_and_rejection_evidence() -> None:
    result = _load_json(SUITE / "MECHANISM_SUITE_RESULTS.json")
    assert isinstance(result, dict)
    assert result["result_hash"] == sha256_value(
        {
            key: value
            for key, value in result.items()
            if key != "result_hash"
        }
    )
    assert result["case_count"] == 6
    assert result["suite_passed"] is True
    assert result["fixture_scientific_qualification_granted"] is False
    assert result["retrospective_scientific_qualification_granted"] is False
    assert result["causal_mechanism_identified"] is False
    assert result["real_world_action_authorized"] is False

    rows = {
        str(row["case_id"]): row
        for row in result["rows"]
    }
    assert set(rows) == {
        "stationary-ar1-recovery",
        "iid-no-recovery",
        "near-unit-root-reject",
        "training-structural-break-reject",
        "validation-structural-break-reject",
        "i34-retrospective-reject",
    }
    assert all(row["expected_outcome_matched"] is True for row in rows.values())
    assert rows["stationary-ar1-recovery"]["selected_candidate_id"] == (
        "logistic.ar1_residual"
    )
    assert rows["stationary-ar1-recovery"]["scientific_acceptance"] is True
    assert rows["iid-no-recovery"]["recovery_triggered"] is False
    assert rows["iid-no-recovery"]["scientific_acceptance"] is True
    for case_id in (
        "near-unit-root-reject",
        "training-structural-break-reject",
        "validation-structural-break-reject",
        "i34-retrospective-reject",
    ):
        assert rows[case_id]["l3_status"] == "FAIL"
        assert rows[case_id]["scientific_acceptance"] is False


def test_v56_mechanism_bundles_and_independent_receipts_replay() -> None:
    result = _load_json(SUITE / "MECHANISM_SUITE_RESULTS.json")
    assert isinstance(result, dict)
    for row in result["rows"]:
        case_id = str(row["case_id"])
        bundle = HybridScientificBundleV56.model_validate_json(
            (SUITE / f"{case_id}.bundle.json").read_text(encoding="utf-8")
        )
        receipts_payload = _load_json(
            SUITE / f"{case_id}.replay_receipts.json"
        )
        assert isinstance(receipts_payload, list)
        receipts = [
            HybridReplayReceiptV56.model_validate(item)
            for item in receipts_payload
        ]

        assert bundle.bundle_hash == bundle.content_hash()
        assert bundle.bundle_hash == row["bundle_hash"]
        assert bundle.graph.graph_hash == row["graph_hash"]
        assert bundle.fixture_only == row["fixture_only"]
        assert bundle.scientific_qualification_granted is False
        assert bundle.real_world_action_authorized is False
        assert len({item.process_id for item in receipts}) == 2
        assert len({item.deterministic_output_hash for item in receipts}) == 1
        assert all(item.fresh_process is True for item in receipts)
        assert all(item.receipt_hash == item.content_hash() for item in receipts)
        assert [item.receipt_hash for item in receipts] == (
            bundle.replay_receipt_hashes
        )
        assert bundle.levels[0].status == "PASS"
        if bool(row["scientific_acceptance"]):
            assert all(item.status == "PASS" for item in bundle.levels)
        else:
            assert bundle.levels[3].status == "FAIL"
