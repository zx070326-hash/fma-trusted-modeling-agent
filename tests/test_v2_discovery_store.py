from __future__ import annotations

import json
import sys

import pytest

from fma.v2.capacity_planning import (
    FIXTURE_TIME,
    capacity_brief_snapshot,
    capacity_planning_mission,
    capacity_problem_hypothesis_draft,
    run_capacity_discovery_fixture,
)
from fma.v2.discovery import ProblemHypothesisDraft
from fma.v2.discovery_store import DiscoveryRunStore
from fma.v2.schemas import DiscoveryRejectionReceipt


def _started_store(tmp_path) -> tuple[DiscoveryRunStore, object, object]:
    mission_contract = capacity_planning_mission()
    snapshot = capacity_brief_snapshot()
    store = DiscoveryRunStore(tmp_path, run_id="discovery-test")
    store.start(mission_contract, occurred_at=FIXTURE_TIME)
    store.ingest_evidence(snapshot, occurred_at=FIXTURE_TIME)
    return store, mission_contract, snapshot


def test_discovery_run_is_replayable_and_admission_is_bound_to_prior_events(tmp_path):
    store, mission_contract, snapshot = _started_store(tmp_path)
    outcome = store.submit_and_admit(
        snapshot,
        capacity_problem_hypothesis_draft(mission_contract, snapshot),
        occurred_at=FIXTURE_TIME,
    )

    assert outcome.status == "admitted"
    assert outcome.hypothesis is not None
    assert store.verify()
    state = store.project_state()
    assert state.event_count == 4
    assert state.evidence_snapshot_hashes == [snapshot.snapshot_hash]
    assert state.admitted_hypothesis_hashes == [outcome.hypothesis.hypothesis_hash]

    reopened = DiscoveryRunStore.open_existing(store.run_directory)
    assert reopened.verify()
    assert reopened.project_state() == state


def test_rejected_draft_receives_a_non_sensitive_terminal_receipt(tmp_path):
    store, mission_contract, snapshot = _started_store(tmp_path)
    mission_hash = mission_contract.mission.mission_spec_hash
    assert mission_hash is not None
    rejected = ProblemHypothesisDraft(
        draft_id="wrong_evidence_draft",
        mission_spec_hash=mission_hash,
        evidence_snapshot_hashes=["0" * 64],
        statement="This candidate cites a snapshot that was not actually supplied.",
        observed_symptoms=["The stated evidence hash is deliberately mismatched"],
        proposed_value="Exercise the rejection path without certifying a hypothesis",
    )

    outcome = store.submit_and_admit(snapshot, rejected, occurred_at=FIXTURE_TIME)

    assert outcome.status == "rejected"
    assert outcome.rejection_receipt is not None
    assert outcome.rejection_receipt.rejection_code == "admission_denied"
    state = store.project_state()
    assert state.admitted_hypothesis_hashes == []
    assert state.rejected_draft_artifact_hashes == [outcome.draft_ref.sha256]
    assert state.event_count == 4
    assert store.verify()


def test_replay_refuses_a_rejection_receipt_for_a_draft_that_would_be_admitted(tmp_path):
    store, mission_contract, snapshot = _started_store(tmp_path)
    snapshot_ref = store._find_snapshot_ref(snapshot)
    draft = capacity_problem_hypothesis_draft(mission_contract, snapshot)
    draft_ref = store.put_artifact("problem_hypothesis_draft", draft)
    store._append_event(
        "problem_draft_submitted", [snapshot_ref, draft_ref], occurred_at=FIXTURE_TIME
    )
    receipt = DiscoveryRejectionReceipt.seal(
        receipt_id="forged_valid_rejection",
        run_id=store.run_id,
        draft_artifact_hash=draft_ref.sha256,
        evidence_snapshot_hash=snapshot.snapshot_hash,
        rejection_code="admission_denied",
        created_at=FIXTURE_TIME,
    )
    receipt_ref = store.put_artifact("discovery_rejection_receipt", receipt)
    store._append_event(
        "problem_draft_rejected",
        [snapshot_ref, draft_ref, receipt_ref],
        rejection_code="admission_denied",
        occurred_at=FIXTURE_TIME,
    )

    with pytest.raises(RuntimeError, match="would pass replayed admission"):
        store.project_state()
    assert not store.verify()


def test_artifact_or_event_tampering_fails_closed(tmp_path):
    store, mission_contract, snapshot = _started_store(tmp_path)
    outcome = store.submit_and_admit(
        snapshot,
        capacity_problem_hypothesis_draft(mission_contract, snapshot),
        occurred_at=FIXTURE_TIME,
    )
    assert outcome.hypothesis_ref is not None

    artifact_path = store.run_directory / outcome.hypothesis_ref.relative_path
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["payload"]["statement"] = "tampered hypothesis statement"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert not store.verify()
    with pytest.raises(RuntimeError, match="integrity"):
        DiscoveryRunStore.open_existing(store.run_directory).project_state()

    clean_store, clean_mission, clean_snapshot = _started_store(tmp_path / "event")
    clean_store.submit_and_admit(
        clean_snapshot,
        capacity_problem_hypothesis_draft(clean_mission, clean_snapshot),
        occurred_at=FIXTURE_TIME,
    )
    lines = clean_store.event_path.read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    last["event_type"] = "problem_draft_rejected"
    lines[-1] = json.dumps(last)
    clean_store.event_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not clean_store.verify()
    with pytest.raises((RuntimeError, ValueError)):
        DiscoveryRunStore.open_existing(clean_store.run_directory)


def test_capacity_discovery_fixture_and_cli_expose_metadata_not_raw_brief(
    tmp_path, capsys, monkeypatch
):
    store, outcome = run_capacity_discovery_fixture(tmp_path / "direct")
    assert outcome.status == "admitted"
    assert store.project_state().event_count == 4

    from fma.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["fma", "v2-discovery-fixture", "--output", str(tmp_path / "cli")],
    )
    assert main() == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["v2_protocol"] == "experimental_discovery_ledger_v2"
    assert payload["admission_status"] == "admitted"
    assert payload["event_chain_verified"] is True
    assert "Demand is five product units" not in output
