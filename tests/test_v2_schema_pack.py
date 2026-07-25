from __future__ import annotations

import json
import sys
from datetime import timedelta

import pytest
from pydantic import ValidationError

from fma.v2.bridge import acceptance_commitment, freeze_legacy_contract
from fma.v2.capacity_planning import (
    FIXTURE_TIME,
    CapacityPlanningTestAuthority,
    build_capacity_planning_fixture,
    capacity_planning_mission,
    capacity_planning_proposal,
    run_capacity_planning_fixture,
)
from fma.v2.schemas import (
    ApprovalRecord,
    EvidenceUseEvent,
    EvidenceUseReservation,
    MissionContract,
    TransactionEvent,
    TransactionProposal,
)


def test_v2_public_proposal_is_sealed_and_has_no_private_acceptance_tests():
    mission_contract = capacity_planning_mission()
    proposal = capacity_planning_proposal(mission_contract)

    proposal.assert_sealed()
    public = proposal.public_legacy_payload()
    assert "acceptance_tests" not in public
    assert "proposal_hash" not in public
    assert public["contract_id"] == "capacity_plan_v2"

    changed = proposal.model_dump()
    changed["question"] = "A changed capacity-planning question invalidates the old hash"
    with pytest.raises(ValidationError, match="proposal_hash"):
        proposal.__class__.model_validate(changed)


def test_bridge_requires_a_private_bundle_bound_to_the_exact_proposal():
    mission_contract = capacity_planning_mission()
    proposal = capacity_planning_proposal(mission_contract)
    authority = CapacityPlanningTestAuthority()
    bundle = authority.issue(proposal)
    commitment = acceptance_commitment(bundle)
    binding = freeze_legacy_contract(proposal, bundle)

    binding.assert_sealed()
    assert binding.contract.frozen_hash
    assert commitment.proposal_hash == proposal.proposal_hash
    assert commitment.acceptance_bundle_hash == bundle.acceptance_bundle_hash
    assert "acceptance_tests" not in commitment.model_dump()

    other_data = proposal.model_dump(exclude={"proposal_hash"})
    other_data["proposal_id"] = "capacity_plan_v2_other_proposal"
    other = proposal.__class__.seal(**other_data)
    with pytest.raises(ValueError, match="another proposal"):
        freeze_legacy_contract(other, bundle)


def test_mission_approval_is_hash_bound_and_expiry_is_checked():
    mission_contract = capacity_planning_mission()
    mission_contract.assert_active(FIXTURE_TIME)

    approval_data = mission_contract.approval.model_dump(exclude={"approval_record_hash"})
    approval_data["expires_at"] = FIXTURE_TIME + timedelta(seconds=1)
    approval = ApprovalRecord.seal(**approval_data)
    active = MissionContract(mission=mission_contract.mission, approval=approval)
    with pytest.raises(ValueError, match="expired"):
        active.assert_active(FIXTURE_TIME + timedelta(seconds=2))

    wrong = approval.model_dump()
    wrong["mission_spec_hash"] = "0" * 64
    wrong.pop("approval_record_hash")
    wrong_approval = ApprovalRecord.seal(**wrong)
    with pytest.raises(ValidationError, match="different mission"):
        MissionContract(mission=mission_contract.mission, approval=wrong_approval)


def test_transaction_and_evidence_lifecycles_use_append_only_events():
    mission_contract = capacity_planning_mission()
    mission_hash = mission_contract.mission.mission_spec_hash
    assert mission_hash is not None
    proposal = TransactionProposal.seal(
        proposal_id="observation_ingest",
        mission_spec_hash=mission_hash,
        base_graph_snapshot_hash="1" * 64,
        operation="ingest_observation",
        rationale_summary="Record a newly received capacity observation",
        created_at=FIXTURE_TIME,
    )
    assert proposal.proposal_hash is not None
    event = TransactionEvent.seal(
        proposal_hash=proposal.proposal_hash,
        sequence=1,
        event_type="authorized",
        payload_hash="2" * 64,
        occurred_at=FIXTURE_TIME,
    )
    assert event.event_hash is not None
    with pytest.raises(ValidationError, match="needs a predecessor"):
        TransactionEvent.seal(
            proposal_hash=proposal.proposal_hash,
            sequence=2,
            event_type="executed",
            payload_hash="3" * 64,
            occurred_at=FIXTURE_TIME,
        )

    reservation = EvidenceUseReservation.seal(
        entry_id="validation_reservation",
        evidence_hash="4" * 64,
        claim_hash="5" * 64,
        candidate_lineage_hash="6" * 64,
        role="validation",
        campaign_id="capacity_campaign",
        reserved_at=FIXTURE_TIME,
    )
    assert reservation.reservation_hash is not None
    consumed = EvidenceUseEvent.seal(
        reservation_hash=reservation.reservation_hash,
        sequence=1,
        event_type="consumed",
        occurred_at=FIXTURE_TIME,
    )
    assert consumed.event_hash is not None


def test_capacity_fixture_reaches_legacy_kernel_without_exposing_tests(tmp_path):
    binding = build_capacity_planning_fixture()
    assert binding.contract.frozen_hash
    outcome = run_capacity_planning_fixture(tmp_path)

    assert outcome.decision.status == "validated"
    assert outcome.solution.values == {"regular_units": 3.0, "overtime_units": 2.0}
    assert outcome.solution.objective_value == pytest.approx(16.0)
    assert outcome.validation.checks["contract_acceptance"].status == "pass"


def test_v2_capacity_fixture_cli_reports_hashes_but_not_private_tests(
    tmp_path, capsys, monkeypatch
):
    from fma.__main__ import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["fma", "v2-capacity-fixture", "--output", str(tmp_path)],
    )
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["v2_protocol"] == "experimental_schema_pack_v2"
    assert payload["decision_status"] == "validated"
    assert "acceptance_tests" not in payload
