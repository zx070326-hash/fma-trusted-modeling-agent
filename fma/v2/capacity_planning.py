from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fma.codex_driver import CodexCLIConfig
from fma.controller import CandidateRunOutcome, ModelingAgent
from fma.hashing import sha256_value
from fma.schemas import OptimizationModelIR, ProblemContract

from .bridge import freeze_legacy_contract
from .codex_discovery import CodexProblemDiscoveryExplorer
from .discovery import ProblemDiscoveryHarness, ProblemHypothesisDraft
from .discovery import ProblemDiscoveryProposal
from .discovery_store import DiscoveryRunStore, ProblemAdmissionOutcome
from .schemas import (
    ApprovalRecord,
    ConceptualModelIR,
    EvidencePedigree,
    EvidenceSnapshot,
    FrozenLegacyBinding,
    MissionContract,
    MissionSpec,
    PrivateAcceptanceBundle,
    ProblemContractProposal,
    ProblemHypothesis,
)


FIXTURE_TIME = datetime(2026, 7, 20, tzinfo=timezone.utc)
CAPACITY_PUBLIC_BRIEF = """Operations brief:
Demand is five product units for this planning period. Regular production can make at most three units and costs two cost units per unit. Overtime production costs five cost units per unit. No inventory, shortage, or external supply is available. Production quantities must be nonnegative integers.
"""


class CapacityPlanningTestAuthority:
    """Fixture-only independent authority for the first V2.0 vertical slice."""

    authority_id = "capacity_fixture_authority"

    def issue(self, proposal: ProblemContractProposal) -> PrivateAcceptanceBundle:
        proposal.assert_sealed()
        if proposal.contract_id != "capacity_plan_v2":
            raise ValueError("capacity fixture authority only accepts capacity_plan_v2")
        expected_decisions = {"regular_units", "overtime_units"}
        if {decision.decision_id for decision in proposal.decisions} != expected_decisions:
            raise ValueError("capacity fixture proposal has unexpected decisions")
        assert proposal.proposal_hash is not None
        return PrivateAcceptanceBundle.seal(
            bundle_id="capacity_plan_v2_private_tests",
            proposal_hash=proposal.proposal_hash,
            authority_id=self.authority_id,
            issued_at=FIXTURE_TIME,
            acceptance_tests=[
                {
                    "test_id": "known_global_optimum",
                    "kind": "known_optimum",
                    "expected_objective": 16,
                    "source_ref": "capacity_fixture:independent_enumeration",
                },
                {
                    "test_id": "known_feasible_optimum",
                    "kind": "assignment_case",
                    "assignment": {"regular_units": 3, "overtime_units": 2},
                    "expected_feasible": True,
                    "expected_objective": 16,
                    "source_ref": "capacity_fixture:independent_hand_check",
                },
                {
                    "test_id": "regular_capacity_counterexample",
                    "kind": "assignment_case",
                    "assignment": {"regular_units": 4, "overtime_units": 1},
                    "expected_feasible": False,
                    "source_ref": "capacity_fixture:capacity_counterexample",
                },
                {
                    "test_id": "demand_counterexample",
                    "kind": "assignment_case",
                    "assignment": {"regular_units": 3, "overtime_units": 1},
                    "expected_feasible": False,
                    "source_ref": "capacity_fixture:demand_counterexample",
                },
            ],
        )


def capacity_planning_mission() -> MissionContract:
    mission = MissionSpec.seal(
        mission_id="capacity_planning_mission",
        version=1,
        knowledge_objectives=["Identify a feasible least-cost capacity plan"],
        intended_decisions=["Choose regular and overtime production quantities"],
        stakeholders_and_value_owners=["fixture_operations_owner"],
        spatial_temporal_scope="One synthetic factory during one planning period",
        approved_evidence_sources=["capacity_fixture:public_brief"],
        resource_budget={"solver_calls": 1},
        validation_budget_reserve={"independent_oracle_calls": 1},
        allowed_actions=["local_compute", "write_local_run_artifacts"],
        forbidden_actions=["external_action"],
        stopping_policy={"when": "validated_or_needs_evidence"},
        created_at=FIXTURE_TIME,
    )
    assert mission.mission_spec_hash is not None
    approval = ApprovalRecord.seal(
        approval_id="capacity_planning_mission_approval",
        mission_spec_hash=mission.mission_spec_hash,
        sequence=1,
        policy_version="capacity_fixture_policy_v1",
        decision="approved",
        approved_scope={"allowed_actions": ["local_compute", "write_local_run_artifacts"]},
        approver_ref="fixture_operations_owner",
        issued_at=FIXTURE_TIME,
    )
    contract = MissionContract(mission=mission, approval=approval)
    contract.assert_active(FIXTURE_TIME)
    return contract


def capacity_codex_discovery_mission() -> MissionContract:
    """Fixture mission that explicitly permits one isolated Codex draft call."""

    mission = MissionSpec.seal(
        mission_id="capacity_codex_discovery_mission",
        version=1,
        knowledge_objectives=["Draft a source-bound capacity-planning problem hypothesis"],
        intended_decisions=["Decide whether a formal capacity model is warranted"],
        stakeholders_and_value_owners=["fixture_operations_owner"],
        spatial_temporal_scope="One synthetic factory during one planning period",
        approved_evidence_sources=["capacity_fixture:public_brief"],
        resource_budget={"codex_calls": 1},
        validation_budget_reserve={"admission_checks": 1},
        allowed_actions=[
            "codex_cli_inference",
            "local_compute",
            "write_local_run_artifacts",
        ],
        forbidden_actions=["external_action"],
        stopping_policy={"when": "draft_recorded_or_no_result"},
        created_at=FIXTURE_TIME,
    )
    assert mission.mission_spec_hash is not None
    approval = ApprovalRecord.seal(
        approval_id="capacity_codex_discovery_approval",
        mission_spec_hash=mission.mission_spec_hash,
        sequence=1,
        policy_version="capacity_codex_discovery_fixture_policy_v1",
        decision="approved",
        approved_scope={
            "allowed_actions": [
                "codex_cli_inference",
                "local_compute",
                "write_local_run_artifacts",
            ]
        },
        approver_ref="fixture_operations_owner",
        issued_at=FIXTURE_TIME,
    )
    contract = MissionContract(mission=mission, approval=approval)
    contract.assert_active(FIXTURE_TIME)
    return contract


def capacity_brief_snapshot() -> EvidenceSnapshot:
    return EvidenceSnapshot.seal(
        snapshot_id="capacity_plan_public_brief",
        pedigree=EvidencePedigree(
            source_kind="fixture",
            source_ref="capacity_fixture:public_brief",
            collector="fixture",
            collected_at=FIXTURE_TIME,
            source_content_hash=sha256_value({"raw_text": CAPACITY_PUBLIC_BRIEF}),
        ),
        content_type="text/plain",
        raw_text=CAPACITY_PUBLIC_BRIEF,
    )


def capacity_problem_hypothesis(
    mission_contract: MissionContract,
) -> ProblemHypothesis:
    mission_contract.assert_active(FIXTURE_TIME)
    snapshot = capacity_brief_snapshot()
    draft = capacity_problem_hypothesis_draft(mission_contract, snapshot)
    return ProblemDiscoveryHarness.admit(
        mission_contract,
        snapshot,
        draft,
        admitted_at=FIXTURE_TIME,
    )


def capacity_problem_hypothesis_draft(
    mission_contract: MissionContract,
    snapshot: EvidenceSnapshot | None = None,
) -> ProblemHypothesisDraft:
    """Fixture draft only; admission remains owned by the discovery harness."""

    mission_contract.assert_active(FIXTURE_TIME)
    snapshot = snapshot or capacity_brief_snapshot()
    mission_hash = mission_contract.mission.mission_spec_hash
    snapshot_hash = snapshot.snapshot_hash
    assert mission_hash is not None
    assert snapshot_hash is not None
    return ProblemHypothesisDraft(
        draft_id="capacity_shortfall_hypothesis",
        mission_spec_hash=mission_hash,
        evidence_snapshot_hashes=[snapshot_hash],
        statement="Regular capacity alone cannot meet the stated demand, so overtime is needed.",
        observed_symptoms=["Demand is five units while regular capacity is three units"],
        proposed_value="Minimize production cost while meeting hard demand",
        assumptions=["Production quantities are nonnegative integers"],
        open_questions=["Does the one-period boundary omit inventory or backlog costs?"],
    )


def run_capacity_discovery_fixture(
    output_root: str | Path,
) -> tuple[DiscoveryRunStore, ProblemAdmissionOutcome]:
    """Exercise the V2 discovery ledger without invoking an LLM or solver."""

    mission_contract = capacity_planning_mission()
    snapshot = capacity_brief_snapshot()
    store = DiscoveryRunStore(output_root, run_id="capacity-discovery")
    store.start(mission_contract, occurred_at=FIXTURE_TIME)
    store.ingest_evidence(snapshot, occurred_at=FIXTURE_TIME)
    outcome = store.submit_and_admit(
        snapshot,
        capacity_problem_hypothesis_draft(mission_contract, snapshot),
        occurred_at=FIXTURE_TIME,
    )
    if outcome.status != "admitted" or not store.verify():
        raise RuntimeError("capacity discovery fixture did not produce a verified admission")
    return store, outcome


def run_codex_capacity_discovery_fixture(
    output_root: str | Path,
    config: CodexCLIConfig,
    *,
    process_runner: Any | None = None,
    cli_locator: Any | None = None,
    prompt_guard: Any | None = None,
) -> tuple[DiscoveryRunStore, ProblemDiscoveryProposal, ProblemAdmissionOutcome | None]:
    """One explicitly authorized Codex draft call against the synthetic brief.

    The fixture ends after discovery admission; it does not construct a formal
    model, invoke the solver, or make a real-world recommendation.
    """

    mission_contract = capacity_codex_discovery_mission()
    snapshot = capacity_brief_snapshot()
    store = DiscoveryRunStore(output_root, run_id="capacity-codex-discovery")
    store.start(mission_contract, occurred_at=FIXTURE_TIME)
    store.ingest_evidence(snapshot, occurred_at=FIXTURE_TIME)
    explorer = CodexProblemDiscoveryExplorer(
        store,
        config,
        process_runner=process_runner,
        cli_locator=cli_locator,
        prompt_guard=prompt_guard,
    )
    try:
        proposal = explorer.propose(
            store.build_problem_discovery_context(snapshot, at=FIXTURE_TIME)
        )
    finally:
        explorer.close()
    if proposal.status != "proposed":
        return store, proposal, None
    assert proposal.draft is not None
    assert proposal.provider_observation_ref is not None
    outcome = store.submit_and_admit(
        snapshot,
        proposal.draft,
        provider_observation_ref=proposal.provider_observation_ref,
    )
    return store, proposal, outcome


def capacity_planning_proposal(mission_contract: MissionContract) -> ProblemContractProposal:
    mission_contract.assert_active(FIXTURE_TIME)
    mission = mission_contract.mission
    assert mission.mission_spec_hash is not None
    hypothesis = capacity_problem_hypothesis(mission_contract)
    assert hypothesis.hypothesis_hash is not None
    conceptual = ConceptualModelIR.seal(
        model_id="capacity_cost_conceptual_model",
        problem_hypothesis_hash=hypothesis.hypothesis_hash,
        entities=["regular production", "overtime production", "demand"],
        mechanisms=["Regular production is cheaper but capacity-limited"],
        assumptions=["No inventory, shortages, or external supply are permitted"],
        observables=["demand", "regular capacity", "unit costs"],
        boundary_conditions=["One planning period", "Integer production quantities"],
        created_at=FIXTURE_TIME,
    )
    assert conceptual.conceptual_model_hash is not None
    return ProblemContractProposal.seal(
        proposal_id="capacity_plan_v2_public_proposal",
        mission_spec_hash=mission.mission_spec_hash,
        problem_hypothesis_hash=hypothesis.hypothesis_hash,
        conceptual_model_hash=conceptual.conceptual_model_hash,
        contract_id="capacity_plan_v2",
        contract_version=1,
        question="How many regular and overtime units minimize cost while meeting demand?",
        system_boundary="One synthetic factory with regular and overtime production only",
        decision_horizon="One planning period",
        decisions=[
            {
                "decision_id": "regular_units",
                "statement": "Number of regular production units",
                "kind": "integer",
                "unit": "product_unit",
                "lower_bound": 0,
                "upper_bound": 3,
                "source_ref": "capacity_fixture:public_brief",
            },
            {
                "decision_id": "overtime_units",
                "statement": "Number of overtime production units",
                "kind": "integer",
                "unit": "product_unit",
                "lower_bound": 0,
                "upper_bound": 5,
                "source_ref": "capacity_fixture:public_brief",
            },
        ],
        clauses=[
            {
                "clause_id": "minimize_total_cost",
                "kind": "objective",
                "statement": "Minimize total regular and overtime production cost",
                "unit": "cost_unit",
                "source_ref": "capacity_fixture:public_brief",
                "acceptance_criterion": "Return the least-cost feasible integer plan",
            },
            {
                "clause_id": "meet_demand",
                "kind": "hard_constraint",
                "statement": "Regular plus overtime production must meet at least five units of demand",
                "unit": "product_unit",
                "source_ref": "capacity_fixture:public_brief",
                "acceptance_criterion": "Require regular_units + overtime_units >= 5",
            },
            {
                "clause_id": "regular_capacity",
                "kind": "hard_constraint",
                "statement": "Regular production cannot exceed three units",
                "unit": "product_unit",
                "source_ref": "capacity_fixture:public_brief",
                "acceptance_criterion": "Require regular_units <= 3",
            },
            {
                "clause_id": "integer_production",
                "kind": "assumption",
                "statement": "Both production quantities are nonnegative integers",
                "unit": "unitless",
                "source_ref": "capacity_fixture:assumption",
                "acceptance_criterion": "Use nonnegative integer variables",
            },
        ],
        public_facts=[
            {
                "fact_id": "regular_unit_cost",
                "statement": "Each regular unit costs two cost units",
                "value": 2,
                "unit": "cost_unit_per_product_unit",
                "source_ref": "capacity_fixture:public_brief",
            },
            {
                "fact_id": "overtime_unit_cost",
                "statement": "Each overtime unit costs five cost units",
                "value": 5,
                "unit": "cost_unit_per_product_unit",
                "source_ref": "capacity_fixture:public_brief",
            },
            {
                "fact_id": "demand_units",
                "statement": "Demand is five product units",
                "value": 5,
                "unit": "product_unit",
                "source_ref": "capacity_fixture:public_brief",
            },
            {
                "fact_id": "regular_capacity_units",
                "statement": "Regular capacity is three product units",
                "value": 3,
                "unit": "product_unit",
                "source_ref": "capacity_fixture:public_brief",
            },
        ],
        permitted_actions=["local_compute", "write_local_run_artifacts"],
        forbidden_actions=["external_action"],
        risk_level="A1",
        created_at=FIXTURE_TIME,
    )


def build_capacity_planning_fixture() -> FrozenLegacyBinding:
    mission_contract = capacity_planning_mission()
    proposal = capacity_planning_proposal(mission_contract)
    bundle = CapacityPlanningTestAuthority().issue(proposal)
    return freeze_legacy_contract(proposal, bundle)


def capacity_planning_ir(contract: ProblemContract) -> OptimizationModelIR:
    contract.assert_frozen()
    assert contract.frozen_hash is not None
    return OptimizationModelIR.seal(
        candidate_id="capacity_plan_v2_milp",
        contract_hash=contract.frozen_hash,
        skeleton_id="capacity_planning_milp",
        lineage={
            "root_kind": "fixture",
            "parent_candidate_ids": [],
            "rationale": "Deterministic V2.0 bridge fixture; it does not measure model generation quality",
        },
        variables=[
            {
                "name": "regular_units",
                "kind": "integer",
                "lower_bound": 0,
                "upper_bound": 3,
                "unit": "product_unit",
            },
            {
                "name": "overtime_units",
                "kind": "integer",
                "lower_bound": 0,
                "upper_bound": 5,
                "unit": "product_unit",
            },
        ],
        objective={
            "sense": "minimize",
            "coefficients": {"regular_units": 2, "overtime_units": 5},
            "constant": 0,
            "unit": "cost_unit",
            "contract_clause_ids": ["minimize_total_cost"],
        },
        constraints=[
            {
                "constraint_id": "demand",
                "coefficients": {"regular_units": 1, "overtime_units": 1},
                "sense": ">=",
                "rhs": 5,
                "lhs_unit": "product_unit",
                "rhs_unit": "product_unit",
                "contract_clause_ids": ["meet_demand"],
            },
            {
                "constraint_id": "regular_limit",
                "coefficients": {"regular_units": 1},
                "sense": "<=",
                "rhs": 3,
                "lhs_unit": "product_unit",
                "rhs_unit": "product_unit",
                "contract_clause_ids": ["regular_capacity"],
            },
        ],
        validation_obligations=[
            "Bind the IR to the frozen legacy contract",
            "Run private acceptance tests only in the trusted verifier",
            "Confirm the optimum by bounded integer enumeration",
        ],
    )


def run_capacity_planning_fixture(output_root: str | Path) -> CandidateRunOutcome:
    """Run only the existing trusted kernel after the V2 bridge has frozen it."""

    binding = build_capacity_planning_fixture()
    return ModelingAgent(output_root).assess_candidate(
        binding.contract,
        capacity_planning_ir(binding.contract),
    )
