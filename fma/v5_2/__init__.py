"""Additive V5.2 graph recovery, candidate governance, and domain expansion.

V5.2 consumes V5/V5.1 artifacts without changing their interpretation.
"""

from .candidate_space import (
    CandidateAdmissionAuthorityV52,
    CandidateAdmissionPolicyV52,
    CandidateAdmissionReceiptV52,
    GeneratedCandidateV52,
    GovernedCandidateRegistryV52,
)
from .evolution_controller import (
    EvolutionDecisionV52,
    EvolutionProposalV52,
    GraphEvolutionControllerV52,
    RecoveryAuthorityV52,
    RecoveryBudgetV52,
    RecoveryEvidenceV52,
    RecoveryStateV52,
)
from .private_qualification import (
    ExternalHostAttestationV52,
    HostAttestationAuthorityV52,
    LocalPrivateRunReceiptV52,
    PrivateEvaluationRequestV52,
    PrivatePromotionAuthorityV52,
    PrivateQualificationReceiptV52,
    PrivateWorkerAuthorityV52,
    PrivateWorkerReceiptV52,
    run_local_private_worker_v52,
)
from .ode_system import (
    ODELevelAdapterV52,
    ODEScientificBundleV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    build_ode_bundle_v52,
    register_ode_adapters_v52,
    run_ode_replays_v52,
)
from .cross_domain_evaluation import (
    AblationArmProcessReceiptV52,
    CrossDomainAblationObservationV52,
    CrossDomainAblationSummaryV52,
    GoldCoverageSummaryV52,
    GoldTaskObservationV52,
    compare_ablation_arms_v52,
    run_fixture_ablation_arm_v52,
    summarize_cross_domain_ablation_v52,
    summarize_gold_coverage_v52,
)

__all__ = [
    "CandidateAdmissionAuthorityV52",
    "CandidateAdmissionPolicyV52",
    "CandidateAdmissionReceiptV52",
    "CrossDomainAblationObservationV52",
    "CrossDomainAblationSummaryV52",
    "EvolutionDecisionV52",
    "EvolutionProposalV52",
    "ExternalHostAttestationV52",
    "GeneratedCandidateV52",
    "GoldCoverageSummaryV52",
    "GoldTaskObservationV52",
    "GovernedCandidateRegistryV52",
    "GraphEvolutionControllerV52",
    "HostAttestationAuthorityV52",
    "LocalPrivateRunReceiptV52",
    "ODELevelAdapterV52",
    "ODEScientificBundleV52",
    "ODEThresholdsV52",
    "ODETimeSeriesSnapshotV52",
    "PrivateEvaluationRequestV52",
    "PrivatePromotionAuthorityV52",
    "PrivateQualificationReceiptV52",
    "PrivateWorkerAuthorityV52",
    "PrivateWorkerReceiptV52",
    "RecoveryAuthorityV52",
    "RecoveryBudgetV52",
    "RecoveryEvidenceV52",
    "RecoveryStateV52",
    "AblationArmProcessReceiptV52",
    "build_ode_bundle_v52",
    "compare_ablation_arms_v52",
    "register_ode_adapters_v52",
    "run_fixture_ablation_arm_v52",
    "run_ode_replays_v52",
    "run_local_private_worker_v52",
    "summarize_cross_domain_ablation_v52",
    "summarize_gold_coverage_v52",
]
