from __future__ import annotations

import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Annotated, Literal

import numpy
import scipy
from pydantic import Field

from .benchmark_cases import BenchmarkCase, BenchmarkSuite, build_fma_bench_v0
from .codex_driver import (
    PROTOCOL_VERSION,
    CodexAgentOutcome,
    CodexCLIConfig,
    CodexDrivenModelingAgent,
    ExplorerProblemView,
    ProcessResult,
)
from .evidence import EvidenceGraph
from .hashing import canonical_json, sha256_value
from .promotion import PromotionEngine, REQUIRED_EVIDENCE_KINDS
from .schemas import (
    ArtifactRef,
    CandidateRunOutcome,
    CompilerCertificate,
    OptimizationModelIR,
    ProblemContract,
    PromotionDecision,
    ReproductionReport,
    SolutionArtifact,
    StrictModel,
    ValidationVector,
)
from .storage import RunStore
from .validation import REQUIRED_HARD_CHECKS, attach_reproduction, validate_candidate


ArmId = Literal["fixture_golden", "fixture_mutant", "live_single", "live_repair"]
ArmRole = Literal["golden_control", "adversarial_control", "capability"]
ObservedStatus = Literal[
    "validated",
    "run_invalid",
    "needs_evidence",
    "no_result",
    "driver_error",
    "permission_denied",
    "needs_approval",
    "evidence_invalid",
    "exception",
]


class ArmMetadata(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    arm_id: ArmId
    role: ArmRole
    implementation_version: Literal["fma_bench_runner_v0"] = "fma_bench_runner_v0"
    max_rounds: Annotated[int, Field(ge=1, le=2)]
    requested_model: str | None = None
    executable_policy: Literal["auto_discovery", "explicit_path", "per_case_fixture"]
    configured_executable: str | None = None
    expected_cli_version: str
    timeout_seconds: Annotated[int, Field(gt=0)]
    max_candidates: Annotated[int, Field(ge=1, le=3)]
    max_input_bytes: Annotated[int, Field(gt=0)]
    max_schema_bytes: Annotated[int, Field(gt=0)]
    max_stdout_bytes: Annotated[int, Field(gt=0)]
    max_stderr_bytes: Annotated[int, Field(gt=0)]
    max_jsonl_line_bytes: Annotated[int, Field(gt=0)]
    max_events: Annotated[int, Field(gt=0)]
    max_oracle_assignments: Annotated[int, Field(gt=0)]
    live_inference: bool

    def content_hash(self) -> str:
        return sha256_value(self)


class HoldoutReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass", "fail", "not_applicable", "error"]
    assignment_count: Annotated[int | None, Field(ge=0)] = None
    feasible_assignment_count: Annotated[int | None, Field(ge=0)] = None
    feasibility_mismatch_count: Annotated[int, Field(ge=0)] = 0
    objective_mismatch_count: Annotated[int, Field(ge=0)] = 0
    reference_optimum: float | None = None
    candidate_optimum: float | None = None
    first_counterexample: dict[str, object] | None = None
    detail: str


class RuntimeInvocationIdentity(StrictModel):
    attempt_index: Annotated[int, Field(ge=1)]
    cli_version: str
    executable_sha256: str
    requested_model: str
    prompt_hash: str
    output_schema_hash: str
    public_view_hash: str


class BenchmarkCaseResult(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_run_id: str
    suite_hash: str
    case_id: str
    case_hash: str
    public_hash: str
    expected_hash: str
    family: str
    task_kind: str
    arm_id: ArmId
    arm_role: ArmRole
    arm_config_hash: str
    repetition: Annotated[int, Field(ge=1)]
    expected_status: Literal["validated", "no_result", "needs_evidence"]
    observed_status: ObservedStatus
    explicit_first_round_no_result: bool
    outer_claimed_validated: bool
    evidence_validated: bool
    evidence_detail: str
    holdout: HoldoutReport
    privacy_passed: bool
    privacy_detail: str
    false_promotion: bool
    exact_terminal_match: bool
    control_passed: bool
    infrastructure_failure: bool
    round_count: Annotated[int, Field(ge=0)]
    candidate_count: Annotated[int, Field(ge=0)]
    private_rejection_count: Annotated[int, Field(ge=0)]
    runtime_invocations: list[RuntimeInvocationIdentity] = Field(default_factory=list)
    metrics: dict[str, int | float | bool | str | None]
    exploration_directory: str
    candidate_run_directories: list[str]
    detail: str


class BenchmarkAggregate(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    suite_hash: str
    arm: ArmMetadata
    arm_config_hash: str
    run_validity: Literal["complete", "partial", "invalid"]
    suite_case_count: Annotated[int, Field(ge=1)]
    scheduled: Annotated[int, Field(ge=0)]
    completed: Annotated[int, Field(ge=0)]
    scored: Annotated[int, Field(ge=0)]
    infrastructure_errors: Annotated[int, Field(ge=0)]
    exact_terminal_correct: Annotated[int, Field(ge=0)]
    control_passed: Annotated[int, Field(ge=0)]
    false_promotion_count: Annotated[int, Field(ge=0)]
    privacy_failure_count: Annotated[int, Field(ge=0)]
    evidence_invalid_count: Annotated[int, Field(ge=0)]
    all_case_accuracy: float | None
    eligible_accuracy: float | None
    control_pass_rate: float | None
    validated_precision: float | None
    validated_recall: float | None
    no_result_precision: float | None
    no_result_recall: float | None
    answerable_coverage: float | None
    selective_accuracy: float | None
    infrastructure_error_rate: float | None
    family_macro_accuracy: float | None
    confusion_matrix: dict[str, dict[str, int]]
    family_accuracy: dict[str, float | None]
    usage_totals: dict[str, int]
    latency_ms_median: float | None
    latency_ms_p90: float | None
    runtime_invocation_count: Annotated[int, Field(ge=0)]
    runtime_executable_sha256s: list[str]
    runtime_cli_versions: list[str]
    runtime_requested_models: list[str]
    runtime_output_schema_hashes: list[str]
    runtime_prompt_set_hash: str | None
    runtime_provenance_passed: bool
    harness_integrity_passed: bool


class BenchmarkRunSummary(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    run_directory: str
    suite_hash: str
    arm: ArmMetadata
    selected_case_ids: list[str]
    repetitions: int
    aggregate: BenchmarkAggregate
    report_path: str
    event_chain_verified: bool


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[position])


def _ir_feasible(ir: OptimizationModelIR, assignment: dict[str, float]) -> bool:
    variables = {variable.name: variable for variable in ir.variables}
    if set(assignment) != set(variables):
        return False
    for name, variable in variables.items():
        value = assignment[name]
        if value < variable.lower_bound - 1e-9 or value > variable.upper_bound + 1e-9:
            return False
        if variable.kind.value in {"integer", "binary"} and abs(value - round(value)) > 1e-9:
            return False
    for constraint in ir.constraints:
        lhs = sum(coefficient * assignment[name] for name, coefficient in constraint.coefficients.items())
        if constraint.sense == "<=" and lhs > constraint.rhs + 1e-9:
            return False
        if constraint.sense == ">=" and lhs < constraint.rhs - 1e-9:
            return False
        if constraint.sense == "==" and abs(lhs - constraint.rhs) > 1e-9:
            return False
    return True


def _ir_objective(ir: OptimizationModelIR, assignment: dict[str, float]) -> float:
    return ir.objective.constant + sum(
        coefficient * assignment[name] for name, coefficient in ir.objective.coefficients.items()
    )


def compare_finite_semantics(
    reference: OptimizationModelIR,
    candidate: OptimizationModelIR,
    *,
    max_assignments: int = 100_000,
) -> HoldoutReport:
    """Compare feasibility on the full finite domain and objective on the shared feasible set."""

    reference.assert_sealed()
    candidate.assert_sealed()
    reference_variables = {variable.name: variable for variable in reference.variables}
    candidate_variables = {variable.name: variable for variable in candidate.variables}
    if set(reference_variables) != set(candidate_variables):
        return HoldoutReport(
            status="fail",
            detail="candidate variable names differ from the sealed holdout reference",
            first_counterexample={
                "reference_variables": sorted(reference_variables),
                "candidate_variables": sorted(candidate_variables),
            },
        )
    for name in sorted(reference_variables):
        left = reference_variables[name]
        right = candidate_variables[name]
        if (
            left.kind != right.kind
            or left.lower_bound != right.lower_bound
            or left.upper_bound != right.upper_bound
        ):
            return HoldoutReport(
                status="fail",
                detail=f"candidate domain differs for {name}",
                first_counterexample={
                    "variable": name,
                    "reference": left.model_dump(mode="json"),
                    "candidate": right.model_dump(mode="json"),
                },
            )

    names = sorted(reference_variables)
    domains: list[range] = []
    assignment_count = 1
    for name in names:
        variable = reference_variables[name]
        if variable.kind.value == "continuous":
            return HoldoutReport(status="error", detail="finite holdout does not support continuous domains")
        domain = range(math.ceil(variable.lower_bound), math.floor(variable.upper_bound) + 1)
        domains.append(domain)
        assignment_count *= len(domain)
        if assignment_count > max_assignments:
            return HoldoutReport(
                status="error",
                assignment_count=assignment_count,
                detail="finite holdout assignment budget exceeded",
            )

    feasibility_mismatches = 0
    objective_mismatches = 0
    feasible_count = 0
    first_counterexample: dict[str, object] | None = None
    reference_best: float | None = None
    candidate_best: float | None = None
    for values in itertools.product(*domains):
        assignment = dict(zip(names, map(float, values), strict=True))
        reference_feasible = _ir_feasible(reference, assignment)
        candidate_feasible = _ir_feasible(candidate, assignment)
        if reference_feasible != candidate_feasible:
            feasibility_mismatches += 1
            if first_counterexample is None:
                first_counterexample = {
                    "kind": "feasibility",
                    "assignment": assignment,
                    "reference_feasible": reference_feasible,
                    "candidate_feasible": candidate_feasible,
                }
            continue
        if not reference_feasible:
            continue
        feasible_count += 1
        reference_value = _ir_objective(reference, assignment)
        candidate_value = _ir_objective(candidate, assignment)
        if abs(reference_value - candidate_value) > 1e-7:
            objective_mismatches += 1
            if first_counterexample is None:
                first_counterexample = {
                    "kind": "objective",
                    "assignment": assignment,
                    "reference_objective": reference_value,
                    "candidate_objective": candidate_value,
                }
        if reference_best is None:
            reference_best = reference_value
            candidate_best = candidate_value
        elif reference.objective.sense == "maximize":
            reference_best = max(reference_best, reference_value)
            assert candidate_best is not None
            candidate_best = max(candidate_best, candidate_value)
        else:
            reference_best = min(reference_best, reference_value)
            assert candidate_best is not None
            candidate_best = min(candidate_best, candidate_value)

    passed = feasibility_mismatches == 0 and objective_mismatches == 0
    return HoldoutReport(
        status="pass" if passed else "fail",
        assignment_count=assignment_count,
        feasible_assignment_count=feasible_count,
        feasibility_mismatch_count=feasibility_mismatches,
        objective_mismatch_count=objective_mismatches,
        reference_optimum=reference_best,
        candidate_optimum=candidate_best,
        first_counterexample=first_counterexample,
        detail=(
            "candidate matches the sealed reference on the full finite domain"
            if passed
            else "candidate differs from the sealed finite-domain reference"
        ),
    )


def _load_envelopes(run_directory: Path, kind: str) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for path in sorted((run_directory / "artifacts").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if sha256_value(envelope) != path.stem:
            raise RuntimeError(f"artifact filename hash mismatch: {path}")
        if envelope.get("kind") == kind:
            payload = envelope.get("payload")
            if not isinstance(payload, dict):
                raise TypeError(f"{kind} artifact payload must be an object")
            payloads.append(payload)
    return payloads


def _supporting_artifacts(
    outcome: CandidateRunOutcome,
) -> tuple[RunStore, dict[str, object], dict[str, object]]:
    store = RunStore.open_existing(outcome.run_directory)
    with EvidenceGraph(Path(outcome.run_directory) / "evidence.sqlite3") as graph:
        if graph.node_status(outcome.claim_node_id) != outcome.decision.status:
            raise RuntimeError("claim status disagrees with the returned promotion decision")
        support = graph.supporting_nodes(outcome.claim_node_id)
        snapshot = graph.snapshot()
    grouped: dict[str, list[dict[str, object]]] = {}
    for node in support:
        grouped.setdefault(str(node["kind"]), []).append(node)
    loaded: dict[str, object] = {}
    for kind in sorted(REQUIRED_EVIDENCE_KINDS):
        nodes = grouped.get(kind, [])
        if len(nodes) != 1 or nodes[0].get("status") != "current":
            raise RuntimeError(f"expected one current {kind} artifact")
        metadata = json.loads(str(nodes[0]["metadata_json"]))
        loaded[kind] = store.load_artifact(
            ArtifactRef(
                kind=kind,
                sha256=str(nodes[0]["artifact_hash"]),
                relative_path=str(metadata["relative_path"]),
            )
        )
    return store, loaded, snapshot


def audit_validated_outcome(
    case: BenchmarkCase, outcome: CandidateRunOutcome
) -> tuple[bool, OptimizationModelIR | None, str]:
    """Rebuild the promotion evidence instead of trusting an outer validated label."""

    try:
        if outcome.decision.status != "validated" or outcome.claim_status != "validated":
            raise RuntimeError("candidate outcome is not internally marked validated")
        if outcome.decision.validation_scope != "synthetic_oracle":
            raise RuntimeError("unexpected validation scope")
        store, loaded, snapshot = _supporting_artifacts(outcome)
        contract = ProblemContract.model_validate(loaded["contract"])
        ir = OptimizationModelIR.model_validate(loaded["model_ir"])
        certificate = CompilerCertificate.model_validate(loaded["compiler_certificate"])
        solution = SolutionArtifact.model_validate(loaded["solution"])
        stored_validation = ValidationVector.model_validate(loaded["validation"])
        reproduction = ReproductionReport.model_validate(loaded["reproduction"])
        environment = loaded["environment"]
        if not isinstance(environment, dict):
            raise TypeError("environment evidence is not an object")
        if contract != case.contract or ir.contract_hash != case.contract.frozen_hash:
            raise RuntimeError("candidate evidence is not bound to this benchmark case")

        reproduction_bound = PromotionEngine._reproduction_is_bound(
            reproduction, certificate, solution
        )
        recomputed = attach_reproduction(
            validate_candidate(contract, ir, certificate, solution),
            passed=reproduction_bound,
            detail=(
                reproduction.detail
                if reproduction_bound
                else "stored reproduction failed benchmark re-binding"
            ),
        )
        if recomputed != stored_validation:
            raise RuntimeError("stored validation differs from benchmark recomputation")
        if not recomputed.hard_gates_pass(REQUIRED_HARD_CHECKS):
            raise RuntimeError("one or more recomputed hard gates failed")
        environment_bound = (
            environment.get("contract_hash") == contract.frozen_hash
            and environment.get("ir_hash") == ir.ir_hash
            and environment.get("compiler") == certificate.compiler_version
            and environment.get("verifier") == recomputed.verifier_version
            and environment.get("promotion_policy") == PromotionEngine.policy_version
            and environment.get("python") == sys.version
            and environment.get("numpy") == numpy.__version__
            and environment.get("scipy") == scipy.__version__
        )
        if not environment_bound:
            raise RuntimeError("environment evidence is not bound to the current verifier")
        if not all(outcome.decision.gate_results.values()):
            raise RuntimeError("returned promotion decision contains a failed gate")

        nodes = {str(node["node_id"]): node for node in snapshot["nodes"]}
        promotion_targets = [
            str(edge["target_id"])
            for edge in snapshot["edges"]
            if edge["source_id"] == outcome.claim_node_id
            and edge["relation"] == "evaluated_by"
        ]
        if len(promotion_targets) != 1:
            raise RuntimeError("expected exactly one promotion artifact")
        promotion_node = nodes[promotion_targets[0]]
        if promotion_node["kind"] != "promotion" or promotion_node["status"] != "current":
            raise RuntimeError("promotion evidence node is invalid")
        promotion_metadata = json.loads(str(promotion_node["metadata_json"]))
        stored_promotion = PromotionDecision.model_validate(
            store.load_artifact(
                ArtifactRef(
                    kind="promotion",
                    sha256=str(promotion_node["artifact_hash"]),
                    relative_path=str(promotion_metadata["relative_path"]),
                )
            )
        )
        if stored_promotion != outcome.decision:
            raise RuntimeError("returned promotion differs from its content-addressed artifact")
        input_snapshot = store.load_artifact(outcome.decision.evidence_snapshot_artifact)
        if not isinstance(input_snapshot, dict) or input_snapshot.get("snapshot_hash") != outcome.decision.evidence_snapshot_hash:
            raise RuntimeError("promotion input snapshot binding failed")
        return True, ir, "promotion evidence, replay, environment, and hard gates were recomputed"
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _candidate_to_draft(ir: OptimizationModelIR, *, mutate: bool) -> dict[str, object]:
    constant = ir.objective.constant + (1 if mutate else 0)
    return {
        "skeleton_id": ir.skeleton_id,
        "evolution_operator": ir.lineage.evolution_operator or "formulate",
        "rationale": (
            "Adversarial fixture changes the objective constant"
            if mutate
            else "Golden fixture mirrors the sealed public model specification"
        ),
        "variables": [variable.model_dump(mode="json") for variable in ir.variables],
        "objective": {
            "sense": ir.objective.sense,
            "coefficients": [
                {"variable": name, "coefficient": coefficient}
                for name, coefficient in sorted(ir.objective.coefficients.items())
            ],
            "constant": constant,
            "unit": ir.objective.unit,
            "contract_clause_ids": ir.objective.contract_clause_ids,
        },
        "constraints": [
            {
                "constraint_id": constraint.constraint_id,
                "coefficients": [
                    {"variable": name, "coefficient": coefficient}
                    for name, coefficient in sorted(constraint.coefficients.items())
                ],
                "sense": constraint.sense,
                "rhs": constraint.rhs,
                "lhs_unit": constraint.lhs_unit,
                "rhs_unit": constraint.rhs_unit,
                "contract_clause_ids": constraint.contract_clause_ids,
            }
            for constraint in ir.constraints
        ],
        "validation_obligations": ir.validation_obligations,
        "unresolved_assumptions": [],
    }


class FixtureProcessRunner:
    """Protocol-faithful fake transport; it does not count as model capability evidence."""

    def __init__(self, case: BenchmarkCase, *, mutate: bool = False) -> None:
        self.case = case
        self.mutate = mutate
        self.prompts: list[str] = []

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        input_text: str | None,
        timeout_seconds: int,
        env: dict[str, str],
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> ProcessResult:
        del cwd, timeout_seconds, env, max_stdout_bytes, max_stderr_bytes
        if argv[-1] == "--version":
            return ProcessResult(0, "codex-cli 0.144.6\n", "")
        if "login" in argv and "status" in argv:
            return ProcessResult(0, "Logged in using fixture credentials\n", "")
        if "mcp" in argv and "list" in argv:
            disabled = any("mcp_servers." in value for value in argv)
            return ProcessResult(
                0,
                json.dumps([{"name": "fixture_mcp", "enabled": not disabled}]),
                "",
            )
        if "exec" not in argv or input_text is None:
            raise AssertionError(f"unexpected fixture process call: {argv}")
        self.prompts.append(input_text)
        request = json.loads(input_text.split("INPUT_JSON\n", 1)[1])
        if self.case.expected_status == "no_result":
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request["request_id"],
                "status": "no_result",
                "candidates": [],
                "no_result_reason": "The public task is outside the bounded protocol or lacks required facts.",
                "notes": [],
            }
        else:
            assert self.case.reference_ir is not None
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request["request_id"],
                "status": "proposed",
                "candidates": [_candidate_to_draft(self.case.reference_ir, mutate=self.mutate)],
                "no_result_reason": "",
                "notes": [],
            }
        events = [
            {"type": "thread.started", "thread_id": "fixture-thread"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "id": "fixture-message",
                    "type": "agent_message",
                    "text": json.dumps(response, ensure_ascii=False),
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 0,
                    "output_tokens": 80,
                    "reasoning_output_tokens": 10,
                },
            },
        ]
        return ProcessResult(0, "\n".join(json.dumps(event) for event in events) + "\n", "")


def _arm_metadata(arm_id: ArmId, config: CodexCLIConfig) -> ArmMetadata:
    live_inference = arm_id.startswith("live_")
    return ArmMetadata(
        arm_id=arm_id,
        role=(
            "golden_control"
            if arm_id == "fixture_golden"
            else "adversarial_control"
            if arm_id == "fixture_mutant"
            else "capability"
        ),
        max_rounds=2 if arm_id == "live_repair" else 1,
        requested_model=config.requested_model if live_inference else "fixture-control",
        executable_policy=(
            "per_case_fixture"
            if not live_inference
            else "explicit_path"
            if config.executable is not None
            else "auto_discovery"
        ),
        configured_executable=(
            str(config.executable.resolve()) if config.executable is not None else None
        ),
        expected_cli_version=config.expected_cli_version,
        timeout_seconds=config.timeout_seconds,
        max_candidates=config.max_candidates,
        max_input_bytes=config.max_input_bytes,
        max_schema_bytes=config.max_schema_bytes,
        max_stdout_bytes=config.max_stdout_bytes,
        max_stderr_bytes=config.max_stderr_bytes,
        max_jsonl_line_bytes=config.max_jsonl_line_bytes,
        max_events=config.max_events,
        max_oracle_assignments=config.max_oracle_assignments,
        live_inference=live_inference,
    )


def _guard_benchmark_prompt(suite: BenchmarkSuite, prompt: str) -> None:
    leaked_case_ids = [
        case.case_id for case in suite.cases if case.privacy_canary in prompt
    ]
    if leaked_case_ids:
        raise PermissionError(
            "benchmark prompt contains private canary for: " + ", ".join(leaked_case_ids)
        )


def _prompt_privacy(
    suite: BenchmarkSuite,
    case: BenchmarkCase,
    outcome: CodexAgentOutcome,
) -> tuple[bool, str]:
    public_text = canonical_json(ExplorerProblemView.from_contract(case.contract))
    try:
        _guard_benchmark_prompt(suite, public_text)
    except PermissionError as exc:
        return False, str(exc)
    if not outcome.exploration_directory:
        return True, "no CLI prompt was created; public projection excluded the canary"
    try:
        directory = Path(outcome.exploration_directory)
        RunStore.open_existing(directory)
        prompts = _load_envelopes(directory, "explorer_public_prompt")
        if not prompts:
            return False, "no auditable public prompt artifact exists"
        for payload in prompts:
            prompt = payload.get("prompt")
            if not isinstance(prompt, str):
                return False, "public prompt artifact is malformed"
            try:
                _guard_benchmark_prompt(suite, prompt)
            except PermissionError as exc:
                return False, str(exc)
        return True, f"{len(prompts)} prompt artifact(s) exclude all suite canaries"
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"prompt privacy audit failed: {type(exc).__name__}: {exc}"


def _usage_metrics(outcome: CodexAgentOutcome, *, simulated: bool) -> dict[str, int | float | bool | str | None]:
    metrics: dict[str, int | float | bool | str | None] = {
        "simulated_usage": simulated,
        "elapsed_ms": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "tool_events_observed": 0,
        "served_model_attested": False,
        "cost_usd": None,
    }
    if not outcome.exploration_directory:
        return metrics
    try:
        receipts = _load_envelopes(Path(outcome.exploration_directory), "exploration_receipt")
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return metrics
    for receipt in receipts:
        metrics["elapsed_ms"] = int(metrics["elapsed_ms"] or 0) + int(receipt.get("elapsed_ms", 0))
        metrics["tool_events_observed"] = int(metrics["tool_events_observed"] or 0) + int(receipt.get("tool_events_observed", 0))
        usage = receipt.get("usage", {})
        if isinstance(usage, dict):
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
                metrics[key] = int(metrics[key] or 0) + int(usage.get(key, 0))
    return metrics


def _runtime_invocation_identities(
    case: BenchmarkCase,
    outcome: CodexAgentOutcome,
) -> list[RuntimeInvocationIdentity]:
    if not outcome.exploration_directory:
        return []
    receipts = _load_envelopes(
        Path(outcome.exploration_directory), "exploration_receipt"
    )
    identities = [
        RuntimeInvocationIdentity(
            attempt_index=int(receipt["attempt_index"]),
            cli_version=str(receipt["cli_version"]),
            executable_sha256=str(receipt["executable_sha256"]),
            requested_model=str(receipt["requested_model"]),
            prompt_hash=str(receipt["prompt_hash"]),
            output_schema_hash=str(receipt["output_schema_hash"]),
            public_view_hash=str(receipt["public_view_hash"]),
        )
        for receipt in receipts
    ]
    identities.sort(key=lambda identity: identity.attempt_index)
    if len({identity.attempt_index for identity in identities}) != len(identities):
        raise RuntimeError("duplicate runtime receipt attempt index")
    if any(identity.public_view_hash != case.public_hash() for identity in identities):
        raise RuntimeError("runtime receipt public view is not bound to the benchmark case")
    return identities


def _explicit_first_round_no_result(outcome: CodexAgentOutcome) -> bool:
    if outcome.status != "no_result" or not outcome.rounds:
        return False
    first = outcome.rounds[0]
    return (
        first.attempt_index == 1
        and first.driver_status == "no_result"
        and not first.proposed_candidate_ids
        and not first.assessed_candidate_ids
    )


def _audit_outcome_binding(
    outcome: CodexAgentOutcome,
    expected_case_output: Path,
) -> tuple[bool, str]:
    """Bind an agent outcome to this exact benchmark case execution directory."""

    try:
        case_root = expected_case_output.resolve(strict=True)
        exploration_root = (case_root / "explorations").resolve(strict=True)
        exploration_directory = Path(outcome.exploration_directory).resolve(strict=True)
    except (OSError, RuntimeError):
        return False, "outcome exploration directory is missing or unresolved"
    if not exploration_directory.is_dir() or not exploration_directory.is_relative_to(
        exploration_root
    ):
        return False, "outcome exploration directory is outside this case execution"

    try:
        exploration_store = RunStore.open_existing(exploration_directory)
        records = [
            json.loads(line)
            for line in exploration_store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        stopped = [record for record in records if record.get("event_type") == "codex_agent_stopped"]
        if len(stopped) != 1 or records[-1].get("event_type") != "codex_agent_stopped":
            return False, "exploration ledger does not have one final stopped event"
        stopped_payload = stopped[0].get("payload")
        if not isinstance(stopped_payload, dict) or not isinstance(
            stopped_payload.get("result"), dict
        ):
            return False, "stopped event does not reference a terminal outcome artifact"
        outcome_ref = ArtifactRef.model_validate(stopped_payload["result"])
        if outcome_ref.kind != "codex_agent_outcome":
            return False, "stopped event references the wrong artifact kind"
        persisted_outcome = CodexAgentOutcome.model_validate(
            exploration_store.load_artifact(outcome_ref)
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False, "persisted terminal outcome or stopped event failed integrity checks"
    if persisted_outcome != outcome:
        return False, "returned outcome differs from the persisted terminal outcome"

    assessed_ids: list[str] = []
    decision_statuses: list[str] = []
    for round_ in outcome.rounds:
        if len(round_.assessed_candidate_ids) != len(round_.decision_statuses):
            return False, "round candidate IDs and decision statuses are not aligned"
        assessed_ids.extend(round_.assessed_candidate_ids)
        decision_statuses.extend(round_.decision_statuses)

    outcome_ids = [candidate.candidate_id for candidate in outcome.candidate_outcomes]
    outcome_statuses = [candidate.decision.status for candidate in outcome.candidate_outcomes]
    if assessed_ids != outcome_ids or decision_statuses != outcome_statuses:
        return False, "round ledger does not bind exactly to the candidate outcomes"

    if outcome.candidate_outcomes:
        try:
            candidate_root = (case_root / "candidate_runs").resolve(strict=True)
            candidate_directories = [
                Path(candidate.run_directory).resolve(strict=True)
                for candidate in outcome.candidate_outcomes
            ]
        except (OSError, RuntimeError):
            return False, "candidate run directory is missing or unresolved"
        if any(
            not directory.is_dir() or not directory.is_relative_to(candidate_root)
            for directory in candidate_directories
        ):
            return False, "candidate evidence is outside this case execution"

    return True, "outcome paths and round ledger are bound to this case execution"


def _score_case(
    *,
    benchmark_run_id: str,
    suite: BenchmarkSuite,
    case: BenchmarkCase,
    arm: ArmMetadata,
    repetition: int,
    outcome: CodexAgentOutcome,
    expected_case_output: Path,
    privacy_passed: bool,
    privacy_detail: str,
) -> BenchmarkCaseResult:
    binding_valid, binding_detail = _audit_outcome_binding(outcome, expected_case_output)
    runtime_invocations: list[RuntimeInvocationIdentity] = []
    if binding_valid:
        try:
            runtime_invocations = _runtime_invocation_identities(case, outcome)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            binding_valid = False
            binding_detail = f"runtime receipt binding failed: {type(exc).__name__}: {exc}"
    if binding_valid and outcome.status in {
        "validated",
        "run_invalid",
        "needs_evidence",
        "no_result",
    } and not runtime_invocations:
        binding_valid = False
        binding_detail = "terminal outcome has no bound successful runtime receipt"
    observed: ObservedStatus = outcome.status if binding_valid else "evidence_invalid"
    outer_validated = outcome.status == "validated"
    evidence_valid = False
    evidence_detail = (
        "not applicable because the outer outcome was not validated"
        if binding_valid
        else binding_detail
    )
    candidate_ir: OptimizationModelIR | None = None
    if outer_validated and binding_valid:
        validated_outcomes = [
            candidate
            for candidate in outcome.candidate_outcomes
            if candidate.decision.status == "validated"
        ]
        if len(validated_outcomes) != 1:
            observed = "evidence_invalid"
            evidence_detail = "outer validated status does not bind exactly one validated candidate"
        else:
            evidence_valid, candidate_ir, evidence_detail = audit_validated_outcome(
                case, validated_outcomes[0]
            )
            if not evidence_valid:
                observed = "evidence_invalid"

    if candidate_ir is not None and case.reference_ir is not None:
        holdout = compare_finite_semantics(case.reference_ir, candidate_ir)
    elif case.reference_ir is None:
        holdout = HoldoutReport(
            status="not_applicable",
            detail="the pre-registered terminal is explicit no_result rather than a candidate model",
        )
    else:
        holdout = HoldoutReport(
            status="not_applicable",
            detail="no evidence-backed promoted candidate was available for private holdout",
        )

    explicit_no_result = binding_valid and _explicit_first_round_no_result(outcome)
    validated_correctly = (
        observed == "validated"
        and evidence_valid
        and holdout.status == "pass"
        and privacy_passed
    )
    if case.expected_status == "validated":
        exact = validated_correctly
    elif case.expected_status == "no_result":
        exact = observed == "no_result" and explicit_no_result and privacy_passed
    else:
        exact = observed == "needs_evidence" and privacy_passed

    false_promotion = outer_validated and (
        case.expected_status != "validated"
        or not evidence_valid
        or holdout.status != "pass"
        or not privacy_passed
    )
    infrastructure_failure = observed in {
        "driver_error",
        "permission_denied",
        "needs_approval",
        "evidence_invalid",
        "exception",
    }
    if arm.role == "adversarial_control" and case.expected_status == "validated":
        control_passed = (
            observed == "run_invalid"
            and not infrastructure_failure
            and any(round_.private_rejections for round_ in outcome.rounds)
        )
    else:
        control_passed = exact
    metrics = _usage_metrics(outcome, simulated=not arm.live_inference)
    candidate_directories = [candidate.run_directory for candidate in outcome.candidate_outcomes]
    return BenchmarkCaseResult(
        benchmark_run_id=benchmark_run_id,
        suite_hash=str(suite.suite_hash),
        case_id=case.case_id,
        case_hash=str(case.sealed_hash),
        public_hash=case.public_hash(),
        expected_hash=case.expected_hash(),
        family=case.family,
        task_kind=case.task_kind,
        arm_id=arm.arm_id,
        arm_role=arm.role,
        arm_config_hash=arm.content_hash(),
        repetition=repetition,
        expected_status=case.expected_status,
        observed_status=observed,
        explicit_first_round_no_result=explicit_no_result,
        outer_claimed_validated=outer_validated,
        evidence_validated=evidence_valid,
        evidence_detail=evidence_detail,
        holdout=holdout,
        privacy_passed=privacy_passed,
        privacy_detail=privacy_detail,
        false_promotion=false_promotion,
        exact_terminal_match=exact,
        control_passed=control_passed,
        infrastructure_failure=infrastructure_failure,
        round_count=len(outcome.rounds),
        candidate_count=sum(len(round_.proposed_candidate_ids) for round_ in outcome.rounds),
        private_rejection_count=sum(
            len(round_.private_rejections) for round_ in outcome.rounds
        ),
        runtime_invocations=runtime_invocations,
        metrics=metrics,
        exploration_directory=outcome.exploration_directory,
        candidate_run_directories=candidate_directories,
        detail=outcome.stop_reason,
    )


def _derived_case_flags(
    result: BenchmarkCaseResult,
    arm: ArmMetadata,
) -> dict[str, bool]:
    infrastructure_failure = result.observed_status in {
        "driver_error",
        "permission_denied",
        "needs_approval",
        "evidence_invalid",
        "exception",
    }
    validated_correctly = (
        result.observed_status == "validated"
        and result.outer_claimed_validated
        and result.evidence_validated
        and result.holdout.status == "pass"
        and result.privacy_passed
    )
    if result.expected_status == "validated":
        exact_terminal_match = validated_correctly
    elif result.expected_status == "no_result":
        exact_terminal_match = (
            result.observed_status == "no_result"
            and result.explicit_first_round_no_result
            and result.privacy_passed
        )
    else:
        exact_terminal_match = (
            result.observed_status == "needs_evidence" and result.privacy_passed
        )
    false_promotion = result.outer_claimed_validated and (
        result.expected_status != "validated"
        or not result.evidence_validated
        or result.holdout.status != "pass"
        or not result.privacy_passed
    )
    if arm.role == "adversarial_control" and result.expected_status == "validated":
        control_passed = (
            result.observed_status == "run_invalid"
            and not infrastructure_failure
            and result.private_rejection_count > 0
        )
    else:
        control_passed = exact_terminal_match
    return {
        "infrastructure_failure": infrastructure_failure,
        "exact_terminal_match": exact_terminal_match,
        "false_promotion": false_promotion,
        "control_passed": control_passed,
    }


def aggregate_results(
    suite: BenchmarkSuite,
    arm: ArmMetadata,
    results: list[BenchmarkCaseResult],
    *,
    benchmark_run_id: str,
    selected_case_ids: list[str],
    repetitions: int,
) -> BenchmarkAggregate:
    if len(selected_case_ids) != len(set(selected_case_ids)):
        raise ValueError("selected benchmark case IDs must be unique")
    suite_cases = {case.case_id: case for case in suite.cases}
    unknown_selected = sorted(set(selected_case_ids) - set(suite_cases))
    if unknown_selected:
        raise ValueError("aggregate contains unknown selected case IDs")
    expected_keys = {
        (repetition, case_id)
        for repetition in range(1, repetitions + 1)
        for case_id in selected_case_ids
    }
    ordered = sorted(results, key=lambda result: (result.repetition, result.case_id))
    keys = [(result.repetition, result.case_id) for result in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate benchmark case result")
    if any(key not in expected_keys for key in keys):
        raise ValueError("benchmark result is outside the selected case/repetition matrix")
    for result in ordered:
        case = suite_cases[result.case_id]
        expected_bindings = {
            "suite_hash": str(suite.suite_hash),
            "case_hash": str(case.sealed_hash),
            "public_hash": case.public_hash(),
            "expected_hash": case.expected_hash(),
            "family": case.family,
            "task_kind": case.task_kind,
            "expected_status": case.expected_status,
            "arm_id": arm.arm_id,
            "arm_role": arm.role,
            "arm_config_hash": arm.content_hash(),
            "benchmark_run_id": benchmark_run_id,
        }
        actual_bindings = {
            key: getattr(result, key) for key in expected_bindings
        }
        if actual_bindings != expected_bindings:
            raise ValueError(f"benchmark result identity binding failed: {result.case_id}")
        if result.observed_status == "validated" and not result.outer_claimed_validated:
            raise ValueError(f"validated status lacks an outer claim: {result.case_id}")
        if result.evidence_validated and not result.outer_claimed_validated:
            raise ValueError(f"evidence validation lacks an outer claim: {result.case_id}")
        if (
            result.explicit_first_round_no_result
            and result.observed_status != "no_result"
        ):
            raise ValueError(f"explicit no-result flag contradicts status: {result.case_id}")
        derived_flags = _derived_case_flags(result, arm)
        actual_flags = {key: getattr(result, key) for key in derived_flags}
        if actual_flags != derived_flags:
            raise ValueError(f"benchmark result derived flags failed: {result.case_id}")
    scheduled = len(selected_case_ids) * repetitions
    completed = len(ordered)
    infrastructure = sum(result.infrastructure_failure for result in ordered)
    scored_results = [result for result in ordered if not result.infrastructure_failure]
    exact = sum(result.exact_terminal_match for result in ordered)
    control_passed = sum(result.control_passed for result in ordered)
    false_promotions = sum(result.false_promotion for result in ordered)
    privacy_failures = sum(not result.privacy_passed for result in ordered)
    evidence_invalid = sum(result.observed_status == "evidence_invalid" for result in ordered)

    confusion: dict[str, dict[str, int]] = {}
    for result in ordered:
        row = confusion.setdefault(result.expected_status, {})
        row[result.observed_status] = row.get(result.observed_status, 0) + 1

    predicted_validated = [result for result in ordered if result.observed_status == "validated"]
    expected_validated = [result for result in ordered if result.expected_status == "validated"]
    true_validated = sum(
        result.expected_status == "validated" and result.exact_terminal_match
        for result in predicted_validated
    )
    predicted_no_result = [
        result
        for result in ordered
        if result.observed_status == "no_result" and result.explicit_first_round_no_result
    ]
    expected_no_result = [result for result in ordered if result.expected_status == "no_result"]
    true_no_result = sum(
        result.expected_status == "no_result" and result.exact_terminal_match
        for result in predicted_no_result
    )
    answered = [
        result
        for result in scored_results
        if not (result.observed_status == "no_result" and result.explicit_first_round_no_result)
    ]

    family_accuracy: dict[str, float | None] = {}
    for family in sorted({case.family for case in suite.cases}):
        family_results = [result for result in ordered if result.family == family]
        family_accuracy[family] = _safe_rate(
            sum(result.exact_terminal_match for result in family_results),
            len(family_results),
        )
    family_values = [value for value in family_accuracy.values() if value is not None]
    usage_keys = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "tool_events_observed",
    )
    usage_totals = {
        key: sum(int(result.metrics.get(key, 0) or 0) for result in ordered)
        for key in usage_keys
    }
    latencies = [int(result.metrics.get("elapsed_ms", 0) or 0) for result in ordered]
    runtime_records = [
        {
            "case_id": result.case_id,
            "repetition": result.repetition,
            **identity.model_dump(mode="json"),
        }
        for result in ordered
        for identity in result.runtime_invocations
    ]
    runtime_executable_sha256s = sorted(
        {str(record["executable_sha256"]) for record in runtime_records}
    )
    runtime_cli_versions = sorted(
        {str(record["cli_version"]) for record in runtime_records}
    )
    runtime_requested_models = sorted(
        {str(record["requested_model"]) for record in runtime_records}
    )
    runtime_output_schema_hashes = sorted(
        {str(record["output_schema_hash"]) for record in runtime_records}
    )
    expected_requested_model = arm.requested_model or "cli_default"
    runtime_provenance_passed = (
        bool(ordered)
        and all(
            len(result.runtime_invocations) == result.round_count
            and result.round_count > 0
            and all(
                identity.public_view_hash == result.public_hash
                for identity in result.runtime_invocations
            )
            for result in ordered
        )
        and len(runtime_executable_sha256s) == 1
        and runtime_cli_versions == [arm.expected_cli_version]
        and runtime_requested_models == [expected_requested_model]
        and len(runtime_output_schema_hashes) == 1
    )
    runtime_prompt_set_hash = (
        sha256_value(runtime_records) if runtime_records else None
    )
    invalid = set(keys) != expected_keys
    run_validity: Literal["complete", "partial", "invalid"]
    if invalid:
        run_validity = "invalid"
    elif len(selected_case_ids) < len(suite.cases):
        run_validity = "partial"
    else:
        run_validity = "complete"
    return BenchmarkAggregate(
        suite_hash=str(suite.suite_hash),
        arm=arm,
        arm_config_hash=arm.content_hash(),
        run_validity=run_validity,
        suite_case_count=len(suite.cases),
        scheduled=scheduled,
        completed=completed,
        scored=len(scored_results),
        infrastructure_errors=infrastructure,
        exact_terminal_correct=exact,
        control_passed=control_passed,
        false_promotion_count=false_promotions,
        privacy_failure_count=privacy_failures,
        evidence_invalid_count=evidence_invalid,
        all_case_accuracy=_safe_rate(exact, scheduled),
        eligible_accuracy=_safe_rate(
            sum(result.exact_terminal_match for result in scored_results), len(scored_results)
        ),
        control_pass_rate=_safe_rate(control_passed, scheduled),
        validated_precision=_safe_rate(true_validated, len(predicted_validated)),
        validated_recall=_safe_rate(true_validated, len(expected_validated)),
        no_result_precision=_safe_rate(true_no_result, len(predicted_no_result)),
        no_result_recall=_safe_rate(true_no_result, len(expected_no_result)),
        answerable_coverage=_safe_rate(
            sum(
                result.expected_status == "validated"
                and result.observed_status == "validated"
                for result in ordered
            ),
            len(expected_validated),
        ),
        selective_accuracy=_safe_rate(
            sum(result.exact_terminal_match for result in answered), len(answered)
        ),
        infrastructure_error_rate=_safe_rate(infrastructure, scheduled),
        family_macro_accuracy=(statistics.mean(family_values) if family_values else None),
        confusion_matrix=confusion,
        family_accuracy=family_accuracy,
        usage_totals=usage_totals,
        latency_ms_median=(statistics.median(latencies) if latencies else None),
        latency_ms_p90=_percentile(latencies, 0.9),
        runtime_invocation_count=len(runtime_records),
        runtime_executable_sha256s=runtime_executable_sha256s,
        runtime_cli_versions=runtime_cli_versions,
        runtime_requested_models=runtime_requested_models,
        runtime_output_schema_hashes=runtime_output_schema_hashes,
        runtime_prompt_set_hash=runtime_prompt_set_hash,
        runtime_provenance_passed=runtime_provenance_passed,
        harness_integrity_passed=(
            not invalid
            and infrastructure == 0
            and false_promotions == 0
            and privacy_failures == 0
            and evidence_invalid == 0
            and runtime_provenance_passed
            and (arm.role == "capability" or control_passed == scheduled)
        ),
    )


def _write_report(
    path: Path,
    suite: BenchmarkSuite,
    arm: ArmMetadata,
    aggregate: BenchmarkAggregate,
    results: list[BenchmarkCaseResult],
) -> Path:
    rows = []
    for result in sorted(results, key=lambda item: (item.repetition, item.case_id)):
        rows.append(
            "| {case} | {family} | {task} | {expected} | {observed} | {explicit} | {evidence} | {holdout} | {privacy} | {control} |".format(
                case=result.case_id,
                family=result.family,
                task=result.task_kind,
                expected=result.expected_status,
                observed=result.observed_status,
                explicit="yes" if result.explicit_first_round_no_result else "no",
                evidence="pass" if result.evidence_validated else "n/a",
                holdout=result.holdout.status,
                privacy="pass" if result.privacy_passed else "FAIL",
                control="pass" if result.control_passed else "FAIL",
            )
        )
    content = f"""# FMA-Bench v0 Report

## Run identity

- Suite: `{suite.suite_version}`
- Suite hash: `{suite.suite_hash}`
- Arm: `{arm.arm_id}` (`{arm.role}`)
- Arm config hash: `{aggregate.arm_config_hash}`
- Run validity: `{aggregate.run_validity}`

## Primary metrics

- Scheduled/completed: `{aggregate.scheduled}/{aggregate.completed}`
- Exact terminal accuracy: `{aggregate.all_case_accuracy}`
- Control pass rate: `{aggregate.control_pass_rate}`
- False promotions: `{aggregate.false_promotion_count}`
- Privacy failures: `{aggregate.privacy_failure_count}`
- Evidence-invalid promotions: `{aggregate.evidence_invalid_count}`
- Validated precision/recall: `{aggregate.validated_precision}` / `{aggregate.validated_recall}`
- Explicit NO_RESULT precision/recall: `{aggregate.no_result_precision}` / `{aggregate.no_result_recall}`
- Runtime provenance passed: `{str(aggregate.runtime_provenance_passed).lower()}`
- Runtime invocations: `{aggregate.runtime_invocation_count}`
- CLI version(s): `{', '.join(aggregate.runtime_cli_versions) or 'none'}`
- Executable SHA-256(s): `{', '.join(aggregate.runtime_executable_sha256s) or 'none'}`
- Output-schema SHA-256(s): `{', '.join(aggregate.runtime_output_schema_hashes) or 'none'}`
- Prompt-set commitment: `{aggregate.runtime_prompt_set_hash}`
- Harness integrity passed: `{str(aggregate.harness_integrity_passed).lower()}`

`Explain` tasks receive mechanical IR scoring only. Free-text rationale quality is deliberately not promoted into `validated@synthetic_oracle`.

## Case results

| Case | Family | Task | Expected | Observed | Explicit NR | Evidence | Holdout | Privacy | Control |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Evidence boundary

The golden and mutant fixture arms validate protocol and harness invariants; they are not model-capability measurements. Only live arms measure Codex behavior. A live `validated` result counts only after content-addressed evidence is reloaded, hard gates are recomputed, and the candidate matches the separate finite-domain holdout. `served_model_attested=false` and `cost_usd=null` remain explicit limitations.
"""
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


class BenchmarkRunner:
    def __init__(
        self,
        output_root: str | Path,
        *,
        suite: BenchmarkSuite | None = None,
        codex_config: CodexCLIConfig | None = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.suite = suite or build_fma_bench_v0()
        self.codex_config = codex_config or CodexCLIConfig()

    def _run_case(
        self,
        case: BenchmarkCase,
        arm: ArmMetadata,
        case_output: Path,
    ) -> CodexAgentOutcome:
        if arm.live_inference:
            return CodexDrivenModelingAgent(
                case_output,
                self.codex_config,
                max_rounds=arm.max_rounds,
                prompt_guard=lambda prompt: _guard_benchmark_prompt(self.suite, prompt),
            ).run(case.contract)

        fixture_path = case_output / "fixture_bin" / "codex.exe"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_bytes(b"fma-bench-fixture-cli-v0")
        process_runner = FixtureProcessRunner(
            case, mutate=arm.arm_id == "fixture_mutant"
        )
        fixture_config = CodexCLIConfig(
            executable=fixture_path,
            requested_model="fixture-control",
            expected_cli_version=self.codex_config.expected_cli_version,
            timeout_seconds=self.codex_config.timeout_seconds,
            max_candidates=self.codex_config.max_candidates,
            max_input_bytes=self.codex_config.max_input_bytes,
            max_schema_bytes=self.codex_config.max_schema_bytes,
            max_stdout_bytes=self.codex_config.max_stdout_bytes,
            max_stderr_bytes=self.codex_config.max_stderr_bytes,
            max_jsonl_line_bytes=self.codex_config.max_jsonl_line_bytes,
            max_events=self.codex_config.max_events,
            max_oracle_assignments=self.codex_config.max_oracle_assignments,
        )
        return CodexDrivenModelingAgent(
            case_output,
            fixture_config,
            max_rounds=1,
            process_runner=process_runner,
            cli_locator=lambda explicit=None: fixture_path,
            prompt_guard=lambda prompt: _guard_benchmark_prompt(self.suite, prompt),
        ).run(case.contract)

    def run(
        self,
        arm_id: ArmId,
        *,
        case_ids: list[str] | None = None,
        repetitions: int = 1,
        live_authorized: bool = False,
    ) -> BenchmarkRunSummary:
        if not 1 <= repetitions <= 5:
            raise ValueError("repetitions must be between one and five")
        arm = _arm_metadata(arm_id, self.codex_config)
        if arm.live_inference and not live_authorized:
            raise PermissionError("live benchmark inference requires explicit authorization")
        by_id = {case.case_id: case for case in self.suite.cases}
        selected_ids = sorted(list(by_id) if case_ids is None else case_ids)
        if not selected_ids:
            raise ValueError("at least one benchmark case must be selected")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected benchmark case IDs must be unique")
        unknown = sorted(set(selected_ids) - set(by_id))
        if unknown:
            raise KeyError(f"unknown benchmark cases: {', '.join(unknown)}")
        selected = [by_id[case_id] for case_id in selected_ids]
        for case in selected:
            _guard_benchmark_prompt(
                self.suite,
                canonical_json(ExplorerProblemView.from_contract(case.contract)),
            )
        for case in selected:
            if case.sealed_hash != case.content_hash():
                raise RuntimeError(f"benchmark case seal failed: {case.case_id}")
        if self.suite.suite_hash != self.suite.content_hash():
            raise RuntimeError("benchmark suite seal failed")

        store = RunStore(self.output_root / "benchmark_runs")
        store.put_artifact("benchmark_public_manifest", self.suite.public_manifest())
        store.put_artifact("benchmark_sealed_commitment", self.suite.sealed_commitment())
        store.put_artifact(
            "benchmark_run_config",
            {
                "suite_hash": self.suite.suite_hash,
                "arm": arm.model_dump(mode="json"),
                "arm_config_hash": arm.content_hash(),
                "selected_case_ids": selected_ids,
                "repetitions": repetitions,
            },
        )
        store.emit(
            "benchmark_created",
            {
                "suite_hash": self.suite.suite_hash,
                "arm_id": arm_id,
                "scheduled": len(selected) * repetitions,
            },
        )
        results: list[BenchmarkCaseResult] = []
        for repetition in range(1, repetitions + 1):
            for case_index, case in enumerate(selected, start=1):
                store.emit(
                    "benchmark_case_started",
                    {"case_id": case.case_id, "repetition": repetition},
                )
                # Keep child paths short enough for legacy Windows MAX_PATH while
                # binding the human-readable case ID in the outer event ledger.
                case_output = store.run_directory / "c" / f"{case_index:02d}r{repetition}"
                try:
                    outcome = self._run_case(case, arm, case_output)
                    privacy_passed, privacy_detail = _prompt_privacy(
                        self.suite, case, outcome
                    )
                    result = _score_case(
                        benchmark_run_id=store.run_id,
                        suite=self.suite,
                        case=case,
                        arm=arm,
                        repetition=repetition,
                        outcome=outcome,
                        expected_case_output=case_output,
                        privacy_passed=privacy_passed,
                        privacy_detail=privacy_detail,
                    )
                except Exception as exc:
                    result = BenchmarkCaseResult(
                        benchmark_run_id=store.run_id,
                        suite_hash=str(self.suite.suite_hash),
                        case_id=case.case_id,
                        case_hash=str(case.sealed_hash),
                        public_hash=case.public_hash(),
                        expected_hash=case.expected_hash(),
                        family=case.family,
                        task_kind=case.task_kind,
                        arm_id=arm.arm_id,
                        arm_role=arm.role,
                        arm_config_hash=arm.content_hash(),
                        repetition=repetition,
                        expected_status=case.expected_status,
                        observed_status="exception",
                        explicit_first_round_no_result=False,
                        outer_claimed_validated=False,
                        evidence_validated=False,
                        evidence_detail=f"{type(exc).__name__}: {exc}",
                        holdout=HoldoutReport(status="error", detail="case execution raised"),
                        privacy_passed=False,
                        privacy_detail="privacy audit did not complete",
                        false_promotion=False,
                        exact_terminal_match=False,
                        control_passed=False,
                        infrastructure_failure=True,
                        round_count=0,
                        candidate_count=0,
                        private_rejection_count=0,
                        runtime_invocations=[],
                        metrics={
                            "simulated_usage": not arm.live_inference,
                            "elapsed_ms": 0,
                            "cost_usd": None,
                            "served_model_attested": False,
                        },
                        exploration_directory="",
                        candidate_run_directories=[],
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                result_ref = store.put_artifact("benchmark_case_result", result)
                store.emit(
                    "benchmark_case_scored",
                    {
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "observed_status": result.observed_status,
                        "control_passed": result.control_passed,
                        "false_promotion": result.false_promotion,
                        "result": result_ref.model_dump(mode="json"),
                    },
                )
                results.append(result)

        aggregate = aggregate_results(
            self.suite,
            arm,
            results,
            benchmark_run_id=store.run_id,
            selected_case_ids=selected_ids,
            repetitions=repetitions,
        )
        store.put_artifact("benchmark_aggregate", aggregate)
        # The full sealed suite is persisted only after every Explorer call has finished.
        store.put_artifact("benchmark_sealed_suite", self.suite)
        report_path = _write_report(
            store.run_directory / "benchmark_report.md",
            self.suite,
            arm,
            aggregate,
            results,
        )
        store.put_artifact(
            "benchmark_report", {"markdown": report_path.read_text(encoding="utf-8")}
        )
        store.emit(
            "benchmark_completed",
            {
                "run_validity": aggregate.run_validity,
                "control_pass_rate": aggregate.control_pass_rate,
                "false_promotion_count": aggregate.false_promotion_count,
                "harness_integrity_passed": aggregate.harness_integrity_passed,
            },
        )
        event_chain_verified = store.verify_event_chain()
        if not event_chain_verified:
            raise RuntimeError("benchmark event hash chain failed")
        return BenchmarkRunSummary(
            run_id=store.run_id,
            run_directory=str(store.run_directory),
            suite_hash=str(self.suite.suite_hash),
            arm=arm,
            selected_case_ids=selected_ids,
            repetitions=repetitions,
            aggregate=aggregate,
            report_path=str(report_path),
            event_chain_verified=True,
        )
