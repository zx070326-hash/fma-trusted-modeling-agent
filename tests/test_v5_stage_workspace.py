from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fma.hashing import sha256_value
from fma.v5.check_registry import AdapterOutcomeV50, CheckRegistryV50
from fma.v5.paper import build_paper
from fma.v5.scaffold import scaffold_task_workspace
from fma.v5.stage_workspace import (
    POLICIES,
    StageWorkspaceError,
    StageWorkspaceV50,
    _evaluate_arithmetic,
)
from fma.v5.workspace_schemas import (
    AdapterExecutionReceiptV50,
    AssumptionRecordV50,
    AssumptionSetV50,
    CandidateFormalizationV50,
    CandidateSetV50,
    CheckResultV50,
    CodeManifestV50,
    DataLedgerEntryV50,
    DataLedgerV50,
    DecisionAssertionV50,
    DecisionFunctionCanaryV50,
    DecisionFunctionSpecV50,
    DecisionDossierV50,
    ModelSpecV50,
    ProcessedArtifactV50,
    ProcessedManifestV50,
    RegimeDiagnosisV50,
    ResultIndexV50,
    ResultRecordV50,
    SymbolRecordV50,
    SymbolTableV50,
    TaskWorkspaceSpecV50,
    UQClaimV50,
    UQSummaryV50,
    ValidationObligationV50,
    ValidationPlanV50,
    WorkflowProfileV50,
)


AUTHORITY_KEY = b"v5-test-authority-key-material!!"
AUTHORITY_KEY_ID = "pytest-authority"
MISSION_HASH = "1" * 64
EVIDENCE_HASH = "2" * 64
ADAPTER_HASH = "a" * 64


class FixtureScientificAdapter:
    """Deterministic computation adapter used only by the synthetic fixture."""

    def __init__(self, obligation: ValidationObligationV50) -> None:
        self.adapter_id = "fixture_domain_adapter"
        self.adapter_version = "1"
        self.check_id = obligation.check_id
        self.level = obligation.level

    def run(self, context: object) -> AdapterOutcomeV50:
        workspace_root = getattr(context, "workspace_root")
        code_manifest = CodeManifestV50.model_validate_json(
            (workspace_root / "results" / "code_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        payload = {
            "check_id": self.check_id,
            "fixture_oracle": True,
            "scientific_transfer_claim": False,
        }
        if self.level == "L0":
            payload["computation_artifact_sha256"] = (
                code_manifest.replay_receipt_hash
            )
        elif self.level == "L2":
            payload["computation_artifact_sha256"] = next(
                iter(code_manifest.toy_oracle_hashes.values())
            )
        return AdapterOutcomeV50(
            status="PASS",
            reason_code="fixture_oracle_pass",
            metrics={"fixture_score": 1.0},
            evidence_payloads=[payload],
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")  # type: ignore[union-attr]
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_workspace(
    tmp_path: Path, *, max_nodes: int = 64, max_outcomes: int = 64
) -> tuple[Path, StageWorkspaceV50]:
    root = scaffold_task_workspace(
        tmp_path / "task",
        "fixture",
        "Evaluate a synthetic forecast while preserving evidence boundaries",
    )
    spec = TaskWorkspaceSpecV50.seal(
        workspace_id="fixture",
        graph_id="v5-fixture",
        objective="Evaluate a synthetic forecast while preserving evidence boundaries",
        mission_hash=MISSION_HASH,
        evidence_snapshot_hash=EVIDENCE_HASH,
        evaluator_epoch="pytest-v1",
        profile=WorkflowProfileV50.seal(),
        evidence_scope="synthetic_fixture",
        max_nodes=max_nodes,
        max_outcomes=max_outcomes,
    )
    workspace = StageWorkspaceV50.create(
        root,
        spec,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
    )
    return root, workspace


def _write_s0(root: Path) -> None:
    _write_json(
        root / "problem" / "contract.json",
        {
            "schema_version": "5.0",
            "mission_hash": MISSION_HASH,
            "evidence_snapshot_hash": EVIDENCE_HASH,
            "question": (
                "Evaluate a synthetic forecast while preserving evidence boundaries"
            ),
        },
    )
    _write_json(
        root / "problem" / "decision_function.json",
        DecisionFunctionSpecV50.seal(
            function_id="forecast_loss",
            input_names=["prediction", "target"],
            expression="(prediction - target) ** 2",
            sense="minimize",
            output_unit="unitless",
            canaries=[
                DecisionFunctionCanaryV50(
                    canary_id="zero_error",
                    inputs={"prediction": 1.0, "target": 1.0},
                    expected=0.0,
                ),
                DecisionFunctionCanaryV50(
                    canary_id="unit_error",
                    inputs={"prediction": 2.0, "target": 1.0},
                    expected=1.0,
                ),
            ],
        ),
    )
    _write_json(
        root / "docs" / "regime.json",
        RegimeDiagnosisV50.seal(
            system_boundary="A closed synthetic scalar forecasting system.",
            state_and_memory="One observed scalar state with one-step memory.",
            uncertainty_and_data="Finite fixture noise with no real-world observations.",
            decision_and_loss="Report-only forecast evaluated by squared error.",
            query_type="prediction",
            downstream_decision="Choose whether the fixture prediction is reportable.",
            decision_function_id="forecast_loss",
            computable_decision_function="mean squared error over registered targets",
            evidence_hashes=["3" * 64],
            limitations=["Synthetic protocol fixture; no scientific transfer claim."],
        ),
    )


def _approve_reviews(
    workspace: StageWorkspaceV50, stage: str, roles: tuple[str, ...]
) -> None:
    manifest = workspace._manifest_for_stage(stage)  # exact committed inputs
    checks = workspace._latest_checks(stage, str(manifest.manifest_hash))
    allowed_inputs = sorted(
        {item.sha256 for item in manifest.files}
        | {
            str(result.result_hash)
            for result in checks.values()
            if result.result_hash is not None
        }
    )
    for index, role in enumerate(roles):
        producer_run_id = f"{stage.lower()}-producer"
        reviewer_run_id = f"{stage.lower()}-{role}-reviewer"
        producer_context_id = f"{stage.lower()}-producer-context"
        reviewer_context_id = f"{stage.lower()}-{role}-context-{index}"
        trace = workspace.commit_evidence(
            "review_transport_trace_v50",
            {
                "stage": stage,
                "role": role,
                "fixture": True,
                "producer_run_id": producer_run_id,
                "reviewer_run_id": reviewer_run_id,
                "producer_context_id": producer_context_id,
                "reviewer_context_id": reviewer_context_id,
                "context_isolated": True,
                "context_isolation_attested": True,
                "allowed_input_hashes": allowed_inputs,
            },
        )
        output = workspace.commit_evidence(
            "review_output_v50",
            {
                "stage": stage,
                "role": role,
                "verdict": "APPROVE",
                "finding_ids": [],
                "fixture": True,
            },
        )
        workspace.issue_review(
            stage=stage,
            review_id=f"{stage.lower()}-{role.replace('_', '-')}",
            role=role,
            producer_run_id=producer_run_id,
            reviewer_run_id=reviewer_run_id,
            producer_context_id=producer_context_id,
            reviewer_context_id=reviewer_context_id,
            prompt_hash=sha256_value({"stage": stage, "role": role, "prompt": 1}),
            output_schema_hash=sha256_value(
                {"stage": stage, "role": role, "schema": 1}
            ),
            allowed_input_hashes=allowed_inputs,
            transport_trace_hash=trace.sha256,
            output_artifact_hash=output.sha256,
            verdict="APPROVE",
        )


def _open_stage(
    workspace: StageWorkspaceV50,
    stage: str,
    *,
    actor: str,
    scientific_checks: list[ValidationObligationV50] | None = None,
) -> str:
    workspace.submit_stage(stage, actor=actor)
    mechanical = workspace.run_mechanical_check(stage)
    assert mechanical.status == "PASS", mechanical
    registry = CheckRegistryV50()
    for obligation in scientific_checks or []:
        if obligation.applicability == "applicable":
            registry.register(FixtureScientificAdapter(obligation))
    for obligation in scientific_checks or []:
        registry.execute(workspace, obligation)
    _approve_reviews(workspace, stage, POLICIES[stage].required_review_roles)
    evaluation = workspace.evaluate_gate(stage)
    assert evaluation.decision == "OPEN", evaluation
    assert evaluation.certificate_hash
    return evaluation.certificate_hash


def _write_s1(
    root: Path, workspace: StageWorkspaceV50
) -> ValidationPlanV50:
    assumptions = AssumptionSetV50(
        assumptions=[
            AssumptionRecordV50(
                assumption_id="stationarity",
                statement="Fixture transition law is stationary.",
                failure_consequence="Forecast calibration becomes invalid.",
                falsification_test="Compare blocked residual distributions.",
                abandon_criterion="Reject after a reproducible distribution shift.",
            )
        ]
    )
    symbols = SymbolTableV50(
        symbols=[
            SymbolRecordV50(
                symbol_id="x",
                meaning="Observed scalar state",
                unit="unitless",
                role="state",
                lower_bound=-100,
                upper_bound=100,
            ),
            SymbolRecordV50(
                symbol_id="theta",
                meaning="Transition parameter",
                unit="unitless",
                role="parameter",
                lower_bound=-1,
                upper_bound=1,
            ),
        ]
    )
    obligations = [
        ValidationObligationV50(
            check_id="l0_repro",
            stage="S3",
            level="L0",
            evidence_class="scientific_computation",
            applicability_rule="Every executable fixture must replay.",
        ),
        ValidationObligationV50(
            check_id="l1_dims",
            stage="S3",
            level="L1",
            evidence_class="scientific_computation",
            applicability_rule="Typed scalar expressions expose unit labels.",
        ),
        ValidationObligationV50(
            check_id="l2_toy",
            stage="S3",
            level="L2",
            evidence_class="scientific_computation",
            applicability_rule="The scalar solver has an analytic toy oracle.",
        ),
        ValidationObligationV50(
            check_id="l3_cross_model",
            stage="S4",
            level="L3",
            evidence_class="scientific_computation",
            applicability="not_applicable",
            applicability_rule="No second implemented model exists in this fixture.",
        ),
        ValidationObligationV50(
            check_id="l3_markov",
            stage="S4",
            level="L3",
            evidence_class="scientific_computation",
            applicability_rule="Ordered residual and history interfaces are available.",
        ),
        ValidationObligationV50(
            check_id="l4_sensitivity",
            stage="S4",
            level="L4",
            evidence_class="scientific_computation",
            applicability_rule="The fixture exposes a finite parameter range.",
        ),
    ]
    plan = ValidationPlanV50.seal(obligations=obligations, frozen_by="verifier")
    candidates = []
    for index, family in enumerate(("ar1", "mean", "trend"), start=1):
        candidates.append(
            CandidateFormalizationV50(
                candidate_id=f"candidate{index}",
                model_family=family,
                mathematical_form=f"x_next = {index} * theta * x",
                assumption_ids=["stationarity"],
                symbol_ids=["theta", "x"],
                data_requirement_ids=["observations"],
                validation_obligation_ids=sorted(
                    item.check_id for item in obligations
                ),
                abandon_criteria=["Fails the frozen fixture oracle."],
                lineage=f"independent fixture context {index}",
            )
        )
    s0_gate = workspace.current_gate("S0")
    assert s0_gate is not None
    generation_receipt_hashes = []
    for index, candidate in enumerate(candidates, start=1):
        trace = workspace.commit_evidence(
            "role_transport_trace_v50",
            {
                "role": "modeler",
                "subject_id": candidate.candidate_id,
                "input_authority_hash": s0_gate,
                "run_id": f"modeler-run-{index}",
                "context_id": f"modeler-context-{index}",
                "candidate_id": candidate.candidate_id,
                "fixture": True,
            },
        )
        output = workspace.commit_evidence(
            "role_output_v50",
            {
                "candidate_id": candidate.candidate_id,
                "candidate_hash": candidate.structural_hash(),
                "fixture": True,
            },
        )
        receipt = workspace.issue_role_execution(
            stage="S1",
            execution_id=f"modeler-execution-{index}",
            role="modeler",
            subject_id=candidate.candidate_id,
            input_authority_hash=s0_gate,
            run_id=f"modeler-run-{index}",
            context_id=f"modeler-context-{index}",
            provider="fixture",
            model="deterministic-fixture",
            prompt_hash=sha256_value(
                {"candidate": candidate.candidate_id, "prompt": 1}
            ),
            output_schema_hash=sha256_value({"candidate_schema": "5.0"}),
            transport_trace_hash=trace.sha256,
            output_artifact_hash=output.sha256,
        )
        assert receipt.receipt_hash
        generation_receipt_hashes.append(receipt.receipt_hash)
    scout_trace = workspace.commit_evidence(
        "role_transport_trace_v50",
        {
            "role": "literature_scout",
            "subject_id": "literature-map",
            "input_authority_hash": s0_gate,
            "run_id": "literature-scout-run",
            "context_id": "literature-scout-context",
            "fixture": True,
        },
    )
    scout_output = workspace.commit_evidence(
        "role_output_v50",
        {
            "subject_id": "literature-map",
            "report_hash": "c" * 64,
            "fixture": True,
        },
    )
    scout_receipt = workspace.issue_role_execution(
        stage="S1",
        execution_id="literature-scout-execution",
        role="literature_scout",
        subject_id="literature-map",
        input_authority_hash=s0_gate,
        run_id="literature-scout-run",
        context_id="literature-scout-context",
        provider="fixture",
        model="deterministic-fixture",
        prompt_hash=sha256_value({"literature_prompt": 1}),
        output_schema_hash=sha256_value({"literature_schema": "5.0"}),
        transport_trace_hash=scout_trace.sha256,
        output_artifact_hash=scout_output.sha256,
    )
    _write_json(
        root / "docs" / "candidates.json",
        CandidateSetV50(
            candidates=candidates,
            generation_receipt_hashes=sorted(generation_receipt_hashes),
            literature_scout_receipt_hash=scout_receipt.receipt_hash,
        ),
    )
    _write_json(root / "docs" / "assumptions.json", assumptions)
    _write_json(root / "docs" / "symbols.json", symbols)
    _write_json(
        root / "docs" / "model_spec.json",
        ModelSpecV50.seal(
            selected_candidate_id="candidate1",
            selected_candidate_structural_hash=candidates[0].structural_hash(),
            selection_rationale="Best fit to the synthetic one-step transition.",
            assumption_ids=["stationarity"],
            symbol_ids=["theta", "x"],
            data_requirement_ids=["observations"],
            declared_conservation_laws=[],
            declared_limit_cases=[
                "theta to zero gives a zero forecast",
                "zero state remains zero",
            ],
            identifiability_risks=["Short series confounds noise and persistence."],
        ),
    )
    _write_json(root / "docs" / "validation_plan.json", plan)
    return plan


def _write_s2(root: Path, plan: ValidationPlanV50) -> None:
    del plan
    raw = root / "data" / "raw" / "observations.csv"
    raw.write_text("t,x\n0,1\n1,0.5\n", encoding="utf-8")
    processed = root / "data" / "processed" / "observations.json"
    _write_json(processed, {"t": [0, 1], "x": [1.0, 0.5]})
    transform = root / "src" / "models" / "prepare_data.py"
    transform.write_text(
        "# Frozen fixture transform.\n"
        "def transform(rows):\n"
        "    return rows\n",
        encoding="utf-8",
    )
    raw_tree_hash = sha256_value({"observations.csv": _sha(raw)})
    processed_hash = _sha(processed)
    transform_params = {"drop_missing": False}
    _write_json(
        root / "data" / "ledger.json",
        DataLedgerV50.seal(
            raw_baseline_tree_hash=raw_tree_hash,
            entries=[
                DataLedgerEntryV50(
                    data_item_id="observations",
                    semantic_name="Synthetic observed scalar series",
                    units="unitless",
                    source_kind="local",
                    source_ref="data/raw/observations.csv",
                    raw_relative_path="data/raw/observations.csv",
                    accessed_at=datetime.now(timezone.utc),
                    license_status="fixture-generated",
                    raw_response_hash=_sha(raw),
                    transform_script_relative_path="src/models/prepare_data.py",
                    transform_script_hash=_sha(transform),
                    transform_params=transform_params,
                    transform_params_hash=sha256_value(transform_params),
                    processed_artifact_hash=processed_hash,
                )
            ],
        ),
    )
    _write_json(
        root / "data" / "processed" / "manifest.json",
        ProcessedManifestV50(
            raw_baseline_tree_hash=raw_tree_hash,
            artifacts=[
                ProcessedArtifactV50(
                    data_item_id="observations",
                    relative_path="data/processed/observations.json",
                    artifact_hash=processed_hash,
                )
            ],
        ),
    )


def _write_s3(root: Path) -> None:
    source = root / "src" / "models" / "fixture_model.py"
    source.write_text(
        "def predict(x: float, theta: float = 0.5) -> float:\n"
        "    return theta * x\n",
        encoding="utf-8",
    )
    source_tree_hash = sha256_value(
        {
            "models/fixture_model.py": _sha(source),
            "models/prepare_data.py": _sha(
                root / "src" / "models" / "prepare_data.py"
            ),
        }
    )
    environment_path = root / "results" / "environment.json"
    fermi_path = root / "results" / "fermi.json"
    toy_path = root / "checks" / "toy_fixture.json"
    replay_path = root / "checks" / "replay_observed.json"
    _write_json(
        environment_path,
        {
            "schema_version": "5.0-environment",
            "python": "pytest-fixture",
            "dependencies": [],
        },
    )
    _write_json(
        fermi_path,
        {
            "schema_version": "5.0-fermi",
            "estimate": 0.25,
            "order_of_magnitude": 0,
        },
    )
    _write_json(
        toy_path,
        {
            "schema_version": "5.0-toy",
            "maximum_absolute_error": 0.0,
            "passed": True,
        },
    )
    replay_command = "python src/models/fixture_model.py"
    _write_json(
        replay_path,
        {
            "schema_version": "5.0-replay",
            "replay_command": replay_command,
            "source_tree_hash": source_tree_hash,
            "environment_hash": _sha(environment_path),
            "random_seed": 0,
            "exit_code": 0,
            "passed": True,
        },
    )
    _write_json(
        root / "results" / "code_manifest.json",
        CodeManifestV50(
            source_tree_hash=source_tree_hash,
            environment_ref="results/environment.json",
            environment_hash=_sha(environment_path),
            replay_command=replay_command,
            replay_receipt_ref="checks/replay_observed.json",
            replay_receipt_hash=_sha(replay_path),
            random_seed=0,
            tolerance_policy="absolute tolerance 1e-12 on the analytic toy",
            fermi_estimate_ref="results/fermi.json",
            fermi_estimate_hash=_sha(fermi_path),
            toy_oracle_refs=["checks/toy_fixture.json"],
            toy_oracle_hashes={
                "checks/toy_fixture.json": _sha(toy_path)
            },
        ),
    )
    forecast_path = root / "results" / "artifacts" / "forecast.json"
    interval_path = (
        root / "results" / "artifacts" / "forecast_interval.json"
    )
    _write_json(
        forecast_path,
        {
            "schema_version": "5.0-result",
            "result_id": "forecast",
            "value": 0.25,
            "interval_low": None,
            "interval_high": None,
            "units": "unitless",
        },
    )
    _write_json(
        interval_path,
        {
            "schema_version": "5.0-result",
            "result_id": "forecast_interval",
            "value": None,
            "interval_low": 0.2,
            "interval_high": 0.3,
            "units": "unitless",
        },
    )
    _write_json(
        root / "results" / "index.json",
        ResultIndexV50(
            records=[
                ResultRecordV50(
                    result_id="forecast",
                    relative_path="results/artifacts/forecast.json",
                    artifact_hash=_sha(forecast_path),
                    value=0.25,
                ),
                ResultRecordV50(
                    result_id="forecast_interval",
                    relative_path="results/artifacts/forecast_interval.json",
                    artifact_hash=_sha(interval_path),
                    interval_low=0.2,
                    interval_high=0.3,
                ),
            ]
        ),
    )


def _write_s4(root: Path, plan: ValidationPlanV50) -> None:
    _write_json(
        root / "results" / "verification_summary.json",
        {
            "schema_version": "5.0",
            "validation_plan_hash": plan.plan_hash,
            "check_ids": [
                "l3_cross_model",
                "l3_markov",
                "l4_sensitivity",
            ],
            "scope": "synthetic_fixture",
        },
    )
    _write_json(
        root / "results" / "uq_summary.json",
        UQSummaryV50(
            claims=[
                UQClaimV50(
                    claim_id="forecast_claim",
                    result_id="forecast",
                    interval_result_id="forecast_interval",
                    support_status="in_support",
                    ensemble_disagreement=0.01,
                )
            ]
        ),
    )


def _write_s5(root: Path) -> None:
    _write_json(
        root / "results" / "decision_dossier.json",
        DecisionDossierV50(
            assertions=[
                DecisionAssertionV50(
                    assertion_id="reportability",
                    statement="The fixture forecast is reportable only as a fixture.",
                    result_ids=["forecast"],
                    uq_claim_ids=["forecast_claim"],
                )
            ],
            high_disagreement_detected=False,
            next_action="draft_report_only",
        ),
    )


def _write_s6(root: Path) -> None:
    _write_json(
        root / "results" / "values.json",
        {
            "forecast": 0.25,
            "forecast_interval_high": 0.3,
            "forecast_interval_low": 0.2,
        },
    )
    (root / "paper" / "main.template.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "Synthetic protocol fixture forecast: {{result.forecast}}.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    build_paper(root)


def _open_through_s2(
    tmp_path: Path,
) -> tuple[Path, StageWorkspaceV50, ValidationPlanV50]:
    root, workspace = _new_workspace(tmp_path)
    _write_s0(root)
    _open_stage(workspace, "S0", actor="model")
    plan = _write_s1(root, workspace)
    _open_stage(workspace, "S1", actor="model")
    _write_s2(root, plan)
    workspace.freeze_raw_inputs(actor="harness")
    _open_stage(workspace, "S2", actor="model")
    return root, workspace, plan


def test_raw_stamp_cannot_unlock_stage(tmp_path: Path) -> None:
    root, workspace = _new_workspace(tmp_path)
    _write_s0(root)
    workspace.submit_stage("S0", actor="model")
    workspace.run_mechanical_check("S0")
    (root / "gates" / "s0.stamp").write_text("PASS\n", encoding="utf-8")

    status = workspace.status()

    assert status.stage_statuses["S0"] == "awaiting_gate_evidence"
    assert status.stage_statuses["S1"] == "pending"
    assert status.current_gate_hashes == {}


def test_s2_rejects_raw_bytes_changed_after_harness_freeze(
    tmp_path: Path,
) -> None:
    root, workspace = _new_workspace(tmp_path)
    _write_s0(root)
    _open_stage(workspace, "S0", actor="model")
    plan = _write_s1(root, workspace)
    _open_stage(workspace, "S1", actor="model")
    _write_s2(root, plan)
    workspace.freeze_raw_inputs(actor="harness")
    (root / "data" / "raw" / "observations.csv").write_text(
        "t,x\n0,99\n",
        encoding="utf-8",
    )

    with pytest.raises(StageWorkspaceError, match="changed after"):
        workspace.submit_stage("S2", actor="model")


def test_workflow_presence_cannot_emit_scientific_pass() -> None:
    with pytest.raises(ValidationError, match="cannot emit a scientific PASS"):
        CheckResultV50(
            check_id="l3_markov",
            stage="S4",
            level="L3",
            evidence_class="workflow_presence",
            applicability="applicable",
            status="PASS",
            reason_code="file_exists",
            input_manifest_hash="1" * 64,
            protocol_hash="2" * 64,
            adapter_id="presence",
            adapter_version="1",
            adapter_code_hash="3" * 64,
            evidence_refs=["4" * 64],
            scope="development",
            executed_by="verifier",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            authority_key_id="test",
        )


def test_required_validation_layers_cannot_all_be_not_applicable() -> None:
    obligations = [
        ValidationObligationV50(
            check_id=f"{stage.lower()}_{level.lower()}",
            stage=stage,
            level=level,
            evidence_class="scientific_computation",
            applicability="not_applicable",
            applicability_rule="Declared irrelevant before execution.",
        )
        for stage, level in (
            ("S3", "L0"),
            ("S3", "L1"),
            ("S3", "L2"),
            ("S4", "L3"),
            ("S4", "L4"),
        )
    ]
    with pytest.raises(
        ValidationError, match="required applicable level coverage"
    ):
        ValidationPlanV50.seal(
            obligations=obligations,
            frozen_by="verifier",
        )


def test_s0_decision_function_dsl_is_computable_and_side_effect_free() -> None:
    assert _evaluate_arithmetic(
        "(prediction - target) ** 2",
        {"prediction": 2.0, "target": 1.0},
    ) == 1.0
    with pytest.raises(ValueError, match="forbidden"):
        _evaluate_arithmetic(
            "__import__('os').system('echo no')",
            {"prediction": 2.0, "target": 1.0},
        )


def test_candidate_identity_or_label_does_not_fake_structural_competition() -> None:
    first = CandidateFormalizationV50(
        candidate_id="candidateA",
        model_family="labelA",
        mathematical_form="x_next = theta * x",
        assumption_ids=["stationarity"],
        symbol_ids=["theta", "x"],
        data_requirement_ids=["observations"],
        validation_obligation_ids=["l0_repro"],
        abandon_criteria=["Fails a frozen oracle."],
        lineage="context A",
    )
    clone = first.model_copy(
        update={
            "candidate_id": "candidateB",
            "model_family": "labelB",
            "lineage": "context B",
        }
    )
    with pytest.raises(ValidationError, match="structurally distinct"):
        CandidateSetV50(candidates=[first, clone])


def test_s0_gate_is_authenticated_and_never_scientific_promotion(
    tmp_path: Path,
) -> None:
    root, workspace = _new_workspace(tmp_path)
    _write_s0(root)
    gate_hash = _open_stage(workspace, "S0", actor="model")

    assert workspace.current_gate("S0") == gate_hash
    assert workspace.graph.project_state().promotions == []
    status = workspace.status()
    assert status.scientific_qualification_granted is False
    assert status.real_world_action_authorized is False


def test_rehashed_forged_gate_fails_authentication(tmp_path: Path) -> None:
    root, workspace = _new_workspace(tmp_path)
    _write_s0(root)
    _open_stage(workspace, "S0", actor="model")
    _, certificate = workspace._artifacts_of_kind(
        "gate_certificate_v50"
    )[0]
    forged = dict(certificate)
    forged["authority_auth_tag"] = "0" * 64
    forged["certificate_hash"] = sha256_value(
        {
            key: value
            for key, value in forged.items()
            if key != "certificate_hash"
        }
    )
    from fma.v5.workspace_schemas import GateCertificateV50

    parsed = GateCertificateV50.model_validate(forged)
    assert workspace.verify_certificate(parsed) is False


def test_upstream_mutation_makes_gate_stale_and_revoke_cascades(
    tmp_path: Path,
) -> None:
    root, workspace = _new_workspace(tmp_path)
    _write_s0(root)
    old_gate = _open_stage(workspace, "S0", actor="model")
    regime = json.loads((root / "docs" / "regime.json").read_text(encoding="utf-8"))
    regime["limitations"].append("post-gate mutation")
    _write_json(root / "docs" / "regime.json", regime)

    assert workspace.current_gate("S0") is None
    assert workspace.status().stale_gate_hashes["S0"] == old_gate
    affected = workspace.invalidate_from("S0", reason="regime diagnosis changed")

    assert len(affected) == 14
    status = workspace.status()
    assert status.frontier_stages == ["S0"]
    assert status.stage_statuses["S1"] == "pending"
    assert workspace.verify()


def test_invalidation_preflights_complete_retry_budget(tmp_path: Path) -> None:
    root, workspace = _new_workspace(tmp_path, max_nodes=14)
    _write_s0(root)
    old_gate = _open_stage(workspace, "S0", actor="model")

    with pytest.raises(StageWorkspaceError, match="node budget"):
        workspace.invalidate_from("S0", reason="retry should not fit")

    assert workspace.current_gate("S0") == old_gate
    assert len(workspace.graph.project_state().nodes) == 14


def test_missing_domain_adapter_stays_needs_evidence(tmp_path: Path) -> None:
    root, workspace, plan = _open_through_s2(tmp_path)
    _write_s3(root)
    workspace.submit_stage("S3", actor="harness")
    workspace.run_mechanical_check("S3")
    registry = CheckRegistryV50()
    missing_receipts = [
        registry.execute(workspace, item)
        for item in plan.obligations
        if item.stage == "S3"
    ]
    _approve_reviews(workspace, "S3", POLICIES["S3"].required_review_roles)

    evaluation = workspace.evaluate_gate("S3")

    assert {item.status for item in missing_receipts} == {"NOT_RUN"}
    assert evaluation.decision == "NEEDS_EVIDENCE"
    assert sorted(evaluation.reasons) == [
        "l0_repro was NOT_RUN",
        "l1_dims was NOT_RUN",
        "l2_toy was NOT_RUN",
    ]
    assert workspace.current_gate("S3") is None
    with pytest.raises(
        StageWorkspaceError, match="requires a current S4 gate"
    ):
        workspace.issue_prediction_seal(
            task_id="fixture",
            training_snapshot_hash="1" * 64,
            candidate_hash="2" * 64,
            prediction_artifact_hash="3" * 64,
            external_registration_hash="4" * 64,
            external_snapshot_hash="5" * 64,
            holdout_commitment_hash="6" * 64,
        )


def test_direct_l0_l4_result_without_execution_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    root, workspace, plan = _open_through_s2(tmp_path)
    _write_s3(root)
    workspace.submit_stage("S3", actor="harness")
    obligation = next(
        item
        for item in plan.obligations
        if item.stage == "S3" and item.applicability == "applicable"
    )
    evidence = workspace.commit_evidence(
        "scientific_fixture_evidence_v50",
        {"check_id": obligation.check_id, "untrusted_direct_claim": True},
    )

    with pytest.raises(StageWorkspaceError, match="execution receipt"):
        workspace.issue_check(
            stage="S3",
            check_id=obligation.check_id,
            level=obligation.level,
            evidence_class=obligation.evidence_class,
            applicability="applicable",
            status="PASS",
            reason_code="self_reported",
            adapter_id="unbound",
            adapter_version="1",
            adapter_code_hash=ADAPTER_HASH,
            evidence_refs=[evidence.sha256],
            scope="synthetic_fixture",
        )


def test_not_run_domain_check_cannot_open_gate(tmp_path: Path) -> None:
    root, workspace, plan = _open_through_s2(tmp_path)
    _write_s3(root)
    workspace.submit_stage("S3", actor="harness")
    workspace.run_mechanical_check("S3")
    obligations = [
        item for item in plan.obligations if item.stage == "S3"
    ]
    registry = CheckRegistryV50()
    for obligation in obligations:
        registry.execute(workspace, obligation)
    _approve_reviews(workspace, "S3", POLICIES["S3"].required_review_roles)

    evaluation = workspace.evaluate_gate("S3")

    assert evaluation.decision == "NEEDS_EVIDENCE"
    assert "l0_repro was NOT_RUN" in evaluation.reasons
    assert "l1_dims was NOT_RUN" in evaluation.reasons
    assert "l2_toy was NOT_RUN" in evaluation.reasons


def test_full_fixture_runs_s0_to_s6_without_scientific_claim(
    tmp_path: Path,
) -> None:
    root, workspace, plan = _open_through_s2(tmp_path)
    _write_s3(root)
    _open_stage(
        workspace,
        "S3",
        actor="harness",
        scientific_checks=[
            item for item in plan.obligations if item.stage == "S3"
        ],
    )
    _write_s4(root, plan)
    _open_stage(
        workspace,
        "S4",
        actor="harness",
        scientific_checks=[
            item for item in plan.obligations if item.stage == "S4"
        ],
    )
    _write_s5(root)
    _open_stage(workspace, "S5", actor="model")
    _write_s6(root)
    _open_stage(workspace, "S6", actor="harness")

    status = workspace.status()

    assert set(status.current_gate_hashes) == set(
        ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
    )
    assert set(status.stage_statuses.values()) == {"gate_open"}
    assert status.frontier_stages == []
    assert status.claim_scope == "workflow_control_only"
    assert status.scientific_qualification_granted is False
    assert status.real_world_action_authorized is False
    assert workspace.graph.project_state().promotions == []
    assert workspace.verify()


def test_new_dynamic_source_file_stales_s3_gate(tmp_path: Path) -> None:
    root, workspace, plan = _open_through_s2(tmp_path)
    _write_s3(root)
    gate_hash = _open_stage(
        workspace,
        "S3",
        actor="harness",
        scientific_checks=[
            item for item in plan.obligations if item.stage == "S3"
        ],
    )

    (root / "src" / "models" / "late_patch.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    assert workspace.current_gate("S3") is None
    assert workspace.status().stale_gate_hashes["S3"] == gate_hash


def test_historical_adapter_receipts_survive_s3_retry(
    tmp_path: Path,
) -> None:
    root, workspace, plan = _open_through_s2(tmp_path)
    _write_s3(root)
    _open_stage(
        workspace,
        "S3",
        actor="harness",
        scientific_checks=[
            item for item in plan.obligations if item.stage == "S3"
        ],
    )
    old_receipt_hashes = {
        item.receipt_hash
        for _, item in workspace._artifacts_of_kind(
            "adapter_execution_receipt_v50",
            AdapterExecutionReceiptV50,
        )
    }

    workspace.invalidate_from("S3", reason="repeat scientific computation")
    _open_stage(
        workspace,
        "S3",
        actor="harness",
        scientific_checks=[
            item for item in plan.obligations if item.stage == "S3"
        ],
    )
    all_receipts = [
        item
        for _, item in workspace._artifacts_of_kind(
            "adapter_execution_receipt_v50",
            AdapterExecutionReceiptV50,
        )
    ]

    assert old_receipt_hashes < {item.receipt_hash for item in all_receipts}
    assert all(
        workspace.verify_adapter_execution(item) for item in all_receipts
    )
    assert workspace.verify()


def test_open_existing_replays_and_wrong_key_fails_auth(tmp_path: Path) -> None:
    root, workspace = _new_workspace(tmp_path)
    with pytest.raises(Exception, match="verification"):
        StageWorkspaceV50.open_existing(
            root,
            authority_key=b"x" * 32,
            authority_key_id="replacement-key",
        )
    _write_s0(root)
    _open_stage(workspace, "S0", actor="model")

    reopened = StageWorkspaceV50.open_existing(
        root,
        authority_key=AUTHORITY_KEY,
        authority_key_id=AUTHORITY_KEY_ID,
    )
    assert reopened.verify()
    assert reopened.current_gate("S0") == workspace.current_gate("S0")

    with pytest.raises(Exception, match="verification"):
        StageWorkspaceV50.open_existing(
            root,
            authority_key=b"x" * 32,
            authority_key_id=AUTHORITY_KEY_ID,
        )
