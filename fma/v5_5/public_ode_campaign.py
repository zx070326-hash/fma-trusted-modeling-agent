"""Code-owned public-only runner for the frozen V5.5 scalar-ODE campaign.

The runner verifies the complete public launch, evaluates the frozen candidate
graph on public observations, obtains authenticated V5.3 replay evidence, and
asks the deterministic V5.4 gate for an eligibility decision.  It accepts no
private-target or provenance decryption key and cannot consume a private
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import scipy
from pydantic import model_validator

from fma.hashing import canonical_json, sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_2.ode_system import (
    ODEFamilyV52,
    ODEThresholdsV52,
    ODETimeSeriesSnapshotV52,
    _parameter_vector,
    _predict,
    fit_ode_v52,
)
from fma.v5_3.custody import PrivateScoreContractV53
from fma.v5_3.ode_forecast import (
    ODEForecastBundleV53,
    ODEForecastPlanV53,
    ODEForecastReplayAuthorityV53,
    build_ode_forecast_bundle_v53,
    run_authenticated_ode_forecast_replays_v53,
)
from fma.v5_4.public_eligibility import (
    PairedForecastLossV54,
    PublicEligibilityAssessmentV54,
    PublicEligibilityAuthorityV54,
    PublicEligibilityContractV54,
    PublicEligibilityInputV54,
    assess_public_eligibility_v54,
)

from .campaign_protocol import (
    CandidateSelectionPolicyV55,
    ProspectiveCampaignProtocolV55,
    PublicLaunchBindingV55,
)
from .split_custody import (
    EncryptedCustodyEnvelopeV55,
    SplitCustodyAttestationV55,
    verify_split_custody_attestation_signature_v55,
)
from .world_bank_custodian import (
    WorldBankPublicManifestV55,
    WorldBankPublicTaskPacketV55,
    WorldBankSelectionSpecV55,
)


FAMILIES: tuple[ODEFamilyV52, ...] = (
    "constant",
    "exponential",
    "gompertz",
    "logistic",
)
HORIZONS = (1, 2, 3, 4)
ORIGINS = tuple(range(12, 25))
COMPLEXITY_ORDER = {family: index for index, family in enumerate(FAMILIES)}
EXPECTED_PUBLIC_FILES = {
    "campaign_protocol_v55.json",
    "candidate_selection_policy_v55.json",
    "custody_public_key.pem",
    "ode_forecast_plan_v53.json",
    "ode_thresholds_v52.json",
    "private_score_contract_v53.json",
    "private_target_envelope_v55.json",
    "public_eligibility_contract_v54.json",
    "public_launch_binding_v55.json",
    "public_snapshot_v52.json",
    "public_task_packet_v55.json",
    "source_provenance_envelope_v55.json",
    "split_custody_attestation_v55.json",
    "world_bank_selection_spec_v55.json",
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class VerifiedPublicLaunchV55:
    protocol: ProspectiveCampaignProtocolV55
    policy: CandidateSelectionPolicyV55
    launch_binding: PublicLaunchBindingV55
    snapshot: ODETimeSeriesSnapshotV52
    thresholds: ODEThresholdsV52
    forecast_plan: ODEForecastPlanV53
    eligibility_contract: PublicEligibilityContractV54
    score_contract: PrivateScoreContractV53
    task_packet: WorldBankPublicTaskPacketV55
    selection_spec: WorldBankSelectionSpecV55
    manifest: WorldBankPublicManifestV55
    target_envelope: EncryptedCustodyEnvelopeV55
    provenance_envelope: EncryptedCustodyEnvelopeV55
    split_attestation: SplitCustodyAttestationV55
    verification: dict[str, object]


class PublicODECampaignResultV55(StrictModel):
    """Terminal public-only projection; never a private qualification."""

    schema_version: Literal["5.5-public-ode-campaign-result"] = (
        "5.5-public-ode-campaign-result"
    )
    task_id: Identifier
    public_manifest_hash: Sha256
    candidate_evidence_hash: Sha256
    forecast_bundle_hash: Sha256
    eligibility_input_hash: Sha256
    eligibility_assessment_hash: Sha256
    eligibility_receipt_hash: Sha256
    rolling_selected_family: Identifier
    v5_3_selected_family: Identifier
    selected_family_alignment: bool
    all_registered_candidate_grids_complete: bool
    public_scientific_acceptance_verified: bool
    public_gate_decision: Literal["ELIGIBLE", "ABSTAIN"]
    graph_recovery_triggered: bool
    private_evaluation_status: Literal[
        "BLOCKED_NOT_RUN",
        "NOT_AUTHORIZED_NOT_RUN",
    ]
    private_evaluations_consumed: Literal[0] = 0
    private_evaluation_budget: Literal[1] = 1
    private_target_plaintext_accessed: Literal[False] = False
    private_target_key_accessed: Literal[False] = False
    source_provenance_plaintext_accessed: Literal[False] = False
    source_provenance_key_accessed: Literal[False] = False
    same_host_role_separation_only: Literal[True] = True
    external_host_established: Literal[False] = False
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    result_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "PublicODECampaignResultV55":
        expected_private_status = (
            "BLOCKED_NOT_RUN"
            if self.public_gate_decision == "ELIGIBLE"
            else "NOT_AUTHORIZED_NOT_RUN"
        )
        if self.private_evaluation_status != expected_private_status:
            raise ValueError("private status differs from the public gate decision")
        if self.result_hash and self.result_hash != self.content_hash():
            raise ValueError("public campaign result hash differs")
        return self

    def content_hash(self) -> str:
        return sha256_value(self.model_dump(mode="json", exclude={"result_hash"}))

    @classmethod
    def seal(cls, **data: object) -> "PublicODECampaignResultV55":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"result_hash"})
        payload["result_hash"] = draft.content_hash()
        return cls(**payload)


def verify_public_launch_v55(public_dir: Path) -> VerifiedPublicLaunchV55:
    """Verify all public bytes and bindings before any local authority key read."""

    root = public_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "public_manifest_v55.json"
    manifest = WorldBankPublicManifestV55.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if not manifest.manifest_hash or manifest.manifest_hash != manifest.content_hash():
        raise ValueError("public manifest is not sealed")
    declared = {entry.path for entry in manifest.files}
    if declared != EXPECTED_PUBLIC_FILES:
        raise ValueError("public manifest file set differs from the frozen launch")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual != EXPECTED_PUBLIC_FILES | {"public_manifest_v55.json"}:
        raise ValueError("public launch directory contains missing or extra files")

    verified_files: list[dict[str, object]] = []
    for entry in manifest.files:
        path = root / entry.path
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("public manifest contains an unsafe path")
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if len(payload) != entry.size_bytes or actual_hash != entry.sha256:
            raise ValueError(f"public artifact differs: {entry.path}")
        verified_files.append(
            {
                "path": entry.path,
                "sha256": actual_hash,
                "size_bytes": len(payload),
            }
        )

    protocol = ProspectiveCampaignProtocolV55.model_validate_json(
        (root / "campaign_protocol_v55.json").read_text(encoding="utf-8")
    )
    policy = CandidateSelectionPolicyV55.model_validate_json(
        (root / "candidate_selection_policy_v55.json").read_text(
            encoding="utf-8"
        )
    )
    launch_binding = PublicLaunchBindingV55.model_validate_json(
        (root / "public_launch_binding_v55.json").read_text(encoding="utf-8")
    )
    snapshot = ODETimeSeriesSnapshotV52.model_validate_json(
        (root / "public_snapshot_v52.json").read_text(encoding="utf-8")
    )
    thresholds = ODEThresholdsV52.model_validate_json(
        (root / "ode_thresholds_v52.json").read_text(encoding="utf-8")
    )
    forecast_plan = ODEForecastPlanV53.model_validate_json(
        (root / "ode_forecast_plan_v53.json").read_text(encoding="utf-8")
    )
    eligibility_contract = PublicEligibilityContractV54.model_validate_json(
        (root / "public_eligibility_contract_v54.json").read_text(
            encoding="utf-8"
        )
    )
    score_contract = PrivateScoreContractV53.model_validate_json(
        (root / "private_score_contract_v53.json").read_text(encoding="utf-8")
    )
    task_packet = WorldBankPublicTaskPacketV55.model_validate_json(
        (root / "public_task_packet_v55.json").read_text(encoding="utf-8")
    )
    selection_spec = WorldBankSelectionSpecV55.model_validate_json(
        (root / "world_bank_selection_spec_v55.json").read_text(
            encoding="utf-8"
        )
    )
    target_envelope = EncryptedCustodyEnvelopeV55.model_validate_json(
        (root / "private_target_envelope_v55.json").read_text(encoding="utf-8")
    )
    provenance_envelope = EncryptedCustodyEnvelopeV55.model_validate_json(
        (root / "source_provenance_envelope_v55.json").read_text(
            encoding="utf-8"
        )
    )
    split_attestation = SplitCustodyAttestationV55.model_validate_json(
        (root / "split_custody_attestation_v55.json").read_text(
            encoding="utf-8"
        )
    )
    custody_public_key = (root / "custody_public_key.pem").read_bytes()

    protocol.assert_sealed()
    policy.assert_sealed()
    launch_binding.assert_sealed()
    snapshot.assert_sealed()
    thresholds.assert_sealed()
    forecast_plan.assert_compatible(snapshot, thresholds)
    eligibility_contract.assert_sealed()
    score_contract.assert_sealed()
    task_packet.assert_sealed()
    selection_spec.assert_sealed()
    target_envelope.assert_sealed()
    provenance_envelope.assert_sealed()
    split_attestation.assert_sealed()

    task_id = manifest.task_id
    if not (
        task_id
        == policy.task_id
        == launch_binding.task_id
        == snapshot.task_id
        == forecast_plan.task_id
        == eligibility_contract.task_id
        == score_contract.case_id
        == task_packet.task_id
        == target_envelope.case_id
        == provenance_envelope.case_id
        == split_attestation.case_id
    ):
        raise ValueError("public launch task bindings differ")
    if not (
        manifest.protocol_hash
        == protocol.protocol_hash
        == policy.protocol_hash
        == launch_binding.protocol_hash
        == task_packet.protocol_hash
        == selection_spec.protocol_hash
        == score_contract.protocol_hash
        == split_attestation.protocol_hash
    ):
        raise ValueError("public launch protocol bindings differ")
    if not (
        manifest.selection_spec_hash
        == selection_spec.selection_spec_hash
        == task_packet.selection_spec_hash
    ):
        raise ValueError("public launch selection-spec bindings differ")
    if list(policy.candidate_families) != list(FAMILIES):
        raise ValueError("candidate family registry differs from the frozen registry")
    if (
        policy.selection_rule
        != "aggregate_normalized_mae_then_rmse_then_complexity"
        or policy.policy_hash
        != launch_binding.candidate_policy_hash
        != eligibility_contract.candidate_selection_rule_hash
    ):
        raise ValueError("candidate selection policy binding differs")
    if not (
        protocol.baseline_id
        == policy.baseline_id
        == launch_binding.baseline_id
        == eligibility_contract.baseline_id
        == "persistence_last_value"
    ):
        raise ValueError("baseline identity differs")
    if (
        launch_binding.public_eligibility_contract_hash
        != eligibility_contract.contract_hash
    ):
        raise ValueError("eligibility contract is not bound to the launch")
    if selection_spec.ode_threshold_hash != thresholds.threshold_hash:
        raise ValueError("selection spec is bound to different ODE thresholds")
    if score_contract.forecast_plan_hash != forecast_plan.plan_hash:
        raise ValueError("private score contract is bound to another forecast plan")
    if split_attestation.score_contract_hash != score_contract.contract_hash:
        raise ValueError("split custody is bound to another score contract")
    if not (
        split_attestation.private_target_envelope_hash
        == target_envelope.envelope_hash
        and split_attestation.source_provenance_envelope_hash
        == provenance_envelope.envelope_hash
    ):
        raise ValueError("split custody envelope bindings differ")
    if target_envelope.domain != "private_targets":
        raise ValueError("private-target envelope domain differs")
    if provenance_envelope.domain != "source_provenance":
        raise ValueError("source-provenance envelope domain differs")
    if not verify_split_custody_attestation_signature_v55(
        attestation=split_attestation,
        custody_public_key_pem=custody_public_key,
    ):
        raise ValueError("split custody signature is invalid")
    if not (
        manifest.fixture_only
        == snapshot.fixture_only
        == task_packet.fixture_only
        == selection_spec.fixture_only
    ):
        raise ValueError("fixture flags differ")

    packet_times = [item.time for item in task_packet.public_observations]
    packet_values = [item.value for item in task_packet.public_observations]
    if packet_times != snapshot.times or not np.allclose(
        packet_values,
        snapshot.observations,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("public packet and typed snapshot differ")
    if [item.target_id for item in task_packet.targets] != [
        item.target_id for item in forecast_plan.targets
    ] or [item.time for item in task_packet.targets] != [
        item.time for item in forecast_plan.targets
    ]:
        raise ValueError("public packet and forecast plan targets differ")

    return VerifiedPublicLaunchV55(
        protocol=protocol,
        policy=policy,
        launch_binding=launch_binding,
        snapshot=snapshot,
        thresholds=thresholds,
        forecast_plan=forecast_plan,
        eligibility_contract=eligibility_contract,
        score_contract=score_contract,
        task_packet=task_packet,
        selection_spec=selection_spec,
        manifest=manifest,
        target_envelope=target_envelope,
        provenance_envelope=provenance_envelope,
        split_attestation=split_attestation,
        verification={
            "public_manifest_sealed": True,
            "complete_exact_file_set": True,
            "all_manifest_hashes_and_sizes_verified": True,
            "all_typed_seals_verified": True,
            "split_custody_signature_verified": True,
            "public_snapshot_packet_alignment_verified": True,
            "private_target_ciphertext_hashed_but_not_decrypted": True,
            "source_provenance_ciphertext_hashed_but_not_decrypted": True,
            "verified_files": verified_files,
        },
    )


def _evaluate_candidate(
    *,
    family: ODEFamilyV52 | None,
    times: np.ndarray,
    values: np.ndarray,
    scale: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for origin in ORIGINS:
        train_times = times[:origin]
        train_values = values[:origin]
        try:
            if family is None:
                predictions = np.repeat(train_values[-1], len(HORIZONS))
                fit_record: dict[str, object] = {
                    "family": "persistence_last_value",
                    "optimizer_converged": True,
                    "last_public_value": float(train_values[-1]),
                }
            else:
                fit = fit_ode_v52(family, train_times, train_values)
                predictions = _predict(
                    family,
                    np.concatenate(([train_times[0]], times[origin : origin + 4])),
                    float(train_values[0]),
                    _parameter_vector(fit),
                )[1:]
                fit_record = fit.model_dump(mode="json")
            if len(predictions) != len(HORIZONS) or not np.all(
                np.isfinite(predictions)
            ):
                raise ValueError("candidate predictions are incomplete or non-finite")
            fits.append({"origin": origin, "fit": fit_record})
            for offset, horizon in enumerate(HORIZONS):
                observed = float(values[origin + offset])
                prediction = float(predictions[offset])
                absolute_error = abs(prediction - observed)
                rows.append(
                    {
                        "origin": origin,
                        "horizon": horizon,
                        "prediction": prediction,
                        "observed": observed,
                        "normalized_absolute_loss": absolute_error / scale,
                        "normalized_squared_error": (
                            (prediction - observed) / scale
                        )
                        ** 2,
                    }
                )
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "origin": origin,
                    "error_type": type(exc).__name__,
                    "error_message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                }
            )

    complete = len(rows) == len(ORIGINS) * len(HORIZONS) and not failures
    losses = np.asarray(
        [float(row["normalized_absolute_loss"]) for row in rows],
        dtype=float,
    )
    squared = np.asarray(
        [float(row["normalized_squared_error"]) for row in rows],
        dtype=float,
    )
    return {
        "candidate_id": family or "persistence_last_value",
        "family": family or "persistence_last_value",
        "training_window_rule": "all_available_public_prefix",
        "grid_complete": complete,
        "aggregate_normalized_mae": (
            float(np.mean(losses)) if complete else None
        ),
        "aggregate_normalized_rmse": (
            float(np.sqrt(np.mean(squared))) if complete else None
        ),
        "rows": rows,
        "fits_by_origin": fits,
        "failures": failures,
    }


def _select_candidate(
    candidates: dict[str, dict[str, object]],
) -> tuple[str, list[dict[str, object]]]:
    complete = [
        item for item in candidates.values() if bool(item["grid_complete"])
    ]
    if not complete:
        raise RuntimeError("no frozen candidate produced a complete public grid")
    ranking = sorted(
        complete,
        key=lambda item: (
            float(item["aggregate_normalized_mae"]),
            float(item["aggregate_normalized_rmse"]),
            COMPLEXITY_ORDER[str(item["family"])],
        ),
    )
    return str(ranking[0]["family"]), [
        {
            "rank": index,
            "family": item["family"],
            "aggregate_normalized_mae": item["aggregate_normalized_mae"],
            "aggregate_normalized_rmse": item["aggregate_normalized_rmse"],
            "complexity_order": COMPLEXITY_ORDER[str(item["family"])],
        }
        for index, item in enumerate(ranking, start=1)
    ]


def _refit_selected_family(
    *,
    family: ODEFamilyV52,
    snapshot: ODETimeSeriesSnapshotV52,
    forecast_plan: ODEForecastPlanV53,
) -> dict[str, object]:
    times = np.asarray(snapshot.times, dtype=float)
    values = np.asarray(snapshot.observations, dtype=float)
    target_times = np.asarray(
        [item.time for item in forecast_plan.targets],
        dtype=float,
    )
    fit = fit_ode_v52(family, times, values)
    predictions = _predict(
        family,
        np.concatenate(([times[0]], target_times)),
        float(values[0]),
        _parameter_vector(fit),
    )[1:]
    if not np.all(np.isfinite(predictions)) or not np.all(predictions > 0):
        raise ValueError("selected-family final predictions are not positive finite")
    return {
        "selected_family": family,
        "family_locked_before_refit": True,
        "fit": fit.model_dump(mode="json"),
        "predictions": [
            {
                "target_id": target.target_id,
                "time": target.time,
                "value": float(predictions[index]),
            }
            for index, target in enumerate(forecast_plan.targets)
        ],
        "prediction_registration_created": False,
        "private_evaluation_performed": False,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }


def _candidate_graph(
    *,
    candidates: dict[str, dict[str, object]],
    baseline: dict[str, object],
    selected_family: str,
) -> dict[str, object]:
    baseline_mae = float(baseline["aggregate_normalized_mae"])
    simple_success = any(
        bool(candidates[family]["grid_complete"])
        and float(candidates[family]["aggregate_normalized_mae"]) < baseline_mae
        for family in ("constant", "exponential")
    )
    recovery_triggered = not simple_success
    return {
        "schema_version": "5.5-public-candidate-graph",
        "nodes": [
            {
                "node_id": "initial-simple-branch",
                "families": ["constant", "exponential"],
                "status": "EVALUATED",
            },
            {
                "node_id": "saturation-recovery-branch",
                "families": ["gompertz", "logistic"],
                "status": "RECOVERY_TRIGGERED"
                if recovery_triggered
                else "PROSPECTIVE_COMPARISON_EVALUATED",
            },
            {
                "node_id": "frozen-selection",
                "selected_family": selected_family,
                "status": "LOCKED_BEFORE_ALL_PUBLIC_REFIT",
            },
        ],
        "edges": [
            {
                "from": "initial-simple-branch",
                "to": "saturation-recovery-branch",
                "condition": (
                    "both_simple_families_failed_to_improve_persistence"
                    if recovery_triggered
                    else "prospective_all_family_comparison"
                ),
            },
            {
                "from": "saturation-recovery-branch",
                "to": "frozen-selection",
                "condition": (
                    "aggregate_normalized_mae_then_rmse_then_complexity"
                ),
            },
        ],
        "recovery_triggered": recovery_triggered,
        "unregistered_family_introduced": False,
        "trailing_window_variant_introduced": False,
        "post_result_threshold_change": False,
    }


def _render_report(
    *,
    result: PublicODECampaignResultV55,
    candidate_evidence: dict[str, object],
    assessment: PublicEligibilityAssessmentV54,
    bundle: ODEForecastBundleV53,
) -> str:
    ranking = candidate_evidence["ranking"]
    lines = [
        "# Iteration 34 public-only ODE campaign",
        "",
        f"- Task: `{result.task_id}`",
        f"- Frozen rolling-origin selection: `{result.rolling_selected_family}`",
        f"- Independent V5.3 selection: `{result.v5_3_selected_family}`",
        f"- Family alignment: `{str(result.selected_family_alignment).lower()}`",
        f"- V5.3 scientific acceptance: `{str(bundle.scientific_acceptance).lower()}`",
        f"- V5.4 public decision: `{result.public_gate_decision}`",
        f"- Private evaluation: `{result.private_evaluation_status}` (0/1 consumed)",
        "",
        "## Frozen candidate ranking",
        "",
        "| Rank | Family | aggregate nMAE | aggregate nRMSE |",
        "|---:|---|---:|---:|",
    ]
    for row in ranking:  # type: ignore[union-attr]
        lines.append(
            "| {rank} | {family} | {mae:.9f} | {rmse:.9f} |".format(
                rank=row["rank"],
                family=row["family"],
                mae=row["aggregate_normalized_mae"],
                rmse=row["aggregate_normalized_rmse"],
            )
        )
    lines.extend(
        [
            "",
            "## Gate evidence",
            "",
            f"- Public mean advantage: `{assessment.metrics.mean_advantage:.9f}`",
            (
                "- Multiplicity-adjusted bootstrap lower bound: "
                f"`{assessment.metrics.selection_adjusted_bootstrap_lower_bound:.9f}`"
            ),
            f"- Origin win fraction: `{assessment.metrics.origin_win_fraction:.6f}`",
            "- Every V5.4 check must pass; failed checks are listed below.",
            "",
        ]
    )
    failed = [name for name, passed in assessment.checks.items() if not passed]
    lines.extend([f"- `{name}`" for name in failed] or ["- None"])
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Public ciphertext hashes were verified, but neither ciphertext was decrypted.",
            "- No private-target or source-provenance key is accepted by this runner.",
            "- Same-host role separation is local protocol evidence, not an external host.",
            "- No private score, scientific qualification, causal claim, or real-world action is authorized.",
            "",
        ]
    )
    return "\n".join(lines)


def run_public_ode_campaign_v55(
    *,
    public_dir: Path,
    replay_secret_path: Path,
    eligibility_private_key_path: Path,
    replay_key_id: str,
    eligibility_key_id: str,
    output_dir: Path,
) -> PublicODECampaignResultV55:
    """Run public science and eligibility, with no private-evaluation capability."""

    launch = verify_public_launch_v55(public_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    times = np.asarray(launch.snapshot.times, dtype=float)
    values = np.asarray(launch.snapshot.observations, dtype=float)
    scale = float(launch.score_contract.quality_scale)
    baseline = _evaluate_candidate(
        family=None,
        times=times,
        values=values,
        scale=scale,
    )
    candidates = {
        family: _evaluate_candidate(
            family=family,
            times=times,
            values=values,
            scale=scale,
        )
        for family in FAMILIES
    }
    selected_family, ranking = _select_candidate(candidates)
    graph = _candidate_graph(
        candidates=candidates,
        baseline=baseline,
        selected_family=selected_family,
    )
    candidate_evidence = {
        "schema_version": "5.5-public-rolling-candidate-evidence",
        "task_id": launch.snapshot.task_id,
        "public_manifest_hash": launch.manifest.manifest_hash,
        "selection_policy_hash": launch.policy.policy_hash,
        "origin_sizes": list(ORIGINS),
        "horizons": list(HORIZONS),
        "quality_scale": scale,
        "candidate_search_count": len(FAMILIES),
        "baseline": baseline,
        "candidates": candidates,
        "ranking": ranking,
        "selected_family": selected_family,
        "graph": graph,
        "selection_uses_public_data_only": True,
        "private_target_plaintext_accessed": False,
        "source_provenance_plaintext_accessed": False,
        "scientific_qualification_granted": False,
        "real_world_action_authorized": False,
    }
    candidate_evidence_hash = sha256_value(candidate_evidence)
    provisional = _refit_selected_family(
        family=selected_family,  # type: ignore[arg-type]
        snapshot=launch.snapshot,
        forecast_plan=launch.forecast_plan,
    )

    temporary_parent = output_dir.resolve().parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=temporary_parent,
        )
    )
    try:
        replay_input_path = temporary / "replay_input_v53.json"
        _write_new(
            replay_input_path,
            _json_bytes(
                {
                    "snapshot": launch.snapshot.model_dump(mode="json"),
                    "thresholds": launch.thresholds.model_dump(mode="json"),
                    "forecast_plan": launch.forecast_plan.model_dump(mode="json"),
                }
            ),
        )
        replay_secret = replay_secret_path.read_bytes()
        if len(replay_secret) < 32:
            raise ValueError("replay authority secret must contain at least 32 bytes")
        replay_authority = ODEForecastReplayAuthorityV53(
            key_id=replay_key_id,
            secret=replay_secret,
        )
        replay_receipts = run_authenticated_ode_forecast_replays_v53(
            replay_input_path,
            authority=replay_authority,
        )
        bundle = build_ode_forecast_bundle_v53(
            snapshot=launch.snapshot,
            thresholds=launch.thresholds,
            forecast_plan=launch.forecast_plan,
            replay_receipts=replay_receipts,
            replay_authority=replay_authority,
        )
        v53_selected_family = bundle.development_bundle.selected_candidate_id
        alignment = selected_family == v53_selected_family
        all_grids_complete = all(
            bool(item["grid_complete"]) for item in candidates.values()
        )
        public_scientific_acceptance = bool(
            bundle.scientific_acceptance
            and alignment
            and all_grids_complete
            and len(candidates) == len(FAMILIES)
        )

        selected = candidates[selected_family]
        paired_by_coordinate = {
            (int(row["origin"]), int(row["horizon"])): row
            for row in baseline["rows"]  # type: ignore[union-attr]
        }
        paired_rows = [
            PairedForecastLossV54(
                origin=int(row["origin"]),
                horizon=int(row["horizon"]),
                candidate_loss=float(row["normalized_absolute_loss"]),
                baseline_loss=float(
                    paired_by_coordinate[
                        (int(row["origin"]), int(row["horizon"]))
                    ]["normalized_absolute_loss"]
                ),
            )
            for row in selected["rows"]  # type: ignore[union-attr]
        ]
        source_hashes = sorted(
            {
                str(launch.manifest.manifest_hash),
                str(launch.snapshot.snapshot_hash),
                str(launch.thresholds.threshold_hash),
                str(launch.forecast_plan.plan_hash),
                str(launch.policy.policy_hash),
                str(bundle.bundle_hash),
                candidate_evidence_hash,
                *[
                    str(receipt.receipt_hash) for receipt in replay_receipts
                ],
            }
        )
        eligibility_input = PublicEligibilityInputV54.seal(
            task_id=launch.snapshot.task_id,
            contract_hash=launch.eligibility_contract.contract_hash,
            candidate_id=selected_family,
            baseline_id=launch.policy.baseline_id,
            candidate_search_count=len(FAMILIES),
            public_scientific_acceptance_verified=(
                public_scientific_acceptance
            ),
            fixture_only=launch.snapshot.fixture_only,
            source_artifact_hashes=source_hashes,
            rows=paired_rows,
        )
        assessment = assess_public_eligibility_v54(
            contract=launch.eligibility_contract,
            evidence=eligibility_input,
        )
        eligibility_authority = PublicEligibilityAuthorityV54(
            key_id=eligibility_key_id,
            private_key_pem=eligibility_private_key_path.read_bytes(),
        )
        receipt = eligibility_authority.issue(
            receipt_id=f"{launch.snapshot.task_id}-public-gate",
            assessment=assessment,
        )
        if not eligibility_authority.verify(
            assessment=assessment,
            receipt=receipt,
        ):
            raise RuntimeError("fresh public eligibility receipt did not verify")

        result = PublicODECampaignResultV55.seal(
            task_id=launch.snapshot.task_id,
            public_manifest_hash=launch.manifest.manifest_hash,
            candidate_evidence_hash=candidate_evidence_hash,
            forecast_bundle_hash=bundle.bundle_hash,
            eligibility_input_hash=eligibility_input.input_hash,
            eligibility_assessment_hash=assessment.assessment_hash,
            eligibility_receipt_hash=receipt.receipt_hash,
            rolling_selected_family=selected_family,
            v5_3_selected_family=v53_selected_family,
            selected_family_alignment=alignment,
            all_registered_candidate_grids_complete=all_grids_complete,
            public_scientific_acceptance_verified=(
                public_scientific_acceptance
            ),
            public_gate_decision=assessment.decision,
            graph_recovery_triggered=bool(graph["recovery_triggered"]),
            private_evaluation_status=(
                "BLOCKED_NOT_RUN"
                if assessment.decision == "ELIGIBLE"
                else "NOT_AUTHORIZED_NOT_RUN"
            ),
            fixture_only=launch.snapshot.fixture_only,
        )

        artifacts: dict[str, bytes] = {
            "candidate_evidence_v55.json": _json_bytes(candidate_evidence),
            "forecast_bundle_v53.json": _json_bytes(bundle),
            "provisional_predictions_v55.json": _json_bytes(provisional),
            "public_eligibility_assessment_v54.json": _json_bytes(assessment),
            "public_eligibility_authority_public_key.pem": (
                eligibility_authority.public_key_pem
            ),
            "public_eligibility_input_v54.json": _json_bytes(eligibility_input),
            "public_eligibility_receipt_v54.json": _json_bytes(receipt),
            "public_launch_verification_v55.json": _json_bytes(
                launch.verification
            ),
            "public_ode_campaign_result_v55.json": _json_bytes(result),
            "replay_receipts_v53.json": _json_bytes(replay_receipts),
            "REPORT.md": (
                _render_report(
                    result=result,
                    candidate_evidence=candidate_evidence,
                    assessment=assessment,
                    bundle=bundle,
                ).encode("utf-8")
            ),
        }
        for name, payload in sorted(artifacts.items()):
            _write_new(temporary / name, payload)
        result_manifest = {
            "schema_version": "5.5-public-ode-result-manifest",
            "task_id": result.task_id,
            "source_module_sha256": _file_hash(Path(__file__)),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "files": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(
                    {
                        **artifacts,
                        "replay_input_v53.json": replay_input_path.read_bytes(),
                    }.items()
                )
            ],
            "private_evaluation_performed": False,
            "scientific_qualification_granted": False,
            "real_world_action_authorized": False,
        }
        result_manifest["manifest_hash"] = sha256_value(result_manifest)
        _write_new(
            temporary / "result_manifest_v55.json",
            _json_bytes(result_manifest),
        )
        # On the supported Windows host, rename fails if a concurrent writer
        # created the target after the create-once precheck.
        os.rename(temporary, output_dir.resolve())
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", required=True)
    parser.add_argument("--replay-secret", required=True)
    parser.add_argument("--eligibility-private-key", required=True)
    parser.add_argument("--replay-key-id", required=True)
    parser.add_argument("--eligibility-key-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_public_ode_campaign_v55(
        public_dir=Path(args.public_dir),
        replay_secret_path=Path(args.replay_secret),
        eligibility_private_key_path=Path(args.eligibility_private_key),
        replay_key_id=args.replay_key_id,
        eligibility_key_id=args.eligibility_key_id,
        output_dir=Path(args.output_dir),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PublicODECampaignResultV55",
    "run_public_ode_campaign_v55",
    "verify_public_launch_v55",
]
