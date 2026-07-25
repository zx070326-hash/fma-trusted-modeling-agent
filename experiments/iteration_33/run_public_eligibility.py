"""Replay and freeze the I33 V5.4 public-eligibility assessment.

This program reads only the frozen I33 public packet and public modeler
artifacts.  It verifies their manifests, typed seals, custody signature, and
paired-loss arithmetic before constructing the deterministic V5.4 evidence
and assessment.  It cannot sign a gate receipt, request private evaluation,
read private targets, or grant scientific qualification.
"""

# ruff: noqa: E402

from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

ITERATION = Path(__file__).resolve().parent
REPO_ROOT = ITERATION.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fma.hashing import canonical_json
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52
from fma.v5_3.ode_forecast import ODEForecastBundleV53
from fma.v5_4.public_eligibility import (
    PairedForecastLossV54,
    PublicEligibilityContractV54,
    PublicEligibilityInputV54,
    assess_public_eligibility_v54,
)


CAMPAIGN_ID = "i33-shadow-85d5123d710d7d19367a"
CAMPAIGN = ITERATION / "campaigns" / CAMPAIGN_ID
PUBLIC = CAMPAIGN / "public"
MODELER = CAMPAIGN / "modeler"
GATE = CAMPAIGN / "public_gate"
EVIDENCE_NAME = "public_eligibility_input_v54.json"
ASSESSMENT_NAME = "public_eligibility_assessment_v54.json"
VERIFICATION_NAME = "public_gate_verification.json"
REPORT_NAME = "GATE_RESULT.md"
MANIFEST_NAME = "public_gate_manifest.json"
GENERATED_NAMES = (
    EVIDENCE_NAME,
    ASSESSMENT_NAME,
    VERIFICATION_NAME,
    REPORT_NAME,
    MANIFEST_NAME,
)


def json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label} differs: {actual!r} != {expected!r}")


def verify_manifest(directory: Path, name: str) -> list[dict[str, object]]:
    manifest = read_json(directory / name)
    results: list[dict[str, object]] = []
    for entry in manifest["files"]:  # type: ignore[index]
        relative = str(entry["path"])  # type: ignore[index]
        path = directory / relative
        require(
            path.resolve().is_relative_to(directory.resolve()), "unsafe manifest path"
        )
        data = path.read_bytes()
        actual_hash = sha256_bytes(data)
        result = {
            "path": relative,
            "expected_sha256": entry["sha256"],  # type: ignore[index]
            "actual_sha256": actual_hash,
            "expected_size_bytes": entry["size_bytes"],  # type: ignore[index]
            "actual_size_bytes": len(data),
            "sha256_match": actual_hash == entry["sha256"],  # type: ignore[index]
            "size_match": len(data) == entry["size_bytes"],  # type: ignore[index]
        }
        require(
            bool(result["sha256_match"]) and bool(result["size_match"]),
            f"{name} does not verify {relative}",
        )
        results.append(result)
    return results


def verify_custody_signature() -> dict[str, object]:
    attestation = read_json(PUBLIC / "custody_attestation.json")
    key_document = read_json(PUBLIC / "custody_public_key.json")
    signing_payload = {
        key: value
        for key, value in attestation.items()
        if key not in {"signature_base64", "signing_payload_sha256"}
    }
    signed_bytes = canonical_json(signing_payload).encode("utf-8")
    signature_valid = False
    try:
        public_key = serialization.load_pem_public_key(
            str(key_document["public_key_pem"]).encode("utf-8")
        )
        public_key.verify(
            base64.b64decode(str(attestation["signature_base64"]), validate=True),
            signed_bytes,
        )
        signature_valid = True
    except (InvalidSignature, TypeError, ValueError):
        signature_valid = False
    public_key_hash = sha256_bytes(str(key_document["public_key_pem"]).encode("utf-8"))
    result = {
        "signature_valid": signature_valid,
        "signing_payload_sha256": sha256_bytes(signed_bytes),
        "signing_payload_hash_match": (
            sha256_bytes(signed_bytes) == attestation["signing_payload_sha256"]
        ),
        "public_key_hash_match": (
            public_key_hash
            == key_document["public_key_sha256"]
            == attestation["custody_public_key_sha256"]
        ),
        "same_host": attestation["same_host"],
        "external_qualification": attestation["external_qualification"],
    }
    require(signature_valid, "custody attestation signature is invalid")
    require(bool(result["signing_payload_hash_match"]), "custody payload hash differs")
    require(bool(result["public_key_hash_match"]), "custody public key hash differs")
    return result


def build_assessment() -> tuple[dict[str, bytes], dict[str, object]]:
    public_manifest_results = verify_manifest(PUBLIC, "hash_manifest.json")
    modeler_manifest_results = verify_manifest(MODELER, "artifact_manifest.json")
    custody = verify_custody_signature()

    protocol_hash = sha256_file(ITERATION / "PROTOCOL.json")
    public_manifest = read_json(PUBLIC / "hash_manifest.json")
    require(
        protocol_hash == public_manifest["protocol_sha256"],
        "public manifest is bound to another protocol",
    )

    contract = PublicEligibilityContractV54.model_validate(
        read_json(PUBLIC / "public_eligibility_contract_v54.json")
    )
    snapshot = ODETimeSeriesSnapshotV52.model_validate(
        read_json(PUBLIC / "public_snapshot_v52.json")
    )
    bundle = ODEForecastBundleV53.model_validate(
        read_json(MODELER / "forecast_bundle_v53.json")
    )
    contract.assert_sealed()
    snapshot.assert_sealed()
    require(bundle.bundle_hash == bundle.content_hash(), "V5.3 bundle is not sealed")
    require(
        contract.task_id == snapshot.task_id == bundle.task_id == CAMPAIGN_ID,
        "typed task bindings differ",
    )
    require(
        bundle.public_snapshot_hash == snapshot.snapshot_hash,
        "V5.3 bundle is bound to another public snapshot",
    )

    policy = read_json(PUBLIC / "candidate_selection_policy.json")
    candidate_results = read_json(MODELER / "candidate_results.json")
    paired = read_json(MODELER / "paired_public_losses.json")
    policy_hash = sha256_file(PUBLIC / "candidate_selection_policy.json")
    require(
        policy_hash == contract.candidate_selection_rule_hash,
        "candidate policy is not bound to V5.4 contract",
    )
    require(
        paired["baseline_id"] == contract.baseline_id == policy["baseline_id"],
        "baseline bindings differ",
    )

    selected_family = str(
        candidate_results["frozen_v53_selection"]["selected_family"]  # type: ignore[index]
    )
    require(
        selected_family == bundle.development_bundle.selected_candidate_id,
        "saved selection differs from typed V5.3 bundle",
    )
    candidate_id = f"initial_{selected_family}"
    candidate_pairs = paired["candidate_pairs"]  # type: ignore[index]
    selected_pairs = [
        item for item in candidate_pairs if item["candidate_id"] == candidate_id
    ]
    require(len(selected_pairs) == 1, "selected candidate pair is absent or ambiguous")
    search_count = int(
        candidate_results["candidate_policy"][  # type: ignore[index]
            "candidate_search_count_excluding_baseline"
        ]
    )
    require(search_count == len(candidate_pairs), "candidate search count differs")
    require(
        search_count <= int(policy["candidate_budget"]), "candidate budget exceeded"
    )

    scale = float(paired["scale"])
    require(math.isfinite(scale) and scale > 0.0, "loss scale is invalid")
    observations = [float(value) for value in snapshot.observations]
    rows: list[PairedForecastLossV54] = []
    arithmetic_checks = 0
    for raw in selected_pairs[0]["rows"]:
        origin = int(raw["origin"])
        horizon = int(raw["horizon"])
        require(raw["candidate_id"] == candidate_id, "candidate row binding differs")
        require(
            raw["baseline_id"] == contract.baseline_id,
            "baseline row binding differs",
        )
        require(
            1 <= origin and 1 <= horizon and origin + horizon <= len(observations),
            "paired-loss coordinate is outside the public snapshot",
        )
        observed = float(raw["observed"])
        prediction = float(raw["prediction"])
        baseline_prediction = float(raw["baseline_prediction"])
        require_close(
            observed,
            observations[origin + horizon - 1],
            "paired observed value",
        )
        require_close(
            baseline_prediction,
            observations[origin - 1],
            "persistence prediction",
        )
        candidate_loss = abs(prediction - observed) / scale
        baseline_loss = abs(baseline_prediction - observed) / scale
        require_close(
            candidate_loss,
            float(raw["candidate_normalized_absolute_loss"]),
            "candidate normalized loss",
        )
        require_close(
            baseline_loss,
            float(raw["baseline_normalized_absolute_loss"]),
            "baseline normalized loss",
        )
        require_close(
            baseline_loss - candidate_loss,
            float(raw["advantage"]),
            "paired advantage",
        )
        rows.append(
            PairedForecastLossV54(
                origin=origin,
                horizon=horizon,
                candidate_loss=candidate_loss,
                baseline_loss=baseline_loss,
            )
        )
        arithmetic_checks += 1

    source_artifact_hashes = sorted(
        {
            sha256_file(MODELER / "artifact_manifest.json"),
            sha256_file(MODELER / "candidate_results.json"),
            sha256_file(MODELER / "forecast_bundle_v53.json"),
            sha256_file(MODELER / "paired_public_losses.json"),
        }
    )
    evidence = PublicEligibilityInputV54.seal(
        task_id=CAMPAIGN_ID,
        contract_hash=contract.contract_hash,
        candidate_id=candidate_id,
        baseline_id=contract.baseline_id,
        candidate_search_count=search_count,
        public_scientific_acceptance_verified=bundle.scientific_acceptance,
        fixture_only=bundle.fixture_only,
        source_artifact_hashes=source_artifact_hashes,
        rows=rows,
    )
    assessment = assess_public_eligibility_v54(
        contract=contract,
        evidence=evidence,
    )
    require(
        assessment.private_evaluation_performed is False
        and assessment.scientific_qualification_granted is False
        and assessment.real_world_action_authorized is False,
        "public assessment exceeded its authority",
    )

    verification = {
        "schema_version": "fma.iteration_33.public_gate_verification.v1",
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": protocol_hash,
        "public_manifest_sha256": sha256_file(PUBLIC / "hash_manifest.json"),
        "modeler_manifest_sha256": sha256_file(MODELER / "artifact_manifest.json"),
        "public_manifest_files": public_manifest_results,
        "modeler_manifest_files": modeler_manifest_results,
        "custody_attestation": custody,
        "typed_seals": {
            "v5_2_public_snapshot": True,
            "v5_3_forecast_bundle": True,
            "v5_4_public_eligibility_contract": True,
        },
        "bindings": {
            "task": True,
            "protocol": True,
            "snapshot_to_bundle": True,
            "candidate_policy_to_contract": True,
            "selected_family_to_bundle": True,
            "baseline": True,
        },
        "paired_loss_arithmetic_rows_verified": arithmetic_checks,
        "private_target_accessed": False,
        "gate_receipt_signed": False,
        "private_evaluation_authorized": False,
        "scientific_qualification_granted": False,
    }
    report_lines = [
        "# I33 V5.4 public eligibility result",
        "",
        f"- Campaign: `{CAMPAIGN_ID}`",
        f"- Decision: **{assessment.decision}**",
        f"- Candidate: `{candidate_id}`",
        f"- Baseline: `{contract.baseline_id}`",
        f"- Public evidence input hash: `{evidence.input_hash}`",
        f"- Public assessment hash: `{assessment.assessment_hash}`",
        f"- Origins / paired rows: {assessment.metrics.origin_count} / "
        f"{assessment.metrics.paired_row_count}",
        f"- Mean paired advantage: {assessment.metrics.mean_advantage:.15f}",
        "- Selection-adjusted bootstrap lower bound: "
        f"{assessment.metrics.selection_adjusted_bootstrap_lower_bound:.15f}",
        f"- Origin win fraction: {assessment.metrics.origin_win_fraction:.15f}",
        "",
        "## Failed checks",
        "",
    ]
    failed = [name for name, passed in assessment.checks.items() if not passed]
    report_lines.extend(f"- `{name}`" for name in failed)
    report_lines.extend(
        [
            "",
            "## Authority boundary",
            "",
            "- The gate used public evidence only.",
            "- No eligibility receipt was signed because the decision is not ELIGIBLE.",
            "- Private evaluation was not authorized or performed; budget consumed is 0.",
            "- This same-host shadow run grants no scientific or external qualification.",
            "",
        ]
    )
    evidence_bytes = json_bytes(evidence.model_dump(mode="json"))
    assessment_bytes = json_bytes(assessment.model_dump(mode="json"))
    verification_bytes = json_bytes(verification)
    report_bytes = ("\n".join(report_lines)).encode("utf-8")
    artifacts = {
        EVIDENCE_NAME: evidence_bytes,
        ASSESSMENT_NAME: assessment_bytes,
        VERIFICATION_NAME: verification_bytes,
        REPORT_NAME: report_bytes,
    }
    manifest = {
        "schema_version": "fma.iteration_33.public_gate_manifest.v1",
        "campaign_id": CAMPAIGN_ID,
        "scope": "deterministic public-only V5.4 assessment; manifest self-hash excluded",
        "runner": {
            "path": "experiments/iteration_33/run_public_eligibility.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "inputs": {
            "public_manifest_sha256": verification["public_manifest_sha256"],
            "modeler_manifest_sha256": verification["modeler_manifest_sha256"],
            "contract_hash": contract.contract_hash,
            "source_artifact_hashes": source_artifact_hashes,
        },
        "files": [
            {
                "path": name,
                "sha256": sha256_bytes(data),
                "size_bytes": len(data),
            }
            for name, data in sorted(artifacts.items())
        ],
        "decision": assessment.decision,
        "public_gate_eligible": assessment.public_gate_eligible,
        "gate_receipt_signed": False,
        "private_evaluation_performed": False,
        "scientific_qualification_granted": False,
        "external_qualification": False,
    }
    artifacts[MANIFEST_NAME] = json_bytes(manifest)
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "decision": assessment.decision,
        "input_hash": evidence.input_hash,
        "assessment_hash": assessment.assessment_hash,
        "failed_checks": failed,
        "private_evaluation_performed": False,
        "private_evaluation_budget_consumed": 0,
        "scientific_qualification_granted": False,
        "artifact_sha256": {
            name: sha256_bytes(data) for name, data in sorted(artifacts.items())
        },
    }
    return artifacts, summary


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--dry-run", "--freeze"}:
        raise SystemExit(
            "usage: python experiments/iteration_33/run_public_eligibility.py "
            "--dry-run|--freeze"
        )
    artifacts, summary = build_assessment()
    if sys.argv[1] == "--dry-run":
        print(canonical_json(summary))
        return
    require(not GATE.exists(), "refusing to alter an existing public_gate directory")
    GATE.mkdir()
    for name in GENERATED_NAMES:
        (GATE / name).write_bytes(artifacts[name])
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
