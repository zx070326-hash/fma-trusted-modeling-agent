"""Cross-domain, repeated gold and mechanism evaluation for V5.2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import t as student_t

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5_1.evaluation_harness import GoldInjectionReceiptV51


MechanismV52 = Literal[
    "backward_revision",
    "competition",
    "independent_review",
    "scientific_adapters",
]
FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


class AblationArmProcessReceiptV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    run_id: Identifier
    domain_id: Identifier
    case_id: Identifier
    repetition: Annotated[int, Field(ge=1)]
    mechanism_id: MechanismV52
    mechanism_enabled: bool
    nuisance_identity_hash: Sha256
    process_id: Annotated[int, Field(ge=1)]
    command_hash: Sha256
    worker_source_hash: Sha256
    stdout_hash: Sha256
    stderr_hash: Sha256
    output_artifact_hash: Sha256
    score: FiniteNumber
    exit_code: int
    failure_code: Identifier | None = None
    wall_time_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    estimated_cost_usd: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None = None
    human_intervention_count: Annotated[int, Field(ge=0)] = 0
    trace_event_hashes: Annotated[list[Sha256], Field(min_length=2)]
    observed_mechanism_event: bool
    fresh_process: Literal[True] = True
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "AblationArmProcessReceiptV52":
        if self.observed_mechanism_event != self.mechanism_enabled:
            raise ValueError("observed mechanism event differs from assignment")
        if self.trace_event_hashes != sorted(set(self.trace_event_hashes)):
            raise ValueError("trace event hashes must be sorted and unique")
        if self.exit_code == 0 and self.failure_code is not None:
            raise ValueError("successful arm cannot contain failure_code")
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("receipt_hash does not match arm receipt")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")

    @classmethod
    def seal(cls, **data: object) -> "AblationArmProcessReceiptV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"receipt_hash"})
        payload["receipt_hash"] = draft.content_hash()
        return cls(**payload)


class CrossDomainAblationObservationV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    observation_id: Identifier
    domain_id: Identifier
    case_id: Identifier
    repetition: Annotated[int, Field(ge=1)]
    mechanism_id: MechanismV52
    control_receipt_hash: Sha256
    treatment_receipt_hash: Sha256
    nuisance_identity_hash: Sha256
    process_receipts_disjoint: bool
    exactly_one_mechanism_changed: bool
    observed_execution_path_delta: bool
    valid_comparison: bool
    score_delta: FiniteNumber
    wall_time_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    estimated_cost_usd: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None = None
    human_intervention_count: Annotated[int, Field(ge=0)]
    failure_codes: list[Identifier] = Field(default_factory=list)
    trace_event_hashes: Annotated[list[Sha256], Field(min_length=4)]
    fixture_only: bool
    scientific_qualification_granted: Literal[False] = False
    reason_codes: list[Identifier] = Field(default_factory=list)
    observation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> "CrossDomainAblationObservationV52":
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason codes must be sorted and unique")
        if self.failure_codes != sorted(set(self.failure_codes)):
            raise ValueError("failure codes must be sorted and unique")
        if self.trace_event_hashes != sorted(set(self.trace_event_hashes)):
            raise ValueError("trace event hashes must be sorted and unique")
        expected = (
            self.process_receipts_disjoint
            and self.exactly_one_mechanism_changed
            and self.observed_execution_path_delta
            and not self.reason_codes
        )
        if self.valid_comparison != expected:
            raise ValueError("valid comparison differs from comparison evidence")
        if (
            self.observation_hash
            and self.observation_hash != self.content_hash()
        ):
            raise ValueError("observation_hash does not match observation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "observation_hash")

    @classmethod
    def seal(cls, **data: object) -> "CrossDomainAblationObservationV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"observation_hash"})
        payload["observation_hash"] = draft.content_hash()
        return cls(**payload)


class CrossDomainAblationSummaryV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    mechanism_id: MechanismV52
    domain_case_counts: dict[Identifier, Annotated[int, Field(ge=0)]]
    total_observations: Annotated[int, Field(ge=0)]
    valid_observations: Annotated[int, Field(ge=0)]
    repetitions_per_case_minimum: Annotated[int, Field(ge=0)]
    mean_score_delta: FiniteNumber | None
    sample_standard_deviation: FiniteNumber | None
    standard_error: FiniteNumber | None
    confidence_interval_95_low: FiniteNumber | None
    confidence_interval_95_high: FiniteNumber | None
    positive_delta_fraction: Annotated[
        float, Field(ge=0, le=1, allow_inf_nan=False)
    ] | None
    total_wall_time_seconds: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ]
    total_estimated_cost_usd: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] | None
    total_human_interventions: Annotated[int, Field(ge=0)]
    failed_arm_count: Annotated[int, Field(ge=0)]
    trace_coverage_complete: bool
    fixture_observations_present: bool
    cross_domain_coverage_satisfied: bool
    repeated_measurement_satisfied: bool
    inference_ready_for_external_review: bool
    general_causal_claim_permitted: Literal[False] = False
    reason_codes: list[Identifier] = Field(default_factory=list)
    observation_hashes: list[Sha256]
    summary_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> "CrossDomainAblationSummaryV52":
        if self.valid_observations > self.total_observations:
            raise ValueError("valid observations exceed total")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason codes must be sorted and unique")
        if self.observation_hashes != sorted(set(self.observation_hashes)):
            raise ValueError("observation hashes must be sorted and unique")
        ready = (
            self.cross_domain_coverage_satisfied
            and self.repeated_measurement_satisfied
            and not self.fixture_observations_present
            and self.valid_observations == self.total_observations
            and self.total_observations > 0
            and self.failed_arm_count == 0
            and self.trace_coverage_complete
        )
        if self.inference_ready_for_external_review != ready:
            raise ValueError("inference readiness differs from evidence")
        if self.summary_hash and self.summary_hash != self.content_hash():
            raise ValueError("summary_hash does not match summary")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "summary_hash")

    @classmethod
    def seal(cls, **data: object) -> "CrossDomainAblationSummaryV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"summary_hash"})
        payload["summary_hash"] = draft.content_hash()
        return cls(**payload)


class GoldTaskObservationV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    domain_id: Identifier
    task_id: Identifier
    through_stage: Literal["S0", "S1", "S2", "S3", "S4", "S5"]
    package_hash: Sha256
    injection_receipt_hash: Sha256
    target_root_hash_after: Sha256
    isolated_target: bool
    process_receipt_hashes: Annotated[list[Sha256], Field(min_length=1)]
    fixture_only: bool
    observation_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> "GoldTaskObservationV52":
        if self.process_receipt_hashes != sorted(
            set(self.process_receipt_hashes)
        ):
            raise ValueError("process receipt hashes must be sorted and unique")
        if (
            self.observation_hash
            and self.observation_hash != self.content_hash()
        ):
            raise ValueError("observation_hash does not match gold observation")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "observation_hash")

    @classmethod
    def from_injection(
        cls,
        *,
        domain_id: str,
        receipt: GoldInjectionReceiptV51,
        isolated_target: bool,
        process_receipt_hashes: list[str],
        fixture_only: bool,
    ) -> "GoldTaskObservationV52":
        if receipt.receipt_hash != receipt.content_hash():
            raise ValueError("gold injection receipt is not sealed")
        draft = cls(
            domain_id=domain_id,
            task_id=receipt.task_id,
            through_stage=receipt.through_stage,
            package_hash=receipt.package_hash,
            injection_receipt_hash=receipt.receipt_hash,
            target_root_hash_after=receipt.target_root_hash_after,
            isolated_target=isolated_target,
            process_receipt_hashes=sorted(set(process_receipt_hashes)),
            fixture_only=fixture_only,
        )
        payload = draft.model_dump(exclude={"observation_hash"})
        payload["observation_hash"] = draft.content_hash()
        return cls(**payload)


class GoldCoverageSummaryV52(StrictModel):
    schema_version: Literal["5.2"] = "5.2"
    domain_ids: Annotated[list[Identifier], Field(min_length=1)]
    task_ids: Annotated[list[Identifier], Field(min_length=1)]
    covered_stages: Annotated[list[Identifier], Field(min_length=1)]
    coverage_matrix: dict[Identifier, list[Identifier]]
    all_targets_isolated: bool
    process_receipts_globally_disjoint: bool
    fixture_observations_present: bool
    multi_domain_coverage_satisfied: bool
    multi_stage_coverage_satisfied: bool
    general_gold_effect_claim_permitted: Literal[False] = False
    observation_hashes: list[Sha256]
    summary_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> "GoldCoverageSummaryV52":
        for values, label in (
            (self.domain_ids, "domain_ids"),
            (self.task_ids, "task_ids"),
            (self.covered_stages, "covered_stages"),
            (self.observation_hashes, "observation_hashes"),
        ):
            if values != sorted(set(values)):
                raise ValueError(f"{label} must be sorted and unique")
        if self.summary_hash and self.summary_hash != self.content_hash():
            raise ValueError("summary_hash does not match gold summary")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "summary_hash")

    @classmethod
    def seal(cls, **data: object) -> "GoldCoverageSummaryV52":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"summary_hash"})
        payload["summary_hash"] = draft.content_hash()
        return cls(**payload)


def compare_ablation_arms_v52(
    control: AblationArmProcessReceiptV52,
    treatment: AblationArmProcessReceiptV52,
) -> CrossDomainAblationObservationV52:
    same_identity = (
        control.domain_id == treatment.domain_id
        and control.case_id == treatment.case_id
        and control.repetition == treatment.repetition
        and control.mechanism_id == treatment.mechanism_id
        and control.nuisance_identity_hash == treatment.nuisance_identity_hash
    )
    disjoint = control.receipt_hash != treatment.receipt_hash
    exactly_one = same_identity and (
        control.mechanism_enabled != treatment.mechanism_enabled
    )
    observed_delta = (
        control.observed_mechanism_event
        != treatment.observed_mechanism_event
        and control.output_artifact_hash != treatment.output_artifact_hash
    )
    reasons: list[str] = []
    if not same_identity:
        reasons.append("nuisance_identity_differs")
    if not disjoint:
        reasons.append("process_receipts_reused")
    if not exactly_one:
        reasons.append("not_exactly_one_mechanism_changed")
    if not observed_delta:
        reasons.append("execution_path_delta_not_observed")
    if (
        control.exit_code != 0
        or treatment.exit_code != 0
        or control.failure_code is not None
        or treatment.failure_code is not None
    ):
        reasons.append("arm_failure_present")
    return CrossDomainAblationObservationV52.seal(
        observation_id=(
            f"{control.domain_id}-{control.case_id}-"
            f"r{control.repetition}-{control.mechanism_id}"
        ),
        domain_id=control.domain_id,
        case_id=control.case_id,
        repetition=control.repetition,
        mechanism_id=control.mechanism_id,
        control_receipt_hash=control.receipt_hash,
        treatment_receipt_hash=treatment.receipt_hash,
        nuisance_identity_hash=control.nuisance_identity_hash,
        process_receipts_disjoint=disjoint,
        exactly_one_mechanism_changed=exactly_one,
        observed_execution_path_delta=observed_delta,
        valid_comparison=not reasons,
        score_delta=treatment.score - control.score,
        wall_time_seconds=(
            control.wall_time_seconds + treatment.wall_time_seconds
        ),
        estimated_cost_usd=(
            control.estimated_cost_usd + treatment.estimated_cost_usd
            if control.estimated_cost_usd is not None
            and treatment.estimated_cost_usd is not None
            else None
        ),
        human_intervention_count=(
            control.human_intervention_count
            + treatment.human_intervention_count
        ),
        failure_codes=sorted(
            set(
                item
                for item in (control.failure_code, treatment.failure_code)
                if item is not None
            )
        ),
        trace_event_hashes=sorted(
            set(control.trace_event_hashes + treatment.trace_event_hashes)
        ),
        fixture_only=control.fixture_only or treatment.fixture_only,
        reason_codes=sorted(reasons),
    )


def summarize_cross_domain_ablation_v52(
    observations: list[CrossDomainAblationObservationV52],
    *,
    minimum_domains: int = 2,
    minimum_cases_per_domain: int = 2,
    minimum_repetitions_per_case: int = 3,
) -> CrossDomainAblationSummaryV52:
    if not observations:
        raise ValueError("at least one ablation observation is required")
    mechanism = observations[0].mechanism_id
    if any(item.mechanism_id != mechanism for item in observations):
        raise ValueError("summary cannot mix mechanisms")
    keys = [
        (item.domain_id, item.case_id, item.repetition)
        for item in observations
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate domain/case/repetition observation")
    domain_cases: dict[str, set[str]] = defaultdict(set)
    repetitions: Counter[tuple[str, str]] = Counter()
    for item in observations:
        domain_cases[item.domain_id].add(item.case_id)
        repetitions[(item.domain_id, item.case_id)] += 1
    cross_domain = (
        len(domain_cases) >= minimum_domains
        and all(
            len(cases) >= minimum_cases_per_domain
            for cases in domain_cases.values()
        )
    )
    repetition_min = min(repetitions.values())
    repeated = repetition_min >= minimum_repetitions_per_case
    valid = [item for item in observations if item.valid_comparison]
    deltas = np.asarray([item.score_delta for item in valid], dtype=float)
    if len(deltas):
        mean = float(np.mean(deltas))
        positive = float(np.mean(deltas > 0))
    else:
        mean = positive = None
    if len(deltas) >= 2:
        standard_deviation = float(np.std(deltas, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(deltas))
        critical = float(student_t.ppf(0.975, df=len(deltas) - 1))
        low = mean - critical * standard_error
        high = mean + critical * standard_error
    else:
        standard_deviation = standard_error = low = high = None
    fixture = any(item.fixture_only for item in observations)
    costs = [item.estimated_cost_usd for item in observations]
    total_cost = (
        float(sum(value for value in costs if value is not None))
        if all(value is not None for value in costs)
        else None
    )
    failed_arm_count = sum(len(item.failure_codes) for item in observations)
    trace_complete = all(len(item.trace_event_hashes) >= 4 for item in observations)
    reasons: list[str] = []
    if not cross_domain:
        reasons.append("cross_domain_case_coverage_insufficient")
    if not repeated:
        reasons.append("repetitions_per_case_insufficient")
    if len(valid) != len(observations):
        reasons.append("invalid_comparisons_present")
    if fixture:
        reasons.append("fixture_only_observations")
    return CrossDomainAblationSummaryV52.seal(
        mechanism_id=mechanism,
        domain_case_counts={
            key: len(value) for key, value in sorted(domain_cases.items())
        },
        total_observations=len(observations),
        valid_observations=len(valid),
        repetitions_per_case_minimum=repetition_min,
        mean_score_delta=mean,
        sample_standard_deviation=standard_deviation,
        standard_error=standard_error,
        confidence_interval_95_low=low,
        confidence_interval_95_high=high,
        positive_delta_fraction=positive,
        total_wall_time_seconds=sum(
            item.wall_time_seconds for item in observations
        ),
        total_estimated_cost_usd=total_cost,
        total_human_interventions=sum(
            item.human_intervention_count for item in observations
        ),
        failed_arm_count=failed_arm_count,
        trace_coverage_complete=trace_complete,
        fixture_observations_present=fixture,
        cross_domain_coverage_satisfied=cross_domain,
        repeated_measurement_satisfied=repeated,
        inference_ready_for_external_review=(
            cross_domain
            and repeated
            and not fixture
            and len(valid) == len(observations)
            and failed_arm_count == 0
            and trace_complete
        ),
        reason_codes=sorted(reasons),
        observation_hashes=sorted(
            item.observation_hash for item in observations
        ),
    )


def summarize_gold_coverage_v52(
    observations: list[GoldTaskObservationV52],
) -> GoldCoverageSummaryV52:
    if not observations:
        raise ValueError("at least one gold observation is required")
    domains = sorted(set(item.domain_id for item in observations))
    tasks = sorted(set(item.task_id for item in observations))
    stages = sorted(set(item.through_stage for item in observations))
    coverage: dict[str, list[str]] = {}
    for domain in domains:
        coverage[domain] = sorted(
            set(
                item.through_stage
                for item in observations
                if item.domain_id == domain
            )
        )
    process_hashes = [
        process_hash
        for item in observations
        for process_hash in item.process_receipt_hashes
    ]
    return GoldCoverageSummaryV52.seal(
        domain_ids=domains,
        task_ids=tasks,
        covered_stages=stages,
        coverage_matrix=coverage,
        all_targets_isolated=all(item.isolated_target for item in observations),
        process_receipts_globally_disjoint=(
            len(process_hashes) == len(set(process_hashes))
        ),
        fixture_observations_present=any(
            item.fixture_only for item in observations
        ),
        multi_domain_coverage_satisfied=len(domains) >= 2,
        multi_stage_coverage_satisfied=len(stages) >= 2,
        observation_hashes=sorted(
            item.observation_hash for item in observations
        ),
    )


def run_fixture_ablation_arm_v52(
    *,
    domain_id: str,
    case_id: str,
    repetition: int,
    mechanism_id: MechanismV52,
    mechanism_enabled: bool,
    nuisance_identity_hash: str,
    fixture_seed: int,
    output_directory: str | Path,
) -> AblationArmProcessReceiptV52:
    """Execute one deterministic control-plane fixture arm in a fresh process."""

    output_dir = Path(output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("cross_domain_worker.py").resolve()
    command = [
        sys.executable,
        str(worker),
        "--domain-id",
        domain_id,
        "--case-id",
        case_id,
        "--repetition",
        str(repetition),
        "--mechanism-id",
        mechanism_id,
        "--mechanism-enabled",
        "1" if mechanism_enabled else "0",
        "--nuisance-identity-hash",
        nuisance_identity_hash,
        "--fixture-seed",
        str(fixture_seed),
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "PATHEXT"}
    }
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="fma-v52-ablation-", dir=output_dir
    ) as temporary:
        completed = subprocess.run(
            command,
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "fixture ablation arm failed; stderr_sha256="
            + hashlib.sha256(completed.stderr.encode()).hexdigest()
        )
    payload = json.loads(completed.stdout)
    assigned_event = sha256_value(
        {
            "event": "arm_assigned",
            "domain_id": domain_id,
            "case_id": case_id,
            "repetition": repetition,
            "mechanism_id": mechanism_id,
            "mechanism_enabled": mechanism_enabled,
            "nuisance_identity_hash": nuisance_identity_hash,
        }
    )
    completed_event = sha256_value(
        {
            "event": "arm_completed",
            "previous_event_hash": assigned_event,
            "output_artifact_hash": payload["output_artifact_hash"],
            "exit_code": completed.returncode,
        }
    )
    return AblationArmProcessReceiptV52.seal(
        run_id=(
            f"{domain_id}-{case_id}-r{repetition}-"
            f"{mechanism_id}-{'on' if mechanism_enabled else 'off'}"
        ),
        domain_id=domain_id,
        case_id=case_id,
        repetition=repetition,
        mechanism_id=mechanism_id,
        mechanism_enabled=mechanism_enabled,
        nuisance_identity_hash=nuisance_identity_hash,
        process_id=int(payload["process_id"]),
        command_hash=sha256_value(
            {
                "domain_id": domain_id,
                "case_id": case_id,
                "repetition": repetition,
                "mechanism_id": mechanism_id,
                "mechanism_enabled": mechanism_enabled,
                "nuisance_identity_hash": nuisance_identity_hash,
                "fixture_seed": fixture_seed,
            }
        ),
        worker_source_hash=hashlib.sha256(worker.read_bytes()).hexdigest(),
        stdout_hash=hashlib.sha256(completed.stdout.encode()).hexdigest(),
        stderr_hash=hashlib.sha256(completed.stderr.encode()).hexdigest(),
        output_artifact_hash=str(payload["output_artifact_hash"]),
        score=float(payload["score"]),
        exit_code=completed.returncode,
        failure_code=None,
        wall_time_seconds=elapsed,
        estimated_cost_usd=None,
        human_intervention_count=0,
        trace_event_hashes=sorted([assigned_event, completed_event]),
        observed_mechanism_event=bool(payload["observed_mechanism_event"]),
        fixture_only=True,
    )
