"""Harness-authenticated V6.2 source acquisition and S2 re-verification.

The public-source records in :mod:`fma.v6.public_source` are content sealed.
That proves internal consistency, not who performed the network transport.
This module adds two domain-separated HMAC envelopes:

* an acquisition receipt issued only around the code-owned fetch path; and
* a fresh S2 receipt issued after replaying the raw bytes for the current
  authenticated raw baseline and attempt.

The authority key remains outside model artifacts.  Either a dedicated
external-to-the-model HMAC key can be supplied or the existing V5 workspace
authority can be reused.  Neither mode is an independent scientific review,
qualification, or real-world action authorization.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import StrictModel
from fma.v2.schemas import Identifier, Sha256
from fma.v5.stage_workspace import StageWorkspaceV50
from fma.v5.workspace_schemas import (
    RawDataBaselineV50,
    TaskWorkspaceSpecV50,
)
from fma.v5_2.ode_system import ODETimeSeriesSnapshotV52

from .public_source import (
    FetchedWorldBankSeriesV62,
    SourceFetcherV62,
    SourceVerificationV62,
    WORLD_BANK_HOST,
    WorldBankSourceContractV62,
    WorldBankSourceReceiptV62,
    fetch_world_bank_series_v62,
    verify_world_bank_source_v62,
)


SOURCE_ACQUISITION_AUTH_PATH = (
    "data/source_provenance_v62/acquisition_authority_receipt.json"
)
S2_SOURCE_REVERIFICATION_PATH = (
    "checks/s2_source_reverification_v62.json"
)

AuthorityModeV62 = Literal["external_hmac", "v5_workspace_hmac"]
TransportModeV62 = Literal[
    "live_https_no_redirect", "fixture_injected"
]
ScientificProvenanceStatusV62 = Literal["FAIL", "NOT_RUN", "HUMAN"]

_KEY_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _hash_without(model: StrictModel, *fields: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude=set(fields)))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _assert_timezone(value: datetime, field_name: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _safe_source_raw_path(
    workspace_root: str | Path,
    receipt: WorldBankSourceReceiptV62,
) -> Path:
    root = Path(workspace_root).resolve()
    candidate = (root / receipt.raw_relative_path).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("source raw path escapes workspace")
    return candidate


class SourceRuntimeIdentityV62(StrictModel):
    """Auditable identity of the process and source-adapter code."""

    schema_version: Literal["6.2-source-runtime-identity"] = (
        "6.2-source-runtime-identity"
    )
    python_implementation: str = Field(min_length=2, max_length=100)
    python_version: str = Field(min_length=3, max_length=100)
    operating_system: str = Field(min_length=2, max_length=200)
    machine: str = Field(min_length=1, max_length=200)
    process_id: Annotated[int, Field(ge=1)]
    python_executable_hash: Sha256
    public_source_code_hash: Sha256
    source_auth_code_hash: Sha256
    runtime_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_runtime(self) -> "SourceRuntimeIdentityV62":
        if self.runtime_hash and self.runtime_hash != self.content_hash():
            raise ValueError("source runtime identity hash differs")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "runtime_hash")

    def assert_sealed(self) -> None:
        if not self.runtime_hash or self.runtime_hash != self.content_hash():
            raise ValueError("source runtime identity is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "SourceRuntimeIdentityV62":
        draft = cls(**data)
        payload = draft.model_dump(exclude={"runtime_hash"})
        payload["runtime_hash"] = draft.content_hash()
        return cls(**payload)


def capture_source_runtime_identity_v62() -> SourceRuntimeIdentityV62:
    """Capture runtime and exact adapter-source hashes, failing closed."""

    executable = Path(sys.executable).resolve(strict=True)
    public_source_path = Path(
        sys.modules[
            "fma.v6.public_source"
        ].__file__  # type: ignore[union-attr]
    ).resolve(strict=True)
    source_auth_path = Path(__file__).resolve(strict=True)
    return SourceRuntimeIdentityV62.seal(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        operating_system=f"{platform.system()} {platform.release()}",
        machine=platform.machine() or "unknown",
        process_id=os.getpid(),
        python_executable_hash=_file_hash(executable),
        public_source_code_hash=_file_hash(public_source_path),
        source_auth_code_hash=_file_hash(source_auth_path),
    )


class SourceAcquisitionReceiptV62(StrictModel):
    """HMAC-authenticated evidence for one code-owned source transport."""

    schema_version: Literal["6.2-source-acquisition-auth"] = (
        "6.2-source-acquisition-auth"
    )
    workspace_id: Identifier
    graph_id: Identifier
    workspace_spec_hash: Sha256
    contract_hash: Sha256
    source_receipt_hash: Sha256
    snapshot_hash: Sha256
    exact_url: str
    raw_response_hash: Sha256
    raw_response_size_bytes: Annotated[int, Field(ge=1)]
    transport_mode: TransportModeV62
    response_status: Literal[200] = 200
    runtime_identity: SourceRuntimeIdentityV62
    fixture_only: bool
    scientific_provenance_status: Literal["NOT_RUN", "HUMAN"]
    acquired_at: datetime
    authenticated_at: datetime
    authority_mode: AuthorityModeV62
    authority_key_id: Identifier
    independent_source_review: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "SourceAcquisitionReceiptV62":
        _assert_timezone(self.acquired_at, "acquired_at")
        _assert_timezone(self.authenticated_at, "authenticated_at")
        if self.authenticated_at < self.acquired_at:
            raise ValueError("source authentication predates acquisition")
        if self.graph_id != f"v5-{self.workspace_id}":
            raise ValueError("source acquisition graph/workspace binding differs")
        parsed = urlparse(self.exact_url)
        if parsed.scheme != "https" or parsed.hostname != WORLD_BANK_HOST:
            raise ValueError("source acquisition URL is not the registered host")
        expected_mode: TransportModeV62 = (
            "fixture_injected"
            if self.fixture_only
            else "live_https_no_redirect"
        )
        if self.transport_mode != expected_mode:
            raise ValueError("source transport mode differs from fixture scope")
        expected_scientific = (
            "NOT_RUN" if self.fixture_only else "HUMAN"
        )
        if self.scientific_provenance_status != expected_scientific:
            raise ValueError("source acquisition scientific status is overstated")
        self.runtime_identity.assert_sealed()
        if self.receipt_hash and not self.authority_auth_tag:
            raise ValueError(
                "source acquisition receipt hash requires an auth tag"
            )
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("source acquisition receipt envelope differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


class S2SourceReverificationReceiptV62(StrictModel):
    """Authenticated fresh replay result for one exact current S2 attempt."""

    schema_version: Literal["6.2-s2-source-reverification"] = (
        "6.2-s2-source-reverification"
    )
    workspace_id: Identifier
    graph_id: Identifier
    workspace_spec_hash: Sha256
    s1_gate_hash: Sha256
    s2_attempt: Annotated[int, Field(ge=1)]
    raw_baseline_hash: Sha256
    source_acquisition_receipt_hash: Sha256
    contract_hash: Sha256
    source_receipt_hash: Sha256
    snapshot_hash: Sha256
    exact_url: str
    raw_response_hash: Sha256
    source_verification_hash: Sha256
    source_checks_hash: Sha256
    replay_status: Literal["PASS", "FAIL"]
    reason_codes: list[Identifier]
    runtime_identity: SourceRuntimeIdentityV62
    fixture_only: bool
    acquisition_authority_verified: Literal[True] = True
    official_live_transport_authenticated: bool
    scientific_provenance_status: ScientificProvenanceStatusV62
    reverified_at: datetime
    authority_mode: AuthorityModeV62
    authority_key_id: Identifier
    independent_source_review: Literal[False] = False
    scientific_qualification_granted: Literal[False] = False
    real_world_action_authorized: Literal[False] = False
    authority_auth_tag: Sha256 | None = None
    receipt_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "S2SourceReverificationReceiptV62":
        _assert_timezone(self.reverified_at, "reverified_at")
        if self.graph_id != f"v5-{self.workspace_id}":
            raise ValueError("S2 source graph/workspace binding differs")
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("S2 source reasons must be sorted and unique")
        if (self.replay_status == "PASS") != (not self.reason_codes):
            raise ValueError("S2 source replay status differs from reasons")
        if self.official_live_transport_authenticated != (
            not self.fixture_only
        ):
            raise ValueError("official transport flag differs from fixture scope")
        expected_scientific: ScientificProvenanceStatusV62 = (
            "FAIL"
            if self.replay_status == "FAIL"
            else "NOT_RUN"
            if self.fixture_only
            else "HUMAN"
        )
        if self.scientific_provenance_status != expected_scientific:
            raise ValueError("S2 source scientific status is overstated")
        self.runtime_identity.assert_sealed()
        if self.receipt_hash and not self.authority_auth_tag:
            raise ValueError(
                "S2 source receipt hash requires an auth tag"
            )
        if self.receipt_hash and self.receipt_hash != self.content_hash():
            raise ValueError("S2 source receipt envelope differs")
        return self

    def unsigned_hash(self) -> str:
        return _hash_without(self, "authority_auth_tag", "receipt_hash")

    def content_hash(self) -> str:
        return _hash_without(self, "receipt_hash")


@dataclass(frozen=True)
class AuthenticatedWorldBankAcquisitionV62:
    fetched: FetchedWorldBankSeriesV62
    authority_receipt: SourceAcquisitionReceiptV62


@dataclass(frozen=True)
class S2WorldBankReverificationV62:
    verification: SourceVerificationV62
    authority_receipt: S2SourceReverificationReceiptV62


class SourceTransportAuthorityV62:
    """Domain-separated HMAC authority kept outside model-visible artifacts."""

    def __init__(
        self,
        *,
        key_id: str,
        secret: bytes,
        authority_mode: AuthorityModeV62 = "external_hmac",
    ) -> None:
        if not _KEY_ID.fullmatch(key_id):
            raise ValueError("invalid source authority key_id")
        if len(secret) < 32:
            raise ValueError("source authority secret needs at least 32 bytes")
        self.key_id = key_id
        self.authority_mode = authority_mode
        self._secret = bytes(secret)

    @classmethod
    def from_stage_workspace(
        cls,
        workspace: StageWorkspaceV50,
    ) -> "SourceTransportAuthorityV62":
        """Reuse the V5 authority with a distinct V6.2 HMAC domain."""

        workspace.spec.assert_sealed()
        return cls(
            key_id=workspace.authority_key_id,
            secret=workspace._authority_key,
            authority_mode="v5_workspace_hmac",
        )

    def _mac(self, kind: str, unsigned_hash: str) -> str:
        return hmac.new(
            self._secret,
            (
                f"fma-v6.2-source:{kind}:{unsigned_hash}"
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _issue(
        self,
        *,
        kind: Literal["acquisition", "s2_reverification"],
        model_type: type[
            SourceAcquisitionReceiptV62
            | S2SourceReverificationReceiptV62
        ],
        data: dict[str, object],
    ) -> SourceAcquisitionReceiptV62 | S2SourceReverificationReceiptV62:
        data["authority_mode"] = self.authority_mode
        data["authority_key_id"] = self.key_id
        unsigned = model_type(**data)
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            kind, unsigned.unsigned_hash()
        )
        tagged = model_type(**payload)
        final_payload = tagged.model_dump(mode="json")
        final_payload["receipt_hash"] = tagged.content_hash()
        return model_type(**final_payload)

    def _verify_auth(
        self,
        *,
        kind: Literal["acquisition", "s2_reverification"],
        receipt: SourceAcquisitionReceiptV62
        | S2SourceReverificationReceiptV62,
    ) -> bool:
        try:
            receipt.runtime_identity.assert_sealed()
            return bool(
                receipt.receipt_hash
                and receipt.receipt_hash == receipt.content_hash()
                and receipt.authority_mode == self.authority_mode
                and receipt.authority_key_id == self.key_id
                and receipt.authority_auth_tag
                and hmac.compare_digest(
                    receipt.authority_auth_tag,
                    self._mac(kind, receipt.unsigned_hash()),
                )
            )
        except (TypeError, ValueError):
            return False

    def acquire_world_bank_series(
        self,
        *,
        workspace_spec: TaskWorkspaceSpecV50,
        task_id: str,
        contract: WorldBankSourceContractV62,
        fetcher: SourceFetcherV62 | None = None,
        retrieved_at: datetime | None = None,
        authenticated_at: datetime | None = None,
    ) -> AuthenticatedWorldBankAcquisitionV62:
        """Execute the registered fetch and authenticate its exact result."""

        workspace_spec.assert_sealed()
        if task_id != workspace_spec.workspace_id:
            raise ValueError("source task_id differs from workspace")
        if not contract.fixture_only and (
            workspace_spec.evidence_scope != "public_data"
        ):
            raise ValueError(
                "live official acquisition requires public_data evidence scope"
            )
        fetched = fetch_world_bank_series_v62(
            task_id=task_id,
            contract=contract,
            fetcher=fetcher,
            retrieved_at=retrieved_at,
        )
        fetched.contract.assert_sealed()
        fetched.receipt.assert_sealed()
        fetched.snapshot.assert_sealed()
        when = authenticated_at or _utc_now()
        issued = self._issue(
            kind="acquisition",
            model_type=SourceAcquisitionReceiptV62,
            data={
                "workspace_id": workspace_spec.workspace_id,
                "graph_id": workspace_spec.graph_id,
                "workspace_spec_hash": workspace_spec.spec_hash,
                "contract_hash": fetched.contract.contract_hash,
                "source_receipt_hash": fetched.receipt.receipt_hash,
                "snapshot_hash": fetched.snapshot.snapshot_hash,
                "exact_url": fetched.contract.exact_url,
                "raw_response_hash": _bytes_hash(fetched.raw_body),
                "raw_response_size_bytes": len(fetched.raw_body),
                "transport_mode": fetched.transport_mode,
                "runtime_identity": capture_source_runtime_identity_v62(),
                "fixture_only": fetched.contract.fixture_only,
                "scientific_provenance_status": (
                    "NOT_RUN" if fetched.contract.fixture_only else "HUMAN"
                ),
                "acquired_at": fetched.receipt.retrieved_at,
                "authenticated_at": when,
            },
        )
        if not isinstance(issued, SourceAcquisitionReceiptV62):
            raise TypeError("source authority issued the wrong receipt type")
        return AuthenticatedWorldBankAcquisitionV62(
            fetched=fetched,
            authority_receipt=issued,
        )

    def _verify_acquisition_static(
        self,
        *,
        workspace_spec: TaskWorkspaceSpecV50,
        contract: WorldBankSourceContractV62,
        source_receipt: WorldBankSourceReceiptV62,
        snapshot: ODETimeSeriesSnapshotV52,
        receipt: SourceAcquisitionReceiptV62,
    ) -> bool:
        try:
            workspace_spec.assert_sealed()
            contract.assert_sealed()
            source_receipt.assert_sealed()
            snapshot.assert_sealed()
        except ValueError:
            return False
        return bool(
            self._verify_auth(kind="acquisition", receipt=receipt)
            and receipt.workspace_id == workspace_spec.workspace_id
            and receipt.graph_id == workspace_spec.graph_id
            and receipt.workspace_spec_hash == workspace_spec.spec_hash
            and receipt.contract_hash == contract.contract_hash
            and receipt.source_receipt_hash == source_receipt.receipt_hash
            and receipt.snapshot_hash == snapshot.snapshot_hash
            and receipt.exact_url
            == contract.exact_url
            == source_receipt.exact_url
            == source_receipt.final_url
            and receipt.raw_response_hash
            == source_receipt.response_bytes_hash
            and receipt.raw_response_size_bytes
            == source_receipt.response_size_bytes
            and receipt.fixture_only
            == contract.fixture_only
            == source_receipt.fixture_only
            == snapshot.fixture_only
            and receipt.acquired_at == source_receipt.retrieved_at
            and snapshot.task_id == workspace_spec.workspace_id
            and source_receipt.snapshot_hash == snapshot.snapshot_hash
        )

    def verify_acquisition(
        self,
        *,
        workspace_spec: TaskWorkspaceSpecV50,
        contract: WorldBankSourceContractV62,
        source_receipt: WorldBankSourceReceiptV62,
        snapshot: ODETimeSeriesSnapshotV52,
        raw_body: bytes,
        receipt: SourceAcquisitionReceiptV62,
    ) -> bool:
        """Verify authority, workspace binding, and the currently supplied bytes."""

        return bool(
            self._verify_acquisition_static(
                workspace_spec=workspace_spec,
                contract=contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                receipt=receipt,
            )
            and _bytes_hash(raw_body) == receipt.raw_response_hash
            and len(raw_body) == receipt.raw_response_size_bytes
        )

    def reverify_world_bank_source_at_s2(
        self,
        *,
        workspace: StageWorkspaceV50,
        raw_baseline: RawDataBaselineV50,
        contract: WorldBankSourceContractV62,
        source_receipt: WorldBankSourceReceiptV62,
        snapshot: ODETimeSeriesSnapshotV52,
        acquisition_receipt: SourceAcquisitionReceiptV62,
        acquisition_authority: "SourceTransportAuthorityV62 | None" = None,
        reverified_at: datetime | None = None,
    ) -> S2WorldBankReverificationV62:
        """Reparse raw bytes and sign the result for the current S2 attempt."""

        _assert_current_s2_context(workspace, raw_baseline)
        acquiring_authority = acquisition_authority or self
        if not acquiring_authority._verify_acquisition_static(
            workspace_spec=workspace.spec,
            contract=contract,
            source_receipt=source_receipt,
            snapshot=snapshot,
            receipt=acquisition_receipt,
        ):
            raise ValueError(
                "source acquisition lacks authenticated authority binding"
            )
        when = reverified_at or _utc_now()
        _assert_timezone(when, "reverified_at")
        if (
            when < raw_baseline.frozen_at
            or when < acquisition_receipt.authenticated_at
        ):
            raise ValueError("S2 source re-verification is not chronologically fresh")
        verification = verify_world_bank_source_v62(
            workspace_root=workspace.root,
            contract=contract,
            receipt=source_receipt,
            snapshot=snapshot,
            verified_at=when,
        )
        issued = self._issue(
            kind="s2_reverification",
            model_type=S2SourceReverificationReceiptV62,
            data={
                "workspace_id": workspace.spec.workspace_id,
                "graph_id": workspace.spec.graph_id,
                "workspace_spec_hash": workspace.spec.spec_hash,
                "s1_gate_hash": raw_baseline.s1_gate_hash,
                "s2_attempt": raw_baseline.s2_attempt,
                "raw_baseline_hash": raw_baseline.baseline_hash,
                "source_acquisition_receipt_hash": (
                    acquisition_receipt.receipt_hash
                ),
                "contract_hash": contract.contract_hash,
                "source_receipt_hash": source_receipt.receipt_hash,
                "snapshot_hash": snapshot.snapshot_hash,
                "exact_url": contract.exact_url,
                "raw_response_hash": source_receipt.response_bytes_hash,
                "source_verification_hash": verification.verification_hash,
                "source_checks_hash": sha256_value(verification.checks),
                "replay_status": verification.status,
                "reason_codes": verification.reason_codes,
                "runtime_identity": capture_source_runtime_identity_v62(),
                "fixture_only": contract.fixture_only,
                "official_live_transport_authenticated": (
                    not contract.fixture_only
                ),
                "scientific_provenance_status": (
                    verification.scientific_provenance_status
                ),
                "reverified_at": verification.verified_at,
            },
        )
        if not isinstance(issued, S2SourceReverificationReceiptV62):
            raise TypeError("source authority issued the wrong receipt type")
        return S2WorldBankReverificationV62(
            verification=verification,
            authority_receipt=issued,
        )

    def verify_s2_reverification(
        self,
        *,
        workspace: StageWorkspaceV50,
        raw_baseline: RawDataBaselineV50,
        contract: WorldBankSourceContractV62,
        source_receipt: WorldBankSourceReceiptV62,
        snapshot: ODETimeSeriesSnapshotV52,
        acquisition_receipt: SourceAcquisitionReceiptV62,
        verification: SourceVerificationV62,
        receipt: S2SourceReverificationReceiptV62,
        acquisition_authority: "SourceTransportAuthorityV62 | None" = None,
    ) -> bool:
        """Verify current-attempt binding and replay the current raw bytes again."""

        try:
            _assert_current_s2_context(workspace, raw_baseline)
            verification.assert_sealed()
            acquiring_authority = acquisition_authority or self
            if not acquiring_authority._verify_acquisition_static(
                workspace_spec=workspace.spec,
                contract=contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                receipt=acquisition_receipt,
            ):
                return False
            replay = verify_world_bank_source_v62(
                workspace_root=workspace.root,
                contract=contract,
                receipt=source_receipt,
                snapshot=snapshot,
                verified_at=receipt.reverified_at,
            )
            return bool(
                self._verify_auth(
                    kind="s2_reverification",
                    receipt=receipt,
                )
                and receipt.workspace_id == workspace.spec.workspace_id
                and receipt.graph_id == workspace.spec.graph_id
                and receipt.workspace_spec_hash == workspace.spec.spec_hash
                and receipt.s1_gate_hash == raw_baseline.s1_gate_hash
                and receipt.s2_attempt == raw_baseline.s2_attempt
                and receipt.raw_baseline_hash == raw_baseline.baseline_hash
                and receipt.source_acquisition_receipt_hash
                == acquisition_receipt.receipt_hash
                and receipt.contract_hash == contract.contract_hash
                and receipt.source_receipt_hash == source_receipt.receipt_hash
                and receipt.snapshot_hash == snapshot.snapshot_hash
                and receipt.exact_url
                == contract.exact_url
                == source_receipt.exact_url
                and receipt.raw_response_hash
                == source_receipt.response_bytes_hash
                and receipt.source_verification_hash
                == verification.verification_hash
                == replay.verification_hash
                and receipt.source_checks_hash
                == sha256_value(verification.checks)
                == sha256_value(replay.checks)
                and receipt.replay_status
                == verification.status
                == replay.status
                and receipt.reason_codes
                == verification.reason_codes
                == replay.reason_codes
                and receipt.fixture_only
                == contract.fixture_only
                == verification.fixture_only
                and receipt.scientific_provenance_status
                == verification.scientific_provenance_status
                == replay.scientific_provenance_status
                and receipt.reverified_at == verification.verified_at
            )
        except (OSError, TypeError, ValueError):
            return False

    def is_s2_reverification_admissible(
        self,
        *,
        workspace: StageWorkspaceV50,
        raw_baseline: RawDataBaselineV50,
        contract: WorldBankSourceContractV62,
        source_receipt: WorldBankSourceReceiptV62,
        snapshot: ODETimeSeriesSnapshotV52,
        acquisition_receipt: SourceAcquisitionReceiptV62,
        verification: SourceVerificationV62,
        receipt: S2SourceReverificationReceiptV62,
        acquisition_authority: "SourceTransportAuthorityV62 | None" = None,
    ) -> bool:
        """Require both authentic current evidence and a fresh replay PASS."""

        return bool(
            verification.status == "PASS"
            and receipt.replay_status == "PASS"
            and self.verify_s2_reverification(
                workspace=workspace,
                raw_baseline=raw_baseline,
                contract=contract,
                source_receipt=source_receipt,
                snapshot=snapshot,
                acquisition_receipt=acquisition_receipt,
                verification=verification,
                receipt=receipt,
                acquisition_authority=acquisition_authority,
            )
        )


def _assert_current_s2_context(
    workspace: StageWorkspaceV50,
    raw_baseline: RawDataBaselineV50,
) -> None:
    workspace.spec.assert_sealed()
    if raw_baseline.baseline_hash is None:
        raise ValueError("S2 source re-verification needs a sealed raw baseline")
    current_s1_gate = workspace.current_gate("S1")
    current_attempt = workspace._latest_attempt("S2")
    current_baseline = workspace._raw_baseline_for_current_s2()
    if (
        current_s1_gate is None
        or not workspace.verify_raw_baseline(raw_baseline)
        or raw_baseline.workspace_spec_hash != workspace.spec.spec_hash
        or raw_baseline.s1_gate_hash != current_s1_gate
        or raw_baseline.s2_attempt != current_attempt
        or current_baseline is None
        or current_baseline.baseline_hash != raw_baseline.baseline_hash
    ):
        raise ValueError(
            "source re-verification is stale for the current S2 attempt"
        )


__all__ = [
    "AuthenticatedWorldBankAcquisitionV62",
    "S2_SOURCE_REVERIFICATION_PATH",
    "SOURCE_ACQUISITION_AUTH_PATH",
    "S2SourceReverificationReceiptV62",
    "S2WorldBankReverificationV62",
    "SourceAcquisitionReceiptV62",
    "SourceRuntimeIdentityV62",
    "SourceTransportAuthorityV62",
    "capture_source_runtime_identity_v62",
]
