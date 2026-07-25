from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Callable, Literal, Mapping
from urllib.request import Request, urlopen
from uuid import uuid4

from pydantic import Field, model_validator

from fma.hashing import sha256_value
from fma.schemas import ArtifactRef, StrictModel
from fma.storage import RunStore

from .schemas import Identifier, Sha256, _assert_timezone


APPROVED_METHOD_URLS = frozenset(
    {
        "https://otexts.com/fpp3/ses.html",
        "https://otexts.com/fpp3/holt.html",
    }
)


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class MethodSourceContractV22(StrictModel):
    """A narrow, exact-URL read contract for untrusted method material."""

    schema_version: Literal["2.2"] = "2.2"
    source_id: Identifier
    source_url: Annotated[str, Field(min_length=12, max_length=512)]
    expected_content_type: Literal["text/html"] = "text/html"
    max_bytes: Annotated[int, Field(ge=1_024, le=1_048_576)] = 524_288
    approved_claim_scope: list[Annotated[str, Field(min_length=3)]] = Field(
        min_length=1
    )
    created_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "MethodSourceContractV22":
        _assert_timezone(self.created_at, "created_at")
        if self.source_url not in APPROVED_METHOD_URLS:
            raise ValueError("method source URL is not on the exact allowlist")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("contract_hash does not match method source contract")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("method source contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "MethodSourceContractV22":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"contract_hash"}),
            contract_hash=draft.content_hash(),
        )


class MethodEvidenceSnapshotV22(StrictModel):
    """Raw web evidence.  Its text is data and never an instruction channel."""

    schema_version: Literal["2.2"] = "2.2"
    snapshot_id: Identifier
    contract_hash: Sha256
    source_url: Annotated[str, Field(min_length=12, max_length=512)]
    final_url: Annotated[str, Field(min_length=12, max_length=512)]
    content_type: Literal["text/html"] = "text/html"
    raw_text: Annotated[str, Field(min_length=1, max_length=1_048_576)]
    response_content_hash: Sha256
    etag: Annotated[str, Field(max_length=512)] = ""
    last_modified: Annotated[str, Field(max_length=512)] = ""
    trust_class: Literal["untrusted_web_data"] = "untrusted_web_data"
    retrieved_at: datetime
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "MethodEvidenceSnapshotV22":
        _assert_timezone(self.retrieved_at, "retrieved_at")
        if self.source_url != self.final_url:
            raise ValueError("method source redirected away from its approved exact URL")
        if self.response_content_hash != sha256_value({"body_utf8": self.raw_text}):
            raise ValueError("response_content_hash does not match method source body")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match method evidence snapshot")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("method evidence snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "MethodEvidenceSnapshotV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class MethodClaimDraftV22(StrictModel):
    """Model-proposed interpretation; admission does not establish truth."""

    schema_version: Literal["2.2"] = "2.2"
    claim_id: Identifier
    evidence_snapshot_hash: Sha256
    statement: Annotated[str, Field(min_length=12)]
    applicability_conditions: list[Annotated[str, Field(min_length=3)]] = Field(
        min_length=1
    )
    exclusions: list[Annotated[str, Field(min_length=3)]] = Field(min_length=1)
    proposed_operator: Literal["exponential_smoothing", "damped_trend"]
    frozen_parameters: dict[str, Annotated[float, Field(allow_inf_nan=False)]]
    created_at: datetime
    draft_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_draft(self) -> "MethodClaimDraftV22":
        _assert_timezone(self.created_at, "created_at")
        if self.proposed_operator == "exponential_smoothing":
            alpha = self.frozen_parameters.get("alpha")
            if alpha is None or not 0 < alpha < 1:
                raise ValueError("exponential smoothing requires frozen alpha in (0, 1)")
        if self.draft_hash and self.draft_hash != self.content_hash():
            raise ValueError("draft_hash does not match method claim draft")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "draft_hash")

    def assert_sealed(self) -> None:
        if not self.draft_hash or self.draft_hash != self.content_hash():
            raise ValueError("method claim draft is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "MethodClaimDraftV22":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"draft_hash"}),
            draft_hash=draft.content_hash(),
        )


class EvolutionOperatorKnowledgeV22(StrictModel):
    """Candidate memory.  Only a separate benchmark may promote its use."""

    schema_version: Literal["2.2"] = "2.2"
    operator_id: Identifier
    source_contract_hash: Sha256
    source_snapshot_hash: Sha256
    source_claim_hash: Sha256
    operator_family: Literal["exponential_smoothing", "damped_trend"]
    frozen_parameters: dict[str, Annotated[float, Field(allow_inf_nan=False)]]
    status: Literal["candidate_requires_hidden_validation"] = (
        "candidate_requires_hidden_validation"
    )
    validation_requirement: Literal["paired_hidden_worldpack_ablation"] = (
        "paired_hidden_worldpack_ablation"
    )
    created_at: datetime
    knowledge_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_knowledge(self) -> "EvolutionOperatorKnowledgeV22":
        _assert_timezone(self.created_at, "created_at")
        if self.knowledge_hash and self.knowledge_hash != self.content_hash():
            raise ValueError("knowledge_hash does not match evolution operator")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "knowledge_hash")

    def assert_sealed(self) -> None:
        if not self.knowledge_hash or self.knowledge_hash != self.content_hash():
            raise ValueError("evolution operator knowledge is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "EvolutionOperatorKnowledgeV22":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"knowledge_hash"}),
            knowledge_hash=draft.content_hash(),
        )


class MethodLearningManifestV22(StrictModel):
    schema_version: Literal["2.2"] = "2.2"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    artifact_refs: list[ArtifactRef] = Field(min_length=4, max_length=4)
    terminal_status: Literal["candidate_requires_hidden_validation"]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "MethodLearningManifestV22":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != 4:
            raise ValueError("method learning manifest references must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match method learning manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("method learning manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "MethodLearningManifestV22":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class MethodFetchResponse:
    status: int
    final_url: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class MethodLearningOutcome:
    store: RunStore
    contract: MethodSourceContractV22
    snapshot: MethodEvidenceSnapshotV22
    claim: MethodClaimDraftV22
    knowledge: EvolutionOperatorKnowledgeV22
    manifest: MethodLearningManifestV22


MethodFetcher = Callable[[str, int], MethodFetchResponse]


def fetch_method_source(
    contract: MethodSourceContractV22,
    *,
    fetcher: MethodFetcher | None = None,
    retrieved_at: datetime | None = None,
) -> MethodEvidenceSnapshotV22:
    contract.assert_sealed()
    response = (fetcher or _default_fetcher)(contract.source_url, contract.max_bytes)
    if response.status != 200:
        raise RuntimeError(f"method source returned HTTP {response.status}")
    if response.final_url != contract.source_url:
        raise RuntimeError("method source redirected away from the exact allowlist URL")
    if len(response.body) > contract.max_bytes:
        raise RuntimeError("method source exceeded max_bytes")
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != contract.expected_content_type:
        raise RuntimeError("method source content type does not match its contract")
    try:
        raw_text = response.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("method source is not valid UTF-8") from exc
    return MethodEvidenceSnapshotV22.seal(
        snapshot_id=f"{contract.source_id}_snapshot",
        contract_hash=contract.contract_hash,
        source_url=contract.source_url,
        final_url=response.final_url,
        content_type="text/html",
        raw_text=raw_text,
        response_content_hash=sha256_value({"body_utf8": raw_text}),
        etag=response.headers.get("etag", "")[:512],
        last_modified=response.headers.get("last-modified", "")[:512],
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
    )


def admit_method_claim(
    contract: MethodSourceContractV22,
    snapshot: MethodEvidenceSnapshotV22,
    claim: MethodClaimDraftV22,
) -> EvolutionOperatorKnowledgeV22:
    """Bind a proposal to evidence while deliberately retaining candidate status."""

    contract.assert_sealed()
    snapshot.assert_sealed()
    claim.assert_sealed()
    if snapshot.contract_hash != contract.contract_hash:
        raise ValueError("method snapshot is bound to another source contract")
    if snapshot.source_url != contract.source_url:
        raise ValueError("method snapshot came from another source URL")
    if claim.evidence_snapshot_hash != snapshot.snapshot_hash:
        raise ValueError("method claim is bound to another evidence snapshot")
    assert contract.contract_hash is not None
    assert snapshot.snapshot_hash is not None
    assert claim.draft_hash is not None
    return EvolutionOperatorKnowledgeV22.seal(
        operator_id=f"{claim.claim_id}_operator",
        source_contract_hash=contract.contract_hash,
        source_snapshot_hash=snapshot.snapshot_hash,
        source_claim_hash=claim.draft_hash,
        operator_family=claim.proposed_operator,
        frozen_parameters=claim.frozen_parameters,
        created_at=claim.created_at,
    )


def capture_method_candidate(
    output_root: str | Path,
    *,
    contract: MethodSourceContractV22,
    claim_id: str,
    statement: str,
    applicability_conditions: list[str],
    exclusions: list[str],
    proposed_operator: Literal["exponential_smoothing", "damped_trend"],
    frozen_parameters: dict[str, float],
    fetcher: MethodFetcher | None = None,
    captured_at: datetime | None = None,
    run_id: str | None = None,
) -> MethodLearningOutcome:
    at = captured_at or datetime.now(timezone.utc)
    snapshot = fetch_method_source(contract, fetcher=fetcher, retrieved_at=at)
    assert snapshot.snapshot_hash is not None
    claim = MethodClaimDraftV22.seal(
        claim_id=claim_id,
        evidence_snapshot_hash=snapshot.snapshot_hash,
        statement=statement,
        applicability_conditions=applicability_conditions,
        exclusions=exclusions,
        proposed_operator=proposed_operator,
        frozen_parameters=frozen_parameters,
        created_at=at,
    )
    knowledge = admit_method_claim(contract, snapshot, claim)
    store = RunStore(
        output_root,
        run_id=run_id or f"method-learning-{contract.source_id}-{uuid4().hex[:10]}",
    )
    refs = [
        store.put_artifact("method_source_contract_v22", contract),
        store.put_artifact("method_evidence_snapshot_v22", snapshot),
        store.put_artifact("method_claim_draft_v22", claim),
        store.put_artifact("evolution_operator_knowledge_v22", knowledge),
    ]
    manifest = MethodLearningManifestV22.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=knowledge.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("method_learning_manifest_v22", manifest)
    store.emit(
        "method_learning_candidate_captured",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_method_learning_run(store.run_directory):
        raise RuntimeError("method learning run failed independent verification")
    return MethodLearningOutcome(store, contract, snapshot, claim, knowledge, manifest)


def verify_method_learning_run(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [
            json.loads(line)
            for line in store.event_path.read_text(encoding="utf-8").splitlines()
        ]
        refs = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in refs:
            store.load_artifact(ref)
        manifests = [ref for ref in refs if ref.kind == "method_learning_manifest_v22"]
        if len(manifests) != 1:
            return False
        manifest = MethodLearningManifestV22.model_validate(store.load_artifact(manifests[0]))
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_one(kind: str, model):
            matches = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(matches) != 1:
                raise RuntimeError(f"manifest needs exactly one {kind}")
            return model.model_validate(store.load_artifact(matches[0]))

        contract = load_one("method_source_contract_v22", MethodSourceContractV22)
        snapshot = load_one("method_evidence_snapshot_v22", MethodEvidenceSnapshotV22)
        claim = load_one("method_claim_draft_v22", MethodClaimDraftV22)
        knowledge = load_one(
            "evolution_operator_knowledge_v22", EvolutionOperatorKnowledgeV22
        )
        contract.assert_sealed()
        snapshot.assert_sealed()
        claim.assert_sealed()
        knowledge.assert_sealed()
        recomputed = admit_method_claim(contract, snapshot, claim)
        return (
            recomputed.knowledge_hash == knowledge.knowledge_hash
            and manifest.terminal_status == knowledge.status
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        return False


def _default_fetcher(url: str, max_bytes: int) -> MethodFetchResponse:
    request = Request(
        url,
        headers={"User-Agent": "FMA-MethodEvidence/2.2 (+local research harness)"},
        method="GET",
    )
    with urlopen(request, timeout=20) as response:  # noqa: S310 - exact allowlist above
        body = response.read(max_bytes + 1)
        return MethodFetchResponse(
            status=int(response.status),
            final_url=str(response.geturl()),
            headers={key.lower(): value for key, value in response.headers.items()},
            body=body,
        )
