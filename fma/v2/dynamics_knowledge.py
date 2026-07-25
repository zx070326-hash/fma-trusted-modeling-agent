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


SourceId = Literal["sindy_pnas", "weak_sindy", "structural_identifiability"]
DesignRule = Literal[
    "explicit_candidate_library",
    "sparsity_requires_basis_match",
    "noise_aware_derivative_estimation",
    "identifiability_before_parameter_claim",
    "trajectory_holdout_before_use",
]


APPROVED_DYNAMICS_SOURCES: dict[str, tuple[str, str]] = {
    "sindy_pnas": (
        "10.1073/pnas.1517384113",
        "https://api.crossref.org/works/10.1073%2Fpnas.1517384113",
    ),
    "weak_sindy": (
        "10.1137/20m1343166",
        "https://api.crossref.org/works/10.1137%2F20m1343166",
    ),
    "structural_identifiability": (
        "10.1155/2019/8497093",
        "https://api.crossref.org/works/10.1155%2F2019%2F8497093",
    ),
}


def _hash_without(model: StrictModel, field: str) -> str:
    return sha256_value(model.model_dump(mode="json", exclude={field}))


class LiteratureSourceContractV24(StrictModel):
    """Exact DOI-metadata read contract; returned text is never instruction authority."""

    schema_version: Literal["2.4"] = "2.4"
    source_id: SourceId
    doi: Annotated[str, Field(min_length=10, max_length=128)]
    api_url: Annotated[str, Field(min_length=20, max_length=512)]
    expected_content_type: Literal["application/json"] = "application/json"
    max_bytes: Annotated[int, Field(ge=4_096, le=1_048_576)] = 524_288
    approved_claim_scope: list[Annotated[str, Field(min_length=8)]] = Field(
        min_length=1
    )
    created_at: datetime
    contract_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "LiteratureSourceContractV24":
        _assert_timezone(self.created_at, "created_at")
        expected = APPROVED_DYNAMICS_SOURCES.get(self.source_id)
        if expected is None or (self.doi.lower(), self.api_url) != expected:
            raise ValueError("literature source is not on the exact DOI/API allowlist")
        if self.contract_hash and self.contract_hash != self.content_hash():
            raise ValueError("contract_hash does not match literature source contract")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "contract_hash")

    def assert_sealed(self) -> None:
        if not self.contract_hash or self.contract_hash != self.content_hash():
            raise ValueError("literature source contract is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "LiteratureSourceContractV24":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"contract_hash"}),
            contract_hash=draft.content_hash(),
        )


class LiteratureEvidenceSnapshotV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    snapshot_id: Identifier
    contract_hash: Sha256
    source_id: SourceId
    doi: Annotated[str, Field(min_length=10, max_length=128)]
    source_url: Annotated[str, Field(min_length=20, max_length=512)]
    final_url: Annotated[str, Field(min_length=20, max_length=512)]
    raw_json: Annotated[str, Field(min_length=2, max_length=1_048_576)]
    response_content_hash: Sha256
    title: Annotated[str, Field(min_length=8, max_length=1_024)]
    publisher: Annotated[str, Field(min_length=1, max_length=512)]
    trust_class: Literal["untrusted_bibliographic_data"] = (
        "untrusted_bibliographic_data"
    )
    retrieved_at: datetime
    snapshot_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "LiteratureEvidenceSnapshotV24":
        _assert_timezone(self.retrieved_at, "retrieved_at")
        if self.source_url != self.final_url:
            raise ValueError("literature endpoint redirected away from its exact URL")
        if self.response_content_hash != sha256_value({"body_utf8": self.raw_json}):
            raise ValueError("response_content_hash does not match literature body")
        payload = json.loads(self.raw_json)
        if payload.get("status") != "ok" or payload.get("message-type") != "work":
            raise ValueError("Crossref payload is not a successful single-work response")
        message = payload.get("message", {})
        if str(message.get("DOI", "")).lower() != self.doi.lower():
            raise ValueError("Crossref payload DOI does not match source contract")
        titles = message.get("title") or []
        if not titles or str(titles[0]) != self.title:
            raise ValueError("literature title was not extracted from the frozen payload")
        if str(message.get("publisher", "")) != self.publisher:
            raise ValueError("literature publisher was not extracted from the frozen payload")
        if self.snapshot_hash and self.snapshot_hash != self.content_hash():
            raise ValueError("snapshot_hash does not match literature evidence")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "snapshot_hash")

    def assert_sealed(self) -> None:
        if not self.snapshot_hash or self.snapshot_hash != self.content_hash():
            raise ValueError("literature evidence snapshot is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "LiteratureEvidenceSnapshotV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"snapshot_hash"}),
            snapshot_hash=draft.content_hash(),
        )


class DynamicsMethodClaimV24(StrictModel):
    """A proposed interpretation, deliberately weaker than verified knowledge."""

    schema_version: Literal["2.4"] = "2.4"
    claim_id: Identifier
    evidence_snapshot_hash: Sha256
    statement: Annotated[str, Field(min_length=16)]
    applicability_conditions: list[Annotated[str, Field(min_length=8)]] = Field(
        min_length=1
    )
    exclusions: list[Annotated[str, Field(min_length=8)]] = Field(min_length=1)
    design_rules: list[DesignRule] = Field(min_length=1)
    status: Literal["candidate_interpretation_only"] = "candidate_interpretation_only"
    created_at: datetime
    claim_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_claim(self) -> "DynamicsMethodClaimV24":
        _assert_timezone(self.created_at, "created_at")
        if len(self.design_rules) != len(set(self.design_rules)):
            raise ValueError("dynamics design rules must be unique within a claim")
        if self.claim_hash and self.claim_hash != self.content_hash():
            raise ValueError("claim_hash does not match dynamics method claim")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "claim_hash")

    def assert_sealed(self) -> None:
        if not self.claim_hash or self.claim_hash != self.content_hash():
            raise ValueError("dynamics method claim is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsMethodClaimV24":
        data.setdefault("created_at", datetime.now(timezone.utc))
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"claim_hash"}),
            claim_hash=draft.content_hash(),
        )


class DynamicsKnowledgeBundleV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    bundle_id: Identifier
    source_contract_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    evidence_snapshot_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    claim_hashes: list[Sha256] = Field(min_length=3, max_length=3)
    exact_design_rules: list[DesignRule] = Field(min_length=5, max_length=5)
    status: Literal["candidate_requires_hidden_dynamics_validation"] = (
        "candidate_requires_hidden_dynamics_validation"
    )
    limitations: list[Annotated[str, Field(min_length=12)]] = Field(min_length=3)
    created_at: datetime
    bundle_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle(self) -> "DynamicsKnowledgeBundleV24":
        _assert_timezone(self.created_at, "created_at")
        for values in (
            self.source_contract_hashes,
            self.evidence_snapshot_hashes,
            self.claim_hashes,
            self.exact_design_rules,
        ):
            if len(values) != len(set(values)):
                raise ValueError("knowledge bundle bindings must be unique")
        if set(self.exact_design_rules) != {
            "explicit_candidate_library",
            "sparsity_requires_basis_match",
            "noise_aware_derivative_estimation",
            "identifiability_before_parameter_claim",
            "trajectory_holdout_before_use",
        }:
            raise ValueError("dynamics knowledge bundle is missing a frozen design rule")
        if self.bundle_hash and self.bundle_hash != self.content_hash():
            raise ValueError("bundle_hash does not match dynamics knowledge bundle")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "bundle_hash")

    def assert_sealed(self) -> None:
        if not self.bundle_hash or self.bundle_hash != self.content_hash():
            raise ValueError("dynamics knowledge bundle is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsKnowledgeBundleV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"bundle_hash"}),
            bundle_hash=draft.content_hash(),
        )


class DynamicsKnowledgeManifestV24(StrictModel):
    schema_version: Literal["2.4"] = "2.4"
    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    artifact_refs: list[ArtifactRef] = Field(min_length=10, max_length=10)
    terminal_status: Literal["candidate_requires_hidden_dynamics_validation"]
    created_at: datetime
    manifest_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_manifest(self) -> "DynamicsKnowledgeManifestV24":
        _assert_timezone(self.created_at, "created_at")
        if len({(ref.kind, ref.sha256) for ref in self.artifact_refs}) != 10:
            raise ValueError("dynamics knowledge manifest references must be unique")
        if self.manifest_hash and self.manifest_hash != self.content_hash():
            raise ValueError("manifest_hash does not match dynamics knowledge manifest")
        return self

    def content_hash(self) -> str:
        return _hash_without(self, "manifest_hash")

    def assert_sealed(self) -> None:
        if not self.manifest_hash or self.manifest_hash != self.content_hash():
            raise ValueError("dynamics knowledge manifest is not sealed")

    @classmethod
    def seal(cls, **data: object) -> "DynamicsKnowledgeManifestV24":
        draft = cls(**data)
        return cls(
            **draft.model_dump(exclude={"manifest_hash"}),
            manifest_hash=draft.content_hash(),
        )


@dataclass(frozen=True)
class LiteratureFetchResponse:
    status: int
    final_url: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class DynamicsKnowledgeOutcome:
    store: RunStore
    contracts: list[LiteratureSourceContractV24]
    snapshots: list[LiteratureEvidenceSnapshotV24]
    claims: list[DynamicsMethodClaimV24]
    bundle: DynamicsKnowledgeBundleV24
    manifest: DynamicsKnowledgeManifestV24


LiteratureFetcher = Callable[[str, int], LiteratureFetchResponse]


def default_dynamics_source_contracts(
    *, created_at: datetime | None = None
) -> list[LiteratureSourceContractV24]:
    at = created_at or datetime.now(timezone.utc)
    scopes = {
        "sindy_pnas": [
            "candidate function libraries and sparsity assumptions in ODE discovery"
        ],
        "weak_sindy": [
            "noise sensitivity of point derivatives and weak-form mitigation"
        ],
        "structural_identifiability": [
            "observability and parameter identifiability boundaries for nonlinear ODEs"
        ],
    }
    return [
        LiteratureSourceContractV24.seal(
            source_id=source_id,
            doi=doi,
            api_url=url,
            approved_claim_scope=scopes[source_id],
            created_at=at,
        )
        for source_id, (doi, url) in APPROVED_DYNAMICS_SOURCES.items()
    ]


def fetch_literature_source(
    contract: LiteratureSourceContractV24,
    *,
    fetcher: LiteratureFetcher | None = None,
    retrieved_at: datetime | None = None,
) -> LiteratureEvidenceSnapshotV24:
    contract.assert_sealed()
    response = (fetcher or _default_fetcher)(contract.api_url, contract.max_bytes)
    if response.status != 200:
        raise RuntimeError(f"literature source returned HTTP {response.status}")
    if response.final_url != contract.api_url:
        raise RuntimeError("literature source redirected away from its exact URL")
    if len(response.body) > contract.max_bytes:
        raise RuntimeError("literature source exceeded max_bytes")
    media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if media_type != contract.expected_content_type:
        raise RuntimeError("literature source content type does not match contract")
    try:
        raw_json = response.body.decode("utf-8")
        payload = json.loads(raw_json)
        message = payload["message"]
        title = str(message["title"][0])
        publisher = str(message["publisher"])
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("literature source is not valid Crossref work JSON") from exc
    return LiteratureEvidenceSnapshotV24.seal(
        snapshot_id=f"{contract.source_id}_snapshot",
        contract_hash=contract.contract_hash,
        source_id=contract.source_id,
        doi=contract.doi,
        source_url=contract.api_url,
        final_url=response.final_url,
        raw_json=raw_json,
        response_content_hash=sha256_value({"body_utf8": raw_json}),
        title=title,
        publisher=publisher,
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
    )


def _claim_for_snapshot(
    snapshot: LiteratureEvidenceSnapshotV24, at: datetime
) -> DynamicsMethodClaimV24:
    assert snapshot.snapshot_hash is not None
    definitions: dict[str, dict[str, object]] = {
        "sindy_pnas": {
            "claim_id": "sindy_library_sparsity_candidate",
            "statement": (
                "Sparse ODE discovery searches a frozen candidate-function library; "
                "success therefore depends on the true dynamics being sparse in that basis."
            ),
            "applicability_conditions": [
                "state trajectories are observed densely enough to construct a design matrix"
            ],
            "exclusions": [
                "the true dynamics are not assumed sparse in an arbitrary chosen basis"
            ],
            "design_rules": [
                "explicit_candidate_library",
                "sparsity_requires_basis_match",
                "trajectory_holdout_before_use",
            ],
        },
        "weak_sindy": {
            "claim_id": "weak_form_noise_candidate",
            "statement": (
                "Pointwise differentiation can amplify measurement noise, so derivative "
                "estimation must be an explicit, validated part of an ODE-discovery policy."
            ),
            "applicability_conditions": ["measured trajectories contain observation noise"],
            "exclusions": [
                "a smoothed derivative estimate is not equivalent to a formal weak-form method"
            ],
            "design_rules": ["noise_aware_derivative_estimation"],
        },
        "structural_identifiability": {
            "claim_id": "identifiability_boundary_candidate",
            "statement": (
                "A fitted trajectory does not by itself establish that hidden states or "
                "parameters are uniquely recoverable from the observed outputs."
            ),
            "applicability_conditions": [
                "a model is used to make state or parameter recovery claims"
            ],
            "exclusions": [
                "numeric rank on one trajectory is not a general structural-identifiability proof"
            ],
            "design_rules": ["identifiability_before_parameter_claim"],
        },
    }
    return DynamicsMethodClaimV24.seal(
        evidence_snapshot_hash=snapshot.snapshot_hash,
        created_at=at,
        **definitions[snapshot.source_id],
    )


def capture_dynamics_knowledge(
    output_root: str | Path,
    *,
    fetcher: LiteratureFetcher | None = None,
    captured_at: datetime | None = None,
    run_id: str | None = None,
) -> DynamicsKnowledgeOutcome:
    at = captured_at or datetime.now(timezone.utc)
    contracts = default_dynamics_source_contracts(created_at=at)
    snapshots = [
        fetch_literature_source(contract, fetcher=fetcher, retrieved_at=at)
        for contract in contracts
    ]
    claims = [_claim_for_snapshot(snapshot, at) for snapshot in snapshots]
    bundle = DynamicsKnowledgeBundleV24.seal(
        bundle_id="dynamics_first_principles_knowledge_v24",
        source_contract_hashes=[contract.contract_hash for contract in contracts],
        evidence_snapshot_hashes=[snapshot.snapshot_hash for snapshot in snapshots],
        claim_hashes=[claim.claim_hash for claim in claims],
        exact_design_rules=[
            "explicit_candidate_library",
            "sparsity_requires_basis_match",
            "noise_aware_derivative_estimation",
            "identifiability_before_parameter_claim",
            "trajectory_holdout_before_use",
        ],
        limitations=[
            "bibliographic metadata proves source identity, not semantic truth of a proposed claim",
            "the bundle is candidate memory until an independent hidden WorldPack tests an exact policy",
            "no source establishes real-world external validity for this implementation",
        ],
        created_at=at,
    )
    store = RunStore(
        output_root,
        run_id=run_id or f"dynamics-knowledge-{uuid4().hex[:10]}",
    )
    refs: list[ArtifactRef] = []
    refs.extend(store.put_artifact("literature_source_contract_v24", item) for item in contracts)
    refs.extend(store.put_artifact("literature_evidence_snapshot_v24", item) for item in snapshots)
    refs.extend(store.put_artifact("dynamics_method_claim_v24", item) for item in claims)
    refs.append(store.put_artifact("dynamics_knowledge_bundle_v24", bundle))
    manifest = DynamicsKnowledgeManifestV24.seal(
        run_id=store.run_id,
        artifact_refs=refs,
        terminal_status=bundle.status,
        created_at=at,
    )
    manifest_ref = store.put_artifact("dynamics_knowledge_manifest_v24", manifest)
    store.emit(
        "dynamics_knowledge_candidate_captured",
        {"manifest_ref": manifest_ref.model_dump(mode="json")},
    )
    if not verify_dynamics_knowledge_run(store.run_directory):
        raise RuntimeError("dynamics knowledge run failed independent verification")
    return DynamicsKnowledgeOutcome(store, contracts, snapshots, claims, bundle, manifest)


def verify_dynamics_knowledge_run(run_directory: str | Path) -> bool:
    try:
        store = RunStore.open_existing(run_directory)
        events = [json.loads(line) for line in store.event_path.read_text(encoding="utf-8").splitlines()]
        committed = [
            ArtifactRef.model_validate(event["payload"])
            for event in events
            if event["event_type"] == "artifact_committed"
        ]
        for ref in committed:
            store.load_artifact(ref)
        manifest_refs = [
            ref for ref in committed if ref.kind == "dynamics_knowledge_manifest_v24"
        ]
        if len(manifest_refs) != 1:
            return False
        manifest = DynamicsKnowledgeManifestV24.model_validate(
            store.load_artifact(manifest_refs[0])
        )
        manifest.assert_sealed()
        if manifest.run_id != store.run_id:
            return False

        def load_many(kind: str, model, count: int):
            refs = [ref for ref in manifest.artifact_refs if ref.kind == kind]
            if len(refs) != count:
                raise RuntimeError(f"manifest needs {count} {kind} artifacts")
            return [model.model_validate(store.load_artifact(ref)) for ref in refs]

        contracts = load_many("literature_source_contract_v24", LiteratureSourceContractV24, 3)
        snapshots = load_many("literature_evidence_snapshot_v24", LiteratureEvidenceSnapshotV24, 3)
        claims = load_many("dynamics_method_claim_v24", DynamicsMethodClaimV24, 3)
        bundles = load_many("dynamics_knowledge_bundle_v24", DynamicsKnowledgeBundleV24, 1)
        bundle = bundles[0]
        for item in [*contracts, *snapshots, *claims, bundle]:
            item.assert_sealed()
        contract_by_id = {item.source_id: item for item in contracts}
        snapshot_by_id = {item.source_id: item for item in snapshots}
        if set(contract_by_id) != set(APPROVED_DYNAMICS_SOURCES) or set(snapshot_by_id) != set(APPROVED_DYNAMICS_SOURCES):
            return False
        if any(
            snapshot.contract_hash != contract_by_id[source_id].contract_hash
            or snapshot.doi.lower() != contract_by_id[source_id].doi.lower()
            for source_id, snapshot in snapshot_by_id.items()
        ):
            return False
        if set(bundle.source_contract_hashes) != {item.contract_hash for item in contracts}:
            return False
        if set(bundle.evidence_snapshot_hashes) != {item.snapshot_hash for item in snapshots}:
            return False
        if set(bundle.claim_hashes) != {item.claim_hash for item in claims}:
            return False
        if any(claim.evidence_snapshot_hash not in set(bundle.evidence_snapshot_hashes) for claim in claims):
            return False
        return manifest.terminal_status == bundle.status
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


def _default_fetcher(url: str, max_bytes: int) -> LiteratureFetchResponse:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "FMA-DynamicsKnowledge/2.4 (+local research harness)",
        },
        method="GET",
    )
    with urlopen(request, timeout=20) as response:  # noqa: S310 - exact allowlist above
        body = response.read(max_bytes + 1)
        return LiteratureFetchResponse(
            status=int(response.status),
            final_url=str(response.geturl()),
            headers={key.lower(): value for key, value in response.headers.items()},
            body=body,
        )
