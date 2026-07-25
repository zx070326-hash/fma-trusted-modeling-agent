"""Graph-native S0--S6 task workspace layered on the FMA V4 authority graph.

The filesystem tree is a human/agent-legible projection.  The authoritative
state is the existing V4 event-sourced graph plus HMAC-authenticated V5
receipts bound to exact file manifests.  A raw ``gates/sN.stamp`` file has no
effect.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from fma.hashing import canonical_json, sha256_value
from fma.schemas import ArtifactRef
from fma.v4.graph_loop import (
    GraphEdgeV40,
    GraphLoopContractV40,
    GraphLoopStoreV40,
    GraphNodeV40,
)

from .paper import PaperBuildReceipt, verify_paper_build
from .workspace_schemas import (
    AdapterExecutionReceiptV50,
    CheckResultV50,
    CodeManifestV50,
    DataLedgerV50,
    DecisionFunctionSpecV50,
    DecisionDossierV50,
    FileBindingV50,
    GateCertificateV50,
    GateEvaluationV50,
    IndependentReviewReceiptV50,
    ModelSpecV50,
    PredictionSealV50,
    ProcessedManifestV50,
    RegimeDiagnosisV50,
    RawDataBaselineV50,
    ResultIndexV50,
    RoleExecutionReceiptV50,
    StageArtifactManifestV50,
    StageId,
    SymbolTableV50,
    TaskWorkspaceSpecV50,
    UQSummaryV50,
    ValidationPlanV50,
    WorkflowStatusV50,
    CandidateSetV50,
    AssumptionSetV50,
    AuthorityGenesisV50,
)


STAGES: tuple[StageId, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
_STAGE_INDEX = {stage: index for index, stage in enumerate(STAGES)}
_NODE_ID = re.compile(r"^s(?P<stage>[0-6])-a(?P<attempt>[1-9][0-9]*)-(?P<kind>work|gate)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StageWorkspaceError(RuntimeError):
    """A fail-closed V5 workspace transition was rejected."""


@dataclass(frozen=True)
class StagePolicy:
    stage: StageId
    required_paths: tuple[str, ...]
    structural_check_id: str
    required_review_roles: tuple[str, ...]

    @property
    def policy_hash(self) -> str:
        return sha256_value(
            {
                "schema_version": "5.0",
                "stage": self.stage,
                "required_paths": list(self.required_paths),
                "structural_check_id": self.structural_check_id,
                "required_review_roles": list(self.required_review_roles),
                "scientific_gate": False,
            }
        )


POLICIES: dict[StageId, StagePolicy] = {
    "S0": StagePolicy(
        "S0",
        (
            "problem/contract.json",
            "problem/decision_function.json",
            "docs/regime.json",
        ),
        "s0_regime_complete",
        ("referee",),
    ),
    "S1": StagePolicy(
        "S1",
        (
            "docs/candidates.json",
            "docs/assumptions.json",
            "docs/symbols.json",
            "docs/model_spec.json",
            "docs/validation_plan.json",
        ),
        "s1_formalization_complete",
        ("referee", "red_team"),
    ),
    "S2": StagePolicy(
        "S2",
        (
            "docs/model_spec.json",
            "docs/validation_plan.json",
            "data/ledger.json",
            "data/processed/manifest.json",
        ),
        "s2_ledger_complete",
        ("data_auditor",),
    ),
    "S3": StagePolicy(
        "S3",
        (
            "docs/validation_plan.json",
            "results/index.json",
            "results/code_manifest.json",
        ),
        "s3_reproducibility_manifest",
        ("numerics_auditor",),
    ),
    "S4": StagePolicy(
        "S4",
        (
            "docs/validation_plan.json",
            "results/index.json",
            "results/verification_summary.json",
            "results/uq_summary.json",
        ),
        "s4_uq_traceability",
        ("red_team",),
    ),
    "S5": StagePolicy(
        "S5",
        (
            "docs/validation_plan.json",
            "results/index.json",
            "results/uq_summary.json",
            "results/decision_dossier.json",
        ),
        "s5_decision_traceability",
        (),
    ),
    "S6": StagePolicy(
        "S6",
        (
            "paper/main.template.tex",
            "results/index.json",
            "results/values.json",
            "paper/build/main.tex",
            "paper/build/main.pdf",
            "paper/build/build_receipt.json",
        ),
        "s6_paper_consistency",
        ("final_red_team",),
    ),
}


_WORK_NODE_KINDS: dict[StageId, str] = {
    "S0": "workflow_plan",
    "S1": "model_proposal",
    "S2": "experiment",
    "S3": "execution",
    "S4": "execution",
    "S5": "model_proposal",
    "S6": "execution",
}
_WORK_NODE_EXECUTORS: dict[StageId, str] = {
    "S0": "model",
    "S1": "model",
    "S2": "model",
    "S3": "harness",
    "S4": "harness",
    "S5": "model",
    "S6": "harness",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree_hash(root: Path, *, include_empty: bool = True) -> str:
    """Hash regular files below ``root`` by safe relative path and bytes."""

    if not root.exists():
        if include_empty:
            return sha256_value({})
        raise StageWorkspaceError(f"tree does not exist: {root}")
    if root.is_symlink() or not root.is_dir():
        raise StageWorkspaceError(f"tree must be a regular directory: {root}")
    bindings: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StageWorkspaceError(f"evidence tree contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            bindings[relative] = _sha256_bytes(path.read_bytes())
    return sha256_value(bindings)


_ARITHMETIC_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}


def _evaluate_arithmetic(expression: str, inputs: dict[str, float]) -> float:
    """Evaluate a deliberately tiny side-effect-free arithmetic DSL."""

    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError("only finite numeric constants are allowed")
            value = float(node.value)
        elif isinstance(node, ast.Name):
            if node.id not in inputs:
                raise ValueError(f"unknown decision function input: {node.id}")
            value = float(inputs[node.id])
        elif isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.Pow):
                value = left**right
            else:
                raise ValueError("decision function uses a forbidden operator")
        elif isinstance(node, ast.UnaryOp):
            operand = visit(node.operand)
            if isinstance(node.op, ast.USub):
                value = -operand
            elif isinstance(node.op, ast.UAdd):
                value = operand
            else:
                raise ValueError("decision function uses a forbidden unary operator")
        elif isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _ARITHMETIC_FUNCTIONS
                or node.keywords
            ):
                raise ValueError("decision function call is forbidden")
            arguments = [visit(argument) for argument in node.args]
            value = float(_ARITHMETIC_FUNCTIONS[node.func.id](*arguments))
        else:
            raise ValueError(
                f"decision function syntax is forbidden: {type(node).__name__}"
            )
        if not math.isfinite(value):
            raise ValueError("decision function produced a non-finite value")
        return value

    return visit(tree)


def _write_json_projection(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


class StageWorkspaceV50:
    """Trusted stage protocol backed by ``GraphLoopStoreV40``."""

    def __init__(
        self,
        root: Path,
        spec: TaskWorkspaceSpecV50,
        graph: GraphLoopStoreV40,
        *,
        authority_key: bytes,
        authority_key_id: str,
    ) -> None:
        if len(authority_key) < 32:
            raise ValueError("authority_key must contain at least 32 bytes")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", authority_key_id):
            raise ValueError("authority_key_id is not a safe identifier")
        self.root = root.resolve()
        self.spec = spec
        self.graph = graph
        self._authority_key = bytes(authority_key)
        self.authority_key_id = authority_key_id
        self._artifact_cache_revision: tuple[Any, ...] | None = None
        self._artifact_index_cache: dict[
            str, list[tuple[ArtifactRef, Any]]
        ] = {}
        self._committed_hash_cache: set[str] = set()
        self._payload_by_hash_cache: dict[str, Any] = {}
        self._typed_artifact_cache: dict[
            tuple[str, type[Any]], list[tuple[ArtifactRef, Any]]
        ] = {}
        self._certificate_verification_cache: dict[str, bool] = {}

    @classmethod
    def create(
        cls,
        workspace_root: str | Path,
        spec: TaskWorkspaceSpecV50,
        *,
        authority_key: bytes,
        authority_key_id: str,
    ) -> "StageWorkspaceV50":
        root = Path(workspace_root).resolve()
        if not root.is_dir():
            raise StageWorkspaceError("task workspace must already be scaffolded")
        spec.assert_sealed()
        if spec.graph_id != f"v5-{spec.workspace_id}":
            raise ValueError("graph_id must be 'v5-' followed by workspace_id")
        control_root = root / ".fma"
        control_root.mkdir(parents=True, exist_ok=True)
        graph_root = control_root / "graph"
        if graph_root.exists() and any(graph_root.iterdir()):
            raise StageWorkspaceError("workspace already contains a graph")
        contract = GraphLoopContractV40.seal(
            graph_id=spec.graph_id,
            layer="modeling",
            evaluator_epoch=spec.evaluator_epoch,
            objective=spec.objective,
            max_nodes=spec.max_nodes,
            max_outcomes=spec.max_outcomes,
            max_failures=spec.max_failures,
            max_promotions=1,
            allowed_actions=spec.permitted_actions,
            forbidden_actions=spec.forbidden_actions,
        )
        graph = GraphLoopStoreV40(graph_root, contract)
        instance = cls(
            root,
            spec,
            graph,
            authority_key=authority_key,
            authority_key_id=authority_key_id,
        )
        graph.put_output("task_workspace_spec_v50", spec)
        instance._commit_authority_genesis()
        instance._add_stage_chain(start_stage="S0", attempt=1, superseded={})
        instance._write_spec_projection()
        instance._write_graph_projection()
        if not instance.verify():
            raise RuntimeError("new V5 stage workspace failed self-verification")
        return instance

    @classmethod
    def open_existing(
        cls,
        workspace_root: str | Path,
        *,
        authority_key: bytes,
        authority_key_id: str,
    ) -> "StageWorkspaceV50":
        root = Path(workspace_root).resolve()
        projection_path = root / ".fma" / "workspace_spec.json"
        if not projection_path.is_file():
            raise FileNotFoundError("workspace_spec.json is missing")
        projection = TaskWorkspaceSpecV50.model_validate_json(
            projection_path.read_text(encoding="utf-8")
        )
        projection.assert_sealed()
        graph = GraphLoopStoreV40.open_existing(
            root / ".fma" / "graph" / projection.graph_id
        )
        authoritative = cls._artifacts_of_kind_from_graph(
            graph, "task_workspace_spec_v50", TaskWorkspaceSpecV50
        )
        if len(authoritative) != 1 or authoritative[0][1] != projection:
            raise StageWorkspaceError(
                "workspace spec projection differs from authoritative graph artifact"
            )
        instance = cls(
            root,
            projection,
            graph,
            authority_key=authority_key,
            authority_key_id=authority_key_id,
        )
        if not instance.verify():
            raise StageWorkspaceError("existing V5 workspace failed verification")
        return instance

    def _write_spec_projection(self) -> None:
        _write_json_projection(
            self.root / ".fma" / "workspace_spec.json",
            self.spec.model_dump(mode="json"),
        )

    def _mac(self, kind: str, unsigned_hash: str) -> str:
        return hmac.new(
            self._authority_key,
            f"fma-v5:{kind}:{unsigned_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _key_commitment(self) -> str:
        return hashlib.sha256(
            b"fma-v5-authority-key-commitment\0" + self._authority_key
        ).hexdigest()

    def _commit_authority_genesis(self) -> AuthorityGenesisV50:
        if self._artifacts_of_kind("authority_genesis_v50"):
            raise StageWorkspaceError("authority genesis already exists")
        unsigned = AuthorityGenesisV50(
            workspace_spec_hash=self.spec.spec_hash,
            graph_id=self.spec.graph_id,
            authority_key_id=self.authority_key_id,
            authority_key_commitment=self._key_commitment(),
            created_at=_utc_now(),
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "authority_genesis_v50", unsigned.unsigned_hash()
        )
        payload["genesis_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "genesis_hash"}
        )
        genesis = AuthorityGenesisV50.model_validate(payload)
        self.graph.put_output("authority_genesis_v50", genesis)
        return genesis

    def _verify_authority_genesis(self) -> bool:
        try:
            artifacts = self._artifacts_of_kind(
                "authority_genesis_v50", AuthorityGenesisV50
            )
            if len(artifacts) != 1:
                return False
            genesis = artifacts[0][1]
            return (
                genesis.workspace_spec_hash == self.spec.spec_hash
                and genesis.graph_id == self.spec.graph_id
                and genesis.authority_key_id == self.authority_key_id
                and genesis.authority_key_commitment == self._key_commitment()
                and self._verify_mac(
                    "authority_genesis_v50",
                    genesis.unsigned_hash(),
                    genesis.authority_auth_tag,
                )
            )
        except (OSError, ValueError, RuntimeError, StageWorkspaceError):
            return False

    def _verify_mac(self, kind: str, unsigned_hash: str, tag: str | None) -> bool:
        if tag is None:
            return False
        return hmac.compare_digest(self._mac(kind, unsigned_hash), tag)

    @staticmethod
    def _artifacts_of_kind_from_graph(
        graph: GraphLoopStoreV40,
        kind: str,
        model_type: type[Any] | None = None,
    ) -> list[tuple[ArtifactRef, Any]]:
        events = graph._read_events(graph.store)
        committed = graph._committed_artifacts(graph.store, events)
        found: list[tuple[ArtifactRef, Any]] = []
        for path in sorted(graph.store.artifact_directory.glob("*.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StageWorkspaceError(f"invalid graph artifact: {path}") from exc
            if envelope.get("kind") != kind:
                continue
            digest = sha256_value(envelope)
            if digest != path.stem:
                raise StageWorkspaceError(f"content-addressed artifact changed: {path}")
            ref = ArtifactRef(
                kind=kind,
                sha256=digest,
                relative_path=f"artifacts/{path.name}",
            )
            if committed.get((kind, digest)) != ref:
                # A content file written before a crashed/interrupted commit is
                # not authority.  Only event-bound artifacts participate in
                # replay; leaving the orphan is safer than deleting evidence.
                continue
            payload: Any = envelope["payload"]
            if model_type is not None:
                payload = model_type.model_validate(payload)
            found.append((ref, payload))
        return found

    def _artifacts_of_kind(
        self, kind: str, model_type: type[Any] | None = None
    ) -> list[tuple[ArtifactRef, Any]]:
        self._refresh_artifact_index()
        raw = self._artifact_index_cache.get(kind, [])
        if model_type is None:
            return list(raw)
        cache_key = (kind, model_type)
        cached = self._typed_artifact_cache.get(cache_key)
        if cached is None:
            cached = [
                (reference, model_type.model_validate(payload))
                for reference, payload in raw
            ]
            self._typed_artifact_cache[cache_key] = cached
        return list(cached)

    def _refresh_artifact_index(self) -> None:
        stat = self.graph.store.event_path.stat()
        artifact_stats = tuple(
            (path.name, item.st_size, item.st_mtime_ns)
            for path in sorted(
                self.graph.store.artifact_directory.glob("*.json")
            )
            for item in (path.stat(),)
        )
        revision = (stat.st_size, stat.st_mtime_ns, artifact_stats)
        if revision == self._artifact_cache_revision:
            return
        events = self.graph._read_events(self.graph.store)
        committed = self.graph._committed_artifacts(self.graph.store, events)
        by_kind: dict[str, list[tuple[ArtifactRef, Any]]] = {}
        payload_by_hash: dict[str, Any] = {}
        for path in sorted(self.graph.store.artifact_directory.glob("*.json")):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StageWorkspaceError(
                    f"invalid graph artifact: {path}"
                ) from exc
            digest = sha256_value(envelope)
            if digest != path.stem:
                raise StageWorkspaceError(
                    f"content-addressed artifact changed: {path}"
                )
            kind = envelope.get("kind")
            if not isinstance(kind, str):
                raise StageWorkspaceError(f"artifact kind is invalid: {path}")
            reference = ArtifactRef(
                kind=kind,
                sha256=digest,
                relative_path=f"artifacts/{path.name}",
            )
            if committed.get((kind, digest)) != reference:
                continue
            payload = envelope["payload"]
            by_kind.setdefault(kind, []).append((reference, payload))
            if digest in payload_by_hash:
                raise StageWorkspaceError(
                    "artifact hash is not unique in the committed store"
                )
            payload_by_hash[digest] = payload
        self._artifact_cache_revision = revision
        self._artifact_index_cache = by_kind
        self._committed_hash_cache = set(payload_by_hash)
        self._payload_by_hash_cache = payload_by_hash
        self._typed_artifact_cache = {}
        self._certificate_verification_cache = {}

    def _committed_artifact_hashes(self) -> set[str]:
        self._refresh_artifact_index()
        return set(self._committed_hash_cache)

    def _artifact_payload_by_hash(self, artifact_hash: str) -> Any:
        self._refresh_artifact_index()
        if artifact_hash not in self._payload_by_hash_cache:
            raise StageWorkspaceError(
                "artifact hash does not identify one committed artifact"
            )
        return self._payload_by_hash_cache[artifact_hash]

    def commit_evidence(self, kind: str, payload: object) -> ArtifactRef:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", kind):
            raise ValueError("evidence kind must be a safe lowercase identifier")
        if kind in {
            "adapter_execution_receipt_v50",
            "gate_certificate_v50",
            "check_result_v50",
            "independent_review_receipt_v50",
        }:
            raise PermissionError("reserved authority artifacts require typed APIs")
        return self.graph.put_output(kind, payload)

    def _node_map(self) -> dict[tuple[StageId, int, str], Any]:
        result: dict[tuple[StageId, int, str], Any] = {}
        for node in self.graph.project_state().nodes:
            match = _NODE_ID.fullmatch(node.node_id)
            if not match:
                continue
            stage = f"S{match.group('stage')}"
            assert stage in STAGES
            key = (stage, int(match.group("attempt")), match.group("kind"))
            if key in result:
                raise StageWorkspaceError("duplicate stage node binding")
            result[key] = node
        return result

    def _latest_attempt(self, stage: StageId) -> int:
        attempts = [
            attempt
            for item_stage, attempt, kind in self._node_map()
            if item_stage == stage and kind == "work"
        ]
        if not attempts:
            raise StageWorkspaceError(f"no graph nodes exist for {stage}")
        return max(attempts)

    def _binding(self, stage: StageId, kind: Literal["work", "gate"]) -> Any:
        attempt = self._latest_attempt(stage)
        try:
            return self._node_map()[(stage, attempt, kind)]
        except KeyError as exc:
            raise StageWorkspaceError(f"missing {stage} {kind} node") from exc

    def _add_stage_chain(
        self,
        *,
        start_stage: StageId,
        attempt: int,
        superseded: dict[tuple[StageId, str], str],
    ) -> None:
        state = self.graph.project_state()
        previous_gate_hash: str | None = None
        start_index = _STAGE_INDEX[start_stage]
        if start_index > 0:
            previous_stage = STAGES[start_index - 1]
            previous_gate = self._binding(previous_stage, "gate")
            previous_gate_hash = str(previous_gate.node_hash)

        new_nodes: dict[tuple[StageId, str], Any] = {}
        for stage in STAGES[start_index:]:
            work_artifact_hash = sha256_value(
                {
                    "workspace_spec_hash": self.spec.spec_hash,
                    "stage": stage,
                    "attempt": attempt,
                    "role": "work",
                }
            )
            work = GraphNodeV40.seal(
                node_id=f"s{_STAGE_INDEX[stage]}-a{attempt}-work",
                layer="modeling",
                node_kind=_WORK_NODE_KINDS[stage],
                executor=_WORK_NODE_EXECUTORS[stage],
                created_by="harness",
                artifact_hash=work_artifact_hash,
                purpose=f"{stage} attempt {attempt} stage work",
            )
            gate = GraphNodeV40.seal(
                node_id=f"s{_STAGE_INDEX[stage]}-a{attempt}-gate",
                layer="modeling",
                node_kind="evaluation",
                executor="verifier",
                created_by="harness",
                artifact_hash=sha256_value(
                    {
                        "workspace_spec_hash": self.spec.spec_hash,
                        "stage": stage,
                        "attempt": attempt,
                        "policy_hash": POLICIES[stage].policy_hash,
                    }
                ),
                purpose=f"{stage} attempt {attempt} workflow gate",
            )
            self.graph.add_node(work)
            self.graph.add_node(gate)
            new_nodes[(stage, "work")] = work
            new_nodes[(stage, "gate")] = gate

        edge_counter = 0

        def add_edge(source: str, target: str, relation: str, rationale: str) -> None:
            nonlocal edge_counter
            edge_counter += 1
            self.graph.add_edge(
                GraphEdgeV40.seal(
                    edge_id=(
                        f"v5-a{attempt}-s{start_index}-e{edge_counter}"
                    ),
                    layer="modeling",
                    source_node_hash=source,
                    target_node_hash=target,
                    relation=relation,
                    rationale=rationale,
                )
            )

        for stage in STAGES[start_index:]:
            work = new_nodes[(stage, "work")]
            gate = new_nodes[(stage, "gate")]
            if previous_gate_hash is not None:
                add_edge(
                    previous_gate_hash,
                    str(work.node_hash),
                    "requires_success",
                    f"{stage} requires the preceding stage gate",
                )
            add_edge(
                str(work.node_hash),
                str(gate.node_hash),
                "evaluated_by",
                f"{stage} work is independently evaluated",
            )
            for kind in ("work", "gate"):
                old_hash = superseded.get((stage, kind))
                if old_hash:
                    add_edge(
                        old_hash,
                        str(new_nodes[(stage, kind)].node_hash),
                        "supersedes",
                        f"{stage} attempt {attempt} preserves retry lineage",
                    )
            previous_gate_hash = str(gate.node_hash)
        self.graph.project_state()

    def _safe_file(self, relative_path: str, *, must_exist: bool = True) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute() or ".." in raw.parts or not raw.parts:
            raise StageWorkspaceError(f"unsafe workspace path: {relative_path}")
        candidate = self.root.joinpath(*raw.parts)
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise StageWorkspaceError(
                f"workspace path escapes root: {relative_path}"
            ) from exc
        current = self.root
        for part in raw.parts:
            current = current / part
            if current.is_symlink():
                raise StageWorkspaceError(
                    f"workspace evidence path may not use symlinks: {relative_path}"
                )
        if must_exist and not candidate.is_file():
            raise StageWorkspaceError(f"required file is missing: {relative_path}")
        return candidate

    def _manifest_paths(
        self, stage: StageId, extra_paths: Iterable[str] = ()
    ) -> list[str]:
        paths = set(POLICIES[stage].required_paths)
        paths.update(extra_paths)
        if stage == "S2":
            for directory in ("data/raw", "data/processed"):
                root = self.root / directory
                if root.exists():
                    for path in root.rglob("*"):
                        if path.is_file():
                            paths.add(path.relative_to(self.root).as_posix())
            ledger_path = self.root / "data" / "ledger.json"
            if ledger_path.is_file():
                try:
                    ledger = DataLedgerV50.model_validate_json(
                        ledger_path.read_text(encoding="utf-8")
                    )
                    for entry in ledger.entries:
                        paths.add(entry.transform_script_relative_path)
                        if entry.raw_relative_path:
                            paths.add(entry.raw_relative_path)
                except (OSError, ValueError):
                    # The mechanical checker will emit the precise schema error.
                    pass
        elif stage == "S3":
            source_root = self.root / "src"
            if source_root.exists():
                for path in source_root.rglob("*"):
                    if path.is_file():
                        paths.add(path.relative_to(self.root).as_posix())
            result_index_path = self.root / "results" / "index.json"
            if result_index_path.is_file():
                try:
                    result_index = ResultIndexV50.model_validate_json(
                        result_index_path.read_text(encoding="utf-8")
                    )
                    paths.update(
                        record.relative_path for record in result_index.records
                    )
                except (OSError, ValueError):
                    pass
            code_manifest_path = self.root / "results" / "code_manifest.json"
            if code_manifest_path.is_file():
                try:
                    code_manifest = CodeManifestV50.model_validate_json(
                        code_manifest_path.read_text(encoding="utf-8")
                    )
                    paths.update(
                        {
                            code_manifest.environment_ref,
                            code_manifest.replay_receipt_ref,
                            code_manifest.fermi_estimate_ref,
                            *code_manifest.toy_oracle_refs,
                        }
                    )
                except (OSError, ValueError):
                    pass
        elif stage == "S6":
            paper_root = self.root / "paper"
            if paper_root.exists():
                for path in paper_root.rglob("*"):
                    if path.is_file():
                        paths.add(path.relative_to(self.root).as_posix())
        return sorted(paths)

    def _capture_manifest(
        self,
        stage: StageId,
        attempt: int,
        paths: Iterable[str],
    ) -> StageArtifactManifestV50:
        predecessor: str | None = None
        if stage != "S0":
            predecessor = self.current_gate(STAGES[_STAGE_INDEX[stage] - 1])
            if predecessor is None:
                raise StageWorkspaceError(
                    f"{stage} is locked: predecessor gate is missing or stale"
                )
        bindings: list[FileBindingV50] = []
        for relative_path in sorted(set(paths)):
            bindings.append(self._snapshot_file_binding(relative_path))
        return StageArtifactManifestV50.seal(
            workspace_spec_hash=self.spec.spec_hash,
            stage=stage,
            attempt=attempt,
            predecessor_gate_hash=predecessor,
            files=bindings,
        )

    def _snapshot_file_binding(self, relative_path: str) -> FileBindingV50:
        path = self._safe_file(relative_path)
        payload = path.read_bytes()
        if len(payload) > 32 * 1024 * 1024:
            raise StageWorkspaceError(
                f"stage file exceeds the 32 MiB snapshot limit: {relative_path}; "
                "use a chunked/external content-addressed data adapter"
            )
        snapshot_ref = self.graph.put_output(
            "workspace_file_snapshot_v50",
            {
                "schema_version": "5.0",
                "relative_path": relative_path,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
                "content_base64": base64.b64encode(payload).decode("ascii"),
            },
        )
        return FileBindingV50(
            relative_path=relative_path,
            sha256=_sha256_bytes(payload),
            size_bytes=len(payload),
            snapshot_artifact_hash=snapshot_ref.sha256,
        )

    def freeze_raw_inputs(
        self,
        *,
        actor: Literal["harness", "human"] = "harness",
    ) -> RawDataBaselineV50:
        """Freeze user/acquisition-owned raw bytes before S2 model work."""

        if actor not in {"harness", "human"}:
            raise PermissionError("model cannot freeze its own raw-data baseline")
        s1_gate_hash = self.current_gate("S1")
        if s1_gate_hash is None:
            raise StageWorkspaceError(
                "raw inputs may be frozen only after the current S1 gate"
            )
        attempt = self._latest_attempt("S2")
        if any(
            baseline.s2_attempt == attempt
            for _, baseline in self._artifacts_of_kind(
                "raw_data_baseline_v50", RawDataBaselineV50
            )
            if self.verify_raw_baseline(baseline)
        ):
            raise StageWorkspaceError(
                "raw input baseline is immutable for this S2 attempt"
            )
        raw_root = self.root / "data" / "raw"
        raw_paths = (
            sorted(
                path.relative_to(self.root).as_posix()
                for path in raw_root.rglob("*")
                if path.is_file()
            )
            if raw_root.exists()
            else []
        )
        bindings = [
            self._snapshot_file_binding(relative_path)
            for relative_path in raw_paths
        ]
        unsigned = RawDataBaselineV50(
            workspace_spec_hash=self.spec.spec_hash,
            s1_gate_hash=s1_gate_hash,
            s2_attempt=attempt,
            raw_tree_hash=_tree_hash(raw_root),
            files=bindings,
            frozen_at=_utc_now(),
            authority_key_id=self.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "raw_data_baseline_v50", unsigned.unsigned_hash()
        )
        payload["baseline_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "baseline_hash"}
        )
        baseline = RawDataBaselineV50.model_validate(payload)
        self.graph.put_output("raw_data_baseline_v50", baseline)
        _write_json_projection(
            self.root
            / ".fma"
            / f"raw_baseline_s2_a{attempt}.json",
            {
                "schema_version": "5.0-projection",
                "baseline_hash": baseline.baseline_hash,
                "raw_tree_hash": baseline.raw_tree_hash,
                "s1_gate_hash": baseline.s1_gate_hash,
                "projection_is_not_authority": True,
            },
        )
        return baseline

    def verify_raw_baseline(self, baseline: RawDataBaselineV50) -> bool:
        try:
            RawDataBaselineV50.model_validate(
                baseline.model_dump(mode="json")
            )
            snapshots_ok = (
                self._verify_file_bindings(baseline.files)
                if baseline.files
                else baseline.raw_tree_hash == sha256_value({})
            )
        except (OSError, ValueError, RuntimeError, StageWorkspaceError):
            return False
        return (
            snapshots_ok
            and baseline.workspace_spec_hash == self.spec.spec_hash
            and baseline.authority_key_id == self.authority_key_id
            and self._verify_mac(
                "raw_data_baseline_v50",
                baseline.unsigned_hash(),
                baseline.authority_auth_tag,
            )
        )

    def _raw_baseline_for_current_s2(self) -> RawDataBaselineV50 | None:
        attempt = self._latest_attempt("S2")
        s1_gate_hash = self.current_gate("S1")
        matches = [
            baseline
            for _, baseline in self._artifacts_of_kind(
                "raw_data_baseline_v50", RawDataBaselineV50
            )
            if baseline.s2_attempt == attempt
            and baseline.s1_gate_hash == s1_gate_hash
            and self.verify_raw_baseline(baseline)
        ]
        return matches[0] if len(matches) == 1 else None

    def submit_stage(
        self,
        stage: StageId,
        *,
        actor: Literal["model", "harness"],
        extra_paths: Iterable[str] = (),
    ) -> StageArtifactManifestV50:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        work = self._binding(stage, "work")
        expected_actor = _WORK_NODE_EXECUTORS[stage]
        if actor != expected_actor:
            raise PermissionError(
                f"{stage} work belongs to {expected_actor}, not {actor}"
            )
        if stage == "S2":
            baseline = self._raw_baseline_for_current_s2()
            if baseline is None:
                raise StageWorkspaceError(
                    "S2 requires one harness-frozen raw input baseline"
                )
            if baseline.raw_tree_hash != _tree_hash(self.root / "data" / "raw"):
                raise StageWorkspaceError(
                    "data/raw changed after the harness baseline was frozen"
                )
        state = self.graph.project_state()
        if work.node_hash not in state.snapshot.frontier_node_hashes:
            raise StageWorkspaceError(f"{stage} work is not on the graph frontier")
        attempt = self._latest_attempt(stage)
        manifest = self._capture_manifest(
            stage, attempt, self._manifest_paths(stage, extra_paths)
        )
        manifest_ref = self.graph.put_output("stage_artifact_manifest_v50", manifest)
        self.graph.record_outcome(
            str(work.node_hash),
            actor=actor,
            status="succeeded",
            output_artifacts=[manifest_ref],
            summary=f"{stage} artifact manifest committed; scientific validity not claimed",
            outcome_id=f"{work.node_id}-outcome",
        )
        self._write_graph_projection()
        return manifest

    def _manifest_for_stage(self, stage: StageId) -> StageArtifactManifestV50:
        work = self._binding(stage, "work")
        state = self.graph.project_state()
        outcomes = [
            outcome
            for outcome in state.outcomes
            if outcome.node_hash == work.node_hash
            and state.snapshot.node_statuses.get(work.node_hash) == "succeeded"
        ]
        if len(outcomes) != 1:
            raise StageWorkspaceError(f"{stage} has no unique successful manifest")
        manifest_refs = [
            ref
            for ref in outcomes[0].output_artifacts
            if ref.kind == "stage_artifact_manifest_v50"
        ]
        if len(manifest_refs) != 1:
            raise StageWorkspaceError(f"{stage} work outcome lacks one manifest")
        manifest = StageArtifactManifestV50.model_validate(
            self.graph.store.load_artifact(manifest_refs[0])
        )
        if manifest.stage != stage:
            raise StageWorkspaceError("stage manifest is bound to another stage")
        return manifest

    def _manifest_is_current(self, manifest: StageArtifactManifestV50) -> bool:
        if manifest.workspace_spec_hash != self.spec.spec_hash:
            return False
        if manifest.manifest_hash != manifest.content_hash():
            return False
        if not self._verify_manifest_snapshots(manifest):
            return False
        recorded_paths = {item.relative_path for item in manifest.files}
        current_required_and_dynamic = set(self._manifest_paths(manifest.stage))
        if not current_required_and_dynamic.issubset(recorded_paths):
            return False
        for binding in manifest.files:
            try:
                path = self._safe_file(binding.relative_path)
            except StageWorkspaceError:
                return False
            payload = path.read_bytes()
            if len(payload) != binding.size_bytes:
                return False
            if _sha256_bytes(payload) != binding.sha256:
                return False
        if manifest.stage != "S0":
            previous = STAGES[_STAGE_INDEX[manifest.stage] - 1]
            if self.current_gate(previous) != manifest.predecessor_gate_hash:
                return False
        return True

    def _verify_manifest_snapshots(
        self, manifest: StageArtifactManifestV50
    ) -> bool:
        return self._verify_file_bindings(manifest.files)

    def _verify_file_bindings(
        self, bindings: Iterable[FileBindingV50]
    ) -> bool:
        try:
            for binding in bindings:
                payload = self._artifact_payload_by_hash(
                    binding.snapshot_artifact_hash
                )
                if not isinstance(payload, dict):
                    return False
                if (
                    payload.get("schema_version") != "5.0"
                    or payload.get("relative_path") != binding.relative_path
                    or payload.get("sha256") != binding.sha256
                    or payload.get("size_bytes") != binding.size_bytes
                ):
                    return False
                encoded = payload.get("content_base64")
                if not isinstance(encoded, str):
                    return False
                decoded = base64.b64decode(encoded, validate=True)
                if (
                    len(decoded) != binding.size_bytes
                    or _sha256_bytes(decoded) != binding.sha256
                ):
                    return False
        except (OSError, ValueError, TypeError, StageWorkspaceError):
            return False
        return True

    def _manifest_file_bytes(
        self, manifest: StageArtifactManifestV50, relative_path: str
    ) -> bytes:
        binding = next(
            (
                item
                for item in manifest.files
                if item.relative_path == relative_path
            ),
            None,
        )
        if binding is None:
            raise StageWorkspaceError(
                f"manifest does not contain {relative_path}"
            )
        payload = self._artifact_payload_by_hash(binding.snapshot_artifact_hash)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("content_base64"), str
        ):
            raise StageWorkspaceError("manifest file snapshot is malformed")
        decoded = base64.b64decode(payload["content_base64"], validate=True)
        if (
            len(decoded) != binding.size_bytes
            or _sha256_bytes(decoded) != binding.sha256
        ):
            raise StageWorkspaceError("manifest file snapshot changed")
        return decoded

    def issue_check(
        self,
        *,
        stage: StageId,
        check_id: str,
        level: str,
        evidence_class: str,
        applicability: str,
        status: str,
        reason_code: str,
        adapter_id: str,
        adapter_version: str,
        adapter_code_hash: str,
        evidence_refs: list[str],
        adapter_execution_receipt_hash: str | None = None,
        subject_hashes: list[str] | None = None,
        thresholds: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        scope: str = "development",
        executed_by: Literal["harness", "verifier"] = "verifier",
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> CheckResultV50:
        if executed_by not in {"harness", "verifier"}:
            raise PermissionError("model cannot issue checker evidence")
        manifest = self._manifest_for_stage(stage)
        committed_hashes = self._committed_artifact_hashes()
        if any(item not in committed_hashes for item in evidence_refs):
            raise StageWorkspaceError(
                "check evidence_refs must be committed graph artifacts"
            )
        receipt: AdapterExecutionReceiptV50 | None = None
        if level in {"L0", "L1", "L2", "L3", "L4"}:
            if adapter_execution_receipt_hash is None:
                raise StageWorkspaceError(
                    "L0-L4 checks require an authenticated adapter execution receipt"
                )
            matches = [
                item
                for _, item in self._artifacts_of_kind(
                    "adapter_execution_receipt_v50",
                    AdapterExecutionReceiptV50,
                )
                if item.receipt_hash == adapter_execution_receipt_hash
            ]
            if len(matches) != 1 or not self.verify_adapter_execution(matches[0]):
                raise StageWorkspaceError(
                    "adapter execution receipt is missing or invalid"
                )
            receipt = matches[0]
            expected_receipt_fields = (
                receipt.stage == stage,
                receipt.check_id == check_id,
                receipt.level == level,
                receipt.input_manifest_hash == manifest.manifest_hash,
                receipt.protocol_hash == POLICIES[stage].policy_hash,
                receipt.adapter_id == adapter_id,
                receipt.adapter_version == adapter_version,
                receipt.adapter_code_hash == adapter_code_hash,
                receipt.applicability == applicability,
                receipt.status == status,
                receipt.evidence_refs == sorted(set(evidence_refs)),
            )
            if not all(expected_receipt_fields):
                raise StageWorkspaceError(
                    "check result does not match its adapter execution receipt"
                )
            if level == "L0" and status == "PASS":
                try:
                    code_manifest = CodeManifestV50.model_validate_json(
                        self._manifest_file_bytes(
                            manifest, "results/code_manifest.json"
                        )
                    )
                except ValueError as exc:
                    raise StageWorkspaceError(
                        "L0 receipt cannot load the frozen code manifest"
                    ) from exc
                bound_replay = False
                for reference in evidence_refs:
                    evidence_payload = self._artifact_payload_by_hash(reference)
                    if (
                        isinstance(evidence_payload, dict)
                        and isinstance(evidence_payload.get("payload"), dict)
                        and evidence_payload["payload"].get(
                            "computation_artifact_sha256"
                        )
                        == code_manifest.replay_receipt_hash
                    ):
                        bound_replay = True
                        break
                if not bound_replay:
                    raise StageWorkspaceError(
                        "L0 PASS must bind the frozen replay receipt hash"
                    )
            started_at = receipt.started_at
            finished_at = receipt.finished_at
        elif adapter_execution_receipt_hash is not None:
            raise StageWorkspaceError(
                "L5 checks cannot cite a scientific adapter execution receipt"
            )
        now = _utc_now()
        unsigned = CheckResultV50(
            check_id=check_id,
            stage=stage,
            level=level,
            evidence_class=evidence_class,
            applicability=applicability,
            status=status,
            reason_code=reason_code,
            subject_hashes=sorted(set(subject_hashes or [])),
            input_manifest_hash=manifest.manifest_hash,
            protocol_hash=POLICIES[stage].policy_hash,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            adapter_code_hash=adapter_code_hash,
            adapter_execution_receipt_hash=adapter_execution_receipt_hash,
            thresholds=thresholds or {},
            metrics=metrics or {},
            evidence_refs=sorted(set(evidence_refs)),
            scope=scope,
            executed_by=executed_by,
            started_at=started_at or now,
            finished_at=finished_at or now,
            authority_key_id=self.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "check_result_v50", unsigned.unsigned_hash()
        )
        payload["result_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "result_hash"}
        )
        result = CheckResultV50.model_validate(payload)
        self.graph.put_output("check_result_v50", result)
        return result

    def _record_adapter_execution(
        self,
        *,
        stage: StageId,
        check_id: str,
        level: str,
        applicability: str,
        status: str,
        execution_mode: str,
        adapter_invoked: bool,
        scientific_computation_performed: bool,
        adapter_id: str,
        adapter_version: str,
        adapter_code_hash: str,
        evidence_refs: list[str],
        started_at: datetime,
        finished_at: datetime,
    ) -> AdapterExecutionReceiptV50:
        """Harness-only primitive used by the typed adapter registry."""

        manifest = self._manifest_for_stage(stage)
        committed_hashes = self._committed_artifact_hashes()
        if any(item not in committed_hashes for item in evidence_refs):
            raise StageWorkspaceError(
                "adapter evidence_refs must be committed graph artifacts"
            )
        unsigned = AdapterExecutionReceiptV50(
            execution_id=(
                f"{stage.lower()}-{check_id}-"
                f"{sha256_value({'manifest': manifest.manifest_hash, 'start': started_at.isoformat()})[:12]}"
            ),
            check_id=check_id,
            stage=stage,
            level=level,
            input_manifest_hash=manifest.manifest_hash,
            protocol_hash=POLICIES[stage].policy_hash,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            adapter_code_hash=adapter_code_hash,
            applicability=applicability,
            status=status,
            execution_mode=execution_mode,
            adapter_invoked=adapter_invoked,
            scientific_computation_performed=scientific_computation_performed,
            evidence_refs=sorted(set(evidence_refs)),
            started_at=started_at,
            finished_at=finished_at,
            authority_key_id=self.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "adapter_execution_receipt_v50", unsigned.unsigned_hash()
        )
        payload["receipt_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
        receipt = AdapterExecutionReceiptV50.model_validate(payload)
        self.graph.put_output("adapter_execution_receipt_v50", receipt)
        return receipt

    def verify_adapter_execution(
        self, receipt: AdapterExecutionReceiptV50
    ) -> bool:
        try:
            AdapterExecutionReceiptV50.model_validate(
                receipt.model_dump(mode="json")
            )
            if any(
                reference not in self._committed_artifact_hashes()
                for reference in receipt.evidence_refs
            ):
                return False
            manifests = [
                item
                for _, item in self._artifacts_of_kind(
                    "stage_artifact_manifest_v50",
                    StageArtifactManifestV50,
                )
                if item.stage == receipt.stage
                and item.manifest_hash == receipt.input_manifest_hash
            ]
            if len(manifests) != 1:
                return False
            manifest = manifests[0]
        except (OSError, ValueError, RuntimeError, StageWorkspaceError):
            return False
        return (
            receipt.input_manifest_hash == manifest.manifest_hash
            and manifest.content_hash() == manifest.manifest_hash
            and self._verify_manifest_snapshots(manifest)
            and receipt.protocol_hash == POLICIES[receipt.stage].policy_hash
            and receipt.authority_key_id == self.authority_key_id
            and self._verify_mac(
                "adapter_execution_receipt_v50",
                receipt.unsigned_hash(),
                receipt.authority_auth_tag,
            )
        )

    def verify_check(self, result: CheckResultV50) -> bool:
        try:
            CheckResultV50.model_validate(result.model_dump(mode="json"))
            committed = self._committed_artifact_hashes()
            if any(reference not in committed for reference in result.evidence_refs):
                return False
            if result.level in {"L0", "L1", "L2", "L3", "L4"}:
                matches = [
                    item
                    for _, item in self._artifacts_of_kind(
                        "adapter_execution_receipt_v50",
                        AdapterExecutionReceiptV50,
                    )
                    if item.receipt_hash
                    == result.adapter_execution_receipt_hash
                ]
                if len(matches) != 1:
                    return False
                receipt = matches[0]
                if (
                    not self.verify_adapter_execution(receipt)
                    or receipt.stage != result.stage
                    or receipt.check_id != result.check_id
                    or receipt.level != result.level
                    or receipt.input_manifest_hash != result.input_manifest_hash
                    or receipt.protocol_hash != result.protocol_hash
                    or receipt.adapter_id != result.adapter_id
                    or receipt.adapter_version != result.adapter_version
                    or receipt.adapter_code_hash != result.adapter_code_hash
                    or receipt.applicability != result.applicability
                    or receipt.status != result.status
                    or receipt.evidence_refs != result.evidence_refs
                    or receipt.started_at != result.started_at
                    or receipt.finished_at != result.finished_at
                ):
                    return False
        except (OSError, ValueError, RuntimeError, StageWorkspaceError):
            return False
        return (
            result.authority_key_id == self.authority_key_id
            and self._verify_mac(
                "check_result_v50",
                result.unsigned_hash(),
                result.authority_auth_tag,
            )
        )

    def issue_review(
        self,
        *,
        stage: StageId,
        review_id: str,
        role: str,
        producer_run_id: str,
        reviewer_run_id: str,
        producer_context_id: str,
        reviewer_context_id: str,
        prompt_hash: str,
        output_schema_hash: str,
        allowed_input_hashes: list[str],
        transport_trace_hash: str,
        output_artifact_hash: str,
        verdict: str,
        finding_ids: list[str] | None = None,
        issued_by: Literal["verifier", "human"] = "verifier",
    ) -> IndependentReviewReceiptV50:
        if issued_by not in {"verifier", "human"}:
            raise PermissionError("model cannot issue review receipts")
        committed_hashes = self._committed_artifact_hashes()
        if transport_trace_hash not in committed_hashes:
            raise StageWorkspaceError("review transport trace is not committed")
        if output_artifact_hash not in committed_hashes:
            raise StageWorkspaceError("review output is not committed")
        manifest = self._manifest_for_stage(stage)
        current_checks = self._latest_checks(
            stage, str(manifest.manifest_hash)
        )
        expected_inputs = sorted(
            {item.sha256 for item in manifest.files}
            | {
                str(result.result_hash)
                for result in current_checks.values()
                if result.result_hash is not None
            }
        )
        if sorted(set(allowed_input_hashes)) != expected_inputs:
            raise StageWorkspaceError(
                "review allowed inputs must exactly match the stage manifest "
                "and current check receipts"
            )
        trace_payload = self._artifact_payload_by_hash(transport_trace_hash)
        if (
            not isinstance(trace_payload, dict)
            or trace_payload.get("producer_run_id") != producer_run_id
            or trace_payload.get("reviewer_run_id") != reviewer_run_id
            or trace_payload.get("producer_context_id") != producer_context_id
            or trace_payload.get("reviewer_context_id") != reviewer_context_id
            or trace_payload.get("context_isolation_attested") is not True
            or trace_payload.get("allowed_input_hashes") != expected_inputs
        ):
            raise StageWorkspaceError(
                "review transport trace does not bind the declared contexts/inputs"
            )
        normalized_findings = sorted(set(finding_ids or []))
        output_payload = self._artifact_payload_by_hash(output_artifact_hash)
        if (
            not isinstance(output_payload, dict)
            or output_payload.get("stage") != stage
            or output_payload.get("role") != role
            or output_payload.get("verdict") != verdict
            or sorted(output_payload.get("finding_ids", []))
            != normalized_findings
        ):
            raise StageWorkspaceError(
                "review output artifact does not bind verdict/findings"
            )
        unsigned = IndependentReviewReceiptV50(
            review_id=review_id,
            stage=stage,
            role=role,
            input_manifest_hash=manifest.manifest_hash,
            producer_run_id=producer_run_id,
            reviewer_run_id=reviewer_run_id,
            producer_context_id=producer_context_id,
            reviewer_context_id=reviewer_context_id,
            prompt_hash=prompt_hash,
            output_schema_hash=output_schema_hash,
            allowed_input_hashes=expected_inputs,
            context_isolation_attested=True,
            transport_trace_hash=transport_trace_hash,
            output_artifact_hash=output_artifact_hash,
            verdict=verdict,
            finding_ids=normalized_findings,
            issued_by=issued_by,
            issued_at=_utc_now(),
            authority_key_id=self.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "independent_review_receipt_v50", unsigned.unsigned_hash()
        )
        payload["receipt_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
        receipt = IndependentReviewReceiptV50.model_validate(payload)
        self.graph.put_output("independent_review_receipt_v50", receipt)
        return receipt

    def issue_role_execution(
        self,
        *,
        stage: StageId,
        execution_id: str,
        role: Literal["modeler", "literature_scout", "writer"],
        subject_id: str,
        input_authority_hash: str,
        run_id: str,
        context_id: str,
        provider: str,
        model: str,
        prompt_hash: str,
        output_schema_hash: str,
        transport_trace_hash: str,
        output_artifact_hash: str,
    ) -> RoleExecutionReceiptV50:
        """Authenticate a completed role call; the model cannot self-issue it."""

        committed_hashes = self._committed_artifact_hashes()
        if transport_trace_hash not in committed_hashes:
            raise StageWorkspaceError("role transport trace is not committed")
        if output_artifact_hash not in committed_hashes:
            raise StageWorkspaceError("role output is not committed")
        trace_payload = self._artifact_payload_by_hash(transport_trace_hash)
        if (
            not isinstance(trace_payload, dict)
            or trace_payload.get("role") != role
            or trace_payload.get("subject_id") != subject_id
            or trace_payload.get("input_authority_hash")
            != input_authority_hash
            or trace_payload.get("run_id") != run_id
            or trace_payload.get("context_id") != context_id
        ):
            raise StageWorkspaceError(
                "role transport trace does not bind the declared execution"
            )
        unsigned = RoleExecutionReceiptV50(
            execution_id=execution_id,
            stage=stage,
            role=role,
            subject_id=subject_id,
            input_authority_hash=input_authority_hash,
            run_id=run_id,
            context_id=context_id,
            provider=provider,
            model=model,
            prompt_hash=prompt_hash,
            output_schema_hash=output_schema_hash,
            transport_trace_hash=transport_trace_hash,
            output_artifact_hash=output_artifact_hash,
            issued_by="harness",
            issued_at=_utc_now(),
            authority_key_id=self.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "role_execution_receipt_v50", unsigned.unsigned_hash()
        )
        payload["receipt_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
        receipt = RoleExecutionReceiptV50.model_validate(payload)
        self.graph.put_output("role_execution_receipt_v50", receipt)
        return receipt

    def verify_role_execution(self, receipt: RoleExecutionReceiptV50) -> bool:
        try:
            RoleExecutionReceiptV50.model_validate(
                receipt.model_dump(mode="json")
            )
            committed = self._committed_artifact_hashes()
            if (
                receipt.transport_trace_hash not in committed
                or receipt.output_artifact_hash not in committed
            ):
                return False
            trace = self._artifact_payload_by_hash(
                receipt.transport_trace_hash
            )
            if (
                not isinstance(trace, dict)
                or trace.get("role") != receipt.role
                or trace.get("subject_id") != receipt.subject_id
                or trace.get("input_authority_hash")
                != receipt.input_authority_hash
                or trace.get("run_id") != receipt.run_id
                or trace.get("context_id") != receipt.context_id
            ):
                return False
        except (OSError, ValueError, RuntimeError, StageWorkspaceError):
            return False
        return (
            receipt.authority_key_id == self.authority_key_id
            and self._verify_mac(
                "role_execution_receipt_v50",
                receipt.unsigned_hash(),
                receipt.authority_auth_tag,
            )
        )

    def issue_prediction_seal(
        self,
        *,
        task_id: str,
        training_snapshot_hash: str,
        candidate_hash: str,
        prediction_artifact_hash: str,
        external_registration_hash: str,
        external_snapshot_hash: str,
        holdout_commitment_hash: str,
    ) -> PredictionSealV50:
        """Bind an external immutable registration after the current S4 gate."""

        s4_gate_hash = self.current_gate("S4")
        if s4_gate_hash is None:
            raise StageWorkspaceError(
                "prediction registration requires a current S4 gate"
            )
        existing = [
            seal
            for _, seal in self._artifacts_of_kind(
                "prediction_seal_v50", PredictionSealV50
            )
            if seal.task_id == task_id and self.verify_prediction_seal(seal)
        ]
        if existing:
            raise StageWorkspaceError(
                "prediction registration is immutable and unique per task"
            )
        unsigned = PredictionSealV50(
            workspace_spec_hash=self.spec.spec_hash,
            s4_gate_hash=s4_gate_hash,
            task_id=task_id,
            training_snapshot_hash=training_snapshot_hash,
            candidate_hash=candidate_hash,
            prediction_artifact_hash=prediction_artifact_hash,
            external_registration_hash=external_registration_hash,
            external_snapshot_hash=external_snapshot_hash,
            holdout_commitment_hash=holdout_commitment_hash,
            registered_at=_utc_now(),
            authority_key_id=self.authority_key_id,
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "prediction_seal_v50", unsigned.unsigned_hash()
        )
        payload["seal_hash"] = sha256_value(
            {key: value for key, value in payload.items() if key != "seal_hash"}
        )
        seal = PredictionSealV50.model_validate(payload)
        self.graph.put_output("prediction_seal_v50", seal)
        return seal

    def verify_prediction_seal(self, seal: PredictionSealV50) -> bool:
        try:
            PredictionSealV50.model_validate(seal.model_dump(mode="json"))
        except ValueError:
            return False
        return (
            seal.workspace_spec_hash == self.spec.spec_hash
            and seal.authority_key_id == self.authority_key_id
            and self._verify_mac(
                "prediction_seal_v50",
                seal.unsigned_hash(),
                seal.authority_auth_tag,
            )
        )

    def verify_review(self, receipt: IndependentReviewReceiptV50) -> bool:
        try:
            IndependentReviewReceiptV50.model_validate(
                receipt.model_dump(mode="json")
            )
        except ValueError:
            return False
        try:
            committed = self._committed_artifact_hashes()
            if (
                receipt.transport_trace_hash not in committed
                or receipt.output_artifact_hash not in committed
            ):
                return False
            output = self._artifact_payload_by_hash(receipt.output_artifact_hash)
            trace = self._artifact_payload_by_hash(receipt.transport_trace_hash)
            evidence_matches = (
                isinstance(output, dict)
                and output.get("stage") == receipt.stage
                and output.get("role") == receipt.role
                and output.get("verdict") == receipt.verdict
                and sorted(output.get("finding_ids", []))
                == receipt.finding_ids
                and isinstance(trace, dict)
                and trace.get("producer_run_id") == receipt.producer_run_id
                and trace.get("reviewer_run_id") == receipt.reviewer_run_id
                and trace.get("producer_context_id")
                == receipt.producer_context_id
                and trace.get("reviewer_context_id")
                == receipt.reviewer_context_id
                and trace.get("context_isolation_attested") is True
                and trace.get("allowed_input_hashes")
                == receipt.allowed_input_hashes
            )
        except (OSError, ValueError, TypeError, StageWorkspaceError):
            return False
        return (
            evidence_matches
            and receipt.authority_key_id == self.authority_key_id
            and self._verify_mac(
                "independent_review_receipt_v50",
                receipt.unsigned_hash(),
                receipt.authority_auth_tag,
            )
        )

    def _load_json_model(self, relative_path: str, model_type: type[Any]) -> Any:
        path = self._safe_file(relative_path)
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise StageWorkspaceError(
                f"{relative_path} does not satisfy {model_type.__name__}: {exc}"
            ) from exc

    def _mechanical_evaluation(self, stage: StageId) -> tuple[bool, list[str]]:
        """Run only structural/provenance checks; never scientific stand-ins."""

        reasons: list[str] = []
        try:
            if stage == "S0":
                contract = json.loads(
                    self._safe_file("problem/contract.json").read_text(
                        encoding="utf-8"
                    )
                )
                if contract.get("mission_hash") != self.spec.mission_hash:
                    reasons.append("problem contract mission_hash mismatch")
                if (
                    contract.get("evidence_snapshot_hash")
                    != self.spec.evidence_snapshot_hash
                ):
                    reasons.append("problem contract evidence_snapshot_hash mismatch")
                if contract.get("question") != self.spec.objective:
                    reasons.append(
                        "problem contract question must equal the frozen objective"
                    )
                regime = self._load_json_model(
                    "docs/regime.json", RegimeDiagnosisV50
                )
                decision_function = self._load_json_model(
                    "problem/decision_function.json", DecisionFunctionSpecV50
                )
                if regime.decision_function_id != decision_function.function_id:
                    reasons.append(
                        "regime diagnosis references another decision function"
                    )
                for canary in decision_function.canaries:
                    actual = _evaluate_arithmetic(
                        decision_function.expression, canary.inputs
                    )
                    if abs(actual - canary.expected) > canary.tolerance:
                        reasons.append(
                            f"decision function canary failed: {canary.canary_id}"
                        )
            elif stage == "S1":
                candidates = self._load_json_model(
                    "docs/candidates.json", CandidateSetV50
                )
                assumptions = self._load_json_model(
                    "docs/assumptions.json", AssumptionSetV50
                )
                symbols = self._load_json_model(
                    "docs/symbols.json", SymbolTableV50
                )
                model = self._load_json_model("docs/model_spec.json", ModelSpecV50)
                plan = self._load_json_model(
                    "docs/validation_plan.json", ValidationPlanV50
                )
                if self.spec.profile.compete and len(candidates.candidates) < 3:
                    reasons.append("COMPETE requires at least three candidates")
                if self.spec.profile.compete and (
                    len(candidates.generation_receipt_hashes) < 3
                ):
                    reasons.append("COMPETE requires three generation receipt hashes")
                role_receipts = {
                    receipt.receipt_hash: receipt
                    for _, receipt in self._artifacts_of_kind(
                        "role_execution_receipt_v50", RoleExecutionReceiptV50
                    )
                    if self.verify_role_execution(receipt)
                    and receipt.stage == "S1"
                    and receipt.input_authority_hash == self.current_gate("S0")
                }
                generation_receipts = [
                    role_receipts.get(receipt_hash)
                    for receipt_hash in candidates.generation_receipt_hashes
                ]
                if any(item is None for item in generation_receipts):
                    reasons.append("candidate generation receipt is unauthenticated")
                else:
                    typed_generation_receipts = [
                        item for item in generation_receipts if item is not None
                    ]
                    if any(
                        item.role != "modeler"
                        for item in typed_generation_receipts
                    ):
                        reasons.append("candidate generation must use modeler roles")
                    if len(
                        {item.context_id for item in typed_generation_receipts}
                    ) != len(typed_generation_receipts):
                        reasons.append("candidate modeler contexts must be distinct")
                    if {
                        item.subject_id for item in typed_generation_receipts
                    } != {item.candidate_id for item in candidates.candidates}:
                        reasons.append(
                            "candidate IDs do not match modeler execution receipts"
                        )
                    if len(typed_generation_receipts) != len(
                        candidates.candidates
                    ):
                        reasons.append(
                            "candidate/modeler execution count does not match"
                        )
                    candidate_by_id_for_receipt = {
                        item.candidate_id: item for item in candidates.candidates
                    }
                    for item in typed_generation_receipts:
                        output = self._artifact_payload_by_hash(
                            item.output_artifact_hash
                        )
                        candidate = candidate_by_id_for_receipt.get(
                            item.subject_id
                        )
                        if (
                            not isinstance(output, dict)
                            or candidate is None
                            or output.get("candidate_id") != item.subject_id
                            or output.get("candidate_hash")
                            != candidate.structural_hash()
                        ):
                            reasons.append(
                                f"modeler output is not bound to {item.subject_id}"
                            )
                scout = role_receipts.get(
                    candidates.literature_scout_receipt_hash
                )
                if scout is None or scout.role != "literature_scout":
                    reasons.append("literature scout execution receipt is missing")
                elif not isinstance(
                    self._artifact_payload_by_hash(scout.output_artifact_hash),
                    dict,
                ):
                    reasons.append("literature scout output is malformed")
                candidate_by_id = {
                    item.candidate_id: item for item in candidates.candidates
                }
                selected = candidate_by_id.get(model.selected_candidate_id)
                if selected is None:
                    reasons.append("selected candidate is not in candidate set")
                elif (
                    model.selected_candidate_structural_hash
                    != selected.structural_hash()
                ):
                    reasons.append(
                        "model spec is not bound to selected candidate structure"
                    )
                assumption_ids = {
                    item.assumption_id for item in assumptions.assumptions
                }
                symbol_ids = {item.symbol_id for item in symbols.symbols}
                if not set(model.assumption_ids).issubset(assumption_ids):
                    reasons.append("model references undeclared assumptions")
                if not set(model.symbol_ids).issubset(symbol_ids):
                    reasons.append("model references undeclared symbols")
                if selected and (
                    model.assumption_ids != selected.assumption_ids
                    or model.symbol_ids != selected.symbol_ids
                    or model.data_requirement_ids
                    != selected.data_requirement_ids
                ):
                    reasons.append(
                        "model spec declarations differ from selected candidate"
                    )
                plan_ids = {item.check_id for item in plan.obligations}
                if selected and not set(
                    selected.validation_obligation_ids
                ).issubset(plan_ids):
                    reasons.append("candidate obligations are absent from frozen plan")
            elif stage == "S2":
                ledger = self._load_json_model("data/ledger.json", DataLedgerV50)
                manifest = self._load_json_model(
                    "data/processed/manifest.json", ProcessedManifestV50
                )
                model = self._load_json_model("docs/model_spec.json", ModelSpecV50)
                plan = self._load_json_model(
                    "docs/validation_plan.json", ValidationPlanV50
                )
                raw_tree_hash = _tree_hash(self.root / "data" / "raw")
                raw_baseline = self._raw_baseline_for_current_s2()
                if raw_baseline is None:
                    reasons.append("authenticated raw input baseline is missing")
                elif raw_baseline.raw_tree_hash != raw_tree_hash:
                    reasons.append("raw tree differs from authenticated baseline")
                if ledger.raw_baseline_tree_hash != raw_tree_hash:
                    reasons.append("ledger raw baseline differs from current raw tree")
                if manifest.raw_baseline_tree_hash != raw_tree_hash:
                    reasons.append("processed manifest raw baseline mismatch")
                ledger_ids = {item.data_item_id for item in ledger.entries}
                if not set(model.data_requirement_ids).issubset(ledger_ids):
                    reasons.append("not every model data requirement has a ledger row")
                artifact_by_id = {
                    item.data_item_id: item for item in manifest.artifacts
                }
                if set(artifact_by_id) != ledger_ids:
                    reasons.append(
                        "processed manifest IDs must exactly match data ledger IDs"
                    )
                baseline_paths = (
                    {item.relative_path for item in raw_baseline.files}
                    if raw_baseline is not None
                    else set()
                )
                ledger_raw_paths = {
                    item.raw_relative_path
                    for item in ledger.entries
                    if item.raw_relative_path is not None
                }
                if baseline_paths != ledger_raw_paths:
                    reasons.append(
                        "raw baseline files must exactly match ledger raw paths"
                    )
                plan_ids = {item.check_id for item in plan.obligations}
                for entry in ledger.entries:
                    transform_script = self._safe_file(
                        entry.transform_script_relative_path
                    )
                    if _sha256_bytes(
                        transform_script.read_bytes()
                    ) != entry.transform_script_hash:
                        reasons.append(
                            f"transform script hash mismatch for {entry.data_item_id}"
                        )
                    if (
                        entry.raw_relative_path is not None
                        and entry.raw_response_hash is not None
                    ):
                        raw_path = self._safe_file(entry.raw_relative_path)
                        if _sha256_bytes(
                            raw_path.read_bytes()
                        ) != entry.raw_response_hash:
                            reasons.append(
                                f"raw file hash mismatch for {entry.data_item_id}"
                            )
                    artifact = artifact_by_id.get(entry.data_item_id)
                    if artifact is None:
                        reasons.append(
                            f"processed manifest misses {entry.data_item_id}"
                        )
                        continue
                    processed_path = self._safe_file(artifact.relative_path)
                    actual_hash = _sha256_bytes(processed_path.read_bytes())
                    if actual_hash != artifact.artifact_hash:
                        reasons.append(
                            f"processed file hash mismatch for {entry.data_item_id}"
                        )
                    if entry.processed_artifact_hash != artifact.artifact_hash:
                        reasons.append(
                            f"ledger/manifest hash mismatch for {entry.data_item_id}"
                        )
                    if (
                        entry.source_kind in {"synthetic", "estimated"}
                        and entry.sensitivity_requirement_id not in plan_ids
                    ):
                        reasons.append(
                            f"synthetic item {entry.data_item_id} lacks planned sensitivity"
                        )
            elif stage == "S3":
                code = self._load_json_model(
                    "results/code_manifest.json", CodeManifestV50
                )
                result_index = self._load_json_model(
                    "results/index.json", ResultIndexV50
                )
                if code.source_tree_hash != _tree_hash(self.root / "src"):
                    reasons.append("source_tree_hash differs from current src tree")
                reference_hashes = {
                    code.environment_ref: code.environment_hash,
                    code.replay_receipt_ref: code.replay_receipt_hash,
                    code.fermi_estimate_ref: code.fermi_estimate_hash,
                    **code.toy_oracle_hashes,
                }
                for relative_path, expected_hash in reference_hashes.items():
                    artifact_path = self._safe_file(relative_path)
                    if _sha256_bytes(artifact_path.read_bytes()) != expected_hash:
                        reasons.append(
                            f"code manifest reference hash mismatch: {relative_path}"
                        )
                replay_payload = json.loads(
                    self._safe_file(code.replay_receipt_ref).read_text(
                        encoding="utf-8"
                    )
                )
                expected_replay_fields = {
                    "replay_command": code.replay_command,
                    "source_tree_hash": code.source_tree_hash,
                    "environment_hash": code.environment_hash,
                    "random_seed": code.random_seed,
                    "exit_code": 0,
                    "passed": True,
                }
                if not isinstance(replay_payload, dict) or any(
                    replay_payload.get(key) != value
                    for key, value in expected_replay_fields.items()
                ):
                    reasons.append(
                        "replay receipt is not bound to command/environment/source"
                    )
                for record in result_index.records:
                    artifact_path = self._safe_file(record.relative_path)
                    if _sha256_bytes(
                        artifact_path.read_bytes()
                    ) != record.artifact_hash:
                        reasons.append(
                            f"result artifact hash mismatch for {record.result_id}"
                        )
                        continue
                    try:
                        artifact_payload = json.loads(
                            artifact_path.read_text(encoding="utf-8")
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        reasons.append(
                            f"result artifact is not JSON for {record.result_id}"
                        )
                        continue
                    expected_fields = {
                        "result_id": record.result_id,
                        "value": record.value,
                        "interval_low": record.interval_low,
                        "interval_high": record.interval_high,
                        "units": record.units,
                    }
                    if not isinstance(artifact_payload, dict) or any(
                        artifact_payload.get(key) != value
                        for key, value in expected_fields.items()
                    ):
                        reasons.append(
                            f"result index/payload mismatch for {record.result_id}"
                        )
            elif stage == "S4":
                verification = json.loads(
                    self._safe_file("results/verification_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                uq = self._load_json_model("results/uq_summary.json", UQSummaryV50)
                plan = self._load_json_model(
                    "docs/validation_plan.json", ValidationPlanV50
                )
                results = self._load_json_model(
                    "results/index.json", ResultIndexV50
                )
                if verification.get("validation_plan_hash") != plan.plan_hash:
                    reasons.append("verification summary is not bound to validation plan")
                planned_ids = {
                    item.check_id
                    for item in plan.obligations
                    if item.stage == "S4" and item.required
                }
                reported_ids = set(verification.get("check_ids", []))
                if not planned_ids.issubset(reported_ids):
                    reasons.append("verification summary omits planned S4 checks")
                result_ids = {item.result_id for item in results.records}
                result_by_id = {
                    item.result_id: item for item in results.records
                }
                for claim in uq.claims:
                    if claim.result_id not in result_ids:
                        reasons.append(f"UQ claim {claim.claim_id} lacks result")
                    if claim.interval_result_id not in result_ids:
                        reasons.append(f"UQ claim {claim.claim_id} lacks interval result")
                    else:
                        interval = result_by_id[claim.interval_result_id]
                        estimate = result_by_id.get(claim.result_id)
                        if (
                            interval.interval_low is None
                            or interval.interval_high is None
                        ):
                            reasons.append(
                                f"UQ claim {claim.claim_id} references a result "
                                "without a finite interval"
                            )
                        if estimate is not None and interval.units != estimate.units:
                            reasons.append(
                                f"UQ claim {claim.claim_id} interval units mismatch"
                            )
            elif stage == "S5":
                dossier = self._load_json_model(
                    "results/decision_dossier.json", DecisionDossierV50
                )
                results = self._load_json_model(
                    "results/index.json", ResultIndexV50
                )
                uq = self._load_json_model("results/uq_summary.json", UQSummaryV50)
                plan = self._load_json_model(
                    "docs/validation_plan.json", ValidationPlanV50
                )
                result_ids = {item.result_id for item in results.records}
                claim_ids = {item.claim_id for item in uq.claims}
                for assertion in dossier.assertions:
                    if not set(assertion.result_ids).issubset(result_ids):
                        reasons.append(
                            f"decision assertion {assertion.assertion_id} has unknown result"
                        )
                    if not set(assertion.uq_claim_ids).issubset(claim_ids):
                        reasons.append(
                            f"decision assertion {assertion.assertion_id} lacks UQ"
                        )
                expected_high_disagreement = any(
                    claim.ensemble_disagreement
                    >= plan.ensemble_disagreement_threshold
                    for claim in uq.claims
                )
                if (
                    dossier.high_disagreement_detected
                    != expected_high_disagreement
                ):
                    reasons.append(
                        "decision disagreement flag differs from frozen threshold"
                    )
                if (
                    any(
                        claim.support_status != "in_support"
                        for claim in uq.claims
                    )
                    and dossier.next_action
                    != plan.unsupported_support_action
                ):
                    reasons.append(
                        "out-of-support UQ must return to data acquisition"
                    )
                if dossier.next_action == "register_prediction":
                    seals = [
                        seal
                        for _, seal in self._artifacts_of_kind(
                            "prediction_seal_v50", PredictionSealV50
                        )
                        if seal.seal_hash == dossier.prediction_seal_hash
                        and self.verify_prediction_seal(seal)
                    ]
                    if len(seals) != 1:
                        reasons.append(
                            "decision dossier prediction hash lacks one "
                            "authenticated external seal"
                        )
                    elif seals[0].s4_gate_hash != self.current_gate("S4"):
                        reasons.append(
                            "prediction seal is bound to a stale S4 gate"
                        )
            elif stage == "S6":
                receipt = PaperBuildReceipt.model_validate_json(
                    self._safe_file(
                        "paper/build/build_receipt.json"
                    ).read_text(encoding="utf-8")
                )
                verification = verify_paper_build(self.root)
                if not verification.ok:
                    reasons.extend(
                        f"paper build mismatch: {item}"
                        for item in verification.mismatches
                    )
                if verification.receipt_hash != receipt.receipt_hash:
                    reasons.append("paper verification receipt hash mismatch")
                result_index = self._load_json_model(
                    "results/index.json", ResultIndexV50
                )
                values_payload = json.loads(
                    self._safe_file("results/values.json").read_text(
                        encoding="utf-8"
                    )
                )
                expected_values: dict[str, float] = {}
                for record in result_index.records:
                    if record.value is not None:
                        expected_values[record.result_id] = record.value
                    if (
                        record.interval_low is not None
                        and record.interval_high is not None
                    ):
                        expected_values[
                            f"{record.result_id}_low"
                        ] = record.interval_low
                        expected_values[
                            f"{record.result_id}_high"
                        ] = record.interval_high
                if values_payload != expected_values:
                    reasons.append(
                        "paper values are not the exact ResultIndex projection"
                    )
        except (
            StageWorkspaceError,
            OSError,
            ValueError,
            TypeError,
            SyntaxError,
            ArithmeticError,
            json.JSONDecodeError,
        ) as exc:
            reasons.append(str(exc))
        return (not reasons, reasons)

    def run_mechanical_check(self, stage: StageId) -> CheckResultV50:
        manifest = self._manifest_for_stage(stage)
        passed, reasons = self._mechanical_evaluation(stage)
        report = {
            "schema_version": "5.0",
            "stage": stage,
            "manifest_hash": manifest.manifest_hash,
            "passed": passed,
            "reasons": reasons,
            "evidence_class": (
                "integrity" if stage in {"S2", "S6"} else "workflow_presence"
            ),
            "scientific_correctness_established": False,
        }
        evidence = self.commit_evidence("mechanical_check_report_v50", report)
        adapter_code_hash = _sha256_bytes(Path(__file__).read_bytes())
        return self.issue_check(
            stage=stage,
            check_id=POLICIES[stage].structural_check_id,
            level="L5",
            evidence_class=report["evidence_class"],
            applicability="applicable",
            status="PASS" if passed else "FAIL",
            reason_code="mechanical_complete" if passed else "mechanical_failure",
            adapter_id="v5_mechanical_registry",
            adapter_version="5.0",
            adapter_code_hash=adapter_code_hash,
            evidence_refs=[evidence.sha256],
            subject_hashes=[item.sha256 for item in manifest.files],
            metrics={"finding_count": len(reasons)},
            scope="development",
            executed_by="harness",
        )

    def _validation_plan(self) -> ValidationPlanV50 | None:
        path = self.root / "docs" / "validation_plan.json"
        if not path.is_file():
            return None
        return self._load_json_model("docs/validation_plan.json", ValidationPlanV50)

    def _required_checks(
        self,
        stage: StageId,
        *,
        manifest: StageArtifactManifestV50 | None = None,
    ) -> dict[str, dict[str, str]]:
        structural_class = (
            "integrity" if stage in {"S2", "S6"} else "workflow_presence"
        )
        required = {
            POLICIES[stage].structural_check_id: {
                "applicability": "applicable",
                "level": "L5",
                "evidence_class": structural_class,
            }
        }
        if stage in {"S3", "S4"}:
            if manifest is None:
                plan = self._validation_plan()
            else:
                try:
                    plan = ValidationPlanV50.model_validate_json(
                        self._manifest_file_bytes(
                            manifest, "docs/validation_plan.json"
                        )
                    )
                except ValueError as exc:
                    raise StageWorkspaceError(
                        "historical validation plan snapshot is invalid"
                    ) from exc
            if plan is None:
                return required
            for obligation in plan.obligations:
                if obligation.stage == stage and obligation.required:
                    required[obligation.check_id] = {
                        "applicability": obligation.applicability,
                        "level": obligation.level,
                        "evidence_class": obligation.evidence_class,
                    }
        return required

    def _latest_checks(
        self, stage: StageId, manifest_hash: str
    ) -> dict[str, CheckResultV50]:
        found: dict[str, CheckResultV50] = {}
        for _, result in self._artifacts_of_kind(
            "check_result_v50", CheckResultV50
        ):
            if result.stage != stage or result.input_manifest_hash != manifest_hash:
                continue
            if not self.verify_check(result):
                continue
            previous = found.get(result.check_id)
            key = (result.finished_at, str(result.result_hash))
            if previous is None or key > (
                previous.finished_at,
                str(previous.result_hash),
            ):
                found[result.check_id] = result
        return found

    def _latest_reviews(
        self, stage: StageId, manifest_hash: str
    ) -> dict[str, IndependentReviewReceiptV50]:
        found: dict[str, IndependentReviewReceiptV50] = {}
        for _, receipt in self._artifacts_of_kind(
            "independent_review_receipt_v50",
            IndependentReviewReceiptV50,
        ):
            if (
                receipt.stage != stage
                or receipt.input_manifest_hash != manifest_hash
                or not self.verify_review(receipt)
            ):
                continue
            previous = found.get(receipt.role)
            key = (receipt.issued_at, str(receipt.receipt_hash))
            if previous is None or key > (
                previous.issued_at,
                str(previous.receipt_hash),
            ):
                found[receipt.role] = receipt
        return found

    def evaluate_gate(
        self,
        stage: StageId,
        *,
        authority: Literal["verifier", "human"] = "verifier",
    ) -> GateEvaluationV50:
        if authority not in {"verifier", "human"}:
            raise PermissionError("model cannot issue gate certificates")
        gate = self._binding(stage, "gate")
        state = self.graph.project_state()
        if gate.node_hash not in state.snapshot.frontier_node_hashes:
            current = self.current_gate(stage)
            if current:
                manifest = self._manifest_for_stage(stage)
                return GateEvaluationV50(
                    stage=stage,
                    manifest_hash=manifest.manifest_hash,
                    decision="OPEN",
                    reasons=["existing current certificate"],
                    certificate_hash=current,
                )
            raise StageWorkspaceError(f"{stage} gate is not on the graph frontier")
        manifest = self._manifest_for_stage(stage)
        if not self._manifest_is_current(manifest):
            return GateEvaluationV50(
                stage=stage,
                manifest_hash=manifest.manifest_hash,
                decision="BLOCKED",
                reasons=["stage manifest is stale"],
            )

        reasons: list[str] = []
        needs_evidence = False
        accepted_checks: list[str] = []
        checks = self._latest_checks(stage, str(manifest.manifest_hash))
        for check_id, expectation in self._required_checks(
            stage, manifest=manifest
        ).items():
            result = checks.get(check_id)
            if result is None:
                reasons.append(f"missing check: {check_id}")
                needs_evidence = True
                continue
            if (
                result.level != expectation["level"]
                or result.evidence_class != expectation["evidence_class"]
                or result.protocol_hash != POLICIES[stage].policy_hash
                or (
                    result.level in {"L0", "L1", "L2", "L3", "L4"}
                    and result.scope != self.spec.evidence_scope
                )
            ):
                reasons.append(f"{check_id} protocol mismatch")
                continue
            planned_applicability = expectation["applicability"]
            if planned_applicability == "not_applicable":
                if (
                    result.status != "NOT_APPLICABLE"
                    or result.applicability != "not_applicable"
                ):
                    reasons.append(
                        f"{check_id} must use frozen NOT_APPLICABLE"
                    )
                    needs_evidence = True
                    continue
            elif result.status == "PASS" and result.applicability == "applicable":
                pass
            elif result.status == "NOT_RUN":
                reasons.append(f"{check_id} was NOT_RUN")
                needs_evidence = True
                continue
            elif result.status == "NOT_APPLICABLE":
                reasons.append(f"{check_id} was not pre-registered as N/A")
                needs_evidence = True
                continue
            else:
                reasons.append(f"{check_id} status is {result.status}")
                continue
            assert result.result_hash is not None
            accepted_checks.append(result.result_hash)

        reviews = self._latest_reviews(stage, str(manifest.manifest_hash))
        accepted_reviews: list[str] = []
        reviewer_contexts: set[str] = set()
        expected_review_inputs = sorted(
            {item.sha256 for item in manifest.files}
            | {
                str(result.result_hash)
                for result in checks.values()
                if result.result_hash is not None
            }
        )
        for role in POLICIES[stage].required_review_roles:
            receipt = reviews.get(role)
            if receipt is None:
                reasons.append(f"missing independent review: {role}")
                needs_evidence = True
                continue
            if receipt.allowed_input_hashes != expected_review_inputs:
                reasons.append(f"{role} review input snapshot is incomplete or stale")
                continue
            if receipt.reviewer_context_id in reviewer_contexts:
                reasons.append(f"review context reused: {role}")
                continue
            reviewer_contexts.add(receipt.reviewer_context_id)
            if receipt.verdict == "APPROVE":
                assert receipt.receipt_hash is not None
                accepted_reviews.append(receipt.receipt_hash)
            elif receipt.verdict == "HUMAN":
                reasons.append(f"{role} requires human judgement")
                needs_evidence = True
            else:
                reasons.append(f"{role} rejected the stage")

        if reasons:
            hard_failure = any(
                "status is FAIL" in reason
                or "status is ERROR" in reason
                or "rejected" in reason
                or "reused" in reason
                or "input snapshot" in reason
                or "protocol mismatch" in reason
                for reason in reasons
            )
            decision = "BLOCKED" if hard_failure else "NEEDS_EVIDENCE"
            return GateEvaluationV50(
                stage=stage,
                manifest_hash=manifest.manifest_hash,
                decision=decision,
                reasons=reasons,
                accepted_check_hashes=sorted(accepted_checks),
                accepted_review_hashes=sorted(accepted_reviews),
            )

        upstream: list[str] = []
        if stage != "S0":
            predecessor = self.current_gate(STAGES[_STAGE_INDEX[stage] - 1])
            if predecessor is None:
                return GateEvaluationV50(
                    stage=stage,
                    manifest_hash=manifest.manifest_hash,
                    decision="BLOCKED",
                    reasons=["predecessor gate became stale"],
                )
            upstream.append(predecessor)
        certificate_check_hashes = sorted(
            str(result.result_hash)
            for result in checks.values()
            if result.result_hash is not None
        )
        unsigned = GateCertificateV50(
            workspace_spec_hash=self.spec.spec_hash,
            stage=stage,
            attempt=manifest.attempt,
            policy_hash=POLICIES[stage].policy_hash,
            manifest=manifest,
            upstream_gate_hashes=sorted(upstream),
            check_result_hashes=certificate_check_hashes,
            reviewer_receipt_hashes=sorted(accepted_reviews),
            graph_gate_node_hash=gate.node_hash,
            graph_snapshot_before_hash=state.snapshot.snapshot_hash,
            evaluator_epoch=self.spec.evaluator_epoch,
            authority=authority,
            authority_key_id=self.authority_key_id,
            issued_at=_utc_now(),
        )
        payload = unsigned.model_dump(mode="json")
        payload["authority_auth_tag"] = self._mac(
            "gate_certificate_v50", unsigned.unsigned_hash()
        )
        payload["certificate_hash"] = sha256_value(
            {
                key: value
                for key, value in payload.items()
                if key != "certificate_hash"
            }
        )
        certificate = GateCertificateV50.model_validate(payload)
        certificate_ref = self.graph.put_output(
            "gate_certificate_v50", certificate
        )
        self.graph.record_outcome(
            str(gate.node_hash),
            actor="verifier",
            status="succeeded",
            output_artifacts=[certificate_ref],
            summary=f"{stage} workflow gate opened; no scientific qualification",
            outcome_id=f"{gate.node_id}-outcome",
        )
        _write_json_projection(
            self.root / "gates" / f"{stage.lower()}.json",
            {
                "schema_version": "5.0-projection",
                "stage": stage,
                "certificate_hash": certificate.certificate_hash,
                "graph_gate_node_hash": certificate.graph_gate_node_hash,
                "manifest_hash": certificate.manifest.manifest_hash,
                "authority": "authenticated_external_key",
                "projection_is_not_authority": True,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        self._write_graph_projection()
        return GateEvaluationV50(
            stage=stage,
            manifest_hash=manifest.manifest_hash,
            decision="OPEN",
            reasons=[],
            accepted_check_hashes=sorted(accepted_checks),
            accepted_review_hashes=sorted(accepted_reviews),
            certificate_hash=certificate.certificate_hash,
        )

    def verify_certificate(self, certificate: GateCertificateV50) -> bool:
        self._refresh_artifact_index()
        certificate_hash = str(certificate.certificate_hash)
        cached = self._certificate_verification_cache.get(certificate_hash)
        if cached is not None:
            return cached
        valid = self._verify_certificate_dependencies(certificate, seen=set())
        self._certificate_verification_cache[certificate_hash] = valid
        return valid

    def _verify_certificate_dependencies(
        self,
        certificate: GateCertificateV50,
        *,
        seen: set[str],
    ) -> bool:
        try:
            GateCertificateV50.model_validate(
                certificate.model_dump(mode="json")
            )
            certificate_hash = str(certificate.certificate_hash)
            if certificate_hash in seen:
                return False
            seen = {*seen, certificate_hash}
            if certificate.workspace_spec_hash != self.spec.spec_hash:
                return False
            if certificate.evaluator_epoch != self.spec.evaluator_epoch:
                return False
            if (
                certificate.policy_hash
                != POLICIES[certificate.stage].policy_hash
            ):
                return False
            if certificate.authority_key_id != self.authority_key_id:
                return False
            if not self._verify_mac(
                "gate_certificate_v50",
                certificate.unsigned_hash(),
                certificate.authority_auth_tag,
            ):
                return False
            if not self._verify_manifest_snapshots(certificate.manifest):
                return False

            check_by_hash = {
                str(result.result_hash): result
                for _, result in self._artifacts_of_kind(
                    "check_result_v50", CheckResultV50
                )
                if result.result_hash is not None
            }
            if any(
                result_hash not in check_by_hash
                for result_hash in certificate.check_result_hashes
            ):
                return False
            certificate_checks = [
                check_by_hash[result_hash]
                for result_hash in certificate.check_result_hashes
            ]
            if any(
                result.stage != certificate.stage
                or result.input_manifest_hash
                != certificate.manifest.manifest_hash
                or not self.verify_check(result)
                for result in certificate_checks
            ):
                return False
            check_by_id: dict[str, CheckResultV50] = {}
            for result in certificate_checks:
                if result.check_id in check_by_id:
                    return False
                check_by_id[result.check_id] = result
            for check_id, expectation in self._required_checks(
                certificate.stage, manifest=certificate.manifest
            ).items():
                result = check_by_id.get(check_id)
                if result is None:
                    return False
                if (
                    result.level != expectation["level"]
                    or result.evidence_class
                    != expectation["evidence_class"]
                    or result.protocol_hash
                    != POLICIES[certificate.stage].policy_hash
                    or (
                        result.level in {"L0", "L1", "L2", "L3", "L4"}
                        and result.scope != self.spec.evidence_scope
                    )
                ):
                    return False
                if expectation["applicability"] == "not_applicable":
                    if (
                        result.status != "NOT_APPLICABLE"
                        or result.applicability != "not_applicable"
                    ):
                        return False
                elif (
                    result.status != "PASS"
                    or result.applicability != "applicable"
                ):
                    return False

            review_by_hash = {
                str(receipt.receipt_hash): receipt
                for _, receipt in self._artifacts_of_kind(
                    "independent_review_receipt_v50",
                    IndependentReviewReceiptV50,
                )
                if receipt.receipt_hash is not None
            }
            if any(
                receipt_hash not in review_by_hash
                for receipt_hash in certificate.reviewer_receipt_hashes
            ):
                return False
            certificate_reviews = [
                review_by_hash[receipt_hash]
                for receipt_hash in certificate.reviewer_receipt_hashes
            ]
            expected_review_inputs = sorted(
                {item.sha256 for item in certificate.manifest.files}
                | set(certificate.check_result_hashes)
            )
            if any(
                receipt.stage != certificate.stage
                or receipt.input_manifest_hash
                != certificate.manifest.manifest_hash
                or receipt.verdict != "APPROVE"
                or receipt.allowed_input_hashes != expected_review_inputs
                or not self.verify_review(receipt)
                for receipt in certificate_reviews
            ):
                return False
            reviews_by_role = {
                receipt.role: receipt for receipt in certificate_reviews
            }
            if len(reviews_by_role) != len(certificate_reviews):
                return False
            if set(POLICIES[certificate.stage].required_review_roles) != set(
                reviews_by_role
            ):
                return False
            contexts = {
                receipt.reviewer_context_id for receipt in certificate_reviews
            }
            if len(contexts) != len(certificate_reviews):
                return False

            all_certificates = self._artifacts_of_kind(
                "gate_certificate_v50", GateCertificateV50
            )
            certificate_artifacts = [
                ref
                for ref, item in all_certificates
                if item.certificate_hash == certificate.certificate_hash
            ]
            if len(certificate_artifacts) != 1:
                return False
            state = self.graph.project_state()
            node = next(
                (
                    item
                    for item in state.nodes
                    if item.node_hash == certificate.graph_gate_node_hash
                ),
                None,
            )
            if (
                node is None
                or node.node_kind != "evaluation"
                or node.executor != "verifier"
            ):
                return False
            matching_outcomes = [
                outcome
                for outcome in state.outcomes
                if outcome.node_hash == certificate.graph_gate_node_hash
                and certificate_artifacts[0] in outcome.output_artifacts
            ]
            if len(matching_outcomes) != 1:
                return False
            if (
                matching_outcomes[0].base_snapshot_hash
                != certificate.graph_snapshot_before_hash
                or matching_outcomes[0].status != "succeeded"
            ):
                return False

            if certificate.stage == "S0":
                if certificate.upstream_gate_hashes:
                    return False
            else:
                if len(certificate.upstream_gate_hashes) != 1:
                    return False
                upstream_hash = certificate.upstream_gate_hashes[0]
                upstream = next(
                    (
                        item
                        for _, item in all_certificates
                        if item.certificate_hash == upstream_hash
                    ),
                    None,
                )
                expected_stage = STAGES[_STAGE_INDEX[certificate.stage] - 1]
                if (
                    upstream is None
                    or upstream.stage != expected_stage
                    or not self._verify_certificate_dependencies(
                        upstream, seen=seen
                    )
                ):
                    return False
            return True
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            StageWorkspaceError,
        ):
            return False

    def _certificate_for_current_node(
        self, stage: StageId
    ) -> GateCertificateV50 | None:
        gate = self._binding(stage, "gate")
        state = self.graph.project_state()
        if state.snapshot.node_statuses.get(gate.node_hash) != "succeeded":
            return None
        outcomes = [
            outcome for outcome in state.outcomes if outcome.node_hash == gate.node_hash
        ]
        if len(outcomes) != 1:
            return None
        refs = [
            ref
            for ref in outcomes[0].output_artifacts
            if ref.kind == "gate_certificate_v50"
        ]
        if len(refs) != 1:
            return None
        try:
            certificate = GateCertificateV50.model_validate(
                self.graph.store.load_artifact(refs[0])
            )
        except (OSError, ValueError, RuntimeError):
            return None
        if certificate.graph_gate_node_hash != gate.node_hash:
            return None
        return certificate

    def current_gate(self, stage: StageId) -> str | None:
        certificate = self._certificate_for_current_node(stage)
        if certificate is None or not self.verify_certificate(certificate):
            return None
        if not self._manifest_is_current(certificate.manifest):
            return None
        if stage != "S0":
            previous = STAGES[_STAGE_INDEX[stage] - 1]
            previous_hash = self.current_gate(previous)
            if previous_hash is None:
                return None
            if certificate.upstream_gate_hashes != [previous_hash]:
                return None
        return certificate.certificate_hash

    def invalidate_from(
        self,
        stage: StageId,
        *,
        reason: str,
        authority: Literal["verifier", "human"] = "verifier",
    ) -> list[str]:
        if len(reason.strip()) < 3:
            raise ValueError("invalidation reason must be explicit")
        state_before = self.graph.project_state()
        remaining_stage_count = len(STAGES) - _STAGE_INDEX[stage]
        required_new_nodes = 2 * remaining_stage_count
        required_future_outcomes = 2 * remaining_stage_count
        if len(state_before.nodes) + required_new_nodes > self.spec.max_nodes:
            raise StageWorkspaceError(
                "cannot invalidate: node budget cannot hold a complete retry chain"
            )
        if (
            len(state_before.outcomes) + required_future_outcomes
            > self.spec.max_outcomes
        ):
            raise StageWorkspaceError(
                "cannot invalidate: outcome budget cannot complete the retry chain"
            )
        work = self._binding(stage, "work")
        old_bindings: dict[tuple[StageId, str], str] = {}
        for item_stage in STAGES[_STAGE_INDEX[stage] :]:
            for kind in ("work", "gate"):
                old_bindings[(item_stage, kind)] = str(
                    self._binding(item_stage, kind).node_hash
                )
        receipt = self.graph.revoke_node(
            str(work.node_hash),
            authority=authority,
            reason=reason,
            revocation_id=f"v5-{stage.lower()}-a{self._latest_attempt(stage)}-revoke",
        )
        next_attempt = max(
            self._latest_attempt(item_stage)
            for item_stage in STAGES[_STAGE_INDEX[stage] :]
        ) + 1
        self._add_stage_chain(
            start_stage=stage,
            attempt=next_attempt,
            superseded=old_bindings,
        )
        decisions_path = self.root / "docs" / "decisions.log"
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                canonical_json(
                    {
                        "event": "stage_invalidated_v50",
                        "stage": stage,
                        "reason": reason,
                        "revocation_hash": receipt.receipt_hash,
                        "affected_node_hashes": receipt.affected_node_hashes,
                        "scientific_qualification_granted": False,
                        "real_world_action_authorized": False,
                    }
                )
                + "\n"
            )
        self._write_graph_projection()
        return receipt.affected_node_hashes

    def reconcile_staleness(
        self, *, reason: str = "artifact snapshot changed after gate"
    ) -> StageId | None:
        """Revoke the earliest graph stage whose signed certificate is stale."""

        for stage in STAGES:
            certificate = self._certificate_for_current_node(stage)
            if certificate is None:
                continue
            if self.current_gate(stage) is None:
                self.invalidate_from(stage, reason=reason)
                return stage
        return None

    def status(self) -> WorkflowStatusV50:
        state = self.graph.project_state()
        current: dict[StageId, str] = {}
        stale: dict[StageId, str] = {}
        stage_statuses: dict[StageId, str] = {}
        frontier: list[StageId] = []
        for stage in STAGES:
            work = self._binding(stage, "work")
            gate = self._binding(stage, "gate")
            gate_status = state.snapshot.node_statuses.get(gate.node_hash, "pending")
            work_status = state.snapshot.node_statuses.get(work.node_hash, "pending")
            certificate = self._certificate_for_current_node(stage)
            current_hash = self.current_gate(stage)
            if current_hash:
                current[stage] = current_hash
                stage_statuses[stage] = "gate_open"
            elif certificate and certificate.certificate_hash:
                stale[stage] = certificate.certificate_hash
                stage_statuses[stage] = "stale"
            elif gate_status in {"failed", "blocked"}:
                stage_statuses[stage] = gate_status
            elif work_status == "succeeded":
                stage_statuses[stage] = "awaiting_gate_evidence"
            elif work.node_hash in state.snapshot.frontier_node_hashes:
                stage_statuses[stage] = "frontier"
                frontier.append(stage)
            else:
                stage_statuses[stage] = str(work_status)
        return WorkflowStatusV50(
            workspace_id=self.spec.workspace_id,
            graph_id=self.spec.graph_id,
            graph_verified=self.graph.verify(),
            stage_statuses=stage_statuses,
            current_gate_hashes=current,
            stale_gate_hashes=stale,
            frontier_stages=frontier,
        )

    def _write_graph_projection(self) -> None:
        state = self.graph.project_state()
        stage_bindings = []
        for (stage, attempt, kind), node in sorted(self._node_map().items()):
            stage_bindings.append(
                {
                    "stage": stage,
                    "attempt": attempt,
                    "role": kind,
                    "node_hash": node.node_hash,
                    "node_kind": node.node_kind,
                    "status": state.snapshot.node_statuses.get(node.node_hash),
                }
            )
        _write_json_projection(
            self.root / ".fma" / "stage_graph_projection.json",
            {
                "schema_version": "5.0-projection",
                "graph_id": self.spec.graph_id,
                "graph_snapshot_hash": state.snapshot.snapshot_hash,
                "graph_event_tip": state.snapshot.last_graph_event_hash,
                "stage_nodes": stage_bindings,
                "projection_is_not_authority": True,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
        )
        status = self.status()
        status_payload = {
            **status.model_dump(mode="json"),
            "projection_is_not_authority": True,
            "graph_snapshot_hash": state.snapshot.snapshot_hash,
        }
        _write_json_projection(
            self.root / "checks" / "status.json",
            status_payload,
        )
        history_path = self.root / ".fma" / "status_history.jsonl"
        with history_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                canonical_json(
                    {
                        "recorded_at": _utc_now().isoformat(),
                        **status_payload,
                    }
                )
                + "\n"
            )

    def verify(self) -> bool:
        try:
            self.spec.assert_sealed()
            if not self.graph.verify():
                return False
            if (
                self.graph.contract.allowed_actions != self.spec.permitted_actions
                or self.graph.contract.forbidden_actions
                != self.spec.forbidden_actions
            ):
                return False
            if not self._verify_authority_genesis():
                return False
            authoritative = self._artifacts_of_kind(
                "task_workspace_spec_v50", TaskWorkspaceSpecV50
            )
            if len(authoritative) != 1 or authoritative[0][1] != self.spec:
                return False
            state = self.graph.project_state()
            if state.promotions:
                # S0--S6 gates must never use V4 scientific promotion.
                return False
            for _, result in self._artifacts_of_kind(
                "check_result_v50", CheckResultV50
            ):
                if not self.verify_check(result):
                    return False
            for _, receipt in self._artifacts_of_kind(
                "adapter_execution_receipt_v50",
                AdapterExecutionReceiptV50,
            ):
                if not self.verify_adapter_execution(receipt):
                    return False
            for _, receipt in self._artifacts_of_kind(
                "independent_review_receipt_v50",
                IndependentReviewReceiptV50,
            ):
                if not self.verify_review(receipt):
                    return False
            for _, receipt in self._artifacts_of_kind(
                "role_execution_receipt_v50", RoleExecutionReceiptV50
            ):
                if not self.verify_role_execution(receipt):
                    return False
            for _, certificate in self._artifacts_of_kind(
                "gate_certificate_v50", GateCertificateV50
            ):
                if not self.verify_certificate(certificate):
                    return False
            for _, seal in self._artifacts_of_kind(
                "prediction_seal_v50", PredictionSealV50
            ):
                if not self.verify_prediction_seal(seal):
                    return False
            for _, baseline in self._artifacts_of_kind(
                "raw_data_baseline_v50", RawDataBaselineV50
            ):
                if not self.verify_raw_baseline(baseline):
                    return False
            return True
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            return False


__all__ = [
    "POLICIES",
    "STAGES",
    "StagePolicy",
    "StageWorkspaceError",
    "StageWorkspaceV50",
]
