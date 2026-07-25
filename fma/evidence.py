from __future__ import annotations

import json
import sqlite3
from collections import deque
from pathlib import Path
from uuid import uuid4

from .hashing import sha256_value
from .storage import utc_now


class EvidenceGraph:
    """Small persistent Claim-Evidence DAG with revocation propagation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                artifact_hash TEXT,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL REFERENCES nodes(node_id),
                target_id TEXT NOT NULL REFERENCES nodes(node_id),
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, relation)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EvidenceGraph":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add_node(
        self,
        kind: str,
        *,
        artifact_hash: str | None = None,
        status: str = "current",
        metadata: dict[str, object] | None = None,
    ) -> str:
        node_id = f"node-{uuid4().hex}"
        now = utc_now()
        self.connection.execute(
            "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                node_id,
                kind,
                artifact_hash,
                status,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )
        self._audit("node_added", {"node_id": node_id, "kind": kind, "status": status})
        self.connection.commit()
        return node_id

    def _reachable(self, start: str, target: str) -> bool:
        queue = deque([start])
        seen: set[str] = set()
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            rows = self.connection.execute(
                "SELECT target_id FROM edges WHERE source_id = ?", (current,)
            ).fetchall()
            queue.extend(row["target_id"] for row in rows)
        return False

    def add_edge(self, source_id: str, target_id: str, relation: str = "supports") -> None:
        if source_id == target_id or self._reachable(target_id, source_id):
            raise ValueError("evidence edges must remain acyclic")
        self.connection.execute(
            "INSERT INTO edges VALUES (?, ?, ?, ?)",
            (source_id, target_id, relation, utc_now()),
        )
        self._audit(
            "edge_added",
            {"source_id": source_id, "target_id": target_id, "relation": relation},
        )
        self.connection.commit()

    def supporting_nodes(self, claim_node_id: str) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT n.* FROM nodes n
            JOIN edges e ON e.source_id = n.node_id
            WHERE e.target_id = ? AND e.relation = 'supports'
            ORDER BY n.kind, n.node_id
            """,
            (claim_node_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def node_status(self, node_id: str) -> str:
        row = self.connection.execute(
            "SELECT status FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        return str(row["status"])

    def _set_claim_status_by_promotion(
        self,
        claim_node_id: str,
        status: str,
        *,
        policy_version: str,
    ) -> None:
        row = self.connection.execute(
            "SELECT kind, status FROM nodes WHERE node_id = ?", (claim_node_id,)
        ).fetchone()
        if row is None or row["kind"] != "claim":
            raise ValueError("promotion can update claim nodes only")
        if row["status"] == "revoked":
            raise RuntimeError("a revoked claim cannot be resurrected by promotion")
        if status not in {"validated", "run_invalid", "needs_evidence"}:
            raise ValueError(f"unsupported promotion status: {status}")
        self.connection.execute(
            "UPDATE nodes SET status = ?, updated_at = ? WHERE node_id = ?",
            (status, utc_now(), claim_node_id),
        )
        self._audit(
            "promotion_applied",
            {
                "claim_node_id": claim_node_id,
                "status": status,
                "policy_version": policy_version,
            },
        )
        self.connection.commit()

    def revoke_node(self, node_id: str, reason: str) -> list[str]:
        """Revoke a node and every downstream conclusion that depends on it."""
        if not self.connection.execute(
            "SELECT 1 FROM nodes WHERE node_id = ?", (node_id,)
        ).fetchone():
            raise KeyError(node_id)
        queue = deque([node_id])
        affected: list[str] = []
        seen: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            affected.append(current)
            self.connection.execute(
                "UPDATE nodes SET status = 'revoked', updated_at = ? WHERE node_id = ?",
                (utc_now(), current),
            )
            rows = self.connection.execute(
                "SELECT target_id FROM edges WHERE source_id = ?", (current,)
            ).fetchall()
            queue.extend(row["target_id"] for row in rows)
        self._audit(
            "revocation_cascade",
            {"root_node_id": node_id, "reason": reason, "affected": affected},
        )
        self.connection.commit()
        return affected

    def snapshot(self) -> dict[str, object]:
        nodes = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM nodes ORDER BY node_id"
            ).fetchall()
        ]
        edges = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM edges ORDER BY source_id, target_id, relation"
            ).fetchall()
        ]
        payload: dict[str, object] = {"nodes": nodes, "edges": edges}
        return {**payload, "snapshot_hash": sha256_value(payload)}

    def _audit(self, event_type: str, payload: dict[str, object]) -> None:
        self.connection.execute(
            "INSERT INTO audit_events(event_type, payload_json, created_at) VALUES (?, ?, ?)",
            (
                event_type,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )
