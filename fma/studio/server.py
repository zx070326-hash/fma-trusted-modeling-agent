"""Loopback HTTP server for the FMA modeling studio."""

from __future__ import annotations

import hmac
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .service import StudioBridgeError, StudioTaskService


_TASK_ROUTE = re.compile(r"/api/v1/tasks/([A-Za-z0-9._-]+)")
_RUN_S0_ROUTE = re.compile(r"/api/v1/tasks/([A-Za-z0-9._-]+)/run-s0")
_RUN_S1_ROUTE = re.compile(r"/api/v1/tasks/([A-Za-z0-9._-]+)/run-s1")


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: StudioTaskService,
        *,
        token: str,
        allowed_origins: set[str],
    ) -> None:
        if len(token) < 24:
            raise ValueError("bridge token must contain at least 24 characters")
        super().__init__(address, StudioRequestHandler)
        self.service = service
        self.bridge_token = token
        self.allowed_origins = set(allowed_origins)


class StudioRequestHandler(BaseHTTPRequestHandler):
    server: StudioHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _origin(self) -> str | None:
        return self.headers.get("Origin")

    def _origin_allowed(self) -> bool:
        origin = self._origin()
        return origin is None or origin in self.server.allowed_origins

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = self._origin()
        if origin and origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, exc: Exception) -> None:
        if isinstance(exc, StudioBridgeError):
            self._send(
                exc.http_status,
                {
                    "status": "error",
                    "type": exc.error_type,
                    "message": str(exc),
                },
            )
            return
        self._send(
            500,
            {
                "status": "error",
                "type": "internal_error",
                "message": "The local bridge failed closed.",
            },
        )

    def _authorized(self) -> bool:
        supplied = self.headers.get("X-FMA-Bridge-Token", "")
        return bool(supplied) and hmac.compare_digest(
            supplied, self.server.bridge_token
        )

    def _require_access(self) -> bool:
        if not self._origin_allowed():
            self._send(
                403,
                {
                    "status": "error",
                    "type": "origin_denied",
                    "message": "Browser origin is not allowed by the local bridge.",
                },
            )
            return False
        if not self._authorized():
            self._send(
                401,
                {
                    "status": "error",
                    "type": "auth_required",
                    "message": "A valid local bridge token is required.",
                },
            )
            return False
        return True

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > 32_768:
            raise ValueError("JSON body must be between 1 and 32768 bytes")
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_OPTIONS(self) -> None:
        if not self._origin_allowed():
            self._send(403, {"status": "error", "type": "origin_denied"})
            return
        self.send_response(204)
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-FMA-Bridge-Token, Idempotency-Key",
        )
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/v1/health":
            if not self._origin_allowed():
                self._send(403, {"status": "error", "type": "origin_denied"})
                return
            self._send(
                200,
                {
                    "status": "ok",
                    "service": "fma-studio-bridge",
                    "api_version": "1",
                    "scope": "loopback_local_execution",
                    "authority_key_exposed": False,
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
            )
            return
        if not self._require_access():
            return
        try:
            if path == "/api/v1/tasks":
                self._send(200, self.server.service.list_tasks())
                return
            match = _TASK_ROUTE.fullmatch(path)
            if match:
                self._send(200, self.server.service.snapshot(match.group(1)))
                return
            self._send(404, {"status": "error", "type": "not_found"})
        except Exception as exc:
            self._error(exc)

    def do_POST(self) -> None:
        if not self._require_access():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/v1/tasks":
                payload = self._body()
                self._send(201, self.server.service.create_task(payload))
                return
            match = _RUN_S0_ROUTE.fullmatch(path)
            if match:
                self._send(202, self.server.service.start_s0(match.group(1)))
                return
            match = _RUN_S1_ROUTE.fullmatch(path)
            if match:
                self._send(202, self.server.service.start_s1(match.group(1)))
                return
            self._send(404, {"status": "error", "type": "not_found"})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(
                400,
                {
                    "status": "error",
                    "type": "invalid_arguments",
                    "message": str(exc),
                },
            )
        except Exception as exc:
            self._error(exc)
