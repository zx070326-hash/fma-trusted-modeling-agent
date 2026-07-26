"""Command-line entry point for the local FMA studio bridge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fma.codex_driver import CodexCLIConfig
from fma.v5.__main__ import _decode_key

from .server import StudioHTTPServer
from .service import StudioTaskService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fma.studio",
        description=(
            "Loopback-only execution bridge for the FMA modeling studio. "
            "The bridge exposes no authority key or private acceptance data."
        ),
    )
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--authority-key-file", required=True)
    parser.add_argument("--authority-key-id", default="studio-local-v1")
    parser.add_argument("--token", default=os.environ.get("FMA_STUDIO_TOKEN"))
    parser.add_argument("--codex-bin")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=["http://localhost:3001", "http://127.0.0.1:3001"],
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("studio bridge may bind only to loopback")
    if not args.token or len(args.token) < 24:
        raise ValueError(
            "set FMA_STUDIO_TOKEN or --token to at least 24 characters"
        )
    key_path = Path(args.authority_key_file).expanduser().resolve(strict=True)
    authority_key = _decode_key(key_path.read_bytes())
    codex_config = CodexCLIConfig(
        executable=Path(args.codex_bin).resolve(strict=True)
        if args.codex_bin
        else None
    )
    service = StudioTaskService(
        args.task_root,
        authority_key=authority_key,
        authority_key_id=args.authority_key_id,
        codex_config=codex_config,
    )
    server = StudioHTTPServer(
        (args.host, args.port),
        service,
        token=args.token,
        allowed_origins=set(args.allowed_origin),
    )
    print(
        f"FMA Studio bridge listening on http://{args.host}:{args.port}; "
        "authority key remains server-side.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
