"""CLI for native Codex paper authoring, build, review, and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fma.codex_driver import DEFAULT_EXPECTED_CLI_VERSION, CodexCLIConfig
from fma.v5.__main__ import _add_authority_flags, _workspace

from .paper_renderer import build_paper_v71, verify_paper_build_v71
from .paper_role_driver import (
    run_native_layout_review_v71,
    run_native_paper_author_v71,
    run_native_semantic_review_v71,
)
from .paper_runtime import (
    PaperDeliveryError,
    audit_paper_content_v71,
    finalize_paper_delivery_v71,
    prepare_paper_delivery_v71,
    project_paper_status_v71,
    verify_paper_delivery_v71,
)


def _json(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    elif hasattr(value, "__dict__"):
        value = value.__dict__
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
    )


def _base(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    _add_authority_flags(parser)


def _codex(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-bin")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--expected-codex-cli-version",
        default=DEFAULT_EXPECTED_CLI_VERSION,
    )
    parser.add_argument("--codex-timeout-seconds", type=int, default=900)


def _publication(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", action="append", required=True)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument(
        "--venue-profile",
        choices=(
            "academic_article",
            "modeling_contest",
            "technical_report",
        ),
        default="academic_article",
    )
    parser.add_argument("--max-pages", type=int, default=24)
    parser.add_argument("--max-revision-rounds", type=int, default=2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fma-paper",
        description=(
            "Post-S6 native Codex paper delivery. Outputs are publication "
            "projections only, never scientific qualification or action authority."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    _base(prepare)
    _publication(prepare)
    prepare.add_argument("--model", default="gpt-5.6-sol")

    author = sub.add_parser("author")
    _base(author)
    _codex(author)

    audit = sub.add_parser("audit")
    _base(audit)

    semantic = sub.add_parser("review-content")
    _base(semantic)
    _codex(semantic)

    build = sub.add_parser("build")
    _base(build)
    build.add_argument("--xelatex", default="xelatex")
    build.add_argument("--pdfinfo", default="pdfinfo")
    build.add_argument("--pdftoppm", default="pdftoppm")
    build.add_argument("--timeout-seconds", type=float, default=180.0)

    layout = sub.add_parser("review-layout")
    _base(layout)
    _codex(layout)

    finalize = sub.add_parser("finalize")
    _base(finalize)

    verify_build = sub.add_parser("verify-build")
    _base(verify_build)

    verify = sub.add_parser("verify")
    _base(verify)

    run = sub.add_parser("run")
    _base(run)
    _publication(run)
    _codex(run)
    run.add_argument("--xelatex", default="xelatex")
    run.add_argument("--pdfinfo", default="pdfinfo")
    run.add_argument("--pdftoppm", default="pdftoppm")
    run.add_argument("--build-timeout-seconds", type=float, default=180.0)
    return parser


def _codex_config(args: argparse.Namespace) -> CodexCLIConfig:
    executable = (
        Path(args.codex_bin).expanduser().resolve(strict=True)
        if args.codex_bin
        else None
    )
    return CodexCLIConfig(
        executable=executable,
        requested_model=args.model,
        expected_cli_version=args.expected_codex_cli_version,
        timeout_seconds=args.codex_timeout_seconds,
    )


def _run(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    if (
        args.command not in {"verify", "verify-build"}
        and workspace.current_gate("S6") is None
    ):
        raise RuntimeError(
            "V7.1 paper delivery requires a current authenticated S6 gate"
        )
    if args.command == "prepare":
        paths = prepare_paper_delivery_v71(
            workspace,
            title_hint=args.title,
            authors=args.author,
            language=args.language,
            venue_profile=args.venue_profile,
            requested_model=args.model,
            max_pages=args.max_pages,
            max_revision_rounds=args.max_revision_rounds,
        )
        finalized = (paths.attempt_root / "delivery_receipt.json").is_file()
        verification = (
            verify_paper_delivery_v71(workspace) if finalized else None
        )
        if verification is not None and not verification.ok:
            raise PaperDeliveryError(
                "existing finalized paper attempt failed verification: "
                + "; ".join(verification.mismatches)
            )
        _json(
            {
                "attempt_id": paths.attempt_id,
                "attempt_root": str(paths.attempt_root),
                "status": (
                    verification.status
                    if verification is not None
                    else "NEEDS_REVISION"
                ),
                "idempotent_reuse": finalized,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            }
        )
    elif args.command == "author":
        _json(
            run_native_paper_author_v71(
                workspace.root, config=_codex_config(args)
            )
        )
    elif args.command == "audit":
        result = audit_paper_content_v71(workspace)
        if result.status != "PASS":
            project_paper_status_v71(workspace.root, "NEEDS_REVISION")
        _json(result)
        return 0 if result.status == "PASS" else 3
    elif args.command == "review-content":
        result = run_native_semantic_review_v71(
            workspace.root, config=_codex_config(args)
        )
        if result.verdict != "APPROVE":
            project_paper_status_v71(
                workspace.root,
                "HUMAN" if result.verdict == "HUMAN" else "NEEDS_REVISION",
            )
        _json(result)
        return 0 if result.verdict == "APPROVE" else 3
    elif args.command == "build":
        _json(
            build_paper_v71(
                workspace.root,
                xelatex_command=args.xelatex,
                pdfinfo_command=args.pdfinfo,
                pdftoppm_command=args.pdftoppm,
                timeout_seconds=args.timeout_seconds,
            )
        )
    elif args.command == "review-layout":
        result = run_native_layout_review_v71(
            workspace.root, config=_codex_config(args)
        )
        if result.verdict != "APPROVE":
            project_paper_status_v71(
                workspace.root,
                "HUMAN" if result.verdict == "HUMAN" else "NEEDS_REVISION",
            )
        _json(result)
        return 0 if result.verdict == "APPROVE" else 3
    elif args.command == "finalize":
        _json(finalize_paper_delivery_v71(workspace))
    elif args.command == "verify-build":
        result = verify_paper_build_v71(workspace.root)
        _json(result)
        return 0 if result.ok else 4
    elif args.command == "verify":
        result = verify_paper_delivery_v71(workspace)
        _json(result)
        return 0 if result.ok else 4
    elif args.command == "run":
        paths = prepare_paper_delivery_v71(
            workspace,
            title_hint=args.title,
            authors=args.author,
            language=args.language,
            venue_profile=args.venue_profile,
            requested_model=args.model,
            max_pages=args.max_pages,
            max_revision_rounds=args.max_revision_rounds,
        )
        if (paths.attempt_root / "delivery_receipt.json").is_file():
            verification = verify_paper_delivery_v71(workspace)
            if not verification.ok:
                raise PaperDeliveryError(
                    "existing finalized paper attempt failed verification: "
                    + "; ".join(verification.mismatches)
                )
            _json(
                {
                    "attempt_id": paths.attempt_id,
                    "status": verification.status,
                    "idempotent_reuse": True,
                    "verification": verification.model_dump(mode="json"),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                }
            )
            return 0
        config = _codex_config(args)
        feedback: list[str] = []
        author = None
        for revision_round in range(args.max_revision_rounds + 1):
            author = run_native_paper_author_v71(
                workspace.root,
                config=config,
                revision_round=revision_round,
                revision_feedback=feedback,
            )
            audit = audit_paper_content_v71(workspace)
            if audit.status != "PASS":
                feedback = [
                    f"content_audit: {message}" for message in audit.errors
                ]
                stopped_at = "content_audit"
                detail = audit.model_dump(mode="json")
            else:
                semantic = run_native_semantic_review_v71(
                    workspace.root, config=config
                )
                if semantic.verdict == "HUMAN":
                    project_paper_status_v71(workspace.root, "HUMAN")
                    _json(
                        {
                            "attempt_id": paths.attempt_id,
                            "status": "HUMAN",
                            "stopped_at": "semantic_review",
                            "revision_round": revision_round,
                            "review": semantic.model_dump(mode="json"),
                        }
                    )
                    return 3
                if semantic.verdict == "REJECT":
                    feedback = [
                        (
                            "semantic_review: "
                            f"{finding.finding_id}: {finding.message}"
                        )
                        for finding in semantic.findings
                    ]
                    stopped_at = "semantic_review"
                    detail = semantic.model_dump(mode="json")
                else:
                    try:
                        build = build_paper_v71(
                            workspace.root,
                            xelatex_command=args.xelatex,
                            pdfinfo_command=args.pdfinfo,
                            pdftoppm_command=args.pdftoppm,
                            timeout_seconds=args.build_timeout_seconds,
                        )
                    except PaperDeliveryError as exc:
                        feedback = [f"paper_build: {exc}"]
                        stopped_at = "paper_build"
                        detail = {"error": str(exc)}
                    else:
                        layout = run_native_layout_review_v71(
                            workspace.root, config=config
                        )
                        if layout.verdict == "HUMAN":
                            project_paper_status_v71(
                                workspace.root, "HUMAN"
                            )
                            _json(
                                {
                                    "attempt_id": paths.attempt_id,
                                    "status": "HUMAN",
                                    "stopped_at": "layout_review",
                                    "revision_round": revision_round,
                                    "build_hash": build.build_hash,
                                    "review": layout.model_dump(mode="json"),
                                }
                            )
                            return 3
                        if layout.verdict == "REJECT":
                            feedback = [
                                f"layout_review: {message}"
                                for message in layout.findings
                            ]
                            stopped_at = "layout_review"
                            detail = layout.model_dump(mode="json")
                        else:
                            receipt = finalize_paper_delivery_v71(workspace)
                            _json(
                                {
                                    "attempt_id": paths.attempt_id,
                                    "status": receipt.status,
                                    "revision_round": revision_round,
                                    "delivery": receipt.model_dump(mode="json"),
                                    "writer_schema": author.schema_version,
                                }
                            )
                            return 0
            if revision_round < args.max_revision_rounds:
                continue
            project_paper_status_v71(workspace.root, "NEEDS_REVISION")
            _json(
                {
                    "attempt_id": paths.attempt_id,
                    "status": "NEEDS_REVISION",
                    "stopped_at": stopped_at,
                    "revision_round": revision_round,
                    "revision_budget_exhausted": True,
                    "detail": detail,
                }
            )
            return 3
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return _run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scientific_qualification_granted": False,
                    "real_world_action_authorized": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
