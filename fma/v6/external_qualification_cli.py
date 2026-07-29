"""Operator CLI for the additive V6.5 external-qualification control plane.

Every state-changing command requires a workspace-authenticated principal
capability and routes through :class:`ExternalQualificationControlPlaneV65`.
The CLI has no V6.3 mutation bypass, signer, key-generation, external-role
private-key, scientific-promotion, or real-world action command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fma.schemas import StrictModel

from .external_control_plane import (
    ExternalControlPrincipalV65,
    ExternalQualificationControlPlaneV65,
)
from .external_qualification import (
    ExternalAggregateEvaluationV63,
    ExternalEvidenceCustodyV63,
    ExternalPredictionRegistrationV63,
    ExternalPredictivePromotionV63,
)


def _json(payload: object, *, stream: Any = None) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")  # type: ignore[union-attr]
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        ),
        file=stream or sys.stdout,
    )


def _decode_workspace_key(payload: bytes) -> bytes:
    stripped = payload.strip()
    if len(stripped) == 64:
        try:
            return bytes.fromhex(stripped.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            pass
    if len(stripped) < 32:
        raise ValueError(
            "workspace authority key must contain at least 32 bytes"
        )
    return stripped


def _load_public_key_manifest(path_value: str) -> dict[str, bytes]:
    path = Path(path_value).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("public key manifest must be a JSON object")
    keys = raw.get("keys", raw)
    if not isinstance(keys, dict) or not keys:
        raise ValueError("public key manifest contains no keys")
    loaded: dict[str, bytes] = {}
    for key_id, specification in keys.items():
        if not isinstance(key_id, str):
            raise ValueError(
                "public key manifest key IDs must be strings"
            )
        if isinstance(specification, dict):
            if set(specification) != {"pem_file"} or not isinstance(
                specification["pem_file"], str
            ):
                raise ValueError(
                    "public key entries must contain only pem_file"
                )
            key_path = (
                path.parent / specification["pem_file"]
            ).resolve()
            loaded[key_id] = key_path.read_bytes()
        elif isinstance(specification, str):
            if "BEGIN PUBLIC KEY" in specification:
                loaded[key_id] = specification.encode("utf-8")
            else:
                loaded[key_id] = (
                    path.parent / specification
                ).resolve().read_bytes()
        else:
            raise ValueError("public key manifest entry is invalid")
    return loaded


def _load_typed_file(
    path_value: str,
    model_type: type[StrictModel],
) -> StrictModel:
    path = Path(path_value).expanduser().resolve()
    return model_type.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def _control_plane(
    args: argparse.Namespace,
    *,
    mutation: bool,
) -> ExternalQualificationControlPlaneV65:
    key = _decode_workspace_key(
        Path(args.authority_key_file).expanduser().resolve().read_bytes()
    )
    principal = None
    if mutation:
        principal = _load_typed_file(
            args.principal_capability_file,
            ExternalControlPrincipalV65,
        )
        assert isinstance(principal, ExternalControlPrincipalV65)
    return ExternalQualificationControlPlaneV65(
        Path(args.workspace),
        authority_key=key,
        authority_key_id=args.authority_key_id,
        trusted_public_keys=_load_public_key_manifest(
            args.public_key_manifest
        ),
        principal=principal,
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--authority-key-file", required=True)
    parser.add_argument("--authority-key-id", required=True)
    parser.add_argument("--public-key-manifest", required=True)


def _add_mutation_common(
    parser: argparse.ArgumentParser,
    *,
    expected_state: bool = True,
) -> None:
    _add_common(parser)
    parser.add_argument("--qualification-id", required=True)
    parser.add_argument("--principal-capability-file", required=True)
    if expected_state:
        parser.add_argument("--expected-state-hash", required=True)
        parser.add_argument("--expected-phase", required=True)


def _add_signed_ingress(
    subparsers: Any,
    command: str,
    help_text: str,
) -> None:
    ingress = subparsers.add_parser(command, help=help_text)
    _add_mutation_common(ingress)
    ingress.add_argument("--artifact-file", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fma.v6.external_qualification_cli",
        description=(
            "V6.5 single-writer external qualification control plane. "
            "No command signs an external-role envelope, grants scientific "
            "qualification, or authorizes a real-world action."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    state = sub.add_parser(
        "state",
        help="pure-read V6.5 control projection",
    )
    _add_common(state)
    state.add_argument("--qualification-id", required=True)

    verify = sub.add_parser(
        "verify",
        help="pure-read newest V6.5 qualification projection",
    )
    _add_common(verify)
    verify.add_argument("--qualification-id", required=True)

    activate = sub.add_parser(
        "activate",
        help="establish the non-retroactive V6.5 ownership boundary",
    )
    _add_mutation_common(activate)

    _add_signed_ingress(
        sub,
        "ingest-custody",
        "ingest one externally signed custody envelope",
    )

    run_prediction = sub.add_parser(
        "run-prediction",
        help="run or exact-resume deterministic current-model prediction",
    )
    _add_mutation_common(run_prediction)

    _add_signed_ingress(
        sub,
        "ingest-registration",
        "ingest one externally signed registration envelope",
    )

    reserve = sub.add_parser(
        "reserve-evaluation",
        help="reserve one evaluator dispatch and emit a local packet",
    )
    _add_mutation_common(reserve)
    reserve.add_argument("--evaluator-key-id", required=True)
    reserve.add_argument("--evaluator-host-id", required=True)

    _add_signed_ingress(
        sub,
        "ingest-evaluation",
        "ingest one externally signed aggregate evaluation",
    )
    _add_signed_ingress(
        sub,
        "ingest-promotion",
        "ingest one externally signed V6.3 promotion decision",
    )

    abort = sub.add_parser(
        "abort-attempt",
        help="abort an unprogressed HUMAN_REQUIRED operation only",
    )
    _add_mutation_common(abort, expected_state=False)
    abort.add_argument("--operation-id", required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    mutation = args.command not in {"state", "verify"}
    control = _control_plane(args, mutation=mutation)
    if args.command == "state":
        state = control.state(qualification_id=args.qualification_id)
        _json(state)
        if state.control_status in {"ABORTED", "INCONSISTENT"}:
            return 4
        if state.control_status in {
            "LEGACY_UNMANAGED",
            "PENDING_FAILURE",
        }:
            return 3
        return 0
    if args.command == "verify":
        projection = control.latest_qualification(
            qualification_id=args.qualification_id
        )
        _json(projection)
        if projection.projection_status == "WORKFLOW_VERIFIED":
            return 0
        if projection.projection_status == "NOT_RUN":
            return 3
        return 4
    if args.command == "activate":
        result: object = control.activate(
            qualification_id=args.qualification_id,
            expected_v63_state_hash=args.expected_state_hash,
            expected_v63_phase=args.expected_phase,
        )
    elif args.command == "ingest-custody":
        custody = _load_typed_file(
            args.artifact_file, ExternalEvidenceCustodyV63
        )
        assert isinstance(custody, ExternalEvidenceCustodyV63)
        result = control.ingest_custody(
            qualification_id=args.qualification_id,
            expected_v63_state_hash=args.expected_state_hash,
            expected_v63_phase=args.expected_phase,
            custody=custody,
        )
    elif args.command == "run-prediction":
        result = control.run_prediction(
            qualification_id=args.qualification_id,
            expected_v63_state_hash=args.expected_state_hash,
            expected_v63_phase=args.expected_phase,
        )
    elif args.command == "ingest-registration":
        registration = _load_typed_file(
            args.artifact_file, ExternalPredictionRegistrationV63
        )
        assert isinstance(registration, ExternalPredictionRegistrationV63)
        result = control.ingest_registration(
            qualification_id=args.qualification_id,
            expected_v63_state_hash=args.expected_state_hash,
            expected_v63_phase=args.expected_phase,
            registration=registration,
        )
    elif args.command == "reserve-evaluation":
        result = control.reserve_evaluation(
            qualification_id=args.qualification_id,
            expected_v63_state_hash=args.expected_state_hash,
            expected_v63_phase=args.expected_phase,
            evaluator_key_id=args.evaluator_key_id,
            evaluator_host_id=args.evaluator_host_id,
        )
    elif args.command == "ingest-evaluation":
        evaluation = _load_typed_file(
            args.artifact_file, ExternalAggregateEvaluationV63
        )
        assert isinstance(evaluation, ExternalAggregateEvaluationV63)
        result = control.ingest_evaluation(
            qualification_id=args.qualification_id,
            expected_v63_state_hash=args.expected_state_hash,
            expected_v63_phase=args.expected_phase,
            evaluation=evaluation,
        )
    elif args.command == "ingest-promotion":
        promotion = _load_typed_file(
            args.artifact_file, ExternalPredictivePromotionV63
        )
        assert isinstance(promotion, ExternalPredictivePromotionV63)
        result = control.ingest_promotion(
            qualification_id=args.qualification_id,
            expected_v63_state_hash=args.expected_state_hash,
            expected_v63_phase=args.expected_phase,
            promotion=promotion,
        )
    elif args.command == "abort-attempt":
        result = control.abort_attempt(
            qualification_id=args.qualification_id,
            operation_id=args.operation_id,
        )
    else:  # pragma: no cover - argparse owns command exhaustiveness.
        raise RuntimeError("unknown control-plane command")
    _json(result)
    status = getattr(result, "control_status", None)
    if status is None and hasattr(result, "state"):
        status = result.state.control_status
    return 0 if status == "ACTIVE" else 4


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        _json(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "v63_protocol_qualification_granted": False,
                "scientific_qualification_granted": False,
                "real_world_action_authorized": False,
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
