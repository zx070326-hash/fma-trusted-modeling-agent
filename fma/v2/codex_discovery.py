from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from fma.codex_driver import (
    CodexCLIConfig,
    CodexCLIError,
    CodexCLIExplorer,
    ProcessOutputLimitExceeded,
    _audit_jsonl,
    _redacted_event_ledger,
    _sha256_bytes,
    _tree_snapshot,
)
from fma.hashing import canonical_json, sha256_value

from .discovery import (
    ProblemDiscoveryContext,
    ProblemDiscoveryExplorer,
    ProblemDiscoveryProposal,
    ProblemHypothesisDraft,
)
from .discovery_store import DiscoveryRunStore
from .schemas import (
    DiscoveryArtifactRef,
    DiscoveryProviderObservation,
    EvidenceSnapshot,
    StrictModel,
)


DISCOVERY_PROTOCOL_VERSION = "fma.codex_problem_discovery.v1"
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "codex_problem_discovery.md"


class CodexProblemDiscoveryResponse(StrictModel):
    protocol_version: Literal[DISCOVERY_PROTOCOL_VERSION]
    request_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    status: Literal["proposed", "no_result"]
    draft: ProblemHypothesisDraft | None = None
    no_result_reason: Annotated[str, Field(max_length=2_000)] = ""
    notes: Annotated[list[str], Field(max_length=16)] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_response(self) -> "CodexProblemDiscoveryResponse":
        if self.status == "proposed":
            if self.draft is None or self.no_result_reason:
                raise ValueError("proposed response needs exactly one draft")
        elif self.draft is not None or not self.no_result_reason.strip():
            raise ValueError("no_result response needs a reason and no draft")
        return self


class CodexProblemDiscoveryExplorer(ProblemDiscoveryExplorer):
    """Draft-only Codex provider built on the audited isolated CLI transport.

    This class owns neither admission nor the discovery state transition.  It
    records one provider observation that the V2 store must bind to a later
    submitted draft before the existing admission gate can run.
    """

    provider_id = "codex_cli"
    provider_version = DISCOVERY_PROTOCOL_VERSION

    def __init__(
        self,
        store: DiscoveryRunStore,
        config: CodexCLIConfig | None = None,
        *,
        process_runner: object | None = None,
        cli_locator: object | None = None,
        prompt_guard: object | None = None,
    ) -> None:
        self.store = store
        self.config = config or CodexCLIConfig(max_candidates=1)
        if self.config.max_candidates != 1:
            raise ValueError("problem discovery permits exactly one draft per invocation")
        # Reuse the existing isolation, auth-only home, readiness checks, event
        # auditing, scratch policy, and bounded process runner.  Its internal
        # trace is supplementary; the V2 observation remains the admission link.
        self._transport = CodexCLIExplorer(
            self.store.run_directory / "codex_transport",
            self.config,
            process_runner=process_runner,  # type: ignore[arg-type]
            cli_locator=cli_locator,  # type: ignore[arg-type]
            prompt_guard=prompt_guard,  # type: ignore[arg-type]
            run_id="transport",
        )

    def close(self) -> None:
        self._transport.close()

    def propose(self, context: ProblemDiscoveryContext) -> ProblemDiscoveryProposal:
        snapshot, approved_context = self.store.assert_problem_discovery_context(context)
        self.store.assert_allowed_actions(
            {"codex_cli_inference", "write_local_run_artifacts"}
        )
        request_id = uuid4().hex
        schema = CodexProblemDiscoveryResponse.model_json_schema()
        schema_text = canonical_json(schema)
        prompt = self._build_prompt(approved_context, request_id)
        prompt_ref = self.store.put_artifact(
            "codex_discovery_prompt",
            {
                "prompt": prompt,
                "excluded_private_fields": [
                    "acceptance tests",
                    "frozen contracts",
                    "solver results",
                    "validation results",
                    "action authority",
                ],
            },
        )
        schema_ref = self.store.put_artifact("codex_discovery_output_schema", schema)
        transport_ref = self.store.put_artifact(
            "codex_discovery_transport_trace",
            {
                "transport_trace_directory": str(self._transport.trace_directory),
                "transport": "fma.codex_driver.CodexCLIExplorer",
                "isolation_profile": "read_only_ephemeral_no_observed_tools",
            },
        )
        trace_refs = [prompt_ref, schema_ref, transport_ref]

        if len(schema_text.encode("utf-8")) > self.config.max_schema_bytes:
            return self._record_terminal(
                snapshot,
                approved_context,
                request_id,
                "error",
                "output_limit_exceeded",
                trace_refs,
            )
        if len(prompt.encode("utf-8")) > self.config.max_input_bytes:
            return self._record_terminal(
                snapshot,
                approved_context,
                request_id,
                "error",
                "input_limit_exceeded",
                trace_refs,
            )

        try:
            executable, cli_version, executable_hash, server_names = (
                self._transport._ensure_readiness()
            )
            invocation_id = f"discovery-{uuid4().hex[:12]}"
            scratch = self._transport._initialize_scratch(invocation_id)
            schema_path = scratch / "problem-discovery-output.schema.json"
            schema_path.write_text(schema_text + "\n", encoding="utf-8")
            before = _tree_snapshot(scratch)
            argv = self._transport._build_argv(
                executable, scratch, schema_path, server_names
            )
            manifest_ref = self.store.put_artifact(
                "codex_discovery_invocation_manifest",
                {
                    "manifest_version": "1.0",
                    "invocation_id": invocation_id,
                    "request_id": request_id,
                    "argv": argv,
                    "argv_hash": sha256_value(argv),
                    "prompt": prompt_ref.model_dump(mode="json"),
                    "schema": schema_ref.model_dump(mode="json"),
                    "transport_trace_directory": str(self._transport.trace_directory),
                    "cli_version": cli_version,
                    "executable_sha256": executable_hash,
                    "environment_policy": {
                        "codex_home": "temporary_auth_only",
                        "child_shell_inherit": "none",
                        "secret_environment_allowlist": False,
                    },
                    "limits": {
                        "timeout_seconds": self.config.timeout_seconds,
                        "max_stdout_bytes": self.config.max_stdout_bytes,
                        "max_stderr_bytes": self.config.max_stderr_bytes,
                        "max_events": self.config.max_events,
                    },
                },
            )
            trace_refs.append(manifest_ref)
            started = time.monotonic()
            result = self._transport._run_process(
                argv,
                cwd=scratch,
                input_text=prompt,
                timeout_seconds=self.config.timeout_seconds,
            )
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
            after = _tree_snapshot(scratch)
            ledger_ref = self.store.put_artifact(
                "codex_discovery_event_ledger", _redacted_event_ledger(result.stdout)
            )
            stream_ref = self.store.put_artifact(
                "codex_discovery_stream_manifest",
                {
                    "manifest_version": "1.0",
                    "invocation_id": invocation_id,
                    "returncode": result.returncode,
                    "elapsed_ms": elapsed_ms,
                    "stdout_bytes": len(result.stdout.encode("utf-8")),
                    "stdout_sha256": _sha256_bytes(result.stdout.encode("utf-8")),
                    "stderr_bytes": len(result.stderr.encode("utf-8")),
                    "stderr_sha256": _sha256_bytes(result.stderr.encode("utf-8")),
                    "scratch_before_hash": sha256_value(before),
                    "scratch_after_hash": sha256_value(after),
                    "scratch_changed": before != after,
                    "event_ledger": ledger_ref.model_dump(mode="json"),
                },
            )
            trace_refs.extend([ledger_ref, stream_ref])
            if result.returncode != 0:
                return self._record_terminal(
                    snapshot,
                    approved_context,
                    request_id,
                    "error",
                    "nonzero_exit",
                    trace_refs,
                )
            if before != after:
                return self._record_terminal(
                    snapshot,
                    approved_context,
                    request_id,
                    "error",
                    "scratch_changed",
                    trace_refs,
                )
            audit = _audit_jsonl(result.stdout, self.config)
            parsed = json.loads(audit.final_message)
            response = CodexProblemDiscoveryResponse.model_validate(parsed)
            if response.request_id != request_id:
                return self._record_terminal(
                    snapshot,
                    approved_context,
                    request_id,
                    "error",
                    "request_binding_failed",
                    trace_refs,
                )
            response_ref = self.store.put_artifact("codex_discovery_response", response)
            trace_refs.append(response_ref)
            if response.status == "no_result":
                return self._record_terminal(
                    snapshot,
                    approved_context,
                    request_id,
                    "no_result",
                    "model_no_result",
                    trace_refs,
                )
            assert response.draft is not None
            draft_ref = self.store.put_artifact(
                "problem_hypothesis_draft", response.draft
            )
            return self._record_proposed(
                snapshot, approved_context, request_id, response.draft, draft_ref, trace_refs
            )
        except CodexCLIError as exc:
            return self._record_terminal(
                snapshot,
                approved_context,
                request_id,
                "error",
                exc.code.value,
                trace_refs,
            )
        except subprocess.TimeoutExpired:
            return self._record_terminal(
                snapshot,
                approved_context,
                request_id,
                "error",
                "timeout",
                trace_refs,
            )
        except ProcessOutputLimitExceeded:
            return self._record_terminal(
                snapshot,
                approved_context,
                request_id,
                "error",
                "output_limit_exceeded",
                trace_refs,
            )
        except PermissionError:
            return self._record_terminal(
                snapshot,
                approved_context,
                request_id,
                "error",
                "policy_violation",
                trace_refs,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return self._record_terminal(
                snapshot,
                approved_context,
                request_id,
                "error",
                "output_schema_or_transport_error",
                trace_refs,
            )

    def _build_prompt(self, context: ProblemDiscoveryContext, request_id: str) -> str:
        input_payload = {
            "protocol_version": DISCOVERY_PROTOCOL_VERSION,
            "operation": "propose_problem_hypothesis_draft",
            "request_id": request_id,
            "mission_spec_hash": context.mission_spec_hash,
            "mission_summary": context.mission_summary,
            "evidence": context.evidence,
            "limits": {
                "max_drafts": 1,
                "tools_allowed": False,
                "execution_allowed": False,
                "admission_allowed": False,
            },
        }
        return (
            _PROMPT_PATH.read_text(encoding="utf-8").rstrip()
            + "\n\nINPUT_JSON\n"
            + canonical_json(input_payload)
            + "\n"
        )

    def _record_proposed(
        self,
        snapshot: EvidenceSnapshot,
        context: ProblemDiscoveryContext,
        request_id: str,
        draft: ProblemHypothesisDraft,
        draft_ref: DiscoveryArtifactRef,
        trace_refs: list[DiscoveryArtifactRef],
    ) -> ProblemDiscoveryProposal:
        observation = DiscoveryProviderObservation.seal(
            observation_id=f"codex_discovery_{request_id[:12]}",
            run_id=self.store.run_id,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            status="proposed",
            mission_spec_hash=context.mission_spec_hash,
            evidence_snapshot_hash=snapshot.snapshot_hash,
            context_hash=sha256_value(context.model_dump(mode="json")),
            request_id=request_id,
            draft_ref=draft_ref,
            trace_refs=trace_refs,
        )
        observation_ref = self.store.record_provider_observation(observation)
        return ProblemDiscoveryProposal(
            status="proposed",
            draft=draft,
            provider_observation_ref=observation_ref,
        )

    def _record_terminal(
        self,
        snapshot: EvidenceSnapshot,
        context: ProblemDiscoveryContext,
        request_id: str,
        status: Literal["no_result", "error"],
        terminal_code: str,
        trace_refs: list[DiscoveryArtifactRef],
    ) -> ProblemDiscoveryProposal:
        terminal_ref = self.store.put_artifact(
            "codex_discovery_terminal_receipt",
            {
                "receipt_version": "1.0",
                "provider_id": self.provider_id,
                "request_id": request_id,
                "status": status,
                "terminal_code": terminal_code,
            },
        )
        observation = DiscoveryProviderObservation.seal(
            observation_id=f"codex_discovery_{request_id[:12]}",
            run_id=self.store.run_id,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            status=status,
            mission_spec_hash=context.mission_spec_hash,
            evidence_snapshot_hash=snapshot.snapshot_hash,
            context_hash=sha256_value(context.model_dump(mode="json")),
            request_id=request_id,
            terminal_code=terminal_code,
            trace_refs=[*trace_refs, terminal_ref],
        )
        observation_ref = self.store.record_provider_observation(observation)
        return ProblemDiscoveryProposal(
            status=status,
            provider_observation_ref=observation_ref,
            terminal_code=terminal_code,
        )
