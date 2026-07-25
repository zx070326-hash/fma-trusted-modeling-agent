from __future__ import annotations

from typing import Protocol

from fma.schemas import ProblemContract

from .schemas import (
    AcceptanceCommitment,
    FrozenLegacyBinding,
    PrivateAcceptanceBundle,
    ProblemContractProposal,
)


class AcceptanceTestAuthority(Protocol):
    """Independent harness role that produces private acceptance material."""

    authority_id: str

    def issue(self, proposal: ProblemContractProposal) -> PrivateAcceptanceBundle: ...


def acceptance_commitment(bundle: PrivateAcceptanceBundle) -> AcceptanceCommitment:
    """Return a public-safe receipt without exposing executable private tests."""

    bundle.assert_sealed()
    assert bundle.acceptance_bundle_hash is not None
    return AcceptanceCommitment(
        authority_id=bundle.authority_id,
        proposal_hash=bundle.proposal_hash,
        acceptance_bundle_hash=bundle.acceptance_bundle_hash,
    )


def freeze_legacy_contract(
    proposal: ProblemContractProposal,
    acceptance_bundle: PrivateAcceptanceBundle,
) -> FrozenLegacyBinding:
    """Freeze the legacy contract only after an independent test bundle is bound.

    The proposal has no acceptance-test field.  The bridge is the sole V2.0
    path that combines the public proposal with harness-private tests before
    calling the existing, content-addressed ``ProblemContract.freeze`` API.
    """

    proposal.assert_sealed()
    acceptance_bundle.assert_sealed()
    if acceptance_bundle.proposal_hash != proposal.proposal_hash:
        raise ValueError("private acceptance bundle is bound to another proposal")
    assert proposal.proposal_hash is not None
    assert acceptance_bundle.acceptance_bundle_hash is not None
    payload = proposal.public_legacy_payload()
    payload["acceptance_tests"] = acceptance_bundle.acceptance_tests
    contract = ProblemContract.freeze(**payload)
    return FrozenLegacyBinding.seal(
        proposal_hash=proposal.proposal_hash,
        acceptance_bundle_hash=acceptance_bundle.acceptance_bundle_hash,
        contract=contract,
    )
