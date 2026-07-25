from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

from fma.v5_2.ode_system import ODEThresholdsV52
from fma.v5_5.campaign_protocol import ProspectiveCampaignProtocolV55
from fma.v5_5.split_custody import SourceProvenanceRecordV55
from fma.v5_5.world_bank_custodian import WorldBankSelectionSpecV55
from fma.v5_6.hybrid_ode import HybridODEThresholdsV56
from fma.v5_6.public_hybrid_campaign import HybridCampaignProtocolV56
from fma.v5_6.unseen_source import (
    UnseenSourceRegistryV56,
    world_bank_source_identity_hash_v56,
)


ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ROOT / "experiments" / "iteration_35" / "scientific"
I34_PROVENANCE = (
    ROOT
    / "experiments"
    / "iteration_34"
    / "provenance_release_v55"
    / "source_provenance_record_v55.json"
)


def test_i35_protocol_freeze_binds_code_thresholds_and_i34_exclusion() -> None:
    registry = UnseenSourceRegistryV56.model_validate_json(
        (SCIENCE / "UNSEEN_SOURCE_REGISTRY_V56.json").read_text(
            encoding="utf-8"
        )
    )
    registry.assert_sealed()
    assert registry.fixture_only is False
    assert registry.required_prior_campaign_ids == ["i34"]

    i34 = SourceProvenanceRecordV55.model_validate_json(
        I34_PROVENANCE.read_text(encoding="utf-8")
    )
    i34.assert_sealed()
    locator = urllib.parse.urlsplit(i34.source_locator)
    parts = locator.path.strip("/").split("/")
    country = parts[parts.index("country") + 1]
    indicator = parts[parts.index("indicator") + 1]
    exclusion = registry.exclusions[0]
    assert exclusion.campaign_id == "i34"
    assert exclusion.source_identity_hash == world_bank_source_identity_hash_v56(
        api_base="https://api.worldbank.org/v2",
        country_code=country,
        indicator_code=indicator,
        period_start=int(i34.public_period_start),
        period_end=int(i34.private_period_end),
    )
    assert exclusion.source_artifact_sha256 == i34.source_artifact_sha256
    assert exclusion.source_provenance_record_hash == i34.record_hash

    v55 = ProspectiveCampaignProtocolV55.model_validate_json(
        (SCIENCE / "PROSPECTIVE_CAMPAIGN_PROTOCOL_V55.json").read_text(
            encoding="utf-8"
        )
    )
    v55.assert_sealed()
    v52 = ODEThresholdsV52.model_validate_json(
        (SCIENCE / "ODE_THRESHOLDS_V52.json").read_text(encoding="utf-8")
    )
    v52.assert_sealed()
    selection = WorldBankSelectionSpecV55.model_validate_json(
        (SCIENCE / "SOURCE_SELECTION_SPEC_V55.json").read_text(
            encoding="utf-8"
        )
    )
    selection.assert_sealed()
    assert selection.protocol_hash == v55.protocol_hash
    assert selection.ode_threshold_hash == v52.threshold_hash
    assert selection.prior_campaign_exclusion_hashes == (
        registry.exclusion_hashes()
    )
    assert selection.fixture_only is False

    hybrid = HybridODEThresholdsV56.model_validate_json(
        (SCIENCE / "HYBRID_THRESHOLDS_V56.json").read_text(
            encoding="utf-8"
        )
    )
    hybrid.assert_sealed()
    raw_hybrid = json.loads(
        (ROOT / "V5_6_HYBRID_THRESHOLDS.json").read_text(encoding="utf-8")
    )
    assert HybridODEThresholdsV56.seal(**raw_hybrid) == hybrid

    protocol = HybridCampaignProtocolV56.model_validate_json(
        (SCIENCE / "HYBRID_CAMPAIGN_PROTOCOL_V56.json").read_text(
            encoding="utf-8"
        )
    )
    protocol.assert_sealed()
    assert protocol.v55_protocol_hash == v55.protocol_hash
    assert protocol.source_registry_hash == registry.registry_hash
    assert protocol.hybrid_threshold_hash == hybrid.threshold_hash
    package = ROOT / "fma" / "v5_6"
    assert protocol.hybrid_adapter_source_sha256 == hashlib.sha256(
        (package / "hybrid_ode.py").read_bytes()
    ).hexdigest()
    assert protocol.unseen_source_adapter_source_sha256 == hashlib.sha256(
        (package / "unseen_source.py").read_bytes()
    ).hexdigest()
    assert protocol.public_runner_source_sha256 == hashlib.sha256(
        (package / "public_hybrid_campaign.py").read_bytes()
    ).hexdigest()
    assert protocol.post_result_threshold_change_allowed is False
    assert protocol.post_result_candidate_change_allowed is False
