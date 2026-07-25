from __future__ import annotations

from pathlib import Path

import pytest

from fma.hashing import canonical_json
from fma.v5_5.campaign_keys import generate_local_custody_material_v55


def test_local_campaign_keys_are_distinct_create_once_and_nonqualifying(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "custody"
    summary = generate_local_custody_material_v55(output_dir=output_dir)
    assert len((output_dir / "selection_seed.bin").read_bytes()) == 32
    target_key = (output_dir / "private_target_aes256.key").read_bytes()
    provenance_key = (
        output_dir / "source_provenance_aes256.key"
    ).read_bytes()
    assert len(target_key) == len(provenance_key) == 32
    assert target_key != provenance_key
    assert summary.external_host_established is False
    assert summary.independent_management_key_control_established is False
    assert summary.scientific_qualification_granted is False
    rendered = canonical_json(summary)
    assert target_key.hex() not in rendered
    assert provenance_key.hex() not in rendered

    with pytest.raises(FileExistsError, match="already exists"):
        generate_local_custody_material_v55(output_dir=output_dir)
