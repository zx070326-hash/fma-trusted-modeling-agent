from __future__ import annotations

from pathlib import Path

import pytest

from fma.codex_wsl import WslCodexRuntimeV65, windows_path_to_wsl_v65


pytestmark = pytest.mark.skipif(
    __import__("os").name != "nt",
    reason="WSL transport is Windows-specific",
)


def test_windows_path_translation_is_absolute_and_drive_bound(
    tmp_path: Path,
) -> None:
    translated = windows_path_to_wsl_v65(tmp_path / "nested" / "item.json")

    assert translated.startswith(f"/mnt/{tmp_path.drive[0].lower()}/")
    assert translated.endswith("/nested/item.json")
    with pytest.raises(ValueError, match="local drive"):
        windows_path_to_wsl_v65(r"\\server\share\item.json")


def test_wsl_runtime_stages_exact_binary_and_translates_driver_paths(
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / "source-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"auth_mode":"test"}\n',
        encoding="utf-8",
    )
    source = tmp_path / "codex"
    source.write_bytes(b"#!/bin/sh\nexit 0\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    schema = scratch / "schema.json"
    schema.write_text("{}\n", encoding="utf-8")

    with WslCodexRuntimeV65(source_codex_home=codex_home) as runtime:
        assert runtime.locate(source) == source.resolve()
        outer = runtime._outer_argv(
            [
                str(source),
                "-C",
                str(scratch),
                "exec",
                "--output-schema",
                str(schema),
                "-",
            ]
        )
        staged = next(
            item for item in outer if item.startswith("/mnt/") and "codex-" in item
        )

        assert outer[0].lower().endswith("wsl.exe")
        assert "env" in outer
        assert "-i" in outer
        assert any(item.startswith("CODEX_HOME=/mnt/") for item in outer)
        assert windows_path_to_wsl_v65(scratch) in outer
        assert windows_path_to_wsl_v65(schema) in outer
        assert staged != windows_path_to_wsl_v65(source)
