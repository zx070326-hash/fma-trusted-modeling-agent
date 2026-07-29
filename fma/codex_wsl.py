"""Windows-to-WSL transport for the Linux Codex binary bundled by the app.

The desktop package can expose an ELF ``codex`` binary through ``where.exe``
even though a normal Windows child process cannot execute it.  This adapter
stages that exact binary into a private temporary directory, gives it an
auth-only Codex home, translates the two filesystem arguments used by the FMA
driver, and launches it through ``wsl.exe``.

It is a process transport only.  It does not relax the existing tool, MCP,
sandbox, approval, output-schema, or fresh-process controls.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

from .codex_driver import (
    ProcessResult,
    _default_process_runner,
)


_PATH_ARGUMENTS = {"-C", "--output-schema"}
_WSL_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def windows_path_to_wsl_v65(value: str | Path) -> str:
    """Translate one absolute local drive path without invoking a shell."""

    path = Path(value).expanduser().resolve()
    drive = path.drive
    if not re.fullmatch(r"[A-Za-z]:", drive):
        raise ValueError("WSL Codex transport requires a local drive path")
    suffix = path.as_posix()[len(drive) :].lstrip("/")
    return f"/mnt/{drive[0].lower()}/{suffix}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WslCodexRuntimeV65:
    """Callable locator/runner pair with one private, auth-only lifetime."""

    provider = "openai_codex_cli_via_wsl"

    def __init__(
        self,
        *,
        source_codex_home: str | Path | None = None,
        wsl_executable: str | Path | None = None,
        distribution: str | None = None,
    ) -> None:
        if os.name != "nt":
            raise OSError("WSL Codex transport is available only on Windows")
        located_wsl = (
            Path(wsl_executable).expanduser()
            if wsl_executable is not None
            else Path(shutil.which("wsl.exe") or "")
        )
        if not str(located_wsl) or not located_wsl.resolve().is_file():
            raise FileNotFoundError("wsl.exe is unavailable")
        self._wsl_executable = located_wsl.resolve(strict=True)
        self._distribution = distribution
        self._source_codex_home = Path(
            source_codex_home
            or os.environ.get("CODEX_HOME")
            or (Path.home() / ".codex")
        ).expanduser()
        self._temporary = tempfile.TemporaryDirectory(
            prefix="fma-codex-wsl-"
        )
        self._root = Path(self._temporary.name)
        self._private_codex_home = self._root / "codex-home"
        self._private_codex_home.mkdir()
        auth_path = self._source_codex_home / "auth.json"
        if not auth_path.is_file():
            self.close()
            raise FileNotFoundError(
                "saved Codex auth.json is unavailable for WSL transport"
            )
        shutil.copy2(auth_path, self._private_codex_home / "auth.json")
        self._staged_by_source: dict[Path, Path] = {}

    @property
    def runtime_identity(self) -> dict[str, str | None]:
        return {
            "transport": "wsl_exec_auth_only",
            "distribution": self._distribution,
            "wsl_executable": str(self._wsl_executable),
            "wsl_executable_sha256": _sha256_file(self._wsl_executable),
        }

    def close(self) -> None:
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "WslCodexRuntimeV65":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def locate(self, explicit: str | Path | None = None) -> Path:
        """Locate a Linux Codex binary without trying to execute it as PE."""

        candidates: list[Path] = []
        if explicit is not None:
            candidates.append(Path(explicit).expanduser())
        elif os.environ.get("FMA_CODEX_BIN"):
            candidates.append(Path(os.environ["FMA_CODEX_BIN"]).expanduser())
        else:
            located = shutil.which("codex")
            if located:
                candidates.append(Path(located))
            codex_home = Path(
                os.environ.get("CODEX_HOME", Path.home() / ".codex")
            )
            release_root = codex_home / "packages" / "standalone" / "releases"
            if release_root.is_dir():
                candidates.extend(
                    sorted(
                        release_root.glob("*/bin/codex"),
                        key=lambda path: path.parent.parent.name,
                        reverse=True,
                    )
                )
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
                with resolved.open("rb") as handle:
                    prefix = handle.read(4)
            except OSError:
                continue
            if resolved.is_file() and (
                prefix == b"\x7fELF" or prefix.startswith(b"#!")
            ):
                return resolved
        raise FileNotFoundError("no Linux Codex binary is available for WSL")

    def _stage(self, source: Path) -> Path:
        source = source.resolve(strict=True)
        existing = self._staged_by_source.get(source)
        if existing is not None:
            return existing
        source_hash = _sha256_file(source)
        staged = self._root / f"codex-{source_hash}"
        shutil.copyfile(source, staged)
        if _sha256_file(staged) != source_hash:
            raise OSError("staged WSL Codex binary hash differs")
        self._staged_by_source[source] = staged
        return staged

    def _outer_argv(self, argv: list[str]) -> list[str]:
        if not argv:
            raise ValueError("Codex argv is empty")
        source = Path(argv[0]).resolve(strict=True)
        staged = self._stage(source)
        translated: list[str] = []
        translate_next = False
        for argument in argv[1:]:
            if translate_next:
                translated.append(windows_path_to_wsl_v65(argument))
                translate_next = False
                continue
            translated.append(argument)
            translate_next = argument in _PATH_ARGUMENTS
        if translate_next:
            raise ValueError("Codex path option lacks its value")

        outer = [str(self._wsl_executable)]
        if self._distribution:
            outer.extend(["--distribution", self._distribution])
        outer.extend(
            [
                "--exec",
                "env",
                "-i",
                "CODEX_HOME="
                + windows_path_to_wsl_v65(self._private_codex_home),
                "HOME=/tmp",
                f"PATH={_WSL_PATH}",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                windows_path_to_wsl_v65(staged),
                *translated,
            ]
        )
        return outer

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        input_text: str | None,
        timeout_seconds: int,
        env: dict[str, str],
        max_stdout_bytes: int,
        max_stderr_bytes: int,
    ) -> ProcessResult:
        return _default_process_runner(
            self._outer_argv(argv),
            cwd=cwd,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
            env=env,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )


__all__ = ["WslCodexRuntimeV65", "windows_path_to_wsl_v65"]
