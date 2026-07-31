"""A small, open-ended mathematical-modeling agent harness."""

from .loop import ModelingLoop, RunResult
from .model import CodexCLIModel, NativeCodexResearcher, ScriptedModel
from .sidecar import NativeSidecar

__all__ = [
    "CodexCLIModel",
    "ModelingLoop",
    "NativeCodexResearcher",
    "NativeSidecar",
    "RunResult",
    "ScriptedModel",
]
__version__ = "0.3.0"
