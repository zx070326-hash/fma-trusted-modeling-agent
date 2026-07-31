"""A small, open-ended mathematical-modeling agent harness."""

from .engine import ModelingEngine, SolveResult
from .model import CodexCLIModel, NativeCodexResearcher, ScriptedModel

__all__ = [
    "CodexCLIModel",
    "ModelingEngine",
    "NativeCodexResearcher",
    "ScriptedModel",
    "SolveResult",
]
__version__ = "0.4.0"
