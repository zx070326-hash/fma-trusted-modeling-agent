"""A small, open-ended mathematical-modeling agent harness."""

from .engine import ModelingEngine, SolveResult
from .model import CodexCLIModel, ScriptedModel

__all__ = [
    "CodexCLIModel",
    "ModelingEngine",
    "ScriptedModel",
    "SolveResult",
]
__version__ = "0.7.0"
