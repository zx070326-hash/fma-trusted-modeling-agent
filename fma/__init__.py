"""Trusted vertical slice for a mathematical-modeling agent."""

from .codex_driver import CodexCLIConfig, CodexDrivenModelingAgent
from .controller import ModelingAgent, StaticExplorer
from .schemas import OptimizationModelIR, ProblemContract

__all__ = [
    "CodexCLIConfig",
    "CodexDrivenModelingAgent",
    "ModelingAgent",
    "OptimizationModelIR",
    "ProblemContract",
    "StaticExplorer",
]

__version__ = "0.3.0"
