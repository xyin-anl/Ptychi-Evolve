"""
Algorithm Discovery Framework.

A streamlined framework for discovering novel regularization algorithms
for ptychographic reconstruction using LLMs.
"""

from .discovery import AlgorithmDiscovery
from .llm_engine import LLMEngine
from .recon_evaluator import ReconEvaluator
from .history import DiscoveryHistory
from .exceptions import (
    PtychiEvolveError,
    GenerationError,
    EvaluationError,
    ConfigurationError,
    PromptError,
    SecurityError,
    VLMError,
    CompressionError,
)

__version__ = "0.0.1"
__all__ = [
    "AlgorithmDiscovery",
    "LLMEngine",
    "ReconEvaluator",
    "DiscoveryHistory",
    "PtychiEvolveError",
    "GenerationError",
    "EvaluationError",
    "ConfigurationError",
    "PromptError",
    "SecurityError",
    "VLMError",
    "CompressionError",
]
