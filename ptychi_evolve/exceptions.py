"""
Custom exceptions for the ptychi-evolve framework.
"""


class PtychiEvolveError(Exception):
    """Base exception for all ptychi-evolve errors."""

    pass


class GenerationError(PtychiEvolveError):
    """Raised when algorithm generation fails."""

    pass


class EvaluationError(PtychiEvolveError):
    """Raised when algorithm evaluation fails."""

    pass


class ConfigurationError(PtychiEvolveError):
    """Raised when configuration is invalid."""

    pass


class PromptError(ConfigurationError):
    """Raised when prompt templates are missing or invalid."""

    pass


class SecurityError(PtychiEvolveError):
    """Raised when security analysis detects potentially unsafe code."""

    pass


class VLMError(PtychiEvolveError):
    """Raised when VLM operations fail."""

    pass


class CompressionError(PtychiEvolveError):
    """Raised when history compression fails."""

    pass
