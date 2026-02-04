"""Core types and protocols for SAIA."""

from llm_saia.core.backend import SAIABackend
from llm_saia.core.types import Critique, Evidence, VerbResult, VerifyResult

__all__ = [
    "Critique",
    "Evidence",
    "SAIABackend",
    "VerifyResult",
    "VerbResult",
]
