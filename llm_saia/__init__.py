"""SAIA: Framework-agnostic verb vocabulary for LLM agents."""

from llm_saia.core.backend import SAIABackend
from llm_saia.core.logger import NullLogger, SAIALogger
from llm_saia.core.types import (
    AgentResponse,
    ChooseResult,
    ClassifyResult,
    ConfirmResult,
    Critique,
    Evidence,
    Message,
    RunConfig,
    TaskResult,
    ToolCall,
    ToolDef,
    VerbConfig,
    VerbResult,
    VerifyResult,
)
from llm_saia.core.verb import Verb
from llm_saia.saia import SAIA

__all__ = [
    # Main class
    "SAIA",
    "SAIABackend",
    # Custom verbs
    "Verb",
    "VerbConfig",
    # Logger
    "NullLogger",
    "SAIALogger",
    # Verb results
    "ChooseResult",
    "ClassifyResult",
    "ConfirmResult",
    "Critique",
    "Evidence",
    "VerifyResult",
    "VerbResult",
    # Task types
    "AgentResponse",
    "Message",
    "RunConfig",
    "TaskResult",
    "ToolCall",
    "ToolDef",
]
