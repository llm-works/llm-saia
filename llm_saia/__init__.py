"""SAIA: Framework-agnostic verb vocabulary for LLM agents."""

from llm_saia.builder import SAIABuilder
from llm_saia.core.backend import AgentResponse, Backend, Message, ToolCall, ToolDef
from llm_saia.core.config import Config, RunConfig, TerminalConfig
from llm_saia.core.errors import (
    BackendError,
    ConfigurationError,
    Error,
    StructuredOutputError,
    ToolExecutionError,
    TruncatedResponseError,
)
from llm_saia.core.logger import Logger, NullLogger
from llm_saia.core.trace import CallbackTracer, Tracer, TracerFactory
from llm_saia.core.types import (
    ChooseResult,
    ClassifyResult,
    Critique,
    DecisionReason,
    Evidence,
    LoopScore,
    TaskResult,
    VerbResult,
    VerifyResult,
)
from llm_saia.core.verb import Verb
from llm_saia.saia import SAIA

__all__ = [
    # Main class
    "SAIA",
    "SAIABuilder",
    "Backend",
    # Custom verbs
    "Verb",
    "Config",
    # Errors
    "Error",
    "BackendError",
    "ConfigurationError",
    "StructuredOutputError",
    "ToolExecutionError",
    "TruncatedResponseError",
    # Logger
    "NullLogger",
    "Logger",
    # Verb results
    "ChooseResult",
    "ClassifyResult",
    "Critique",
    "Evidence",
    "VerifyResult",
    "VerbResult",
    # Task types
    "AgentResponse",
    "DecisionReason",
    "LoopScore",
    "Message",
    "RunConfig",
    "TaskResult",
    "ToolCall",
    "ToolDef",
    # Terminal
    "TerminalConfig",
    # Tracing
    "CallbackTracer",
    "Tracer",
    "TracerFactory",
]
