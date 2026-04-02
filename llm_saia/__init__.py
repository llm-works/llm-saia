"""SAIA: Framework-agnostic verb vocabulary for LLM agents."""

from llm_saia.builder import SAIABuilder
from llm_saia.core.backend import AgentResponse, Backend, ToolDef
from llm_saia.core.config import CallOptions, Config, TerminalConfig
from llm_saia.core.conversation import Message, Role, ToolCall
from llm_saia.core.errors import (
    BackendError,
    ConfigurationError,
    Error,
    StructuredOutputError,
    ToolExecutionError,
    TruncatedResponseError,
)
from llm_saia.core.guard import Guarded, OutputGuard, OutputGuardError
from llm_saia.core.logger import Logger, NullLogger
from llm_saia.core.trace import CallbackTracer, Tracer, TracerFactory
from llm_saia.core.types import (
    ChooseResult,
    ClassifyResult,
    ConversationLike,
    Critique,
    DecisionReason,
    Evidence,
    FindResult,
    ListConversation,
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
    "OutputGuardError",
    "StructuredOutputError",
    "ToolExecutionError",
    "TruncatedResponseError",
    # Guards
    "Guarded",
    "OutputGuard",
    # Logger
    "NullLogger",
    "Logger",
    # Verb results
    "ChooseResult",
    "ClassifyResult",
    "Critique",
    "Evidence",
    "FindResult",
    "VerifyResult",
    "VerbResult",
    # Task types
    "AgentResponse",
    "DecisionReason",
    "LoopScore",
    "Message",
    "CallOptions",
    "Role",
    "TaskResult",
    "ToolCall",
    "ToolDef",
    # Conversation protocol
    "ConversationLike",
    "ListConversation",
    # Terminal
    "TerminalConfig",
    # Tracing
    "CallbackTracer",
    "Tracer",
    "TracerFactory",
]
