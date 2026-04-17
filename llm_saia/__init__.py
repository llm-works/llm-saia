"""SAIA: Framework-agnostic verb vocabulary for LLM agents."""

from .builder import SAIABuilder
from .core.backend import AgentResponse, Backend, ToolDef
from .core.config import CallOptions, Config, TerminalConfig
from .core.conversation import Message, Role, ToolCall
from .core.errors import (
    BackendError,
    ConfigurationError,
    Error,
    StructuredOutputError,
    ToolExecutionError,
    TruncatedResponseError,
)
from .core.guard import (
    UNLIMITED,
    Guarded,
    IterationContext,
    IterationGuard,
    OutputGuard,
    OutputGuardError,
)
from .core.logger import Logger, NullLogger
from .core.trace import (
    CallbackTracer,
    GuardOutcome,
    LLMCall,
    Step,
    ToolOutcome,
    Tracer,
    TracerFactory,
    VerbTrace,
)
from .core.types import (
    AsyncConversationLike,
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
from .core.verb import Verb
from .saia import SAIA

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
    "IterationContext",
    "IterationGuard",
    "OutputGuard",
    "UNLIMITED",
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
    "AsyncConversationLike",
    "ConversationLike",
    "ListConversation",
    # Terminal
    "TerminalConfig",
    # Tracing
    "CallbackTracer",
    "GuardOutcome",
    "LLMCall",
    "Step",
    "ToolOutcome",
    "Tracer",
    "TracerFactory",
    "VerbTrace",
]
