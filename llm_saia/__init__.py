# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-saia Authors
#
# Maintained by LLM Works LLC (https://llm-works.ai) and contributors.

"""SAIA: Framework-agnostic verb vocabulary for LLM agents."""

from .builder import SAIABuilder
from .core.backend import Backend, ChatResponse, ToolDef
from .core.config import (
    CallOptions,
    Config,
    JsonParser,
    TerminalConfig,
    ToolGate,
    ToolGateContext,
    ToolGateContextFactory,
)
from .core.conversation import Message, Role, ToolCall
from .core.errors import (
    BackendError,
    ConfigurationError,
    Error,
    PauseRequested,
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
    ConversationFactory,
    ConversationLike,
    Critique,
    DecisionReason,
    Evidence,
    FindResult,
    ListConversation,
    LoopScore,
    SerializableConversationLike,
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
    "JsonParser",
    # Errors
    "Error",
    "BackendError",
    "ConfigurationError",
    "OutputGuardError",
    "PauseRequested",
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
    "ChatResponse",
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
    "ConversationFactory",
    "ConversationLike",
    "ListConversation",
    "SerializableConversationLike",
    # Terminal
    "TerminalConfig",
    # Tool visibility gates
    "ToolGate",
    "ToolGateContext",
    "ToolGateContextFactory",
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
