# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ConversationLike` protocol for pluggable conversation management (compaction, persistence)
- `ListConversation` default implementation of `ConversationLike`
- `Role` enum for message roles (`USER`, `ASSISTANT`, `SYSTEM`, `TOOL`)
- `core/conversation.py` module for conversation/message types
- All 13 verbs accept optional `conversation` keyword argument for external conversation management
- Direct-call paths (no tools) also track messages through the conversation object
- Guard retries thread conversation (retry exchanges visible in caller's conversation)
- Parse retries isolate failed attempts from caller's conversation (only final exchange propagated)
- `find` verb - filter items matching criteria, returns `FindResult(indices, reason)`

### Changed
- All internal imports converted from absolute (`from llm_saia.core.X`) to relative (`from .X`)
  to avoid resolving against an installed package instead of the local development source
- Extracted `OutputGuardMixin` (`core/guards.py`) and `VerbLoggingMixin` (`core/logging.py`) from
  `core/verb.py` to keep the base class manageable
- Guard revalidation loops now have a convergence cap (`_MAX_REVALIDATION_ROUNDS=10`) to prevent
  infinite loops when guard retries keep producing results that fail other guards
- Narrowed `except Exception` to `except TypeError` in field guard extraction (`get_type_hints`)
- Truncation heuristic for structured output errors now also checks `JSONDecodeError.pos` to verify
  the error is actually at/near EOF before classifying as truncated
- **BREAKING**: Message role for tool results changed from `"tool_result"` to `"tool"` (aligns with OpenAI convention; tool calls remain in assistant messages via `tool_calls` field)
- Moved `Message`, `ToolCall` from `backend.py` to new `conversation.py` module
- `llm_saia.core` now re-exports `AgentResponse`, `Message`, `ToolCall`, `ToolDef` (stable public API for downstream consumers)
- Schema support for `Literal[...]` types (maps to JSON enum)
- Schema support for `Enum` types (maps to JSON enum)
- Schema support for nested dataclasses (recursive conversion)
- Schema support for `list[MyDataclass]` (recursive parsing)
- Parse retry with feedback - retry on `StructuredOutputError` with LLM feedback (opt-in)
- `with_parse_retries(n)` fluent API for enabling retry attempts (default: 0 = disabled)
- Output guards - validators with automatic retry for both text and structured output
- `with_guard()` fluent API for adding output guards (chainable)
- `with_guards(*guards)` for adding multiple guards at once
- `Guarded` class for field-level guards via `Annotated[str, Guarded(guard1, guard2)]`
- Pre-built guards: `english_only()`, `max_length()`, `no_emoji()`, `no_markdown()`, `no_preamble()`, `ascii_only()`
- `OutputGuard` dataclass for custom validators with retry instructions
- `OutputGuardError` raised when all guard retries exhausted

### Fixed
- Schema generation now raises clear `TypeError` for recursive dataclasses
- Schema generation validates Literal values are same type (prevents invalid JSON schema)
- Parsing raises `TypeError` for type mismatches instead of silent coercion
- `max_call_tokens` now respected in direct structured output completion

## [0.2.0] - 2026-03-16

### Added
- `temperature` parameter to `Backend.chat()` protocol for sampling variance control
- `CallOptions` dataclass for all per-call configuration (replaces `RunConfig`)
- `Configurable` interface providing fluent `with_*()` methods
- `with_temperature()` method for per-call temperature override
- `with_system()` method for per-call system prompt override
- `.temperature()` method on `SAIABuilder` for fluent configuration

### Changed
- **BREAKING**: Renamed `RunConfig` → `CallOptions`
- **BREAKING**: Renamed `Config.run` → `Config.call`
- **BREAKING**: Moved `system`, `temperature`, `request_id` from `Config` to `CallOptions`
- **BREAKING**: Renamed `saia.run_config` → `saia.call_options`
- **BREAKING**: Renamed `with_run_config()` → `with_call_options()`
- **BREAKING**: Renamed `with_timeout_secs()` → `with_timeout()`
- `SAIA` now inherits from `Configurable` interface

## [0.1.0] - 2026-02-25

### Added
- Core verb vocabulary for LLM interactions:
  - `ask` - Query an artifact with a question
  - `verify` - Check if artifact satisfies predicate (returns `VerifyResult`)
  - `critique` - Generate strongest counter-argument (returns `Critique`)
  - `refine` - Improve artifact based on feedback
  - `synthesize` - Combine multiple artifacts into one
  - `decompose` - Break complex task into subtasks
  - `extract` - Pull structured data from text
  - `classify` - Categorize into predefined classes (returns `ClassifyResult`)
  - `choose` - Select best option from choices (returns `ChooseResult`)
  - `constrain` - Parse into structured schema
  - `ground` - Anchor claims to source evidence (returns `Evidence`)
  - `instruct` - Execute open-ended instructions
- Memory verbs: `store` and `recall` for session-scoped memory
- `complete()` verb for tool-calling loops with terminal detection
- Builder pattern configuration via `SAIA.builder()`
- Runtime modifiers: `with_single_call()`, `with_max_iterations()`, `with_timeout_secs()`,
  `with_request_id()`
- Protocol-based `Backend` abstraction for LLM providers
- Structured output parsing with dataclass schemas
- Tracing infrastructure: `Tracer`, `CallbackTracer`, `TracerFactory`
- Custom exception hierarchy: `Error`, `BackendError`, `ConfigurationError`,
  `StructuredOutputError`, `ToolExecutionError`, `TruncatedResponseError`
- `compose()` utility for chaining verb operations
- Iteration trace infrastructure with type-safe decision reasons
- PEP 561 `py.typed` marker for type checker support
- Examples: `investigate.py`, `build.py`, `build_multi.py`, `agent.py`, `analyze.py`

### Changed
- Python 3.11+ required
- mypy strict mode compliance
- 93% test coverage
- CI/CD with GitHub Actions (lint, test, coverage, release)

[Unreleased]: https://github.com/llm-works/llm-saia/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/llm-works/llm-saia/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/llm-works/llm-saia/releases/tag/v0.1.0
