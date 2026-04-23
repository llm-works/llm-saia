# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ChatResponse.model` — resolved model name returned by the backend. Populated when a backend
  resolves a placeholder like `"auto"` to a concrete model; closes the cost-attribution gap where
  downstream consumers had no way to know which model actually ran.
- `ChatResponse.raw` — escape hatch carrying the unmodified backend-native response object.
  Consumers reading from it accept coupling to a specific backend in exchange for access to
  vendor-specific fields (cache tokens, thinking/reasoning blocks, adapter metadata, etc.) that
  SAIA intentionally does not surface at the language-layer level.
- `AsyncConversationLike` protocol for non-blocking conversation append. Extends `ConversationLike`
  with `append_async()` method. All verbs use `append_async()` when the conversation supports it,
  allowing compaction strategies that involve I/O (e.g., LLM-based summarization) to run without
  blocking the event loop.
- `Complete` verb now accepts optional `conversation: ConversationLike` parameter. When provided,
  messages are appended to both an internal history (returned in `TaskResult.history`) and the
  external conversation. The LLM sees `conversation.as_messages()` which may be compacted,
  enabling long-running tool loops without unbounded context growth. `TaskResult.history` always
  contains the complete uncompacted history.
- `IterationContext` passed to `IterationGuard` validators, providing access to `response`,
  `iteration`, `max_iterations`, and `remaining` property. Enables guards that adapt behavior
  based on loop progress (e.g., force terminal tool when iterations are running low).
- `require_confirmation` parameter for terminal tools. Set to `False` to complete immediately
  on first terminal tool call without requiring a confirmation call. Many models respond to
  confirmation prompts with text instead of a tool call, causing `terminal_data` to be `None`.
  Use `.terminal_tool("name", require_confirmation=False)` to avoid this issue.
- `with_tracer()` fluent API for per-call tracer override (consistent with other `with_*` methods)
- Built-in iteration guards in `llm_saia.guards`:
  - `terminal_status(tool, status_field, failure_values)` - reject terminal calls with failure status
  - `terminal_schema(tools, terminal_tool)` - validate terminal args against JSON schema
  - `contradiction(terminal_tool)` - detect hedging language when terminal tool is called
- `IterationContext.parse_error` field for detecting parse retry context. When set, the guard is
  running in parse retry mode (structured output failed to parse).
- `IterationGuard.parse_max_retries` field. Guards with `parse_max_retries > 0` participate in
  parse retry (retrying when structured output JSON parsing fails). When multiple guards have
  `parse_max_retries > 0`, their retry budgets are summed (e.g., two guards with 2 retries each
  allow up to 5 attempts). All participating guards are evaluated on each attempt; their feedback
  is combined.
- Built-in `schema_retry(max_retries=2)` guard - retry with escalating feedback when JSON
  parsing fails. Migration from `with_parse_retries(n)`: use `.with_guard(schema_retry(n))`
- **Trace-level observability** for debugging stuck loops and agent behavior. Enable trace logging
  to see detailed execution flow:
  - Tool results returned to LLM (up to 50k chars per item)
  - Guard triggers with feedback content injected into conversation
  - Message assembly showing what's sent to LLM each iteration (counts by role, last user
    message, recent tool results)
  - Verb lifecycle (start, completion with duration and step count)
  - Controller decisions with terminal tool data

### Changed
- **BREAKING**: Renamed `AgentResponse` → `ChatResponse` to match the `Backend.chat()` method
  name and align with `LLMCall` already used in `trace.py`. The old name is removed; update
  imports and type hints to `ChatResponse`.
- **BREAKING**: `IterationGuard.validator` signature changed from `Callable[[ChatResponse], str | None]`
  to `Callable[[IterationContext], str | None]`. Access response via `ctx.response`.
- **BREAKING**: `lg: Logger` is now the first field in `Config` (was after `backend`). This
  follows the project convention that logger is always the first parameter.
- Logger is now required in `Config` (defaults to `NullLogger()` via builder). Removed all
  internal `if self._lg:` checks.

### Removed
- **BREAKING**: `retries()` / `with_retries()` removed. Terminal failure retry behavior
  should be implemented via `IterationGuard` instead.
- **BREAKING**: `parse_retries()` / `with_parse_retries()` removed. Use the built-in
  `schema_retry()` guard instead: `.with_guard(schema_retry(2))`
- **BREAKING**: `CallOptions.max_retries`, `CallOptions.retry_escalation`,
  `CallOptions.parse_retries` removed.
- Controller no longer handles terminal failure retries or confirmation contradiction
  detection internally. These behaviors can be implemented via guards for opt-in use.

## [0.3.0] - 2026-04-13

### Added
- `IterationGuard` for per-iteration constraints in tool loops (runs after each LLM response,
  unlike `OutputGuard` which validates the final result). Added via `with_guard()`/`with_guards()`.
- Tree-structured tracing with `VerbTrace`/`Step` hierarchy and derived aggregates
  (`total_llm_calls`, `parse_retries`, `guard_retries`, `total_tokens`)
- Escalating guard retries: `OutputGuard.retry_instruction` now accepts
  `str | Callable[[int, Any, str], str]` for dynamic, attempt-aware retry instructions.
  `resolve_instruction(attempt, result, error)` dispatches between static and callable forms.
- `escalate=True` parameter on all built-in guards (`max_length`, `english_only`, `no_markdown`,
  `no_preamble`, `no_emoji`, `ascii_only`) for increasingly forceful retry instructions.
  `max_length(escalate=True)` includes current char count in each retry.
- `with_tools(tools, executor=)` fluent API for per-call tool override (enables benchmarking
  scenarios like BFCL where each test case defines its own function schemas)
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
- **BREAKING**: All 12 simple verbs now return `VerbResult[T]` instead of bare `T`. Access the
  value via `.value` and the execution trace via `.trace`. `Complete` still returns `TaskResult`
  but now also carries `.trace`. `VerbTrace` is always populated (never None).
- **BREAKING**: `TaskResult.trace_id` and `TaskResult.request_id` removed — now accessed via
  `result.trace.trace_id` and `result.trace.request_id`.
- `VerbTrace.to_dict()` and `VerbTrace.to_json()` for serialization.
- `Tracer.write()` now accepts `Step | VerbTrace` records (consumers dispatch on `record["type"]`)
- All internal imports converted from absolute (`from llm_saia.core.X`) to relative (`from .X`)
  to avoid resolving against an installed package instead of the local development source
- Guard revalidation capped at 10 rounds to prevent infinite loops
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

[Unreleased]: https://github.com/llm-works/llm-saia/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/llm-works/llm-saia/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/llm-works/llm-saia/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/llm-works/llm-saia/releases/tag/v0.1.0
