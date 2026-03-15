# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Runtime modifiers: `with_single_call()`, `with_max_iterations()`, `with_timeout()`,
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

[Unreleased]: https://github.com/llm-works/llm-saia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/llm-works/llm-saia/releases/tag/v0.1.0
