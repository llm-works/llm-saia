# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial implementation of llm-saia
- Core types: `VerbResult`, `VerifyResult`, `Critique`, `Evidence`
- Backend protocol: `SAIABackend` abstract base class
- Anthropic backend with structured output via tool_use
- OpenClaw backend for multi-model support (Claude, OpenRouter, Ollama)
- LLM verbs:
  - `ask` - Query an artifact with a question
  - `constrain` - Parse response into structured schema
  - `verify` - Check if artifact satisfies predicate
  - `critique` - Generate strongest counter-argument
  - `refine` - Improve artifact based on feedback
  - `synthesize` - Combine multiple artifacts into structured output
  - `ground` - Anchor artifact against sources for evidence
  - `decompose` - Break down task into subtasks
- Memory verbs:
  - `store` - Save value to in-memory storage
  - `recall` - Retrieve values by key substring match
- Main `SAIA` class wiring all verbs together
- Example script demonstrating investigation workflow
- Unit tests with 92% coverage
