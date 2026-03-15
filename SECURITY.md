# Security Policy

## Reporting Security Vulnerabilities

**We take security seriously.** If you discover a security vulnerability in SAIA, please report it
responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting:
1. Go to the repository's **Security** tab
2. Click **"Report a vulnerability"**
3. Submit your confidential report

### What to Include

Please include the following information in your report:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** assessment
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up

### Response Timeline

- **Initial Response:** Within 48 hours
- **Triage & Assessment:** Within 7 days
- **Fix & Disclosure:** Coordinated with reporter (typically 30-90 days)

### Recognition

We maintain a security hall of fame for responsible disclosure. Contributors will be credited
(unless they prefer to remain anonymous).

---

## Supported Versions

| Version | Supported          | End of Support |
| ------- | ------------------ | -------------- |
| 0.x     | :white_check_mark: | TBD            |

**Note:** This is a pre-1.0 release. Security patches will be provided for the latest 0.x version
only.

---

## Security Considerations

SAIA is a framework for building LLM agents. Security in LLM applications involves multiple layers,
and SAIA provides protections at the framework level while leaving application-level security to
developers.

### 1. Structured Output Parsing

**Protection against malformed LLM responses.**

SAIA uses dataclass schemas with strict JSON parsing:

```python
from llm_saia import SAIA, VerifyResult

# Structured output with type-safe parsing
result = await saia.verify(artifact, predicate)
# Returns VerifyResult(passed: bool, reason: str) - guaranteed structure
```

- JSON schema validation before parsing
- Truncation detection (`TruncatedResponseError`) when max_tokens is too low
- Type-safe dataclass results prevent string parsing errors

### 2. Tool Execution

**Protection against unsafe tool invocations.**

When using `complete()` with tools:

```python
saia = (
    SAIA.builder()
    .backend(backend)
    .tools(tool_defs, executor)  # You control the executor
    .build()
)
```

**Your responsibilities:**
- Validate tool arguments before execution
- Implement appropriate sandboxing for dangerous operations
- Rate-limit expensive tool calls
- Log tool executions for audit trails

**SAIA provides:**
- Max iteration limits to prevent infinite loops
- Timeout mechanisms to prevent runaway executions
- Tracing infrastructure for debugging and auditing
- Terminal detection to stop when task is complete

### 3. Backend Credentials

**Best practices for LLM API credentials.**

SAIA does not handle credentials - backends manage their own authentication:

```python
# Backend handles credentials via environment
import os
from your_backend import AnthropicBackend

backend = AnthropicBackend(api_key=os.environ["ANTHROPIC_API_KEY"])
```

**Never:**
- Hardcode API keys in source code
- Commit credentials to version control
- Log API keys or tokens

### 4. Prompt Injection

**Understanding the threat model.**

Prompt injection is an **application-level concern**, not a framework concern. SAIA provides the
building blocks; your application must handle untrusted input appropriately.

**Mitigations you should implement:**
- Validate and sanitize user input before including in prompts
- Use structured verbs (`verify`, `extract`) instead of open-ended `instruct` for untrusted input
- Implement output validation for critical decisions
- Consider multi-agent verification for high-stakes operations

```python
# Example: Using verify() to check LLM output before acting
result = await saia.verify(
    llm_output,
    "contains no harmful instructions and follows expected format"
)
if not result.passed:
    raise ValueError(f"Output validation failed: {result.reason}")
```

### 5. Resource Limits

**Protection against resource exhaustion.**

SAIA provides configurable limits:

```python
saia = (
    SAIA.builder()
    .backend(backend)
    .max_iterations(10)       # Limit tool loop iterations
    .max_call_tokens(4096)    # Limit tokens per LLM call
    .build()
)

# Runtime limits
result = await saia.with_timeout(30).instruct(task)
result = await saia.with_max_iterations(5).complete(prompt, tools)
```

---

## Security Checklist

Use this checklist when deploying SAIA-based applications:

### Development

- [ ] Store API keys in environment variables, not code
- [ ] Implement input validation for user-provided prompts
- [ ] Add output validation for critical operations
- [ ] Set appropriate iteration and timeout limits
- [ ] Enable tracing for debugging and auditing

### Deployment

- [ ] Use secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
- [ ] Configure appropriate resource limits
- [ ] Set up monitoring for unusual patterns (high iteration counts, timeouts)
- [ ] Implement rate limiting at the application level
- [ ] Review tool executor implementations for security

### Tool Security

If your application uses `complete()` with tools:

- [ ] Validate all tool arguments before execution
- [ ] Implement least-privilege for tool capabilities
- [ ] Sandbox dangerous operations (file system, network, shell)
- [ ] Log all tool executions with arguments
- [ ] Rate-limit expensive or dangerous tools

---

## Threat Model

### In Scope

SAIA protects against:

- **Malformed responses** - Strict JSON parsing with schema validation
- **Runaway execution** - Iteration limits and timeouts
- **Truncated output** - Detection and clear error messages
- **Type confusion** - Dataclass-based results with static typing

### Out of Scope

SAIA **does not** protect against:

- **Prompt injection** - Application responsibility to validate input
- **Credential leakage** - Backend implementations manage credentials
- **Tool misuse** - Your executor must validate and sandbox tools
- **LLM hallucination** - Use `verify()` and `ground()` to check outputs
- **Network attacks** - Backend implementations handle transport security

### Assumptions

SAIA assumes:

- **Trusted backends** - Backend implementations are secure
- **Trusted tools** - Tool executors validate their inputs
- **Developer awareness** - You understand LLM security risks
- **Proper configuration** - Appropriate limits are set

---

## Dependency Security

### Runtime Dependencies

**SAIA's core library has zero runtime dependencies.** It is pure Python with no external packages
required.

The `Backend` protocol allows you to bring your own LLM client implementation. This design:
- Minimizes supply chain attack surface
- Lets you control which LLM SDK versions you use
- Avoids dependency conflicts in your project

### Development Dependencies

Development and testing require additional packages (pytest, mypy, ruff, etc.) which are not
installed when you `pip install llm-saia`.

### Example Code

The `examples/` directory is **not included in the published package**. Examples in the repository
use `httpx` and `llm-infer` for demonstration backends, but these are not dependencies of the
library itself.

---

## Contact

For security-related questions or concerns:

- **Security Issues:** Use GitHub's private vulnerability reporting
- **General Questions:** GitHub Discussions
- **Documentation:** See README.md and examples/

---

**Last Updated:** 2026-02-20
