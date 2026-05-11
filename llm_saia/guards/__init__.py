"""Pre-built guards for LLM output validation and iteration control.

Output guards validate final results and retry on failure:
    >>> from llm_saia.guards import english_only, max_length
    >>> result = await saia.with_guard(english_only()).summarize(article)

Parse guards retry when structured output parsing fails:
    >>> from llm_saia.guards import schema_retry
    >>> result = await saia.with_guard(schema_retry()).summarize(article)

Iteration guards run during tool loops and inject feedback:
    >>> from llm_saia.guards import terminal_status, terminal_schema
    >>> result = await (
    ...     saia
    ...     .with_guard(terminal_status("done", "status", ("stuck",)))
    ...     .complete(task)
    ... )
"""

# Re-export _ordinal for tests that import it directly
from ._helpers import ordinal as _ordinal
from .iteration import (
    contradiction,
    narrative,
    terminal_compliance,
    terminal_deadline,
    terminal_schema,
    terminal_status,
)
from .output import (
    ascii_only,
    english_only,
    max_length,
    no_emoji,
    no_markdown,
    no_preamble,
)
from .parse import schema_retry

__all__ = [
    # Output guards
    "ascii_only",
    "english_only",
    "max_length",
    "no_emoji",
    "no_markdown",
    "no_preamble",
    # Parse guards
    "schema_retry",
    # Iteration guards - terminal tool validation
    "contradiction",
    "terminal_schema",
    "terminal_status",
    # Iteration guards - behavioral
    "narrative",
    "terminal_compliance",
    "terminal_deadline",
    # Internal (exported for tests)
    "_ordinal",
]
