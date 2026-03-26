"""Pre-built output guards with sensible default instructions.

Example:
    >>> from llm_saia.guards import english_only, max_length
    >>> result = await (
    ...     saia
    ...     .with_guard(english_only())
    ...     .with_guard(max_length(500))
    ...     .summarize(article)
    ... )
"""

from __future__ import annotations

import re

from llm_saia.core.guard import OutputGuard

__all__ = [
    "english_only",
    "max_length",
    "no_markdown",
    "no_preamble",
    "ascii_only",
]


def english_only(max_retries: int = 1) -> OutputGuard:
    """Require English-only output (no CJK, Arabic, Cyrillic, etc.).

    Args:
        max_retries: Max retry attempts. Default 1.

    Returns:
        OutputGuard configured for English-only validation.
    """
    return OutputGuard(
        validator=_is_english,
        retry_instruction="Respond ONLY in English. Do not use any other language or script.",
        max_retries=max_retries,
        name="english_only",
    )


def max_length(n: int, max_retries: int = 2) -> OutputGuard:
    """Limit response to n characters.

    Args:
        n: Maximum character count.
        max_retries: Max retry attempts. Default 2 (length often needs multiple tries).

    Returns:
        OutputGuard configured for length validation.
    """

    def check(text: str) -> str | None:
        length = len(str(text))
        if length <= n:
            return None
        return f"Response is {length} chars (max {n})"

    return OutputGuard(
        validator=check,
        retry_instruction=f"Your response is too long. Keep it under {n} characters.",
        max_retries=max_retries,
        name="max_length",
    )


def no_markdown(max_retries: int = 1) -> OutputGuard:
    """Plain text only - no markdown formatting.

    Args:
        max_retries: Max retry attempts. Default 1.

    Returns:
        OutputGuard configured to reject markdown.
    """
    return OutputGuard(
        validator=_has_no_markdown,
        retry_instruction=(
            "Respond in plain text only. "
            "No markdown formatting: no headers (#), no bullet points (- or *), "
            "no bold (**), no code blocks (```)."
        ),
        max_retries=max_retries,
        name="no_markdown",
    )


def no_preamble(max_retries: int = 1) -> OutputGuard:
    """No conversational preamble - start directly with content.

    Args:
        max_retries: Max retry attempts. Default 1.

    Returns:
        OutputGuard configured to reject preambles.
    """
    return OutputGuard(
        validator=_has_no_preamble,
        retry_instruction=(
            "Start directly with the answer. "
            "Do not begin with phrases like 'Sure!', 'Here is...', "
            "'I'd be happy to...', 'Certainly!', etc."
        ),
        max_retries=max_retries,
        name="no_preamble",
    )


def ascii_only(max_retries: int = 1) -> OutputGuard:
    """ASCII characters only (printable + whitespace).

    Args:
        max_retries: Max retry attempts. Default 1.

    Returns:
        OutputGuard configured for ASCII-only validation.
    """
    return OutputGuard(
        validator=_is_ascii,
        retry_instruction="Use only basic ASCII characters. No special symbols or Unicode.",
        max_retries=max_retries,
        name="ascii_only",
    )


# --- Validator implementations ---

# Punctuation commonly used in English text that should be allowed
_ALLOWED_PUNCTUATION = {
    0x2018,  # '
    0x2019,  # '
    0x201C,  # "
    0x201D,  # "
    0x2014,  # —
    0x2013,  # –
    0x2026,  # …
    0x00B0,  # ° (degrees)
    0x00A9,  # © (copyright)
    0x00AE,  # ® (registered)
    0x2122,  # ™ (trademark)
}


def _is_english(text: str) -> str | None:
    """Check for non-Latin script characters."""
    text_str = str(text)
    for char in text_str:
        cp = ord(char)
        # Allow: Basic Latin, Latin-1 Supplement, Latin Extended-A/B, common punctuation
        if cp > 0x024F and cp not in _ALLOWED_PUNCTUATION:
            return f"Contains non-English character: '{char}' (U+{cp:04X})"
    return None


def _has_no_markdown(text: str) -> str | None:
    """Check for markdown patterns."""
    text_str = str(text)
    patterns = [
        (r"^#{1,6}\s", "headers (#)"),
        (r"^\s*[-*]\s", "bullet points"),
        (r"\*\*[^*]+\*\*", "bold text (**)"),
        (r"```", "code blocks"),
        (r"`[^`]+`", "inline code"),
        (r"\[.+\]\(.+\)", "links"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, text_str, re.MULTILINE):
            return f"Contains markdown: {name}"
    return None


def _has_no_preamble(text: str) -> str | None:
    """Check for conversational preamble."""
    preambles = [
        "sure",
        "certainly",
        "of course",
        "absolutely",
        "here is",
        "here's",
        "here are",
        "i'd be happy to",
        "i would be happy to",
        "i can help",
        "let me",
        "great question",
        "good question",
    ]
    text_str = str(text)
    first_line = text_str.split("\n")[0].lower().strip()
    for p in preambles:
        if first_line.startswith(p):
            preview = first_line[:50] + "..." if len(first_line) > 50 else first_line
            return f"Starts with preamble: '{preview}'"
    return None


def _is_ascii(text: str) -> str | None:
    """Check for non-ASCII characters."""
    text_str = str(text)
    for char in text_str:
        if ord(char) > 127:
            return f"Contains non-ASCII: '{char}' (U+{ord(char):04X})"
    return None
