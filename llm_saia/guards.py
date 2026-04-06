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
from collections.abc import Callable
from typing import Any

from .core.guard import OutputGuard

__all__ = [
    "ascii_only",
    "english_only",
    "max_length",
    "no_emoji",
    "no_markdown",
    "no_preamble",
]


def english_only(max_retries: int = 1, *, escalate: bool = True) -> OutputGuard:
    """Require English-only output (no CJK, Arabic, Cyrillic, etc.).

    Args:
        max_retries: Max retry attempts. Default 1.
        escalate: Use increasingly forceful retry instructions. Default False.
    """
    static = (
        "Your response contained non-English characters. "
        "Respond ONLY in English using Latin script. "
        "Do not use any other language or script."
    )
    return OutputGuard(
        validator=_is_english,
        retry_instruction=_escalating(
            "respond in English",
            static,
            "ENGLISH ONLY. Latin script. ZERO non-English characters. Do it NOW.",
        )
        if escalate
        else static,
        max_retries=max_retries,
        name="english_only",
    )


def max_length(n: int, max_retries: int = 2, *, escalate: bool = True) -> OutputGuard:
    """Limit response to n characters.

    Args:
        n: Maximum character count.
        max_retries: Max retry attempts. Default 2 (length often needs multiple tries).
        escalate: Use increasingly forceful retry instructions. Default False.
    """

    def check(result: Any) -> str | None:
        length = len(str(result))
        return f"Response is {length} chars (max {n})" if length > n else None

    def instruction(attempt: int, result: Any, error: str) -> str:
        current = len(str(result))
        if attempt == 0:
            return f"Response is {current} chars (limit: {n}). Shorten to under {n} characters."
        return (
            f"YOU HAVE FAILED TO STAY UNDER {n} CHARS {attempt + 1} TIMES. "
            f"Current: {current}. Limit: {n}. "
            f"Strip EVERYTHING non-essential. No adjectives, no qualifiers, no preamble. "
            f"Just the raw answer. DO IT NOW."
        )

    static = (
        f"Your response exceeded the maximum length. "
        f"Shorten it to under {n} characters while preserving the key information."
    )
    return OutputGuard(
        validator=check,
        retry_instruction=instruction if escalate else static,
        max_retries=max_retries,
        name="max_length",
    )


def no_markdown(max_retries: int = 1, *, escalate: bool = True) -> OutputGuard:
    """Plain text only - no markdown formatting.

    Args:
        max_retries: Max retry attempts. Default 1.
        escalate: Use increasingly forceful retry instructions. Default False.
    """
    static = (
        "Your response contained markdown formatting. "
        "Respond in plain text only. Do not use: "
        "headers (#), bullet points (- or *), bold (**), or code blocks (```)."
    )
    return OutputGuard(
        validator=_has_no_markdown,
        retry_instruction=_escalating(
            "remove markdown",
            static,
            "NO MARKDOWN. No #, no **, no ```, no bullets. PLAIN TEXT ONLY. Do it NOW.",
        )
        if escalate
        else static,
        max_retries=max_retries,
        name="no_markdown",
    )


def no_preamble(max_retries: int = 1, *, escalate: bool = True) -> OutputGuard:
    """No conversational preamble - start directly with content.

    Args:
        max_retries: Max retry attempts. Default 1.
        escalate: Use increasingly forceful retry instructions. Default False.
    """
    static = (
        "Your response started with a conversational preamble. "
        "Start directly with the answer. Do not begin with phrases like "
        "'Sure!', 'Here is...', 'I'd be happy to...', 'Certainly!', etc."
    )
    return OutputGuard(
        validator=_has_no_preamble,
        retry_instruction=_escalating(
            "skip the preamble",
            static,
            "START WITH THE ANSWER. No 'Sure', no 'Here is', no greetings. "
            "First word must be content. Do it NOW.",
        )
        if escalate
        else static,
        max_retries=max_retries,
        name="no_preamble",
    )


def no_emoji(max_retries: int = 1, *, escalate: bool = True) -> OutputGuard:
    """No emoji characters allowed.

    Args:
        max_retries: Max retry attempts. Default 1.
        escalate: Use increasingly forceful retry instructions. Default False.
    """
    static = "Your response contained emoji. Do not use emoji or emoticons. Use plain text only."
    return OutputGuard(
        validator=_has_no_emoji,
        retry_instruction=_escalating(
            "remove emoji",
            static,
            "ZERO EMOJI. Not a single one. Remove every emoji character. Text only. Do it NOW.",
        )
        if escalate
        else static,
        max_retries=max_retries,
        name="no_emoji",
    )


def ascii_only(max_retries: int = 1, *, escalate: bool = True) -> OutputGuard:
    """ASCII characters only (printable + whitespace).

    Args:
        max_retries: Max retry attempts. Default 1.
        escalate: Use increasingly forceful retry instructions. Default False.
    """
    static = (
        "Your response contained non-ASCII characters. "
        "Use only basic ASCII characters (a-z, A-Z, 0-9, standard punctuation). "
        "No accented letters, special symbols, or Unicode."
    )
    return OutputGuard(
        validator=_is_ascii,
        retry_instruction=_escalating(
            "use ASCII only",
            static,
            "ASCII ONLY. a-z A-Z 0-9 and basic punctuation. "
            "ZERO special characters, ZERO accented letters. Do it NOW.",
        )
        if escalate
        else static,
        max_retries=max_retries,
        name="ascii_only",
    )


# --- Escalation helper ---


def _escalating(requirement: str, polite: str, forceful: str) -> Callable[[int, Any, str], str]:
    """Create an escalating instruction with failure count feedback.

    Args:
        requirement: Short name for the requirement (e.g., "use English only").
        polite: Instruction for first attempt.
        forceful: Aggressive instruction for subsequent attempts.
    """

    def instruction(attempt: int, result: Any, error: str) -> str:
        if attempt == 0:
            return polite
        return f"YOU HAVE FAILED TO {requirement.upper()} {attempt + 1} TIMES. {forceful}"

    return instruction


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


def _is_english(result: Any) -> str | None:
    """Check for non-Latin script characters."""
    text = str(result)
    for char in text:
        cp = ord(char)
        # Allow: Basic Latin, Latin-1 Supplement, Latin Extended-A/B, common punctuation
        if cp > 0x024F and cp not in _ALLOWED_PUNCTUATION:
            return f"Contains non-English character: '{char}' (U+{cp:04X})"
    return None


def _has_no_markdown(result: Any) -> str | None:
    """Check for markdown patterns."""
    text = str(result)
    patterns = [
        (r"^#{1,6}\s", "headers (#)"),
        (r"^\s*[-*]\s", "bullet points"),
        (r"\*\*[^*]+\*\*", "bold text (**)"),
        (r"```", "code blocks"),
        (r"`[^`]+`", "inline code"),
        (r"\[.+\]\(.+\)", "links"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, text, re.MULTILINE):
            return f"Contains markdown: {name}"
    return None


def _has_no_preamble(result: Any) -> str | None:
    """Check for conversational preamble."""
    text = str(result)
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
    first_line = text.split("\n")[0].lower().strip()
    for p in preambles:
        if first_line.startswith(p):
            preview = first_line[:50] + "..." if len(first_line) > 50 else first_line
            return f"Starts with preamble: '{preview}'"
    return None


def _is_ascii(result: Any) -> str | None:
    """Check for non-ASCII characters."""
    text = str(result)
    for char in text:
        if ord(char) > 127:
            return f"Contains non-ASCII: '{char}' (U+{ord(char):04X})"
    return None


# Emoji Unicode ranges (covers most common emoji)
_EMOJI_RANGES = [
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
    (0x1F680, 0x1F6FF),  # Transport and Map Symbols
    (0x1F1E0, 0x1F1FF),  # Flags (regional indicators)
    (0x2600, 0x26FF),  # Misc Symbols
    (0x2700, 0x27BF),  # Dingbats
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x1FA00, 0x1FA6F),  # Chess Symbols, Extended-A
    (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
    (0x231A, 0x231B),  # Watch, Hourglass
    (0x23E9, 0x23F3),  # Media control symbols
    (0x23F8, 0x23FA),  # Media control symbols
    (0x25AA, 0x25AB),  # Small squares
    (0x25B6, 0x25B6),  # Play button
    (0x25C0, 0x25C0),  # Reverse button
    (0x25FB, 0x25FE),  # Medium squares
    (0x2614, 0x2615),  # Umbrella, Hot beverage
    (0x2648, 0x2653),  # Zodiac signs
    (0x267F, 0x267F),  # Wheelchair
    (0x2693, 0x2693),  # Anchor
    (0x26A1, 0x26A1),  # High voltage
    (0x26AA, 0x26AB),  # Circles
    (0x26BD, 0x26BE),  # Soccer, Baseball
    (0x26C4, 0x26C5),  # Snowman, Sun
    (0x26CE, 0x26CE),  # Ophiuchus
    (0x26D4, 0x26D4),  # No entry
    (0x26EA, 0x26EA),  # Church
    (0x26F2, 0x26F3),  # Fountain, Golf
    (0x26F5, 0x26F5),  # Sailboat
    (0x26FA, 0x26FA),  # Tent
    (0x26FD, 0x26FD),  # Fuel pump
    (0x2702, 0x2702),  # Scissors
    (0x2705, 0x2705),  # Check mark
    (0x2708, 0x270D),  # Airplane to Writing hand
    (0x270F, 0x270F),  # Pencil
    (0x2712, 0x2712),  # Black nib
    (0x2714, 0x2714),  # Check mark
    (0x2716, 0x2716),  # X mark
    (0x271D, 0x271D),  # Latin cross
    (0x2721, 0x2721),  # Star of David
    (0x2728, 0x2728),  # Sparkles
    (0x2733, 0x2734),  # Eight spoked asterisk
    (0x2744, 0x2744),  # Snowflake
    (0x2747, 0x2747),  # Sparkle
    (0x274C, 0x274C),  # Cross mark
    (0x274E, 0x274E),  # Cross mark
    (0x2753, 0x2755),  # Question marks
    (0x2757, 0x2757),  # Exclamation mark
    (0x2763, 0x2764),  # Heart exclamation, Heart
    (0x2795, 0x2797),  # Plus, Minus, Division
    (0x27A1, 0x27A1),  # Right arrow
    (0x27B0, 0x27B0),  # Curly loop
    (0x27BF, 0x27BF),  # Double curly loop
    (0x2934, 0x2935),  # Arrows
    (0x2B05, 0x2B07),  # Arrows
    (0x2B1B, 0x2B1C),  # Squares
    (0x2B50, 0x2B50),  # Star
    (0x2B55, 0x2B55),  # Circle
    (0x3030, 0x3030),  # Wavy dash
    (0x303D, 0x303D),  # Part alternation mark
    (0x3297, 0x3297),  # Circled Ideograph Congratulation
    (0x3299, 0x3299),  # Circled Ideograph Secret
    (0xFE0F, 0xFE0F),  # Variation selector (emoji presentation)
    (0x200D, 0x200D),  # Zero-width joiner (for ZWJ sequences)
]


def _is_emoji(cp: int) -> bool:
    """Check if code point is in emoji ranges."""
    for start, end in _EMOJI_RANGES:
        if start <= cp <= end:
            return True
    return False


def _has_no_emoji(result: Any) -> str | None:
    """Check for emoji characters."""
    text = str(result)
    for char in text:
        cp = ord(char)
        if _is_emoji(cp):
            return f"Contains emoji: '{char}' (U+{cp:04X})"
    return None
