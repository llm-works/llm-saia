"""Output guard mixin for verb classes.

Provides guard application, retry logic, and field-level guard extraction.
Separated from verb.py to keep the base class manageable.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from .guard import Guarded, OutputGuard, OutputGuardError

if TYPE_CHECKING:
    from .config import CallOptions
    from .conversation import ConversationLike
    from .logger import Logger
    from .trace import VerbTrace

T = TypeVar("T")


class OutputGuardMixin:
    """Mixin providing output guard application and retry logic.

    Expects the host class to provide:
      - ``_get_call_options(run)`` -> CallOptions
      - ``_complete_structured_attempt(prompt, schema, run, conversation=)`` -> T
      - ``_complete_text_attempt(prompt, run, phase=, conversation=)`` -> str
      - ``_lg`` property -> Logger | None
    """

    # Maximum revalidation rounds before raising (prevents infinite loops when
    # guard retries keep producing results that fail other guards).
    _MAX_REVALIDATION_ROUNDS = 10

    # Stubs for attributes/methods provided by the host class (Verb).

    _lg: Logger | None

    def _get_call_options(self, override: CallOptions | None = None) -> CallOptions:
        raise NotImplementedError

    async def _complete_structured_attempt(
        self,
        prompt: str,
        schema: type[T],
        run: CallOptions | None = None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
        _phase: str = "attempt",
    ) -> T:
        raise NotImplementedError

    async def _complete_text_attempt(
        self,
        prompt: str,
        run: CallOptions | None = None,
        phase: str = "direct",
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> str:
        raise NotImplementedError

    # -- Structured output guards --

    async def _apply_guards(
        self,
        prompt: str,
        result: T,
        schema: type[T],
        run: CallOptions | None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> T:
        """Apply output guards sequentially, retrying on failure.

        Applies both instance-level guards (from with_guard/with_guards) and
        field-level guards (from Annotated[..., Guarded(...)] type hints).

        If a guard retry produces a different result, all guards are re-validated
        from the beginning to ensure the new result passes all guards.
        """
        config = self._get_call_options(run)
        instance_guards = config.output_guards
        field_guards = self._extract_field_guards(schema)

        for _round in range(self._MAX_REVALIDATION_ROUNDS):
            original_result = result

            result, changed = await self._apply_instance_guards_once(
                prompt, result, original_result, schema, instance_guards, run, conversation, _trace
            )
            if not changed:
                result, changed = await self._apply_field_guards_once(
                    prompt, result, original_result, schema, field_guards, run, conversation, _trace
                )
            if not changed:
                return result

        raise OutputGuardError(
            "revalidation", "guards did not converge", self._MAX_REVALIDATION_ROUNDS
        )

    async def _apply_instance_guards_once(
        self,
        prompt: str,
        result: T,
        original_result: T,
        schema: type[T],
        guards: tuple[OutputGuard, ...],
        run: CallOptions | None,
        conversation: ConversationLike | None,
        _trace: VerbTrace | None = None,
    ) -> tuple[T, bool]:
        """Apply instance-level guards once, returning (result, changed)."""
        for guard in guards:
            result = await self._apply_single_guard(
                prompt, result, schema, guard, run, conversation=conversation, _trace=_trace
            )
            if result != original_result:
                return result, True
        return result, False

    async def _apply_field_guards_once(
        self,
        prompt: str,
        result: T,
        original_result: T,
        schema: type[T],
        field_guards: dict[str, tuple[OutputGuard, ...]],
        run: CallOptions | None,
        conversation: ConversationLike | None,
        _trace: VerbTrace | None = None,
    ) -> tuple[T, bool]:
        """Apply field-level guards once, returning (result, changed)."""
        for field_name, guards in field_guards.items():
            for guard in guards:
                result = await self._apply_field_guard(
                    prompt,
                    result,
                    schema,
                    field_name,
                    guard,
                    run,
                    conversation=conversation,
                    _trace=_trace,
                )
                if result != original_result:
                    return result, True
            if result != original_result:
                return result, True
        return result, False

    def _extract_field_guards(self, schema: type[T]) -> dict[str, tuple[OutputGuard, ...]]:
        """Extract field-level guards from Annotated type hints.

        Returns:
            Dict mapping field name to tuple of guards for that field.
        """
        result: dict[str, tuple[OutputGuard, ...]] = {}

        try:
            hints = get_type_hints(schema, include_extras=True)
        except TypeError:
            # Schema doesn't support type hints (e.g., not a dataclass)
            return result

        for field_name, hint in hints.items():
            if get_origin(hint) is Annotated:
                args = get_args(hint)
                for arg in args[1:]:  # Skip the actual type (first arg)
                    if isinstance(arg, Guarded):
                        result[field_name] = arg.guards
                        break  # Only one Guarded per field

        return result

    async def _apply_field_guard(
        self,
        prompt: str,
        result: T,
        schema: type[T],
        field_name: str,
        guard: OutputGuard,
        run: CallOptions | None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> T:
        """Apply a guard to a specific field of the result."""
        gname = f"{field_name}.{guard.name}" if guard.name else field_name
        for attempt in range(1 + guard.max_retries):
            field_value = getattr(result, field_name, None)
            error = self._run_validator(guard, field_value)
            if error is None:
                return result
            if attempt >= guard.max_retries:
                raise OutputGuardError(gname, error, attempt + 1)
            self._log_guard_retry(gname, attempt + 1, guard.max_retries, error)
            retry_prompt = self._build_field_guard_retry_prompt(
                prompt, result, field_name, field_value, guard, error, attempt
            )
            result = await self._complete_structured_attempt(
                retry_prompt,
                schema,
                run,
                conversation=conversation,
                _trace=_trace,
                _phase="guard_retry",
            )
        return result  # Unreachable, satisfies type checker

    @staticmethod
    def _run_validator(guard: OutputGuard, value: Any) -> str | None:
        """Run guard validator, converting exceptions to error strings."""
        try:
            return guard.validator(value)
        except Exception as e:
            return f"Validator raised {type(e).__name__}: {e}"

    def _build_field_guard_retry_prompt(
        self,
        original_prompt: str,
        failed_result: Any,
        field_name: str,
        field_value: Any,
        guard: OutputGuard,
        error: str,
        attempt: int = 0,
    ) -> str:
        """Build retry prompt for field-specific guard failure."""
        result_str = str(failed_result)
        if len(result_str) > 300:
            result_str = result_str[:300] + "..."

        field_str = str(field_value)
        if len(field_str) > 200:
            field_str = field_str[:200] + "..."

        instruction = guard.resolve_instruction(attempt, field_value, error)
        return (
            f"{original_prompt}\n\n"
            f"---\n\n"
            f"Your previous response did not meet requirements.\n\n"
            f"Issue with field '{field_name}': {error}\n\n"
            f"Field value was:\n```\n{field_str}\n```\n\n"
            f"{instruction}"
        )

    # -- Instance-level structured guard --

    async def _apply_single_guard(
        self,
        prompt: str,
        result: T,
        schema: type[T],
        guard: OutputGuard,
        run: CallOptions | None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> T:
        """Apply one guard with retries.

        Note: Guard retries call _complete_structured_attempt directly, bypassing
        parse_retries. This is intentional - guards run after parsing succeeds,
        so JSON structure is expected to be stable on retry.
        """
        for attempt in range(1 + guard.max_retries):
            error = self._run_validator(guard, result)
            if error is None:
                return result  # Passed

            if attempt < guard.max_retries:
                self._log_guard_retry(guard.name, attempt + 1, guard.max_retries, error)
                retry_prompt = self._build_guard_retry_prompt(prompt, result, guard, error, attempt)
                result = await self._complete_structured_attempt(
                    retry_prompt,
                    schema,
                    run,
                    conversation=conversation,
                    _trace=_trace,
                    _phase="guard_retry",
                )
            else:
                raise OutputGuardError(guard.name, error, attempt + 1)

        return result  # Unreachable, satisfies type checker

    # -- Text guards --

    async def _apply_text_guards(
        self,
        prompt: str,
        text: str,
        run: CallOptions | None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> str:
        """Apply output guards to text completion results.

        Similar to _apply_guards but for plain text (not structured output).
        Guards validate the text string directly.
        """
        config = self._get_call_options(run)
        guards = config.output_guards
        if not guards:
            return text

        for _round in range(self._MAX_REVALIDATION_ROUNDS):
            original_text = text
            for guard in guards:
                text = await self._apply_single_text_guard(
                    prompt, text, guard, run, conversation=conversation, _trace=_trace
                )
                if text != original_text:
                    break
            else:
                return text

        raise OutputGuardError(
            "revalidation", "guards did not converge", self._MAX_REVALIDATION_ROUNDS
        )

    async def _apply_single_text_guard(
        self,
        prompt: str,
        text: str,
        guard: OutputGuard,
        run: CallOptions | None,
        conversation: ConversationLike | None = None,
        _trace: VerbTrace | None = None,
    ) -> str:
        """Apply one guard to text with retries."""
        for attempt in range(1 + guard.max_retries):
            error = self._run_validator(guard, text)
            if error is None:
                return text  # Passed

            if attempt < guard.max_retries:
                self._log_guard_retry(guard.name, attempt + 1, guard.max_retries, error)
                retry_prompt = self._build_guard_retry_prompt(prompt, text, guard, error, attempt)
                text = await self._complete_text_attempt(
                    retry_prompt,
                    run,
                    phase="guard_retry",
                    conversation=conversation,
                    _trace=_trace,
                )
            else:
                raise OutputGuardError(guard.name, error, attempt + 1)

        return text  # Unreachable, satisfies type checker

    # -- Shared helpers --

    def _build_guard_retry_prompt(
        self,
        original_prompt: str,
        failed_result: Any,
        guard: OutputGuard,
        error: str,
        attempt: int = 0,
    ) -> str:
        """Build retry prompt with guard feedback."""
        result_str = str(failed_result)
        if len(result_str) > 300:
            result_str = result_str[:300] + "..."

        instruction = guard.resolve_instruction(attempt, failed_result, error)
        return (
            f"{original_prompt}\n\n"
            f"---\n\n"
            f"Your previous response did not meet requirements.\n\n"
            f"Issue: {error}\n\n"
            f"Your response was:\n```\n{result_str}\n```\n\n"
            f"{instruction}"
        )

    def _log_guard_retry(
        self, name: str | None, attempt: int, max_retries: int, error: str
    ) -> None:
        """Log guard retry attempt."""
        if self._lg:
            self._lg.debug(
                "guard retry",
                extra={
                    "guard": name,
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "error": error,
                },
            )
