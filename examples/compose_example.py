"""Example showing compose() for clean prompt building.

This demonstrates how compose() simplifies prompt construction in agent code,
replacing manual string concatenation and conditional logic.
"""

from typing import Any

from llm_saia import SAIA
from llm_saia.core.backend import AgentResponse, Backend, Message


class DemoBackend(Backend):
    """Demo backend for examples."""

    async def chat(  # type: ignore[empty-body]
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> AgentResponse:
        """Placeholder chat method."""
        pass


def create_demo_saia() -> SAIA:
    """Create SAIA instance with demo backend."""
    return SAIA.builder().backend(DemoBackend()).build()


def example_simple(saia: SAIA) -> None:
    """Example 1: Simple composition."""
    identity = "You are a helpful assistant"
    task = "Explain quantum computing"

    prompt = saia.compose(identity, task)
    print("Example 1 - Simple:")
    print(prompt)
    print()


def example_with_none(saia: SAIA) -> None:
    """Example 2: Filtering None values."""
    identity = "You are a helpful assistant"
    past_context = None  # No past executions yet
    task = "Explain quantum computing"

    prompt = saia.compose(identity, past_context, task)
    print("Example 2 - With None (filtered out):")
    print(prompt)
    print()


def example_with_context(saia: SAIA) -> None:
    """Example 3: With populated context."""
    identity = "You are a helpful assistant"
    past_context = "Previous findings:\n- User prefers concise answers"
    task = "Explain quantum computing"

    prompt = saia.compose(identity, past_context, task)
    print("Example 3 - With context:")
    print(prompt)
    print()


def example_custom_separator(saia: SAIA) -> None:
    """Example 4: Custom separator."""
    steps = ["Understand the problem", "Design a solution", "Implement it"]

    prompt = saia.compose(*steps, separator=" → ")
    print("Example 4 - Custom separator:")
    print(prompt)
    print()


def format_past_solutions(solutions: list[str]) -> str:
    """Format past solutions for prompt context."""
    if not solutions:
        return ""
    return "\n".join(f"- {s}" for s in solutions)


def example_agent_pattern(saia: SAIA) -> None:
    """Example 5: Agent pattern (like DefaultAgent)."""
    past_solutions = ["Solution 1", "Solution 2"]
    default_prompt = "Analyze the latest data"

    context = format_past_solutions(past_solutions)
    prompt = saia.compose(context, default_prompt)

    print("Example 5 - Agent pattern:")
    print(prompt)


def main() -> None:
    """Show compose() usage patterns."""
    saia = create_demo_saia()

    example_simple(saia)
    example_with_none(saia)
    example_with_context(saia)
    example_custom_separator(saia)
    example_agent_pattern(saia)


if __name__ == "__main__":
    main()
