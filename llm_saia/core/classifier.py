"""Task state classification for the Complete verb loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from llm_saia.core.config import Config


class TaskState(Enum):
    """Possible states of a task during execution."""

    COMPLETED = "completed"  # Task is done, agent provided final answer
    STUCK = "stuck"  # Agent is blocked, confused, or giving up
    WANTS_CONTINUE = "wants_continue"  # Agent wants to continue but didn't use tools
    ASKING = "asking"  # Agent is asking permission to proceed


@dataclass
class TaskStateResult:
    """Result of task state classification."""

    state: TaskState
    confidence: float  # 0.0 to 1.0
    reason: str


class TaskStateClassifier(Protocol):
    """Protocol for task state classifiers."""

    async def classify(
        self,
        task: str,
        content: str,
        tool_names: list[str],
    ) -> TaskStateResult:
        """Classify the current state of a task based on agent response.

        Args:
            task: The original task description.
            content: The agent's response content (no tool calls).
            tool_names: Names of available tools.

        Returns:
            TaskStateResult with state, confidence, and reason.
        """
        ...


class LLMTaskStateClassifier:
    """Task state classifier using the Classify verb."""

    _CRITERIA_SUFFIX = (
        "Classify the agent's response:\n"
        "- completed: The agent has finished the task and provided a final answer\n"
        "- stuck: The agent is blocked, confused, giving up, or says it cannot proceed\n"
        "- wants_continue: The agent wants to continue working but didn't use tools "
        "(e.g., says 'let me check', 'I will use', writes tool calls as text)\n"
        "- asking: The agent is asking permission before proceeding "
        "(e.g., 'shall I?', 'would you like me to?', 'do you want me to?')"
    )

    def __init__(self, config: Config) -> None:
        """Initialize with a config for creating the Classify verb."""
        self._config = config

    async def classify(
        self,
        task: str,
        content: str,
        tool_names: list[str],
    ) -> TaskStateResult:
        """Classify task state using LLM-based classification."""
        from llm_saia.verbs.classify import Classify

        tools_desc = ", ".join(tool_names) if tool_names else "none"
        criteria = (
            f"Original task: {task}\n\nAvailable tools: {tools_desc}\n\n" + self._CRITERIA_SUFFIX
        )

        classifier = Classify(self._config)
        result = await classifier(
            text=content,
            categories=[s.value for s in TaskState],
            criteria=criteria,
        )

        return TaskStateResult(
            state=self._parse_state(result.category),
            confidence=result.confidence,
            reason=result.reason,
        )

    def _parse_state(self, category: str) -> TaskState:
        """Parse category string to TaskState enum, with fallback."""
        try:
            return TaskState(category)
        except ValueError:
            return TaskState.WANTS_CONTINUE
