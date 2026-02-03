"""SAIA: Framework-agnostic verb vocabulary for LLM agents."""

from typing import Any, TypeVar

from llm_saia.core.protocols import SAIABackend
from llm_saia.core.types import Critique, Evidence, VerbResult, VerifyResult
from llm_saia.verbs import (
    ask as _ask,
)
from llm_saia.verbs import (
    constrain as _constrain,
)
from llm_saia.verbs import (
    critique as _critique,
)
from llm_saia.verbs import (
    decompose as _decompose,
)
from llm_saia.verbs import (
    ground as _ground,
)
from llm_saia.verbs import (
    recall as _recall,
)
from llm_saia.verbs import (
    refine as _refine,
)
from llm_saia.verbs import (
    store as _store,
)
from llm_saia.verbs import (
    synthesize as _synthesize,
)
from llm_saia.verbs import (
    verify as _verify,
)

__all__ = [
    "SAIA",
    "SAIABackend",
    "Critique",
    "Evidence",
    "VerifyResult",
    "VerbResult",
]

T = TypeVar("T")


class SAIA:
    """Framework-agnostic verb vocabulary for LLM agents.

    SAIA provides a consistent set of semantic verbs for interacting with LLMs,
    independent of the underlying framework (Anthropic, OpenAI, LangChain, etc.).

    Example:
        >>> from llm_saia import SAIA
        >>> from llm_saia.backends.anthropic import AnthropicBackend
        >>>
        >>> saia = SAIA(backend=AnthropicBackend())
        >>> result = await saia.verify("Python is fast", "factually accurate")
        >>> print(f"Verified: {result.passed}")
    """

    def __init__(self, backend: SAIABackend):
        """Initialize SAIA with a backend.

        Args:
            backend: The LLM backend to use for verb execution.
        """
        self._backend = backend
        self._memory: dict[str, Any] = {}

    # --- LLM Verbs ---

    async def ask(self, artifact: Any, question: str) -> str:
        """ASK verb: Query an artifact with a question.

        Args:
            artifact: The artifact to query (will be converted to string).
            question: The question to ask about the artifact.

        Returns:
            The LLM's response to the question.
        """
        return await _ask(self._backend, artifact, question)

    async def constrain(self, response: str, schema: type[T]) -> T:
        """CONSTRAIN verb: Parse response into structured schema.

        Args:
            response: The unstructured response to parse.
            schema: A dataclass type to parse the response into.

        Returns:
            An instance of the schema type populated from the response.
        """
        return await _constrain(self._backend, response, schema)

    async def verify(self, artifact: Any, predicate: str) -> VerifyResult:
        """VERIFY verb: Check if artifact satisfies predicate.

        Args:
            artifact: The artifact to verify.
            predicate: The condition to check (e.g., "factually accurate").

        Returns:
            VerifyResult with passed (bool) and reason (str).
        """
        return await _verify(self._backend, artifact, predicate)

    async def critique(self, artifact: Any) -> Critique:
        """CRITIQUE verb: Generate strongest counter-argument.

        Args:
            artifact: The artifact to critique (claim, argument, etc.).

        Returns:
            Critique with counter_argument, weaknesses, and strength score.
        """
        return await _critique(self._backend, artifact)

    async def refine(self, artifact: Any, feedback: str) -> str:
        """REFINE verb: Improve artifact based on feedback.

        Args:
            artifact: The artifact to refine.
            feedback: Feedback describing how to improve the artifact.

        Returns:
            The refined artifact as a string.
        """
        return await _refine(self._backend, artifact, feedback)

    async def synthesize(self, artifacts: list[Any], schema: type[T]) -> T:
        """SYNTHESIZE verb: Combine multiple artifacts into structured output.

        Args:
            artifacts: List of artifacts to synthesize.
            schema: A dataclass type for the structured output.

        Returns:
            An instance of the schema type combining information from all artifacts.
        """
        return await _synthesize(self._backend, artifacts, schema)

    async def ground(self, artifact: Any, sources: list[Any]) -> list[Evidence]:
        """GROUND verb: Anchor artifact against sources for evidence.

        Args:
            artifact: The artifact (claim, hypothesis) to ground.
            sources: List of sources to check against.

        Returns:
            List of Evidence objects extracted from sources.
        """
        return await _ground(self._backend, artifact, sources)

    async def decompose(self, task: str) -> list[str]:
        """DECOMPOSE verb: Break down task into subtasks.

        Args:
            task: The task description to decompose.

        Returns:
            List of subtask descriptions.
        """
        return await _decompose(self._backend, task)

    # --- Memory Verbs ---

    def recall(self, query: str) -> list[Any]:
        """RECALL verb: Retrieve values from memory matching query.

        Args:
            query: The search query (substring match on keys).

        Returns:
            List of values whose keys contain the query.
        """
        return _recall(self._memory, query)

    def store(self, key: str, value: Any) -> None:
        """STORE verb: Save a value to memory.

        Args:
            key: The key to store under.
            value: The value to store.
        """
        _store(self._memory, key, value)
