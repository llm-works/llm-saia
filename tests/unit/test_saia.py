"""Tests for the main SAIA class."""

from dataclasses import dataclass

import pytest

from llm_saia.core.errors import StructuredOutputError
from llm_saia.core.types import ChooseResult, ClassifyResult, Critique, VerbResult, VerifyResult
from llm_saia.guards import schema_retry
from llm_saia.verbs.decompose import DecomposeResult
from tests.unit.conftest import MockBackend, make_saia

pytestmark = pytest.mark.unit


class TestSAIA:
    def test_init(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        assert saia._config.backend is mock_backend
        assert saia._memory == {}

    async def test_ask(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("the answer")

        result = await saia.ask("artifact", "question")

        assert result.value == "the answer"

    async def test_extract(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Output:
            data: str

        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(Output, Output(data="extracted"))

        result = await saia.extract("raw content", Output)

        assert result.value.data == "extracted"

    async def test_constrain(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("constrained output")

        result = await saia.constrain("text", ["rule1", "rule2"])

        assert result.value == "constrained output"

    async def test_classify(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.classify("text", ["cat_a", "cat_b"])

        assert isinstance(result.value, ClassifyResult)
        assert result.value.category == "test_category"

    async def test_choose(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.choose(["option_a", "option_b"])

        assert isinstance(result.value, ChooseResult)
        assert result.value.choice == "option_a"

    async def test_instruct(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("Done.")

        result = await saia.instruct("Complete the task")

        assert result.value == "Done."

    async def test_verify(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.verify("claim", "predicate")

        assert isinstance(result.value, VerifyResult)
        assert result.value.passed is True

    async def test_critique(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.critique("argument")

        assert isinstance(result.value, Critique)
        assert result.value.counter_argument == "test counter"

    async def test_refine(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("improved")

        result = await saia.refine("original", "feedback")

        assert result.value == "improved"

    async def test_synthesize(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Combined:
            result: str

        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(Combined, Combined(result="merged"))

        result = await saia.synthesize(["a", "b"], Combined)

        assert result.value.result == "merged"

    async def test_ground(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.ground("hypothesis", ["source1"])

        assert len(result.value) == 1
        assert result.value[0].content == "test content"

    async def test_decompose(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(
            DecomposeResult, DecomposeResult(subtasks=["task1", "task2"])
        )

        result = await saia.decompose("big task")

        assert result.value == ["task1", "task2"]

    def test_store_and_recall(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        saia.store("key", "value")
        result = saia.recall("key")

        assert result == ["value"]

    def test_recall_empty(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = saia.recall("nonexistent")

        assert result == []

    def test_compose_simple(self, mock_backend: MockBackend) -> None:
        """Test basic composition with multiple layers."""
        saia = make_saia(mock_backend)

        result = saia.compose("You are helpful", "Context here", "Do this task")

        assert result == "You are helpful\n\nContext here\n\nDo this task"

    def test_compose_filters_none(self, mock_backend: MockBackend) -> None:
        """Test that None values are filtered out."""
        saia = make_saia(mock_backend)

        result = saia.compose("Identity", None, "Task")

        assert result == "Identity\n\nTask"

    def test_compose_filters_empty(self, mock_backend: MockBackend) -> None:
        """Test that empty strings are filtered out."""
        saia = make_saia(mock_backend)

        result = saia.compose("Identity", "", "Task")

        assert result == "Identity\n\nTask"

    def test_compose_custom_separator(self, mock_backend: MockBackend) -> None:
        """Test composition with custom separator."""
        saia = make_saia(mock_backend)

        result = saia.compose("Step 1", "Step 2", "Step 3", separator=" -> ")

        assert result == "Step 1 -> Step 2 -> Step 3"

    def test_compose_all_empty(self, mock_backend: MockBackend) -> None:
        """Test composition when all layers are empty."""
        saia = make_saia(mock_backend)

        result = saia.compose(None, "", None)

        assert result == ""

    def test_compose_single_layer(self, mock_backend: MockBackend) -> None:
        """Test composition with a single layer."""
        saia = make_saia(mock_backend)

        result = saia.compose("Single layer")

        assert result == "Single layer"


@dataclass
class _Judgment:
    verdict: str
    confidence: float


class TestCompleteStructured:
    """Tests for SAIA.complete_structured — the public structured primitive."""

    async def test_returns_verbresult_with_parsed_value(self, mock_backend: MockBackend) -> None:
        """Parses backend response against schema and wraps in VerbResult."""
        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(_Judgment, _Judgment(verdict="yes", confidence=0.9))

        result = await saia.complete_structured("Is this a cat?", _Judgment)

        assert isinstance(result, VerbResult)
        assert result.value.verdict == "yes"
        assert result.value.confidence == 0.9

    async def test_sends_prompt_verbatim(self, mock_backend: MockBackend) -> None:
        """Unlike domain verbs, no framing is prepended to the prompt."""
        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 1.0))
        prompt = "Return {'verdict': 'y', 'confidence': 1.0}"

        await saia.complete_structured(prompt, _Judgment)

        # Prompt appears verbatim as the user message content (Extract, by
        # contrast, would wrap it with "Extract the following...").
        assert mock_backend.last_prompt == prompt

    async def test_emits_trace(self, mock_backend: MockBackend) -> None:
        """Result carries a VerbTrace tagged with the internal verb name."""
        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.5))

        result = await saia.complete_structured("prompt", _Judgment)

        assert result.trace is not None
        assert result.trace.trace_id
        assert result.trace.verb == "_PromptVerb"
        assert len(result.trace.steps) >= 1

    async def test_with_temperature_threads_through(self, mock_backend: MockBackend) -> None:
        """with_* chaining is preserved: config threads to the backend call."""
        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.5))

        await saia.with_temperature(0.3).complete_structured("prompt", _Judgment)

        assert mock_backend.last_temperature == 0.3

    async def test_with_request_id_propagates_to_trace(self, mock_backend: MockBackend) -> None:
        """with_request_id reaches the emitted VerbTrace."""
        saia = make_saia(mock_backend).with_request_id("req-xyz")
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.5))

        result = await saia.complete_structured("prompt", _Judgment)

        assert result.trace.request_id == "req-xyz"

    async def test_raises_on_malformed_json_without_guard(self, mock_backend: MockBackend) -> None:
        """Without a retry guard, a parse failure surfaces as StructuredOutputError."""
        saia = make_saia(mock_backend)
        mock_backend.queue_raw_structured("not json at all")

        with pytest.raises(StructuredOutputError):
            await saia.complete_structured("prompt", _Judgment)

    async def test_retries_with_schema_retry_guard(self, mock_backend: MockBackend) -> None:
        """with_guard(schema_retry()) retries on parse failure and succeeds."""
        saia = make_saia(mock_backend)
        # First response bad, second good.
        mock_backend.queue_raw_structured("not json")
        mock_backend.set_structured_response(_Judgment, _Judgment("y", 0.5))

        result = await saia.with_guard(schema_retry(max_retries=2)).complete_structured(
            "prompt", _Judgment
        )

        assert result.value.verdict == "y"
