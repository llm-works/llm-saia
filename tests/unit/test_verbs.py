"""Tests for individual verb implementations."""

from dataclasses import dataclass

import pytest

from llm_saia.core.logger import NullLogger
from llm_saia.core.types import (
    ChooseResult,
    ClassifyResult,
    Critique,
    Evidence,
    FindResult,
    VerifyResult,
)
from llm_saia.verbs import (
    Ask,
    Choose,
    Classify,
    Config,
    Constrain,
    Critique_,
    Decompose,
    Extract,
    Find,
    Ground,
    Instruct,
    Refine,
    Synthesize,
    Verify,
    recall,
    store,
)
from llm_saia.verbs.decompose import DecomposeResult
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


def make_config(backend: MockBackend) -> Config:
    """Create a Config with no tools (direct backend calls)."""
    return Config(lg=NullLogger(), backend=backend, tools=[], executor=None)


class TestAsk:
    async def test_ask_includes_artifact_and_question(self, mock_backend: MockBackend) -> None:
        ask = Ask(make_config(mock_backend))
        result = await ask("test artifact", "what is this?")

        assert "test artifact" in mock_backend.last_prompt
        assert "what is this?" in mock_backend.last_prompt
        assert result.value == "mock response"
        assert result.trace.trace_id
        assert result.trace.verb == "Ask"
        assert len(result.trace.steps) >= 1

    async def test_ask_returns_completion(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("custom answer")
        ask = Ask(make_config(mock_backend))
        result = await ask("artifact", "question")

        assert result.value == "custom answer"


_FIND_RESPONSE = "_FindResponse"


class TestFind:
    async def test_find_returns_result(self, mock_backend: MockBackend) -> None:
        find = Find(make_config(mock_backend))
        result = await find(["item a", "item b", "item c"], "matches test criteria")

        assert isinstance(result.value, FindResult)
        # Default mock returns matching_numbers=[1, 3], which becomes indices=[0, 2]
        assert result.value.indices == [0, 2]
        assert result.value.reason == "test find reason"
        assert result.trace.trace_id
        assert result.trace.verb == "Find"
        assert len(result.trace.steps) >= 1

    async def test_find_includes_items_and_criteria_in_prompt(
        self, mock_backend: MockBackend
    ) -> None:
        find = Find(make_config(mock_backend))
        await find(["alpha", "beta", "gamma"], "starts with vowel")

        assert "alpha" in mock_backend.last_prompt
        assert "beta" in mock_backend.last_prompt
        assert "gamma" in mock_backend.last_prompt
        assert "starts with vowel" in mock_backend.last_prompt

    async def test_find_empty_items_returns_empty(self, mock_backend: MockBackend) -> None:
        find = Find(make_config(mock_backend))
        result = await find([], "any criteria")

        assert result.value.indices == []
        assert "No items" in result.value.reason

    async def test_find_filters_invalid_indices(self, mock_backend: MockBackend) -> None:
        mock_backend.set_structured_response(
            _FIND_RESPONSE, {"matching_numbers": [1, 5, 10], "reason": "includes invalid"}
        )
        find = Find(make_config(mock_backend))
        result = await find(["a", "b", "c"], "criteria")

        # Only index 1 is valid (maps to 0), 5 and 10 are out of range
        assert result.value.indices == [0]

    async def test_find_deduplicates_indices(self, mock_backend: MockBackend) -> None:
        mock_backend.set_structured_response(
            _FIND_RESPONSE, {"matching_numbers": [2, 1, 2, 1], "reason": "has duplicates"}
        )
        find = Find(make_config(mock_backend))
        result = await find(["a", "b", "c"], "criteria")

        # Duplicates removed, sorted order
        assert result.value.indices == [0, 1]

    async def test_find_rejects_too_many_items(self, mock_backend: MockBackend) -> None:
        from llm_saia.verbs.find import MAX_ITEMS

        find = Find(make_config(mock_backend))
        items = [f"item {i}" for i in range(MAX_ITEMS + 1)]

        with pytest.raises(ValueError, match="Too many items"):
            await find(items, "criteria")


class TestExtract:
    async def test_extract_uses_structured_output(self, mock_backend: MockBackend) -> None:
        @dataclass
        class TestSchema:
            value: str

        mock_backend.set_structured_response(TestSchema, TestSchema(value="extracted"))
        extract = Extract(make_config(mock_backend))
        result = await extract("raw content", TestSchema)

        assert result.value.value == "extracted"
        assert "raw content" in mock_backend.last_prompt

    async def test_extract_includes_instructions(self, mock_backend: MockBackend) -> None:
        @dataclass
        class TestSchema:
            value: str

        mock_backend.set_structured_response(TestSchema, TestSchema(value="extracted"))
        extract = Extract(make_config(mock_backend))
        result = await extract("content", TestSchema, instructions="Focus on names")

        assert result.value.value == "extracted"
        assert "Focus on names" in mock_backend.last_prompt

    async def test_extract_with_custom_json_parser(self, mock_backend: MockBackend) -> None:
        @dataclass
        class TestSchema:
            value: str

        # Simulate malformed JSON with unescaped newline
        malformed = '{"value": "line1\nline2"}'
        mock_backend.queue_raw_structured(malformed)

        def custom_parser(content: str) -> dict:
            # Parser that fixes newlines then parses
            import json

            return json.loads(content.replace("\n", "\\n"))

        config = Config(
            lg=NullLogger(),
            backend=mock_backend,
            tools=[],
            executor=None,
            json_parser=custom_parser,
        )
        extract = Extract(config)
        result = await extract("content", TestSchema)

        assert result.value.value == "line1\nline2"

    async def test_extract_default_parser_fails_on_malformed(
        self, mock_backend: MockBackend
    ) -> None:
        from llm_saia.core.errors import StructuredOutputError

        @dataclass
        class TestSchema:
            value: str

        malformed = '{"value": "line1\nline2"}'
        mock_backend.queue_raw_structured(malformed)

        extract = Extract(make_config(mock_backend))
        with pytest.raises(StructuredOutputError):
            await extract("content", TestSchema)

    async def test_extract_custom_parser_always_called(self, mock_backend: MockBackend) -> None:
        @dataclass
        class TestSchema:
            value: str

        mock_backend.set_structured_response(TestSchema, TestSchema(value="valid"))
        parser_called = False

        def custom_parser(content: str) -> dict:
            nonlocal parser_called
            parser_called = True
            import json

            return json.loads(content)

        config = Config(
            lg=NullLogger(),
            backend=mock_backend,
            tools=[],
            executor=None,
            json_parser=custom_parser,
        )
        extract = Extract(config)
        result = await extract("content", TestSchema)

        assert result.value.value == "valid"
        assert parser_called


class TestConstrain:
    async def test_constrain_enforces_rules(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("Constrained text with citations.")
        constrain = Constrain(make_config(mock_backend))
        result = await constrain("Original text", ["cite sources", "no speculation"])

        assert result.value == "Constrained text with citations."
        assert "cite sources" in mock_backend.last_prompt
        assert "no speculation" in mock_backend.last_prompt
        assert "Original text" in mock_backend.last_prompt

    async def test_constrain_empty_rules_returns_unchanged(self, mock_backend: MockBackend) -> None:
        constrain = Constrain(make_config(mock_backend))
        result = await constrain("Original text", [])

        assert result.value == "Original text"
        # Backend should not be called for empty rules
        assert mock_backend.last_prompt == ""


class TestClassify:
    async def test_classify_returns_result(self, mock_backend: MockBackend) -> None:
        classify = Classify(make_config(mock_backend))
        result = await classify("some text", ["cat_a", "cat_b"])

        assert isinstance(result.value, ClassifyResult)
        assert result.value.category == "test_category"
        assert result.value.confidence == 0.9

    async def test_classify_includes_categories_in_prompt(self, mock_backend: MockBackend) -> None:
        classify = Classify(make_config(mock_backend))
        await classify("text to classify", ["positive", "negative", "neutral"])

        assert "positive" in mock_backend.last_prompt
        assert "negative" in mock_backend.last_prompt
        assert "neutral" in mock_backend.last_prompt
        assert "text to classify" in mock_backend.last_prompt

    async def test_classify_includes_criteria_when_provided(
        self, mock_backend: MockBackend
    ) -> None:
        classify = Classify(make_config(mock_backend))
        await classify("text", ["a", "b"], criteria="Based on sentiment")

        assert "Based on sentiment" in mock_backend.last_prompt


class TestChoose:
    async def test_choose_returns_result(self, mock_backend: MockBackend) -> None:
        choose = Choose(make_config(mock_backend))
        result = await choose(["option_a", "option_b"])

        assert isinstance(result.value, ChooseResult)
        assert result.value.choice == "option_a"
        assert result.value.reason == "test choice reason"

    async def test_choose_includes_options_in_prompt(self, mock_backend: MockBackend) -> None:
        choose = Choose(make_config(mock_backend))
        await choose(["continue", "stop", "retry"])

        assert "continue" in mock_backend.last_prompt
        assert "stop" in mock_backend.last_prompt
        assert "retry" in mock_backend.last_prompt

    async def test_choose_includes_context_when_provided(self, mock_backend: MockBackend) -> None:
        choose = Choose(make_config(mock_backend))
        await choose(["a", "b"], context="We are at step 3.")

        assert "We are at step 3." in mock_backend.last_prompt

    async def test_choose_includes_criteria_when_provided(self, mock_backend: MockBackend) -> None:
        choose = Choose(make_config(mock_backend))
        await choose(["a", "b"], criteria="Pick the safest option")

        assert "Pick the safest option" in mock_backend.last_prompt


class TestInstruct:
    async def test_instruct_returns_response(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("I will wrap up now.")
        instruct = Instruct(make_config(mock_backend))
        result = await instruct("Wrap up your work")

        assert result.value == "I will wrap up now."

    async def test_instruct_includes_directive_in_prompt(self, mock_backend: MockBackend) -> None:
        instruct = Instruct(make_config(mock_backend))
        await instruct("Complete the task immediately")

        assert "Complete the task immediately" in mock_backend.last_prompt

    async def test_instruct_includes_context_when_provided(self, mock_backend: MockBackend) -> None:
        instruct = Instruct(make_config(mock_backend))
        await instruct("Summarize findings", context="You found 3 issues.")

        assert "Summarize findings" in mock_backend.last_prompt
        assert "You found 3 issues." in mock_backend.last_prompt


class TestVerify:
    async def test_verify_returns_result(self, mock_backend: MockBackend) -> None:
        verify = Verify(make_config(mock_backend))
        result = await verify("claim", "is accurate")

        assert isinstance(result.value, VerifyResult)
        assert result.value.passed is True
        assert result.value.reason == "test reason"

    async def test_verify_includes_artifact_and_predicate(self, mock_backend: MockBackend) -> None:
        verify = Verify(make_config(mock_backend))
        await verify("my claim", "factually correct")

        assert "my claim" in mock_backend.last_prompt
        assert "factually correct" in mock_backend.last_prompt


class TestCritique:
    async def test_critique_returns_result(self, mock_backend: MockBackend) -> None:
        critique = Critique_(make_config(mock_backend))
        result = await critique("argument to critique")

        assert isinstance(result.value, Critique)
        assert result.value.counter_argument == "test counter"
        assert result.value.weaknesses == ["weakness 1"]
        assert result.value.strength == 0.5

    async def test_critique_includes_artifact(self, mock_backend: MockBackend) -> None:
        critique = Critique_(make_config(mock_backend))
        await critique("specific argument")

        assert "specific argument" in mock_backend.last_prompt


class TestRefine:
    async def test_refine_includes_artifact_and_feedback(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("refined artifact")
        refine = Refine(make_config(mock_backend))
        result = await refine("original", "make it better")

        assert result.value == "refined artifact"
        assert "original" in mock_backend.last_prompt
        assert "make it better" in mock_backend.last_prompt


class TestSynthesize:
    async def test_synthesize_includes_all_artifacts(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Summary:
            combined: str

        mock_backend.set_structured_response(Summary, Summary(combined="synthesis"))
        synthesize = Synthesize(make_config(mock_backend))
        result = await synthesize(["art1", "art2", "art3"], Summary)

        assert result.value.combined == "synthesis"
        assert "art1" in mock_backend.last_prompt
        assert "art2" in mock_backend.last_prompt
        assert "art3" in mock_backend.last_prompt


class TestGround:
    async def test_ground_returns_evidence_per_source(self, mock_backend: MockBackend) -> None:
        ground = Ground(make_config(mock_backend))
        result = await ground("hypothesis", ["source1", "source2"])

        assert len(result.value) == 2
        assert all(isinstance(e, Evidence) for e in result.value)

    async def test_ground_includes_artifact_and_source(self, mock_backend: MockBackend) -> None:
        ground = Ground(make_config(mock_backend))
        await ground("my hypothesis", ["my source"])

        assert "my hypothesis" in mock_backend.last_prompt
        assert "my source" in mock_backend.last_prompt


class TestDecompose:
    async def test_decompose_returns_subtasks(self, mock_backend: MockBackend) -> None:
        mock_backend.set_structured_response(
            DecomposeResult, DecomposeResult(subtasks=["step1", "step2"])
        )
        decompose = Decompose(make_config(mock_backend))
        result = await decompose("complex task")

        assert result.value == ["step1", "step2"]
        assert "complex task" in mock_backend.last_prompt
        assert result.trace.trace_id
        assert result.trace.verb == "Decompose"
        assert len(result.trace.steps) >= 1


class TestMemory:
    def test_store_and_recall(self) -> None:
        memory: dict[str, object] = {}
        store(memory, "key1", "value1")
        store(memory, "other_key", "value2")

        assert recall(memory, "key1") == ["value1"]
        assert recall(memory, "key") == ["value1", "value2"]
        assert recall(memory, "nonexistent") == []

    def test_recall_case_insensitive(self) -> None:
        memory: dict[str, object] = {}
        store(memory, "MyKey", "value")

        assert recall(memory, "mykey") == ["value"]
        assert recall(memory, "MYKEY") == ["value"]
