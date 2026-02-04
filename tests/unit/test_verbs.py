"""Tests for individual verb implementations."""

from dataclasses import dataclass

import pytest

from llm_saia.core.types import (
    ChooseResult,
    ClassifyResult,
    ConfirmResult,
    Critique,
    Evidence,
    VerifyResult,
)
from llm_saia.verbs import (
    Ask,
    Choose,
    Classify,
    Confirm,
    Constrain,
    Critique_,
    Decompose,
    Ground,
    Instruct,
    Parse,
    Refine,
    Synthesize,
    VerbConfig,
    Verify,
    recall,
    store,
)
from llm_saia.verbs.decompose import DecomposeResult
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


def make_config(backend: MockBackend) -> VerbConfig:
    """Create a VerbConfig with no tools (direct backend calls)."""
    return VerbConfig(backend=backend, tools=[], executor=None, system=None)


class TestAsk:
    async def test_ask_includes_artifact_and_question(self, mock_backend: MockBackend) -> None:
        ask = Ask(make_config(mock_backend))
        result = await ask("test artifact", "what is this?")

        assert "test artifact" in mock_backend.last_prompt
        assert "what is this?" in mock_backend.last_prompt
        assert result == "mock response"

    async def test_ask_returns_completion(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("custom answer")
        ask = Ask(make_config(mock_backend))
        result = await ask("artifact", "question")

        assert result == "custom answer"


class TestParse:
    async def test_parse_uses_structured_output(self, mock_backend: MockBackend) -> None:
        @dataclass
        class TestSchema:
            value: str

        mock_backend.set_structured_response(TestSchema, TestSchema(value="parsed"))
        parse = Parse(make_config(mock_backend))
        result = await parse("raw response", TestSchema)

        assert result.value == "parsed"
        assert "raw response" in mock_backend.last_prompt


class TestConstrain:
    async def test_constrain_enforces_rules(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("Constrained text with citations.")
        constrain = Constrain(make_config(mock_backend))
        result = await constrain("Original text", ["cite sources", "no speculation"])

        assert result == "Constrained text with citations."
        assert "cite sources" in mock_backend.last_prompt
        assert "no speculation" in mock_backend.last_prompt
        assert "Original text" in mock_backend.last_prompt


class TestClassify:
    async def test_classify_returns_result(self, mock_backend: MockBackend) -> None:
        classify = Classify(make_config(mock_backend))
        result = await classify("some text", ["cat_a", "cat_b"])

        assert isinstance(result, ClassifyResult)
        assert result.category == "test_category"
        assert result.confidence == 0.9

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


class TestConfirm:
    async def test_confirm_returns_result(self, mock_backend: MockBackend) -> None:
        confirm = Confirm(make_config(mock_backend))
        result = await confirm("the task is complete")

        assert isinstance(result, ConfirmResult)
        assert result.confirmed is True
        assert result.reason == "test confirmation reason"

    async def test_confirm_includes_claim_in_prompt(self, mock_backend: MockBackend) -> None:
        confirm = Confirm(make_config(mock_backend))
        await confirm("this claim should be confirmed")

        assert "this claim should be confirmed" in mock_backend.last_prompt

    async def test_confirm_includes_context_when_provided(self, mock_backend: MockBackend) -> None:
        confirm = Confirm(make_config(mock_backend))
        await confirm("task complete", context="The agent processed all items.")

        assert "task complete" in mock_backend.last_prompt
        assert "The agent processed all items." in mock_backend.last_prompt


class TestChoose:
    async def test_choose_returns_result(self, mock_backend: MockBackend) -> None:
        choose = Choose(make_config(mock_backend))
        result = await choose(["option_a", "option_b"])

        assert isinstance(result, ChooseResult)
        assert result.choice == "option_a"
        assert result.reason == "test choice reason"

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

        assert result == "I will wrap up now."

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

        assert isinstance(result, VerifyResult)
        assert result.passed is True
        assert result.reason == "test reason"

    async def test_verify_includes_artifact_and_predicate(self, mock_backend: MockBackend) -> None:
        verify = Verify(make_config(mock_backend))
        await verify("my claim", "factually correct")

        assert "my claim" in mock_backend.last_prompt
        assert "factually correct" in mock_backend.last_prompt


class TestCritique:
    async def test_critique_returns_result(self, mock_backend: MockBackend) -> None:
        critique = Critique_(make_config(mock_backend))
        result = await critique("argument to critique")

        assert isinstance(result, Critique)
        assert result.counter_argument == "test counter"
        assert result.weaknesses == ["weakness 1"]
        assert result.strength == 0.5

    async def test_critique_includes_artifact(self, mock_backend: MockBackend) -> None:
        critique = Critique_(make_config(mock_backend))
        await critique("specific argument")

        assert "specific argument" in mock_backend.last_prompt


class TestRefine:
    async def test_refine_includes_artifact_and_feedback(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("refined artifact")
        refine = Refine(make_config(mock_backend))
        result = await refine("original", "make it better")

        assert result == "refined artifact"
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

        assert result.combined == "synthesis"
        assert "art1" in mock_backend.last_prompt
        assert "art2" in mock_backend.last_prompt
        assert "art3" in mock_backend.last_prompt


class TestGround:
    async def test_ground_returns_evidence_per_source(self, mock_backend: MockBackend) -> None:
        ground = Ground(make_config(mock_backend))
        result = await ground("hypothesis", ["source1", "source2"])

        assert len(result) == 2
        assert all(isinstance(e, Evidence) for e in result)

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

        assert result == ["step1", "step2"]
        assert "complex task" in mock_backend.last_prompt


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
