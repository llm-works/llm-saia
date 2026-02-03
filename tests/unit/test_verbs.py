"""Tests for individual verb implementations."""

from dataclasses import dataclass

import pytest

from llm_saia.core.types import Critique, Evidence, VerifyResult
from llm_saia.verbs import (
    ask,
    constrain,
    critique,
    decompose,
    ground,
    recall,
    refine,
    store,
    synthesize,
    verify,
)
from llm_saia.verbs.decompose import DecomposeResult
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class TestAsk:
    async def test_ask_includes_artifact_and_question(self, mock_backend: MockBackend) -> None:
        result = await ask(mock_backend, "test artifact", "what is this?")

        assert "test artifact" in mock_backend.last_prompt
        assert "what is this?" in mock_backend.last_prompt
        assert result == "mock response"

    async def test_ask_returns_completion(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("custom answer")
        result = await ask(mock_backend, "artifact", "question")

        assert result == "custom answer"


class TestConstrain:
    async def test_constrain_uses_structured_output(self, mock_backend: MockBackend) -> None:
        @dataclass
        class TestSchema:
            value: str

        mock_backend.set_structured_response(TestSchema, TestSchema(value="parsed"))
        result = await constrain(mock_backend, "raw response", TestSchema)

        assert result.value == "parsed"
        assert "raw response" in mock_backend.last_prompt


class TestVerify:
    async def test_verify_returns_result(self, mock_backend: MockBackend) -> None:
        result = await verify(mock_backend, "claim", "is accurate")

        assert isinstance(result, VerifyResult)
        assert result.passed is True
        assert result.reason == "test reason"

    async def test_verify_includes_artifact_and_predicate(self, mock_backend: MockBackend) -> None:
        await verify(mock_backend, "my claim", "factually correct")

        assert "my claim" in mock_backend.last_prompt
        assert "factually correct" in mock_backend.last_prompt


class TestCritique:
    async def test_critique_returns_result(self, mock_backend: MockBackend) -> None:
        result = await critique(mock_backend, "argument to critique")

        assert isinstance(result, Critique)
        assert result.counter_argument == "test counter"
        assert result.weaknesses == ["weakness 1"]
        assert result.strength == 0.5

    async def test_critique_includes_artifact(self, mock_backend: MockBackend) -> None:
        await critique(mock_backend, "specific argument")

        assert "specific argument" in mock_backend.last_prompt


class TestRefine:
    async def test_refine_includes_artifact_and_feedback(self, mock_backend: MockBackend) -> None:
        mock_backend.set_complete_response("refined artifact")
        result = await refine(mock_backend, "original", "make it better")

        assert result == "refined artifact"
        assert "original" in mock_backend.last_prompt
        assert "make it better" in mock_backend.last_prompt


class TestSynthesize:
    async def test_synthesize_includes_all_artifacts(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Summary:
            combined: str

        mock_backend.set_structured_response(Summary, Summary(combined="synthesis"))
        result = await synthesize(mock_backend, ["art1", "art2", "art3"], Summary)

        assert result.combined == "synthesis"
        assert "art1" in mock_backend.last_prompt
        assert "art2" in mock_backend.last_prompt
        assert "art3" in mock_backend.last_prompt


class TestGround:
    async def test_ground_returns_evidence_per_source(self, mock_backend: MockBackend) -> None:
        result = await ground(mock_backend, "hypothesis", ["source1", "source2"])

        assert len(result) == 2
        assert all(isinstance(e, Evidence) for e in result)

    async def test_ground_includes_artifact_and_source(self, mock_backend: MockBackend) -> None:
        await ground(mock_backend, "my hypothesis", ["my source"])

        assert "my hypothesis" in mock_backend.last_prompt
        assert "my source" in mock_backend.last_prompt


class TestDecompose:
    async def test_decompose_returns_subtasks(self, mock_backend: MockBackend) -> None:
        mock_backend.set_structured_response(
            DecomposeResult, DecomposeResult(subtasks=["step1", "step2"])
        )
        result = await decompose(mock_backend, "complex task")

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
