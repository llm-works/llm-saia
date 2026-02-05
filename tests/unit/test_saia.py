"""Tests for the main SAIA class."""

from dataclasses import dataclass

import pytest

from llm_saia.core.types import ChooseResult, ClassifyResult, ConfirmResult, Critique, VerifyResult
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

        assert result == "the answer"

    async def test_extract(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Output:
            data: str

        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(Output, Output(data="extracted"))

        result = await saia.extract("raw content", Output)

        assert result.data == "extracted"

    async def test_constrain(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("constrained output")

        result = await saia.constrain("text", ["rule1", "rule2"])

        assert result == "constrained output"

    async def test_classify(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.classify("text", ["cat_a", "cat_b"])

        assert isinstance(result, ClassifyResult)
        assert result.category == "test_category"

    async def test_confirm(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.confirm("claim")

        assert isinstance(result, ConfirmResult)
        assert result.confirmed is True

    async def test_choose(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.choose(["option_a", "option_b"])

        assert isinstance(result, ChooseResult)
        assert result.choice == "option_a"

    async def test_instruct(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("Done.")

        result = await saia.instruct("Complete the task")

        assert result == "Done."

    async def test_verify(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.verify("claim", "predicate")

        assert isinstance(result, VerifyResult)
        assert result.passed is True

    async def test_critique(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.critique("argument")

        assert isinstance(result, Critique)
        assert result.counter_argument == "test counter"

    async def test_refine(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_complete_response("improved")

        result = await saia.refine("original", "feedback")

        assert result == "improved"

    async def test_synthesize(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Combined:
            result: str

        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(Combined, Combined(result="merged"))

        result = await saia.synthesize(["a", "b"], Combined)

        assert result.result == "merged"

    async def test_ground(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = await saia.ground("hypothesis", ["source1"])

        assert len(result) == 1
        assert result[0].content == "test content"

    async def test_decompose(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)
        mock_backend.set_structured_response(
            DecomposeResult, DecomposeResult(subtasks=["task1", "task2"])
        )

        result = await saia.decompose("big task")

        assert result == ["task1", "task2"]

    def test_store_and_recall(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        saia.store("key", "value")
        result = saia.recall("key")

        assert result == ["value"]

    def test_recall_empty(self, mock_backend: MockBackend) -> None:
        saia = make_saia(mock_backend)

        result = saia.recall("nonexistent")

        assert result == []
