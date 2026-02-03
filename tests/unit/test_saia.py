"""Tests for the main SAIA class."""

from dataclasses import dataclass

import pytest

from llm_saia import SAIA
from llm_saia.core.types import Critique, VerifyResult
from llm_saia.verbs.decompose import DecomposeResult
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class TestSAIA:
    def test_init(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)
        assert saia._backend is mock_backend
        assert saia._memory == {}

    async def test_ask(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)
        mock_backend.set_complete_response("the answer")

        result = await saia.ask("artifact", "question")

        assert result == "the answer"

    async def test_constrain(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Output:
            data: str

        saia = SAIA(backend=mock_backend)
        mock_backend.set_structured_response(Output, Output(data="parsed"))

        result = await saia.constrain("raw", Output)

        assert result.data == "parsed"

    async def test_verify(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)

        result = await saia.verify("claim", "predicate")

        assert isinstance(result, VerifyResult)
        assert result.passed is True

    async def test_critique(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)

        result = await saia.critique("argument")

        assert isinstance(result, Critique)
        assert result.counter_argument == "test counter"

    async def test_refine(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)
        mock_backend.set_complete_response("improved")

        result = await saia.refine("original", "feedback")

        assert result == "improved"

    async def test_synthesize(self, mock_backend: MockBackend) -> None:
        @dataclass
        class Combined:
            result: str

        saia = SAIA(backend=mock_backend)
        mock_backend.set_structured_response(Combined, Combined(result="merged"))

        result = await saia.synthesize(["a", "b"], Combined)

        assert result.result == "merged"

    async def test_ground(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)

        result = await saia.ground("hypothesis", ["source1"])

        assert len(result) == 1
        assert result[0].content == "test content"

    async def test_decompose(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)
        mock_backend.set_structured_response(
            DecomposeResult, DecomposeResult(subtasks=["task1", "task2"])
        )

        result = await saia.decompose("big task")

        assert result == ["task1", "task2"]

    def test_store_and_recall(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)

        saia.store("key", "value")
        result = saia.recall("key")

        assert result == ["value"]

    def test_recall_empty(self, mock_backend: MockBackend) -> None:
        saia = SAIA(backend=mock_backend)

        result = saia.recall("nonexistent")

        assert result == []
