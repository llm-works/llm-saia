"""Tests for OpenClaw backend implementation."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_saia.backends.openclaw import OpenClawBackend

pytestmark = pytest.mark.unit


class TestOpenClawBackendInit:
    def test_default_gateway_url(self) -> None:
        backend = OpenClawBackend()
        assert backend._gateway_url == "http://127.0.0.1:18789"

    def test_custom_gateway_url(self) -> None:
        backend = OpenClawBackend(gateway_url="http://192.168.1.100:8080")
        assert backend._gateway_url == "http://192.168.1.100:8080"

    def test_token_from_param(self) -> None:
        backend = OpenClawBackend(token="test-token")
        assert backend._token == "test-token"

    def test_gateway_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://env-gateway:9000")
        backend = OpenClawBackend()
        assert backend._gateway_url == "http://env-gateway:9000"

    def test_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "env-token")
        backend = OpenClawBackend()
        assert backend._token == "env-token"

    def test_param_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://env-gateway:9000")
        backend = OpenClawBackend(gateway_url="http://param-gateway:8000")
        assert backend._gateway_url == "http://param-gateway:8000"


class TestOpenClawBackendComplete:
    @pytest.fixture
    def mock_response(self) -> MagicMock:
        response = MagicMock()
        response.json.return_value = {"ok": True, "result": {"text": "Hello, world!"}}
        response.raise_for_status = MagicMock()
        return response

    async def test_complete_returns_text(self, mock_response: MagicMock) -> None:
        backend = OpenClawBackend()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await backend.complete("test prompt")

            assert result == "Hello, world!"
            mock_client.post.assert_called_once_with(
                "/tools/invoke",
                json={
                    "tool": "llm-task",
                    "args": {"action": "text", "prompt": "test prompt", "max_tokens": 4096},
                },
            )

    async def test_complete_extracts_from_output(self) -> None:
        backend = OpenClawBackend()
        response = MagicMock()
        response.json.return_value = {"ok": True, "result": {"output": "output text"}}
        response.raise_for_status = MagicMock()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_get_client.return_value = mock_client

            result = await backend.complete("prompt")
            assert result == "output text"

    async def test_complete_extracts_from_details(self) -> None:
        backend = OpenClawBackend()
        response = MagicMock()
        response.json.return_value = {
            "ok": True,
            "result": {"details": {"text": "details text"}},
        }
        response.raise_for_status = MagicMock()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_get_client.return_value = mock_client

            result = await backend.complete("prompt")
            assert result == "details text"

    async def test_complete_raises_on_empty_response(self) -> None:
        backend = OpenClawBackend()
        response = MagicMock()
        response.json.return_value = {"ok": True, "result": {}}
        response.raise_for_status = MagicMock()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="No text in OpenClaw response"):
                await backend.complete("prompt")


class TestOpenClawBackendCompleteStructured:
    @dataclass
    class SampleSchema:
        """Sample schema for structured output."""

        name: str
        value: int

    async def test_complete_structured_returns_dataclass(self) -> None:
        backend = OpenClawBackend()
        response = MagicMock()
        response.json.return_value = {
            "ok": True,
            "result": {"json": {"name": "test", "value": 42}},
        }
        response.raise_for_status = MagicMock()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_get_client.return_value = mock_client

            result = await backend.complete_structured("prompt", self.SampleSchema)

            assert isinstance(result, self.SampleSchema)
            assert result.name == "test"
            assert result.value == 42

    async def test_complete_structured_sends_schema(self) -> None:
        backend = OpenClawBackend()
        response = MagicMock()
        response.json.return_value = {
            "ok": True,
            "result": {"json": {"name": "test", "value": 42}},
        }
        response.raise_for_status = MagicMock()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_get_client.return_value = mock_client

            await backend.complete_structured("prompt", self.SampleSchema)

            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["tool"] == "llm-task"
            assert call_args[1]["json"]["args"]["action"] == "json"
            assert "schema" in call_args[1]["json"]["args"]

    async def test_complete_structured_extracts_from_output(self) -> None:
        backend = OpenClawBackend()
        response = MagicMock()
        response.json.return_value = {
            "ok": True,
            "result": {"output": {"name": "output", "value": 99}},
        }
        response.raise_for_status = MagicMock()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_get_client.return_value = mock_client

            result = await backend.complete_structured("prompt", self.SampleSchema)
            assert result.name == "output"
            assert result.value == 99


class TestOpenClawBackendInvokeTool:
    async def test_invoke_tool_raises_on_error(self) -> None:
        backend = OpenClawBackend()
        response = MagicMock()
        response.json.return_value = {"ok": False, "error": "Tool not found"}
        response.raise_for_status = MagicMock()

        with patch.object(backend, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="OpenClaw tool invocation failed"):
                await backend._invoke_tool("unknown-tool", {})


class TestOpenClawBackendContextManager:
    async def test_context_manager(self) -> None:
        async with OpenClawBackend() as backend:
            assert backend._client is None  # Client created lazily

    async def test_close(self) -> None:
        backend = OpenClawBackend()
        # Create a mock client
        mock_client = AsyncMock()
        backend._client = mock_client

        await backend.close()

        mock_client.aclose.assert_called_once()
        assert backend._client is None

    async def test_close_acquires_lock(self) -> None:
        backend = OpenClawBackend()
        mock_client = AsyncMock()
        backend._client = mock_client

        # Track if close() properly acquires the lock
        lock_acquired = False
        original_lock = backend._client_lock

        class TrackedLock:
            async def __aenter__(self) -> None:
                nonlocal lock_acquired
                lock_acquired = True
                await original_lock.__aenter__()

            async def __aexit__(self, *args: object) -> None:
                await original_lock.__aexit__(*args)

        backend._client_lock = TrackedLock()  # type: ignore[assignment]
        await backend.close()

        assert lock_acquired, "close() should acquire the lock"

    async def test_close_idempotent(self) -> None:
        backend = OpenClawBackend()
        mock_client = AsyncMock()
        backend._client = mock_client

        # Calling close twice should be safe
        await backend.close()
        await backend.close()

        mock_client.aclose.assert_called_once()
