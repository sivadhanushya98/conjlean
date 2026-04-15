"""
Tests for conjlean.models — LLM client factory and provider-specific clients.

All external SDK calls (Anthropic, OpenAI, Gemini) are mocked so no real API
requests are made. Tests verify correct message routing, response extraction,
batch fan-out, and rate-limit retry behaviour.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conjlean.config import ConjLeanConfig
from conjlean.models import (
    AnthropicClient,
    GeminiClient,
    HuggingFaceClient,
    OpenAIClient,
    VLLMClient,
    create_client,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str) -> MagicMock:
    """Build a mock Anthropic SDK response object."""
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.usage = MagicMock(input_tokens=10, output_tokens=5)
    return resp


def _make_openai_response(text: str) -> MagicMock:
    """Build a mock OpenAI SDK response object."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return resp


def _make_gemini_response(text: str) -> MagicMock:
    """Build a mock google.generativeai response object."""
    resp = MagicMock()
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


class TestAnthropicClient:
    """Tests for AnthropicClient.complete and complete_batch."""

    @pytest.fixture()
    def mock_async_anthropic(self) -> MagicMock:
        """Patch anthropic.AsyncAnthropic with a mock that returns a canned response."""
        mock_sdk = MagicMock()
        mock_sdk.messages.create = AsyncMock(
            return_value=_make_anthropic_response("Hello from Anthropic")
        )
        return mock_sdk

    @pytest.mark.asyncio()
    async def test_complete_returns_string(self, mock_async_anthropic: MagicMock) -> None:
        """AnthropicClient.complete returns the assistant text as a plain string."""
        with patch("anthropic.AsyncAnthropic", return_value=mock_async_anthropic), \
             patch("anthropic.RateLimitError", Exception):
            client = AnthropicClient(api_key="test-key", model="claude-sonnet-4-6")
            client._client = mock_async_anthropic

        result = await client.complete(
            messages=[{"role": "user", "content": "Formalize this"}],
            temperature=0.2,
            max_tokens=512,
        )
        assert result == "Hello from Anthropic"

    @pytest.mark.asyncio()
    async def test_complete_strips_system_message(self, mock_async_anthropic: MagicMock) -> None:
        """System messages are passed via the 'system' kwarg, not in the messages list."""
        with patch("anthropic.AsyncAnthropic", return_value=mock_async_anthropic), \
             patch("anthropic.RateLimitError", Exception):
            client = AnthropicClient(api_key="test-key", model="claude-sonnet-4-6")
            client._client = mock_async_anthropic

        await client.complete(
            messages=[
                {"role": "system", "content": "You are a math assistant."},
                {"role": "user", "content": "Prove it."},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        call_kwargs = mock_async_anthropic.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a math assistant."
        non_system_roles = [m["role"] for m in call_kwargs["messages"]]
        assert "system" not in non_system_roles

    @pytest.mark.asyncio()
    async def test_complete_batch_returns_all_results(self, mock_async_anthropic: MagicMock) -> None:
        """complete_batch returns one result per input message list."""
        with patch("anthropic.AsyncAnthropic", return_value=mock_async_anthropic), \
             patch("anthropic.RateLimitError", Exception):
            client = AnthropicClient(api_key="test-key", model="claude-sonnet-4-6")
            client._client = mock_async_anthropic

        messages_list = [[{"role": "user", "content": f"msg {i}"}] for i in range(4)]
        results = await client.complete_batch(messages_list, temperature=0.2, max_tokens=256)
        assert len(results) == 4
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio()
    async def test_rate_limit_retry(self) -> None:
        """AnthropicClient retries once on RateLimitError then succeeds."""

        class _FakeRateLimitError(Exception):
            pass

        call_count = 0

        async def _create(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeRateLimitError("rate limited")
            return _make_anthropic_response("retry success")

        mock_sdk = MagicMock()
        mock_sdk.messages.create = _create

        with patch("anthropic.AsyncAnthropic", return_value=mock_sdk), \
             patch("anthropic.RateLimitError", _FakeRateLimitError), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            client = AnthropicClient(api_key="key", model="claude-sonnet-4-6")
            client._client = mock_sdk
            client._RateLimitError = _FakeRateLimitError

        result = await client.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=64,
        )
        assert result == "retry success"
        assert call_count == 2

    @pytest.mark.asyncio()
    async def test_rate_limit_exhausted_raises_runtime_error(self) -> None:
        """After _MAX_RETRIES rate limit errors, RuntimeError is raised."""

        class _FakeRateLimitError(Exception):
            pass

        async def _always_rate_limit(**kwargs: Any) -> None:
            raise _FakeRateLimitError("always limited")

        mock_sdk = MagicMock()
        mock_sdk.messages.create = _always_rate_limit

        with patch("anthropic.AsyncAnthropic", return_value=mock_sdk), \
             patch("anthropic.RateLimitError", _FakeRateLimitError), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            client = AnthropicClient(api_key="key", model="claude-sonnet-4-6")
            client._client = mock_sdk
            client._RateLimitError = _FakeRateLimitError

        with pytest.raises(RuntimeError, match="rate limit"):
            await client.complete(
                messages=[{"role": "user", "content": "test"}],
                temperature=0.0,
                max_tokens=64,
            )


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


class TestOpenAIClient:
    """Tests for OpenAIClient.complete and complete_batch."""

    @pytest.fixture()
    def mock_async_openai(self) -> MagicMock:
        """Patch openai.AsyncOpenAI with a mock that returns a canned response."""
        mock_sdk = MagicMock()
        mock_sdk.chat = MagicMock()
        mock_sdk.chat.completions = MagicMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_make_openai_response("Hello from OpenAI")
        )
        return mock_sdk

    @pytest.mark.asyncio()
    async def test_complete_returns_string(self, mock_async_openai: MagicMock) -> None:
        """OpenAIClient.complete returns the assistant text as a plain string."""
        with patch("openai.AsyncOpenAI", return_value=mock_async_openai), \
             patch("openai.RateLimitError", Exception):
            client = OpenAIClient(api_key="test-key", model="gpt-4")
            client._client = mock_async_openai

        result = await client.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=128,
        )
        assert result == "Hello from OpenAI"

    @pytest.mark.asyncio()
    async def test_complete_passes_messages_verbatim(self, mock_async_openai: MagicMock) -> None:
        """OpenAI messages list is passed through unchanged (no system stripping)."""
        with patch("openai.AsyncOpenAI", return_value=mock_async_openai), \
             patch("openai.RateLimitError", Exception):
            client = OpenAIClient(api_key="test-key", model="gpt-4")
            client._client = mock_async_openai

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        await client.complete(messages=msgs, temperature=0.0, max_tokens=64)
        call_kwargs = mock_async_openai.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 2

    @pytest.mark.asyncio()
    async def test_complete_batch_length(self, mock_async_openai: MagicMock) -> None:
        """complete_batch returns a result for each input message list."""
        with patch("openai.AsyncOpenAI", return_value=mock_async_openai), \
             patch("openai.RateLimitError", Exception):
            client = OpenAIClient(api_key="test-key", model="gpt-4")
            client._client = mock_async_openai

        msgs_list = [[{"role": "user", "content": "hello"}]] * 5
        results = await client.complete_batch(msgs_list, temperature=0.0, max_tokens=64)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# VLLMClient
# ---------------------------------------------------------------------------


class TestVLLMClient:
    """Tests for VLLMClient — thin wrapper around OpenAIClient."""

    def test_is_openai_client_subclass(self) -> None:
        """VLLMClient is a subclass of OpenAIClient."""
        assert issubclass(VLLMClient, OpenAIClient)

    def test_instantiates_with_base_url(self) -> None:
        """VLLMClient stores the custom base_url via OpenAIClient."""
        with patch("openai.AsyncOpenAI"), patch("openai.RateLimitError", Exception):
            client = VLLMClient(
                base_url="http://localhost:8000/v1",
                model="Qwen/Qwen2.5-Math-72B-Instruct",
            )
        assert client._model == "Qwen/Qwen2.5-Math-72B-Instruct"


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------


class TestGeminiClient:
    """Tests for GeminiClient message conversion and completion."""

    @pytest.fixture(autouse=True)
    def _patch_genai(self) -> None:
        """Inject a fake google.generativeai module into sys.modules for all tests."""
        import sys

        mock_genai = MagicMock()
        mock_genai.types = MagicMock()
        mock_genai.types.GenerationConfig = MagicMock()
        mock_genai.GenerativeModel = MagicMock()

        # Insert both the top-level google and the sub-module so the import
        # inside GeminiClient.__init__ succeeds regardless of install state.
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai
        sys.modules.setdefault("google", mock_google)
        sys.modules["google.generativeai"] = mock_genai

        self._mock_genai = mock_genai
        yield
        # Tear down: only remove the sub-module we injected; leave "google" alone
        sys.modules.pop("google.generativeai", None)

    @pytest.mark.asyncio()
    async def test_complete_returns_string(self) -> None:
        """GeminiClient.complete returns the model reply text."""
        with patch("asyncio.to_thread", new_callable=AsyncMock, return_value="Gemini reply"):
            client = GeminiClient(api_key="gemini-test", model="gemini-1.5-pro")
            client._genai = self._mock_genai

            result = await client.complete(
                messages=[{"role": "user", "content": "Prove this theorem."}],
                temperature=0.1,
                max_tokens=512,
            )
        assert result == "Gemini reply"

    def test_convert_messages_system_extraction(self) -> None:
        """_convert_messages extracts system content into system_instruction."""
        msgs = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Explain this."},
            {"role": "assistant", "content": "Here is the explanation."},
        ]
        system_instr, contents = GeminiClient._convert_messages(msgs)
        assert system_instr == "You are a math tutor."
        roles = [c["role"] for c in contents]
        assert "user" in roles
        assert "model" in roles
        assert "system" not in roles

    def test_convert_messages_user_becomes_user(self) -> None:
        """User messages are preserved as Gemini 'user' role."""
        msgs = [{"role": "user", "content": "Hello"}]
        _, contents = GeminiClient._convert_messages(msgs)
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == ["Hello"]

    def test_convert_messages_assistant_becomes_model(self) -> None:
        """Assistant messages are mapped to Gemini 'model' role."""
        msgs = [{"role": "assistant", "content": "Answer"}]
        _, contents = GeminiClient._convert_messages(msgs)
        assert contents[0]["role"] == "model"


# ---------------------------------------------------------------------------
# create_client factory
# ---------------------------------------------------------------------------


class TestCreateClient:
    """Tests for the create_client factory function."""

    def test_anthropic_provider_returns_anthropic_client(self) -> None:
        """create_client returns AnthropicClient for provider='anthropic'."""
        cfg = ConjLeanConfig(provider="anthropic")
        cfg.api_keys.anthropic = "sk-test"
        with patch("anthropic.AsyncAnthropic"), patch("anthropic.RateLimitError", Exception):
            client = create_client(cfg)
        assert isinstance(client, AnthropicClient)

    def test_openai_provider_returns_openai_client(self) -> None:
        """create_client returns OpenAIClient for provider='openai'."""
        cfg = ConjLeanConfig(provider="openai")
        cfg.api_keys.openai = "sk-openai-test"
        with patch("openai.AsyncOpenAI"), patch("openai.RateLimitError", Exception):
            client = create_client(cfg)
        assert isinstance(client, OpenAIClient)

    def test_vllm_provider_returns_vllm_client(self) -> None:
        """create_client returns VLLMClient for provider='vllm'."""
        cfg = ConjLeanConfig(provider="vllm")
        with patch("openai.AsyncOpenAI"), patch("openai.RateLimitError", Exception):
            client = create_client(cfg)
        assert isinstance(client, VLLMClient)

    def test_gemini_provider_returns_gemini_client(self) -> None:
        """create_client returns GeminiClient for provider='gemini'."""
        import sys

        mock_genai = MagicMock()
        mock_genai.types = MagicMock()
        mock_genai.types.GenerationConfig = MagicMock()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai
        sys.modules.setdefault("google", mock_google)
        sys.modules["google.generativeai"] = mock_genai

        try:
            cfg = ConjLeanConfig(provider="gemini")
            cfg.api_keys.gemini = "gemini-key"
            client = create_client(cfg)
            assert isinstance(client, GeminiClient)
        finally:
            sys.modules.pop("google.generativeai", None)

    def test_missing_anthropic_key_raises_runtime_error(self) -> None:
        """create_client raises RuntimeError when Anthropic key is absent."""
        cfg = ConjLeanConfig(provider="anthropic")
        # Ensure no env var override
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        cfg.api_keys.anthropic = ""
        with pytest.raises(RuntimeError, match="Anthropic API key is required"):
            create_client(cfg)

    def test_missing_openai_key_raises_runtime_error(self) -> None:
        """create_client raises RuntimeError when OpenAI key is absent."""
        cfg = ConjLeanConfig(provider="openai")
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        cfg.api_keys.openai = ""
        with pytest.raises(RuntimeError, match="OpenAI API key is required"):
            create_client(cfg)

    def test_wrong_type_raises_type_error(self) -> None:
        """create_client raises TypeError when config is not ConjLeanConfig."""
        with pytest.raises(TypeError):
            create_client({"provider": "anthropic"})  # type: ignore[arg-type]

    @pytest.mark.asyncio()
    async def test_complete_batch_uses_gather(self) -> None:
        """complete_batch resolves N concurrent tasks using asyncio.gather."""
        call_count = 0

        class _CountingClient(OpenAIClient):
            async def complete(self, messages, temperature, max_tokens):  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                return f"result_{call_count}"

        with patch("openai.AsyncOpenAI"), patch("openai.RateLimitError", Exception):
            client = _CountingClient(api_key="key", model="gpt-4")

        n = 7
        msgs_list = [[{"role": "user", "content": f"q{i}"}] for i in range(n)]
        results = await client.complete_batch(msgs_list, temperature=0.0, max_tokens=64)
        assert len(results) == n
        assert call_count == n
