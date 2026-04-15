"""
Multi-provider asynchronous LLM client for the ConjLean pipeline.

Supports Anthropic, OpenAI, Google Gemini, Hugging Face Inference API, and
self-hosted vLLM (OpenAI-compatible).  All clients share the same abstract
interface so pipeline stages can swap providers via configuration without
changing call sites.

Rate-limit errors trigger automatic exponential-backoff retry (up to 3
attempts) for every provider.  Batch completion uses ``asyncio.gather`` for
concurrent fan-out, maximising throughput during generation and proof-search
stages.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _backoff_delay(attempt: int) -> float:
    """
    Compute exponential-backoff delay in seconds for a given attempt index.

    Args:
        attempt: Zero-based attempt index (0 → first retry).

    Returns:
        Seconds to sleep before the next attempt.
    """
    return _BACKOFF_BASE_SECONDS ** (attempt + 1)


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Heuristically detect rate-limit / quota exceptions across provider SDKs.

    Args:
        exc: Exception raised by a provider SDK call.

    Returns:
        True if the exception looks like a rate-limit or transient quota error.
    """
    type_name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "ratelimit" in type_name
        or "rate_limit" in type_name
        or "429" in message
        or "resource_exhausted" in message
        or "quota" in message
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """
    Abstract interface for all LLM provider clients.

    Concrete subclasses must implement ``complete`` (single completion) and
    ``complete_batch`` (parallel fan-out).  All completions are async to
    prevent blocking the event loop during I/O-bound API calls.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Request a single chat completion.

        Args:
            messages: OpenAI-style message list
                (``[{"role": "...", "content": "..."}]``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the completion.

        Returns:
            The assistant's reply as a plain string.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """

    @abstractmethod
    async def complete_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """
        Request completions for multiple message lists concurrently.

        Implementations must use ``asyncio.gather`` to fan out requests.

        Args:
            messages_list: A list where each element is a message list
                compatible with ``complete``.
            temperature: Sampling temperature applied to all requests.
            max_tokens: Maximum tokens per completion.

        Returns:
            List of assistant reply strings, in the same order as input.
        """


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """
    Async LLM client backed by the Anthropic SDK (``AsyncAnthropic``).

    Args:
        api_key: Anthropic API key.
        model: Model identifier (e.g. ``claude-sonnet-4-6``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from anthropic import AsyncAnthropic, RateLimitError
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._RateLimitError = RateLimitError

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Send a chat completion request to the Anthropic Messages API.

        System messages are extracted and passed via the dedicated ``system``
        parameter; the remaining messages form the ``messages`` list.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the completion.

        Returns:
            The assistant reply text.
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        system_text: str | Any = system_parts[0] if system_parts else ""

        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=non_system,
        )
        if system_text:
            kwargs["system"] = system_text

        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    "Anthropic request | model=%s | messages=%d | max_tokens=%d",
                    self._model,
                    len(non_system),
                    max_tokens,
                )
                response = await self._client.messages.create(**kwargs)
                content = response.content[0].text if response.content else ""
                logger.debug(
                    "Anthropic response | input_tokens=%d | output_tokens=%d",
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
                return content
            except self._RateLimitError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"Anthropic rate limit exceeded after {_MAX_RETRIES} retries"
                    ) from exc
                delay = _backoff_delay(attempt)
                logger.warning("Anthropic rate limit hit; retrying in %.1fs (attempt %d)", delay, attempt + 1)
                await asyncio.sleep(delay)

        raise RuntimeError("Unreachable: retry loop exhausted without raising")

    async def complete_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """
        Concurrently complete multiple message lists via ``asyncio.gather``.

        Args:
            messages_list: List of message lists.
            temperature: Sampling temperature for all requests.
            max_tokens: Max tokens per completion.

        Returns:
            Ordered list of assistant reply strings.
        """
        tasks = [self.complete(msgs, temperature, max_tokens) for msgs in messages_list]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    """
    Async LLM client backed by the OpenAI SDK (``AsyncOpenAI``).

    Supports both the public OpenAI API and any OpenAI-compatible endpoint
    (e.g. vLLM) via the ``base_url`` override.

    Args:
        api_key: OpenAI API key (or placeholder for vLLM).
        model: Model identifier.
        base_url: Optional custom base URL (overrides the default OpenAI endpoint).
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI, RateLimitError
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._model = model
        self._RateLimitError = RateLimitError

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Send a chat completion request to the OpenAI Chat Completions API.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the completion.

        Returns:
            The assistant reply text.
        """
        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    "OpenAI request | model=%s | messages=%d | max_tokens=%d",
                    self._model,
                    len(messages),
                    max_tokens,
                )
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                usage = response.usage
                if usage:
                    logger.debug(
                        "OpenAI response | prompt_tokens=%d | completion_tokens=%d",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    )
                return content
            except self._RateLimitError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"OpenAI rate limit exceeded after {_MAX_RETRIES} retries"
                    ) from exc
                delay = _backoff_delay(attempt)
                logger.warning("OpenAI rate limit hit; retrying in %.1fs (attempt %d)", delay, attempt + 1)
                await asyncio.sleep(delay)

        raise RuntimeError("Unreachable: retry loop exhausted without raising")

    async def complete_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """
        Concurrently complete multiple message lists via ``asyncio.gather``.

        Args:
            messages_list: List of message lists.
            temperature: Sampling temperature for all requests.
            max_tokens: Max tokens per completion.

        Returns:
            Ordered list of assistant reply strings.
        """
        tasks = [self.complete(msgs, temperature, max_tokens) for msgs in messages_list]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# vLLM (OpenAI-compatible)
# ---------------------------------------------------------------------------


class VLLMClient(OpenAIClient):
    """
    Thin subclass of ``OpenAIClient`` targeting a self-hosted vLLM server.

    The vLLM server exposes an OpenAI-compatible API, so the only difference
    is pointing ``base_url`` at the local server and using the vLLM model name.

    Args:
        base_url: Base URL of the vLLM server (e.g. ``http://localhost:8000/v1``).
        model: Model name as registered in the vLLM server.
        api_key: Placeholder API key (vLLM does not enforce keys by default).
    """

    def __init__(self, base_url: str, model: str, api_key: str = "vllm") -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiClient(LLMClient):
    """
    Async LLM client backed by the ``google-generativeai`` SDK.

    Converts OpenAI-style ``system`` / ``user`` / ``assistant`` message dicts
    to Gemini's ``contents`` format before each API call.

    Args:
        api_key: Google Gemini API key.
        model: Gemini model identifier (e.g. ``gemini-1.5-pro``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiClient. "
                "Install it with: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """
        Convert OpenAI-style messages to Gemini's content format.

        System messages are concatenated into a single system instruction
        string.  User and assistant turns are mapped to Gemini ``"user"`` and
        ``"model"`` roles respectively.

        Args:
            messages: OpenAI-style message list.

        Returns:
            Tuple of ``(system_instruction, gemini_contents)`` where
            ``gemini_contents`` is a list of ``{"role": ..., "parts": [...]}``
            dicts suitable for ``GenerativeModel.generate_content``.
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system_instruction = "\n\n".join(system_parts)

        role_map = {"user": "user", "assistant": "model"}
        gemini_contents = [
            {"role": role_map[m["role"]], "parts": [m["content"]]}
            for m in messages
            if m["role"] in role_map
        ]
        return system_instruction, gemini_contents

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Send a completion request to the Gemini GenerativeAI API.

        The SDK call is wrapped in ``asyncio.to_thread`` because the
        ``google-generativeai`` library is synchronous.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            The model reply text.
        """
        system_instruction, contents = self._convert_messages(messages)
        generation_config = self._genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model_kwargs: dict[str, Any] = {"model_name": self._model_name}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction

        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    "Gemini request | model=%s | turns=%d | max_tokens=%d",
                    self._model_name,
                    len(contents),
                    max_tokens,
                )

                def _sync_call() -> str:
                    model_instance = self._genai.GenerativeModel(**model_kwargs)
                    response = model_instance.generate_content(
                        contents,
                        generation_config=generation_config,
                    )
                    return response.text

                text = await asyncio.to_thread(_sync_call)
                logger.debug("Gemini response received | model=%s", self._model_name)
                return text
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    if attempt == _MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"Gemini rate limit exceeded after {_MAX_RETRIES} retries"
                        ) from exc
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Gemini rate limit hit; retrying in %.1fs (attempt %d)", delay, attempt + 1
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise RuntimeError("Unreachable: retry loop exhausted without raising")

    async def complete_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """
        Concurrently complete multiple message lists via ``asyncio.gather``.

        Args:
            messages_list: List of message lists.
            temperature: Sampling temperature for all requests.
            max_tokens: Max tokens per completion.

        Returns:
            Ordered list of model reply strings.
        """
        tasks = [self.complete(msgs, temperature, max_tokens) for msgs in messages_list]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


class HuggingFaceClient(LLMClient):
    """
    Async LLM client backed by the Hugging Face ``InferenceClient``.

    The HF InferenceClient is synchronous; calls are dispatched in a thread
    pool via ``asyncio.to_thread`` to keep the event loop non-blocking.

    Args:
        token: Hugging Face hub API token.
        model: Model repository ID (e.g. ``meta-llama/Llama-3-70b-instruct``).
    """

    def __init__(self, token: str, model: str) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_hub' package is required for HuggingFaceClient. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        self._client = InferenceClient(model=model, token=token)
        self._model = model

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Send a chat completion request to the Hugging Face Inference API.

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens in the completion.

        Returns:
            The assistant reply text.
        """
        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    "HuggingFace request | model=%s | messages=%d | max_tokens=%d",
                    self._model,
                    len(messages),
                    max_tokens,
                )

                def _sync_call() -> str:
                    response = self._client.chat_completion(
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or ""

                text = await asyncio.to_thread(_sync_call)
                logger.debug("HuggingFace response received | model=%s", self._model)
                return text
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    if attempt == _MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"HuggingFace rate limit exceeded after {_MAX_RETRIES} retries"
                        ) from exc
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "HuggingFace rate limit hit; retrying in %.1fs (attempt %d)",
                        delay,
                        attempt + 1,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise RuntimeError("Unreachable: retry loop exhausted without raising")

    async def complete_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """
        Concurrently complete multiple message lists via ``asyncio.gather``.

        Args:
            messages_list: List of message lists.
            temperature: Sampling temperature for all requests.
            max_tokens: Max tokens per completion.

        Returns:
            Ordered list of assistant reply strings.
        """
        tasks = [self.complete(msgs, temperature, max_tokens) for msgs in messages_list]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Local HuggingFace (transformers, direct GPU)
# ---------------------------------------------------------------------------


class LocalHFClient(LLMClient):
    """
    Async LLM client that loads a HuggingFace model locally via ``transformers``.

    Designed for Lambda Labs / local GPU inference where no external API is
    needed.  The model is loaded once at construction time with
    ``device_map="auto"`` so multi-GPU setups are handled automatically.
    Inference runs inside ``asyncio.to_thread`` to keep the event loop
    non-blocking.

    Args:
        model: HuggingFace model repository ID
            (e.g. ``"Qwen/Qwen2.5-Math-7B-Instruct"``).
        torch_dtype: PyTorch dtype string — ``"bfloat16"`` (default, best for
            A100/H100), ``"float16"``, or ``"float32"``.
        device_map: Passed directly to ``from_pretrained`` — ``"auto"``
            distributes layers across all visible GPUs / CPU.
        max_new_tokens: Hard cap on generated tokens (overrides ``max_tokens``
            when the caller passes a larger value).
        load_in_4bit: Enable bitsandbytes 4-bit quantisation (halves VRAM,
            small quality hit). Requires ``bitsandbytes`` to be installed.
        load_in_8bit: Enable bitsandbytes 8-bit quantisation. Mutually
            exclusive with ``load_in_4bit``.
        trust_remote_code: Passed to ``from_pretrained``; required by some
            models (e.g. Qwen).
    """

    def __init__(
        self,
        model: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 4096,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for LocalHFClient. "
                "Install them with: pip install transformers torch"
            ) from exc

        import torch  # re-import for dtype resolution

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(
                f"torch_dtype must be one of {list(dtype_map)}, got {torch_dtype!r}"
            )
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive.")

        logger.info(
            "Loading local HF model | model=%s | dtype=%s | device_map=%s | 4bit=%s | 8bit=%s",
            model,
            torch_dtype,
            device_map,
            load_in_4bit,
            load_in_8bit,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        # Pad on the left for batch generation; use eos as pad if no explicit pad token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype_map[torch_dtype],
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore[attr-defined]
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for 4-bit quantisation: pip install bitsandbytes"
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore[attr-defined]
            except ImportError as exc:
                raise ImportError(
                    "bitsandbytes is required for 8-bit quantisation: pip install bitsandbytes"
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self._model_obj = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self._tokenizer = tokenizer
        self._model_name = model
        self._max_new_tokens = max_new_tokens
        self._torch = torch

        logger.info("Local HF model loaded | model=%s", model)

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """
        Apply the model's built-in chat template to an OpenAI-style message list.

        Falls back to a simple concatenation if the tokenizer does not have a
        registered chat template.

        Args:
            messages: OpenAI-style ``[{"role": ..., "content": ...}]`` list.

        Returns:
            A single prompt string ready to tokenise and feed to the model.
        """
        if self._tokenizer.chat_template:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Minimal fallback: join turns with role prefixes.
        lines = []
        for m in messages:
            prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
                m["role"], m["role"].capitalize()
            )
            lines.append(f"{prefix}: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _generate_sync(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Synchronous single-sequence generation.  Called via ``asyncio.to_thread``.

        Args:
            prompt: Pre-tokenised prompt string.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Returns:
            Decoded assistant reply (prompt prefix stripped).
        """
        import torch  # local import for thread safety

        inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model_obj.device
        )
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": min(max_tokens, self._max_new_tokens),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model_obj.generate(**inputs, **gen_kwargs)

        # Strip the input prefix from the output.
        new_ids = output_ids[0][input_len:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    async def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Generate a completion asynchronously (inference runs in a thread pool).

        Args:
            messages: OpenAI-style message list.
            temperature: Sampling temperature (0 → greedy).
            max_tokens: Maximum new tokens to generate.

        Returns:
            The model's reply as a plain string.
        """
        prompt = self._apply_chat_template(messages)
        logger.debug(
            "LocalHF request | model=%s | prompt_chars=%d | max_tokens=%d",
            self._model_name,
            len(prompt),
            max_tokens,
        )
        reply = await asyncio.to_thread(self._generate_sync, prompt, temperature, max_tokens)
        logger.debug("LocalHF response | model=%s | reply_chars=%d", self._model_name, len(reply))
        return reply

    async def complete_batch(
        self,
        messages_list: list[list[dict]],
        temperature: float,
        max_tokens: int,
    ) -> list[str]:
        """
        Generate completions for multiple prompts concurrently.

        Each request runs in its own thread via ``asyncio.to_thread`` so the
        event loop is never blocked.  On a multi-GPU node vLLM is more
        efficient for true batch inference; this implementation is simpler and
        works on any transformer model without extra serving infrastructure.

        Args:
            messages_list: List of OpenAI-style message lists.
            temperature: Sampling temperature for all requests.
            max_tokens: Max new tokens per completion.

        Returns:
            Ordered list of assistant reply strings.
        """
        tasks = [self.complete(msgs, temperature, max_tokens) for msgs in messages_list]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_client(config: "ConjLeanConfig") -> LLMClient:  # noqa: F821
    """
    Instantiate the appropriate ``LLMClient`` for the active provider.

    The provider is read from ``config.provider``.  API keys and model names
    are resolved from the nested ``config.api_keys`` and ``config.models``
    objects.

    Args:
        config: A fully validated ``ConjLeanConfig`` instance.

    Returns:
        A concrete ``LLMClient`` ready to issue async completions.

    Raises:
        ValueError: If ``config.provider`` is not one of the supported values.
        RuntimeError: If the required API key is empty for the active provider.
    """
    from conjlean.config import ConjLeanConfig  # local import avoids circular dependency

    if not isinstance(config, ConjLeanConfig):
        raise TypeError(f"Expected ConjLeanConfig, got {type(config).__name__}")

    provider = config.get_active_provider()
    models = config.models
    keys = config.api_keys

    logger.info("Creating LLM client | provider=%s", provider)

    if provider == "anthropic":
        if not keys.anthropic:
            raise RuntimeError(
                "Anthropic API key is required. Set api_keys.anthropic in config.yaml "
                "or export ANTHROPIC_API_KEY."
            )
        return AnthropicClient(api_key=keys.anthropic, model=models.conjecture_gen)

    if provider == "openai":
        if not keys.openai:
            raise RuntimeError(
                "OpenAI API key is required. Set api_keys.openai in config.yaml "
                "or export OPENAI_API_KEY."
            )
        return OpenAIClient(api_key=keys.openai, model=models.conjecture_gen)

    if provider == "gemini":
        if not keys.gemini:
            raise RuntimeError(
                "Gemini API key is required. Set api_keys.gemini in config.yaml "
                "or export GEMINI_API_KEY."
            )
        return GeminiClient(api_key=keys.gemini, model=models.conjecture_gen)

    if provider == "huggingface":
        if not keys.huggingface:
            raise RuntimeError(
                "Hugging Face token is required. Set api_keys.huggingface in config.yaml "
                "or export HF_TOKEN."
            )
        return HuggingFaceClient(token=keys.huggingface, model=models.conjecture_gen)

    if provider == "vllm":
        return VLLMClient(
            base_url=config.vllm.base_url,
            model=config.vllm.model,
        )

    if provider == "local_hf":
        lhf = config.local_hf
        return LocalHFClient(
            model=lhf.model,
            torch_dtype=lhf.torch_dtype,
            device_map=lhf.device_map,
            max_new_tokens=lhf.max_new_tokens,
            load_in_4bit=lhf.load_in_4bit,
            load_in_8bit=lhf.load_in_8bit,
            trust_remote_code=lhf.trust_remote_code,
        )

    raise ValueError(
        f"Unsupported provider: {provider!r}. "
        "Must be one of: anthropic, openai, gemini, huggingface, vllm, local_hf."
    )
