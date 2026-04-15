"""
Pydantic v2-based configuration management for the ConjLean pipeline.

Loads settings from a YAML file and merges environment-variable overrides
for all API keys.  The canonical config schema mirrors ``configs/config.yaml``
exactly, providing field-level validation, typed access, and IDE auto-complete
throughout the codebase.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ProviderConfig(BaseModel):
    """
    API credentials for every supported LLM provider.

    Each field accepts a literal value from the YAML or falls back to the
    corresponding environment variable when the YAML value is empty.

    Attributes:
        anthropic: Anthropic API key (env: ``ANTHROPIC_API_KEY``).
        openai: OpenAI API key (env: ``OPENAI_API_KEY``).
        gemini: Google Gemini API key (env: ``GEMINI_API_KEY``).
        huggingface: Hugging Face hub token (env: ``HF_TOKEN``).
    """

    anthropic: str = Field(default="", description="Anthropic API key")
    openai: str = Field(default="", description="OpenAI API key")
    gemini: str = Field(default="", description="Google Gemini API key")
    huggingface: str = Field(default="", description="Hugging Face hub token")

    @model_validator(mode="after")
    def _apply_env_overrides(self) -> "ProviderConfig":
        """Merge environment-variable values for any empty API key fields."""
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "huggingface": "HF_TOKEN",
        }
        for field_name, env_var in env_map.items():
            current = getattr(self, field_name)
            if not current:
                env_val = os.environ.get(env_var, "")
                if env_val:
                    object.__setattr__(self, field_name, env_val)
                    logger.debug("Loaded %s from environment variable %s", field_name, env_var)
        return self


class VLLMConfig(BaseModel):
    """
    Connection settings for a vLLM-hosted OpenAI-compatible inference server.

    Attributes:
        base_url: Base URL of the vLLM server (e.g. ``http://localhost:8000/v1``).
        model: Model ID served by the vLLM instance.
    """

    base_url: str = Field(default="http://localhost:8000/v1", description="vLLM server base URL")
    model: str = Field(
        default="Qwen/Qwen2.5-Math-72B-Instruct", description="Model name served by vLLM"
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"vllm.base_url must be an HTTP(S) URL, got: {v!r}")
        return v.rstrip("/")


class LocalHFConfig(BaseModel):
    """
    Settings for direct local HuggingFace model inference via ``transformers``.

    Use this when running on Lambda Labs or any machine with local GPU access
    and no external API budget.  The model is downloaded from the HF Hub on
    first use and cached in ``~/.cache/huggingface/``.

    Recommended math models (set ``model`` to one of these):

    +-----------------------------------------+------+----------------------------+
    | Model                                   | VRAM | Notes                      |
    +=========================================+======+============================+
    | Qwen/Qwen2.5-Math-7B-Instruct           |  16G | 1× A100, best 7B math      |
    | Qwen/Qwen2.5-Math-72B-Instruct          | 160G | 2× A100 80G, SOTA math     |
    | deepseek-ai/DeepSeek-Math-7B-Instruct   |  16G | 1× A100, strong alt        |
    | Qwen/QwQ-32B                            |  80G | 1× A100 80G, reasoning     |
    | meta-llama/Meta-Llama-3.1-70B-Instruct  | 160G | 2× A100 80G, general       |
    +-----------------------------------------+------+----------------------------+

    Attributes:
        model: HuggingFace model ID.
        torch_dtype: Weight precision — ``"bfloat16"`` recommended for A100/H100.
        device_map: Passed to ``from_pretrained`` — ``"auto"`` auto-shards
            across all GPUs.
        max_new_tokens: Hard cap on tokens generated per call.
        load_in_4bit: Enable 4-bit quantisation (requires ``bitsandbytes``).
            Use to run 72B on 2× A100 40G.
        load_in_8bit: Enable 8-bit quantisation (requires ``bitsandbytes``).
        trust_remote_code: Required for Qwen models.
    """

    model: str = Field(
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        description="HuggingFace model repo ID",
    )
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16", description="Model weight dtype"
    )
    device_map: str = Field(default="auto", description="Device map for from_pretrained")
    max_new_tokens: int = Field(default=4096, ge=1, description="Max new tokens per call")
    load_in_4bit: bool = Field(default=False, description="Enable 4-bit quantisation")
    load_in_8bit: bool = Field(default=False, description="Enable 8-bit quantisation")
    trust_remote_code: bool = Field(default=True, description="Required for Qwen / custom models")


class ModelsConfig(BaseModel):
    """
    Model identifiers for each pipeline stage.

    Attributes:
        conjecture_gen: Model used for conjecture generation prompts.
        formalizer: Model used for Lean 4 formalization prompts.
        proof_gen: Model used for LLM-guided proof search (layer 3).
    """

    conjecture_gen: str = Field(default="claude-sonnet-4-6")
    formalizer: str = Field(default="claude-sonnet-4-6")
    proof_gen: str = Field(default="claude-sonnet-4-6")


class LeanConfig(BaseModel):
    """
    Settings for the Lean 4 / Mathlib environment.

    Attributes:
        project_dir: Path to the Lean project directory containing
            ``lakefile.toml`` (may be relative to the repo root).
        build_timeout: Maximum seconds allowed for a ``lake build`` invocation.
        repl_timeout: Default per-command timeout (seconds) for the REPL.
    """

    project_dir: str = Field(default="./lean", description="Path to Lean 4 project root")
    build_timeout: int = Field(default=300, ge=1, description="lake build timeout in seconds")
    repl_timeout: int = Field(default=30, ge=1, description="REPL per-command timeout in seconds")

    @field_validator("build_timeout", "repl_timeout")
    @classmethod
    def _positive_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timeout values must be positive integers")
        return v


class PipelineConfig(BaseModel):
    """
    High-level parameters controlling the end-to-end pipeline run.

    Attributes:
        domains: List of mathematical domains to generate conjectures for.
        conjectures_per_domain: Target number of raw conjectures per domain.
        max_formalization_retries: Maximum LLM retries for Lean formalization.
        max_proof_retries: Maximum total proof-search retries per conjecture.
    """

    domains: list[str] = Field(
        default_factory=lambda: ["number_theory", "inequality"],
        description="Mathematical domains to target",
    )
    conjectures_per_domain: int = Field(
        default=100, ge=1, description="Raw conjectures to generate per domain"
    )
    max_formalization_retries: int = Field(
        default=5, ge=0, description="LLM formalization retry budget"
    )
    max_proof_retries: int = Field(
        default=3, ge=0, description="Proof-search retry budget per conjecture"
    )


class GenerationConfig(BaseModel):
    """
    Sampling parameters for the conjecture generation stage.

    Attributes:
        temperature: LLM sampling temperature (higher → more diverse).
        batch_size: Number of conjectures to request per LLM call.
    """

    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    batch_size: int = Field(default=20, ge=1, description="Conjectures per generation batch")


class ProofSearchConfig(BaseModel):
    """
    Per-layer timeout budgets for the layered proof-search strategy.

    Attributes:
        layer0_timeout: Timeout (s) for automated tactic combinators (``decide``, ``norm_num``).
        layer1_timeout: Timeout (s) for combined / preset tactic sequences.
        layer2_timeout: Timeout (s) for symbolic proof search (``aesop``, ``omega``).
        layer3_timeout: Timeout (s) for LLM-guided proof generation per round.
        layer3_max_rounds: Maximum dialogue rounds with the LLM in layer 3.
    """

    layer0_timeout: int = Field(default=5, ge=1, description="Layer-0 auto tactic timeout")
    layer1_timeout: int = Field(default=30, ge=1, description="Layer-1 combo tactic timeout")
    layer2_timeout: int = Field(default=60, ge=1, description="Layer-2 search timeout")
    layer3_timeout: int = Field(default=120, ge=1, description="Layer-3 LLM proof timeout")
    layer3_max_rounds: int = Field(
        default=3, ge=1, description="Max LLM dialogue rounds in layer 3"
    )


class OutputConfig(BaseModel):
    """
    Output and logging settings.

    Attributes:
        save_dir: Directory where pipeline results (JSON, Lean files) are written.
        save_lean_proofs: If True, write verified ``.lean`` proof files to ``save_dir``.
        log_level: Python logging level string (``DEBUG``, ``INFO``, ``WARNING``, etc.).
    """

    save_dir: str = Field(default="./data/results", description="Output directory for results")
    save_lean_proofs: bool = Field(default=True, description="Persist verified Lean proof files")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Root logger verbosity"
    )


class ConjLeanConfig(BaseModel):
    """
    Root configuration object for the ConjLean pipeline.

    Loaded from a YAML file and enriched with environment-variable overrides.
    All nested configs are validated by Pydantic v2 at construction time so
    failures surface immediately on startup rather than mid-run.

    Attributes:
        provider: Active LLM provider name.
        models: Model IDs for each pipeline stage.
        api_keys: API credentials (merged with environment variables).
        vllm: vLLM server settings (used only when ``provider == "vllm"``).
        lean: Lean 4 / Mathlib environment settings.
        pipeline: High-level pipeline parameters.
        generation: Conjecture generation sampling parameters.
        proof_search: Per-layer proof-search budgets.
        output: Output and logging settings.
    """

    provider: Literal["anthropic", "openai", "gemini", "huggingface", "vllm", "local_hf"] = Field(
        default="anthropic", description="Active LLM provider"
    )
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    api_keys: ProviderConfig = Field(default_factory=ProviderConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    local_hf: LocalHFConfig = Field(default_factory=LocalHFConfig)
    lean: LeanConfig = Field(default_factory=LeanConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    proof_search: ProofSearchConfig = Field(default_factory=ProofSearchConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ConjLeanConfig":
        """
        Load and validate a ConjLeanConfig from a YAML file.

        Environment variables are merged into ``api_keys`` automatically by
        the nested ``ProviderConfig`` validator after YAML loading.

        Args:
            path: Absolute or relative path to the YAML configuration file.

        Returns:
            A fully validated ``ConjLeanConfig`` instance.

        Raises:
            FileNotFoundError: If ``path`` does not point to an existing file.
            yaml.YAMLError: If the file is not valid YAML.
            pydantic.ValidationError: If any field fails validation.
        """
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")

        with config_path.open("r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        logger.info("Loaded configuration from %s", config_path.resolve())
        return cls(**raw)

    def get_active_provider(self) -> str:
        """
        Return the name of the currently active LLM provider.

        Returns:
            The provider string (one of ``anthropic``, ``openai``, ``gemini``,
            ``huggingface``, ``vllm``).
        """
        return self.provider
