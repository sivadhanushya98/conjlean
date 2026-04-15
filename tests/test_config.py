"""
Tests for conjlean.config — YAML loading, defaults, env-var overrides,
and Pydantic validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from conjlean.config import ConjLeanConfig, ProviderConfig, VLLMConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_YAML = str(_REPO_ROOT / "configs" / "config.yaml")


# ---------------------------------------------------------------------------
# Loading from YAML
# ---------------------------------------------------------------------------


class TestFromYaml:
    """Validate ConjLeanConfig.from_yaml loading behaviour."""

    def test_loads_from_disk(self) -> None:
        """from_yaml loads the canonical config.yaml without raising."""
        cfg = ConjLeanConfig.from_yaml(_CONFIG_YAML)
        assert isinstance(cfg, ConjLeanConfig)

    def test_provider_is_anthropic(self) -> None:
        """The default provider in config.yaml is 'anthropic'."""
        cfg = ConjLeanConfig.from_yaml(_CONFIG_YAML)
        assert cfg.provider == "anthropic"

    def test_domains_include_number_theory(self) -> None:
        """config.yaml includes 'number_theory' in the domains list."""
        cfg = ConjLeanConfig.from_yaml(_CONFIG_YAML)
        assert "number_theory" in cfg.pipeline.domains

    def test_domains_include_inequality(self) -> None:
        """config.yaml includes 'inequality' in the domains list."""
        cfg = ConjLeanConfig.from_yaml(_CONFIG_YAML)
        assert "inequality" in cfg.pipeline.domains

    def test_missing_file_raises(self) -> None:
        """from_yaml raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            ConjLeanConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_model_ids_loaded(self) -> None:
        """Model IDs from config.yaml are loaded into the models sub-config."""
        cfg = ConjLeanConfig.from_yaml(_CONFIG_YAML)
        assert cfg.models.conjecture_gen != ""
        assert cfg.models.formalizer != ""
        assert cfg.models.proof_gen != ""


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    """Validate default field values on ConjLeanConfig."""

    def test_default_provider(self) -> None:
        """ConjLeanConfig() defaults provider to 'anthropic'."""
        cfg = ConjLeanConfig()
        assert cfg.provider == "anthropic"

    def test_default_domains(self) -> None:
        """Default pipeline.domains list contains number_theory and inequality."""
        cfg = ConjLeanConfig()
        assert "number_theory" in cfg.pipeline.domains
        assert "inequality" in cfg.pipeline.domains

    def test_default_conjectures_per_domain(self) -> None:
        """Default pipeline.conjectures_per_domain is 100."""
        cfg = ConjLeanConfig()
        assert cfg.pipeline.conjectures_per_domain == 100

    def test_default_max_formalization_retries(self) -> None:
        """Default pipeline.max_formalization_retries is 5."""
        cfg = ConjLeanConfig()
        assert cfg.pipeline.max_formalization_retries == 5

    def test_default_generation_temperature(self) -> None:
        """Default generation.temperature is 0.8."""
        cfg = ConjLeanConfig()
        assert cfg.generation.temperature == pytest.approx(0.8)

    def test_default_generation_batch_size(self) -> None:
        """Default generation.batch_size is 20."""
        cfg = ConjLeanConfig()
        assert cfg.generation.batch_size == 20

    def test_default_lean_project_dir(self) -> None:
        """Default lean.project_dir is './lean'."""
        cfg = ConjLeanConfig()
        assert cfg.lean.project_dir == "./lean"

    def test_default_output_log_level(self) -> None:
        """Default output.log_level is 'INFO'."""
        cfg = ConjLeanConfig()
        assert cfg.output.log_level == "INFO"

    def test_default_vllm_base_url(self) -> None:
        """Default vllm.base_url is 'http://localhost:8000/v1'."""
        cfg = ConjLeanConfig()
        assert cfg.vllm.base_url == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    """Validate that env-var values propagate into ProviderConfig."""

    def test_anthropic_key_from_env(self) -> None:
        """ANTHROPIC_API_KEY in environment is picked up by ProviderConfig."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-anthropic-123"}):
            cfg = ProviderConfig()
        assert cfg.anthropic == "sk-test-anthropic-123"

    def test_openai_key_from_env(self) -> None:
        """OPENAI_API_KEY in environment is picked up by ProviderConfig."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-openai-456"}):
            cfg = ProviderConfig()
        assert cfg.openai == "sk-test-openai-456"

    def test_gemini_key_from_env(self) -> None:
        """GEMINI_API_KEY in environment is picked up by ProviderConfig."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-abc"}):
            cfg = ProviderConfig()
        assert cfg.gemini == "gemini-abc"

    def test_hf_token_from_env(self) -> None:
        """HF_TOKEN in environment is picked up by ProviderConfig."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf-token-xyz"}):
            cfg = ProviderConfig()
        assert cfg.huggingface == "hf-token-xyz"

    def test_env_key_visible_in_conjlean_config(self) -> None:
        """An env-set API key appears in ConjLeanConfig.api_keys after construction."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-pipeline-999"}):
            cfg = ConjLeanConfig()
        assert cfg.api_keys.anthropic == "sk-test-pipeline-999"

    def test_yaml_value_takes_priority_over_empty_env(self) -> None:
        """Explicit YAML value is preserved when the env var is absent."""
        raw = {"api_keys": {"anthropic": "yaml-key-001"}}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            cfg = ConjLeanConfig(**raw)
        assert cfg.api_keys.anthropic == "yaml-key-001"


# ---------------------------------------------------------------------------
# get_active_provider
# ---------------------------------------------------------------------------


class TestGetActiveProvider:
    """Validate get_active_provider returns the configured provider string."""

    def test_returns_anthropic(self) -> None:
        """get_active_provider returns 'anthropic' for default config."""
        cfg = ConjLeanConfig(provider="anthropic")
        assert cfg.get_active_provider() == "anthropic"

    def test_returns_openai(self) -> None:
        """get_active_provider returns 'openai' when provider is openai."""
        cfg = ConjLeanConfig(provider="openai")
        assert cfg.get_active_provider() == "openai"

    def test_returns_gemini(self) -> None:
        """get_active_provider returns 'gemini' when provider is gemini."""
        cfg = ConjLeanConfig(provider="gemini")
        assert cfg.get_active_provider() == "gemini"

    def test_returns_vllm(self) -> None:
        """get_active_provider returns 'vllm' when provider is vllm."""
        cfg = ConjLeanConfig(provider="vllm")
        assert cfg.get_active_provider() == "vllm"

    def test_returns_huggingface(self) -> None:
        """get_active_provider returns 'huggingface' when provider is huggingface."""
        cfg = ConjLeanConfig(provider="huggingface")
        assert cfg.get_active_provider() == "huggingface"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    """Validate that illegal config values raise Pydantic ValidationError."""

    def test_invalid_provider_raises(self) -> None:
        """An unsupported provider value raises a ValidationError."""
        with pytest.raises(ValidationError):
            ConjLeanConfig(provider="invalid_provider")  # type: ignore[arg-type]

    def test_invalid_vllm_base_url_raises(self) -> None:
        """A non-HTTP vllm.base_url raises a ValidationError."""
        with pytest.raises(ValidationError):
            VLLMConfig(base_url="ftp://not-http")

    def test_negative_build_timeout_raises(self) -> None:
        """A non-positive build_timeout raises a ValidationError."""
        with pytest.raises(ValidationError):
            from conjlean.config import LeanConfig
            LeanConfig(build_timeout=0)

    def test_invalid_log_level_raises(self) -> None:
        """An invalid log_level string raises a ValidationError."""
        with pytest.raises(ValidationError):
            from conjlean.config import OutputConfig
            OutputConfig(log_level="VERBOSE")  # type: ignore[arg-type]

    def test_zero_conjectures_per_domain_raises(self) -> None:
        """conjectures_per_domain < 1 raises a ValidationError."""
        with pytest.raises(ValidationError):
            from conjlean.config import PipelineConfig
            PipelineConfig(conjectures_per_domain=0)
