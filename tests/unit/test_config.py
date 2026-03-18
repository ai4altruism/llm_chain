"""Unit tests for the configuration module."""

import pytest
from pydantic import SecretStr

from llm_chain.config import Settings, ProviderType
from llm_chain.exceptions import ConfigurationError


class TestProviderType:
    """Tests for the ProviderType enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.CLAUDE.value == "claude"
        assert ProviderType.GEMINI.value == "gemini"

    def test_provider_from_string(self):
        """Test creating provider from string."""
        assert ProviderType("openai") == ProviderType.OPENAI
        assert ProviderType("claude") == ProviderType.CLAUDE
        assert ProviderType("gemini") == ProviderType.GEMINI


class TestSettings:
    """Tests for the Settings class."""

    def test_default_values(self, monkeypatch):
        """Test default settings values."""
        # Clear all env vars
        for key in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "INITIAL_PROVIDER",
            "REVIEW_PROVIDER",
        ]:
            monkeypatch.delenv(key, raising=False)

        settings = Settings()

        assert settings.initial_provider == ProviderType.OPENAI
        assert settings.review_provider == ProviderType.OPENAI
        assert settings.openai_api_key is None
        assert settings.openai_model_initial == "gpt-4o"
        assert settings.default_max_tokens == 1024
        assert settings.default_temperature == 0.7

    def test_load_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        monkeypatch.setenv("INITIAL_PROVIDER", "claude")
        monkeypatch.setenv("OPENAI_MODEL_INITIAL", "gpt-4-turbo")
        monkeypatch.setenv("DEFAULT_MAX_TOKENS", "2048")

        settings = Settings()

        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-test123"
        assert settings.initial_provider == ProviderType.CLAUDE
        assert settings.openai_model_initial == "gpt-4-turbo"
        assert settings.default_max_tokens == 2048

    def test_api_key_placeholder_rejected(self, monkeypatch):
        """Test that placeholder API keys are treated as None."""
        placeholders = [
            "your_api_key_here",
            "sk-...",
            "sk-ant-...",
            "<your-api-key>",
        ]

        for placeholder in placeholders:
            monkeypatch.setenv("OPENAI_API_KEY", placeholder)
            settings = Settings()
            assert settings.openai_api_key is None, f"Placeholder '{placeholder}' should be None"

    def test_empty_api_key_is_none(self, monkeypatch):
        """Test that empty API keys are treated as None."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        settings = Settings()
        assert settings.openai_api_key is None

        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        settings = Settings()
        assert settings.openai_api_key is None

    def test_get_model_for_provider(self, test_settings):
        """Test getting model names for providers."""
        assert test_settings.get_model_for_provider(ProviderType.OPENAI, "initial") == "gpt-4o"
        assert test_settings.get_model_for_provider(ProviderType.OPENAI, "review") == "gpt-4o"
        assert (
            test_settings.get_model_for_provider(ProviderType.CLAUDE, "initial")
            == "claude-sonnet-4-5-20250929"
        )
        assert (
            test_settings.get_model_for_provider(ProviderType.GEMINI, "initial")
            == "gemini-2.0-flash"
        )

    def test_get_model_invalid_stage(self, test_settings):
        """Test that invalid stage raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stage"):
            test_settings.get_model_for_provider(ProviderType.OPENAI, "invalid")

    def test_get_api_key_for_provider(self, test_settings):
        """Test getting API keys for providers."""
        openai_key = test_settings.get_api_key_for_provider(ProviderType.OPENAI)
        assert openai_key is not None
        assert openai_key.get_secret_value() == "test-openai-key"

        claude_key = test_settings.get_api_key_for_provider(ProviderType.CLAUDE)
        assert claude_key is not None
        assert claude_key.get_secret_value() == "test-anthropic-key"

        gemini_key = test_settings.get_api_key_for_provider(ProviderType.GEMINI)
        assert gemini_key is not None
        assert gemini_key.get_secret_value() == "test-google-key"

    def test_validate_provider_config_success(self, test_settings):
        """Test provider validation with valid config."""
        # Should not raise
        test_settings.validate_provider_config(ProviderType.OPENAI)
        test_settings.validate_provider_config(ProviderType.CLAUDE)
        test_settings.validate_provider_config(ProviderType.GEMINI)

    def test_validate_provider_config_missing_key(self, minimal_settings):
        """Test provider validation with missing API key."""
        # OpenAI should pass
        minimal_settings.validate_provider_config(ProviderType.OPENAI)

        # Claude should fail
        with pytest.raises(ConfigurationError, match="API key not configured"):
            minimal_settings.validate_provider_config(ProviderType.CLAUDE)

        # Gemini should fail
        with pytest.raises(ConfigurationError, match="API key not configured"):
            minimal_settings.validate_provider_config(ProviderType.GEMINI)


class TestSettingsCustomModels:
    """Tests for custom model configuration."""

    def test_custom_openai_models(self, monkeypatch):
        """Test custom OpenAI model names."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL_INITIAL", "gpt-4-turbo")
        monkeypatch.setenv("OPENAI_MODEL_REVIEW", "gpt-4o-mini")

        settings = Settings()

        assert settings.openai_model_initial == "gpt-4-turbo"
        assert settings.openai_model_review == "gpt-4o-mini"

    def test_custom_anthropic_models(self, monkeypatch):
        """Test custom Anthropic model names."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("ANTHROPIC_MODEL_INITIAL", "claude-3-opus")
        monkeypatch.setenv("ANTHROPIC_MODEL_REVIEW", "claude-3-haiku")

        settings = Settings()

        assert settings.anthropic_model_initial == "claude-3-opus"
        assert settings.anthropic_model_review == "claude-3-haiku"

    def test_custom_gemini_models(self, monkeypatch):
        """Test custom Gemini model names."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL_INITIAL", "gemini-1.5-pro")
        monkeypatch.setenv("GEMINI_MODEL_REVIEW", "gemini-1.5-flash")

        settings = Settings()

        assert settings.gemini_model_initial == "gemini-1.5-pro"
        assert settings.gemini_model_review == "gemini-1.5-flash"
