"""Integration tests for the ProviderFactory."""

import pytest
from unittest.mock import patch

from llm_chain.config import Settings, ProviderType
from llm_chain.exceptions import ConfigurationError
from llm_chain.providers import (
    ProviderFactory,
    LLMProvider,
    OpenAIProvider,
    ClaudeProvider,
    GeminiProvider,
)


class TestProviderFactoryInit:
    """Tests for ProviderFactory initialization."""

    def test_init_with_settings(self):
        """Test factory initialization with explicit settings."""
        settings = Settings(
            openai_api_key="test-key",
            initial_provider=ProviderType.OPENAI,
        )
        factory = ProviderFactory(settings)
        assert factory.settings is settings

    def test_init_without_settings(self):
        """Test factory initialization creates default settings."""
        factory = ProviderFactory()
        assert factory.settings is not None
        assert isinstance(factory.settings, Settings)


class TestProviderFactoryRegistry:
    """Tests for the provider registry functionality."""

    def test_get_registered_providers(self):
        """Test listing registered providers."""
        providers = ProviderFactory.get_registered_providers()
        assert ProviderType.OPENAI in providers
        assert ProviderType.CLAUDE in providers
        assert ProviderType.GEMINI in providers

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        # Create a simple mock provider class
        class CustomProvider(LLMProvider):
            def __init__(self, api_key: str, model: str = "custom"):
                self._model = model

            @property
            def provider_name(self) -> str:
                return "custom"

            @property
            def model_name(self) -> str:
                return self._model

            async def generate(self, messages, config=None):
                return "custom response"

            async def generate_stream(self, messages, config=None):
                yield "custom"

        # Store original registry state
        original_registry = ProviderFactory._registry.copy()

        try:
            # Register custom provider for an existing type (for testing)
            ProviderFactory.register(ProviderType.OPENAI, CustomProvider)
            assert ProviderFactory._registry[ProviderType.OPENAI] is CustomProvider
        finally:
            # Restore original registry
            ProviderFactory._registry = original_registry


class TestProviderFactoryCreate:
    """Tests for provider creation."""

    def test_create_openai_provider(self):
        """Test creating an OpenAI provider."""
        settings = Settings(openai_api_key="test-openai-key")
        factory = ProviderFactory(settings)

        provider = factory.create(ProviderType.OPENAI, stage="initial")

        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "openai"
        assert provider.model_name == settings.openai_model_initial

    def test_create_claude_provider(self):
        """Test creating a Claude provider."""
        settings = Settings(anthropic_api_key="test-anthropic-key")
        factory = ProviderFactory(settings)

        provider = factory.create(ProviderType.CLAUDE, stage="initial")

        assert isinstance(provider, ClaudeProvider)
        assert provider.provider_name == "claude"
        assert provider.model_name == settings.anthropic_model_initial

    def test_create_gemini_provider(self):
        """Test creating a Gemini provider."""
        settings = Settings(google_api_key="test-google-key")
        factory = ProviderFactory(settings)

        provider = factory.create(ProviderType.GEMINI, stage="initial")

        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"
        assert provider.model_name == settings.gemini_model_initial

    def test_create_with_review_stage(self):
        """Test creating provider with review stage model."""
        settings = Settings(
            openai_api_key="test-key",
            openai_model_initial="gpt-4o",
            openai_model_review="gpt-4o-mini",
        )
        factory = ProviderFactory(settings)

        initial = factory.create(ProviderType.OPENAI, stage="initial")
        review = factory.create(ProviderType.OPENAI, stage="review")

        assert initial.model_name == "gpt-4o"
        assert review.model_name == "gpt-4o-mini"

    def test_create_without_api_key_raises_error(self):
        """Test that creating without API key raises ConfigurationError."""
        settings = Settings()  # No API keys set
        factory = ProviderFactory(settings)

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create(ProviderType.OPENAI, stage="initial")

        assert "API key not configured" in str(exc_info.value)
        assert "openai" in str(exc_info.value)


class TestProviderFactoryCreateByName:
    """Tests for provider creation by name string."""

    def test_create_by_name_openai(self):
        """Test creating provider by 'openai' name."""
        settings = Settings(openai_api_key="test-key")
        factory = ProviderFactory(settings)

        provider = factory.create_by_name("openai")

        assert isinstance(provider, OpenAIProvider)

    def test_create_by_name_claude(self):
        """Test creating provider by 'claude' name."""
        settings = Settings(anthropic_api_key="test-key")
        factory = ProviderFactory(settings)

        provider = factory.create_by_name("claude")

        assert isinstance(provider, ClaudeProvider)

    def test_create_by_name_gemini(self):
        """Test creating provider by 'gemini' name."""
        settings = Settings(google_api_key="test-key")
        factory = ProviderFactory(settings)

        provider = factory.create_by_name("gemini")

        assert isinstance(provider, GeminiProvider)

    def test_create_by_name_case_insensitive(self):
        """Test that provider names are case-insensitive."""
        settings = Settings(openai_api_key="test-key")
        factory = ProviderFactory(settings)

        provider1 = factory.create_by_name("OPENAI")
        provider2 = factory.create_by_name("OpenAI")
        provider3 = factory.create_by_name("openai")

        assert all(isinstance(p, OpenAIProvider) for p in [provider1, provider2, provider3])

    def test_create_by_name_invalid_raises_error(self):
        """Test that invalid provider name raises ConfigurationError."""
        settings = Settings()
        factory = ProviderFactory(settings)

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_by_name("invalid_provider")

        assert "Invalid provider name" in str(exc_info.value)
        assert "invalid_provider" in str(exc_info.value)


class TestProviderFactoryCreateChainProviders:
    """Tests for creating chain provider pairs."""

    def test_create_chain_providers_same_provider(self):
        """Test creating chain providers when both use same provider."""
        settings = Settings(
            openai_api_key="test-key",
            initial_provider=ProviderType.OPENAI,
            review_provider=ProviderType.OPENAI,
        )
        factory = ProviderFactory(settings)

        initial, review = factory.create_chain_providers()

        assert isinstance(initial, OpenAIProvider)
        assert isinstance(review, OpenAIProvider)

    def test_create_chain_providers_mixed_providers(self):
        """Test creating chain providers with different providers."""
        settings = Settings(
            openai_api_key="test-openai-key",
            anthropic_api_key="test-anthropic-key",
            initial_provider=ProviderType.OPENAI,
            review_provider=ProviderType.CLAUDE,
        )
        factory = ProviderFactory(settings)

        initial, review = factory.create_chain_providers()

        assert isinstance(initial, OpenAIProvider)
        assert isinstance(review, ClaudeProvider)

    def test_create_chain_providers_all_three(self):
        """Test that we can create providers for all three backends."""
        settings = Settings(
            openai_api_key="test-openai-key",
            anthropic_api_key="test-anthropic-key",
            google_api_key="test-google-key",
            initial_provider=ProviderType.GEMINI,
            review_provider=ProviderType.CLAUDE,
        )
        factory = ProviderFactory(settings)

        initial, review = factory.create_chain_providers()

        assert isinstance(initial, GeminiProvider)
        assert isinstance(review, ClaudeProvider)


class TestProviderFactoryCache:
    """Tests for provider caching functionality."""

    def test_cache_returns_same_instance(self):
        """Test that cached providers return same instance."""
        settings = Settings(openai_api_key="test-key")
        factory = ProviderFactory(settings)

        provider1 = factory.create(ProviderType.OPENAI, stage="initial", use_cache=True)
        provider2 = factory.create(ProviderType.OPENAI, stage="initial", use_cache=True)

        assert provider1 is provider2

    def test_cache_different_stages_different_instances(self):
        """Test that different stages create different cached instances."""
        settings = Settings(openai_api_key="test-key")
        factory = ProviderFactory(settings)

        initial = factory.create(ProviderType.OPENAI, stage="initial", use_cache=True)
        review = factory.create(ProviderType.OPENAI, stage="review", use_cache=True)

        assert initial is not review

    def test_no_cache_creates_new_instance(self):
        """Test that without cache flag, new instances are created."""
        settings = Settings(openai_api_key="test-key")
        factory = ProviderFactory(settings)

        provider1 = factory.create(ProviderType.OPENAI, stage="initial", use_cache=False)
        provider2 = factory.create(ProviderType.OPENAI, stage="initial", use_cache=False)

        assert provider1 is not provider2

    def test_clear_cache(self):
        """Test clearing the provider cache."""
        settings = Settings(openai_api_key="test-key")
        factory = ProviderFactory(settings)

        provider1 = factory.create(ProviderType.OPENAI, stage="initial", use_cache=True)
        factory.clear_cache()
        provider2 = factory.create(ProviderType.OPENAI, stage="initial", use_cache=True)

        assert provider1 is not provider2

    def test_get_cache_info(self):
        """Test getting cache information."""
        settings = Settings(openai_api_key="test-key")
        factory = ProviderFactory(settings)

        factory.create(ProviderType.OPENAI, stage="initial", use_cache=True)

        cache_info = factory.get_cache_info()

        assert "openai:initial" in cache_info
        assert cache_info["openai:initial"] == "OpenAIProvider"


class TestProviderFactoryIntegration:
    """Integration tests combining multiple factory features."""

    def test_full_workflow(self):
        """Test a complete workflow using the factory."""
        settings = Settings(
            openai_api_key="test-openai-key",
            anthropic_api_key="test-anthropic-key",
            google_api_key="test-google-key",
            openai_model_initial="gpt-4o",
            openai_model_review="gpt-4o-mini",
            initial_provider=ProviderType.OPENAI,
            review_provider=ProviderType.CLAUDE,
        )
        factory = ProviderFactory(settings)

        # Create chain providers
        initial, review = factory.create_chain_providers(use_cache=True)

        # Verify types
        assert isinstance(initial, OpenAIProvider)
        assert isinstance(review, ClaudeProvider)

        # Verify models
        assert initial.model_name == "gpt-4o"
        assert review.model_name == "claude-sonnet-4-5-20250929"

        # Verify cache
        cache_info = factory.get_cache_info()
        assert len(cache_info) == 2

        # Create another provider
        gemini = factory.create(ProviderType.GEMINI, stage="initial")
        assert isinstance(gemini, GeminiProvider)

    def test_settings_environment_loading(self):
        """Test that factory respects environment-loaded settings."""
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "env-openai-key",
            "INITIAL_PROVIDER": "openai",
        }):
            settings = Settings()
            factory = ProviderFactory(settings)

            provider = factory.create(ProviderType.OPENAI, stage="initial")

            assert isinstance(provider, OpenAIProvider)
            assert provider.provider_name == "openai"
