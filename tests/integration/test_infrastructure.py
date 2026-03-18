"""Integration tests for infrastructure setup."""

import pytest
from llm_chain import (
    Settings,
    LLMProvider,
    Message,
    Role,
    LLMChainError,
    ProviderError,
    ConfigurationError,
)
from llm_chain.providers.base import GenerationConfig


class TestModuleImports:
    """Test that all modules can be imported correctly."""

    def test_import_main_module(self):
        """Test importing the main llm_chain module."""
        import llm_chain

        assert hasattr(llm_chain, "__version__")
        assert llm_chain.__version__ == "2.0.0"

    def test_import_settings(self):
        """Test importing Settings class."""
        from llm_chain.config import Settings, ProviderType

        assert Settings is not None
        assert ProviderType is not None

    def test_import_providers(self):
        """Test importing provider classes."""
        from llm_chain.providers import LLMProvider, Message, Role

        assert LLMProvider is not None
        assert Message is not None
        assert Role is not None

    def test_import_exceptions(self):
        """Test importing exception classes."""
        from llm_chain.exceptions import (
            LLMChainError,
            ProviderError,
            StreamingError,
            ConfigurationError,
            RateLimitError,
            AuthenticationError,
            ModelNotFoundError,
        )

        # All should be importable
        assert all(
            [
                LLMChainError,
                ProviderError,
                StreamingError,
                ConfigurationError,
                RateLimitError,
                AuthenticationError,
                ModelNotFoundError,
            ]
        )


class TestProviderInterface:
    """Integration tests for the provider interface."""

    @pytest.mark.asyncio
    async def test_mock_provider_full_workflow(self, mock_provider_factory):
        """Test a complete workflow with the mock provider."""
        # Create provider
        provider = mock_provider_factory(
            name="test-provider",
            model="test-model-v1",
            response="This is the complete response.",
            stream_chunks=["This ", "is ", "streaming."],
        )

        # Build conversation
        messages = [
            Message.system("You are a test assistant."),
            Message.user("What is 2+2?"),
        ]

        # Test non-streaming generation
        async with provider as p:
            result = await p.generate(messages)
            assert result == "This is the complete response."
            assert p.provider_name == "test-provider"
            assert p.model_name == "test-model-v1"

        # Test streaming generation
        chunks = []
        async for chunk in provider.generate_stream(messages):
            chunks.append(chunk)

        assert "".join(chunks) == "This is streaming."

    @pytest.mark.asyncio
    async def test_provider_with_config(self, mock_provider, sample_messages):
        """Test provider respects generation config."""
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.0,
            stop_sequences=["END"],
        )

        await mock_provider.generate(sample_messages, config)

        # Verify config was passed
        assert len(mock_provider.generate_calls) == 1
        _, passed_config = mock_provider.generate_calls[0]
        assert passed_config is config
        assert passed_config.max_tokens == 100


class TestSettingsIntegration:
    """Integration tests for settings management."""

    def test_settings_with_all_providers(self, test_settings):
        """Test settings with all providers configured."""
        from llm_chain.config import ProviderType

        # All providers should be configured
        for provider in ProviderType:
            key = test_settings.get_api_key_for_provider(provider)
            assert key is not None

            # Should not raise
            test_settings.validate_provider_config(provider)

    def test_settings_model_lookup(self, test_settings):
        """Test model lookup for different providers and stages."""
        from llm_chain.config import ProviderType

        # Test all combinations
        for provider in ProviderType:
            for stage in ["initial", "review"]:
                model = test_settings.get_model_for_provider(provider, stage)
                assert isinstance(model, str)
                assert len(model) > 0


class TestExceptionHandling:
    """Integration tests for exception handling patterns."""

    def test_exception_chain(self):
        """Test exception chaining works correctly."""
        original = ValueError("Original error")

        try:
            try:
                raise original
            except ValueError as e:
                raise ProviderError(
                    "Provider failed",
                    provider="test",
                    original_error=e,
                ) from e
        except ProviderError as pe:
            assert pe.original_error is original
            assert pe.__cause__ is original

    def test_catch_by_hierarchy(self):
        """Test catching exceptions by hierarchy."""
        errors_caught = []

        # Should be catchable as LLMChainError
        try:
            raise ConfigurationError("Missing key")
        except LLMChainError as e:
            errors_caught.append(type(e).__name__)

        try:
            raise ProviderError("API failed", provider="openai")
        except LLMChainError as e:
            errors_caught.append(type(e).__name__)

        assert errors_caught == ["ConfigurationError", "ProviderError"]
