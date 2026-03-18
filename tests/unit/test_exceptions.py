"""Unit tests for custom exceptions."""

import pytest
from llm_chain.exceptions import (
    LLMChainError,
    ConfigurationError,
    ProviderError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class TestLLMChainError:
    """Tests for the base LLMChainError."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        error = LLMChainError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_inheritance(self):
        """Test that LLMChainError inherits from Exception."""
        assert issubclass(LLMChainError, Exception)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error(self):
        """Test configuration error creation."""
        error = ConfigurationError("Missing API key")
        assert str(error) == "Missing API key"
        assert isinstance(error, LLMChainError)


class TestProviderError:
    """Tests for ProviderError."""

    def test_basic_provider_error(self):
        """Test basic provider error."""
        error = ProviderError("API call failed")
        assert "API call failed" in str(error)
        assert error.provider is None
        assert error.original_error is None

    def test_provider_error_with_provider(self):
        """Test provider error with provider name."""
        error = ProviderError("API call failed", provider="openai")
        assert "[openai]" in str(error)
        assert "API call failed" in str(error)
        assert error.provider == "openai"

    def test_provider_error_with_original(self):
        """Test provider error with original exception."""
        original = ValueError("Invalid input")
        error = ProviderError("Wrapped error", original_error=original)
        assert error.original_error is original

    def test_provider_error_full(self):
        """Test provider error with all attributes."""
        original = RuntimeError("Network failed")
        error = ProviderError(
            "Request failed",
            provider="claude",
            original_error=original,
        )
        assert "[claude]" in str(error)
        assert "Request failed" in str(error)
        assert error.provider == "claude"
        assert error.original_error is original


class TestStreamingError:
    """Tests for StreamingError."""

    def test_streaming_error(self):
        """Test streaming error creation."""
        error = StreamingError("Stream interrupted", provider="gemini")
        assert isinstance(error, ProviderError)
        assert "[gemini]" in str(error)
        assert "Stream interrupted" in str(error)


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error_basic(self):
        """Test basic rate limit error."""
        error = RateLimitError("Too many requests", provider="openai")
        assert isinstance(error, ProviderError)
        assert error.retry_after is None

    def test_rate_limit_error_with_retry(self):
        """Test rate limit error with retry-after."""
        error = RateLimitError(
            "Rate limited",
            provider="openai",
            retry_after=30.0,
        )
        assert error.retry_after == 30.0
        assert error.provider == "openai"


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_auth_error(self):
        """Test authentication error creation."""
        error = AuthenticationError("Invalid API key", provider="claude")
        assert isinstance(error, ProviderError)
        assert "[claude]" in str(error)


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_model_not_found(self):
        """Test model not found error creation."""
        error = ModelNotFoundError("Model 'gpt-5' not found", provider="openai")
        assert isinstance(error, ProviderError)
        assert "gpt-5" in str(error)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_inherit_from_base(self):
        """Test that all exceptions inherit from LLMChainError."""
        exceptions = [
            ConfigurationError,
            ProviderError,
            StreamingError,
            RateLimitError,
            AuthenticationError,
            ModelNotFoundError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, LLMChainError)

    def test_provider_errors_inherit(self):
        """Test that provider-related errors inherit from ProviderError."""
        provider_exceptions = [
            StreamingError,
            RateLimitError,
            AuthenticationError,
            ModelNotFoundError,
        ]

        for exc_class in provider_exceptions:
            assert issubclass(exc_class, ProviderError)

    def test_can_catch_by_base(self):
        """Test that all errors can be caught by base class."""
        errors = [
            ConfigurationError("config"),
            ProviderError("provider"),
            StreamingError("stream"),
            RateLimitError("rate"),
            AuthenticationError("auth"),
            ModelNotFoundError("model"),
        ]

        for error in errors:
            with pytest.raises(LLMChainError):
                raise error
