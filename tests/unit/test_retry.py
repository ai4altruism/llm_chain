"""Unit tests for retry utilities."""

import pytest
from unittest.mock import AsyncMock, patch

from llm_chain.retry import (
    RetryConfig,
    is_retryable,
    get_retry_after,
    retry_async,
    with_retry,
    DEFAULT_RETRY_CONFIG,
)
from llm_chain.exceptions import (
    RateLimitError,
    ProviderError,
    AuthenticationError,
    ModelNotFoundError,
)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.calculate_delay(0) == 1.0   # 1 * 2^0 = 1
        assert config.calculate_delay(1) == 2.0   # 1 * 2^1 = 2
        assert config.calculate_delay(2) == 4.0   # 1 * 2^2 = 4
        assert config.calculate_delay(3) == 8.0   # 1 * 2^3 = 8

    def test_calculate_delay_max_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=10.0, max_delay=30.0, jitter=False)

        assert config.calculate_delay(0) == 10.0
        assert config.calculate_delay(1) == 20.0
        assert config.calculate_delay(2) == 30.0  # Capped at max
        assert config.calculate_delay(10) == 30.0  # Still capped

    def test_calculate_delay_with_retry_after(self):
        """Test using server-provided retry-after."""
        config = RetryConfig(base_delay=1.0, jitter=False)

        # Server says wait 10 seconds
        assert config.calculate_delay(0, retry_after=10.0) == 10.0

        # Server value capped at max_delay
        config.max_delay = 5.0
        assert config.calculate_delay(0, retry_after=10.0) == 5.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(base_delay=1.0, jitter=True)

        # With jitter, delay should be between 0.5x and 1.5x base
        delays = [config.calculate_delay(0) for _ in range(100)]

        assert all(0.5 <= d <= 1.5 for d in delays)
        # Should have some variance
        assert len(set(delays)) > 1


class TestIsRetryable:
    """Tests for is_retryable function."""

    def test_rate_limit_error_is_retryable(self):
        """Test that RateLimitError is retryable."""
        error = RateLimitError("Rate limited", provider="test")
        assert is_retryable(error) is True

    def test_auth_error_not_retryable(self):
        """Test that AuthenticationError is not retryable."""
        error = AuthenticationError("Invalid key", provider="test")
        assert is_retryable(error) is False

    def test_model_not_found_not_retryable(self):
        """Test that ModelNotFoundError is not retryable."""
        error = ModelNotFoundError("Model not found", provider="test")
        assert is_retryable(error) is False

    def test_timeout_error_is_retryable(self):
        """Test that timeout errors are retryable."""
        error = ProviderError("Connection timeout", provider="test")
        assert is_retryable(error) is True

    def test_connection_error_is_retryable(self):
        """Test that connection errors are retryable."""
        error = ProviderError("Connection refused", provider="test")
        assert is_retryable(error) is True

    def test_503_error_is_retryable(self):
        """Test that 503 errors are retryable."""
        error = ProviderError("503 Service Unavailable", provider="test")
        assert is_retryable(error) is True

    def test_generic_error_not_retryable(self):
        """Test that generic errors are not retryable."""
        error = ProviderError("Unknown error", provider="test")
        assert is_retryable(error) is False


class TestGetRetryAfter:
    """Tests for get_retry_after function."""

    def test_rate_limit_with_retry_after(self):
        """Test extracting retry_after from RateLimitError."""
        error = RateLimitError("Rate limited", provider="test", retry_after=30.0)
        assert get_retry_after(error) == 30.0

    def test_rate_limit_without_retry_after(self):
        """Test RateLimitError without retry_after."""
        error = RateLimitError("Rate limited", provider="test")
        assert get_retry_after(error) is None

    def test_other_error(self):
        """Test other errors return None."""
        error = ProviderError("Error", provider="test")
        assert get_retry_after(error) is None


class TestRetryAsync:
    """Tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        """Test successful execution on first try."""
        func = AsyncMock(return_value="success")

        result = await retry_async(func, "arg1", kwarg1="value1")

        assert result == "success"
        func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Test success after a retryable failure."""
        func = AsyncMock(
            side_effect=[
                RateLimitError("Rate limited", provider="test"),
                "success",
            ]
        )
        config = RetryConfig(base_delay=0.01, jitter=False)

        result = await retry_async(func, config=config)

        assert result == "success"
        assert func.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries is respected."""
        func = AsyncMock(
            side_effect=RateLimitError("Rate limited", provider="test")
        )
        config = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)

        with pytest.raises(RateLimitError):
            await retry_async(func, config=config)

        assert func.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self):
        """Test that non-retryable errors are not retried."""
        func = AsyncMock(
            side_effect=AuthenticationError("Invalid key", provider="test")
        )

        with pytest.raises(AuthenticationError):
            await retry_async(func)

        func.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_default_config(self):
        """Test that default config is used when not specified."""
        func = AsyncMock(return_value="success")

        result = await retry_async(func)

        assert result == "success"


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""
        @with_retry()
        async def my_func():
            return "success"

        result = await my_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_with_retry(self):
        """Test decorator retries on failure."""
        call_count = 0

        @with_retry(RetryConfig(max_retries=3, base_delay=0.01, jitter=False))
        async def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited", provider="test")
            return "success"

        result = await my_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        @with_retry()
        async def my_documented_func():
            """This is my docstring."""
            return "success"

        assert my_documented_func.__name__ == "my_documented_func"
        assert my_documented_func.__doc__ == "This is my docstring."


class TestDefaultRetryConfig:
    """Tests for default retry configuration."""

    def test_default_config_exists(self):
        """Test that DEFAULT_RETRY_CONFIG is defined."""
        assert DEFAULT_RETRY_CONFIG is not None
        assert isinstance(DEFAULT_RETRY_CONFIG, RetryConfig)

    def test_default_config_values(self):
        """Test default config has sensible values."""
        assert DEFAULT_RETRY_CONFIG.max_retries == 3
        assert DEFAULT_RETRY_CONFIG.base_delay == 1.0
        assert DEFAULT_RETRY_CONFIG.max_delay == 60.0
