"""Retry utilities for handling transient failures."""

import asyncio
import random
from functools import wraps
from typing import TypeVar, Callable, Awaitable, Any

from llm_chain.exceptions import RateLimitError, ProviderError


T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter to delays.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum retry attempts (default: 3).
            base_delay: Base delay in seconds (default: 1.0).
            max_delay: Maximum delay in seconds (default: 60.0).
            exponential_base: Exponential backoff base (default: 2.0).
            jitter: Add random jitter to delays (default: True).
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate delay for a given attempt.

        Args:
            attempt: Current attempt number (0-indexed).
            retry_after: Optional retry-after value from server.

        Returns:
            Delay in seconds before next retry.
        """
        # Use server-provided retry-after if available
        if retry_after is not None and retry_after > 0:
            delay = min(retry_after, self.max_delay)
        else:
            # Exponential backoff
            delay = self.base_delay * (self.exponential_base ** attempt)
            delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


def is_retryable(error: Exception) -> bool:
    """Determine if an error is retryable.

    Args:
        error: The exception to check.

    Returns:
        True if the error is retryable, False otherwise.
    """
    # Rate limit errors are always retryable
    if isinstance(error, RateLimitError):
        return True

    # Check for transient provider errors
    if isinstance(error, ProviderError):
        error_msg = str(error).lower()
        transient_indicators = [
            "timeout",
            "connection",
            "temporary",
            "unavailable",
            "overloaded",
            "502",
            "503",
            "504",
        ]
        return any(indicator in error_msg for indicator in transient_indicators)

    return False


def get_retry_after(error: Exception) -> float | None:
    """Extract retry-after value from an error.

    Args:
        error: The exception to check.

    Returns:
        Retry-after value in seconds, or None if not available.
    """
    if isinstance(error, RateLimitError):
        return error.retry_after
    return None


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """Execute an async function with retry logic.

    Args:
        func: Async function to execute.
        *args: Positional arguments for the function.
        config: Retry configuration (uses default if None).
        **kwargs: Keyword arguments for the function.

    Returns:
        Result of the function.

    Raises:
        The last exception if all retries are exhausted.
    """
    config = config or DEFAULT_RETRY_CONFIG
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            # Check if we should retry
            if attempt >= config.max_retries or not is_retryable(e):
                raise

            # Calculate delay
            retry_after = get_retry_after(e)
            delay = config.calculate_delay(attempt, retry_after)

            # Wait before retrying
            await asyncio.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise last_error  # type: ignore


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator to add retry logic to an async function.

    Args:
        config: Retry configuration (uses default if None).

    Returns:
        Decorator function.

    Example:
        ```python
        @with_retry(RetryConfig(max_retries=5))
        async def my_api_call():
            return await client.request()
        ```
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_async(func, *args, config=config, **kwargs)
        return wrapper
    return decorator
