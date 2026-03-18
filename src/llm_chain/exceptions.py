"""Custom exceptions for LLM Chain."""


class LLMChainError(Exception):
    """Base exception for all LLM Chain errors."""

    pass


class ConfigurationError(LLMChainError):
    """Raised when there is a configuration problem.

    Examples:
        - Missing API key
        - Invalid model name
        - Invalid provider selection
    """

    pass


class ProviderError(LLMChainError):
    """Raised when an LLM provider API call fails.

    Attributes:
        provider: The name of the provider that failed.
        original_error: The underlying exception from the provider SDK.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.provider:
            parts.insert(0, f"[{self.provider}]")
        return " ".join(parts)


class StreamingError(ProviderError):
    """Raised when streaming fails mid-response.

    This can happen due to:
        - Network interruption
        - API timeout
        - Rate limiting during stream
    """

    pass


class RateLimitError(ProviderError):
    """Raised when the API rate limit is exceeded.

    Attributes:
        retry_after: Suggested wait time in seconds before retrying.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        original_error: Exception | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, provider, original_error)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when authentication with the provider fails.

    This typically indicates an invalid or expired API key.
    """

    pass


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available.

    This can happen if:
        - The model name is misspelled
        - The model is not available in your region
        - You don't have access to the model
    """

    pass
