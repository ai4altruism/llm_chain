"""LLM Chain - A multi-provider LLM chaining assistant with async streaming support."""

from llm_chain.config import Settings, ProviderType
from llm_chain.exceptions import (
    LLMChainError,
    ProviderError,
    ConfigurationError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)
from llm_chain.providers.base import LLMProvider, Message, Role, GenerationConfig
from llm_chain.providers.openai import OpenAIProvider
from llm_chain.providers.claude import ClaudeProvider
from llm_chain.providers.gemini import GeminiProvider
from llm_chain.providers.factory import ProviderFactory
from llm_chain.chaining import ChainingService, ChainResult, StreamChunk, ChainConfig
from llm_chain.retry import (
    RetryConfig,
    retry_async,
    with_retry,
    is_retryable,
    get_retry_after,
    DEFAULT_RETRY_CONFIG,
)

__version__ = "2.0.0"

__all__ = [
    # Config
    "Settings",
    "ProviderType",
    # Exceptions
    "LLMChainError",
    "ProviderError",
    "ConfigurationError",
    "StreamingError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    # Provider base
    "LLMProvider",
    "Message",
    "Role",
    "GenerationConfig",
    # Providers
    "OpenAIProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "ProviderFactory",
    # Chaining
    "ChainingService",
    "ChainResult",
    "StreamChunk",
    "ChainConfig",
    # Retry
    "RetryConfig",
    "retry_async",
    "with_retry",
    "is_retryable",
    "get_retry_after",
    "DEFAULT_RETRY_CONFIG",
]
