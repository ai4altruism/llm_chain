"""LLM Chain - A multi-provider LLM chaining assistant with async streaming support."""

from llm_chain.config import Settings
from llm_chain.exceptions import (
    LLMChainError,
    ProviderError,
    ConfigurationError,
    StreamingError,
)
from llm_chain.providers.base import LLMProvider, Message, Role, GenerationConfig
from llm_chain.providers.openai import OpenAIProvider
from llm_chain.providers.claude import ClaudeProvider

__version__ = "2.0.0"

__all__ = [
    # Config
    "Settings",
    # Exceptions
    "LLMChainError",
    "ProviderError",
    "ConfigurationError",
    "StreamingError",
    # Provider base
    "LLMProvider",
    "Message",
    "Role",
    "GenerationConfig",
    # Providers
    "OpenAIProvider",
    "ClaudeProvider",
]
