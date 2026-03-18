"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class Role(str, Enum):
    """Message roles for LLM conversations."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A message in an LLM conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
    """

    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 to 2.0).
        top_p: Nucleus sampling parameter.
        stop_sequences: List of sequences that stop generation.
    """

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers (OpenAI, Claude, Gemini) must implement this interface
    to ensure consistent behavior across different backends.

    Example:
        ```python
        class MyProvider(LLMProvider):
            async def generate(self, messages, config=None):
                # Implementation
                pass

            async def generate_stream(self, messages, config=None):
                # Implementation
                yield "chunk"
        ```
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider (e.g., 'openai', 'claude', 'gemini')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of messages forming the conversation.
            config: Optional generation configuration.

        Returns:
            The generated text response.

        Raises:
            ProviderError: If the API call fails.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            messages: List of messages forming the conversation.
            config: Optional generation configuration.

        Yields:
            Text chunks as they are generated.

        Raises:
            ProviderError: If the API call fails.
            StreamingError: If streaming fails mid-response.
        """
        ...
        # Required for async generator type hint
        yield ""  # pragma: no cover

    async def __aenter__(self) -> "LLMProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - override to clean up resources."""
        pass
