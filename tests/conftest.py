"""Shared pytest fixtures for LLM Chain tests."""

import pytest
from typing import AsyncIterator

from llm_chain.config import Settings, ProviderType
from llm_chain.providers.base import LLMProvider, Message, GenerationConfig, Role


class MockProvider(LLMProvider):
    """A mock LLM provider for testing.

    This provider returns predictable responses without making real API calls.
    """

    def __init__(
        self,
        name: str = "mock",
        model: str = "mock-model",
        response: str = "Mock response",
        stream_chunks: list[str] | None = None,
    ):
        self._name = name
        self._model = model
        self._response = response
        self._stream_chunks = stream_chunks or ["Mock ", "streaming ", "response"]
        self.generate_calls: list[tuple[list[Message], GenerationConfig | None]] = []
        self.stream_calls: list[tuple[list[Message], GenerationConfig | None]] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        self.generate_calls.append((messages, config))
        return self._response

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls.append((messages, config))
        for chunk in self._stream_chunks:
            yield chunk


@pytest.fixture
def mock_provider() -> MockProvider:
    """Create a mock LLM provider."""
    return MockProvider()


@pytest.fixture
def mock_provider_factory():
    """Factory fixture to create mock providers with custom settings."""

    def _factory(
        name: str = "mock",
        model: str = "mock-model",
        response: str = "Mock response",
        stream_chunks: list[str] | None = None,
    ) -> MockProvider:
        return MockProvider(
            name=name,
            model=model,
            response=response,
            stream_chunks=stream_chunks,
        )

    return _factory


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create a sample list of messages for testing."""
    return [
        Message.system("You are a helpful assistant."),
        Message.user("Hello, how are you?"),
    ]


@pytest.fixture
def sample_config() -> GenerationConfig:
    """Create a sample generation config for testing."""
    return GenerationConfig(
        max_tokens=512,
        temperature=0.5,
        top_p=0.9,
    )


@pytest.fixture
def test_settings(monkeypatch) -> Settings:
    """Create test settings with mock API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("INITIAL_PROVIDER", "openai")
    monkeypatch.setenv("REVIEW_PROVIDER", "claude")
    return Settings()


@pytest.fixture
def minimal_settings(monkeypatch) -> Settings:
    """Create minimal settings with only OpenAI configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    # Clear other keys
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return Settings()
