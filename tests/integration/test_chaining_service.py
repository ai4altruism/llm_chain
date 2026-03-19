"""Integration tests for the ChainingService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_chain.chaining import (
    ChainingService,
    ChainConfig,
    ChainResult,
    StreamChunk,
)
from llm_chain.providers.base import LLMProvider, Message, GenerationConfig, Role


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        response: str = "Mock response",
        stream_chunks: list[str] | None = None,
    ):
        self._name = name
        self._response = response
        self._stream_chunks = stream_chunks or ["chunk1", "chunk2", "chunk3"]
        self._generate_calls: list[list[Message]] = []
        self._stream_calls: list[list[Message]] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return f"{self._name}-model"

    async def generate(self, messages: list[Message], config=None) -> str:
        self._generate_calls.append(messages)
        return self._response

    async def generate_stream(self, messages: list[Message], config=None):
        self._stream_calls.append(messages)
        for chunk in self._stream_chunks:
            yield chunk


class TestChainingServiceInit:
    """Tests for ChainingService initialization."""

    def test_init_with_providers(self):
        """Test initialization with providers."""
        initial = MockProvider(name="initial")
        review = MockProvider(name="review")

        service = ChainingService(initial, review)

        assert service.initial_provider is initial
        assert service.review_provider is review
        assert service.config is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        initial = MockProvider()
        review = MockProvider()
        config = ChainConfig(
            review_system_prompt="Custom system prompt",
            review_instruction="Custom instruction: {initial_response}",
        )

        service = ChainingService(initial, review, config)

        assert service.config.review_system_prompt == "Custom system prompt"


class TestChainConfig:
    """Tests for ChainConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChainConfig()

        assert "critical reviewer" in config.review_system_prompt.lower()
        assert "{initial_response}" in config.review_instruction
        assert config.generation_config is None

    def test_custom_config(self):
        """Test custom configuration."""
        gen_config = GenerationConfig(max_tokens=500, temperature=0.5)
        config = ChainConfig(
            review_system_prompt="Be brief",
            review_instruction="Review: {initial_response}",
            generation_config=gen_config,
        )

        assert config.review_system_prompt == "Be brief"
        assert config.generation_config.max_tokens == 500


class TestChainingServiceProcess:
    """Tests for the process method."""

    @pytest.mark.asyncio
    async def test_process_basic(self):
        """Test basic process flow."""
        initial = MockProvider(name="initial", response="Initial response")
        review = MockProvider(name="review", response="Review response")

        service = ChainingService(initial, review)
        messages = [Message.user("Test prompt")]

        result = await service.process(messages)

        assert isinstance(result, ChainResult)
        assert result.initial_response == "Initial response"
        assert result.review_response == "Review response"
        assert result.initial_provider == "initial"
        assert result.review_provider == "review"

    @pytest.mark.asyncio
    async def test_process_calls_initial_with_messages(self):
        """Test that initial provider receives original messages."""
        initial = MockProvider()
        review = MockProvider()

        service = ChainingService(initial, review)
        messages = [
            Message.system("Be helpful"),
            Message.user("Hello"),
        ]

        await service.process(messages)

        assert len(initial._generate_calls) == 1
        assert initial._generate_calls[0] == messages

    @pytest.mark.asyncio
    async def test_process_calls_review_with_initial_response(self):
        """Test that review provider receives initial response."""
        initial = MockProvider(response="This is the initial answer")
        review = MockProvider()
        config = ChainConfig(review_instruction="Review this: {initial_response}")

        service = ChainingService(initial, review, config)

        await service.process([Message.user("Test")])

        assert len(review._generate_calls) == 1
        review_messages = review._generate_calls[0]

        # Should have system message and user message
        assert len(review_messages) == 2
        assert review_messages[0].role == Role.SYSTEM
        assert review_messages[1].role == Role.USER
        assert "This is the initial answer" in review_messages[1].content

    @pytest.mark.asyncio
    async def test_process_with_custom_generation_config(self):
        """Test process with custom generation config."""
        initial = MockProvider()
        review = MockProvider()
        gen_config = GenerationConfig(max_tokens=100)

        service = ChainingService(initial, review)

        # The mock doesn't use config, but this tests the interface
        result = await service.process([Message.user("Test")], config=gen_config)

        assert result is not None


class TestChainingServiceProcessStream:
    """Tests for the process_stream method."""

    @pytest.mark.asyncio
    async def test_process_stream_basic(self):
        """Test basic streaming flow."""
        initial = MockProvider(
            name="initial",
            stream_chunks=["Hello", " ", "World"],
        )
        review = MockProvider(
            name="review",
            stream_chunks=["Good", " response"],
        )

        service = ChainingService(initial, review)
        messages = [Message.user("Test")]

        chunks = []
        async for chunk in service.process_stream(messages):
            chunks.append(chunk)

        # Should have: 3 initial + 1 final marker + 2 review + 1 final marker
        assert len(chunks) == 7

    @pytest.mark.asyncio
    async def test_process_stream_stages(self):
        """Test that chunks are properly marked by stage."""
        initial = MockProvider(stream_chunks=["A", "B"])
        review = MockProvider(stream_chunks=["X", "Y"])

        service = ChainingService(initial, review)

        chunks = []
        async for chunk in service.process_stream([Message.user("Test")]):
            chunks.append(chunk)

        # Check initial stage chunks
        assert chunks[0].stage == "initial"
        assert chunks[0].content == "A"
        assert chunks[1].stage == "initial"
        assert chunks[1].content == "B"
        assert chunks[2].stage == "initial"
        assert chunks[2].is_final is True

        # Check review stage chunks
        assert chunks[3].stage == "review"
        assert chunks[3].content == "X"
        assert chunks[4].stage == "review"
        assert chunks[4].content == "Y"
        assert chunks[5].stage == "review"
        assert chunks[5].is_final is True

    @pytest.mark.asyncio
    async def test_process_stream_collects_initial_for_review(self):
        """Test that initial response is collected and passed to review."""
        initial = MockProvider(stream_chunks=["Hello", " ", "World"])
        review = MockProvider(stream_chunks=["OK"])
        config = ChainConfig(review_instruction="Review: {initial_response}")

        service = ChainingService(initial, review, config)

        # Consume all chunks
        async for _ in service.process_stream([Message.user("Test")]):
            pass

        # Check review was called with collected initial response
        assert len(review._stream_calls) == 1
        review_user_msg = review._stream_calls[0][1]
        assert "Hello World" in review_user_msg.content

    @pytest.mark.asyncio
    async def test_process_stream_chunk_type(self):
        """Test that StreamChunk has correct attributes."""
        initial = MockProvider(stream_chunks=["test"])
        review = MockProvider(stream_chunks=["review"])

        service = ChainingService(initial, review)

        async for chunk in service.process_stream([Message.user("Test")]):
            assert isinstance(chunk, StreamChunk)
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "stage")
            assert hasattr(chunk, "is_final")
            break


class TestChainingServiceInitialOnly:
    """Tests for initial-only processing methods."""

    @pytest.mark.asyncio
    async def test_process_initial_only(self):
        """Test processing with only initial provider."""
        initial = MockProvider(response="Just initial")
        review = MockProvider()

        service = ChainingService(initial, review)

        result = await service.process_initial_only([Message.user("Test")])

        assert result == "Just initial"
        # Review should not be called
        assert len(review._generate_calls) == 0

    @pytest.mark.asyncio
    async def test_process_initial_only_stream(self):
        """Test streaming with only initial provider."""
        initial = MockProvider(stream_chunks=["A", "B", "C"])
        review = MockProvider()

        service = ChainingService(initial, review)

        chunks = []
        async for chunk in service.process_initial_only_stream([Message.user("Test")]):
            chunks.append(chunk)

        assert chunks == ["A", "B", "C"]
        # Review should not be called
        assert len(review._stream_calls) == 0


class TestChainingServiceContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        initial = MockProvider()
        review = MockProvider()

        async with ChainingService(initial, review) as service:
            assert service is not None
            result = await service.process([Message.user("Test")])
            assert result is not None


class TestChainingServiceIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_full_chain_workflow(self):
        """Test complete chaining workflow."""
        initial = MockProvider(
            name="openai",
            response="Quantum computing uses qubits...",
        )
        review = MockProvider(
            name="claude",
            response="The explanation is good but could include more details...",
        )

        config = ChainConfig(
            review_system_prompt="You are an expert reviewer.",
            review_instruction=(
                "Analyze this response about quantum computing:\n\n"
                "{initial_response}\n\n"
                "Provide constructive feedback."
            ),
        )

        service = ChainingService(initial, review, config)

        messages = [
            Message.system("Explain complex topics simply."),
            Message.user("What is quantum computing?"),
        ]

        result = await service.process(messages)

        assert "qubits" in result.initial_response
        assert "good" in result.review_response
        assert result.initial_provider == "openai"
        assert result.review_provider == "claude"

    @pytest.mark.asyncio
    async def test_stream_full_workflow(self):
        """Test complete streaming workflow."""
        initial = MockProvider(
            name="openai",
            stream_chunks=["Quantum ", "computing ", "is..."],
        )
        review = MockProvider(
            name="claude",
            stream_chunks=["Good ", "explanation."],
        )

        service = ChainingService(initial, review)

        messages = [Message.user("Explain quantum computing")]

        initial_content = []
        review_content = []

        async for chunk in service.process_stream(messages):
            if chunk.stage == "initial" and not chunk.is_final:
                initial_content.append(chunk.content)
            elif chunk.stage == "review" and not chunk.is_final:
                review_content.append(chunk.content)

        assert "".join(initial_content) == "Quantum computing is..."
        assert "".join(review_content) == "Good explanation."

    @pytest.mark.asyncio
    async def test_mixed_providers(self):
        """Test chaining with different provider types."""
        # Simulate different providers
        initial = MockProvider(name="gemini", response="Gemini response")
        review = MockProvider(name="claude", response="Claude review")

        service = ChainingService(initial, review)

        result = await service.process([Message.user("Test")])

        assert result.initial_provider == "gemini"
        assert result.review_provider == "claude"
