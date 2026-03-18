"""Unit tests for the abstract LLMProvider base class."""

import pytest
from llm_chain.providers.base import LLMProvider, Message, GenerationConfig, Role


class TestMessage:
    """Tests for the Message dataclass."""

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello!")
        assert msg.role == Role.USER
        assert msg.content == "Hello!"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"

    def test_direct_construction(self):
        """Test direct Message construction."""
        msg = Message(role=Role.USER, content="Test")
        assert msg.role == Role.USER
        assert msg.content == "Test"


class TestGenerationConfig:
    """Tests for the GenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.stop_sequences == []

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["STOP", "END"],
        )
        assert config.max_tokens == 512
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.stop_sequences == ["STOP", "END"]


class TestRole:
    """Tests for the Role enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"

    def test_role_is_string(self):
        """Test that roles are string-compatible."""
        assert str(Role.SYSTEM) == "Role.SYSTEM"
        assert Role.SYSTEM == "system"  # str enum comparison


class TestMockProvider:
    """Tests using the mock provider to verify interface."""

    @pytest.mark.asyncio
    async def test_generate(self, mock_provider, sample_messages):
        """Test the generate method."""
        result = await mock_provider.generate(sample_messages)
        assert result == "Mock response"
        assert len(mock_provider.generate_calls) == 1
        assert mock_provider.generate_calls[0][0] == sample_messages

    @pytest.mark.asyncio
    async def test_generate_with_config(self, mock_provider, sample_messages, sample_config):
        """Test generate with custom config."""
        result = await mock_provider.generate(sample_messages, sample_config)
        assert result == "Mock response"
        assert mock_provider.generate_calls[0][1] == sample_config

    @pytest.mark.asyncio
    async def test_generate_stream(self, mock_provider, sample_messages):
        """Test the streaming generate method."""
        chunks = []
        async for chunk in mock_provider.generate_stream(sample_messages):
            chunks.append(chunk)

        assert chunks == ["Mock ", "streaming ", "response"]
        assert len(mock_provider.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_provider_properties(self, mock_provider):
        """Test provider name and model properties."""
        assert mock_provider.provider_name == "mock"
        assert mock_provider.model_name == "mock-model"

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_provider):
        """Test async context manager protocol."""
        async with mock_provider as provider:
            assert provider is mock_provider
            result = await provider.generate([Message.user("test")])
            assert result == "Mock response"

    @pytest.mark.asyncio
    async def test_mock_factory(self, mock_provider_factory):
        """Test the mock provider factory fixture."""
        provider = mock_provider_factory(
            name="custom",
            model="custom-model",
            response="Custom response",
            stream_chunks=["A", "B", "C"],
        )

        assert provider.provider_name == "custom"
        assert provider.model_name == "custom-model"

        result = await provider.generate([Message.user("test")])
        assert result == "Custom response"

        chunks = [c async for c in provider.generate_stream([Message.user("test")])]
        assert chunks == ["A", "B", "C"]
