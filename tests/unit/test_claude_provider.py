"""Unit tests for the Claude provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from anthropic import AsyncAnthropic, APIError
from anthropic import AuthenticationError as AnthropicAuthError
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic import NotFoundError as AnthropicNotFoundError

from llm_chain.providers.claude import ClaudeProvider
from llm_chain.providers.base import Message, GenerationConfig, Role
from llm_chain.exceptions import (
    ProviderError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class TestClaudeProviderInit:
    """Tests for ClaudeProvider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = ClaudeProvider(api_key="test-key")
        assert provider.provider_name == "claude"
        assert provider.model_name == "claude-sonnet-4-5-20250929"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        provider = ClaudeProvider(api_key="test-key", model="claude-3-opus")
        assert provider.model_name == "claude-3-opus"

    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        provider = ClaudeProvider(
            api_key="test-key",
            base_url="https://custom.api.com",
        )
        # Anthropic client normalizes the URL
        assert "custom.api.com" in str(provider._client.base_url)


class TestClaudeProviderMessageConversion:
    """Tests for message format conversion."""

    def test_extract_no_system_message(self):
        """Test extracting messages without system prompt."""
        provider = ClaudeProvider(api_key="test-key")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        system, converted = provider._extract_system_and_messages(messages)

        assert system is None
        assert converted == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_extract_with_system_message(self):
        """Test extracting messages with system prompt."""
        provider = ClaudeProvider(api_key="test-key")
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
        ]

        system, converted = provider._extract_system_and_messages(messages)

        assert system == "You are helpful."
        assert converted == [{"role": "user", "content": "Hello"}]

    def test_extract_multiple_system_messages_uses_last(self):
        """Test that multiple system messages use the last one."""
        provider = ClaudeProvider(api_key="test-key")
        messages = [
            Message.system("First system prompt"),
            Message.user("Hello"),
            Message.system("Second system prompt"),
            Message.user("More"),
        ]

        system, converted = provider._extract_system_and_messages(messages)

        assert system == "Second system prompt"
        assert len(converted) == 2

    def test_extract_empty_messages(self):
        """Test extracting from empty message list."""
        provider = ClaudeProvider(api_key="test-key")
        system, converted = provider._extract_system_and_messages([])

        assert system is None
        assert converted == []


class TestClaudeProviderGenerationParams:
    """Tests for generation parameter building."""

    def test_default_params(self):
        """Test default generation parameters."""
        provider = ClaudeProvider(api_key="test-key")
        params = provider._get_generation_params(None)

        assert params["max_tokens"] == 1024
        assert params["temperature"] == 0.7
        assert params["top_p"] == 1.0
        assert "stop_sequences" not in params

    def test_custom_params(self):
        """Test custom generation parameters."""
        provider = ClaudeProvider(api_key="test-key")
        config = GenerationConfig(
            max_tokens=512,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["END", "STOP"],
        )

        params = provider._get_generation_params(config)

        assert params["max_tokens"] == 512
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9
        assert params["stop_sequences"] == ["END", "STOP"]

    def test_empty_stop_sequences(self):
        """Test that empty stop sequences are not included."""
        provider = ClaudeProvider(api_key="test-key")
        config = GenerationConfig(stop_sequences=[])

        params = provider._get_generation_params(config)

        assert "stop_sequences" not in params


class TestClaudeProviderGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        provider = ClaudeProvider(api_key="test-key")

        # Mock the response with content blocks
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello! How can I help?"
        mock_response.content = [mock_text_block]

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await provider.generate([Message.user("Hello")])

            assert result == "Hello! How can I help?"
            mock_create.assert_called_once()

            # Verify the call arguments
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        provider = ClaudeProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"
        mock_response.content = [mock_text_block]

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            messages = [
                Message.system("You are a helpful assistant."),
                Message.user("Hello"),
            ]
            await provider.generate(messages)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["system"] == "You are a helpful assistant."
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_generate_multiple_content_blocks(self):
        """Test generation with multiple content blocks."""
        provider = ClaudeProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_text_block1 = MagicMock()
        mock_text_block1.type = "text"
        mock_text_block1.text = "Hello "
        mock_text_block2 = MagicMock()
        mock_text_block2.type = "text"
        mock_text_block2.text = "World!"
        mock_response.content = [mock_text_block1, mock_text_block2]

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.generate([Message.user("Hello")])
            assert result == "Hello World!"

    @pytest.mark.asyncio
    async def test_generate_with_config(self):
        """Test generation with custom config."""
        provider = ClaudeProvider(api_key="test-key")
        config = GenerationConfig(max_tokens=100, temperature=0.0)

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"
        mock_response.content = [mock_text_block]

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            await provider.generate([Message.user("Test")], config)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_auth_error(self):
        """Test authentication error handling."""
        provider = ClaudeProvider(api_key="invalid-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        error = AnthropicAuthError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(AuthenticationError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "claude"
            assert exc_info.value.original_error is error

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self):
        """Test rate limit error handling."""
        provider = ClaudeProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}

        error = AnthropicRateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "claude"
            assert exc_info.value.retry_after == 60.0

    @pytest.mark.asyncio
    async def test_generate_model_not_found(self):
        """Test model not found error handling."""
        provider = ClaudeProvider(api_key="test-key", model="nonexistent-model")

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}

        error = AnthropicNotFoundError(
            message="Model not found",
            response=mock_response,
            body={"error": {"message": "Model not found"}},
        )

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(ModelNotFoundError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert "nonexistent-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """Test generic API error handling."""
        provider = ClaudeProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}

        error = APIError(
            message="Internal server error",
            request=MagicMock(),
            body={"error": {"message": "Internal server error"}},
        )

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "claude"


class TestClaudeProviderGenerateStream:
    """Tests for the streaming generate method."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        """Test successful streaming generation."""
        provider = ClaudeProvider(api_key="test-key")

        # Create mock stream context manager
        mock_stream = AsyncMock()

        async def mock_text_stream():
            for text in ["Hello", " there", "!"]:
                yield text

        mock_stream.text_stream = mock_text_stream()

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_stream
        mock_context_manager.__aexit__.return_value = None

        with patch.object(
            provider._client.messages,
            "stream",
            return_value=mock_context_manager,
        ):
            result = []
            async for chunk in provider.generate_stream([Message.user("Hello")]):
                result.append(chunk)

            assert result == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_generate_stream_with_system_prompt(self):
        """Test streaming with system prompt."""
        provider = ClaudeProvider(api_key="test-key")

        mock_stream = AsyncMock()

        async def mock_text_stream():
            yield "Response"

        mock_stream.text_stream = mock_text_stream()

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_stream
        mock_context_manager.__aexit__.return_value = None

        with patch.object(
            provider._client.messages,
            "stream",
            return_value=mock_context_manager,
        ) as mock_stream_method:
            messages = [
                Message.system("Be helpful"),
                Message.user("Test"),
            ]
            chunks = [c async for c in provider.generate_stream(messages)]

            call_kwargs = mock_stream_method.call_args.kwargs
            assert call_kwargs["system"] == "Be helpful"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Test"}]

    @pytest.mark.asyncio
    async def test_generate_stream_with_config(self):
        """Test streaming with custom config."""
        provider = ClaudeProvider(api_key="test-key")
        config = GenerationConfig(max_tokens=50)

        mock_stream = AsyncMock()

        async def mock_text_stream():
            yield "Test"

        mock_stream.text_stream = mock_text_stream()

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_stream
        mock_context_manager.__aexit__.return_value = None

        with patch.object(
            provider._client.messages,
            "stream",
            return_value=mock_context_manager,
        ) as mock_stream_method:
            chunks = [c async for c in provider.generate_stream([Message.user("Test")], config)]

            call_kwargs = mock_stream_method.call_args.kwargs
            assert call_kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_generate_stream_auth_error(self):
        """Test streaming authentication error."""
        provider = ClaudeProvider(api_key="invalid-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        error = AnthropicAuthError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.side_effect = error

        with patch.object(
            provider._client.messages,
            "stream",
            return_value=mock_context_manager,
        ):
            with pytest.raises(AuthenticationError):
                async for _ in provider.generate_stream([Message.user("Hello")]):
                    pass

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self):
        """Test streaming API error becomes StreamingError."""
        provider = ClaudeProvider(api_key="test-key")

        error = APIError(
            message="Stream failed",
            request=MagicMock(),
            body={"error": {"message": "Stream failed"}},
        )

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.side_effect = error

        with patch.object(
            provider._client.messages,
            "stream",
            return_value=mock_context_manager,
        ):
            with pytest.raises(StreamingError) as exc_info:
                async for _ in provider.generate_stream([Message.user("Hello")]):
                    pass

            assert exc_info.value.provider == "claude"


class TestClaudeProviderContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        provider = ClaudeProvider(api_key="test-key")

        with patch.object(provider._client, "close", new_callable=AsyncMock) as mock_close:
            async with provider as p:
                assert p is provider

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_error(self):
        """Test context manager cleans up on error."""
        provider = ClaudeProvider(api_key="test-key")

        with patch.object(provider._client, "close", new_callable=AsyncMock) as mock_close:
            with pytest.raises(ValueError):
                async with provider:
                    raise ValueError("Test error")

            mock_close.assert_called_once()


class TestClaudeProviderIntegration:
    """Integration-style tests for ClaudeProvider."""

    @pytest.mark.asyncio
    async def test_full_conversation(self):
        """Test a full multi-turn conversation."""
        provider = ClaudeProvider(api_key="test-key")

        messages = [
            Message.system("You are a helpful math tutor."),
            Message.user("What is 2+2?"),
            Message.assistant("2+2 equals 4."),
            Message.user("What about 3+3?"),
        ]

        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "3+3 equals 6."
        mock_response.content = [mock_text_block]

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await provider.generate(messages)

            assert result == "3+3 equals 6."

            # Verify system was extracted and messages were converted
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["system"] == "You are a helpful math tutor."
            # System message should be excluded from messages list
            assert len(call_kwargs["messages"]) == 3
            assert call_kwargs["messages"][-1]["role"] == "user"
