"""Unit tests for the OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai import AsyncOpenAI, APIError
from openai import AuthenticationError as OpenAIAuthError
from openai import RateLimitError as OpenAIRateLimitError
from openai import NotFoundError as OpenAINotFoundError

from llm_chain.providers.openai import OpenAIProvider
from llm_chain.providers.base import Message, GenerationConfig, Role
from llm_chain.exceptions import (
    ProviderError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class TestOpenAIProviderInit:
    """Tests for OpenAIProvider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.provider_name == "openai"
        assert provider.model_name == "gpt-4o"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4-turbo")
        assert provider.model_name == "gpt-4-turbo"

    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
        )
        # OpenAI client normalizes URL with trailing slash
        assert str(provider._client.base_url).rstrip("/") == "https://custom.api.com/v1"

    def test_init_with_organization(self):
        """Test initialization with organization."""
        provider = OpenAIProvider(
            api_key="test-key",
            organization="org-123",
        )
        assert provider._client.organization == "org-123"


class TestOpenAIProviderMessageConversion:
    """Tests for message format conversion."""

    def test_convert_single_message(self):
        """Test converting a single message."""
        provider = OpenAIProvider(api_key="test-key")
        messages = [Message.user("Hello")]

        result = provider._convert_messages(messages)

        assert result == [{"role": "user", "content": "Hello"}]

    def test_convert_multiple_messages(self):
        """Test converting multiple messages."""
        provider = OpenAIProvider(api_key="test-key")
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        result = provider._convert_messages(messages)

        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_convert_empty_messages(self):
        """Test converting empty message list."""
        provider = OpenAIProvider(api_key="test-key")
        result = provider._convert_messages([])
        assert result == []


class TestOpenAIProviderGenerationParams:
    """Tests for generation parameter building."""

    def test_default_params(self):
        """Test default generation parameters."""
        provider = OpenAIProvider(api_key="test-key")
        params = provider._get_generation_params(None)

        assert params["max_tokens"] == 1024
        assert params["temperature"] == 0.7
        assert params["top_p"] == 1.0
        assert "stop" not in params

    def test_custom_params(self):
        """Test custom generation parameters."""
        provider = OpenAIProvider(api_key="test-key")
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
        assert params["stop"] == ["END", "STOP"]

    def test_empty_stop_sequences(self):
        """Test that empty stop sequences are not included."""
        provider = OpenAIProvider(api_key="test-key")
        config = GenerationConfig(stop_sequences=[])

        params = provider._get_generation_params(config)

        assert "stop" not in params


class TestOpenAIProviderGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        provider = OpenAIProvider(api_key="test-key")

        # Mock the response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! How can I help?"

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await provider.generate([Message.user("Hello")])

            assert result == "Hello! How can I help?"
            mock_create.assert_called_once()

            # Verify the call arguments
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_generate_with_config(self):
        """Test generation with custom config."""
        provider = OpenAIProvider(api_key="test-key")
        config = GenerationConfig(max_tokens=100, temperature=0.0)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            await provider.generate([Message.user("Test")], config)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_empty_content(self):
        """Test generation with None content returns empty string."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.generate([Message.user("Hello")])
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_auth_error(self):
        """Test authentication error handling."""
        provider = OpenAIProvider(api_key="invalid-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        error = OpenAIAuthError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(AuthenticationError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "openai"
            assert exc_info.value.original_error is error

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self):
        """Test rate limit error handling."""
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}

        error = OpenAIRateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "openai"
            assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_generate_model_not_found(self):
        """Test model not found error handling."""
        provider = OpenAIProvider(api_key="test-key", model="nonexistent-model")

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}

        error = OpenAINotFoundError(
            message="Model not found",
            response=mock_response,
            body={"error": {"message": "Model not found"}},
        )

        with patch.object(
            provider._client.chat.completions,
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
        provider = OpenAIProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}

        error = APIError(
            message="Internal server error",
            request=MagicMock(),
            body={"error": {"message": "Internal server error"}},
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "openai"


class TestOpenAIProviderGenerateStream:
    """Tests for the streaming generate method."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        """Test successful streaming generation."""
        provider = OpenAIProvider(api_key="test-key")

        # Create mock stream chunks
        chunks = []
        for text in ["Hello", " there", "!"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)

        # Add final chunk with no content
        final_chunk = MagicMock()
        final_chunk.choices = [MagicMock()]
        final_chunk.choices[0].delta.content = None
        chunks.append(final_chunk)

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ):
            result = []
            async for chunk in provider.generate_stream([Message.user("Hello")]):
                result.append(chunk)

            assert result == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_generate_stream_with_config(self):
        """Test streaming with custom config."""
        provider = OpenAIProvider(api_key="test-key")
        config = GenerationConfig(max_tokens=50)

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Test"
            yield chunk

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ) as mock_create:
            chunks = [c async for c in provider.generate_stream([Message.user("Test")], config)]

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["stream"] is True
            assert call_kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_generate_stream_empty_choices(self):
        """Test streaming with empty choices."""
        provider = OpenAIProvider(api_key="test-key")

        async def mock_stream():
            # Chunk with empty choices
            chunk1 = MagicMock()
            chunk1.choices = []
            yield chunk1

            # Normal chunk
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = "Hello"
            yield chunk2

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ):
            result = [c async for c in provider.generate_stream([Message.user("Test")])]
            assert result == ["Hello"]

    @pytest.mark.asyncio
    async def test_generate_stream_auth_error(self):
        """Test streaming authentication error."""
        provider = OpenAIProvider(api_key="invalid-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        error = OpenAIAuthError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(AuthenticationError):
                async for _ in provider.generate_stream([Message.user("Hello")]):
                    pass

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self):
        """Test streaming API error becomes StreamingError."""
        provider = OpenAIProvider(api_key="test-key")

        error = APIError(
            message="Stream failed",
            request=MagicMock(),
            body={"error": {"message": "Stream failed"}},
        )

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(StreamingError) as exc_info:
                async for _ in provider.generate_stream([Message.user("Hello")]):
                    pass

            assert exc_info.value.provider == "openai"


class TestOpenAIProviderContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(provider._client, "close", new_callable=AsyncMock) as mock_close:
            async with provider as p:
                assert p is provider

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_with_error(self):
        """Test context manager cleans up on error."""
        provider = OpenAIProvider(api_key="test-key")

        with patch.object(provider._client, "close", new_callable=AsyncMock) as mock_close:
            with pytest.raises(ValueError):
                async with provider:
                    raise ValueError("Test error")

            mock_close.assert_called_once()


class TestOpenAIProviderIntegration:
    """Integration-style tests for OpenAIProvider."""

    @pytest.mark.asyncio
    async def test_full_conversation(self):
        """Test a full multi-turn conversation."""
        provider = OpenAIProvider(api_key="test-key")

        messages = [
            Message.system("You are a helpful math tutor."),
            Message.user("What is 2+2?"),
            Message.assistant("2+2 equals 4."),
            Message.user("What about 3+3?"),
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "3+3 equals 6."

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            result = await provider.generate(messages)

            assert result == "3+3 equals 6."

            # Verify all messages were sent
            call_kwargs = mock_create.call_args.kwargs
            assert len(call_kwargs["messages"]) == 4
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][-1]["role"] == "user"
