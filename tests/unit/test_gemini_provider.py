"""Unit tests for the Gemini provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import types
from google.genai.errors import APIError, ClientError

from llm_chain.providers.gemini import GeminiProvider
from llm_chain.providers.base import Message, GenerationConfig, Role
from llm_chain.exceptions import (
    ProviderError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class TestGeminiProviderInit:
    """Tests for GeminiProvider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = GeminiProvider(api_key="test-key")
        assert provider.provider_name == "gemini"
        assert provider.model_name == "gemini-2.0-flash"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-pro")
        assert provider.model_name == "gemini-1.5-pro"


class TestGeminiProviderMessageConversion:
    """Tests for message format conversion."""

    def test_extract_no_system_message(self):
        """Test extracting messages without system prompt."""
        provider = GeminiProvider(api_key="test-key")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        system, contents = provider._extract_system_and_contents(messages)

        assert system is None
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[1].role == "model"  # assistant -> model

    def test_extract_with_system_message(self):
        """Test extracting messages with system prompt."""
        provider = GeminiProvider(api_key="test-key")
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
        ]

        system, contents = provider._extract_system_and_contents(messages)

        assert system == "You are helpful."
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_extract_multiple_system_messages_uses_last(self):
        """Test that multiple system messages use the last one."""
        provider = GeminiProvider(api_key="test-key")
        messages = [
            Message.system("First system prompt"),
            Message.user("Hello"),
            Message.system("Second system prompt"),
            Message.user("More"),
        ]

        system, contents = provider._extract_system_and_contents(messages)

        assert system == "Second system prompt"
        assert len(contents) == 2

    def test_extract_empty_messages(self):
        """Test extracting from empty message list."""
        provider = GeminiProvider(api_key="test-key")
        system, contents = provider._extract_system_and_contents([])

        assert system is None
        assert contents == []

    def test_role_mapping(self):
        """Test that roles are correctly mapped to Gemini format."""
        provider = GeminiProvider(api_key="test-key")
        messages = [
            Message.user("User message"),
            Message.assistant("Assistant message"),
            Message.user("Another user message"),
        ]

        _, contents = provider._extract_system_and_contents(messages)

        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"


class TestGeminiProviderGenerationConfig:
    """Tests for generation config building."""

    def test_default_config(self):
        """Test default generation config."""
        provider = GeminiProvider(api_key="test-key")
        config = provider._get_generation_config(None)

        assert config.max_output_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 1.0

    def test_custom_config(self):
        """Test custom generation config."""
        provider = GeminiProvider(api_key="test-key")
        gen_config = GenerationConfig(
            max_tokens=512,
            temperature=0.5,
            top_p=0.9,
            stop_sequences=["END", "STOP"],
        )

        config = provider._get_generation_config(gen_config)

        assert config.max_output_tokens == 512
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.stop_sequences == ["END", "STOP"]

    def test_empty_stop_sequences(self):
        """Test that empty stop sequences are handled."""
        provider = GeminiProvider(api_key="test-key")
        gen_config = GenerationConfig(stop_sequences=[])

        config = provider._get_generation_config(gen_config)

        # Empty list shouldn't be set
        assert not config.stop_sequences


class TestGeminiProviderGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful generation."""
        provider = GeminiProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.text = "Hello! How can I help?"

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_generate:
            result = await provider.generate([Message.user("Hello")])

            assert result == "Hello! How can I help?"
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        provider = GeminiProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.text = "Response"

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_generate:
            messages = [
                Message.system("You are a helpful assistant."),
                Message.user("Hello"),
            ]
            await provider.generate(messages)

            call_kwargs = mock_generate.call_args.kwargs
            assert call_kwargs["config"].system_instruction == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_empty_response(self):
        """Test generation with empty response."""
        provider = GeminiProvider(api_key="test-key")

        mock_response = MagicMock()
        mock_response.text = None

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.generate([Message.user("Hello")])
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_with_config(self):
        """Test generation with custom config."""
        provider = GeminiProvider(api_key="test-key")
        config = GenerationConfig(max_tokens=100, temperature=0.0)

        mock_response = MagicMock()
        mock_response.text = "Response"

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_generate:
            await provider.generate([Message.user("Test")], config)

            call_kwargs = mock_generate.call_args.kwargs
            assert call_kwargs["config"].max_output_tokens == 100
            assert call_kwargs["config"].temperature == 0.0

    @pytest.mark.asyncio
    async def test_generate_auth_error(self):
        """Test authentication error handling."""
        provider = GeminiProvider(api_key="invalid-key")

        error = ClientError(401, {"error": {"message": "API key not valid"}})

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(AuthenticationError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "gemini"
            assert exc_info.value.original_error is error

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self):
        """Test rate limit error handling."""
        provider = GeminiProvider(api_key="test-key")

        error = APIError(429, {"error": {"message": "Rate limit exceeded"}})

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_generate_quota_error(self):
        """Test quota exceeded error handling."""
        provider = GeminiProvider(api_key="test-key")

        error = APIError(429, {"error": {"message": "Quota exceeded for the day"}})

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_generate_model_not_found(self):
        """Test model not found error handling."""
        provider = GeminiProvider(api_key="test-key", model="nonexistent-model")

        error = APIError(404, {"error": {"message": "Model not found"}})

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(ModelNotFoundError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert "nonexistent-model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """Test generic API error handling."""
        provider = GeminiProvider(api_key="test-key")

        error = APIError(500, {"error": {"message": "Internal server error"}})

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(ProviderError) as exc_info:
                await provider.generate([Message.user("Hello")])

            assert exc_info.value.provider == "gemini"


class TestGeminiProviderGenerateStream:
    """Tests for the streaming generate method."""

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        """Test successful streaming generation."""
        provider = GeminiProvider(api_key="test-key")

        # Create mock chunks
        chunks = []
        for text in ["Hello", " there", "!"]:
            chunk = MagicMock()
            chunk.text = text
            chunks.append(chunk)

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ):
            result = []
            async for chunk in provider.generate_stream([Message.user("Hello")]):
                result.append(chunk)

            assert result == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_generate_stream_with_system_prompt(self):
        """Test streaming with system prompt."""
        provider = GeminiProvider(api_key="test-key")

        chunk = MagicMock()
        chunk.text = "Response"

        async def mock_stream():
            yield chunk

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ) as mock_stream_method:
            messages = [
                Message.system("Be helpful"),
                Message.user("Test"),
            ]
            chunks = [c async for c in provider.generate_stream(messages)]

            call_kwargs = mock_stream_method.call_args.kwargs
            assert call_kwargs["config"].system_instruction == "Be helpful"

    @pytest.mark.asyncio
    async def test_generate_stream_with_config(self):
        """Test streaming with custom config."""
        provider = GeminiProvider(api_key="test-key")
        config = GenerationConfig(max_tokens=50)

        chunk = MagicMock()
        chunk.text = "Test"

        async def mock_stream():
            yield chunk

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ) as mock_stream_method:
            chunks = [c async for c in provider.generate_stream([Message.user("Test")], config)]

            call_kwargs = mock_stream_method.call_args.kwargs
            assert call_kwargs["config"].max_output_tokens == 50

    @pytest.mark.asyncio
    async def test_generate_stream_empty_chunks(self):
        """Test streaming with empty chunks."""
        provider = GeminiProvider(api_key="test-key")

        # Create mock chunks with some empty
        chunks_data = [("Hello", True), ("", False), (" World", True)]
        chunks = []
        for text, has_text in chunks_data:
            chunk = MagicMock()
            chunk.text = text if has_text else None
            chunks.append(chunk)

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ):
            result = []
            async for chunk in provider.generate_stream([Message.user("Hello")]):
                result.append(chunk)

            assert result == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_generate_stream_auth_error(self):
        """Test streaming authentication error."""
        provider = GeminiProvider(api_key="invalid-key")

        error = ClientError(401, {"error": {"message": "Authentication failed"}})

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(AuthenticationError):
                async for _ in provider.generate_stream([Message.user("Hello")]):
                    pass

    @pytest.mark.asyncio
    async def test_generate_stream_api_error(self):
        """Test streaming API error becomes StreamingError."""
        provider = GeminiProvider(api_key="test-key")

        error = APIError(500, {"error": {"message": "Stream failed unexpectedly"}})

        with patch.object(
            provider._client.aio.models,
            "generate_content_stream",
            new_callable=AsyncMock,
            side_effect=error,
        ):
            with pytest.raises(StreamingError) as exc_info:
                async for _ in provider.generate_stream([Message.user("Hello")]):
                    pass

            assert exc_info.value.provider == "gemini"


class TestGeminiProviderContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        provider = GeminiProvider(api_key="test-key")

        async with provider as p:
            assert p is provider

    @pytest.mark.asyncio
    async def test_context_manager_with_error(self):
        """Test context manager cleans up on error."""
        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(ValueError):
            async with provider:
                raise ValueError("Test error")


class TestGeminiProviderIntegration:
    """Integration-style tests for GeminiProvider."""

    @pytest.mark.asyncio
    async def test_full_conversation(self):
        """Test a full multi-turn conversation."""
        provider = GeminiProvider(api_key="test-key")

        messages = [
            Message.system("You are a helpful math tutor."),
            Message.user("What is 2+2?"),
            Message.assistant("2+2 equals 4."),
            Message.user("What about 3+3?"),
        ]

        mock_response = MagicMock()
        mock_response.text = "3+3 equals 6."

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_generate:
            result = await provider.generate(messages)

            assert result == "3+3 equals 6."

            # Verify system was extracted
            call_kwargs = mock_generate.call_args.kwargs
            assert call_kwargs["config"].system_instruction == "You are a helpful math tutor."
            # System message should be excluded from contents
            assert len(call_kwargs["contents"]) == 3
            assert call_kwargs["contents"][-1].role == "user"

    @pytest.mark.asyncio
    async def test_conversation_role_mapping(self):
        """Test that conversation roles are properly mapped."""
        provider = GeminiProvider(api_key="test-key")

        messages = [
            Message.user("First user message"),
            Message.assistant("First assistant response"),
            Message.user("Second user message"),
        ]

        mock_response = MagicMock()
        mock_response.text = "Response"

        with patch.object(
            provider._client.aio.models,
            "generate_content",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_generate:
            await provider.generate(messages)

            call_kwargs = mock_generate.call_args.kwargs
            contents = call_kwargs["contents"]

            assert contents[0].role == "user"
            assert contents[1].role == "model"
            assert contents[2].role == "user"
