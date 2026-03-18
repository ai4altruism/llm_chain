"""Anthropic Claude provider implementation using the latest SDK patterns."""

from typing import AsyncIterator

from anthropic import AsyncAnthropic, APIError
from anthropic import AuthenticationError as AnthropicAuthError
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic import NotFoundError as AnthropicNotFoundError

from llm_chain.providers.base import LLMProvider, Message, GenerationConfig, Role
from llm_chain.exceptions import (
    ProviderError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider using AsyncAnthropic client.

    This provider implements the LLMProvider interface for Anthropic's Messages API.
    It supports both standard and streaming responses using async/await patterns.

    Note: Claude handles system prompts differently - they are passed as a separate
    parameter rather than as a message in the conversation. This provider
    automatically extracts system messages and handles them appropriately.

    Example:
        ```python
        from llm_chain.providers.claude import ClaudeProvider
        from llm_chain.providers.base import Message

        provider = ClaudeProvider(api_key="sk-ant-...", model="claude-sonnet-4-5-20250929")

        messages = [
            Message.system("You are helpful."),
            Message.user("Hello!")
        ]
        response = await provider.generate(messages)
        print(response)

        # Streaming
        async for chunk in provider.generate_stream(messages):
            print(chunk, end="", flush=True)
        ```

    Attributes:
        _client: The AsyncAnthropic client instance.
        _model: The model name to use for completions.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
    ):
        """Initialize the Claude provider.

        Args:
            api_key: Anthropic API key.
            model: Model name to use (default: claude-sonnet-4-5-20250929).
            base_url: Optional custom base URL for API requests.
        """
        self._model = model
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "claude"

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model

    def _extract_system_and_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        """Extract system prompt and convert messages to Claude format.

        Claude requires system prompts to be passed separately from the messages.
        This method extracts any system messages and converts the rest to
        Claude's message format.

        Args:
            messages: List of Message objects.

        Returns:
            Tuple of (system_prompt, converted_messages).
            system_prompt is None if no system message was provided.
        """
        system_prompt = None
        converted_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Claude only supports one system prompt - use the last one
                system_prompt = msg.content
            else:
                # Claude uses "user" and "assistant" roles
                converted_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return system_prompt, converted_messages

    def _get_generation_params(self, config: GenerationConfig | None) -> dict:
        """Build generation parameters from config.

        Args:
            config: Optional generation configuration.

        Returns:
            Dict of parameters for the API call.
        """
        if config is None:
            config = GenerationConfig()

        params = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.stop_sequences:
            params["stop_sequences"] = config.stop_sequences

        return params

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response from Claude.

        Args:
            messages: List of messages forming the conversation.
            config: Optional generation configuration.

        Returns:
            The generated text response.

        Raises:
            AuthenticationError: If the API key is invalid.
            RateLimitError: If rate limits are exceeded.
            ModelNotFoundError: If the model doesn't exist.
            ProviderError: For other API errors.
        """
        system_prompt, converted_messages = self._extract_system_and_messages(messages)

        try:
            kwargs = {
                "model": self._model,
                "messages": converted_messages,
                **self._get_generation_params(config),
            }

            if system_prompt is not None:
                kwargs["system"] = system_prompt

            response = await self._client.messages.create(**kwargs)

            # Extract text from content blocks
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)

            return "".join(text_parts)

        except AnthropicAuthError as e:
            raise AuthenticationError(
                "Invalid Anthropic API key",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except AnthropicRateLimitError as e:
            # Try to extract retry-after from headers if available
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass

            raise RateLimitError(
                "Anthropic rate limit exceeded",
                provider=self.provider_name,
                original_error=e,
                retry_after=retry_after,
            ) from e

        except AnthropicNotFoundError as e:
            raise ModelNotFoundError(
                f"Model '{self._model}' not found",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except APIError as e:
            raise ProviderError(
                f"Anthropic API error: {e.message}",
                provider=self.provider_name,
                original_error=e,
            ) from e

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from Claude.

        Uses Claude's messages.stream() context manager for efficient streaming.

        Args:
            messages: List of messages forming the conversation.
            config: Optional generation configuration.

        Yields:
            Text chunks as they are generated.

        Raises:
            AuthenticationError: If the API key is invalid.
            RateLimitError: If rate limits are exceeded.
            ModelNotFoundError: If the model doesn't exist.
            StreamingError: If streaming fails mid-response.
            ProviderError: For other API errors.
        """
        system_prompt, converted_messages = self._extract_system_and_messages(messages)

        try:
            kwargs = {
                "model": self._model,
                "messages": converted_messages,
                **self._get_generation_params(config),
            }

            if system_prompt is not None:
                kwargs["system"] = system_prompt

            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except AnthropicAuthError as e:
            raise AuthenticationError(
                "Invalid Anthropic API key",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except AnthropicRateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass

            raise RateLimitError(
                "Anthropic rate limit exceeded",
                provider=self.provider_name,
                original_error=e,
                retry_after=retry_after,
            ) from e

        except AnthropicNotFoundError as e:
            raise ModelNotFoundError(
                f"Model '{self._model}' not found",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except APIError as e:
            raise StreamingError(
                f"Anthropic streaming error: {e.message}",
                provider=self.provider_name,
                original_error=e,
            ) from e

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up the client on context manager exit."""
        await self._client.close()
