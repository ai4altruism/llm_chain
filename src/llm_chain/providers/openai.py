"""OpenAI provider implementation using the latest SDK patterns."""

from typing import AsyncIterator

from openai import AsyncOpenAI, APIError, AuthenticationError as OpenAIAuthError
from openai import RateLimitError as OpenAIRateLimitError
from openai import NotFoundError as OpenAINotFoundError

from llm_chain.providers.base import LLMProvider, Message, GenerationConfig, Role
from llm_chain.exceptions import (
    ProviderError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class OpenAIProvider(LLMProvider):
    """OpenAI provider using AsyncOpenAI client.

    This provider implements the LLMProvider interface for OpenAI's Chat Completions API.
    It supports both standard and streaming responses using async/await patterns.

    Example:
        ```python
        from llm_chain.providers.openai import OpenAIProvider
        from llm_chain.providers.base import Message

        provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

        messages = [Message.user("Hello!")]
        response = await provider.generate(messages)
        print(response)

        # Streaming
        async for chunk in provider.generate_stream(messages):
            print(chunk, end="", flush=True)
        ```

    Attributes:
        _client: The AsyncOpenAI client instance.
        _model: The model name to use for completions.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        organization: str | None = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model name to use (default: gpt-4o).
            base_url: Optional custom base URL for API requests.
            organization: Optional organization ID.
        """
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal Message format to OpenAI format.

        Args:
            messages: List of Message objects.

        Returns:
            List of dicts in OpenAI message format.
        """
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

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
            params["stop"] = config.stop_sequences

        return params

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response from OpenAI.

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
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._convert_messages(messages),
                **self._get_generation_params(config),
            )
            return response.choices[0].message.content or ""

        except OpenAIAuthError as e:
            raise AuthenticationError(
                "Invalid OpenAI API key",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except OpenAIRateLimitError as e:
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
                "OpenAI rate limit exceeded",
                provider=self.provider_name,
                original_error=e,
                retry_after=retry_after,
            ) from e

        except OpenAINotFoundError as e:
            raise ModelNotFoundError(
                f"Model '{self._model}' not found",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e.message}",
                provider=self.provider_name,
                original_error=e,
            ) from e

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from OpenAI.

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
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=self._convert_messages(messages),
                stream=True,
                **self._get_generation_params(config),
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIAuthError as e:
            raise AuthenticationError(
                "Invalid OpenAI API key",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except OpenAIRateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after_str = e.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass

            raise RateLimitError(
                "OpenAI rate limit exceeded",
                provider=self.provider_name,
                original_error=e,
                retry_after=retry_after,
            ) from e

        except OpenAINotFoundError as e:
            raise ModelNotFoundError(
                f"Model '{self._model}' not found",
                provider=self.provider_name,
                original_error=e,
            ) from e

        except APIError as e:
            raise StreamingError(
                f"OpenAI streaming error: {e.message}",
                provider=self.provider_name,
                original_error=e,
            ) from e

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up the client on context manager exit."""
        await self._client.close()
