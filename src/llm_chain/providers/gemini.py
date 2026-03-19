"""Google Gemini provider implementation using the google-genai SDK."""

from typing import AsyncIterator

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from llm_chain.providers.base import LLMProvider, Message, GenerationConfig, Role
from llm_chain.exceptions import (
    ProviderError,
    StreamingError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
)


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the google-genai SDK.

    This provider implements the LLMProvider interface for Google's Gemini API.
    It supports both standard and streaming responses using async/await patterns.

    Note: Gemini uses different role names than OpenAI/Claude:
    - 'user' maps to 'user'
    - 'assistant' maps to 'model'
    - System messages are handled via system_instruction parameter

    Example:
        ```python
        from llm_chain.providers.gemini import GeminiProvider
        from llm_chain.providers.base import Message

        provider = GeminiProvider(api_key="...", model="gemini-2.0-flash")

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
        _client: The genai.Client instance.
        _model: The model name to use for completions.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
    ):
        """Initialize the Gemini provider.

        Args:
            api_key: Google API key.
            model: Model name to use (default: gemini-2.0-flash).
        """
        self._model = model
        self._api_key = api_key
        self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model

    def _extract_system_and_contents(
        self, messages: list[Message]
    ) -> tuple[str | None, list[types.Content]]:
        """Extract system instruction and convert messages to Gemini format.

        Gemini uses 'user' and 'model' roles instead of 'user' and 'assistant'.
        System messages are handled via the system_instruction parameter.

        Args:
            messages: List of Message objects.

        Returns:
            Tuple of (system_instruction, contents).
            system_instruction is None if no system message was provided.
        """
        system_instruction = None
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Gemini uses system_instruction parameter
                system_instruction = msg.content
            else:
                # Map assistant -> model for Gemini
                role = "model" if msg.role == Role.ASSISTANT else "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        return system_instruction, contents

    def _get_generation_config(
        self, config: GenerationConfig | None
    ) -> types.GenerateContentConfig:
        """Build generation config from our GenerationConfig.

        Args:
            config: Optional generation configuration.

        Returns:
            Gemini GenerateContentConfig object.
        """
        if config is None:
            config = GenerationConfig()

        gen_config = types.GenerateContentConfig(
            max_output_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        if config.stop_sequences:
            gen_config.stop_sequences = config.stop_sequences

        return gen_config

    def _handle_error(self, error: Exception, is_streaming: bool = False) -> None:
        """Handle Gemini API errors and convert to our exception types.

        Args:
            error: The exception from the Gemini API.
            is_streaming: Whether this occurred during streaming.

        Raises:
            AuthenticationError: For authentication failures.
            RateLimitError: For rate limit errors.
            ModelNotFoundError: For model not found errors.
            StreamingError: For streaming-specific errors.
            ProviderError: For other API errors.
        """
        error_message = str(error).lower()

        # Check for authentication errors
        if "api key" in error_message or "authentication" in error_message or "401" in error_message:
            raise AuthenticationError(
                "Invalid Google API key",
                provider=self.provider_name,
                original_error=error,
            ) from error

        # Check for rate limit errors
        if "rate limit" in error_message or "quota" in error_message or "429" in error_message:
            # Try to extract retry-after if available
            retry_after = None
            raise RateLimitError(
                "Google API rate limit exceeded",
                provider=self.provider_name,
                original_error=error,
                retry_after=retry_after,
            ) from error

        # Check for model not found errors
        if "not found" in error_message or "404" in error_message or "model" in error_message and "invalid" in error_message:
            raise ModelNotFoundError(
                f"Model '{self._model}' not found",
                provider=self.provider_name,
                original_error=error,
            ) from error

        # Generic error handling
        if is_streaming:
            raise StreamingError(
                f"Gemini streaming error: {error}",
                provider=self.provider_name,
                original_error=error,
            ) from error
        else:
            raise ProviderError(
                f"Gemini API error: {error}",
                provider=self.provider_name,
                original_error=error,
            ) from error

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response from Gemini.

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
        system_instruction, contents = self._extract_system_and_contents(messages)
        gen_config = self._get_generation_config(config)

        # Add system instruction if present
        if system_instruction:
            gen_config.system_instruction = system_instruction

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=gen_config,
            )

            # Extract text from the response
            if response.text:
                return response.text
            return ""

        except (APIError, ClientError) as e:
            self._handle_error(e, is_streaming=False)
        except Exception as e:
            self._handle_error(e, is_streaming=False)

        return ""  # Unreachable, but keeps type checker happy

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from Gemini.

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
        system_instruction, contents = self._extract_system_and_contents(messages)
        gen_config = self._get_generation_config(config)

        # Add system instruction if present
        if system_instruction:
            gen_config.system_instruction = system_instruction

        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=gen_config,
            ):
                if chunk.text:
                    yield chunk.text

        except (APIError, ClientError) as e:
            self._handle_error(e, is_streaming=True)
        except Exception as e:
            self._handle_error(e, is_streaming=True)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources on context manager exit.

        Note: The google-genai client doesn't require explicit cleanup,
        but we implement this for interface consistency.
        """
        pass
