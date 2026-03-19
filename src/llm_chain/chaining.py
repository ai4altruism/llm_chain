"""Async chaining service for multi-stage LLM processing."""

from dataclasses import dataclass, field
from typing import AsyncIterator

from llm_chain.providers.base import LLMProvider, Message, GenerationConfig


@dataclass
class ChainResult:
    """Result of a chaining operation.

    Attributes:
        initial_response: The response from the initial provider.
        review_response: The response from the review provider.
        initial_provider: Name of the provider used for initial response.
        review_provider: Name of the provider used for review.
    """

    initial_response: str
    review_response: str
    initial_provider: str
    review_provider: str


@dataclass
class StreamChunk:
    """A chunk from the streaming chaining process.

    Attributes:
        content: The text content of the chunk.
        stage: Which stage this chunk is from ('initial' or 'review').
        is_final: Whether this is the final chunk for this stage.
    """

    content: str
    stage: str  # 'initial' or 'review'
    is_final: bool = False


@dataclass
class ChainConfig:
    """Configuration for the chaining service.

    Attributes:
        review_system_prompt: System prompt for the review stage.
        review_instruction: Instruction template for the reviewer.
            Use {initial_response} as placeholder for the initial response.
        generation_config: Optional generation config for both stages.
    """

    review_system_prompt: str = (
        "You are a critical reviewer. Your task is to analyze and critique "
        "the following response, identifying strengths, weaknesses, and areas "
        "for improvement."
    )
    review_instruction: str = (
        "Please review and critique the following response:\n\n"
        "---\n{initial_response}\n---\n\n"
        "Provide a balanced critique covering:\n"
        "1. Strengths of the response\n"
        "2. Weaknesses or gaps\n"
        "3. Suggestions for improvement"
    )
    generation_config: GenerationConfig | None = None


class ChainingService:
    """Async service for chaining multiple LLM providers.

    This service implements a two-stage processing pipeline where:
    1. An initial provider generates a response to the user's prompt
    2. A review provider critiques and analyzes the initial response

    The service supports both standard (wait for complete response) and
    streaming (real-time chunks) modes.

    Example:
        ```python
        from llm_chain.chaining import ChainingService, ChainConfig
        from llm_chain.providers import ProviderFactory

        factory = ProviderFactory(settings)
        initial, review = factory.create_chain_providers()

        service = ChainingService(initial, review)

        # Standard processing
        result = await service.process([Message.user("Explain quantum computing")])
        print(result.initial_response)
        print(result.review_response)

        # Streaming processing
        async for chunk in service.process_stream(messages):
            print(f"[{chunk.stage}] {chunk.content}", end="")
        ```

    Attributes:
        initial_provider: Provider for generating initial responses.
        review_provider: Provider for generating reviews/critiques.
        config: Configuration for the chaining process.
    """

    def __init__(
        self,
        initial_provider: LLMProvider,
        review_provider: LLMProvider,
        config: ChainConfig | None = None,
    ):
        """Initialize the chaining service.

        Args:
            initial_provider: Provider for initial response generation.
            review_provider: Provider for review/critique generation.
            config: Optional chaining configuration.
        """
        self.initial_provider = initial_provider
        self.review_provider = review_provider
        self.config = config or ChainConfig()

    def _build_review_messages(self, initial_response: str) -> list[Message]:
        """Build the messages for the review stage.

        Args:
            initial_response: The response from the initial stage.

        Returns:
            List of messages for the review provider.
        """
        review_prompt = self.config.review_instruction.format(
            initial_response=initial_response
        )

        return [
            Message.system(self.config.review_system_prompt),
            Message.user(review_prompt),
        ]

    async def process(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> ChainResult:
        """Process messages through the chain.

        This method:
        1. Sends messages to the initial provider
        2. Takes the response and sends it to the review provider
        3. Returns both responses

        Args:
            messages: The conversation messages for the initial provider.
            config: Optional generation config (overrides service config).

        Returns:
            ChainResult containing both responses.

        Raises:
            ProviderError: If either provider fails.
        """
        gen_config = config or self.config.generation_config

        # Stage 1: Get initial response
        initial_response = await self.initial_provider.generate(
            messages, gen_config
        )

        # Stage 2: Get review of the initial response
        review_messages = self._build_review_messages(initial_response)
        review_response = await self.review_provider.generate(
            review_messages, gen_config
        )

        return ChainResult(
            initial_response=initial_response,
            review_response=review_response,
            initial_provider=self.initial_provider.provider_name,
            review_provider=self.review_provider.provider_name,
        )

    async def process_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Process messages through the chain with streaming output.

        This method streams the response in real-time:
        1. First yields chunks from the initial provider
        2. Then yields chunks from the review provider

        Each chunk includes metadata about which stage it's from.

        Args:
            messages: The conversation messages for the initial provider.
            config: Optional generation config (overrides service config).

        Yields:
            StreamChunk objects containing text and metadata.

        Raises:
            ProviderError: If either provider fails.
            StreamingError: If streaming fails mid-response.
        """
        gen_config = config or self.config.generation_config

        # Stage 1: Stream initial response
        initial_response_parts: list[str] = []

        async for chunk in self.initial_provider.generate_stream(
            messages, gen_config
        ):
            initial_response_parts.append(chunk)
            yield StreamChunk(content=chunk, stage="initial")

        # Mark end of initial stage
        yield StreamChunk(content="", stage="initial", is_final=True)

        # Combine initial response for review
        initial_response = "".join(initial_response_parts)

        # Stage 2: Stream review response
        review_messages = self._build_review_messages(initial_response)

        async for chunk in self.review_provider.generate_stream(
            review_messages, gen_config
        ):
            yield StreamChunk(content=chunk, stage="review")

        # Mark end of review stage
        yield StreamChunk(content="", stage="review", is_final=True)

    async def process_initial_only(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> str:
        """Process messages through only the initial provider.

        Useful when you only need the initial response without review.

        Args:
            messages: The conversation messages.
            config: Optional generation config.

        Returns:
            The initial response string.
        """
        gen_config = config or self.config.generation_config
        return await self.initial_provider.generate(messages, gen_config)

    async def process_initial_only_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream response from only the initial provider.

        Args:
            messages: The conversation messages.
            config: Optional generation config.

        Yields:
            Text chunks from the initial provider.
        """
        gen_config = config or self.config.generation_config
        async for chunk in self.initial_provider.generate_stream(
            messages, gen_config
        ):
            yield chunk

    async def __aenter__(self) -> "ChainingService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - clean up both providers."""
        await self.initial_provider.__aexit__(exc_type, exc_val, exc_tb)
        await self.review_provider.__aexit__(exc_type, exc_val, exc_tb)
