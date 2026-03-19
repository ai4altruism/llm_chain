"""Provider factory for dynamic provider instantiation."""

from typing import TypeVar

from llm_chain.config import Settings, ProviderType
from llm_chain.exceptions import ConfigurationError
from llm_chain.providers.base import LLMProvider
from llm_chain.providers.openai import OpenAIProvider
from llm_chain.providers.claude import ClaudeProvider
from llm_chain.providers.gemini import GeminiProvider


T = TypeVar("T", bound=LLMProvider)


class ProviderFactory:
    """Factory for creating LLM provider instances.

    This factory uses a registry pattern to map provider types to their
    implementation classes. It supports creating providers by name string
    and handles configuration loading from Settings.

    Example:
        ```python
        from llm_chain.providers.factory import ProviderFactory
        from llm_chain.config import Settings

        settings = Settings()
        factory = ProviderFactory(settings)

        # Create provider by type enum
        provider = factory.create(ProviderType.OPENAI, stage="initial")

        # Create provider by string name
        provider = factory.create_by_name("claude", stage="review")

        # Create both providers for chaining
        initial, review = factory.create_chain_providers()
        ```

    Attributes:
        _settings: The application settings instance.
        _registry: Mapping of provider types to their implementation classes.
    """

    # Class-level registry of provider types to implementation classes
    _registry: dict[ProviderType, type[LLMProvider]] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.CLAUDE: ClaudeProvider,
        ProviderType.GEMINI: GeminiProvider,
    }

    def __init__(self, settings: Settings | None = None):
        """Initialize the factory with settings.

        Args:
            settings: Application settings. If None, creates new Settings instance.
        """
        self._settings = settings or Settings()
        self._provider_cache: dict[str, LLMProvider] = {}

    @property
    def settings(self) -> Settings:
        """Return the settings instance."""
        return self._settings

    @classmethod
    def register(cls, provider_type: ProviderType, provider_class: type[LLMProvider]) -> None:
        """Register a provider class for a given type.

        This allows extending the factory with custom provider implementations.

        Args:
            provider_type: The provider type enum value.
            provider_class: The provider implementation class.

        Example:
            ```python
            ProviderFactory.register(ProviderType.OPENAI, CustomOpenAIProvider)
            ```
        """
        cls._registry[provider_type] = provider_class

    @classmethod
    def get_registered_providers(cls) -> list[ProviderType]:
        """Return list of registered provider types.

        Returns:
            List of ProviderType values that have registered implementations.
        """
        return list(cls._registry.keys())

    def _get_provider_class(self, provider_type: ProviderType) -> type[LLMProvider]:
        """Get the provider class for a given type.

        Args:
            provider_type: The provider type to look up.

        Returns:
            The provider implementation class.

        Raises:
            ConfigurationError: If the provider type is not registered.
        """
        if provider_type not in self._registry:
            raise ConfigurationError(
                f"No provider registered for type: {provider_type.value}. "
                f"Available providers: {[p.value for p in self._registry.keys()]}"
            )
        return self._registry[provider_type]

    def create(
        self,
        provider_type: ProviderType,
        stage: str = "initial",
        *,
        use_cache: bool = False,
    ) -> LLMProvider:
        """Create a provider instance.

        Args:
            provider_type: The type of provider to create.
            stage: The stage ('initial' or 'review') to determine model selection.
            use_cache: If True, return cached instance if available.

        Returns:
            A configured provider instance.

        Raises:
            ConfigurationError: If the provider is not properly configured.
        """
        cache_key = f"{provider_type.value}:{stage}"

        if use_cache and cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        # Validate configuration
        self._settings.validate_provider_config(provider_type)

        # Get API key and model
        api_key = self._settings.get_api_key_for_provider(provider_type)
        model = self._settings.get_model_for_provider(provider_type, stage)

        # Get the provider class and instantiate
        provider_class = self._get_provider_class(provider_type)

        # All providers take api_key and model as first two args
        provider = provider_class(
            api_key=api_key.get_secret_value(),  # type: ignore[union-attr]
            model=model,
        )

        if use_cache:
            self._provider_cache[cache_key] = provider

        return provider

    def create_by_name(
        self,
        provider_name: str,
        stage: str = "initial",
        *,
        use_cache: bool = False,
    ) -> LLMProvider:
        """Create a provider instance by name string.

        Args:
            provider_name: The provider name ('openai', 'claude', 'gemini').
            stage: The stage ('initial' or 'review') to determine model selection.
            use_cache: If True, return cached instance if available.

        Returns:
            A configured provider instance.

        Raises:
            ConfigurationError: If the provider name is invalid or not configured.
        """
        try:
            provider_type = ProviderType(provider_name.lower())
        except ValueError:
            valid_names = [p.value for p in ProviderType]
            raise ConfigurationError(
                f"Invalid provider name: '{provider_name}'. "
                f"Valid providers: {valid_names}"
            ) from None

        return self.create(provider_type, stage, use_cache=use_cache)

    def create_chain_providers(
        self,
        *,
        use_cache: bool = False,
    ) -> tuple[LLMProvider, LLMProvider]:
        """Create both initial and review providers based on settings.

        This is a convenience method for creating the two providers needed
        for the chaining workflow.

        Args:
            use_cache: If True, return cached instances if available.

        Returns:
            Tuple of (initial_provider, review_provider).

        Raises:
            ConfigurationError: If either provider is not properly configured.
        """
        initial_provider = self.create(
            self._settings.initial_provider,
            stage="initial",
            use_cache=use_cache,
        )
        review_provider = self.create(
            self._settings.review_provider,
            stage="review",
            use_cache=use_cache,
        )
        return initial_provider, review_provider

    def clear_cache(self) -> None:
        """Clear all cached provider instances."""
        self._provider_cache.clear()

    def get_cache_info(self) -> dict[str, str]:
        """Return information about cached providers.

        Returns:
            Dict mapping cache keys to provider class names.
        """
        return {
            key: provider.__class__.__name__
            for key, provider in self._provider_cache.items()
        }
