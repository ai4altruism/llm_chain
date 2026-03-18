"""Configuration management using Pydantic Settings."""

from enum import Enum
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Environment variables can be set directly or loaded from a .env file.
    All API keys are handled securely using SecretStr.

    Example:
        ```python
        settings = Settings()
        print(settings.initial_provider)  # ProviderType.OPENAI

        # Access API keys securely
        api_key = settings.openai_api_key.get_secret_value()
        ```
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Provider Selection
    initial_provider: ProviderType = Field(
        default=ProviderType.OPENAI,
        description="Provider for initial response generation",
    )
    review_provider: ProviderType = Field(
        default=ProviderType.OPENAI,
        description="Provider for review/critique generation",
    )

    # OpenAI Configuration
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key",
    )
    openai_model_initial: str = Field(
        default="gpt-4o",
        description="OpenAI model for initial responses",
    )
    openai_model_review: str = Field(
        default="gpt-4o",
        description="OpenAI model for review responses",
    )

    # Anthropic Configuration
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key",
    )
    anthropic_model_initial: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model for initial responses",
    )
    anthropic_model_review: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model for review responses",
    )

    # Google Gemini Configuration
    google_api_key: SecretStr | None = Field(
        default=None,
        description="Google API key for Gemini",
    )
    gemini_model_initial: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for initial responses",
    )
    gemini_model_review: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for review responses",
    )

    # Generation Defaults
    default_max_tokens: Annotated[int, Field(ge=1, le=32768)] = Field(
        default=1024,
        description="Default maximum tokens for generation",
    )
    default_temperature: Annotated[float, Field(ge=0.0, le=2.0)] = Field(
        default=0.7,
        description="Default sampling temperature",
    )

    @field_validator("openai_api_key", "anthropic_api_key", "google_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate that API keys are not placeholder values."""
        if v is None:
            return None
        if v.strip() == "":
            return None
        placeholder_values = [
            "your_api_key_here",
            "sk-...",
            "sk-ant-...",
            "<your-api-key>",
        ]
        if v.lower() in [p.lower() for p in placeholder_values]:
            return None
        return v

    def get_model_for_provider(self, provider: ProviderType, stage: str) -> str:
        """Get the model name for a specific provider and stage.

        Args:
            provider: The LLM provider.
            stage: Either 'initial' or 'review'.

        Returns:
            The model name string.

        Raises:
            ValueError: If stage is invalid.
        """
        if stage not in ("initial", "review"):
            raise ValueError(f"Invalid stage: {stage}. Must be 'initial' or 'review'.")

        model_map = {
            ProviderType.OPENAI: {
                "initial": self.openai_model_initial,
                "review": self.openai_model_review,
            },
            ProviderType.CLAUDE: {
                "initial": self.anthropic_model_initial,
                "review": self.anthropic_model_review,
            },
            ProviderType.GEMINI: {
                "initial": self.gemini_model_initial,
                "review": self.gemini_model_review,
            },
        }
        return model_map[provider][stage]

    def get_api_key_for_provider(self, provider: ProviderType) -> SecretStr | None:
        """Get the API key for a specific provider.

        Args:
            provider: The LLM provider.

        Returns:
            The SecretStr API key or None if not configured.
        """
        key_map = {
            ProviderType.OPENAI: self.openai_api_key,
            ProviderType.CLAUDE: self.anthropic_api_key,
            ProviderType.GEMINI: self.google_api_key,
        }
        return key_map[provider]

    def validate_provider_config(self, provider: ProviderType) -> None:
        """Validate that a provider is properly configured.

        Args:
            provider: The LLM provider to validate.

        Raises:
            ConfigurationError: If the provider is not properly configured.
        """
        from llm_chain.exceptions import ConfigurationError

        api_key = self.get_api_key_for_provider(provider)
        if api_key is None:
            raise ConfigurationError(
                f"API key not configured for provider: {provider.value}. "
                f"Set the appropriate environment variable."
            )
