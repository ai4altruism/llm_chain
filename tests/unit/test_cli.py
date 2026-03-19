"""Unit tests for the CLI module."""

import pytest
from unittest.mock import patch, MagicMock

from llm_chain.cli import (
    CLIState,
    COMMANDS,
    cmd_help,
    cmd_config,
    cmd_provider,
    cmd_model,
    cmd_streaming,
    cmd_review,
    cmd_clear,
    cmd_quit,
    process_input,
)
from llm_chain.config import Settings, ProviderType


class TestCLIState:
    """Tests for CLIState class."""

    def test_init_default(self):
        """Test default initialization."""
        state = CLIState()
        assert state.settings is not None
        assert state.factory is not None
        assert state.streaming is True
        assert state.show_review is True

    def test_init_with_settings(self):
        """Test initialization with custom settings."""
        settings = Settings(openai_api_key="test-key")
        state = CLIState(settings)
        assert state.settings is settings

    def test_invalidate_service(self):
        """Test service invalidation."""
        state = CLIState()
        state._service = MagicMock()
        state.invalidate_service()
        assert state._service is None


class TestCommandRegistry:
    """Tests for the command registry."""

    def test_all_commands_registered(self):
        """Test that all expected commands are registered."""
        expected_commands = [
            "/help",
            "/config",
            "/provider",
            "/model",
            "/streaming",
            "/review",
            "/clear",
            "/quit",
            "/exit",
        ]
        for cmd in expected_commands:
            assert cmd in COMMANDS, f"Command {cmd} not registered"

    def test_commands_are_callable(self):
        """Test that all registered commands are callable."""
        for cmd, handler in COMMANDS.items():
            assert callable(handler), f"Handler for {cmd} is not callable"


class TestCmdHelp:
    """Tests for /help command."""

    def test_help_returns_true(self):
        """Test that help command returns True (continue)."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = cmd_help(state, [])
        assert result is True


class TestCmdConfig:
    """Tests for /config command."""

    def test_config_returns_true(self):
        """Test that config command returns True (continue)."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = cmd_config(state, [])
        assert result is True

    def test_config_shows_settings(self):
        """Test that config displays settings."""
        settings = Settings(
            openai_api_key="test-key",
            initial_provider=ProviderType.CLAUDE,
        )
        state = CLIState(settings)
        with patch("llm_chain.cli.console") as mock_console:
            cmd_config(state, [])
            # Verify print was called
            mock_console.print.assert_called()


class TestCmdProvider:
    """Tests for /provider command."""

    def test_provider_requires_two_args(self):
        """Test that provider command requires two arguments."""
        state = CLIState()
        with patch("llm_chain.cli.console") as mock_console:
            result = cmd_provider(state, [])
            assert result is True
            mock_console.print.assert_called()
            assert "Usage" in str(mock_console.print.call_args)

    def test_provider_invalid_stage(self):
        """Test provider command with invalid stage."""
        state = CLIState()
        with patch("llm_chain.cli.console") as mock_console:
            result = cmd_provider(state, ["invalid", "openai"])
            assert result is True
            assert "initial" in str(mock_console.print.call_args).lower()

    def test_provider_invalid_provider(self):
        """Test provider command with invalid provider name."""
        state = CLIState()
        with patch("llm_chain.cli.console") as mock_console:
            result = cmd_provider(state, ["initial", "invalid"])
            assert result is True
            assert "invalid" in str(mock_console.print.call_args).lower()

    def test_provider_set_initial(self):
        """Test setting initial provider."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = cmd_provider(state, ["initial", "claude"])
        assert result is True
        assert state.settings.initial_provider == ProviderType.CLAUDE

    def test_provider_set_review(self):
        """Test setting review provider."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = cmd_provider(state, ["review", "gemini"])
        assert result is True
        assert state.settings.review_provider == ProviderType.GEMINI

    def test_provider_invalidates_service(self):
        """Test that changing provider invalidates service."""
        state = CLIState()
        state._service = MagicMock()
        with patch("llm_chain.cli.console"):
            cmd_provider(state, ["initial", "claude"])
        assert state._service is None


class TestCmdModel:
    """Tests for /model command."""

    def test_model_requires_two_args(self):
        """Test that model command requires two arguments."""
        state = CLIState()
        with patch("llm_chain.cli.console") as mock_console:
            result = cmd_model(state, [])
            assert result is True
            assert "Usage" in str(mock_console.print.call_args)

    def test_model_invalid_stage(self):
        """Test model command with invalid stage."""
        state = CLIState()
        with patch("llm_chain.cli.console") as mock_console:
            result = cmd_model(state, ["invalid", "gpt-4"])
            assert result is True
            assert "initial" in str(mock_console.print.call_args).lower()

    def test_model_set_openai_initial(self):
        """Test setting OpenAI initial model."""
        settings = Settings(initial_provider=ProviderType.OPENAI)
        state = CLIState(settings)
        with patch("llm_chain.cli.console"):
            result = cmd_model(state, ["initial", "gpt-4-turbo"])
        assert result is True
        assert state.settings.openai_model_initial == "gpt-4-turbo"

    def test_model_set_claude_review(self):
        """Test setting Claude review model."""
        settings = Settings(review_provider=ProviderType.CLAUDE)
        state = CLIState(settings)
        with patch("llm_chain.cli.console"):
            result = cmd_model(state, ["review", "claude-3-opus"])
        assert result is True
        assert state.settings.anthropic_model_review == "claude-3-opus"

    def test_model_set_gemini_initial(self):
        """Test setting Gemini initial model."""
        settings = Settings(initial_provider=ProviderType.GEMINI)
        state = CLIState(settings)
        with patch("llm_chain.cli.console"):
            result = cmd_model(state, ["initial", "gemini-1.5-pro"])
        assert result is True
        assert state.settings.gemini_model_initial == "gemini-1.5-pro"


class TestCmdStreaming:
    """Tests for /streaming command."""

    def test_streaming_toggle(self):
        """Test toggling streaming mode."""
        state = CLIState()
        assert state.streaming is True

        with patch("llm_chain.cli.console"):
            cmd_streaming(state, [])
        assert state.streaming is False

        with patch("llm_chain.cli.console"):
            cmd_streaming(state, [])
        assert state.streaming is True

    def test_streaming_set_on(self):
        """Test setting streaming on explicitly."""
        state = CLIState()
        state.streaming = False

        with patch("llm_chain.cli.console"):
            cmd_streaming(state, ["on"])
        assert state.streaming is True

    def test_streaming_set_off(self):
        """Test setting streaming off explicitly."""
        state = CLIState()

        with patch("llm_chain.cli.console"):
            cmd_streaming(state, ["off"])
        assert state.streaming is False


class TestCmdReview:
    """Tests for /review command."""

    def test_review_toggle(self):
        """Test toggling review mode."""
        state = CLIState()
        assert state.show_review is True

        with patch("llm_chain.cli.console"):
            cmd_review(state, [])
        assert state.show_review is False

        with patch("llm_chain.cli.console"):
            cmd_review(state, [])
        assert state.show_review is True

    def test_review_set_on(self):
        """Test setting review on explicitly."""
        state = CLIState()
        state.show_review = False

        with patch("llm_chain.cli.console"):
            cmd_review(state, ["on"])
        assert state.show_review is True

    def test_review_set_off(self):
        """Test setting review off explicitly."""
        state = CLIState()

        with patch("llm_chain.cli.console"):
            cmd_review(state, ["off"])
        assert state.show_review is False


class TestCmdClear:
    """Tests for /clear command."""

    def test_clear_returns_true(self):
        """Test that clear command returns True."""
        state = CLIState()
        with patch("llm_chain.cli.console") as mock_console:
            result = cmd_clear(state, [])
        assert result is True
        mock_console.clear.assert_called_once()


class TestCmdQuit:
    """Tests for /quit and /exit commands."""

    def test_quit_returns_false(self):
        """Test that quit command returns False (exit)."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = cmd_quit(state, [])
        assert result is False


class TestProcessInput:
    """Tests for process_input function."""

    @pytest.mark.asyncio
    async def test_empty_input_continues(self):
        """Test that empty input returns True."""
        state = CLIState()
        result = await process_input(state, "")
        assert result is True

    @pytest.mark.asyncio
    async def test_whitespace_input_continues(self):
        """Test that whitespace-only input returns True."""
        state = CLIState()
        result = await process_input(state, "   ")
        assert result is True

    @pytest.mark.asyncio
    async def test_help_command(self):
        """Test processing /help command."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = await process_input(state, "/help")
        assert result is True

    @pytest.mark.asyncio
    async def test_quit_command(self):
        """Test processing /quit command."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = await process_input(state, "/quit")
        assert result is False

    @pytest.mark.asyncio
    async def test_exit_command(self):
        """Test processing /exit command."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = await process_input(state, "/exit")
        assert result is False

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        """Test processing unknown command."""
        state = CLIState()
        with patch("llm_chain.cli.console") as mock_console:
            result = await process_input(state, "/unknown")
        assert result is True
        assert "Unknown command" in str(mock_console.print.call_args)

    @pytest.mark.asyncio
    async def test_command_case_insensitive(self):
        """Test that commands are case-insensitive."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = await process_input(state, "/HELP")
        assert result is True

    @pytest.mark.asyncio
    async def test_provider_command_with_args(self):
        """Test processing /provider command with arguments."""
        state = CLIState()
        with patch("llm_chain.cli.console"):
            result = await process_input(state, "/provider initial claude")
        assert result is True
        assert state.settings.initial_provider == ProviderType.CLAUDE


class TestCLIIntegration:
    """Integration tests for CLI components."""

    def test_full_provider_switch_workflow(self):
        """Test switching providers through CLI commands."""
        state = CLIState()

        # Start with default (OpenAI)
        assert state.settings.initial_provider == ProviderType.OPENAI
        assert state.settings.review_provider == ProviderType.OPENAI

        # Switch initial to Claude
        with patch("llm_chain.cli.console"):
            cmd_provider(state, ["initial", "claude"])
        assert state.settings.initial_provider == ProviderType.CLAUDE

        # Switch review to Gemini
        with patch("llm_chain.cli.console"):
            cmd_provider(state, ["review", "gemini"])
        assert state.settings.review_provider == ProviderType.GEMINI

    def test_model_and_provider_workflow(self):
        """Test setting model after provider switch."""
        state = CLIState()

        # Switch to Claude and set model
        with patch("llm_chain.cli.console"):
            cmd_provider(state, ["initial", "claude"])
            cmd_model(state, ["initial", "claude-3-opus"])

        assert state.settings.initial_provider == ProviderType.CLAUDE
        assert state.settings.anthropic_model_initial == "claude-3-opus"
