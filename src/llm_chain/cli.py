"""Interactive CLI for LLM Chain.

This module provides a REPL-style interface for interacting with
multiple LLM providers with streaming support.
"""

import asyncio
import sys
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from llm_chain.config import Settings, ProviderType
from llm_chain.providers.factory import ProviderFactory
from llm_chain.providers.base import Message
from llm_chain.chaining import ChainingService, ChainConfig
from llm_chain.exceptions import LLMChainError, ConfigurationError


# Rich console for output
console = Console()


class CLIState:
    """Maintains the state of the CLI session."""

    def __init__(self, settings: Settings | None = None):
        """Initialize CLI state.

        Args:
            settings: Optional settings instance.
        """
        self.settings = settings or Settings()
        self.factory = ProviderFactory(self.settings)
        self.streaming = True
        self.show_review = True
        self._service: ChainingService | None = None

    @property
    def service(self) -> ChainingService:
        """Get or create the chaining service."""
        if self._service is None:
            self._rebuild_service()
        return self._service  # type: ignore

    def _rebuild_service(self) -> None:
        """Rebuild the chaining service with current settings."""
        try:
            initial, review = self.factory.create_chain_providers()
            self._service = ChainingService(initial, review)
        except ConfigurationError as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            self._service = None

    def invalidate_service(self) -> None:
        """Invalidate the current service to force rebuild."""
        self._service = None


# Command handlers
CommandHandler = Callable[[CLIState, list[str]], bool]


def cmd_help(state: CLIState, args: list[str]) -> bool:
    """Show help information."""
    help_text = """
[bold cyan]LLM Chain Interactive CLI[/bold cyan]

[bold]Commands:[/bold]
  /help                          Show this help message
  /config                        Show current configuration
  /provider <stage> <name>       Set provider (stage: initial|review, name: openai|claude|gemini)
  /model <stage> <name>          Set model for a provider stage
  /streaming [on|off]            Toggle streaming mode
  /review [on|off]               Toggle review stage
  /clear                         Clear the screen
  /quit, /exit                   Exit the CLI

[bold]Usage:[/bold]
  Type your message and press Enter to send it to the LLM.
  The initial response will be followed by a review/critique.

[bold]Examples:[/bold]
  /provider initial claude       Use Claude for initial responses
  /provider review gemini        Use Gemini for reviews
  /model initial gpt-4o          Set initial model to gpt-4o
  /streaming off                 Disable streaming output
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))
    return True


def cmd_config(state: CLIState, args: list[str]) -> bool:
    """Show current configuration."""
    s = state.settings

    config_text = f"""
[bold]Provider Configuration:[/bold]
  Initial Provider: [cyan]{s.initial_provider.value}[/cyan]
  Review Provider:  [cyan]{s.review_provider.value}[/cyan]

[bold]Models:[/bold]
  OpenAI Initial:    {s.openai_model_initial}
  OpenAI Review:     {s.openai_model_review}
  Claude Initial:    {s.anthropic_model_initial}
  Claude Review:     {s.anthropic_model_review}
  Gemini Initial:    {s.gemini_model_initial}
  Gemini Review:     {s.gemini_model_review}

[bold]API Keys:[/bold]
  OpenAI:    {'[green]✓ Set[/green]' if s.openai_api_key else '[red]✗ Not set[/red]'}
  Anthropic: {'[green]✓ Set[/green]' if s.anthropic_api_key else '[red]✗ Not set[/red]'}
  Google:    {'[green]✓ Set[/green]' if s.google_api_key else '[red]✗ Not set[/red]'}

[bold]Session Settings:[/bold]
  Streaming: {'[green]On[/green]' if state.streaming else '[yellow]Off[/yellow]'}
  Review:    {'[green]On[/green]' if state.show_review else '[yellow]Off[/yellow]'}
"""
    console.print(Panel(config_text, title="Configuration", border_style="green"))
    return True


def cmd_provider(state: CLIState, args: list[str]) -> bool:
    """Set the provider for a stage."""
    if len(args) != 2:
        console.print("[red]Usage: /provider <initial|review> <openai|claude|gemini>[/red]")
        return True

    stage, provider_name = args[0].lower(), args[1].lower()

    if stage not in ("initial", "review"):
        console.print("[red]Stage must be 'initial' or 'review'[/red]")
        return True

    try:
        provider_type = ProviderType(provider_name)
    except ValueError:
        console.print(f"[red]Invalid provider: {provider_name}. Use openai, claude, or gemini[/red]")
        return True

    # Update settings
    if stage == "initial":
        state.settings.initial_provider = provider_type
    else:
        state.settings.review_provider = provider_type

    state.invalidate_service()
    console.print(f"[green]Set {stage} provider to {provider_name}[/green]")
    return True


def cmd_model(state: CLIState, args: list[str]) -> bool:
    """Set the model for a stage."""
    if len(args) != 2:
        console.print("[red]Usage: /model <initial|review> <model-name>[/red]")
        return True

    stage, model_name = args[0].lower(), args[1]

    if stage not in ("initial", "review"):
        console.print("[red]Stage must be 'initial' or 'review'[/red]")
        return True

    # Update the appropriate model setting
    provider = state.settings.initial_provider if stage == "initial" else state.settings.review_provider

    if provider == ProviderType.OPENAI:
        if stage == "initial":
            state.settings.openai_model_initial = model_name
        else:
            state.settings.openai_model_review = model_name
    elif provider == ProviderType.CLAUDE:
        if stage == "initial":
            state.settings.anthropic_model_initial = model_name
        else:
            state.settings.anthropic_model_review = model_name
    elif provider == ProviderType.GEMINI:
        if stage == "initial":
            state.settings.gemini_model_initial = model_name
        else:
            state.settings.gemini_model_review = model_name

    state.invalidate_service()
    console.print(f"[green]Set {stage} model to {model_name}[/green]")
    return True


def cmd_streaming(state: CLIState, args: list[str]) -> bool:
    """Toggle streaming mode."""
    if args:
        state.streaming = args[0].lower() in ("on", "true", "1", "yes")
    else:
        state.streaming = not state.streaming

    status = "on" if state.streaming else "off"
    console.print(f"[green]Streaming mode: {status}[/green]")
    return True


def cmd_review(state: CLIState, args: list[str]) -> bool:
    """Toggle review stage."""
    if args:
        state.show_review = args[0].lower() in ("on", "true", "1", "yes")
    else:
        state.show_review = not state.show_review

    status = "on" if state.show_review else "off"
    console.print(f"[green]Review stage: {status}[/green]")
    return True


def cmd_clear(state: CLIState, args: list[str]) -> bool:
    """Clear the screen."""
    console.clear()
    return True


def cmd_quit(state: CLIState, args: list[str]) -> bool:
    """Exit the CLI."""
    console.print("[yellow]Goodbye![/yellow]")
    return False


# Command registry
COMMANDS: dict[str, CommandHandler] = {
    "/help": cmd_help,
    "/config": cmd_config,
    "/provider": cmd_provider,
    "/model": cmd_model,
    "/streaming": cmd_streaming,
    "/review": cmd_review,
    "/clear": cmd_clear,
    "/quit": cmd_quit,
    "/exit": cmd_quit,
}


async def process_message_streaming(state: CLIState, user_input: str) -> None:
    """Process user message with streaming output."""
    messages = [Message.user(user_input)]

    try:
        if state.show_review:
            # Full chain with streaming
            console.print("\n[bold cyan]Initial Response:[/bold cyan]")

            initial_text = Text()
            review_text = Text()
            current_stage = "initial"

            async for chunk in state.service.process_stream(messages):
                if chunk.is_final:
                    if chunk.stage == "initial" and state.show_review:
                        console.print(initial_text)
                        console.print("\n[bold magenta]Review:[/bold magenta]")
                    elif chunk.stage == "review":
                        console.print(review_text)
                    current_stage = "review" if chunk.stage == "initial" else "done"
                else:
                    if current_stage == "initial":
                        initial_text.append(chunk.content)
                        console.print(chunk.content, end="")
                    elif current_stage == "review":
                        review_text.append(chunk.content)
                        console.print(chunk.content, end="")

            console.print()  # Final newline
        else:
            # Initial only with streaming
            console.print("\n[bold cyan]Response:[/bold cyan]")
            async for chunk in state.service.process_initial_only_stream(messages):
                console.print(chunk, end="")
            console.print("\n")

    except LLMChainError as e:
        console.print(f"\n[red]Error: {e}[/red]")


async def process_message_standard(state: CLIState, user_input: str) -> None:
    """Process user message without streaming."""
    messages = [Message.user(user_input)]

    try:
        if state.show_review:
            with console.status("[bold cyan]Generating response...[/bold cyan]"):
                result = await state.service.process(messages)

            console.print("\n[bold cyan]Initial Response:[/bold cyan]")
            console.print(Markdown(result.initial_response))

            console.print("\n[bold magenta]Review:[/bold magenta]")
            console.print(Markdown(result.review_response))
        else:
            with console.status("[bold cyan]Generating response...[/bold cyan]"):
                response = await state.service.process_initial_only(messages)

            console.print("\n[bold cyan]Response:[/bold cyan]")
            console.print(Markdown(response))

        console.print()

    except LLMChainError as e:
        console.print(f"\n[red]Error: {e}[/red]")


async def process_input(state: CLIState, user_input: str) -> bool:
    """Process user input (command or message).

    Args:
        state: CLI state.
        user_input: User's input string.

    Returns:
        True to continue, False to exit.
    """
    user_input = user_input.strip()

    if not user_input:
        return True

    # Check for commands
    if user_input.startswith("/"):
        parts = user_input.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in COMMANDS:
            return COMMANDS[cmd](state, args)
        else:
            console.print(f"[red]Unknown command: {cmd}. Type /help for available commands.[/red]")
            return True

    # Process as message to LLM
    if state.streaming:
        await process_message_streaming(state, user_input)
    else:
        await process_message_standard(state, user_input)

    return True


async def run_repl(state: CLIState) -> None:
    """Run the interactive REPL loop."""
    # Set up prompt with history and completion
    command_completer = WordCompleter(
        list(COMMANDS.keys()) + ["initial", "review", "openai", "claude", "gemini", "on", "off"],
        ignore_case=True,
    )

    session: PromptSession[str] = PromptSession(
        history=FileHistory(".llm_chain_history"),
        auto_suggest=AutoSuggestFromHistory(),
        completer=command_completer,
    )

    # Welcome message
    console.print(Panel(
        "[bold]Welcome to LLM Chain Interactive CLI[/bold]\n\n"
        "Type your message to chat with the LLM, or use commands:\n"
        "  /help    - Show available commands\n"
        "  /config  - Show current configuration\n"
        "  /quit    - Exit the CLI",
        title="LLM Chain",
        border_style="cyan",
    ))

    # Check if we have valid configuration
    try:
        _ = state.service  # Try to create service
    except Exception as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")
        console.print("[yellow]Use /config to check your settings and /provider to configure.[/yellow]\n")

    # Main loop
    while True:
        try:
            user_input = await session.prompt_async(
                "\n[bold green]You>[/bold green] ",
            )

            should_continue = await process_input(state, user_input)
            if not should_continue:
                break

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /quit to exit[/yellow]")
            continue
        except EOFError:
            break


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    try:
        state = CLIState()
        asyncio.run(run_repl(state))
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
