# LLM Chain

A modern, async Python library for multi-provider LLM chaining with streaming support. Chain responses from OpenAI, Anthropic Claude, and Google Gemini in a two-stage pipeline where one model generates and another reviews.

## Features

- **Multi-Provider Support**: OpenAI (GPT-4o), Anthropic Claude, and Google Gemini
- **Async/Await Throughout**: Built on modern Python async patterns
- **Streaming Responses**: Real-time token streaming for all providers
- **Two-Stage Chaining**: Initial response generation + automated review/critique
- **Interactive CLI**: REPL interface with provider switching and streaming display
- **Flexible Configuration**: Environment variables and runtime configuration
- **Comprehensive Testing**: 212 tests with 80%+ coverage

## Installation

### Requirements

- Python 3.10+
- API keys for your chosen providers

### Install from Source

```bash
git clone https://github.com/ai4altruism/llm_chain.git
cd llm_chain
pip install -e ".[dev]"
```

### Configure API Keys

Create a `.env` file or set environment variables:

```bash
# Required: At least one provider API key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional: Provider selection (defaults to openai)
INITIAL_PROVIDER=openai    # openai, claude, or gemini
REVIEW_PROVIDER=claude     # openai, claude, or gemini

# Optional: Model overrides
OPENAI_MODEL_INITIAL=gpt-4o
OPENAI_MODEL_REVIEW=gpt-4o
ANTHROPIC_MODEL_INITIAL=claude-sonnet-4-5-20250929
ANTHROPIC_MODEL_REVIEW=claude-sonnet-4-5-20250929
GEMINI_MODEL_INITIAL=gemini-2.0-flash
GEMINI_MODEL_REVIEW=gemini-2.0-flash
```

## Quick Start

### Interactive CLI

```bash
llm-chain
```

```
╭─────────────────── LLM Chain ───────────────────╮
│ Welcome to LLM Chain Interactive CLI            │
│                                                 │
│ Type your message to chat with the LLM          │
│   /help    - Show available commands            │
│   /config  - Show current configuration         │
│   /quit    - Exit the CLI                       │
╰─────────────────────────────────────────────────╯

You> What is quantum computing?

Initial Response:
Quantum computing harnesses quantum mechanical phenomena...

Review:
The explanation is accurate and accessible. It covers the key concepts...

You> /provider initial claude
Set initial provider to claude

You> /quit
Goodbye!
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/config` | Display current configuration |
| `/provider <stage> <name>` | Set provider (stage: initial\|review, name: openai\|claude\|gemini) |
| `/model <stage> <name>` | Set model for current provider |
| `/streaming [on\|off]` | Toggle streaming mode |
| `/review [on\|off]` | Toggle review stage |
| `/clear` | Clear the screen |
| `/quit`, `/exit` | Exit the CLI |

### Programmatic Usage

```python
import asyncio
from llm_chain import (
    Settings,
    ProviderFactory,
    ChainingService,
    Message,
)

async def main():
    # Load settings from environment
    settings = Settings()

    # Create providers using factory
    factory = ProviderFactory(settings)
    initial, review = factory.create_chain_providers()

    # Create chaining service
    service = ChainingService(initial, review)

    # Process a message
    messages = [Message.user("Explain machine learning in simple terms")]
    result = await service.process(messages)

    print("Initial Response:")
    print(result.initial_response)
    print("\nReview:")
    print(result.review_response)

asyncio.run(main())
```

### Streaming Example

```python
async def streaming_example():
    factory = ProviderFactory(Settings())
    initial, review = factory.create_chain_providers()
    service = ChainingService(initial, review)

    messages = [Message.user("Write a haiku about coding")]

    async for chunk in service.process_stream(messages):
        if chunk.is_final:
            print(f"\n--- End of {chunk.stage} ---")
        else:
            print(chunk.content, end="", flush=True)

asyncio.run(streaming_example())
```

### Using Individual Providers

```python
from llm_chain import OpenAIProvider, ClaudeProvider, GeminiProvider, Message

async def single_provider():
    # OpenAI
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")
    response = await provider.generate([Message.user("Hello!")])

    # Streaming
    async for chunk in provider.generate_stream([Message.user("Tell me a story")]):
        print(chunk, end="")

asyncio.run(single_provider())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Interactive CLI                          │
│                    (REPL Interface)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   ChainingService                            │
│              (Async Pipeline Orchestrator)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   ProviderFactory                            │
│           (Dynamic Provider Instantiation)                   │
└───────┬─────────────────┼─────────────────┬─────────────────┘
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│ OpenAIProvider│ │ClaudeProvider │ │GeminiProvider │
│   (Async)     │ │   (Async)     │ │   (Async)     │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Provider Comparison

| Feature | OpenAI | Claude | Gemini |
|---------|--------|--------|--------|
| Default Model | gpt-4o | claude-sonnet-4-5-20250929 | gemini-2.0-flash |
| Streaming | Yes | Yes | Yes |
| System Prompts | In messages | Separate parameter | system_instruction |
| Role Names | user/assistant | user/assistant | user/model |

## Configuration Reference

### Settings Class

```python
from llm_chain import Settings

settings = Settings(
    # Provider selection
    initial_provider="openai",      # or "claude", "gemini"
    review_provider="claude",

    # API keys (can also use environment variables)
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    google_api_key="...",

    # Model configuration
    openai_model_initial="gpt-4o",
    openai_model_review="gpt-4o",
    anthropic_model_initial="claude-sonnet-4-5-20250929",
    anthropic_model_review="claude-sonnet-4-5-20250929",
    gemini_model_initial="gemini-2.0-flash",
    gemini_model_review="gemini-2.0-flash",

    # Generation defaults
    default_max_tokens=1024,
    default_temperature=0.7,
)
```

### Generation Config

```python
from llm_chain import GenerationConfig

config = GenerationConfig(
    max_tokens=2048,
    temperature=0.5,
    top_p=0.9,
    stop_sequences=["END", "STOP"],
)

result = await service.process(messages, config=config)
```

### Chain Config

```python
from llm_chain import ChainConfig

chain_config = ChainConfig(
    review_system_prompt="You are an expert code reviewer.",
    review_instruction="Review this code:\n\n{initial_response}\n\nProvide feedback.",
)

service = ChainingService(initial, review, config=chain_config)
```

## Error Handling

```python
from llm_chain import (
    LLMChainError,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    StreamingError,
    ConfigurationError,
)

try:
    result = await service.process(messages)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ModelNotFoundError:
    print("Model not available")
except StreamingError:
    print("Streaming interrupted")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Development

### Setup

```bash
git clone https://github.com/ai4altruism/llm_chain.git
cd llm_chain
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=llm_chain --cov-report=term-missing

# Specific test file
pytest tests/unit/test_openai_provider.py -v
```

### Code Quality

```bash
# Format
ruff format src tests

# Lint
ruff check src tests

# Type check
mypy src
```

## Project Structure

```
llm_chain/
├── src/llm_chain/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # Settings and configuration
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── chaining.py          # ChainingService implementation
│   ├── cli.py               # Interactive CLI
│   └── providers/
│       ├── __init__.py      # Provider exports
│       ├── base.py          # Abstract LLMProvider
│       ├── factory.py       # ProviderFactory
│       ├── openai.py        # OpenAI implementation
│       ├── claude.py        # Anthropic implementation
│       └── gemini.py        # Google implementation
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── docs/
│   └── SOFTWARE_DEVELOPMENT_PLAN.md
└── pyproject.toml           # Project configuration
```

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

Copyright (c) 2025 AI for Altruism Inc.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Contact

- **GitHub Issues**: [github.com/ai4altruism/llm_chain/issues](https://github.com/ai4altruism/llm_chain/issues)
- **Email**: team@ai4altruism.org

## Acknowledgments

Built with:
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Google GenAI SDK](https://github.com/googleapis/python-genai)
- [prompt-toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)
- [Rich](https://github.com/Textualize/rich)
