# Software Development Plan (SDP)
## LLM Chain Assistant - Multi-Provider Modernization

**Version:** 1.0
**Date:** 2026-03-17
**Project:** llm_chain

---

## 1. Executive Summary

This document outlines the development plan to modernize the LLM Chain Assistant with:
- Multi-provider support (OpenAI, Anthropic Claude, Google Gemini)
- Async/await architecture throughout
- Streaming response support
- Interactive CLI (REPL-style) interface
- Comprehensive testing infrastructure (unit + integration)

---

## 2. Technical Research Summary

### 2.1 OpenAI API (Latest)

**SDK:** `openai` (v1.x+)
**Key Changes:**
- Client-based instantiation: `OpenAI()` / `AsyncOpenAI()`
- Chat Completions API remains supported (Responses API is newer alternative)
- Async streaming via `async for` iteration

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()
stream = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
async for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

**Sources:**
- [OpenAI Python SDK GitHub](https://github.com/openai/openai-python)
- [OpenAI Streaming Guide](https://developers.openai.com/api/docs/guides/streaming-responses/)

### 2.2 Anthropic Claude API (Latest)

**SDK:** `anthropic` (with optional `[aiohttp]` for better async)
**Python:** 3.9+ required
**Key Features:**
- `AsyncAnthropic` client for async operations
- `client.messages.stream()` - context manager with event iteration
- `client.messages.create(..., stream=True)` - lower memory alternative

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
async with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Sources:**
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Claude Streaming Messages](https://docs.anthropic.com/en/api/messages-streaming)

### 2.3 Google Gemini API (Latest)

**SDK:** `google-genai` (GA as of May 2025)
**Note:** Legacy `google-generativeai` deprecated November 2025
**Key Features:**
- Async via `client.aio.models.generate_content()`
- Streaming via `generate_content_stream()` suffix methods

```python
from google import genai

client = genai.Client()
async for chunk in await client.aio.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents="Hello"
):
    print(chunk.text, end="")
```

**Sources:**
- [Google Gen AI SDK](https://github.com/googleapis/python-genai)
- [Gemini API Libraries](https://ai.google.dev/gemini-api/docs/libraries)

---

## 3. Architecture Overview

### 3.1 Target Architecture

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
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
              ┌───────────▼───────────┐
              │   LLMProvider (ABC)   │
              │  - generate()         │
              │  - generate_stream()  │
              └───────────────────────┘
```

### 3.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Provider Interface | Abstract Base Class | Type safety, clear contract |
| Async Framework | Native asyncio | Standard library, broad compatibility |
| Streaming | Async generators | Natural fit with async/await |
| CLI Framework | `prompt_toolkit` | Rich REPL features, async support |
| Testing | pytest + pytest-asyncio | Industry standard, async support |
| Config | pydantic-settings | Validation, environment loading |

---

## 4. Sprint Plan

### Sprint Overview

| Sprint | Focus | Branch |
|--------|-------|--------|
| 1 | Project Infrastructure & Testing Setup | `sprint-1/infrastructure` |
| 2 | OpenAI Provider Modernization | `sprint-2/openai-provider` |
| 3 | Anthropic Claude Provider | `sprint-3/claude-provider` |
| 4 | Google Gemini Provider | `sprint-4/gemini-provider` |
| 5 | Provider Factory & Configuration | `sprint-5/provider-factory` |
| 6 | Async Chaining Service | `sprint-6/async-chaining` |
| 7 | Interactive CLI | `sprint-7/interactive-cli` |
| 8 | Polish & Documentation | `sprint-8/polish` |

---

### Sprint 1: Project Infrastructure & Testing Setup

**Branch:** `sprint-1/infrastructure`

**Objectives:**
- Modernize project structure with `pyproject.toml`
- Set up pytest infrastructure with async support
- Create abstract provider interface
- Establish base async patterns

**Tasks:**

1. **Project Configuration**
   - Create `pyproject.toml` with modern Python packaging
   - Define dependency groups (core, dev, test)
   - Configure pytest, black, ruff

2. **Directory Structure**
   ```
   src/
   ├── llm_chain/
   │   ├── __init__.py
   │   ├── providers/
   │   │   ├── __init__.py
   │   │   └── base.py          # Abstract LLMProvider
   │   ├── config.py            # Pydantic settings
   │   └── exceptions.py        # Custom exceptions
   tests/
   ├── __init__.py
   ├── conftest.py              # Shared fixtures
   ├── unit/
   │   └── __init__.py
   └── integration/
       └── __init__.py
   ```

3. **Abstract Provider Interface**
   ```python
   class LLMProvider(ABC):
       @abstractmethod
       async def generate(self, messages: list[Message]) -> str: ...

       @abstractmethod
       async def generate_stream(self, messages: list[Message]) -> AsyncIterator[str]: ...
   ```

4. **Testing Infrastructure**
   - pytest configuration with asyncio mode
   - Shared fixtures for mocking
   - Test utilities for async operations

**Deliverables:**
- [ ] `pyproject.toml` with all dependencies
- [ ] Abstract `LLMProvider` base class
- [ ] Pydantic configuration model
- [ ] pytest setup with sample tests
- [ ] Pre-commit hooks (optional)

**Dependencies:** None

---

### Sprint 2: OpenAI Provider Modernization

**Branch:** `sprint-2/openai-provider`

**Objectives:**
- Implement OpenAI provider using latest SDK patterns
- Full async support with `AsyncOpenAI`
- Streaming via async generators
- Comprehensive unit tests

**Tasks:**

1. **OpenAI Provider Implementation**
   - Create `providers/openai.py`
   - Use `AsyncOpenAI` client
   - Implement `generate()` method
   - Implement `generate_stream()` method

2. **Configuration**
   - Add OpenAI-specific config fields
   - API key validation
   - Model selection (default: gpt-4o)

3. **Unit Tests**
   - Mock `AsyncOpenAI` client
   - Test successful generation
   - Test streaming chunk handling
   - Test error scenarios (rate limit, auth, etc.)

**Deliverables:**
- [ ] `OpenAIProvider` class
- [ ] OpenAI configuration integration
- [ ] Unit tests with >90% coverage
- [ ] Error handling for API failures

**Dependencies:** Sprint 1

---

### Sprint 3: Anthropic Claude Provider

**Branch:** `sprint-3/claude-provider`

**Objectives:**
- Implement Claude provider using latest SDK
- Async support with `AsyncAnthropic`
- Streaming via message stream context manager
- Unit tests

**Tasks:**

1. **Claude Provider Implementation**
   - Create `providers/claude.py`
   - Use `AsyncAnthropic` client
   - Implement `generate()` method
   - Implement `generate_stream()` with `messages.stream()`

2. **Message Format Adaptation**
   - Map internal message format to Claude's format
   - Handle system prompts (Claude uses separate parameter)

3. **Configuration**
   - Add Anthropic-specific config fields
   - API key validation
   - Model selection (default: claude-sonnet-4-5-20250929)

4. **Unit Tests**
   - Mock `AsyncAnthropic` client
   - Test message format conversion
   - Test streaming behavior
   - Test error scenarios

**Deliverables:**
- [ ] `ClaudeProvider` class
- [ ] Claude configuration integration
- [ ] Unit tests with >90% coverage
- [ ] Message format mapping utilities

**Dependencies:** Sprint 1

---

### Sprint 4: Google Gemini Provider

**Branch:** `sprint-4/gemini-provider`

**Objectives:**
- Implement Gemini provider using new `google-genai` SDK
- Async support via `client.aio`
- Streaming via `generate_content_stream()`
- Unit tests

**Tasks:**

1. **Gemini Provider Implementation**
   - Create `providers/gemini.py`
   - Use `genai.Client()` with `client.aio`
   - Implement `generate()` method
   - Implement `generate_stream()` method

2. **Message Format Adaptation**
   - Map internal message format to Gemini's `contents` format
   - Handle role mappings (user/model vs user/assistant)

3. **Configuration**
   - Add Gemini-specific config fields
   - API key validation
   - Model selection (default: gemini-2.0-flash)

4. **Unit Tests**
   - Mock `genai.Client`
   - Test content format conversion
   - Test streaming behavior
   - Test error scenarios

**Deliverables:**
- [ ] `GeminiProvider` class
- [ ] Gemini configuration integration
- [ ] Unit tests with >90% coverage
- [ ] Content format mapping utilities

**Dependencies:** Sprint 1

---

### Sprint 5: Provider Factory & Configuration

**Branch:** `sprint-5/provider-factory`

**Objectives:**
- Create factory for dynamic provider instantiation
- Unified configuration for all providers
- Support for mixed provider chains
- Integration tests

**Tasks:**

1. **Provider Factory**
   - Create `providers/factory.py`
   - Registry pattern for providers
   - Instantiate by provider name string
   - Lazy initialization

2. **Unified Configuration**
   - Single config model with all provider options
   - Environment variable support for all keys
   - Validation for provider-specific requirements

3. **Provider Selection**
   - Configure initial/review providers independently
   - Allow same or different providers
   - Runtime provider switching

4. **Integration Tests**
   - Test factory creates correct provider types
   - Test configuration loading from environment
   - Test mixed provider configurations

**Deliverables:**
- [ ] `ProviderFactory` class
- [ ] Unified configuration model
- [ ] Integration tests for factory
- [ ] Environment variable documentation

**Dependencies:** Sprints 2, 3, 4

---

### Sprint 6: Async Chaining Service

**Branch:** `sprint-6/async-chaining`

**Objectives:**
- Modernize `ChainingService` with async/await
- Support streaming through the pipeline
- Mixed provider chain support
- Integration tests

**Tasks:**

1. **Async Chaining Service**
   - Rewrite `ChainingService` as async
   - Accept any `LLMProvider` instances
   - `async def process()` method

2. **Streaming Pipeline**
   - `async def process_stream()` method
   - Yield initial response chunks
   - Then yield review response chunks
   - Optional: parallel initial + review

3. **Prompt Management**
   - Update `PromptManager` for async file I/O (optional)
   - Flexible review prompt construction

4. **Integration Tests**
   - Test with mocked providers
   - Test streaming output
   - Test mixed provider chains
   - Test error propagation

**Deliverables:**
- [ ] Async `ChainingService` class
- [ ] Streaming pipeline support
- [ ] Updated `PromptManager`
- [ ] Integration tests

**Dependencies:** Sprint 5

---

### Sprint 7: Interactive CLI

**Branch:** `sprint-7/interactive-cli`

**Objectives:**
- REPL-style interactive interface
- Real-time streaming display
- Provider selection commands
- Graceful error handling

**Tasks:**

1. **CLI Framework Setup**
   - Use `prompt_toolkit` for rich REPL
   - Async event loop integration
   - Command parsing

2. **Core Commands**
   - `/provider [initial|review] [openai|claude|gemini]` - switch providers
   - `/model [initial|review] [model-name]` - switch models
   - `/config` - show current configuration
   - `/help` - show available commands
   - `/quit` or `/exit` - exit REPL

3. **Streaming Display**
   - Real-time token display
   - Visual distinction between initial/review
   - Progress indicators

4. **User Experience**
   - Command history
   - Auto-completion for commands
   - Colored output
   - Error messages

**Deliverables:**
- [ ] Interactive CLI entry point
- [ ] Command parser and handlers
- [ ] Streaming output display
- [ ] User documentation for CLI

**Dependencies:** Sprint 6

---

### Sprint 8: Polish & Documentation

**Branch:** `sprint-8/polish`

**Objectives:**
- Comprehensive error handling
- User documentation
- Code cleanup and consistency
- Final integration tests

**Tasks:**

1. **Error Handling**
   - Graceful API error recovery
   - User-friendly error messages
   - Retry logic for transient failures

2. **Documentation**
   - Update README.md
   - CLI usage guide
   - Configuration reference
   - Provider comparison

3. **Code Quality**
   - Consistent code style (ruff/black)
   - Type hints throughout
   - Docstrings for public APIs

4. **Final Testing**
   - End-to-end integration tests
   - Manual testing checklist
   - Performance baseline

**Deliverables:**
- [ ] Updated README.md
- [ ] CLI usage documentation
- [ ] Clean, consistent codebase
- [ ] Full test coverage report

**Dependencies:** Sprint 7

---

## 5. Dependencies

### Python Version
- **Minimum:** Python 3.10 (for modern typing features)
- **Recommended:** Python 3.11+

### Core Dependencies

```toml
[project.dependencies]
openai = ">=1.0.0"
anthropic = { version = ">=0.30.0", extras = ["aiohttp"] }
google-genai = { version = ">=1.0.0", extras = ["aiohttp"] }
pydantic-settings = ">=2.0.0"
prompt-toolkit = ">=3.0.0"
rich = ">=13.0.0"  # For colored CLI output
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "respx>=0.21.0",  # For httpx mocking
    "ruff>=0.3.0",
    "black>=24.0.0",
    "mypy>=1.8.0",
]
```

---

## 6. Git Workflow

### Branch Strategy

```
main
  └── sprint-1/infrastructure
        └── (PR & merge)
  └── sprint-2/openai-provider
        └── (PR & merge)
  └── sprint-3/claude-provider
        └── (PR & merge)
  ... and so on
```

### Workflow Per Sprint

1. **Start Sprint:**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b sprint-N/feature-name
   ```

2. **During Sprint:**
   - Regular commits with descriptive messages
   - Run tests locally before pushing

3. **Complete Sprint:**
   ```bash
   git push -u origin sprint-N/feature-name
   ```
   - Create PR on GitHub
   - Wait for review/merge confirmation

4. **After Merge:**
   - User confirms merge complete
   - Proceed to next sprint

---

## 7. Testing Strategy

### Unit Tests
- Mock all external API calls
- Test each provider independently
- Test configuration validation
- Test error handling paths

### Integration Tests
- Test provider factory
- Test chaining service with mock providers
- Test CLI command parsing
- Test streaming assembly

### Test Coverage Target
- **Unit Tests:** >90% coverage
- **Integration Tests:** Critical paths covered
- **Overall:** >85% coverage

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| API breaking changes | Medium | Pin SDK versions, test against latest |
| Rate limiting during tests | Low | Use mocks, have skip markers for live tests |
| Async complexity | Medium | Thorough testing, simple patterns |
| Provider API differences | Medium | Adapter pattern, good abstraction |

---

## 9. Success Criteria

- [ ] All three providers (OpenAI, Claude, Gemini) working
- [ ] Streaming works for all providers
- [ ] Mixed provider chains functional
- [ ] Interactive CLI operational
- [ ] Test coverage >85%
- [ ] Documentation complete
- [ ] Clean git history with per-sprint PRs

---

## Appendix A: Message Format Mapping

### Internal Format (Canonical)
```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
```

### OpenAI Format
```python
{"role": "system|user|assistant", "content": "..."}
```

### Claude Format
```python
# System prompt is separate parameter
system="..."
messages=[{"role": "user|assistant", "content": "..."}]
```

### Gemini Format
```python
# Uses 'user' and 'model' roles
contents=[{"role": "user|model", "parts": [{"text": "..."}]}]
# Or simple string for single turn
contents="..."
```

---

## Appendix B: Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_INITIAL=gpt-4o
OPENAI_MODEL_REVIEW=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_INITIAL=claude-sonnet-4-5-20250929
ANTHROPIC_MODEL_REVIEW=claude-sonnet-4-5-20250929

# Google
GOOGLE_API_KEY=...
GEMINI_MODEL_INITIAL=gemini-2.0-flash
GEMINI_MODEL_REVIEW=gemini-2.0-flash

# Chain Configuration
INITIAL_PROVIDER=openai|claude|gemini
REVIEW_PROVIDER=openai|claude|gemini
```
