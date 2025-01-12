# LLM Chain Assistant

A Python application that demonstrates a fundamental pattern in LLM application development: using multiple language models in a cooperative arrangement. This project implements a two-stage GPT processing pipeline, where one model generates initial responses while a second model reviews and critiques those responses.

This application serves as an educational template and building block, designed to help developers understand:

- How to work with Large Language Models (LLMs) through APIs in a production-like environment
- How to implement "LLM teaming" patterns where multiple models work together
- Basic patterns for prompt engineering and response chaining
- Clean architecture principles in AI application development

While simple in scope, this codebase reflects patterns that can be expanded into more sophisticated applications, such as:

- Multi-stage content generation and verification systems
- AI-powered content moderation platforms
- Complex decision-making pipelines with multiple specialist models
- Self-reviewing AI systems

## Features

- Two-stage LLM processing pipeline
- Configurable LLM models for both stages
- Separate system and user prompts loaded from configuration files
- Environment-based configuration
- Logging and error handling
- Clean, modular architecture

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt-chain-assistant.git
cd gpt-chain-assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
   - Create a `config` directory in the project root
   - Create a `.env` file in the `config` directory with the following content:
```
OPENAI_API_KEY=your_openai_api_key_here
GPT_MODEL_INITIAL=gpt-4
GPT_MODEL_REVIEW=gpt-4o
```

4. Create prompt files:
   - In the `config` directory, create:
     - `system_prompt.txt`: Contains the system prompt for the initial model
     - `user_prompt.txt`: Contains the base user prompt

## Project Structure

```
gpt-chain-assistant/
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── prompt_manager.py
│   ├── openai_client.py
│   └── chaining_service.py
│
├── config/
│   ├── .env
│   ├── system_prompt.txt
│   └── user_prompt.txt
│
├── main.py
└── requirements.txt
```

## Usage

1. Ensure all configuration files are set up properly.

2. Run the application:
```bash
python main.py
```

The application will:
1. Load the configuration and prompts
2. Process the user input through the initial model
3. Send the initial response to the review model for critique
4. Output both the initial response and the review

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `GPT_MODEL_INITIAL`: Model to use for initial response (default: gpt-4)
- `GPT_MODEL_REVIEW`: Model to use for review (default: gpt-4o)

### Prompt Files

- `system_prompt.txt`: Contains the system prompt that defines the assistant's role
- `user_prompt.txt`: Contains the base user prompt that precedes the actual user input

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- A4A Team <team@ai4altruism.org>

## Acknowledgments

- Built with Python and the OpenAI Python client library

## Disclaimer

This project is not affiliated with or endorsed by OpenAI. Please ensure you comply with OpenAI's use policies when using this application with their API.
