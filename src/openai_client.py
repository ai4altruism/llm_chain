# src/openai_client.py

import logging
from openai import OpenAI, OpenAIError
from src.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    A client to interact with the OpenAI ChatCompletion API.

    Attributes:
        client (OpenAI): An instance of the OpenAI client.
        model (str): The OpenAI model to use for generating responses.
    """

    def __init__(self, config: ConfigLoader, model: str = None):
        """
        Initializes the OpenAIClient with the given configuration and model.

        Args:
            config (ConfigLoader): Configuration loader instance.
            model (str, optional): Specific model to override the default.
        """
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model or config.GPT_MODEL_INITIAL

    def generate_response(self, messages: list) -> str:
        """
        Generates a response from the OpenAI API based on the provided messages.

        Args:
            messages (list): A list of message dictionaries for the conversation.

        Returns:
            str: The content of the generated response or an empty string on failure.
        """
        try:
            #logger.info(f"Sending messages to OpenAI: {messages}")  # Logging the messages sent
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            return ""
