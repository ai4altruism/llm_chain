# src/prompt_manager.py

import os

class PromptManager:
    """
    Manages loading and formatting of system and user prompts.
    """
    def __init__(self, system_prompt_path: str = None, user_prompt_path: str = None):
        if system_prompt_path is None or user_prompt_path is None:
            # Determine the absolute paths based on the current file's location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            system_prompt_path = os.path.join(current_dir, '..', 'config', 'system_prompt.txt')
            user_prompt_path = os.path.join(current_dir, '..', 'config', 'user_prompt.txt')
        
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path

    def load_system_prompt(self) -> str:
        """
        Loads the system prompt from the specified file.

        Returns:
            str: The content of the system prompt.
        """
        if not os.path.exists(self.system_prompt_path):
            raise FileNotFoundError(f"System prompt file not found at path: {self.system_prompt_path}")
        
        with open(self.system_prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def load_user_prompt(self) -> str:
        """
        Loads the user prompt from the specified file.

        Returns:
            str: The content of the user prompt.
        """
        if not os.path.exists(self.user_prompt_path):
            raise FileNotFoundError(f"User prompt file not found at path: {self.user_prompt_path}")
        
        with open(self.user_prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    def get_full_prompt(self, user_input: str) -> list:
        """
        Combines system and user prompts with user input to form the message structure.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            list: A list of message dictionaries for the OpenAI API.
        """
        return [
            {
                "role": "system",
                "content": self.load_system_prompt()
            },
            {
                "role": "user",
                "content": f"{self.load_user_prompt()} {user_input}"
            }
        ]
