# src/config_loader.py

import os
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self, env_path: str = None):
        if env_path is None:
            # Get the absolute path to the .env file based on the current file's location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(current_dir, '..', 'config', '.env')
        
        load_dotenv(dotenv_path=env_path)
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.GPT_MODEL_INITIAL = os.getenv('GPT_MODEL_INITIAL', 'gpt-4')
        self.GPT_MODEL_REVIEW = os.getenv('GPT_MODEL_REVIEW', 'gpt-4o')

        if not self.OPENAI_API_KEY or self.OPENAI_API_KEY == 'your_openai_api_key_here':
            raise ValueError("OPENAI_API_KEY not found or not set in the environment variables.")
