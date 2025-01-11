# src/chaining_service.py

from src.openai_client import OpenAIClient
from src.prompt_manager import PromptManager

class ChainingService:
    def __init__(self, client_initial: OpenAIClient, client_review: OpenAIClient, prompt_manager: PromptManager):
        self.client_initial = client_initial
        self.client_review = client_review
        self.prompt_manager = prompt_manager

    def process(self, user_input: str) -> dict:
        """
        Processes user input by generating an initial response and then reviewing it.

        Args:
            user_input (str): The input provided by the user.

        Returns:
            dict: A dictionary containing the initial and review responses.
        """
        # Initial model response
        messages = self.prompt_manager.get_full_prompt(user_input)
        initial_response = self.client_initial.generate_response(messages)
        
        print("User Prompt:", messages[1]['content'])  # Added print statement for the initial user prompt text

        # Review model response
        review_prompt = "Please review the following response for accuracy and clarity:\n\n" + initial_response
        system_review_msg = {
            "role": "system",
            "content": "You are an assistant that reviews and critiques responses for accuracy and clarity."
        }
        user_review_msg = {
            "role": "user",
            "content": review_prompt
        }
        review_response = self.client_review.generate_response([system_review_msg, user_review_msg])

        return {
            "initial_response": initial_response,
            "review_response": review_response
        }
