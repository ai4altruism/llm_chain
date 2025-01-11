# main.py

from src.config_loader import ConfigLoader
from src.prompt_manager import PromptManager
from src.openai_client import OpenAIClient
from src.chaining_service import ChainingService

def main():
    # Load configurations
    config = ConfigLoader()

    # Initialize managers and clients
    prompt_manager = PromptManager()
    openai_client_initial = OpenAIClient(config, model=config.GPT_MODEL_INITIAL)
    openai_client_review = OpenAIClient(config, model=config.GPT_MODEL_REVIEW)
    chaining_service = ChainingService(openai_client_initial, openai_client_review, prompt_manager)

    # Example user input
    user_input = "Please answer the following question(s):"
    #print("User Input:", user_input)  # Added print statement for the initial user prompt

    # Process input through chaining service
    results = chaining_service.process(user_input)

    print("Initial Response:")
    print(results['initial_response'])
    print("\nReview Response:")
    print(results['review_response'])

if __name__ == "__main__":
    main()
