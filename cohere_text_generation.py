# Import the necessary Python package with the functions.
from dotenv import find_dotenv, load_dotenv

import cohere

import os


_ = load_dotenv(find_dotenv())

# Create a Cohere client object
api_key = os.environ.get('COHERE_API_KEY')
co = cohere.Client(api_key=api_key)


# cohere dotenv
if __name__ == '__main__':
    # Provide it with extra arguments like system prompt (preamble), chat_history, max_tokens, and temperature.
    response = co.chat(
        model="command-r-plus",
        preamble="You are a happy chatbot that puts a positive spin on everything.",  # system prompt (preamble)
        chat_history=[
            {"role": "USER", "text": "Hey, my name is Abid!"},
            {"role": "CHATBOT", "text": "Hey Abid! How can I help you today?"}  # chat_history
        ],
        message="I can't swim?",
        max_tokens=150,  # max_tokens
        temperature=0.7  # temperature
    )
    # Based on the additional arguments, the model has modified the response.
    print(response.text)