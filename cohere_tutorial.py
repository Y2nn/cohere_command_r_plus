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
    # To generate a response using the .chat() function, provide the model name and the message.
    response = co.chat(
        model="command-r-plus",
        message="Please help me write an email to the angry boss, who thinks I made the changes to the data pipeline "
                "but didn't. It was James."
    )
    # The response generation should be fast and highly relevant.
    print(response.text)
