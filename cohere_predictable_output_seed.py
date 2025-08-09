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
    # Predictable output is a unique feature of Cohere.
    # By setting the seed argument, we can make the model generate the same response to the same prompt.
    # Usually, when you ask an LLM the same question twice, you will receive a different answer.
    # Setting the seed ensures consistent and reproducible results, like any machine learning model.
    response = co.chat(
        model="command-r",
        message="say a random name",
        seed=55
    )

    # In the following example, by setting the seed argument to 55, you will always receive "Matilda" as a response.
    # To test our theory, we can asked the model the same question with the same seed 2 times
    # and it has produced the same result.
    print(response.text)