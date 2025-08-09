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
    # Stream the response by using the .chat_stream() function.
    response = co.chat_stream(
        model="command-r-plus",
        message="Tell me something interesting about the galaxy?"
    )

    # This function allows us to generate responses in real time, producing tokens as they become available,
    # which enhances the perceived speed of the model.
    for event in response:
        if event.event_type == "text-generation":
            print(event.text, end="")
        elif event.event_type == "stream-end":
            print(event.finish_reason, end="")
