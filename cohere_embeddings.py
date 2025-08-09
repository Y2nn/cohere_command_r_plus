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
    """Text embeddings are numerical representations of text that capture semantic meaning, 
        allowing for efficient similarity search and analysis of textual data."""

    texts = ['I love you', 'I hate you', 'Who are you?']

    # Cohere’s .embed() function can convert text into embedding vectors for search queries.
    response = co.embed(
        model='embed-english-v3.0',  # model name
        texts=texts,  # list of texts
        input_type='search_query',  # input_type
        embedding_types=['float']  # embedding_types
    )

    embeddings = response.embeddings.float  # All text embeddings
    print(embeddings[2][:5])

    """Cohere also allows us to convert multiple language texts into embeddings"""

    texts = [
        'I love you', 'Te quiero', 'Ich liebe dich',
        'Ti amo', 'Я тебя люблю', ' 我爱你 ',
        '愛してる', 'أحبك', 'मैं तुमसे प्यार करता हूँ'
    ]

    # To do so, change the embedding model to “embed-multilingual-v3.0” and set the input_type to “classification”.
    response = co.embed(
        model='embed-multilingual-v3.0',  # model name
        texts=texts,
        input_type='classification',  # input_type
        embedding_types=['float'])

    embeddings = response.embeddings.float  # All text embeddings
    print(embeddings[2][:5])
