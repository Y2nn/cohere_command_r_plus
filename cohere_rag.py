# Import the necessary Python package with the functions.
from dotenv import find_dotenv, load_dotenv

import cohere

import os


_ = load_dotenv(find_dotenv())

# Create a Cohere client object
api_key = os.environ.get('COHERE_API_KEY')
co = cohere.Client(api_key=api_key)


anime = [
    {
        "title": "Naruto Popularity Analysis",
        "snippet": "Naruto's global success: massive manga sales, top anime ratings, extensive merchandise, and a dedicated fanbase. The series' impact on anime culture and its influence on subsequent shonen series is undeniable."
    },
    {
        "title": "One Piece Popularity Analysis",
        "snippet": "One Piece's record-breaking manga sales and its status as a long-running anime phenomenon highlight its popularity. The series' captivating story and characters have made it a staple in the anime community."
    },
    {
        "title": "Attack on Titan Popularity Analysis",
        "snippet": "Attack on Titan's intense storyline and high-quality animation have garnered a massive following. Its success in both manga and anime formats demonstrates its widespread appeal."
    },
    {
        "title": "My Hero Academia Popularity Analysis",
        "snippet": "My Hero Academia's rapid rise to fame is marked by its engaging characters and compelling plot. The series has achieved impressive manga sales and anime viewership."
    }
]


# cohere dotenv
if __name__ == '__main__':
    # The Cohere API offers a built-in function for performing RAG.
    # Provide the .chat() function with a documents argument.
    # Use anime research documents as an example.
    # Each document should contain the title and snippet keys.
    # To generate precise and contextual answers,
    # we will provide the anime documents to the documents argument in the .chat() function.
    # Upon asking a question, it will run a similarity search on the documents to generate context-aware answers.
    response = co.chat(
        model="command-r-plus",
        message="Which Anime series have most engaging characters?",
        documents=anime
    )

    # As you can see, the model uses documents to generate highly accurate answers.
    # print(response.text)

    # If you want to know what is happening in the background and how the model generates the response,
    # you can simply print the whole response with metadata.
    # Notice the ChatCitation part and how Cohere’s .chat() function has used snippets from the documents
    # to generate the response.
    # print(response)

    # We can also connect tools and connectors to the .chat() function.
    # In the following example, we connect an internet search engine to the model to generate an updated answer.
    response = co.chat(
        model="command-r-plus",
        message="Which Anime series have most engaging characters?",
        connectors=[{"id": "web-search"}]
    )

    # In this case, the model looks up the information on the Internet and then provides the model with the context
    # to generate updated and accurate results.
    print(response.text)

    """You can discover more about the strengths of LLMs with effective information retrieval mechanisms"""