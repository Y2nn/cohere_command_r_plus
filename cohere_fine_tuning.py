# Import the necessary Python package with the functions.
from cohere.finetuning import FinetunedModel, Settings, BaseModel
import cohere

from dotenv import find_dotenv, load_dotenv

import os


_ = load_dotenv(find_dotenv())

# Create a Cohere client object
api_key = os.environ.get('COHERE_API_KEY')
co = cohere.Client(api_key=api_key)


# cohere dotenv
if __name__ == '__main__':
    """
    the Cohere API allows us to fine-tune the model on a custom dataset. 
    To do so, we upload the data and run the fine-tuning function. It is that simple. 

    For our example, we will generate two custom datasets using the ChatGPT (GPT-4o) model (image: 'two custom datasets')
    Using the generated data, create a “positive_bot_train” and a “positive_bot_eval” JSONL file.
    """

    # Provide the file location to the .datasets.create() function.
    # The function also requires the dataset's name and fine-tuning type.
    my_dataset = co.datasets.create(
        name="Happy assistant",
        type="chat-finetune-input",
        data=open("./data/positive_bot_train.jsonl", "rb"),
        eval_data=open("./data/positive_bot_eval.jsonl", "rb")
    )

    result = co.wait(my_dataset)
    print(result)

    """
    As we can see, 
    the function has validated the dataset and uploaded it to the Cohere cloud storage 
    (image: 'custom dataset uploaded to cohere cloud storage').
    """

    # The .finetuning.create_finetuned_model() function will initiate the fine-tuning process in the cloud.
    # start training a custom model using the dataset
    finetuned_model = co.finetuning.create_finetuned_model(
        request=FinetunedModel(
            name="happy-chat-bot-model",  # model name
            settings=Settings(
                base_model=BaseModel(
                    base_type="BASE_TYPE_CHAT",  # base_type
                ),
                dataset_id=my_dataset.id,  # dataset_id
            ),
        ),
    )

    """
    To view the progress of model fine-tuning, 
    go to the Cohere dashboard and click on the “Fine-tuning” option on the left panel (image: fine-tuning). 
    
    It can take a few minutes to fine-tune the model and generate the evaluation report (image: evaluation report).
    
    Once the model is fine-tuned, we get an overview of the results. 
    As you can see in the above image, the model's accuracy is quite low. 
    Why? We have only provided it with a two-row dataset. 
    To improve the accuracy, try providing it with real-world data of at least 1000 rows.
    """

    # To access our fine-tuned model, we can provide the model ID to the .chat() function.
    # You can find the model ID by going to the dashboard, selecting the “Fine-tuning” option,
    # then "YOUR MODELS," and copying the model ID from the list of fine-tuned models.
    response = co.chat(
        model="8f34d596-b94e-4395-afad-1db35b2b0b53-ft",  # model ID
        preamble="You are a happy chatbot that puts a positive spin on everything.",
        message="I burned my finger while barbecuing.",
        max_tokens=100
    )

    print(response.text)
