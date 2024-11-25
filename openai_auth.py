import os
import openai
from dotenv import load_dotenv

def setup_api_key():
    # Load the .env file
    load_dotenv()
   
    # Check if openai.api_key is already set
    if openai.api_key:
        return openai.api_key
    
    try:
        # Try to get the API key from the environment variable
        api_key = os.environ['OPENAI_API_KEY']
    except KeyError:
        # If the environment variable is not set, try to read the API key from a file
        try:
            with open("G:/My Drive/code/API_SECRET", "r") as file:
                api_key = file.read().strip()
        except FileNotFoundError:
            with open("D:/Users/Dan/GoogleDrivePersonal/code/API_SECRET", "r") as file:
                api_key = file.read().strip()

        # Set the environment variable
        os.environ['OPENAI_API_KEY'] = api_key

    return api_key
