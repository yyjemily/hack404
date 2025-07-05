from dotenv import load_dotenv
import os
from google import genai

# Load environment variables from .env file
load_dotenv()

# Get the key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❗ API key not found. Please check your .env file.")
else:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-1.5-flash", contents= """
        You are an expert dental radiologist with 15 years of experience. 
        Analyze dental X-rays and provide clinical insights.

        Question: What are the key features to look for when diagnosing tooth decay in OPG X-rays?
        """
    )
    print(response.text)


