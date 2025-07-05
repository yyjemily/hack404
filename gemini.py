from dotenv import load_dotenv
import os
from google import genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Get the key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key not found. Please check your .env file.")

client = genai.Client(api_key=api_key)

uploaded_file = Image.open("cavity1.jpg")

prompt = """
You are a dental diagnostic assistant. Your task is to analyze dental images (X-rays, intraoral photos, or CBCT slices) and provide a preliminary diagnostic report. Focus on identifying common oral health conditions such as cavities, gum disease, bone loss, missing teeth, impacted teeth, and other abnormalities. Be concise, professional, and clinically accurate.

Instructions: Describe any dental issues visible in the image.

Output format: [Short diagnostic report, e.g., "Cavity present on tooth 16. Mild bone loss near molars. Tooth 27 is missing."]
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[prompt, uploaded_file]
)

print(response.text)