from google import genai
from dotenv import load_dotenv
import os

# Explicitly load .env from the project folder
load_dotenv(dotenv_path=".env", override=True)

api_key = os.getenv("GEMINI_API_KEY")

print("API key detected:", bool(api_key))

if not api_key:
    raise ValueError("Gemini API key not found!")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send a test message
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Introduce yourself as an AI banking assistant for IDBI Bank."
)

print("\n========== GEMINI RESPONSE ==========")
print(response.text)

client.close()
