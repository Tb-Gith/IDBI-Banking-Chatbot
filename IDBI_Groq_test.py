
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "openai/gpt-oss-20b"

print("Groq chatbot connected successfully!")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        print("\nGroq:", response.choices[0].message.content)

    except Exception as e:
        print("Error:", e)
