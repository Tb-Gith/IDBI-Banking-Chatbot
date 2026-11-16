
import time
import os
from pathlib import Path

import chromadb
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ==========================================
# STEP 1: PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


# ==========================================
# STEP 2: LOAD EXISTING VECTOR DATABASE
# ==========================================

print("Loading IDBI chatbot resources...")

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

chroma_client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = chroma_client.get_collection(
    name="idbi_documents"
)

print("Existing ChromaDB loaded!")
print("Total stored chunks:", collection.count())


# ==========================================
# STEP 3: CONNECT TO GROQ
# ==========================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    import streamlit as st
    api_key = st.secrets["GROQ_API_KEY"]

llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

MODEL_NAME = "openai/gpt-oss-20b"


# ==========================================
# STEP 4: GENERATE GROQ RESPONSE
# ==========================================

def generate_llm_response(prompt, max_tokens=1000):

    for attempt in range(3):

        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content

            if isinstance(answer, str) and answer.strip():
                return answer.strip()

        except Exception as e:
            print(f"Groq API error: {e}")

        if attempt < 2:
            time.sleep(3)

    return None


# ==========================================
# STEP 5: IDBI BANKING CHATBOT
# ==========================================

def ask_idbi_chatbot(question, chat_history):

    # 1. Rewrite follow-up questions

    if chat_history:

        history_text = ""

        for message in chat_history[-10:]:
            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        rewrite_prompt = f"""
You are a multilingual banking question-rewriting assistant.

Based on the conversation history, rewrite the
customer's latest question into a complete,
standalone banking question.

Rules:
1. Do not answer the question.
2. Return only the rewritten question.
3. Preserve the language of the latest customer question.
4. Do not translate it into English.
5. Preserve important banking terms and details.

CONVERSATION:
{history_text}

LATEST QUESTION:
{question}

STANDALONE QUESTION:
"""

        rewritten_question = generate_llm_response(
            rewrite_prompt,
            max_tokens=200
        )

        search_question = rewritten_question or question

    else:
        search_question = question


    # 2. Generate question embedding

    query_embedding = embedding_model.encode(
        [f"query: {search_question}"]
    ).tolist()


    # 3. Retrieve relevant banking information

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=4
    )


    # 4. Prepare retrieved context with citations

    context_parts = []

    for i in range(len(results["documents"][0])):

        text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        context_parts.append(
            f"""
Information: {text}
Source: {metadata['source']}
Page: {metadata['page']}
"""
        )

    context = "\n".join(context_parts)


    # 5. Prepare conversation history

    history_text = ""

    for message in chat_history[-10:]:
        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    # 6. Create the final prompt

    prompt = f"""
You are an AI Banking Customer Assistance
Chatbot for IDBI Bank.

Answer the customer's latest question using ONLY
the information provided in the retrieved context.

You must also consider the conversation history
to understand follow-up questions.

Rules:
1. Do not invent banking information.
2. Do not use general knowledge to fill information gaps.
3. If the answer is unavailable in the context,
   clearly say that it could not be found in the
   available IDBI documents.
4. Never claim to access customer accounts,
   balances, transactions or perform banking actions.
5. Give clear and simple answers.
6. Mention the source document and page number
   when providing information.
7. Do not invent source references.
8. Understand references to earlier topics using
   the conversation history.
9. Respond in the same language used in the
   customer's latest question.
10. Support English, Hindi, Tamil, Telugu,
    Kannada, Malayalam, Bengali and Marathi.
11. Do not translate the answer into English
    unless the customer requests it.
12. Keep source document names and page
    references accurate and unchanged.

CONVERSATION HISTORY:
{history_text}

RETRIEVED BANKING INFORMATION:
{context}

LATEST CUSTOMER QUESTION:
{question}

ANSWER:
"""


    # 7. Generate the answer

    answer = generate_llm_response(
        prompt,
        max_tokens=1000
    )

    if answer is None:
        return None


    # 8. Update conversation history

    chat_history.append({
        "role": "Customer",
        "content": question
    })

    chat_history.append({
        "role": "IDBI Assistant",
        "content": answer
    })

    if len(chat_history) > 10:
        del chat_history[:-10]


    return answer
