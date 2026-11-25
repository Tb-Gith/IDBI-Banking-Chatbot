
import time
import os
import re
import html
from pathlib import Path

import chromadb
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
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GROQ_API_KEY not found. Check your .env file "
            "or Streamlit Cloud secrets."
        )

llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

MODEL_NAME = "openai/gpt-oss-20b"


# ==========================================
# STEP 4: RESPONSE CLEANING
# ==========================================

def clean_response(answer):

    if not answer:
        return None

    answer = html.unescape(str(answer))

    # Convert HTML line breaks into normal line breaks.
    answer = re.sub(
        r"(?i)<br\s*/?>",
        "\n",
        answer
    )

    answer = re.sub(
        r"(?i)</p\s*>",
        "\n\n",
        answer
    )

    answer = re.sub(
        r"(?i)<p\s*>",
        "",
        answer
    )

    # Remove remaining HTML tags.
    answer = re.sub(
        r"<[^>]+>",
        "",
        answer
    )

    # Remove excessive spaces and blank lines.
    answer = re.sub(
        r"[ \t]+\n",
        "\n",
        answer
    )

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    return answer.strip()


# ==========================================
# STEP 5: GENERATE GROQ RESPONSE
# ==========================================

def generate_llm_response(prompt, max_tokens=1000):

    for attempt in range(3):

        try:

            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful, accurate and "
                            "multilingual IDBI Bank customer "
                            "assistance assistant."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content

            if isinstance(answer, str) and answer.strip():
                return clean_response(answer)

        except Exception as e:

            print(f"Groq API error: {e}")

        if attempt < 2:
            time.sleep(3)

    return None


# ==========================================
# STEP 6: CUSTOMER MESSAGE DETECTION
# ==========================================

def normalize_message(message):

    message = message.lower().strip()

    message = re.sub(
        r"[^\w\s]",
        "",
        message
    )

    return re.sub(r"\s+", " ", message).strip()


def is_greeting(message):

    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "hii",
        "hiii",
        "vanakkam",
        "வணக்கம்",
        "नमस्ते",
        "नमस्कार"
    }

    return normalize_message(message) in greetings


def is_help_request(message):

    help_commands = {
        "help",
        "help me",
        "show help",
        "show available services",
        "available services",
        "list services",
        "list all services",
        "what can you do",
        "what can you help me with",
        "how can you help me",
        "what services do you provide",
        "what services are available",
        "what are your services",
        "your services",
        "your capabilities",
        "what do you offer",
        "என்ன செய்ய முடியும்",
        "உதவி",
        "मदद",
        "आप क्या कर सकते हैं"
    }

    return normalize_message(message) in help_commands


def is_thank_you(message):

    thank_you_messages = {
        "thanks",
        "thank you",
        "thankyou",
        "thanks a lot",
        "thank you so much",
        "much appreciated",
        "நன்றி",
        "धन्यवाद"
    }

    return normalize_message(message) in thank_you_messages


def is_goodbye(message):

    goodbye_messages = {
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "take care",
        "bye bye",
        "போயிட்டு வரேன்",
        "अलविदा"
    }

    return normalize_message(message) in goodbye_messages


# ==========================================
# STEP 7: AVAILABLE SERVICES RESPONSE
# ==========================================

def get_services_response():

    return """
Welcome to **IDBI Bank AI Assistant!** 👋

I can help you understand IDBI Bank's banking products,
services and procedures using the available bank documents.

### Here's what you can ask me about:

**1. Accounts**
- Account-related information
- Account features and facilities

**2. Deposits**
- Deposit products
- Deposit-related terms and conditions

**3. Loans**
- Loan-related information
- Eligibility and procedures, where documented

**4. Cards**
- Banking card features and related information

**5. Charges**
- Banking charges and applicable fees
- Related terms and conditions

**6. Banking Procedures**
- Information about banking processes
- Step-by-step guidance based on available documents

### How to get started

You can ask questions naturally, for example:

- "What types of accounts are available?"
- "Tell me about IDBI deposit schemes."
- "What information is available about loans?"
- "Explain the charges associated with banking services."

You can also ask questions in **English, Hindi, Tamil,
Telugu, Kannada, Malayalam, Bengali and Marathi**.

**A quick note:** I provide banking information and guidance.
I cannot access personal accounts, check balances, transfer money
or perform actual banking transactions.

What would you like to know today?
"""


# ==========================================
# STEP 8: SAVE CONVERSATION
# ==========================================

def build_sources_section(metadata_list):
    """Build clickable source links using URLs stored in ChromaDB metadata."""
    unique_sources = {}

    for metadata in metadata_list:
        source = metadata.get("source", "IDBI Bank document")
        page = metadata.get("page", "Unknown")
        source_url = metadata.get("source_url")

        # Do not fabricate a URL when old database metadata has no source_url.
        if not source_url or not str(source_url).startswith("https://www.idbi.bank.in/"):
            continue

        key = (source, source_url)
        unique_sources.setdefault(key, set()).add(str(page))

    if not unique_sources:
        return ""

    lines = ["\n\n---\n\n### Sources"]
    for (source, source_url), pages in unique_sources.items():
        page_text = ", ".join(sorted(pages, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)))
        lines.append(f"- [{source}]({source_url}) — Page(s): {page_text}")

    return "\n".join(lines)


def save_conversation(question, answer, chat_history):

    chat_history.append({
        "role": "Customer",
        "content": question
    })

    chat_history.append({
        "role": "IDBI Assistant",
        "content": answer
    })

    # Retain recent conversation messages.
    if len(chat_history) > 10:
        del chat_history[:-10]


# ==========================================
# STEP 9: MAIN IDBI CHATBOT
# ==========================================

def ask_idbi_chatbot(question, chat_history):

    question = question.strip()

    if not question:
        return None


    # --------------------------------------
    # 1. Handle greetings without RAG
    # --------------------------------------

    if is_greeting(question):

        answer = get_services_response()

        save_conversation(
            question,
            answer,
            chat_history
        )

        return answer


    # --------------------------------------
    # 2. Handle help commands without RAG
    # --------------------------------------

    if is_help_request(question):

        answer = get_services_response()

        save_conversation(
            question,
            answer,
            chat_history
        )

        return answer


    # --------------------------------------
    # 3. Handle thank-you messages
    # --------------------------------------

    if is_thank_you(question):

        answer = (
            "You're welcome! 😊\n\n"
            "I'm happy to help. Feel free to ask me "
            "anything about IDBI Bank's accounts, deposits, "
            "loans, cards, charges or banking procedures."
        )

        save_conversation(
            question,
            answer,
            chat_history
        )

        return answer


    # --------------------------------------
    # 4. Handle goodbye messages
    # --------------------------------------

    if is_goodbye(question):

        answer = (
            "Thank you for using IDBI Bank AI Assistant! "
            "Have a great day. 😊"
        )

        save_conversation(
            question,
            answer,
            chat_history
        )

        return answer


    # --------------------------------------
    # 5. Rewrite follow-up questions
    # --------------------------------------

    if chat_history:

        history_text = ""

        for message in chat_history[-10:]:

            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        rewrite_prompt = f"""
You are a multilingual banking question-rewriting assistant.

Rewrite the customer's latest question as a complete,
standalone question using the conversation history.

Rules:
1. Do not answer the question.
2. Return only the rewritten question.
3. Preserve the language of the latest customer question.
4. Do not translate it into English.
5. Preserve important banking terms and details.
6. If the question is already complete, return it unchanged.

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


    # --------------------------------------
    # 6. Generate question embedding
    # --------------------------------------

    query_embedding = embedding_model.encode(
        [f"query: {search_question}"]
    ).tolist()


    # --------------------------------------
    # 7. Retrieve relevant banking information
    # --------------------------------------

    total_chunks = collection.count()

    if total_chunks == 0:

        answer = (
            "I'm sorry, but the IDBI Bank documents are "
            "currently unavailable. Please try again later."
        )

        save_conversation(
            question,
            answer,
            chat_history
        )

        return answer

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(4, total_chunks)
    )


    # --------------------------------------
    # 8. Prepare retrieved context and sources
    # --------------------------------------

    context_parts = []

    retrieved_documents = results.get(
        "documents", [[]]
    )[0]

    retrieved_metadata = results.get(
        "metadatas", [[]]
    )[0]

    for document, metadata in zip(
        retrieved_documents,
        retrieved_metadata
    ):

        source = metadata.get(
            "source",
            "Unknown document"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        source_url = metadata.get("source_url", "")

        context_parts.append(
            f"""
Information:
{document}

Source document: {source}
Page number: {page}
Official source URL: {source_url}
"""
        )

    context = "\n".join(context_parts)

    if not context.strip():

        answer = (
            "I couldn't find relevant information in the "
            "available IDBI Bank documents for your question. "
            "Could you try asking in a different way?"
        )

        save_conversation(
            question,
            answer,
            chat_history
        )

        return answer


    # --------------------------------------
    # 9. Prepare conversation history
    # --------------------------------------

    history_text = ""

    for message in chat_history[-10:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    # --------------------------------------
    # 10. Create improved response prompt
    # --------------------------------------

    prompt = f"""
You are the IDBI Bank AI Customer Assistance Chatbot.

Your job is to help customers understand IDBI Bank's
products, services and banking procedures.

Use the retrieved IDBI Bank information to answer
the customer's latest question.

IMPORTANT ANSWERING RULES:

1. Answer using ONLY the retrieved banking information.
2. Never invent banking information, rates, charges,
   eligibility criteria, limits, timelines or procedures.
3. If the requested information is not available,
   politely explain that it could not be found in
   the available IDBI Bank documents.
4. Never claim to access personal accounts, balances,
   transactions or perform banking operations.
5. Never request passwords, PINs or OTPs.
6. Answer the actual question directly.
7. Do not provide unrelated information.
8. Consider conversation history when understanding
   follow-up questions.
9. Respond in the same language as the latest question.
10. Support English, Hindi, Tamil, Telugu, Kannada,
    Malayalam, Bengali and Marathi.
11. Do not translate into English unless requested.
12. Preserve source document names and page numbers
    exactly as provided.
13. Do not generate URLs or a Sources section yourself.
    Clickable source links will be added by the application
    using the retrieved document metadata.
14. Never invent source references.

RESPONSE STYLE AND FORMATTING:

1. Use clean Markdown only. Never generate HTML tags.
2. Keep answers concise but sufficiently informative.
3. Use short paragraphs.
4. Use **bold** for important terms and key information.
5. Use bullet points when listing multiple items.
6. Use numbered lists for step-by-step procedures.
7. Use clear headings when an answer has multiple sections.
8. Use tables only when comparing multiple products
   or features where a table improves readability.
9. Avoid repeating information.
10. Avoid unnecessary greetings and generic introductions.
11. Do not repeat the customer's question.
12. Explain technical banking terms in simple language.
13. If the question requires several steps, explain
    them in the correct order.
14. If information is incomplete, clearly mention
    what is available and what is missing.

CONVERSATION HISTORY:
{history_text}

RETRIEVED IDBI BANK INFORMATION:
{context}

LATEST CUSTOMER QUESTION:
{question}

Write a clear, natural and helpful customer-facing answer:
"""


    # --------------------------------------
    # 11. Generate final answer
    # --------------------------------------

    answer = generate_llm_response(
        prompt,
        max_tokens=1000
    )

    if answer is None:
        return (
            "Sorry! I'm temporarily unable to process "
            "your request. Please try again in a moment."
        )


    # --------------------------------------
    # 12. Append clickable source links
    # --------------------------------------

    sources_section = build_sources_section(retrieved_metadata)
    final_answer = answer + sources_section


    # --------------------------------------
    # 13. Save conversation
    # --------------------------------------

    save_conversation(
        question,
        final_answer,
        chat_history
    )

    return final_answer
