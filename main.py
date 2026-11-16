import time
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer
import os
import pymupdf 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# STEP 1: Load all PDF documents
documents_folder = "documents"

pages = []
processed_files = []

for filename in os.listdir(documents_folder):

    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(documents_folder, filename)

        with pymupdf.open(pdf_path) as document:

            processed_files.append(filename)

            for page_number, page in enumerate(document):

                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "source": filename,
                        "page": page_number + 1,
                        "text": text
                    })

print("========== PDF LOADING RESULTS ==========")
print("PDFs processed:", len(processed_files))
print("Pages extracted:", len(pages))


# STEP 2: Split extracted text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = []
chunk_metadata = []

for page in pages:

    page_chunks = splitter.split_text(page["text"])

    for chunk in page_chunks:

        chunks.append(chunk)

        chunk_metadata.append({
            "source": page["source"],
            "page": page["page"]
        })


print("\n========== CHUNKING RESULTS ==========")
print("Total chunks generated:", len(chunks))

if chunks:
    print("\nFirst chunk:")
    print(chunks[0])

    print("\nFirst chunk metadata:")
    print(chunk_metadata[0])
else:
    print("No text was extracted. Please check your PDF files.")

# STEP 3: Generate embeddings

print("\n========== EMBEDDING GENERATION ==========")

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

# STEP 4: Create ChromaDB database

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Rebuild the collection when running the ingestion script again
try:
    chroma_client.delete_collection(name="idbi_documents")
except Exception:
    pass

collection = chroma_client.create_collection(name="idbi_documents")

# Generate embeddings and store them in batches
for start in range(0, len(chunks), 100):

    batch_chunks = chunks[start:start + 100]
    batch_metadata = chunk_metadata[start:start + 100]

    passage_chunks = [
    f"passage: {chunk}" for chunk in batch_chunks
]

    embeddings = embedding_model.encode(
    passage_chunks,
    show_progress_bar=False
    ).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(start, start + len(batch_chunks))
    ]

    collection.add(
        ids=ids,
        documents=batch_chunks,
        metadatas=batch_metadata,
        embeddings=embeddings
    )

print("Embeddings generated successfully!")
print("Total chunks stored:", collection.count())
print("Database saved in: chroma_db")

# STEP 5: Test semantic search

print("\n========== SEMANTIC SEARCH ==========")

query_text = "What is the procedure to open a savings account?"

# Convert the question into an embedding
query_embedding = embedding_model.encode(
    [f"query: {query_text}"]
).tolist()

# Search ChromaDB for the 3 most relevant chunks
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

# Display the retrieved information
for i in range(len(results["documents"][0])):

    print(f"\nResult {i + 1}")
    print("-----------------------------")

    print("Relevant Information:")
    print(results["documents"][0][i])

    print("\nSource Document:")
    print(results["metadatas"][0][i]["source"])

    print("Page Number:")
    print(results["metadatas"][0][i]["page"])

    print("Distance:")
    print(results["distances"][0][i])

# STEP 6: Connect to Groq  

load_dotenv()


llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "openai/gpt-oss-20b"


print("\nGroq connected successfully!")
print("Model:", MODEL_NAME)


def generate_llm_response(prompt, max_tokens=1000):
    """Send a prompt to Groq, with a small retry limit."""

    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    max_tokens=1000,
)

            message = response.choices[0].message
            answer = message.content

            if isinstance(answer, str) and answer.strip():
                return answer.strip()

            # Diagnostic information without printing private reasoning.
            print(
                "The model returned no visible answer. "
                f"Finish reason: {response.choices[0].finish_reason}"
            )
            print("Response usage:", response.usage)

        except Exception as e:
            print(f"groq API error: {e}")
        if attempt < 2:
            print("Retrying in 3 seconds...")
            time.sleep(3)

    return None



# STEP 7: IDBI Banking Assistant

# Store previous questions and answers
chat_history = []


def ask_idbi_chatbot(question):

    # 1. Convert follow-up question into a standalone question
    if chat_history:

        history_text = ""

        for message in chat_history[-10:]:
            history_text += f"{message['role']}: {message['content']}\n"

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

        # If rewriting fails, use the customer's original question.
        search_question = rewritten_question or question

    else:
        search_question = question

    # 2. Generate embedding for the standalone question
    query_embedding = embedding_model.encode(
    [f"query: {search_question}"]
).tolist()

    # 3. Retrieve relevant information from ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=4
    )

    # 4. Prepare retrieved information with source references
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

    # 5. Prepare previous conversation
    history_text = ""

    for message in chat_history[-10:]:
        history_text += (
            f"{message['role']}: {message['content']}\n"
        )

    # 6. Send context and conversation history to the Groq model
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

    
    # Generate the final answer using Groq
    answer = generate_llm_response(prompt, max_tokens=1000)

    if answer is None:
        print("Sorry! The banking assistant is temporarily unavailable.")
        return



    # 7. Save the latest question and answer in memory
    chat_history.append({
        "role": "Customer",
        "content": question
    })

    chat_history.append({
        "role": "IDBI Assistant",
        "content": answer
    })

    # Keep only the latest 10 messages
    if len(chat_history) > 10:
        del chat_history[:-10]

    # 8. Display answer
    print("\n========== IDBI CHATBOT ==========")
    print(answer)


# STEP 8: Start the chatbot

print("\nIDBI BANK AI ASSISTANT IS READY!")
print("Type 'reset' to start a new conversation.")
print("Type 'exit' to close the chatbot.")

while True:

    question = input("\nYou: ").strip()

    if question.lower() in [
        "exit", "quit", "bye", "abort", "stop", "end"
    ]:
        print("Chatbot: Thank you for using IDBI AI Assistant!")
        break

    if question.lower() == "reset":
        chat_history.clear()
        print("Chatbot: Conversation history cleared!")
        continue

    if not question:
        continue

    ask_idbi_chatbot(question)
