import re
import html

import streamlit as st
from chatbot import ask_idbi_chatbot


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="IDBI Bank AI Assistant",
    page_icon="assets/IDBI_LOGO.png",
    layout="centered"
)


# ==========================================
# RESPONSE FORMATTING
# ==========================================

def format_response(response):
    """Clean HTML artifacts while preserving Markdown links."""

    if not response:
        return ""

    response = html.unescape(str(response))

    response = re.sub(r"(?i)<br\s*/?>", "\n", response)
    response = re.sub(r"(?i)</?p\s*>", "\n", response)

    # Remove HTML tags without removing Markdown URL syntax.
    response = re.sub(
        r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^>]*)?>",
        "",
        response
    )

    response = re.sub(r"\n{3,}", "\n\n", response)

    return response.strip()


# ==========================================
# SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.image("assets/IDBI_LOGO.png", width=180)
    st.title("IDBI Bank AI Assistant")
    st.write("Your AI-powered banking information assistant.")

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("Powered by Groq + RAG")
    st.caption("Information is retrieved from available IDBI Bank documents.")


# ==========================================
# MAIN HEADER
# ==========================================

col1, col2 = st.columns([1, 5], vertical_alignment="center")

with col1:
    st.image("assets/IDBI_LOGO.png", width=100)

with col2:
    st.title("IDBI Bank AI Assistant")

st.write(
    "Hello! 👋 Ask me anything about IDBI Bank's "
    "accounts, deposits, loans, cards, charges and banking services."
)

st.caption(
    "You can ask questions in English, Hindi, Tamil "
    "and other supported Indian languages."
)

st.divider()


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.chat_history:
    if message["role"] == "Customer":
        with st.chat_message("user"):
            st.markdown(message["content"])

    elif message["role"] == "IDBI Assistant":
        with st.chat_message("assistant"):
            st.markdown(format_response(message["content"]))


# ==========================================
# CHAT INPUT
# ==========================================

user_question = st.chat_input("Ask your banking question...")


# ==========================================
# PROCESS USER QUESTION
# ==========================================

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Finding information..."):
            try:
                answer = ask_idbi_chatbot(
                    user_question,
                    st.session_state.chat_history
                )

                if answer:
                    st.markdown(format_response(answer))
                else:
                    st.error(
                        "Sorry! The banking assistant is temporarily "
                        "unavailable. Please try again."
                    )

            except Exception as e:
                st.error(
                    "Something went wrong while processing your question."
                )
                st.exception(e)
