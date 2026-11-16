
import streamlit as st
from chatbot import ask_idbi_chatbot


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="IDBI Bank AI Assistant",
    page_icon="C:\\Users\\tharu\\Documents\\idbi bank logo.png",
    layout="centered"
)


# ==========================================
# SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    col1, col2 = st.columns([1, 7])

with col1:
    st.image("assets/idbi_logo.png", width=240)

with col2:
    st.title("IDBI Bank AI Assistant")
    st.write(
        "Your AI-powered banking information assistant."
    )

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):

        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption("Powered by Groq + RAG")

    st.caption(
        "Information is retrieved from available "
        "IDBI Bank documents."
    )



# ==========================================
# MAIN HEADER
# ==========================================

col1, col2 = st.columns([2, 6], vertical_alignment="center")

with col1:
    st.image("assets/idbi_logo.png", width=160)

with col2:
    st.title("IDBI Bank AI Assistant")

st.write(
    "Hello! 👋 Ask me anything about IDBI Bank's "
    "accounts, deposits, loans, cards, charges "
    "and banking services."
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
            st.markdown(message["content"])


# ==========================================
# CHAT INPUT
# ==========================================

user_question = st.chat_input(
    "Ask your banking question..."
)


# ==========================================
# PROCESS USER QUESTION
# ==========================================

if user_question:

    # Display customer question

    with st.chat_message("user"):
        st.markdown(user_question)


    # Generate chatbot response

    with st.chat_message("assistant"):

        with st.spinner("Finding information..."):

            try:

                answer = ask_idbi_chatbot(
                    user_question,
                    st.session_state.chat_history
                )

                if answer:

                    st.markdown(answer)

                else:

                    st.error(
                        "Sorry! The banking assistant is "
                        "temporarily unavailable. Please try again."
                    )

            except Exception as e:

                st.error(
                    "Something went wrong while processing "
                    "your question."
                )

                st.exception(e)
