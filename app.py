# Streamlit Web Application
import logging
import sys
from pathlib import Path

import streamlit as st
from streamlit import session_state
from streamlit_option_menu import option_menu

import embeddings
from kb.conversation import Conversation, Role
from kb.model import KBQueryRunner
from utils import setup_logger, get_logger

# Configure Streamlit page
st.set_page_config(
    page_title="Knowledge Base",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
        .stProgress .st-bo {
            background-color: #00a0dc;
        }
        .success-text {
            color: #00c853;
        }
        .warning-text {
            color: #ffd700;
        }
        .error-text {
            color: #ff5252;
        }
        .st-emotion-cache-1v0mbdj.e115fcil1 {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize logger
@st.cache_resource
def init_logger() -> logging.Logger:
    log_level = logging.DEBUG if len(sys.argv) == 2 and sys.argv[1] == "debug" else logging.INFO
    setup_logger(log_level)
    return get_logger()

logger = init_logger()
logger.debug("Logger initialized")

@st.cache_resource
def init_store() -> embeddings.EmbeddingsStore:
    """Initialize the embeddings store."""
    logger.info("Initializing the embeddings store")
    return embeddings.EmbeddingsStore()

@st.cache_resource
def get_generator() -> embeddings.EmbeddingsGenerator:
    """A cached instance of the embeddings generator."""
    logger.info("Initializing the embeddings generator")
    return embeddings.EmbeddingsGenerator()

def render_conversation():
    for message in st.session_state.conversation.messages:
        for role, text in message.items():
            with st.chat_message(role):
                st.write(text)

def clear_conversation():
    if "conversation" in st.session_state:
        st.session_state.conversation.reset()


def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation()

    header_cols = st.columns([1, 2])
    with header_cols[0]:
        st.image(Path("static/mastersIcon.png"), width=120)
    with header_cols[1]:
        st.title("Bertrand Knowledge Base")
    # # Sidebar navigation
    with st.sidebar:
        st.image(
            Path("static/logo.png"),
            width=50,
        )
        st.title("Knowledge Base")
        selected = option_menu(
            menu_title="Navigation",
            options=["Upload File", "Query the KB", "About"],
            icons=["cloud-upload", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Upload File":
        st.header("📄 Document Upload")
        st.write("Upload a document to get AI-powered insights.")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF tech paper to analyze",
        )

        if uploaded_file:
            try:
                with st.spinner("Extracting data into our KB..."):
                    gen = get_generator()
                    vectors, chunks = gen.from_file(uploaded_file)
                    store = init_store()
                    store.store_embeddings(chunks, vectors)
                st.info("Tech paper uploaded successfully! Processing...")
            except Exception as e:
                st.error(f"Error handling file upload: {str(e)}")
                logger.error(f"Upload error: {str(e)}", exc_info=True)

    elif selected == "Query the KB":
        st.header("🔍 Query the Knowledge Base")
        st.write("Ask a technical question to get AI-powered answers.")
        if st.button("Clear conversation"):
            clear_conversation()
        prompt = st.chat_input("Enter your question:")
        if prompt:
            session_state.conversation.add_message(Role.USER, prompt)
            try:
                with st.spinner("Querying the KB..."):
                    runner = KBQueryRunner(generator=get_generator(), store=init_store())
                    response = runner.query(prompt)
                    st.session_state.conversation.add_message(Role.ASSISTANT, response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Query error: {str(e)}", exc_info=True)
            render_conversation()
    elif selected == "About":
        st.header("About Bertrand Knowledge Base")
        st.write(
            """
        Welcome to Bertrand, a cutting-edge local KB system powered by:

        - **Ollama (llama3.2)**: Advanced language model for natural language processing
        - **Qdrant**: Vector DB store for text embeddings
        - **Streamlit**: Modern web interface for easy interaction

        Our system uses specialized AI agents to:
        1. 📄 Extract information from tech papers
        2. 🔍 Analyze the content
        3. 🎯 Enrich your query to the LLM with the most relevant information
        4. 👥 Query the LLM with enriched data
        5. 💡 Provide answers to detailed technical questions

        Upload your library of tech papers to experience AI-powered Knowledge!
        """
        )


if __name__ == "__main__":
    main()
