# Streamlit Web Application
import logging
import sys

import streamlit as st
from streamlit_option_menu import option_menu

import embeddings
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
logger.debug("Logger initialized.")

@st.cache_resource
def init_store() -> embeddings.EmbeddingsStore:
    """Initialize the embeddings store."""
    logger.info("Initializing the embeddings store.")
    return embeddings.EmbeddingsStore()


@st.cache_resource
def get_generator() -> embeddings.EmbeddingsGenerator:
    """A cached instance of the embeddings generator."""
    logger.info("Initializing the embeddings generator.")
    return embeddings.EmbeddingsGenerator()


def main():
    st.title("🤖 Bertrand Knowledge Base")
    st.write("Hello, World!")
    # # Sidebar navigation
    with st.sidebar:
        st.image(
            "https://img.icons8.com/resume",
            width=50,
        )
        st.title("Knowledge Base")
        selected = option_menu(
            menu_title="Navigation",
            options=["Upload File", "About"],
            icons=["cloud-upload", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

        if selected == "Upload File":
            st.header("📄 Document Analysis")
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
