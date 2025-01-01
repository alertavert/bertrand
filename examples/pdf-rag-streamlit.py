# demo-ui-streamlit.py

import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "./data/BOI.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

@st.cache_data
def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None

@st.cache_data
def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db

@st.cache_resource
def create_retriever(_vector_db, _llm):
    """Create a multi-query _retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        _vector_db.as_retriever(), _llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

@st.cache_resource
def create_chain(_retriever, _llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": _retriever, "question": RunnablePassthrough()}
            | prompt
            | _llm
            | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain

@st.cache_resource
def init():
    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Load the vector database
    vector_db = load_vector_db()
    if vector_db is None:
        st.error("Failed to load or create the vector database.")
        return

    # Create the _retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain
    chain = create_chain(retriever, llm)
    return chain

@st.cache_data(show_spinner="Thinking about this...")
def get_response(_chain, user_input):
    response = _chain.invoke(input=user_input)
    return response

def main():
    # Initialize the chain
    chain = init()
    st.title("Document Assistant")

    # User input
    user_input = st.text_input("Enter a question:", "")

    if user_input:
        try:
            # Get the response
            response = get_response(chain, user_input)

            st.markdown("## Assistant Response")
            st.write(response)

            st.markdown("## Document Preview")
            option = st.radio("Would you like to see the original document?", ("Yes", "No"))
            if option == "Yes":
                st.markdown("**Original Document:**")
                st.write("This was the document")
            else:
                st.info("You can view the original document by selecting 'Yes' above.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()
