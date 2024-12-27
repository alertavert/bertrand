

Here’s a high-level guide to building a Retrieval-Augmented Generation (RAG) pipeline locally using the tools you mentioned:

# Steps to Build the RAG Pipeline

## 1. Ingest and Chunk PDF Files

Use unstructured.io to read and process PDF files. The library can extract content and metadata from PDFs.

```python
from unstructured.partition.pdf import partition_pdf

def chunk_pdf(file_path, chunk_size=512):
    """Ingest a PDF file, extract text, and chunk it."""
    elements = partition_pdf(filename=file_path)
    text = " ".join([element.text for element in elements if element.text])
    
    # Split text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks
```

## 2. Generate Embeddings

Use an embedding model compatible with your LLM (e.g., models from sentence-transformers, OpenAI, or others supported by Pinecone). If Ollama supports embedding generation, use its API.
```python
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for text chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings
```

## 3. Store in a Vector Database

You can use Pinecone for storing the embeddings. If you prefer a fully local solution, use alternatives like FAISS.

Pinecone Example:
```python title=Pinecone
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")  # Replace with your Pinecone key and environment
index_name = "rag-vector-store"

# Create a new index (if not exists)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)

index = pinecone.Index(index_name)

def store_embeddings(chunks, embeddings):
    """Store text chunks and their embeddings in Pinecone."""
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        index.upsert([(f"chunk-{i}", embedding, {"text": chunk})])
```


FAISS Example:

```python title=FAISS
import faiss
import numpy as np

def create_faiss_index(embeddings):
    """Create a FAISS index for the embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index
```

## 4. Query the Vector Store

Retrieve the most relevant chunks using cosine similarity.

Pinecone Query Example:
```python title="Query Pinecone"
def query_pinecone(query_text, model, top_k=5):
    """Query Pinecone for the most relevant chunks."""
    query_embedding = model.encode([query_text])[0]
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return results
```
FAISS Query Example:
```python title="Query FAISS"
def query_faiss(query_text, model, faiss_index, chunks, top_k=5):
    """Query FAISS for the most relevant chunks."""
    query_embedding = model.encode([query_text])[0]
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)
    results = [(chunks[i], distances[0][i]) for i in indices[0]]
    return results
```

## 5. Integrate with Local LLM (Ollama)

You can use the retrieved chunks to provide context for a query to your locally running LLM with Ollama.
```python title=ollama
import subprocess

def query_ollama(prompt, context):
    """Query Ollama with a contextualized prompt."""
    full_prompt = f"{context}\n\n{prompt}"
    process = subprocess.run(["ollama", "run", full_prompt], capture_output=True, text=True)
    return process.stdout
```

### Summary of the Tech Stack
1.	PDF Processing: `unstructured.io` for ingestion and chunking.
2.	Embedding Generation: `sentence-transformers` or Ollama (if supported).
3.	Vector Store: `Pinecone` for cloud-based, `FAISS` for local.
4.	Query Execution: Retrieve relevant context from the vector store.
5.	LLM Integration: `Ollama` for generating responses.

# Vector DB

## Qdrant

Qdrant is a vector database designed for real-time, high-performance similarity search. It offers a simple API and runs efficiently on local machines.
	•	Key Features:
	•	Lightweight and fast.
	•	REST API with gRPC support.
	•	Compatible with embeddings from most AI frameworks.
	•	Installation: Use Docker or pip for Python bindings.

```shell
docker run -p 6333:6333 qdrant/qdrant
```

[Qdrant docs](https://qdrant.tech/documentation/)

**Why Qdrant?**
	•	Ease of Use: Simple setup with REST and Python APIs.
	•	Local Storage: Supports persistent storage, so you don’t need to worry about data loss.
	•	Low Complexity: Lightweight, no complex dependencies.
	•	Incremental Updates: Adding new embeddings (from additional PDFs) is straightforward.
	•	Performance: Optimized for local environments and fast similarity searches.

**Recommended Setup:**
	•	Run Qdrant locally via Docker or as a standalone binary.
	•	Use Qdrant’s Python client for embedding storage and querying.

# System configuration

## System Architecture Overview

*1.	PDF Ingestion and Chunking:*
	•	Use unstructured.io to extract text and split it into manageable chunks.
*2.	Embedding Generation:*
	•	Use a suitable embedding model (e.g., from sentence-transformers).
*3.	Vector Store (Qdrant):*
	•	Store embeddings and associated metadata in a local Qdrant instance (running via Docker).
*4.	Query Execution:*
	•	Retrieve relevant chunks from Qdrant based on user input.
*5.	LLM Querying (Ollama):*
	•	Use Ollama to generate context-aware answers using retrieved chunks.
*6.	Frontend (Streamlit):*
	•	Provide a user interface for querying the knowledge base and displaying results.

## Implementation

```shell title='Run Qdrant'
docker run -d -p 6333:6333 qdrant/qdrant
```

Created the `bertie` virtualenv (see `requirements.txt`)



# References

- [Unstructured.io Docs](https://unstructured.io/)
- [Pinecone Documentation](https://docs.pinecone.io/guides/get-started/overview)
- [faiss: A library for efficient similarity search and clustering of dense vectors.](https://github.com/facebookresearch/faiss)
- [SentenceTransformers Documentation](https://www.sbert.net/)

