# Embeddings module init file

from constants import *
import numpy
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf


class FileIngestor(object):
    """Ingestor for PDF files."""

    def __init__(
            self,
            file_path: str,
            chunk_size: int = CHUNK_SIZE,
            embedding_model: str = EMBED_MODEL
    ) -> None:
        self.path = file_path
        self.chunk_size = chunk_size
        self.model = embedding_model

    def _chunk_pdf(self) -> List[str]:
        """Ingest a PDF, extract text, and chunk it."""
        elements: List[Element] = partition_pdf(filename=self.path)
        text: str = " ".join([element.text for element in elements if element.text])
        chunks: List[str] = [text[i:i + self.chunk_size]
                             for i in range(0, len(text), self.chunk_size)]
        return chunks

    def _generate_embeddings(self, chunks) -> numpy.ndarray:
        """Generate embeddings for text chunks."""
        model = SentenceTransformer(self.model)
        embeddings = model.encode(chunks)
        return embeddings

    def __call__(self) -> Tuple[numpy.ndarray, List[str]]:
        chunks = self._chunk_pdf()
        embeddings = self._generate_embeddings(chunks)
        return embeddings, chunks


class EmbeddingsClient(object):
    """Client for storing embeddings in Qdrant."""

    def __init__(
            self,
            url: str = QDRANT_URL,
            collection: str = QDRANT_COLLECTION,
            embeddings_dim: int = EMBED_DIM,
    ) -> None:
        self._client = QdrantClient(url=url)
        self.collection = collection
        self.embeddings_dim = embeddings_dim
        self._init_collection()

    def _init_collection(self) -> None:
        """Initialize the collection in Qdrant."""
        if not self._client.collection_exists(collection_name=self.collection):
            vector_configs = VectorParams(
                size=EMBED_DIM,  # Dimensionality of the embedding vectors
                distance=Distance.COSINE  # Distance metric to use for similarity search
            )
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=vector_configs,
            )

    def store_embeddings(self, chunks, embeddings) -> None:
        """Store chunks and embeddings in Qdrant."""
        points = [
            PointStruct(id=i, vector=embedding, payload={"text": chunk})
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self._client.upsert(collection_name=self.collection, points=points)
