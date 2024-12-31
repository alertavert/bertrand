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


class EmbeddingsGenerator(object):
    """Generates embeddings from strings or entire files."""

    def __init__(
            self,
            chunk_size: int = CHUNK_SIZE,
            embedding_model: str = EMBED_MODEL
    ) -> None:
        self.chunk_size: int = chunk_size
        self.model: SentenceTransformer = SentenceTransformer(embedding_model)

    def _chunk_text(self, text: List[str]) -> List[str]:
        """Repartition text into chunks of length `self.chunk_size`.

        :param text: A list of text strings (possibly, just one) of various lengths.
        :return: List of text chunks of length `self.chunk_size`.
        """
        text: str = " ".join(text)
        chunks: List[str] = [text[i:i + self.chunk_size]
                             for i in range(0, len(text), self.chunk_size)]
        return chunks

    def _generate_embeddings(self, chunks: List[str]) -> numpy.ndarray:
        """Generate embeddings for text chunks.

        :param chunks: List of text chunks as strings of length `self.chunk_size`.
        :return: Numpy array of embeddings, one for each chunk.
        """
        embeddings = self.model.encode(chunks)
        return embeddings

    def from_file(self, file_path: str) -> Tuple[numpy.ndarray, List[str]]:
        """Generate embeddings from a PDF file.

        :param file_path: Path to the PDF file.
        :return: Tuple of embeddings, one vector per chunk, and the respective text chunks, a list of strings.
        """
        elements: List[Element] = partition_pdf(filename=file_path)
        text: List[str] = [element.text for element in elements if element.text]
        return self.from_text(text)

    def from_text(self, text: str | List[str]) -> Tuple[numpy.ndarray, List[str]]:
        """Generate embeddings from a text.

        The text can be a single string, or a list of strings, which will be joined together.
        In either case, the text will be chunked into pieces of length `self.chunk_size` and the
        respective embeddings will be generated.

        :param text: List of text strings, each must be of length `self.chunk_size`.
        :return: Respective embeddings, one vector per element in the list, in order.
        """
        if isinstance(text, str):
            text = [text]
        chunks = self._chunk_text(text)
        embeddings = self._generate_embeddings(chunks)
        return embeddings, chunks

    def query_vect(self, query: str) -> numpy.ndarray:
        """Generate embeddings from a query string.

        Helper method to simplify the process of generating embeddings from a query string to
        be used in `EmbeddingsStore.search`.
        The `query` string will be converted into a single embedding vector, and should be
        of length `self.chunk_size` or less (if longer, the additional generated embeddings
        will be ignored).

        :param query: The query string.
        :return: The respective embedding vector.
        """
        qv, _ = self.from_text(query)
        return qv[0]

# TODO: Use Pydantic to validate the input data
class EmbeddingsStore(object):
    """Manages embeddings stored in Qdrant."""

    def __init__(
            self,
            url: str = QDRANT_URL,
            collection: str = QDRANT_COLLECTION,
            embeddings_dim: int = EMBED_DIM,
    ) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._embeddings_dim = embeddings_dim
        self._init_collection()

    @property
    def dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        return self._embeddings_dim

    @property
    def collection(self) -> str:
        """Return the name of the collection."""
        return self._collection

    def _init_collection(self) -> None:
        """Initialize the collection in Qdrant."""
        if not self._client.collection_exists(collection_name=self._collection):
            vector_configs = VectorParams(
                size=EMBED_DIM,  # Dimensionality of the embedding vectors
                distance=Distance.COSINE  # Distance metric to use for similarity search
            )
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=vector_configs,
            )

    def store_embeddings(self, chunks, embeddings) -> None:
        """Store chunks and embeddings in Qdrant."""
        points = [
            PointStruct(id=i, vector=embedding, payload={"text": chunk})
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, embeddings: numpy.ndarray, k: int = TOP_K, threshold: float = MIN_THRESHOLD) -> List[Tuple[str, float]]:
        """Search for the most similar embeddings to the query.

        This uses Qdrant's `search` method to find the `k` most similar embeddings to the query, and is
        thus limited to using a single embedding vector as the query.

        :param embeddings: The query string encoded as an embedding (use `EmbeddingsGenerator.query_vect`).
        :param k: The number of most similar embeddings to return.
        :return: List of tuples, each containing the text and the similarity score.
        """
        results = self._client.search(
            collection_name=self._collection,
            query_vector=embeddings,
            limit=k,
        )
        return [(result.payload["text"], result.score) for result in results]
