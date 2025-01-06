from typing import List, Tuple

import numpy
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from constants import *
from utils import get_logger


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
        self.log = get_logger()

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

    def store_embeddings(self, chunks: List[str], embeddings: numpy.ndarray) -> None:
        """Store chunks and embeddings in Qdrant."""
        if embeddings.shape[0] == 0:
            self.log.warning("No embeddings to store")
            return
        self.log.debug(f"Storing {embeddings.shape} embeddings")
        points = [
            PointStruct(id=i, vector=embedding, payload={"text": chunk})
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, embeddings: numpy.ndarray, k: int = TOP_K, threshold: float = MIN_THRESHOLD) -> List[Tuple[str, float]]:
        """Search for the most similar embeddings to the query.

        This uses Qdrant's `search` method to find the `k` most similar embeddings to the query, and is
        thus limited to using a single embedding vector as the query.

        :param threshold: The maximum distance threshold for a result to be considered.
        :param embeddings: The query string encoded as an embedding (use `EmbeddingsGenerator.query_vect`).
        :param k: The number of most similar embeddings to return.
        :return: List of tuples, each containing the text and the similarity score.
        """
        results = self._client.search(
            collection_name=self._collection,
            query_vector=embeddings,
            limit=k,
        )
        # TODO: use the threshold to filter the results
        return [(result.payload["text"], result.score) for result in results]
