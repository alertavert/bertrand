# System constants
#
# These should only be rarely changed, if ever.

# Retrieval embedding model and constants
EMBED_MODEL: str = "all-MiniLM-L6-v2"
EMBED_DIM: int = 384
CHUNK_SIZE: int = 512

# Qdrant client constants
QDRANT_URL: str = "http://localhost:6333"
QDRANT_COLLECTION: str = "tech_articles"
