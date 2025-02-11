# System constants
#
# These should only be rarely changed, if ever.

# Retrieval embedding model and constants
EMBED_MODEL: str = "all-MiniLM-L6-v2"
EMBED_DIM: int = 384
CHUNK_SIZE: int = 1024

# The LLM model to use.
LLM_MODEL: str = "llama3.2"
# Use this to run the DeepSeek R1 model instead.
#  LLM_MODEL: str = "deepseek-r1"

# Qdrant client constants
QDRANT_URL: str = "http://localhost:6333"
QDRANT_COLLECTION: str = "tech_articles_1024"

# Query defaults, can be overridden at runtime

# The number of results to return
TOP_K = 5
# The minimum similarity threshold for a result to be considered
MIN_THRESHOLD = 0.3

# Logging
APP_NAME = "bertrand"
LOG_FILE = f"{APP_NAME}"
LOG_DIR = f"/tmp/{APP_NAME}/logs/"
