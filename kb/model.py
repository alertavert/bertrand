from typing import List, Tuple

import ollama

from constants import (
    LLM_MODEL,
)
from embeddings import EmbeddingsGenerator, EmbeddingsStore
from utils import get_logger

Log = get_logger()

class KBQueryRunner:
    """Main interface towards the LLM model and the embeddings store."""

    def __init__(self, generator: EmbeddingsGenerator, store: EmbeddingsStore):
        self.generator = generator
        self.store = store

    def generate_context(self, prompt: str) -> str:
        """Generate embeddings from the prompt and generate the context."""
        # Generate embeddings from the prompt
        embeddings, _ = self.generator.from_text(prompt)
        # Retrieve the chunks that match the prompt from the Vector DB
        matches: List[Tuple[str, float]] = self.store.search(embeddings[0])
        # Generate the context
        for m, score in matches:
            Log.debug(f"({score:.2f}) {m[:100]}")
        context = "\n---\n".join([match for match, _ in matches])
        Log.debug(f"Generated context with {len(matches)} matches")
        return context

    def query(self, prompt: str) -> str:
        """Query the LLM model with the prompt, generates the RAG context and returns the response."""
        # Generate the context
        context = self.generate_context(prompt)
        # Query the model
        Log.debug(f"Querying the {LLM_MODEL} model with prompt ({len(prompt)}) and context ({len(context)})")
        if len(context) > 0:
            prompt=f"{prompt}\nConsider this additional information:{context}"
        res = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
        )
        Log.debug(f"Response received: {len(res.get('response', ''))} characters.")
        if "response" not in res:
            Log.warning("LLM returned no response")
        return res.get("response", "No response found.")
