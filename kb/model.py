from typing import List, Tuple

import ollama

from constants import (
    LLM_MODEL,
)
from embeddings import EmbeddingsGenerator, EmbeddingsStore

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
        context = "\n---\n".join([match[0] for match in matches])
        return context

    def query(self, prompt: str) -> str:
        """Query the LLM model with the prompt, generates the RAG context and returns the response."""
        # Generate the context
        context = self.generate_context(prompt)
        # Query the model
        res = ollama.generate(
            model=LLM_MODEL,
            prompt=f"""
                Please answer the following question:
                {prompt}
                When answering, also consider this additional information:
                {context}
                """,
        )
        return res.get("response", "No response found.")
