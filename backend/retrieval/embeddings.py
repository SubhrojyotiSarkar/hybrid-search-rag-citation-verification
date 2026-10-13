"""Text embedding behind a small interface, so the model can be swapped or faked in tests."""

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dimension) float32 array of L2-normalized vectors."""
        ...


class SentenceTransformerEmbedder:
    """Sentence Transformers model running on CPU.

    Vectors are L2-normalized, so the inner product of two vectors is their cosine
    similarity. Queries and passages use the same model (MiniLM is symmetric).
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = int(self.model.get_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
