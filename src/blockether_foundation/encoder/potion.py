"""
Potion-8M encoder module for text embeddings using model2vec.

This module provides a singleton encoder that's initialized once and reused
across the application for generating text embeddings. Direct instantiation
is prohibited - use the class methods instead.
"""

import asyncio
import logging
from pathlib import Path

import numpy as np
from agno.knowledge.embedder import Embedder
from model2vec import StaticModel
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity  # type: ignore

logger = logging.getLogger(__name__)


class PotionEncoder:
    """
    Singleton encoder for generating text embeddings using Potion-8M model.

    This class provides static methods for encoding text into embeddings
    using the model2vec StaticModel. The model is loaded once on first use
    and cached for subsequent calls.

    Direct instantiation is prohibited - use class methods only.
    """

    def __init__(self):
        """Private constructor - do not instantiate directly!"""
        raise RuntimeError(
            "PotionEncoder cannot be instantiated. "
            "Use class methods like PotionEncoder.encode() directly."
        )

    # Class-level attributes for singleton pattern
    _model: StaticModel | None = None
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize the model if not already loaded."""
        if not cls._initialized:
            try:
                # Get the path to the static model - same approach as templates
                module_path = Path(__file__).parent.parent
                local_model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

                logger.info(f"Loading encoder model from: {local_model_path}")
                cls._model = StaticModel.from_pretrained(str(local_model_path))
                cls._initialized = True
                logger.info("Encoder model loaded successfully")
            except Exception as e:
                logger.exception("Failed to load encoder model")
                raise RuntimeError(f"Could not initialize encoder model: {e}") from e

    @classmethod
    def encode(cls, text: str | list[str]) -> np.ndarray:  # type: ignore
        """
        Encode text into embeddings.

        Args:
            text: Single text string or list of texts to encode

        Returns:
            Numpy array of embeddings. Shape (1, embedding_dim) for single text,
            or (n_texts, embedding_dim) for multiple texts.

        Raises:
            RuntimeError: If model initialization fails
        """
        # Ensure model is initialized
        cls._initialize()

        if not cls._model:
            raise RuntimeError("Encoder model is not initialized")

        if isinstance(text, str):
            return cls._model.encode([text])  # type: ignore

        return cls._model.encode(text)  # type: ignore

    @classmethod
    def encode_single(cls, text: str) -> np.ndarray:  # type: ignore
        """
        Encode a single text into an embedding vector.

        Args:
            text: Single text string to encode

        Returns:
            1D numpy array of the embedding vector

        Raises:
            RuntimeError: If model initialization fails
        """
        embeddings = cls.encode(text)  # type: ignore
        return embeddings[0]  # type: ignore[no-any-return]

    @classmethod
    def cosine_similarity(
        cls,
        embedding1: np.ndarray,  # type: ignore
        embedding2: np.ndarray,  # type: ignore
    ) -> float:
        """
        Calculate cosine similarity between two embeddings using sklearn.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1

        Raises:
            ValueError: If embeddings have different shapes
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(
                f"Embedding shapes must match: {embedding1.shape} != {embedding2.shape}"
            )

        # Reshape for sklearn (needs 2D arrays)
        emb1 = embedding1.reshape(1, -1)  # type: ignore
        emb2 = embedding2.reshape(1, -1)  # type: ignore

        similarity = sklearn_cosine_similarity(emb1, emb2)[0, 0]
        return float(similarity)


class PotionAgnoVectorEmbedder(Embedder):
    dimensions: int | None = 256  # Potion-8M-base embedding size
    enable_batch: bool = True
    batch_size: int = 50  # Number of texts to process in each API call

    def get_embedding(self, text: str) -> list[float]:
        embedding = PotionEncoder.encode_single(text)  # type: ignore
        return embedding.tolist()

    async def async_get_embedding(self, text: str) -> list[float]:
        """Async version using thread executor for CPU-bound operations."""

        loop = asyncio.get_event_loop()
        # Run the CPU-bound operation in a thread executor
        return await loop.run_in_executor(None, self.get_embedding, text)

    def get_embedding_and_usage(self, text: str) -> tuple[list[float], None]:
        embedding = self.get_embedding(text)
        return embedding, None

    async def async_get_embedding_and_usage(self, text: str) -> tuple[list[float], None]:
        """Async version using thread executor for CPU-bound operations."""

        loop = asyncio.get_event_loop()
        # Run the CPU-bound operation in a thread executor
        return await loop.run_in_executor(None, self.get_embedding_and_usage, text)
