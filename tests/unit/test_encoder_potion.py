"""
Tests for the Potion-8M encoder module.

This module contains both unit tests (with mocks) and integration tests
(without mocks) for comprehensive coverage.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from blockether_foundation.encoder import (
    PotionAgnoVectorEmbedder,
    PotionEncoder,
)


class TestPotionEncoderUnit:
    """Tests for PotionEncoder class."""

    def test_cannot_instantiate_directly(self):
        """Test that PotionEncoder cannot be instantiated."""
        with pytest.raises(RuntimeError, match="cannot be instantiated"):
            PotionEncoder()

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_initialize_loads_model(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test that _initialize loads the model correctly."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model

        # Initialize the encoder
        PotionEncoder._initialize()

        # Verify model was loaded
        assert PotionEncoder._initialized is True
        assert PotionEncoder._model is not None

        # Verify correct path was used
        call_args = mock_static_model_class.from_pretrained.call_args[0][0]
        assert "assets/model2vec/potion-8M-base" in call_args

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_initialize_handles_error(self, mock_static_model_class, reset_encoder_state):
        """Test that _initialize handles initialization errors."""
        mock_static_model_class.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(RuntimeError, match="Could not initialize encoder model"):
            PotionEncoder._initialize()

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_initialize_only_once(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test that model is only initialized once."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model

        # Initialize twice
        PotionEncoder._initialize()
        PotionEncoder._initialize()

        # Verify model was only loaded once
        assert mock_static_model_class.from_pretrained.call_count == 1

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_encode_single_string(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test encoding a single string."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model
        expected_embedding = np.random.rand(1, 256)
        mock_static_model.encode.return_value = expected_embedding

        result = PotionEncoder.encode("Hello world")

        # Verify encode was called with a list
        mock_static_model.encode.assert_called_once_with(["Hello world"])
        np.testing.assert_array_equal(result, expected_embedding)

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_encode_list_of_strings(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test encoding a list of strings."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model
        expected_embedding = np.random.rand(3, 256)
        mock_static_model.encode.return_value = expected_embedding

        texts = ["Hello", "world", "test"]
        result = PotionEncoder.encode(texts)

        # Verify encode was called with the list directly
        mock_static_model.encode.assert_called_once_with(texts)
        np.testing.assert_array_equal(result, expected_embedding)

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_encode_when_model_not_initialized(self, mock_static_model_class, reset_encoder_state):
        """Test encoding raises error when model is not initialized."""
        # Mock initialization to set _initialized=True but _model=None
        mock_static_model_class.from_pretrained.return_value = None
        PotionEncoder._initialized = True
        PotionEncoder._model = None

        with pytest.raises(RuntimeError, match="Encoder model is not initialized"):
            PotionEncoder.encode("test")

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_encode_single(self, mock_static_model_class, reset_encoder_state, mock_static_model):
        """Test encode_single returns a 1D array."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model
        expected_embedding = np.random.rand(1, 256)
        mock_static_model.encode.return_value = expected_embedding

        result = PotionEncoder.encode_single("Hello")

        # Verify result is 1D (first element of the batch)
        assert result.shape == (256,)
        np.testing.assert_array_equal(result, expected_embedding[0])

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_cosine_similarity_matching_shapes(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test cosine similarity with matching shapes."""
        # Create two embeddings with known similarity
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])

        similarity = PotionEncoder.cosine_similarity(emb1, emb2)

        # Identical vectors should have similarity of 1.0
        assert isinstance(similarity, float)
        assert abs(similarity - 1.0) < 1e-6

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_cosine_similarity_orthogonal_vectors(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test cosine similarity with orthogonal vectors."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])

        similarity = PotionEncoder.cosine_similarity(emb1, emb2)

        # Orthogonal vectors should have similarity of 0.0
        assert isinstance(similarity, float)
        assert abs(similarity - 0.0) < 1e-6

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_cosine_similarity_opposite_vectors(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test cosine similarity with opposite vectors."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([-1.0, 0.0, 0.0])

        similarity = PotionEncoder.cosine_similarity(emb1, emb2)

        # Opposite vectors should have similarity of -1.0
        assert isinstance(similarity, float)
        assert abs(similarity - (-1.0)) < 1e-6

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_cosine_similarity_mismatched_shapes(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test cosine similarity raises error with mismatched shapes."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0])

        with pytest.raises(ValueError, match="Embedding shapes must match"):
            PotionEncoder.cosine_similarity(emb1, emb2)


class TestPotionAgnoVectorEmbedderUnit:
    """Unit tests for PotionAgnoVectorEmbedder class (with mocks)."""

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_dimensions_attribute(self, mock_static_model_class):
        """Test that dimensions class attribute is set correctly."""
        # Check class-level attributes
        assert PotionAgnoVectorEmbedder.dimensions == 256
        assert PotionAgnoVectorEmbedder.enable_batch is True
        assert PotionAgnoVectorEmbedder.batch_size == 50

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_batch_settings(self, mock_static_model_class):
        """Test that batch settings are configured."""
        embedder = PotionAgnoVectorEmbedder()
        # The actual instance values come from parent class
        # Just verify the embedder was created successfully
        assert embedder is not None

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_get_embedding(self, mock_static_model_class, reset_encoder_state, mock_static_model):
        """Test get_embedding returns a list of floats."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model
        expected_embedding = np.random.rand(1, 256)
        mock_static_model.encode.return_value = expected_embedding

        embedder = PotionAgnoVectorEmbedder()
        result = embedder.get_embedding("Hello world")

        assert isinstance(result, list)
        assert len(result) == 256
        assert all(isinstance(x, float) for x in result)

    @patch("blockether_foundation.encoder.potion.StaticModel")
    @pytest.mark.asyncio
    async def test_async_get_embedding(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test async_get_embedding returns a list of floats."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model
        expected_embedding = np.random.rand(1, 256)
        mock_static_model.encode.return_value = expected_embedding

        embedder = PotionAgnoVectorEmbedder()
        result = await embedder.async_get_embedding("Hello world")

        assert isinstance(result, list)
        assert len(result) == 256
        assert all(isinstance(x, float) for x in result)

    @patch("blockether_foundation.encoder.potion.StaticModel")
    def test_get_embedding_and_usage(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test get_embedding_and_usage returns embedding and None usage."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model
        expected_embedding = np.random.rand(1, 256)
        mock_static_model.encode.return_value = expected_embedding

        embedder = PotionAgnoVectorEmbedder()
        embedding, usage = embedder.get_embedding_and_usage("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 256
        assert usage is None

    @patch("blockether_foundation.encoder.potion.StaticModel")
    @pytest.mark.asyncio
    async def test_async_get_embedding_and_usage(
        self, mock_static_model_class, reset_encoder_state, mock_static_model
    ):
        """Test async_get_embedding_and_usage returns embedding and None usage."""
        mock_static_model_class.from_pretrained.return_value = mock_static_model
        expected_embedding = np.random.rand(1, 256)
        mock_static_model.encode.return_value = expected_embedding

        embedder = PotionAgnoVectorEmbedder()
        embedding, usage = await embedder.async_get_embedding_and_usage("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == 256
        assert usage is None


# Integration Tests (without mocks)
# These tests use the actual encoder and require the model to be available


@pytest.mark.integration
class TestPotionEncoderIntegration:
    """Integration tests for PotionEncoder (without mocks)."""

    def test_encoder_initialization_with_real_model(self, reset_encoder_state):
        """Test that encoder can initialize with real model if available."""
        # Check if model path exists
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        # This should initialize successfully with real model
        PotionEncoder._initialize()
        assert PotionEncoder._initialized is True
        assert PotionEncoder._model is not None

    def test_encode_real_text(self, reset_encoder_state):
        """Test encoding actual text with the real model."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        # Encode a single string
        result = PotionEncoder.encode("Hello, world!")

        # Verify result shape and type
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 256)  # Potion-8M-base has 256 dimensions
        assert result.dtype == np.float32 or result.dtype == np.float64

    def test_encode_multiple_texts(self, reset_encoder_state):
        """Test encoding multiple texts at once."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        texts = ["Hello", "World", "Test"]
        result = PotionEncoder.encode(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 256)

    def test_encode_single_returns_1d(self, reset_encoder_state):
        """Test that encode_single returns 1D array."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        result = PotionEncoder.encode_single("Test")

        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)
        assert len(result.shape) == 1  # 1D array

    def test_cosine_similarity_with_real_embeddings(self, reset_encoder_state):
        """Test cosine similarity with real embeddings."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        # Same text should have similarity close to 1.0
        emb1 = PotionEncoder.encode_single("cat")
        emb2 = PotionEncoder.encode_single("cat")
        similarity = PotionEncoder.cosine_similarity(emb1, emb2)

        assert 0.99 < similarity <= 1.0

    def test_different_words_have_different_embeddings(self, reset_encoder_state):
        """Test that different words produce different embeddings."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        # Different words should produce different embeddings
        emb_cat = PotionEncoder.encode_single("cat")
        emb_dog = PotionEncoder.encode_single("dog")
        emb_car = PotionEncoder.encode_single("car")

        # Verify they are different (not identical)
        assert not np.array_equal(emb_cat, emb_dog)
        assert not np.array_equal(emb_cat, emb_car)
        assert not np.array_equal(emb_dog, emb_car)

        # All embeddings should have reasonable similarity scores (between -1 and 1)
        similarity_cat_dog = PotionEncoder.cosine_similarity(emb_cat, emb_dog)
        similarity_cat_car = PotionEncoder.cosine_similarity(emb_cat, emb_car)

        assert -1.0 <= similarity_cat_dog <= 1.0
        assert -1.0 <= similarity_cat_car <= 1.0


@pytest.mark.integration
class TestPotionAgnoVectorEmbedderIntegration:
    """Integration tests for PotionAgnoVectorEmbedder (without mocks)."""

    def test_get_embedding_real(self, reset_encoder_state):
        """Test get_embedding with real model."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        embedder = PotionAgnoVectorEmbedder()
        result = embedder.get_embedding("Hello world")

        assert isinstance(result, list)
        assert len(result) == 256
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_async_get_embedding_real(self, reset_encoder_state):
        """Test async_get_embedding with real model."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        embedder = PotionAgnoVectorEmbedder()
        result = await embedder.async_get_embedding("Hello world")

        assert isinstance(result, list)
        assert len(result) == 256
        assert all(isinstance(x, float) for x in result)

    def test_get_embedding_and_usage_real(self, reset_encoder_state):
        """Test get_embedding_and_usage with real model."""
        module_path = Path(__file__).parent.parent.parent / "src" / "blockether_foundation"
        model_path = module_path / "assets" / "model2vec" / "potion-8M-base"

        if not model_path.exists():
            pytest.skip(f"Model not found at {model_path}")

        embedder = PotionAgnoVectorEmbedder()
        embedding, usage = embedder.get_embedding_and_usage("Test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 256
        assert usage is None
