"""Tests for classifier module."""

import numpy as np
import pytest
import torch

from chirpe.models.classifier import CHRClassifier


class TestCHRClassifier:
    """Test CHRClassifier class."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "I feel sad today",
            "I am happy now",
        ]

    def test_initialization(self):
        """Test classifier initialization."""
        # Skip if no model available
        pytest.importorskip("transformers")

        classifier = CHRClassifier(
            model_name="bert-base-uncased",
            num_labels=2,
        )

        assert classifier.model_name == "bert-base-uncased"
        assert classifier.num_labels == 2
        assert classifier.tokenizer is not None
        assert classifier.model is not None

    def test_tokenize(self, sample_texts):
        """Test tokenization."""
        pytest.importorskip("transformers")

        classifier = CHRClassifier(model_name="bert-base-uncased")
        tokens = classifier.tokenize(sample_texts)

        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == len(sample_texts)

    def test_predict(self, sample_texts):
        """Test prediction."""
        pytest.importorskip("transformers")

        classifier = CHRClassifier(model_name="bert-base-uncased")
        predictions, probs = classifier.predict(sample_texts, return_probs=True)

        assert len(predictions) == len(sample_texts)
        assert probs.shape == (len(sample_texts), 2)
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probs.flatten())

    def test_predict_without_probs(self, sample_texts):
        """Test prediction without probabilities."""
        pytest.importorskip("transformers")

        classifier = CHRClassifier(model_name="bert-base-uncased")
        predictions = classifier.predict(sample_texts, return_probs=False)

        assert len(predictions) == len(sample_texts)
        assert isinstance(predictions, np.ndarray)

    def test_predict_with_segments(self):
        """Test prediction with segments."""
        pytest.importorskip("transformers")

        classifier = CHRClassifier(model_name="bert-base-uncased")

        segments = [
            {"domain": "P1", "summary": "I hear voices"},
            {"domain": "P2", "summary": "I feel paranoid"},
        ]

        results = classifier.predict_with_segments(segments)

        assert "prediction" in results
        assert "confidence" in results
        assert "segment_predictions" in results
        assert "num_segments" in results
        assert results["num_segments"] == len(segments)


class TestEnsembleClassifier:
    """Test EnsembleClassifier class."""

    def test_initialization(self):
        """Test ensemble initialization."""
        # This would require actual model paths
        # For now, just verify the class exists
        from chirpe.models.classifier import EnsembleClassifier
        assert EnsembleClassifier is not None
