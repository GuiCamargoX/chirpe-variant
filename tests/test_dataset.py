"""Tests for dataset module."""

import pytest
import torch

from chirpe.data.dataset import CHRPDataset, load_data, split_data


class TestCHRPDataset:
    """Test CHRPDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return [
            {"summary": "This is a test summary", "label": "CHR-P"},
            {"summary": "Another test summary", "label": "Healthy"},
            {"summary": "Third summary here", "label": "CHR-P"},
        ]

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer."""
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
                }

        return MockTokenizer()

    def test_initialization(self, sample_data, mock_tokenizer):
        """Test dataset initialization."""
        dataset = CHRPDataset(sample_data, mock_tokenizer)
        assert len(dataset) == len(sample_data)

    def test_getitem(self, sample_data, mock_tokenizer):
        """Test getting items."""
        dataset = CHRPDataset(sample_data, mock_tokenizer)
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["labels"], torch.Tensor)

    def test_label_mapping(self, sample_data, mock_tokenizer):
        """Test label mapping."""
        dataset = CHRPDataset(sample_data, mock_tokenizer)

        item1 = dataset[0]  # CHR-P -> 1
        item2 = dataset[1]  # Healthy -> 0

        assert item1["labels"].item() == 1
        assert item2["labels"].item() == 0

    def test_get_class_weights(self, sample_data, mock_tokenizer):
        """Test class weights calculation."""
        dataset = CHRPDataset(sample_data, mock_tokenizer)
        weights = dataset.get_class_weights()

        assert isinstance(weights, torch.Tensor)
        assert len(weights) == 2  # Binary classification


class TestDataUtils:
    """Test data utility functions."""

    def test_split_data(self):
        """Test data splitting."""
        data = [{"id": i, "label": i % 2} for i in range(100)]

        train, val, test = split_data(data, train_ratio=0.6, val_ratio=0.2, seed=42)

        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20
        assert len(train) + len(val) + len(test) == len(data)

        # Check no overlap
        train_ids = {item["id"] for item in train}
        val_ids = {item["id"] for item in val}
        test_ids = {item["id"] for item in test}

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
