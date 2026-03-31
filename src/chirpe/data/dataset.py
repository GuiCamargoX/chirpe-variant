"""Dataset classes for CHiRPE."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CHRPDataset(Dataset):
    """Dataset for Clinical High-Risk for Psychosis classification."""

    LABEL_MAP = {"CHR-P": 1, "Healthy": 0, "Control": 0, "HC": 0}

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        text_column: str = "summary",
        label_column: str = "label",
    ):
        """Initialize the dataset.

        Args:
            data: List of data samples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            text_column: Column containing text
            label_column: Column containing labels
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.data[idx]

        text = item[self.text_column]
        label_str = item[self.label_column]

        # Convert label
        if isinstance(label_str, str):
            label = self.LABEL_MAP.get(label_str, 0)
        else:
            label = int(label_str)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
            "participant_id": item.get("participant_id", f"unknown_{idx}"),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        labels = []
        for item in self.data:
            label_str = item[self.label_column]
            if isinstance(label_str, str):
                labels.append(self.LABEL_MAP.get(label_str, 0))
            else:
                labels.append(int(label_str))

        counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float)


class TranscriptLevelDataset(Dataset):
    """Dataset for transcript-level classification with segment voting."""

    LABEL_MAP = {"CHR-P": 1, "Healthy": 0, "Control": 0, "HC": 0}

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512,
        text_column: str = "summary",
        label_column: str = "label",
    ):
        """Initialize the dataset.

        Args:
            data: List of data samples with segments
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            text_column: Column containing text
            label_column: Column containing labels
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

        # Flatten segments for batch processing
        self.flat_data = []
        for item in data:
            segments = item.get("segments", [item])
            for seg in segments:
                self.flat_data.append({
                    "text": seg.get(text_column, seg.get("text", "")),
                    "label": item[label_column],
                    "participant_id": item.get("participant_id", "unknown"),
                    "domain": seg.get("domain", "unknown"),
                })

    def __len__(self) -> int:
        return len(self.flat_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.flat_data[idx]

        text = item["text"]
        label_str = item["label"]

        # Convert label
        if isinstance(label_str, str):
            label = self.LABEL_MAP.get(label_str, 0)
        else:
            label = int(label_str)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
            "participant_id": item["participant_id"],
            "domain": item["domain"],
        }

    def get_original_item(self, participant_id: str) -> Optional[Dict]:
        """Get original data item by participant ID."""
        for item in self.data:
            if item.get("participant_id") == participant_id:
                return item
        return None


def load_data(data_dir: Path, split: str = "train") -> List[Dict]:
    """Load data from JSON file.

    Args:
        data_dir: Directory containing data files
        split: Data split (train/val/test)

    Returns:
        List of data samples
    """
    file_path = data_dir / f"{split}.json"

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    with open(file_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def split_data(
    data: List[Dict],
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test sets.

    Args:
        data: Full dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed

    Returns:
        Tuple of (train, val, test) data lists
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data))

    n_train = int(len(data) * train_ratio)
    n_val = int(len(data) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    logger.info(
        f"Split data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )
    return train_data, val_data, test_data
