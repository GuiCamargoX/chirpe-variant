"""BERT-based classifier for CHR-P prediction."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
)

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "bert": "bert-base-uncased",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "mentalbert": "mental/mental-bert-base-uncased",
}


class CHRClassifier:
    """Wrapper around Hugging Face sequence classification models.

    The class provides convenience helpers for tokenization, batched
    predictions, and segment-level voting strategies used by CHiRPE flows.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """Initialize the classifier.

        Args:
            model_name: Model name or path
            num_labels: Number of classification labels
            device: Device to use (cuda/cpu)
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Resolve model name
        if model_name.lower() in MODEL_MAP:
            self.model_name = MODEL_MAP[model_name.lower()]

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)

        logger.info(f"Loaded model {self.model_name} on {self.device}")

    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Tokenize input texts.

        Args:
            texts: Input text or list of texts

        Returns:
            Tokenized tensor dictionary ready for model forward pass.
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def predict(
        self, texts: Union[str, List[str]], return_probs: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict labels for input texts.

        Args:
            texts: Input text or list of texts
            return_probs: Whether to return probabilities

        Returns:
            Predicted label indices, optionally with probability matrix of shape
            `[n_samples, num_labels]`.
        """
        self.model.eval()

        # Tokenize
        inputs = self.tokenize(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get predictions
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = np.argmax(probs, axis=-1)

        if return_probs:
            return predictions, probs
        return predictions

    def predict_with_segments(
        self, segments: List[Dict], voting: str = "majority"
    ) -> Dict:
        """Predict from multiple segments with voting.

        Args:
            segments: List of segment dictionaries with `summary` keys.
            voting: Transcript-level aggregation strategy:
                - `majority`: majority vote over segment predictions
                - `average`: argmax over mean segment probabilities

        Returns:
            Dictionary containing transcript prediction and per-segment outputs.
        """
        summaries = [seg["summary"] for seg in segments]
        predictions, probs = self.predict(summaries, return_probs=True)

        if voting == "majority":
            final_pred = int(np.bincount(predictions).argmax())
        elif voting == "average":
            avg_probs = np.mean(probs, axis=0)
            final_pred = int(np.argmax(avg_probs))
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

        return {
            "prediction": final_pred,
            "confidence": float(np.max(probs, axis=1).mean()),
            "segment_predictions": predictions.tolist(),
            "segment_probabilities": probs.tolist(),
            "num_segments": len(segments),
        }

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save model and tokenizer.

        Args:
            output_dir: Directory to save to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Saved model to {output_dir}")

    def load(self, model_dir: Union[str, Path]) -> None:
        """Load model and tokenizer.

        Args:
            model_dir: Directory to load from
        """
        model_dir = Path(model_dir)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)

        logger.info(f"Loaded model from {model_dir}")

    def get_embeddings(
        self, texts: Union[str, List[str]], layer: int = -1
    ) -> np.ndarray:
        """Get hidden state embeddings.

        Args:
            texts: Input text or list of texts
            layer: Which layer to extract (-1 for last)

        Returns:
            Embeddings array
        """
        self.model.eval()

        # Tokenize
        inputs = self.tokenize(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model.base_model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings


class EnsembleClassifier:
    """Ensemble of multiple BERT classifiers."""

    def __init__(self, model_paths: List[Union[str, Path]], device: Optional[str] = None):
        """Initialize ensemble.

        Args:
            model_paths: List of model directories
            device: Device to use
        """
        self.classifiers = []
        for path in model_paths:
            clf = CHRClassifier(path, device=device)
            clf.load(path)
            self.classifiers.append(clf)

    def predict(
        self, texts: Union[str, List[str]], method: str = "average"
    ) -> np.ndarray:
        """Predict using ensemble.

        Args:
            texts: Input text or list of texts
            method: Ensemble method:
                - `average`: average class probabilities, then argmax
                - `vote`: per-model hard voting

        Returns:
            Predicted labels
        """
        all_probs = []
        for clf in self.classifiers:
            _, probs = clf.predict(texts, return_probs=True)
            all_probs.append(probs)

        if method == "average":
            avg_probs = np.mean(all_probs, axis=0)
            return np.argmax(avg_probs, axis=-1)
        elif method == "vote":
            all_preds = [np.argmax(p, axis=-1) for p in all_probs]
            votes = np.array(all_preds).T
            # Majority vote
            return np.array([np.bincount(v).argmax() for v in votes])
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
