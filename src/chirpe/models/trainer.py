"""Training utilities for CHiRPE models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from chirpe.data.dataset import CHRPDataset
from chirpe.models.classifier import CHRClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Thin orchestration layer around Hugging Face `Trainer`.

    This class centralizes training arguments, metric computation, evaluation,
    and optional cross-validation loops used by CHiRPE scripts and CLI flows.
    """

    def __init__(
        self,
        classifier: CHRClassifier,
        output_dir: str = "./outputs",
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        num_epochs: int = 3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        logging_steps: int = 50,
        seed: int = 42,
    ):
        """Initialize the trainer.

        Args:
            classifier: The classifier to train
            output_dir: Output directory for checkpoints
            learning_rate: Learning rate
            batch_size: Training batch size
            num_epochs: Number of training epochs
            weight_decay: Weight decay for optimizer
            warmup_ratio: Warmup ratio for scheduler
            logging_steps: Logging frequency
            seed: Random seed
        """
        self.classifier = classifier
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.seed = seed

        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            eval_pred: Tuple-like object returned by Hugging Face Trainer,
                containing logits and label IDs.

        Returns:
            Dictionary of scalar metric values.
        """
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        }

        # ROC-AUC (only if both classes present)
        if len(np.unique(labels)) > 1:
            metrics["auc"] = roc_auc_score(labels, probs)

        return metrics

    def train(
        self,
        train_dataset: CHRPDataset,
        val_dataset: Optional[CHRPDataset] = None,
        use_class_weights: bool = True,
    ) -> None:
        """Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset. If provided, epoch-level
                evaluation and early stopping are enabled.
            use_class_weights: Whether to use class weights for imbalance
        """
        # Class weights are logged for visibility; custom loss wiring is not
        # currently injected into Trainer in this implementation.
        if use_class_weights:
            class_weights = train_dataset.get_class_weights()
            logger.info(f"Using class weights: {class_weights}")
        else:
            class_weights = None

        # Training arguments
        from transformers import IntervalStrategy

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.logging_steps,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.seed,
            report_to="none",  # Disable wandb/tensorboard for simplicity
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.classifier.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3)
            ] if val_dataset else [],
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save model/tokenizer in a stable location consumed by CLI prediction.
        trainer.save_model(self.output_dir / "best_model")
        self.classifier.tokenizer.save_pretrained(self.output_dir / "best_model")

        logger.info(f"Training complete. Model saved to {self.output_dir / 'best_model'}")

    def evaluate(self, test_dataset: CHRPDataset) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            test_dataset: Test dataset

        Returns:
            Dictionary of evaluation metrics (Trainer-prefixed keys included).
        """
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_eval_batch_size=self.batch_size,
            seed=self.seed,
            report_to="none",
        )

        trainer = Trainer(
            model=self.classifier.model,
            args=training_args,
            compute_metrics=self.compute_metrics,
        )

        results = trainer.evaluate(test_dataset)
        logger.info(f"Evaluation results: {results}")

        return results

    def cross_validate(
        self,
        dataset: CHRPDataset,
        n_splits: int = 5,
    ) -> List[Dict[str, float]]:
        """Perform k-fold cross-validation.

        Args:
            dataset: Full dataset
            n_splits: Number of folds

        Returns:
            List of per-fold evaluation metric dictionaries.
        """
        from sklearn.model_selection import StratifiedKFold

        # Extract labels for stratification
        labels = [item["labels"].item() for item in dataset]
        indices = np.arange(len(dataset))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
            logger.info(f"Fold {fold + 1}/{n_splits}")

            # Create subsets from original raw entries to preserve tokenization
            # settings and label handling.
            train_data = [dataset.data[i] for i in train_idx]
            val_data = [dataset.data[i] for i in val_idx]

            train_subset = CHRPDataset(
                train_data,
                dataset.tokenizer,
                dataset.max_length,
                dataset.text_column,
                dataset.label_column,
            )
            val_subset = CHRPDataset(
                val_data,
                dataset.tokenizer,
                dataset.max_length,
                dataset.text_column,
                dataset.label_column,
            )

            # Train and evaluate
            fold_output = self.output_dir / f"fold_{fold + 1}"
            self.output_dir = fold_output

            self.train(train_subset, val_subset)
            fold_results = self.evaluate(val_subset)
            results.append(fold_results)

        # Aggregate means/stds for logging visibility.
        aggregated = {}
        for key in results[0].keys():
            values = [r[key] for r in results if key in r]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

        logger.info(f"Cross-validation results: {aggregated}")
        return results


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])

    return metrics
