"""Public model-layer API for classifier and trainer components."""

from chirpe.models.classifier import CHRClassifier
from chirpe.models.trainer import ModelTrainer

__all__ = [
    "CHRClassifier",
    "ModelTrainer",
]
