"""CHiRPE: Clinical High-Risk Prediction with Explainability.

A human-centred NLP framework for predicting Clinical High-Risk for Psychosis (CHR-P)
from semi-structured clinical interview transcripts.
"""

__version__ = "0.1.0"
__author__ = "CHiRPE Team"
__email__ = "stephanie.fong@unimelb.edu.au"

from chirpe.data.preprocessor import TranscriptPreprocessor
from chirpe.models.classifier import CHRClassifier
from chirpe.explanations.shap_generator import SHAPExplainer

__all__ = [
    "TranscriptPreprocessor",
    "CHRClassifier",
    "SHAPExplainer",
]
