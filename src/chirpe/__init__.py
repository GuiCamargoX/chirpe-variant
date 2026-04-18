"""CHiRPE: Clinical High-Risk Prediction with Explainability.

A human-centred NLP framework for predicting Clinical High-Risk for Psychosis (CHR-P)
from semi-structured clinical interview transcripts.

Note:
    Importing `chirpe` eagerly imports key classes for convenience. Lightweight
    scripts may prefer importing submodules directly (for example,
    `from chirpe.data.preprocessor import TranscriptPreprocessor`) to avoid
    unnecessary initialization overhead.
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
