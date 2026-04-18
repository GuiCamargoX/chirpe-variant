"""Public explainability API (SHAP and narrative generators)."""

from chirpe.explanations.shap_generator import SHAPExplainer
from chirpe.explanations.narrative import NarrativeGenerator

__all__ = [
    "SHAPExplainer",
    "NarrativeGenerator",
]
