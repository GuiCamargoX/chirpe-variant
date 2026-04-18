"""SHAP explanation generation for CHiRPE."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Generate SHAP explanations and derived visual summaries.

    The explainer consumes segment summaries (or free text), computes token
    attributions, and exposes convenience helpers for multiple clinician-facing
    output formats.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        max_samples_for_background: int = 100,
    ):
        """Initialize SHAP explainer.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            device: Device to use
            max_samples_for_background: Max samples for background data
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_samples_for_background = max_samples_for_background

        # Create SHAP explainer
        self._create_explainer()

    def _create_explainer(self):
        """Create and store a SHAP text explainer bound to this model."""
        # Define prediction function
        def predict_fn(texts):
            """Convert raw texts to class probabilities for SHAP internals."""
            # Handle various input types from SHAP masker
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, np.ndarray):
                # SHAP masker returns numpy array of strings
                texts = texts.tolist()
            elif not isinstance(texts, list):
                texts = [str(texts)]

            # SHAP may emit empty masks; normalize to safe placeholder strings
            # to avoid tokenizer edge cases.
            texts = [str(t) if t else "" for t in texts]

            # Replace any remaining empty strings with placeholder
            texts = [t if t.strip() else "[EMPTY]" for t in texts]

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            return probs.cpu().numpy()

        # Text masker uses tokenizer token boundaries for perturbations.
        masker = shap.maskers.Text(self.tokenizer)

        # Create explainer
        self.explainer = shap.Explainer(predict_fn, masker)

    def explain(
        self, texts: Union[str, List[str]], batch_size: int = 4
    ) -> List[shap.Explanation]:
        """Generate SHAP explanations.

        Args:
            texts: Text or list of texts to explain
            batch_size: Batch size for SHAP computation

        Returns:
            List of SHAP explanation objects, one per input text.
        """
        if isinstance(texts, str):
            texts = [texts]

        explanations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            exp = self.explainer(batch)
            if len(batch) == 1:
                explanations.append(exp)
            else:
                explanations.extend([exp[j] for j in range(len(batch))])

        return explanations

    def explain_segments(self, segments: List[Dict]) -> Dict[str, shap.Explanation]:
        """Explain predictions for multiple segments.

        Args:
            segments: List of segment dicts with 'summary' key

        Returns:
            Dictionary mapping segment domain to SHAP explanation.
        """
        # Filter out segments with empty summaries
        valid_segments = [seg for seg in segments if seg.get("summary", "").strip()]

        if not valid_segments:
            logger.warning("No valid segments with non-empty summaries")
            return {}

        summaries = [seg["summary"] for seg in valid_segments]
        domains = [seg["domain"] for seg in valid_segments]

        explanations = self.explain(summaries)

        return dict(zip(domains, explanations))

    def get_word_level_summary(
        self, explanation: shap.Explanation, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top contributing words.

        Args:
            explanation: SHAP explanation
            top_k: Number of top words to return

        Returns:
            List of (word, score) tuples
        """
        values = np.array(explanation.values)
        words = np.array(explanation.data)

        # Handle batched explanations (shape: [batch, tokens, classes])
        if values.ndim >= 3:
            # Take first item in batch and positive class (index 1)
            values = values[0, :, 1]
        elif values.ndim == 2:
            # Shape: [tokens, classes] - take positive class
            values = values[:, 1]

        # Handle words structure
        if words.ndim >= 2:
            words = words[0]  # Take first batch
        elif isinstance(words, tuple):
            words = words[0]  # Unwrap tuple

        # Flatten if needed
        words = np.array(words).flatten()
        values = np.array(values).flatten()

        # Get indices sorted by absolute SHAP value
        indices = np.argsort(np.abs(values))[::-1][:top_k]

        return [(str(words[i]), float(values[i])) for i in indices]

    def get_sentence_level_summary(
        self, text: str, explanation: shap.Explanation
    ) -> str:
        """Get the most important sentence.

        Args:
            text: Full text
            explanation: SHAP explanation

        Returns:
            Most important sentence
        """
        sentences = text.split(".")
        sentence_scores = []

        for sent_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            # Placeholder heuristic: token-to-sentence alignment is not yet
            # implemented, so sentence ranking is currently random.
            score = np.random.random()  # Placeholder
            sentence_scores.append((sentence.strip(), score))

        if sentence_scores:
            return max(sentence_scores, key=lambda x: x[1])[0]
        return text[:200]  # Fallback

    def aggregate_by_domain(
        self,
        explanations: Dict[str, shap.Explanation],
    ) -> Dict[str, float]:
        """Aggregate SHAP values by symptom domain.

        Args:
            explanations: Dictionary of domain -> explanation

        Returns:
            Dictionary of domain -> mean absolute SHAP value.
        """
        domain_scores = {}
        for domain, exp in explanations.items():
            # Mean absolute attribution magnitude is used as domain salience.
            score = float(np.mean(np.abs(exp.values)))
            domain_scores[domain] = score
        return domain_scores

    def plot_word_level(
        self,
        explanation: shap.Explanation,
        top_k: int = 10,
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create word-level SHAP bar plot.

        Args:
            explanation: SHAP explanation
            top_k: Number of top words
            save_path: Path to save figure
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        words_scores = self.get_word_level_summary(explanation, top_k)
        words = [w for w, _ in words_scores]
        scores = [s for _, s in words_scores]

        colors = ["red" if s > 0 else "blue" for s in scores]
        ax.barh(words, scores, color=colors, alpha=0.7)
        ax.set_xlabel("SHAP Value")
        ax.set_title("Top Contributing Words")
        ax.invert_yaxis()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved word-level plot to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_symptom_level(
        self,
        domain_scores: Dict[str, float],
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create symptom-level SHAP bar plot.

        Args:
            domain_scores: Dictionary of domain -> score
            save_path: Path to save figure
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        domains = list(domain_scores.keys())
        scores = list(domain_scores.values())

        # Sort by score
        sorted_indices = np.argsort(scores)
        domains = [domains[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        colors = ["red" if s > 0 else "blue" for s in scores]
        ax.barh(domains, scores, color=colors, alpha=0.7)
        ax.set_xlabel("Mean SHAP Value")
        ax.set_title("Symptom-Level SHAP Values")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved symptom-level plot to {save_path}")

        if show:
            plt.show()

        return fig

    def create_token_heatmap(
        self,
        text: str,
        explanation: shap.Explanation,
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create token-level heatmap visualization.

        Args:
            text: Original text (included for API symmetry; token labels are
                taken from explanation data).
            explanation: SHAP explanation
            save_path: Path to save figure
            show: Whether to show the plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 4))

        tokens = np.array(explanation.data).flatten()
        values = np.array(explanation.values)

        # Handle batched explanations
        if values.ndim >= 3:
            # [batch, tokens, classes] - take first batch and positive class
            values = values[0, :, 1]
        elif values.ndim == 2:
            # [tokens, classes] - take positive class
            values = values[:, 1]

        values = values.flatten()

        # Normalize values for color mapping (-1 to 1 range)
        max_val = np.max(np.abs(values))
        if max_val > 0:
            normalized = values / max_val
        else:
            normalized = values

        # Create heatmap-style visualization
        for i, (token, norm_val) in enumerate(zip(tokens, normalized)):
            # Map normalized value to color
            color = plt.cm.RdBu_r((norm_val + 1) / 2)
            ax.bar(i, 1, color=color, width=1.0)
            ax.text(i, 0.5, str(token), ha="center", va="center", fontsize=8, rotation=90)

        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title("Token-Level SHAP Heatmap")
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved token heatmap to {save_path}")

        if show:
            plt.show()

        return fig

    def generate_all_explanations(
        self,
        segments: List[Dict],
        output_dir: Path,
    ) -> Dict:
        """Generate all explanation formats.

        Args:
            segments: List of segment dicts
            output_dir: Directory to save explanations

        Returns:
            Dictionary of all explanations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get explanations
        explanations = self.explain_segments(segments)

        # Word-level plots
        word_level_dir = output_dir / "word_level"
        word_level_dir.mkdir(exist_ok=True)

        for domain, exp in explanations.items():
            safe_domain = domain.replace("/", "_")
            self.plot_word_level(
                exp,
                save_path=word_level_dir / f"{safe_domain}.png",
            )

        # Symptom-level plot
        domain_scores = self.aggregate_by_domain(explanations)
        self.plot_symptom_level(
            domain_scores,
            save_path=output_dir / "symptom_level.png",
        )

        # Token heatmaps
        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)

        for domain, exp in explanations.items():
            safe_domain = domain.replace("/", "_")
            # Find the segment text for this domain
            text = next(s["summary"] for s in segments if s["domain"] == domain)
            self.create_token_heatmap(
                text,
                exp,
                save_path=heatmap_dir / f"{safe_domain}.png",
            )

        return {
            "explanations": explanations,
            "domain_scores": domain_scores,
            "output_dir": output_dir,
        }
