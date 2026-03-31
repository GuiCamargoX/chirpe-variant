#!/usr/bin/env python3
"""Inference script for loading and using trained CHiRPE models."""

import argparse
import json
import logging
from pathlib import Path

from chirpe.models.classifier import CHRClassifier
from chirpe.data.preprocessor import TranscriptPreprocessor
from chirpe.explanations.shap_generator import SHAPExplainer
from chirpe.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load a trained CHiRPE model.

    Args:
        model_path: Path to the saved model directory

    Returns:
        Loaded classifier
    """
    logger.info(f"Loading model from {model_path}")

    classifier = CHRClassifier(model_name=model_path)
    classifier.load(model_path)

    # Load model info if available
    model_info_path = Path(model_path) / "model_info.json"
    if model_info_path.exists():
        with open(model_info_path) as f:
            model_info = json.load(f)
        logger.info(f"Model type: {model_info.get('model_type', 'unknown')}")
        logger.info(f"Max length: {model_info.get('max_length', 512)}")

    logger.info("Model loaded successfully!")
    return classifier


def predict_transcript(classifier, text: str):
    """Make prediction on a single text.

    Args:
        classifier: Loaded classifier
        text: Input text

    Returns:
        Prediction result
    """
    prediction, probability = classifier.predict([text], return_probs=True)

    label = "CHR-P" if prediction[0] == 1 else "Healthy"
    confidence = probability[0][prediction[0]]

    return {
        "label": label,
        "label_id": int(prediction[0]),
        "confidence": float(confidence),
        "probabilities": {
            "Healthy": float(probability[0][0]),
            "CHR-P": float(probability[0][1]),
        }
    }


def explain_prediction(classifier, text: str, output_dir: Path = None):
    """Generate SHAP explanation for a prediction.

    Args:
        classifier: Loaded classifier
        text: Input text
        output_dir: Optional directory to save explanations

    Returns:
        Explanation dictionary
    """
    explainer = SHAPExplainer(
        classifier.model,
        classifier.tokenizer,
        device=classifier.device
    )

    explanations = explainer.explain([text])
    exp = explanations[0]

    # Get top words
    top_words = explainer.get_word_level_summary(exp, top_k=10)

    result = {
        "top_words": [{"word": w, "score": float(s)} for w, s in top_words],
    }

    # Save plots if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        explainer.plot_word_level(
            exp,
            top_k=10,
            save_path=output_dir / "word_level_shap.png"
        )

        result["plots_saved_to"] = str(output_dir)

    return result


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="CHiRPE Inference - Load and use trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Text to classify (or path to JSON file with transcript)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate SHAP explanations",
    )
    parser.add_argument(
        "--explain-output-dir",
        type=Path,
        help="Directory to save explanation plots",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)

    # Load model
    classifier = load_model(str(args.model_path))

    # Get input
    if args.input:
        if Path(args.input).exists() and str(args.input).endswith('.json'):
            # Load from JSON file
            with open(args.input) as f:
                data = json.load(f)
            if isinstance(data, dict) and "text" in data:
                text = data["text"]
            elif isinstance(data, dict) and "transcript" in data:
                # Process transcript
                text = " ".join([u["text"] for u in data["transcript"]])
            else:
                text = str(data)
        else:
            text = args.input
    else:
        # Interactive mode
        print("\nEnter text to classify (or 'quit' to exit):")
        text = input("> ")
        if text.lower() == 'quit':
            return

    # Predict
    logger.info(f"Input text: {text[:100]}...")
    result = predict_transcript(classifier, text)

    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.4f}")
    print("="*50)

    # Generate explanations
    if args.explain:
        logger.info("Generating explanations...")
        explanation = explain_prediction(
            classifier,
            text,
            output_dir=args.explain_output_dir
        )

        print("\nTop Contributing Words:")
        for item in explanation["top_words"][:5]:
            print(f"  {item['word']}: {item['score']:.4f}")

        if args.explain_output_dir:
            print(f"\nExplanations saved to: {args.explain_output_dir}")

        result["explanation"] = explanation

    # Save output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return result


if __name__ == "__main__":
    main()
