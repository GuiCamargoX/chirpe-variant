"""Command-line interface for CHiRPE."""

import argparse
import json
import logging
import sys
from pathlib import Path

from chirpe.data.dataset import load_data, split_data
from chirpe.data.dataset import CHRPDataset as Dataset
from chirpe.models.classifier import CHRClassifier
from chirpe.models.trainer import ModelTrainer
from chirpe.utils.config import load_config
from chirpe.utils.logging_utils import setup_logging
from chirpe.utils.metrics import calculate_all_metrics, print_metrics

logger = logging.getLogger(__name__)


def train_cli():
    """CLI for training models."""
    parser = argparse.ArgumentParser(
        description="Train CHiRPE classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.output_dir / "train.log")
    logger.info(f"Starting training with config: {args.config}")

    # Load config
    config = load_config(args.config)

    # Load data
    train_data = load_data(args.data_dir, "train")
    val_data = load_data(args.data_dir, "val")

    if not train_data:
        logger.error("No training data found")
        sys.exit(1)

    if not val_data:
        logger.warning("No validation data found, will split from training data")
        from chirpe.data.dataset import split_data as ds_split

        train_data, val_data, _ = ds_split(train_data, train_ratio=0.8, val_ratio=0.2)

    # Preprocess data
    from chirpe.data.preprocessor import TranscriptPreprocessor
    from chirpe.data.segmentation import SymptomSegmenter

    logger.info("Preprocessing data...")
    preprocessor = TranscriptPreprocessor(
        segmentation_threshold=config["preprocessing"]["segmentation_threshold"],
        use_llm_summarizer=False,  # Use simple summarizer for CLI
    )
    segmenter = SymptomSegmenter(threshold=config["preprocessing"]["segmentation_threshold"])

    def preprocess_data(data, max_segments=3):
        """Preprocess data into segments."""
        processed = []
        for item in data:
            result = preprocessor.process_transcript(item["transcript"], item["participant_id"])
            # Flatten to segment-level
            for seg in result.get("segments", [])[:max_segments]:
                processed.append({
                    "participant_id": item["participant_id"],
                    "summary": seg["summary"],
                    "label": item["label"],
                })
        return processed

    max_segments = config.get("preprocessing", {}).get("max_segments_per_transcript", 3)
    train_data = preprocess_data(train_data, max_segments)
    val_data = preprocess_data(val_data, max_segments)

    logger.info(f"Preprocessed: {len(train_data)} train, {len(val_data)} val segments")

    # Initialize classifier
    classifier = CHRClassifier(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        max_length=config["model"]["max_length"],
    )

    # Create datasets
    train_dataset = Dataset(
        train_data,
        classifier.tokenizer,
        max_length=config["model"]["max_length"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
    )
    val_dataset = Dataset(
        val_data,
        classifier.tokenizer,
        max_length=config["model"]["max_length"],
        text_column=config["data"]["text_column"],
        label_column=config["data"]["label_column"],
    )

    # Initialize trainer
    trainer = ModelTrainer(
        classifier=classifier,
        output_dir=args.output_dir,
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        num_epochs=config["training"]["num_epochs"],
        weight_decay=config["training"]["weight_decay"],
        warmup_ratio=config["training"]["warmup_ratio"],
        logging_steps=config["training"]["logging_steps"],
        seed=config["training"]["seed"],
    )

    # Train
    trainer.train(train_dataset, val_dataset)

    # Save config for later prediction (inside best_model dir)
    import json
    config_save_path = args.output_dir / "best_model" / "chirpe_config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_save_path}")

    logger.info("Training complete!")


def predict_cli():
    """CLI for making predictions."""
    parser = argparse.ArgumentParser(
        description="Make predictions with CHiRPE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to input transcript JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./predictions"),
        help="Output directory",
    )
    parser.add_argument(
        "--generate-explanations",
        action="store_true",
        help="Generate SHAP explanations",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting prediction")

    # Load config if available
    config = {}
    # Try model dir first, then parent dir (for backwards compat)
    config_path = Path(args.model_path) / "chirpe_config.json"
    if not config_path.exists() and (Path(args.model_path).parent / "chirpe_config.json").exists():
        config_path = Path(args.model_path).parent / "chirpe_config.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"No config found at {config_path}, using defaults")

    # Load model
    classifier = CHRClassifier(model_name=str(args.model_path))
    classifier.load(args.model_path)

    # Load input
    with open(args.input_file, "r") as f:
        input_data = json.load(f)

    # Handle both single transcript (dict) and batch (list)
    if isinstance(input_data, list):
        input_data = input_data[0]  # Use first transcript for prediction

    # Preprocess transcript if it has raw interview data
    if "transcript" in input_data and isinstance(input_data["transcript"], list):
        logger.info("Preprocessing transcript (segmenting + summarizing)...")
        from chirpe.data.preprocessor import TranscriptPreprocessor

        # Use config values if available, otherwise defaults
        seg_threshold = config.get("preprocessing", {}).get("segmentation_threshold", 0.8)
        # Check if we should use LLM summarizer
        use_llm = False
        if config.get("llm", {}).get("use_api", False):
            use_llm = True
        elif config.get("ultra_quick", {}).get("use_simple_summarizer_fallback", True):
            use_llm = False

        preprocessor = TranscriptPreprocessor(
            segmentation_threshold=seg_threshold,
            use_llm_summarizer=use_llm,
        )
        logger.info(f"Using segmentation_threshold={seg_threshold}, use_llm_summarizer={use_llm}")

        result = preprocessor.process_transcript(
            input_data["transcript"],
            input_data.get("participant_id", "unknown")
        )
        segments = result.get("segments", [])
        logger.info(f"Extracted {len(segments)} segments from transcript")

        if not segments:
            logger.error("No segments found in transcript")
            sys.exit(1)

        # Predict with segments
        results = classifier.predict_with_segments(segments)
        results["participant_id"] = input_data.get("participant_id", "unknown")
        results["num_segments"] = len(segments)
        results["segments"] = [
            {"domain": s["domain"], "summary": s["summary"]} for s in segments
        ]

        # Store segments for explanations
        input_data["segments"] = segments

    elif "segments" in input_data:
        # Pre-segmented data
        segments = input_data["segments"]
        results = classifier.predict_with_segments(segments)
    else:
        # Simple text input
        text = input_data.get("text", "")
        pred, probs = classifier.predict(text, return_probs=True)
        results = {
            "prediction": int(pred[0]),
            "confidence": float(probs[0][pred[0]]),
            "probabilities": probs[0].tolist(),
        }

    # Generate explanations if requested
    if args.generate_explanations:
        from chirpe.explanations.shap_generator import SHAPExplainer

        explainer = SHAPExplainer(
            classifier.model,
            classifier.tokenizer,
            classifier.device,
        )

        if "segments" in input_data:
            explanations = explainer.generate_all_explanations(
                input_data["segments"],
                args.output_dir / "explanations",
            )
            results["explanations"] = str(explanations["output_dir"])

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Prediction complete! Results saved to {output_file}")
    print(json.dumps(results, indent=2))


def evaluate_cli():
    """CLI for evaluating models."""
    parser = argparse.ArgumentParser(
        description="Evaluate CHiRPE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing test data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./evaluation"),
        help="Output directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger.info("Starting evaluation")

    # Load model
    classifier = CHRClassifier(model_name=str(args.model_path))
    classifier.load(args.model_path)

    # Load test data
    test_data = load_data(args.data_dir, "test")

    if not test_data:
        logger.error("No test data found")
        sys.exit(1)

    # Preprocess data (same as training)
    from chirpe.data.preprocessor import TranscriptPreprocessor
    from chirpe.data.segmentation import SymptomSegmenter

    logger.info("Preprocessing test data...")
    preprocessor = TranscriptPreprocessor(
        segmentation_threshold=0.8,
        use_llm_summarizer=False,  # Use simple summarizer for speed
    )

    processed = []
    for item in test_data:
        result = preprocessor.process_transcript(item["transcript"], item["participant_id"])
        # Flatten to segment-level
        for seg in result.get("segments", [])[:3]:
            processed.append({
                "participant_id": item["participant_id"],
                "summary": seg["summary"],
                "label": item["label"],
            })

    logger.info(f"Preprocessed: {len(processed)} test segments")

    # Create dataset
    test_dataset = Dataset(
        processed,
        classifier.tokenizer,
        max_length=512,
        text_column="summary",
    )

    # Evaluate
    trainer = ModelTrainer(classifier, output_dir=args.output_dir)
    results = trainer.evaluate(test_dataset)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print_metrics(results)


if __name__ == "__main__":
    # Default to train CLI if called directly
    train_cli()
