#!/usr/bin/env python3
"""Train CHiRPE using a YAML config file.

Usage:
    python scripts/train_from_config.py --config configs/ultra_quick_config.yaml
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from chirpe.data.preprocessor import TranscriptPreprocessor
from chirpe.data.dataset import load_data
from chirpe.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def limit_data(data, max_samples):
    """Limit data to max_samples."""
    if max_samples and len(data) > max_samples:
        return data[:max_samples]
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Train CHiRPE from config file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory with train/val/test.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/config_training"),
        help="Output directory",
    )

    args = parser.parse_args()

    print("="*70)
    print("CHiRPE Training from Config")
    print("="*70)

    # Load config
    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    print(f"\nConfig: {args.config}")
    print(f"Model: {config['model']['name']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")

    # Load data
    print("\nLoading data...")
    train_data = load_data(args.data_dir, "train")
    val_data = load_data(args.data_dir, "val")
    test_data = load_data(args.data_dir, "test")

    # Limit samples for ultra-quick mode
    max_train = config["data"].get("max_train_samples")
    max_val = config["data"].get("max_val_samples")
    max_test = config["data"].get("max_test_samples")

    train_data = limit_data(train_data, max_train)
    val_data = limit_data(val_data, max_val)
    test_data = limit_data(test_data, max_test)

    print(f"Using {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Preprocess
    print("\n" + "="*70)
    print("Preprocessing...")
    print("="*70)

    # Check if using Gemma
    llm_config = config.get("llm", {})
    use_gemma = llm_config.get("model_name", "").startswith("google/gemma")

    if use_gemma:
        print("Using Gemma for LLM summarization")
        from transformers import AutoModelForCausalLM, AutoTokenizer as LLMTokenizer

        llm_model_name = llm_config["model_name"]
        print(f"Loading {llm_model_name}...")

        llm_tokenizer = LLMTokenizer.from_pretrained(llm_model_name)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        preprocessor = TranscriptPreprocessor(
            segmentation_threshold=config["preprocessing"]["segmentation_threshold"],
            use_llm_summarizer=True,
            llm_model_name=llm_model_name,
            use_api=False,
        )
    else:
        print("Using simple summarizer")
        preprocessor = TranscriptPreprocessor(
            segmentation_threshold=config["preprocessing"]["segmentation_threshold"],
            use_llm_summarizer=False,
        )
        llm_model = None
        llm_tokenizer = None

    # Process data
    from chirpe.data.segmentation import SymptomSegmenter
    segmenter = SymptomSegmenter(threshold=config["preprocessing"]["segmentation_threshold"])

    max_segments = config["preprocessing"].get("max_segments_per_transcript", 3)

    def process_split(data, name):
        processed = []
        for i, item in enumerate(data, 1):
            print(f"\n[{name}] {item['participant_id']} ({i}/{len(data)})")

            segments = segmenter.segment_transcript(item["transcript"])

            for seg in segments[:max_segments]:
                text = seg.get_text()

                if use_gemma and llm_model and llm_tokenizer:
                    # Use Gemma for summarization
                    prompt = f"""<bos><start_of_turn>user
Summarize this clinical interview segment in one sentence:

{text}
<end_of_turn>
<start_of_turn>model
"""
                    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = llm_model.generate(**inputs, max_new_tokens=100, do_sample=False)

                    input_len = inputs["input_ids"].shape[1]
                    summary = llm_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                else:
                    # Simple fallback
                    summary = text.replace("I ", "The patient ").replace("my ", "their ")[:150]

                print(f"  {summary[:60]}...")

                processed.append({
                    "participant_id": item["participant_id"],
                    "summary": summary,
                    "label": 1 if item["label"] == "CHR-P" else 0,
                })

        return processed

    train_processed = process_split(train_data, "Train")
    val_processed = process_split(val_data, "Val")
    test_processed = process_split(test_data, "Test")

    print(f"\n✓ Preprocessed: {len(train_processed)} train, {len(val_processed)} val")

    # Train classifier
    print("\n" + "="*70)
    print("Training classifier...")
    print("="*70)

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["model"]["num_labels"],
    )

    class ConfigDataset:
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            encoding = self.tokenizer(
                item["summary"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(item["label"], dtype=torch.long),
            }

    train_dataset = ConfigDataset(train_processed, tokenizer, config["model"]["max_length"])
    val_dataset = ConfigDataset(val_processed, tokenizer, config["model"]["max_length"])

    # Training args
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=config["training"]["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("\nTraining...")
    trainer.train()

    # Save model
    print("\nSaving model...")
    final_model_dir = args.output_dir / "final_model"
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save config used
    with open(final_model_dir / "config_used.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel saved to: {final_model_dir}")
    print(f"\nTo load:")
    print(f"  from chirpe.models.classifier import CHRClassifier")
    print(f"  classifier = CHRClassifier('{final_model_dir}')")
    print(f"  classifier.load('{final_model_dir}')")


if __name__ == "__main__":
    main()
