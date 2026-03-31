# CHiRPE: Clinical High-Risk Prediction with Explainability

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> [!IMPORTANT]
> **Unofficial Implementation**: This repository contains an unofficial implementation of the CHiRPE framework based on the paper. The official implementation can be found in a separate repository.

CHiRPE is a human-centred NLP framework for predicting Clinical High-Risk for Psychosis (CHR-P) from semi-structured clinical interview transcripts. It integrates symptom domain segmentation, LLM summarisation, BERT-based classification, and clinician-friendly SHAP explanations.

## Overview

The CHiRPE pipeline processes PSYCHS interview transcripts to:
1. **Segment** transcripts into 15 symptom domains using fuzzy string matching
2. **Summarise** segments using LLMs (rephrased to third-person)
3. **Classify** using domain-specific BERT models (BERT, ClinicalBERT, MentalBERT)
4. **Explain** predictions using novel SHAP formats co-designed with clinicians

## Features

- **Symptom Domain Segmentation**: Maps interviewer utterances to 15 PSYCHS symptom domains
- **LLM Summarisation**: Rephrases interview segments to match BERT pretraining data
- **Multi-Model Classification**: Supports BERT, ClinicalBERT, and MentalBERT
- **Clinician-Friendly Explanations**:
  - Word-level SHAP plots
  - Token-level heatmaps
  - Symptom-level bar plots
  - Sentence-level summaries
  - Narrative summaries (hybrid graph + text)

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

#### Using Conda

```bash
conda create -n chirpe-env python=3.9 -y
conda activate chirpe-env
pip install -e ".[dev]"
```

## Quick Start

### 1. Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py --n-participants 100 --output-dir data/synthetic
```

### 2. Train a Model

```bash
chirpe-train --config configs/bert_config.yaml --data-dir data/synthetic --output-dir outputs/
```

### 3. Make Predictions

```bash
chirpe-predict --model-path outputs/best_model --input-file data/synthetic/test_transcript.json --output-dir predictions/
```

### 4. Evaluate Explanations

```bash
chirpe-evaluate --predictions predictions/results.json --output-dir evaluation/
```

## Project Structure

```
chirpe/
├── src/chirpe/          # Core source code
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model definitions and training
│   ├── explanations/    # SHAP explanation generation
│   └── utils/           # Utility functions
├── tests/               # Unit tests
├── configs/             # Configuration files
├── data/                # Data directory
│   ├── raw/             # Raw transcripts
│   ├── processed/       # Processed data
│   └── synthetic/       # Synthetic test data
├── notebooks/           # Jupyter notebooks for exploration
└── scripts/             # Utility scripts
```

## Configuration

Models are configured via YAML files. See `configs/` for examples:

- `bert_config.yaml`: Standard BERT configuration


## Data Format

### Input Transcript Format

```json
{
  "participant_id": "P001",
  "transcript": [
    {
      "speaker": "interviewer",
      "text": "Have you ever felt like people are talking about you?",
      "timestamp": "00:01:23"
    },
    {
      "speaker": "interviewee",
      "text": "Sometimes I feel like they're whispering about me...",
      "timestamp": "00:01:45"
    }
  ],
  "label": "CHR-P"
}
```


## Explanation Formats

CHiRPE generates five explanation formats:

1. **Word-level SHAP plots**: Traditional bar plots of top contributing words
2. **Token-level heatmaps**: Colour-coded transcript highlighting
3. **Symptom-level plots**: Domain-wise SHAP aggregations
4. **Sentence-level summaries**: Highest-impact sentences
5. **Narrative summaries**: LLM-generated clinical narratives with quotes

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=chirpe
```

## Citation

If you use CHiRPE in your research, please cite:

```bibtex
@article{fong2025chirpe,
  title={CHiRPE: A Step Towards Real-World Clinical NLP with Clinician-Oriented Model Explanations},
  author={Fong, Stephanie and Wang, Zimu and Oliveira, Guilherme C. and others},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

