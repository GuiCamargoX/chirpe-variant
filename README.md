# CHiRPE: Clinical High-Risk Prediction with Explainability

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> [!IMPORTANT]
> **Unofficial Implementation**: This repository contains an unofficial implementation of the CHiRPE framework based on the paper. The official implementation can be found in a separate repository.

CHiRPE is a human-centred NLP framework for predicting Clinical High-Risk for Psychosis (CHR-P) from semi-structured clinical interview transcripts. It integrates symptom domain segmentation, LLM summarisation, BERT-based classification, and clinician-friendly SHAP explanations.

## Overview 🔎

The CHiRPE pipeline processes PSYCHS interview transcripts to:
1. **Segment** transcripts into 15 symptom domains using fuzzy string matching
2. **Summarise** segments using LLMs (rephrased to third-person)
3. **Classify** each segment with a single selected backbone (BERT, ClinicalBERT, or MentalBERT) and aggregate with voting
4. **Explain** predictions using novel SHAP formats co-designed with clinicians

## Features ✨

- **Symptom Domain Segmentation**: Maps interviewer utterances to 15 PSYCHS symptom domains
- **LLM Summarisation**: Rephrases interview segments to match BERT pretraining data
- **Single-Model Classification**: Choose one backbone per run (BERT, ClinicalBERT, or MentalBERT)
- **Clinician-Friendly Explanations**:
  - Word-level SHAP plots
  - Token-level heatmaps
  - Symptom-level bar plots
  - Sentence-level summaries
  - Narrative summaries (hybrid graph + text)

## Installation ⚙️

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

## Quick Start 🚀

### 1. Generate Synthetic Data

```bash
python scripts/dataprep/generate_synthetic_data.py --n-participants 100 --output-dir data/synthetic --split
```

### 2. Train a Model

```bash
chirpe-train --config configs/bert_config.yaml --data-dir data/synthetic --output-dir outputs/run-name
```

### 3. Make Predictions

```bash
chirpe-predict --model-path outputs/run-name/best_model --input-file path/to/transcript.json --output-dir predictions
```

### 4. Evaluate Model

```bash
chirpe-evaluate --model-path outputs/run-name/best_model --data-dir data/synthetic --output-dir evaluation
```

## ONNX Export and Triton Serving (Classifier Only) 🧩

This repository includes scripts to export a trained classifier to ONNX and serve it with Triton Inference Server.

### 1. Install ONNX and Triton client dependencies

```bash
pip install onnx onnxruntime onnxscript tritonclient[http]
```

> [!NOTE]
> `scripts/onnx/export_triton_onnx.py` supports `transformers` 4.x and 5.x via the modern `dynamo=True` exporter path.

### 2. Export model to Triton repository format

```bash
python scripts/onnx/export_triton_onnx.py \
  --model-dir outputs/test-config-save/best_model \
  --triton-repo outputs/triton_model_repository \
  --model-name chirpe_classifier \
  --version 1 \
  --max-batch-size 32 \
  --opset 18
```

This generates:

- `outputs/triton_model_repository/chirpe_classifier/config.pbtxt`
- `outputs/triton_model_repository/chirpe_classifier/1/model.onnx`
- `outputs/triton_model_repository/chirpe_classifier/1/export_metadata.json`

### 3. Verify ONNX parity against PyTorch

```bash
python scripts/validation/verify_onnx_parity.py \
  --hf-model-dir outputs/test-config-save/best_model \
  --onnx-model-path outputs/triton_model_repository/chirpe_classifier/1/model.onnx \
  --num-samples 16 \
  --atol 1e-4 \
  --rtol 1e-3 \
  --report-file outputs/triton_model_repository/chirpe_classifier/1/parity_report.json
```

### 4. Direct ONNX conversion tutorial notebook

See `notebooks/02_convert_onnx_tutorial.ipynb` for a beginner-friendly step-by-step workflow that covers:

- exporting the classifier to ONNX directly in notebook code
- exporting tokenizer ONNX with ORT Extensions
- merging tokenizer + classifier into a fused string-input ONNX model
- running local fused inference with `onnxruntime`

### 5. Start Triton Inference Server

```bash
docker run --rm --net=host \
  -v ${PWD}/outputs/triton_model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.10-py3 \
  tritonserver --model-repository=/models
```

### 6. CHiRPE fused ONNX quickstart notebook

See `notebooks/03_quickstart_fused_onnx.ipynb` for a CHiRPE pipeline walkthrough:

- raw transcript loading and format check
- CHiRPE preprocessing (`TranscriptPreprocessor`)
- fused ONNX inference on segment summaries
- transcript-level majority/average voting

> [!IMPORTANT]
> Triton serves classifier inference only (tokenized tensor inputs -> logits). Transcript preprocessing,
> segmentation/summarisation, and segment-level voting remain in application code.

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

## Notebooks 📓

- `notebooks/01_quickstart.ipynb`: classic CHiRPE quick start (data -> train -> predict -> explain)
- `notebooks/02_convert_onnx_tutorial.ipynb`: direct tutorial to build fused ONNX (tokenizer + classifier)
- `notebooks/03_quickstart_fused_onnx.ipynb`: CHiRPE quickstart using fused ONNX for segment inference

## Configuration 🧪

Models are configured via YAML files. See `configs/` for examples:

- `bert_config.yaml`: Standard BERT configuration


## Data Format 📝

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


## Explanation Formats 🧠

CHiRPE generates five explanation formats:

1. **Word-level SHAP plots**: Traditional bar plots of top contributing words
2. **Token-level heatmaps**: Colour-coded transcript highlighting
3. **Symptom-level plots**: Domain-wise SHAP aggregations
4. **Sentence-level summaries**: Highest-impact sentences
5. **Narrative summaries**: LLM-generated clinical narratives with quotes

## Testing ✅

Run the test suite:

```bash
pytest tests/ -v --cov=chirpe
```

## Citation 📚

If you use CHiRPE in your research, please cite:

```bibtex
@inproceedings{fong-etal-2026-chirpe,
    title = "{CH}i{RPE}: A Step Towards Real-World Clinical {NLP} with Clinician-Oriented Model Explanations",
    author = "Fong, Stephanie  and
      Wang, Zimu  and
      Oliveira, Guilherme C  and
      Zhao, Xiangyu  and
      Jiang, Yiwen  and
      Liu, Jiahe  and
      Colton, Beau-Luke  and
      Woods, Scott W.  and
      Shenton, Martha  and
      Nelson, Barnaby  and
      Ge, Zongyuan  and
      Dwyer, Dominic",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 2: Short Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-short.46/",
    doi = "10.18653/v1/2026.eacl-short.46",
    pages = "646--658",
    ISBN = "979-8-89176-381-4",
}
```

## License 📄

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
