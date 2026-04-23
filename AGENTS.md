IMPORTANT: Run any code/commands only in the conda environment `chirp`.
IMPORTANT: When invoking commands via conda, always use `conda run --no-capture-output -n chirp ...`.

# AGENTS.md

## Setup
- Install in editable mode with dev tools: `pip install -e ".[dev]"`. Python `>=3.9` is required.
- `pyproject.toml` is the main source of truth for tooling in this repo; there is no CI, pre-commit config, or other repo-local agent instruction file.
- `data/` and `outputs/` are local artifacts and are gitignored.

## Commands
- Generate usable synthetic splits before training: `python scripts/dataprep/generate_synthetic_data.py --n-participants 100 --output-dir data/synthetic --split`
- Main training entrypoint: `chirpe-train --config configs/bert_config.yaml --data-dir data/synthetic --output-dir outputs/run-name`
- Main prediction entrypoint: `chirpe-predict --model-path outputs/run-name/best_model --input-file path/to/transcript.json --output-dir predictions`
- Main evaluation entrypoint: `chirpe-evaluate --model-path outputs/run-name/best_model --data-dir data/synthetic --output-dir evaluation`
- Fast config-driven training path: `python scripts/train/train_from_config.py --config configs/ultra_quick_config.yaml --data-dir data/synthetic --output-dir outputs/quick-run`
- Export classifier-only ONNX to Triton layout: `python scripts/onnx/export_triton_onnx.py --model-dir outputs/run-name/best_model --triton-repo outputs/triton_model_repository --model-name chirpe_classifier --version 1 --max-batch-size 32 --opset 18`
- Build fused string-input ONNX models: `python scripts/onnx/build_fused_string_onnx.py --checkpoint-root outputs/string_onnx_checkpoints --classifier-repo outputs/string_onnx_baseline_classifier --output-repo outputs/string_onnx_fused --report-root reports/fused_smoke`
- Verify HF vs ONNX logit parity: `python scripts/validation/verify_onnx_parity.py --hf-model-dir outputs/run-name/best_model --onnx-model-path outputs/triton_model_repository/chirpe_classifier/1/model.onnx --report-file reports/baseline/bert.json`
- Smoke test Triton ORT Extensions loading: `python scripts/validation/smoke_test_triton_ort_extensions.py --run-docker`
- Benchmark Triton classifier serving baseline: `python scripts/validation/benchmark_triton_classifier_baseline.py --model-repo outputs/triton_model_repository --model-name chirpe_classifier --hf-model-dir outputs/run-name/best_model --report-file reports/perf/bert.json`
- Validate fused string-input Triton models end-to-end: `python scripts/validation/integrate_fused_triton_models.py --model-repo outputs/string_onnx_fused --report-file reports/triton_fused_integration/summary.json`
- Compare fused Triton logits/labels vs Python reference: `python scripts/validation/verify_fused_triton_parity.py --model-repo outputs/string_onnx_fused --checkpoints-root outputs/string_onnx_checkpoints --data-file data/synthetic/test.json --report-root reports/parity_fused`
- Benchmark fused Triton string-input serving: `python scripts/validation/benchmark_triton_fused.py --model-repo outputs/string_onnx_fused --model-name chirpe_bert_string --baseline-report reports/perf/bert.json --report-file reports/perf_fused/bert.json`
- Run fused Triton robustness suite: `python scripts/validation/robustness_fused_triton.py --model-repo outputs/string_onnx_fused --report-file reports/robustness_fused/summary.json`
- Run tokenizer ONNX feasibility spike: `python scripts/experiments/tokenizer_onnx_feasibility_spike.py`
- Do not rely on `python -m chirpe.cli` for normal use; `src/chirpe/cli.py` defaults to `train_cli()` when run directly.

## Important Quirks
- `chirpe-train` and `chirpe-evaluate` expect `train.json`, `val.json`, and `test.json` inside `--data-dir`. Without `--split`, synthetic generation only writes `all_data.json`.
- `README.md`'s evaluation example is stale. `chirpe-evaluate` does not take `--predictions`; use `--model-path` and `--data-dir`.
- The two training flows save different metadata:
  - `chirpe-train` writes `best_model/chirpe_config.json`
- `scripts/train/train_from_config.py` writes `final_model/config_used.json`
- `chirpe-predict` only auto-loads `chirpe_config.json` from the model dir or its parent. Models from `scripts/train/train_from_config.py` therefore run with defaults unless that config is copied or renamed.
- `tests/test_classifier.py` is not a cheap unit test: it instantiates real Hugging Face models such as `bert-base-uncased`.
- API-backed LLM paths in `src/chirpe/data/summarizer.py` and `src/chirpe/explanations/narrative.py` still use `openai.ChatCompletion.create` even though the repo depends on `openai>=1.0.0`; treat those code paths as likely stale before relying on them.
- Avoid `import chirpe` in lightweight scripts or tests; `src/chirpe/__init__.py` eagerly imports the preprocessor, classifier, and SHAP explainer.

## Architecture
- `src/chirpe/data/`: JSON loading, PSYCHS-domain segmentation, summarization, preprocessing.
- `src/chirpe/models/`: `CHRClassifier` and the Hugging Face `Trainer` wrapper in `ModelTrainer`.
- `src/chirpe/explanations/`: SHAP-based explanation generation and narrative generation.
- `src/chirpe/cli.py`: installed `chirpe-train`, `chirpe-predict`, and `chirpe-evaluate` entrypoints.
- Real training/eval flow is transcript -> `TranscriptPreprocessor` -> segment summaries -> `CHRPDataset` -> BERT classifier.
- Segmentation matches interviewer utterances to PSYCHS domains with fuzzywuzzy in `src/chirpe/data/segmentation.py`; thresholds are configured as `0-1` floats but converted internally to fuzzywuzzy's `0-100` scale.
- Training and evaluation flatten transcripts into segment-level samples and cap segments per transcript (`preprocessing.max_segments_per_transcript`, default `3` in the CLI flow).

## Verification
- Full test run: `pytest`
- Faster focused checks: `pytest tests/test_dataset.py tests/test_segmentation.py`
- Tooling configured in `pyproject.toml`: `black`, `isort`, `mypy`, `pytest`
