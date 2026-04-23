# String-In ONNX Deployment Runbook

Date: 2026-04-23

## 1) Naming and Versioning Convention

Use the following Triton model names and version layout:

- Classifier-only baseline: `chirpe_<backbone>_baseline`
- Fused string-input model: `chirpe_<backbone>_string`
- Backbones: `bert`, `clinicalbert`, `mentalbert`
- Triton version directory: integer folder (current: `1`)

Canonical fused repository layout:

```text
outputs/string_onnx_fused/
  chirpe_bert_string/
    config.pbtxt
    1/model.onnx
  chirpe_clinicalbert_string/
    config.pbtxt
    1/model.onnx
  chirpe_mentalbert_string/
    config.pbtxt
    1/model.onnx
```

## 2) Export and Build

Run commands in the `chirp` conda environment.

1. Export classifier-only ONNX models (baseline repository).
2. Build fused tokenizer+classifier ONNX models.
3. Generate Triton configs and validate readiness.

Reference commands:

```bash
conda run --no-capture-output -n chirp python scripts/onnx/export_triton_onnx.py --model-dir outputs/string_onnx_checkpoints/bert/best_model --triton-repo outputs/string_onnx_baseline_classifier --model-name chirpe_bert_baseline --version 1 --opset 18
conda run --no-capture-output -n chirp python scripts/onnx/build_fused_string_onnx.py --checkpoint-root outputs/string_onnx_checkpoints --classifier-repo outputs/string_onnx_baseline_classifier --output-repo outputs/string_onnx_fused --report-root reports/fused_smoke
conda run --no-capture-output -n chirp python scripts/validation/integrate_fused_triton_models.py --model-repo outputs/string_onnx_fused --report-file reports/triton_fused_integration/summary.json
```

## 3) Validation Gates

Required checks before promotion:

- Parity: `scripts/validation/verify_fused_triton_parity.py`
- Performance: `scripts/validation/benchmark_triton_fused.py`
- Robustness: `scripts/validation/robustness_fused_triton.py`

Required artifacts:

- `reports/parity_fused/*.json`
- `reports/perf_fused/*.json`
- `reports/robustness_fused/summary.json`

## 4) Deploy

1. Copy vetted fused repository to deployment host.
2. Mount ORT Extensions shared library inside Triton container (path used in `config.pbtxt`: `/opt/ortx/libortextensions.so`).
3. Start Triton with the fused model repository.
4. Confirm:
   - `/v2/health/ready` returns `200`
   - each model `/v2/models/<name>/versions/1/ready` returns `200`
   - infer endpoint returns logits shape `[1, 2]` for valid BYTES payloads.

## 5) Rollback

If parity/perf/robustness regressions are detected:

1. Switch serving target back to classifier-only baseline repository (`outputs/string_onnx_baseline_classifier`).
2. Redeploy Triton with baseline model names (`chirpe_<backbone>_baseline`).
3. Re-run baseline smoke/parity checks to confirm recovery.
4. Open incident note with failing artifact paths and container logs.

## 6) Operator Checklist

- [ ] Confirm model repository paths and version folders are correct.
- [ ] Confirm `config.pbtxt` includes `model_operations.op_library_filename` for ORT Extensions.
- [ ] Confirm parity reports are `passed` for all three backbones.
- [ ] Confirm robustness report is `passed` for all three backbones.
- [ ] Confirm performance report reviewed and tradeoffs accepted.
- [ ] Confirm rollback target and command path are available before cutover.
- [ ] Confirm ML owner sign-off.
- [ ] Confirm infra owner sign-off.
