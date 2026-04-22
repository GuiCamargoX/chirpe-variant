# Triton ORT Extensions Smoke Test Report

Date: 2026-04-22

## Goal
Verify checklist item: Triton can load ONNX Runtime Extensions custom-op library.

## Commands Executed
```bash
conda run --no-capture-output -n chirp docker pull nvcr.io/nvidia/tritonserver:24.10-py3
conda run --no-capture-output -n chirp python scripts/smoke_test_triton_ort_extensions.py --run-docker
```

## Result
- Status: **passed**
- Triton loaded the tokenizer custom-op model successfully (`READY`).

## What Failed First
- Using the Python wheel library (`_extensions_pydll...so`) failed with:
  `undefined symbol: PyInstanceMethod_Type`

## Fix Applied
1. Built ORT Extensions shared library from source with tokenizer-focused options.
2. Used that built library in the smoke test:
   - Host path used: `outputs/onnxruntime-extensions-src/build-shared/lib/libortextensions.so`
3. Re-ran smoke test:
   `conda run --no-capture-output -n chirp python scripts/smoke_test_triton_ort_extensions.py --run-docker --ortx-library-host-path outputs/onnxruntime-extensions-src/build-shared/lib/libortextensions.so`

## Evidence
- `server_ready: true`
- `model_ready: true`
- Triton model status:
  `ortx_tokenizer_smoketest | 1 | READY`

## Notes
- The script now supports `--ortx-library-host-path` and mounts that library into Triton.
- This completes checklist item: deployment can load ORT Extensions custom-op library in Triton.
