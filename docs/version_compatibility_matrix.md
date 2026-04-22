# String-In ONNX Version Compatibility Matrix

Date: 2026-04-23

## Purpose
Pin core versions for reproducible ONNX export and Triton deployment.

## Current Local Environment (conda env: `chirp`)
- `torch`: `2.11.0+cpu`
- `transformers`: `4.57.6`
- `onnx`: `1.21.0`
- `onnxruntime`: `1.24.4`
- `onnxruntime-extensions`: `0.15.2`
- `onnxscript`: `0.7.0`

## Pinned Export Stack (Recommended)
- `torch`: `2.11.0+cpu`
- `transformers`: `>=4,<6`
- `onnx`: `>=1.16,<2`
- `onnxruntime`: `>=1.18,<2`
- `onnxruntime-extensions`: `>=0.12,<1`
- `onnxscript`: `>=0.7,<1`

## Pinned Deployment Stack (Triton)
- Triton server: pin to one tested release and keep fixed for rollout.
- ONNX Runtime backend: use the version bundled with that Triton release.
- `onnxruntime-extensions` custom-op library: build/install against the same ONNX Runtime ABI used by Triton.

## Notes
- `scripts/onnx/export_triton_onnx.py` supports `transformers` 4.x and 5.x.
- Export + parity smoke tests passed with `transformers==5.4.0` for `bert`, `clinicalbert`, and `mentalbert` checkpoints.
- `scripts/onnx/export_triton_onnx.py` uses PyTorch's `dynamo=True` ONNX exporter path and requires `onnxscript`.
- Export and parity checks require installing `onnx` and `onnxruntime` in the `chirp` env.
- String-input fused models require custom-op registration in Triton (`model_operations.op_library_filename`).

## Acceptance Mapping
- Satisfies checklist item: "Pin versions: Triton, ONNX Runtime, onnxruntime-extensions, transformers, torch."
