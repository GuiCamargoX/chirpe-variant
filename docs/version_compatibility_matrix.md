# String-In ONNX Version Compatibility Matrix

Date: 2026-04-22

## Purpose
Pin core versions for reproducible ONNX export and Triton deployment.

## Current Local Environment (conda env: `chirp`)
- `torch`: `2.11.0+cpu`
- `transformers`: `5.4.0`
- `onnx`: missing
- `onnxruntime`: missing
- `onnxruntime-extensions`: missing

## Pinned Export Stack (Recommended)
- `torch`: `2.11.0+cpu`
- `transformers`: `<5` (script requires 4.x)
- `onnx`: `>=1.16,<2`
- `onnxruntime`: `>=1.18,<2`
- `onnxruntime-extensions`: `>=0.12,<1`

## Pinned Deployment Stack (Triton)
- Triton server: pin to one tested release and keep fixed for rollout.
- ONNX Runtime backend: use the version bundled with that Triton release.
- `onnxruntime-extensions` custom-op library: build/install against the same ONNX Runtime ABI used by Triton.

## Notes
- `scripts/export_triton_onnx.py` currently exits if `transformers>=5`.
- Export and parity checks require installing `onnx` and `onnxruntime` in the `chirp` env.
- String-input fused models require custom-op registration in Triton (`model_operations.op_library_filename`).

## Acceptance Mapping
- Satisfies checklist item: "Pin versions: Triton, ONNX Runtime, onnxruntime-extensions, transformers, torch."
