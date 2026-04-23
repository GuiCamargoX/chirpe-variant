# String-In ONNX Project Checklist (CHiRPE)

## 0) Scope Lock + Prerequisites
- [x] Confirm scope: single-model per run + segment voting (no multi-model ensemble).
- [x] Confirm target artifacts: 3 fused ONNX models (`bert`, `clinicalbert`, `mentalbert`).
- [x] Confirm deployment can load ORT Extensions custom-op library in Triton.
- [x] Pin versions: Triton, ONNX Runtime, onnxruntime-extensions, transformers, torch.
- **Acceptance:** Written scope and version matrix approved.

## 1) Baseline Freeze (Current Classifier-Only Path)
- [x] Export classifier-only ONNX with current script (`scripts/onnx/export_triton_onnx.py`) for each backbone.
- [x] Run parity baseline (`scripts/validation/verify_onnx_parity.py`) and save JSON reports.
- [x] Collect baseline serving metrics (p50/p95 latency, throughput, memory).
- **Acceptance:** Baseline reports exist for all 3 backbones and are reproducible.

## 2) Tokenizer ONNX Feasibility Spike
- [x] Prototype tokenizer graph generation using ORT Extensions (`gen_processing_models` / tutorial flow).
- [x] Validate tokenizer output parity vs HF (`input_ids`, `attention_mask`, `token_type_ids`).
- [x] Test edge strings (empty, unicode, punctuation-heavy, long text).
- **Acceptance:** Tokenizer parity is exact on the test corpus; no unsupported-op errors.

## 3) Build Fused Tokenizer + Classifier ONNX
- [x] Create one fused ONNX per backbone (`string -> logits`).
- [x] Run `onnx.checker.check_model` and ORT session smoke test for string input.
- [x] Save metadata for input/output names and shapes.
- **Acceptance:** Each fused model loads in ORT and returns logits from raw text input.

## 4) Triton Integration
- [x] Add Triton model repository layout for each fused model.
- [x] Generate/verify `config.pbtxt` with `TYPE_STRING` input.
- [x] Register ORT Extensions library in model config (`model_operations.op_library_filename`).
- [x] Start Triton and verify model readiness and infer endpoint behavior.
- **Acceptance:** All 3 models are READY and infer from raw strings successfully.

## 5) End-to-End Parity Validation
- [x] Compare fused Triton path against Python reference pipeline for logits and labels.
- [x] Validate transcript-level final decision parity (segment voting outcome).
- [x] Evaluate across realistic transcript segment data.
- **Acceptance:**
  - [x] Logits within agreed tolerance (`atol`/`rtol`).
  - [x] Label agreement >= 99.5% (target 100%).
  - [x] Transcript-level voting agreement >= 99.5% (target 100%).

## 6) Performance and Reliability Validation
- [x] Benchmark fused vs current path (latency/throughput/memory) at batch sizes 1/4/8.
- [x] Run robustness suite: empty strings, long inputs, malformed payloads, concurrency.
- [x] Document regressions and mitigations (batching, CPU/GPU allocation, model config tuning).
- **Acceptance:** Meets agreed performance budget (or approved tradeoff); no crash/fatal errors.

## 7) Release Readiness
- [x] Finalize model naming/versioning convention for all backbones.
- [x] Document export + deploy + rollback runbook.
- [x] Prepare operator checklist for production rollout.
- [ ] Sign-off from ML and infra owners.
- **Acceptance:** Reproducible end-to-end runbook and sign-off complete.

## Suggested Deliverables
- [x] `reports/baseline/<backbone>.json`
- [x] `reports/parity_fused/<backbone>.json`
- [x] `reports/perf/<backbone>.json`
- [x] `docs/string_in_onnx_runbook.md`
- [x] `docs/version_compatibility_matrix.md`
