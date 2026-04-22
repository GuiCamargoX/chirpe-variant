# String-In ONNX Project Checklist (CHiRPE)

## 0) Scope Lock + Prerequisites
- [ ] Confirm scope: single-model per run + segment voting (no multi-model ensemble).
- [ ] Confirm target artifacts: 3 fused ONNX models (`bert`, `clinicalbert`, `mentalbert`).
- [ ] Confirm deployment can load ORT Extensions custom-op library in Triton.
- [ ] Pin versions: Triton, ONNX Runtime, onnxruntime-extensions, transformers, torch.
- **Acceptance:** Written scope and version matrix approved.

## 1) Baseline Freeze (Current Classifier-Only Path)
- [ ] Export classifier-only ONNX with current script (`scripts/export_triton_onnx.py`) for each backbone.
- [ ] Run parity baseline (`scripts/verify_onnx_parity.py`) and save JSON reports.
- [ ] Collect baseline serving metrics (p50/p95 latency, throughput, memory).
- **Acceptance:** Baseline reports exist for all 3 backbones and are reproducible.

## 2) Tokenizer ONNX Feasibility Spike
- [ ] Prototype tokenizer graph generation using ORT Extensions (`gen_processing_models` / tutorial flow).
- [ ] Validate tokenizer output parity vs HF (`input_ids`, `attention_mask`, `token_type_ids`).
- [ ] Test edge strings (empty, unicode, punctuation-heavy, long text).
- **Acceptance:** Tokenizer parity is exact on the test corpus; no unsupported-op errors.

## 3) Build Fused Tokenizer + Classifier ONNX
- [ ] Create one fused ONNX per backbone (`string -> logits`).
- [ ] Run `onnx.checker.check_model` and ORT session smoke test for string input.
- [ ] Save metadata for input/output names and shapes.
- **Acceptance:** Each fused model loads in ORT and returns logits from raw text input.

## 4) Triton Integration
- [ ] Add Triton model repository layout for each fused model.
- [ ] Generate/verify `config.pbtxt` with `TYPE_STRING` input.
- [ ] Register ORT Extensions library in model config (`model_operations.op_library_filename`).
- [ ] Start Triton and verify model readiness and infer endpoint behavior.
- **Acceptance:** All 3 models are READY and infer from raw strings successfully.

## 5) End-to-End Parity Validation
- [ ] Compare fused Triton path against Python reference pipeline for logits and labels.
- [ ] Validate transcript-level final decision parity (segment voting outcome).
- [ ] Evaluate across realistic transcript segment data.
- **Acceptance:**
  - [ ] Logits within agreed tolerance (`atol`/`rtol`).
  - [ ] Label agreement >= 99.5% (target 100%).
  - [ ] Transcript-level voting agreement >= 99.5% (target 100%).

## 6) Performance and Reliability Validation
- [ ] Benchmark fused vs current path (latency/throughput/memory) at batch sizes 1/4/8.
- [ ] Run robustness suite: empty strings, long inputs, malformed payloads, concurrency.
- [ ] Document regressions and mitigations (batching, CPU/GPU allocation, model config tuning).
- **Acceptance:** Meets agreed performance budget (or approved tradeoff); no crash/fatal errors.

## 7) Release Readiness
- [ ] Finalize model naming/versioning convention for all backbones.
- [ ] Document export + deploy + rollback runbook.
- [ ] Prepare operator checklist for production rollout.
- [ ] Sign-off from ML and infra owners.
- **Acceptance:** Reproducible end-to-end runbook and sign-off complete.

## Suggested Deliverables
- [ ] `reports/baseline/<backbone>.json`
- [ ] `reports/parity_fused/<backbone>.json`
- [ ] `reports/perf/<backbone>.json`
- [ ] `docs/string_in_onnx_runbook.md`
- [ ] `docs/version_compatibility_matrix.md`
