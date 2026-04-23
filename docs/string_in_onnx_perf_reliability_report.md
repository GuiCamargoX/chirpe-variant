# String-In ONNX Performance and Reliability Report

Date: 2026-04-23

## Scope

This report captures Section 6 validation for fused Triton string-input models:

- `chirpe_bert_string`
- `chirpe_clinicalbert_string`
- `chirpe_mentalbert_string`

Artifacts:

- `reports/perf_fused/bert.json`
- `reports/perf_fused/clinicalbert.json`
- `reports/perf_fused/mentalbert.json`
- `reports/robustness_fused/summary.json`

## Performance Summary

Benchmark mode used for fused models is concurrency levels `1/4/8` (reported as batch sizes) with one string per infer request.

Reason: the current ORT Extensions tokenizer op accepts single-string infer requests in this deployment graph; multi-string infer tensors are not used for fused serving benchmarks.

Highlights from `reports/perf_fused/*.json`:

- Batch/concurrency `1`: fused latency and throughput are similar to or better than baseline.
- Batch/concurrency `4` and `8`: fused throughput is lower than classifier-only baseline due to tokenizer work executed per request.
- Peak container memory is stable around ~4.69-4.75 GiB across runs.

## Reliability Summary

Robustness suite passed for all three fused models (`reports/robustness_fused/summary.json`):

- Valid payloads: empty, whitespace, punctuation-heavy, unicode, and long text all return `200` with `[1, 2]` logits.
- Malformed payloads: missing inputs, wrong input name, wrong datatype, and shape mismatch are rejected with expected error responses.
- Concurrency check: 8 workers x 20 requests each completed with zero failed requests for all models.

## Regressions and Mitigations

1. **Tokenizer batching behavior**
   - Observation: fused serving currently runs one text per infer request in benchmark mode.
   - Impact: lower throughput than classifier-only baseline at higher load levels.
   - Mitigation: use request concurrency for scaling; keep single-string infer payloads in production config.

2. **Long-input stability (fixed)**
   - Observation: very long strings previously caused runtime shape errors in fused execution.
   - Fix: fused adapter now clips tokenizer outputs to model `max_length` from `chirpe_config.json` during export.
   - Result: long input case passes in robustness validation.

3. **GPU availability**
   - Observation: current validation environment runs Triton on CPU (no NVIDIA driver detected in container logs).
   - Mitigation: for production GPU deployments, repeat the same scripts on target GPU nodes and compare against this CPU baseline.
