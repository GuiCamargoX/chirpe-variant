# String-In ONNX Blocker Resolution

Date: 2026-04-23

## Goal
Resolve blockers before starting checklist Section 1 (baseline freeze).

## Blocker 1: `transformers` 5.x export compatibility
- Validation run with `transformers==5.4.0` failed in `scripts/onnx/export_triton_onnx.py` during `torch.onnx.export`.
- Failure occurred in DistilBERT masking path (`IndexError` in `transformers/masking_utils.py`).
- Resolution: downgraded to `transformers==4.57.6` in conda env `chirp`.
- Post-fix smoke test passed:
  - `outputs/triton_model_repository_tf4_smoke/chirpe_classifier_tf4_smoke/1/model.onnx`

## Blocker 2: Missing quick checkpoints for the three target backbones
- Generated synthetic split dataset for fast fine-tuning:
  - `data/synthetic_string_onnx_quick/{train,val,test}.json`
- Added quick training configs:
  - `configs/string_onnx_quick_bert.yaml`
  - `configs/string_onnx_quick_clinicalbert.yaml`
  - `configs/string_onnx_quick_mentalbert.yaml`
- Trained checkpoints:
  - `outputs/string_onnx_checkpoints/bert/best_model`
  - `outputs/string_onnx_checkpoints/clinicalbert/best_model`
  - `outputs/string_onnx_checkpoints/mentalbert/best_model`

## Blocker 3: MentalBERT upstream model access
- Attempt to load `mental/mental-bert-base-uncased` failed with `403` (gated repo access).
- Temporary unblock: quick mentalbert config uses `bert-base-uncased` with `type: mentalbert`.
- This keeps artifact naming/paths aligned for pipeline work while access is pending.

## Notes for Section 1
- Export/parity can proceed immediately for all three checkpoint directories above.
- Smoke export + parity succeeded for all three quick checkpoints under `transformers==4.57.6`:
  - `outputs/string_onnx_blocker_smoke/chirpe_bert_smoke/1/parity_report.json`
  - `outputs/string_onnx_blocker_smoke/chirpe_clinicalbert_smoke/1/parity_report.json`
  - `outputs/string_onnx_blocker_smoke/chirpe_mentalbert_smoke/1/parity_report.json`
- For production-grade `mentalbert` parity, replace the fallback checkpoint once gated access is granted.
