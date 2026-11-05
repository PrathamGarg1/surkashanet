# Hinglish data and conversational context

Hinglish version of MACD, the conversational test slices, and the benchmarks behind the gated window of `../extension_v2/`.

| Path | Content |
|---|---|
| `codemix_hinglish/` | MACD comments converted to Romanized, code-mixed Hinglish with a Qwen-based pipeline on Bedrock: `train/val/test.csv`, per-split metadata and conversion reports |
| `macd_hindi_cache/` | MACD Hindi train/val/test splits used for training and evaluation |
| `codemix_eval/` | all evaluation reports: frozen Hinglish test, narrative benchmark (current message / whole chat / gated window), gate benchmarks v2 to v4, Bedrock comparison with Nova Micro and Qwen3 32B, scorecard |
| `scripts/` | conversion (`codemix_convert_bedrock.py`, `run_codemix_full.sh`), Modal training (`train_codemix_modal.py`), evaluation, gate and narrative benchmarks, audits, JS parity test |
| `notebooks/` | Modal training notebook of the Hinglish-aware binary model |
| `ablations/` | reports of two data-cleaning and synthetic-augmentation experiments that did not improve the frozen Hinglish test; their data are not stored (regenerate with `scripts/run_hard_aug_pipeline.sh`) |

Key files:

- `codemix_eval/narrative_kv_benchmark.json`: accuracy on single messages (6,368), split narratives (1,500), benign chats (800) and abuse after small talk (800) for current message only, whole chat and the gated window.
- `codemix_eval/bedrock_llm_report.json`: local INT8 model against Nova Micro and Qwen3 32B on small subsets (20 to 80 examples).
- `codemix_eval/scorecard.json`: accuracy of the shipped INT8 model on the Hindi, Hinglish, English and mixed test sets.

Scripts assume this folder as their root and look for the model files in `../extension_v2/assets/models/custom-macd-model/` (not stored in git). The HinGE data used by `eval_hinge_qwen.py` is a third-party dataset and is not included.
