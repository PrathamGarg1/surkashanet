# SurakshaNet — Final Training & Evaluation Report

_Generated: 2026-05-10 10:10:29 UTC_

## Recipe

- Base model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Training data: MACD Hindi train+val (ShareChatAI/MACD) + Davidson 2017 English (90% train split)
- Hyperparameters: epochs=7, lr=3e-05, batch=32, weight_decay=0.01, warmup_ratio=0.1, lr_scheduler=cosine, label_smoothing=0.1, class_weights=balanced, metric_for_best=f1_macro

## Leakage check

| Test set | N | Train overlap | sha256(text)[..16] |
|---|---:|---:|---|
| test_a_macd_hindi | 6728 | 0 | `5b92920ad1df6d1b…` |
| test_b_davidson_10pct | 2478 | 0 | `bba5b8aa21fef89c…` |
| test_c_combined | 9206 | 0 | `9d59795fd1b950cd…` |

All test sets are confirmed disjoint from training data.

## Size comparison

| Stage | File | Size |
|---|---|---:|
| PyTorch FP32 | weights file | 448.8 MB |
| ONNX FP32 | `model.onnx` | 449.1 MB |
| ONNX INT8 (shipped) | `model_quantized.onnx` | 112.8 MB |

INT8 is **4.0×** smaller than the PyTorch FP32 weights.

## Accuracy on held-out test sets

### Test A — MACD Hindi (`hindi_test.csv`, full)

| Model | Accuracy | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| pytorch_fp32 | 84.66% | 84.66% | 84.80% | 84.79% |
| onnx_fp32 | 84.66% | 84.66% | 84.80% | 84.79% |
| onnx_int8 | 84.38% | 84.38% | 84.59% | 84.54% |

### Test B — Davidson 10% holdout (English)

| Model | Accuracy | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| pytorch_fp32 | 96.49% | 93.86% | 93.12% | 94.66% |
| onnx_fp32 | 96.49% | 93.86% | 93.12% | 94.66% |
| onnx_int8 | 96.57% | 94.06% | 92.95% | 95.27% |

### Test C — Combined (A + B)

| Model | Accuracy | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| pytorch_fp32 | 87.84% | 87.46% | 87.13% | 87.98% |
| onnx_fp32 | 87.84% | 87.46% | 87.13% | 87.98% |
| onnx_int8 | 87.68% | 87.32% | 86.96% | 87.94% |

## CPU latency (batch = 1)

| Model | ms / sample |
|---|---:|
| pytorch_fp32 | 35.554 |
| onnx_fp32 | 6.381 |
| onnx_int8 | 5.204 |

## Quantization quality delta (Test A)

| Metric | PyTorch FP32 | ONNX FP32 | ONNX INT8 | INT8 - FP32 (pp) |
|---|---:|---:|---:|---:|
| Accuracy | 84.66% | 84.66% | 84.38% | -0.28 |
| Macro F1 | 84.66% | 84.66% | 84.38% | -0.28 |

## NeurIPS baseline comparison (MACD Test A)

_The published baseline numbers from the MACD paper (Maity et al., NeurIPS 2022) are not configured in `_common.py`. Edit `NEURIPS_BASELINE` to fill in `macd_hindi_test_accuracy` and `macd_hindi_test_f1_macro` and re-run this script to regenerate the table below._

## Deployment artifact

The INT8 ONNX model and tokenizer have been written to `assets/models/custom-macd-model/`. The Chrome extension loads this directly via `@xenova/transformers` from the service worker; nothing is fetched from the network at runtime.
