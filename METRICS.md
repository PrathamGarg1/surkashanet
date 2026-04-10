# SurakshaNet v2 — Benchmark Metrics

Verified benchmarks collected from the live Chrome extension run on WhatsApp Web (April 2026).

---

## 1. Inference Latency

Measured via `console.time('inference')` + `performance.now()` in the Service Worker, directly wrapping the ONNX `classifier()` call.

| Condition | Latency |
|---|---|
| Cold start (first activation) | ~114 ms |
| Warm / steady-state — typical messages | **15 – 45 ms** |
| **Median (warm, real-world messages)** | **~28 ms** |
| Long inputs (code blocks, full URLs) | 100 – 650 ms |

> **Methodology**: `console.time('inference')` placed immediately before and after `await classifier(text)` inside the MV3 Service Worker. Values recorded across 50+ real WhatsApp messages.

---

## 2. RAM Usage

Measured via Chrome Task Manager (`Shift+Esc`) while the extension was actively classifying messages on WhatsApp Web.

| Process | Memory Footprint |
|---|---|
| Extension Service Worker (ONNX WASM heap) | **~89 MB** |
| WhatsApp Web tab (baseline, no extension) | ~604 MB |

> The 89 MB service worker footprint is dominated by the INT8 quantized ONNX model loaded into the WebAssembly linear memory heap. No GPU memory is used — all inference is CPU/WASM.

---

## 3. Model Accuracy (Post-INT8 Quantization)

Evaluated on a held-out, isolated test set combining:
- **ShareChat MACD** (Hindi abuse, ~15k samples)
- **Davidson et al. ICWSM '17** (English hate speech, 10% split)

| Metric | PyTorch (FP32) | ONNX INT8 |
|---|---|---|
| Accuracy | ~88.2% | **87.79%** |
| F1 (Macro) | ~0.882 | **≥ 0.875** |
| Accuracy drop after quantization | — | **< 0.5%** |

> **Note**: The test split was strictly isolated — never seen during training or validation. INT8 quantization applied via `optimum[onnxruntime]` AVX2 dynamic quantization.

---

## Live Detection Spot-Check

Results observed during the live WhatsApp Web run:

| Message | Expected | Result | Confidence |
|---|---|---|---|
| `"तू गधे की औलाद है।"` | Abusive | ✅ HIGH | 98.6% |
| `"तेरे गाल पर तमाचा मारूँगा"` | Abusive | ✅ HIGH | 98.6% |
| `"You are stupid / sexually assaulting..."` | Abusive | ✅ HIGH | 97.8% |
| `"This is terrible and disgusting..."` | Abusive | ✅ HIGH | 98.1% |
| `"how the fuc..."` | Abusive | ✅ MEDIUM | 78.1% |
| `"De kar aana"` | Clean | ✅ Clean | — |
| `"Koi na"` | Clean | ✅ Clean | — |
| `"Ticket to NDLS and metro"` | Clean | ✅ Clean | — |

---

## Environment

| Parameter | Value |
|---|---|
| Model | `paraphrase-multilingual-MiniLM-L12-v2` (fine-tuned) |
| Quantization | INT8 dynamic (AVX2, via `optimum[onnxruntime]`) |
| Runtime | ONNX Runtime Web (WASM, single-threaded) |
| Extension | Chrome MV3 Service Worker |
| Test Platform | macOS, Chrome 124, WhatsApp Web |
