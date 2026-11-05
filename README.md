# SurakshaNet

Privacy-first, on-device detection of abusive messages in English, Hindi and Hinglish, with a chat-thread benchmark for women and child safety.

SurakshaNet runs a quantized multilingual MiniLM inside a Chrome extension (ONNX Runtime Web / Transformers.js, WebAssembly), so message text never leaves the browser. This repository holds the system from the accepted MobiSys '26 poster (v1), its extension (v2: Hinglish-aware model and a gated conversational window), and a new five-class, context-thread dataset and the models trained on it.

- Poster: *SurakshaNet: Privacy-First, Edge-Deployed Multilingual Abusive Language Detection*, MobiSys Companion '26, [doi:10.1145/3812835.3814864](https://doi.org/10.1145/3812835.3814864) (`paper/accepted_poster_MobiSys26.pdf`)
- Extended report (this repository's paper): `paper/SurakshaNet_V2.pdf`, source in `paper/`

## Repository layout

| Folder | What it contains |
|---|---|
| [`paper/`](paper) | LaTeX source of the extended report, TikZ figures, bibliography, PDFs |
| [`poster_v1/`](poster_v1) | The system of the accepted poster: MV3 extension (`src/`), Modal training pipeline (`training/`), v1 reports (`REPORTS/`) |
| [`extension_v2/`](extension_v2) | Extension with the Hinglish-aware model and the gated conversational window (`sw.js`, `narrative_cache.js`) |
| [`hinglish_context/`](hinglish_context) | MACD-to-Hinglish dataset, conversational test slices, gate and cloud-LLM benchmarks, training scripts |
| [`thread_benchmark/`](thread_benchmark) | **New:** 105,000-thread five-class dataset, generation pipeline, thread models, INT8 export, evaluation |

## What is new

1. **Five-class thread dataset (105,000 threads).** Classes: benign, friendly sarcasm, hostile abuse, sexual harassment, predatory grooming. Three scripts (English, Hindi in Devanagari, Hinglish), thread lengths 3 / 5 / 7 / 10, 17 columns per record. The label belongs to the *last message given the earlier ones*; about half of the sarcasm and hostile threads end in an identical line, so only the context separates them. See [`thread_benchmark/dataset/DATASET_CARD.md`](thread_benchmark/dataset/DATASET_CARD.md).
2. **Generation method.** A seeded scenario sampler (31 subtypes, 40 topics, named speakers, audience, target gender and age) feeds Qwen3-32B on Amazon Bedrock. Rule-based validators check turn count, refusals, script, profanity and explicit words outside the hostile class, and sexualised words in grooming threads; threads are de-duplicated, and pair lines are split by line. Code: [`thread_benchmark/scripts/`](thread_benchmark/scripts).
3. **Thread-aware models.** A fine-tuned MiniLM that reads the last message plus earlier turns, a last-message-only control, and a frozen MiniLM + GRU variant (198,021 trained weights, JavaScript runtime with parity test). Dynamic INT8 export of both routes.
4. **Gated conversational window for the extension.** The window of up to 8 turns / 96 tokens is scored only when the new turn is short and history exists; the final score is the maximum of the message and window scores.

## Headline results

Five-class models, held-out synthetic test set (10,563 threads):

| Model | Size | Accuracy | Macro-F1 |
|---|---|---|---|
| Thread model, INT8 | 118.2 MB | 97.88% | 0.979 |
| Thread model, FP32 | 470.8 MB | 97.96% | 0.980 |
| Frozen MiniLM + GRU, INT8 | 118 MB + 0.8 MB | 94.91% | 0.950 |
| Last message only | 470.8 MB | 83.18% | 0.828 |

Extension model (binary, Hinglish-aware): 470.6 MB → 118.4 MB with INT8 (75% smaller) and accuracy unchanged within 0.15 points (Hindi 85.1%, Hinglish 84.4%, English 96.7%, mixed 86.5%). On the frozen Hinglish conversational test, the gated window raises split-narrative detection from 28.9% (current message only) to 76.1%.

These five-class scores describe the synthetic distribution. A first zero-shot check on real conversation data (CAD, ConvAbuse, PRISM) gave AUROC 0.47 to 0.74 ([`thread_benchmark/external_eval/RESULTS.md`](thread_benchmark/external_eval/RESULTS.md)); the dataset is a controlled benchmark and training material to be mixed with real data, not evidence of real-world accuracy.

Every number is backed by a report file: `thread_benchmark/quant/int8_report.json`, `thread_benchmark/gru/out/benchmark_report.json`, `hinglish_context/codemix_eval/*.json`, `poster_v1/REPORTS/`. Details in [`thread_benchmark/RESULTS.md`](thread_benchmark/RESULTS.md).

## Quick start

```bash
pip install -r requirements.txt

# check the dataset (105,000 threads, 17 columns)
python thread_benchmark/scripts/flatten_dataset.py        # writes dataset/all/{train,val,test}.jsonl

# train, quantize, evaluate (needs a GPU or patience; see thread_benchmark/README.md)
python thread_benchmark/scripts/train_v2.py --mode thread --epochs 2
python thread_benchmark/gru/embed_lines.py && python thread_benchmark/gru/train_gru.py
python thread_benchmark/quant/export_quantize.py && python thread_benchmark/quant/eval_quant.py

# JavaScript parity tests
node thread_benchmark/gru/parity_test.mjs
node hinglish_context/scripts/test_narrative_cache.mjs

# paper (acmart; needs TeX Live)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Model weights (`*.onnx`, about 118 MB each) are not stored in git. The extensions expect them under `extension_v2/assets/models/custom-macd-model/` (v2) and `poster_v1/assets/models/custom-macd-model/` (v1); `poster_v1/training/` and `thread_benchmark/quant/export_quantize.py` regenerate them. Building the extensions: `npm ci && npm run build` inside `extension_v2/` or `poster_v1/`, then load `dist/` as an unpacked extension.

## Notes on data and use

- The five-class dataset is synthetic (LLM-generated, checked by automatic rules, not by human annotators). It contains abusive, harassing and grooming-style text but no real people. Grooming threads use only non-sexual cues (secrecy, isolation, age probing). Use it only to build protective classifiers.
- `poster_v1/` and `hinglish_context/` are built on MACD (Gupta et al., NeurIPS 2022 Datasets and Benchmarks) and Davidson et al. (ICWSM 2017); `hinglish_context/codemix_hinglish/` is a machine-converted derivative of MACD. Check the original licences before redistributing.
- Parental login and Child Protection Mode of the poster system live on the branch `aamod/login-signup` and are not part of `main`.

## Authors

Pratham Garg, Aamod Jain, Geeta Yadav (IIT Ropar). Contact: 2023csb1147@iitrpr.ac.in
