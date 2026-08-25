# SurakshaNet — BTP Presentation **PART A** (Slides **1–15**)

Use this file alone for Notebook LM when the limit is **15 slides per generation**.  
**Part B:** `REPORTS/BTP_PRESENTATION_FINAL_PART_B.md` — slides 16–34 + appendices.

After generating PDFs from Part A and Part B, merge in slide order.  
**Master copy** (full deck in one file): `REPORTS/BTP_PRESENTATION_FINAL.md`.

---

## Numerics rule

- Use **poster / pipeline figure** values everywhere **except** **Slide 14 (quantization table)** — those rows come only from **`REPORTS/final_report.md`**: **−0.28 percentage points**. **Do not** use the poster’s **“<0.11%”** on Slide 14 right panel.
- **Training rows:** Modal logs **42,488**; poster may say **42,487**.

---

## Graphics in Part A

| Slide | Asset |
|------:|-------|
| **7** | `REPORTS/paper2.png` |
| **10** | `REPORTS/loss_curves.png` |
| **13** | `REPORTS/confusion_matrices.png` |
| **15** | `REPORTS/paper1.jpeg` |

*(Slides 16+ assets listed in Part B.)*

---

## Master layout defaults

**Slide size:** 16:9 · **Margins:** ~5% L/R · **Title:** 28–32 pt bold · **Body:** 18–22 pt · **≤5 bullets/slide** · **Figures:** proportional, centered.

---

## Slide 1 — Title

**Layout:** Title upper-middle; subtitle; names bottom-center.

**Title:**

```
SurakshaNet
```

**Subtitle line 1:**

```
Privacy-First, Edge-Deployed Multilingual Abusive Language Detection
```

**Subtitle line 2:**

```
B.Tech Project — End-Semester Evaluation · Computer Science & Engineering · IIT Ropar
```

**Names:**

```
Aamod Jain · Pratham Garg · Dr. Geeta Yadav
```

**Graphic:** Optional IIT Ropar + Chrome logos.

---

## Slide 2 — Agenda

**Title:** `Outline`

**Bullets:**

```
• Mid-semester recap vs end-semester objectives
• Problem statement & contributions
• End-to-end pipeline (datasets → edge deployment → inference → forensics)
• Training setup & evaluation integrity (held-out tests, zero leakage)
• Results — accuracy vs MACD paper; multilingual headline; compression & quantization
• Runtime performance & live deployment
• Technical depth — validation design, Modal logs, export chain, browser call graph
• MobiSys ’26 poster acceptance
• Limitations, conclusion, Q&A
```

---

## Slide 3 — Mid-semester recap

**Title:** `Mid-Semester Recap`

**Bullets:**

```
• Chrome extension (Manifest V3): monitored DOM on WhatsApp Web / similar web apps
• Inline alert banners injected next to toxic message nodes
• Toxicity model: generic off-the-shelf classifier (not trained on MACD)
• Direction set: privacy-first, on-device inference; groundwork for evidence logging
```

---

## Slide 4 — End-semester objectives vs delivered

**Title:** `End-Semester Objectives → Delivered`

**Headers:** `Objective` | `Delivered`

| Objective | Delivered |
|-----------|-----------|
| Train on MACD-aligned Hindi + English data | MACD Hindi + Davidson English combined; **42,488** training rows (Modal logs); deterministic Davidson 90/10 split |
| Replace generic toxicity detector | Fine-tuned **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** |
| Browser-deployable model | **INT8 ONNX** + **onnxruntime-web** / WASM in MV3 **service worker**; shipped artifact **~113 MB** |
| Rigorous evaluation | Held-out **Test A / B / C**; **zero** train↔test text overlap (hash checks) |
| Academic dissemination | **MobiSys ’26 Posters** acceptance (ACM) |

**Graphic:** Horizontal timeline — Mid-sem prototype → MACD MiniLM + benchmarks → INT8 + MobiSys.

---

## Slide 5 — Problem & motivation

**Title:** `Problem & Motivation`

**Bullets:**

```
• Gender-based harassment is prevalent in code-mixed Hinglish / Hindi online contexts
• Cloud moderation APIs: privacy exposure, latency, user content leaves the device
• Requirement: real-time detection inside the browser without uploading raw chats
• Goal: multilingual abuse classification plus locally auditable, forensic-grade incident artifacts
```

**Graphic:** Cloud + slash → laptop + padlock (~30% width).

---

## Slide 6 — Contributions

**Title:** `Contributions`

**Bullets:**

```
(1) Chrome extension prototype for in-browser abuse detection on live web messaging UIs
(2) Training pipeline: ShareChat MACD (Hindi/Hinglish) + Davidson English → unified binary labels
(3) Aggressive compression (INT8 ONNX) + WASM inference + forensic evidence & reporting stack
```

---

## Slide 7 — Full pipeline figure (`paper2.png`)

**Title:** `SurakshaNet — End-to-End Pipeline`

**Subtitle (~18 pt):**

```
Multilingual data → MiniLM training & INT8 quantization → Chrome extension → live classification → forensic reporting
```

**Main graphic:** **`REPORTS/paper2.png`** — ~92% width, ~70–75% slide height; do not crop five stages.

**Footer caption (14–16 pt, grey):**

```
Figure: Five-stage pipeline — dataset fusion (MACD + Davidson, 42k+ instances) → Modal A10G training & 76% size reduction (471 MB → 113 MB) → MV3 + WASM (~89 MB RAM, 28 ms median) → MutationObserver + SHIELD tiers → AES-256 & court-admissible PDF workflow.
```

**Speaker note:** If figure says “Interference”, say **“Inference”** aloud.

---

## Slide 8 — Datasets & splits

**Title:** `Datasets & Evaluation Splits`

**Table:**

| Source | Split used in project | Held-out test size |
|--------|----------------------|-------------------|
| **ShareChat MACD** | `hindi_train` / `hindi_val` / **`hindi_test`** | **6,728** |
| **Davidson et al. 2017** | 90% train · **10% deterministic holdout** (`seed = 42`) | **2,478** |
| **Combined stress test** | Tests concatenated | **9,206** |

**Footnote:**

```
Training uses MACD hindi_val for checkpoint selection only. hindi_test is never used during training. Davidson 3-class labels mapped to MACD-binary abusive / non-abusive.
```

---

## Slide 9 — Evaluation integrity

**Title:** `Evaluation Integrity — Train/Test Leakage`

**Table:**

| Test set | **N** | Rows overlapping training texts |
|----------|------:|--------------------------------:|
| MACD `hindi_test` | **6,728** | **0** |
| Davidson 10% holdout | **2,478** | **0** |
| Combined | **9,206** | **0** |

**Line below:**

```
Every test row’s text is hashed and checked against the full training-text set before any metric is computed.
```

---

## Slide 10 — Training configuration + loss curves

**Title:** `Training Configuration`

**Left (~45%):**

```
• Base model: paraphrase-multilingual-MiniLM-L12-v2
• Modal **A10G** GPU training
• **7** epochs · batch **32** · lr **3×10⁻⁵** · cosine schedule · **10%** warmup
• Label smoothing **0.1** · **balanced class weights** · checkpoint selected by **macro-F1** on validation
```

**Right (~52%):** **`REPORTS/loss_curves.png`** — ~65–70% slide height, top-aligned.

---

## Slide 11 — Hindi-only vs MACD paper

**Title:** `Hindi-Only Held-Out Test — Comparison to Published MACD Baseline`

**Table:**

| Metric | Value |
|--------|------:|
| **MACD paper** — XLM-R, Hindi test | **86.34%** accuracy |
| **SurakshaNet** — Test A (`hindi_test`), FP32 | **84.66%** accuracy |
| **Gap (poster)** | **−1.68%** |

**Note box:**

```
Interpretation: −1.68% = −1.68 percentage points (86.34 − 84.66).
Design: MiniLM fits browser WASM constraints; XLM-R-scale models impractical for edge RAM (poster rationale).
```

**Graphic:** Two bars **86.34** vs **84.66**.

---

## Slide 12 — Multilingual headline (poster)

**Title:** `Hindi + English Combined Setting (Poster Headline)`

**Large centre:**

```
88.01%
```

**Below:**

```
Accuracy — SurakshaNet, Hindi + English (poster reported)
```

**Small note:**

```
MACD paper does not report a comparable English-inclusive single-number baseline for this combined setting (poster wording).
```

---

## Slide 13 — Confusion matrices

**Title:** `Confusion Matrices — Held-Out Tests`

**Subtitle:**

```
Rows = true label · Columns = predicted · Classes: abusive / non-abusive · Models: PT / ONNX FP32 / INT8 · Tests: A · B · C
```

**Graphic:** **`REPORTS/confusion_matrices.png`** — full width, ~78% height.

---

## Slide 14 — Compression & quantization

**Title:** `Compression & Quantization Impact`

**Two equal panels.**

**Left — heading:** `Model size (poster / pipeline figure)`

```
471 MB  →  113 MB
76% reduction
```

Mini horizontal bars + **76% SIZE REDUCTION**.

**Right — heading:** `Accuracy impact of INT8 quantization — Test A only (measured, REPORTS)`

| | Accuracy | Macro F1 |
|---|----------|----------|
| **Before INT8 (FP32)** | **84.66%** | **84.66%** |
| **After INT8** | **84.38%** | **84.38%** |
| **Change** | **−0.28 percentage points** | **−0.28 percentage points** |

**Critical note:** `Use ONLY −0.28 pp — do not quote poster “<0.11%” here.`

---

## Slide 15 — Live deployment (`paper1.jpeg`)

**Title:** `Live Deployment — WhatsApp Web (Screenshot)`

**Graphic:** **`REPORTS/paper1.jpeg`** — full width, ~80% height.

**Caption:**

```
English abusive phrase + Hinglish threat → CRITICAL THREAT / HIGH · confidence (e.g. 95%) · benign Hindi unflagged · Block · Report · Save Evidence
```

---

_End of Part A (15 slides). Continue with `BTP_PRESENTATION_FINAL_PART_B.md`._
