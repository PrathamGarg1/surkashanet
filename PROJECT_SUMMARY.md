# SurakshaNet v2: Comprehensive Progress & Technical Report

## 1. Executive Overview
SurakshaNet v2 is a privacy-first, edge-deployed multilingual abusive language detection and evidence management system. Engineered as a browser extension, it operates entirely offline using state-of-the-art model quantization and WebAssembly (WASM). It performs real-time interception of toxic content on major social platforms (WhatsApp Web, Twitter/X, Instagram) without routing user data through external APIs, prioritizing absolute privacy and zero network latency.

## 2. Model Architecture & Cloud Pipeline
The core intelligence engine uses a highly optimized, fine-tuned transformer (`paraphrase-multilingual-MiniLM-L12-v2`).

### 2.1. Cloud-Automated Training
- **Infrastructure:** Dockerized pipeline orchestrated natively on **Modal** cloud instances utilizing high-performance A10G GPUs.
- **Hyperparameters:** Fine-tuning was executed with a Learning Rate of `2e-5`, a Batch Size of `16`, over `3` Epochs, implementing a weight decay of `0.01` for optimal regularization.

### 2.2. Edge Deployment & Quantization
- **Weight Reduction:** The native PyTorch model (~471MB) undergoes aggressive **ONNX INT8** static quantization, reducing the footprint to **~40-60MB** without significant accuracy loss.
- **In-Browser Inference:** The `onnxruntime-web` framework is bundled into the Chrome Extension Service Worker (core script size: ~892.52 KB). The WASM-backed model loads directly into device memory, scanning DOM text-nodes natively.

## 3. Data Engineering & Multilingual Context
Handling the code-mixed reality of modern platforms requires specialized, high-quality data. 

*The primary training corpus (approx. 42,000 instances) natively integrates:*

| Language Focus | Academic/Industrial Dataset | Purpose |
| :--- | :--- | :--- |
| **Hindi & Deep Regional Context** | **ShareChat MACD** | Identifying deep regional context and obscure Devanagari profanity directly from a major Indian social network. |
| **Pure English Context** | **Davidson et al. (ICWSM '17)** | High-fidelity English detection to differentiate severe hate speech from general offensive language. |

By aligning these datasets into a strict binary classification schema (*0 = abusive, 1 = non-abusive*), the system achieves exceptional understanding of both native Devanagari slang and standard English context.

## 4. Empirical Evaluation & Metrics
The model was mathematically validated via an isolated execution (`evaluate_model_modal.py`) against a massive 9,207-instance test set, constructed from the ShareChat MACD test splits and a strict 10% hold-out of the English corpus.

**Definitive Metrics (ONNX INT8 Edge Model):**
- **Overall Accuracy:** `87.79%`
- **F1-Macro Score:** `87.39%`

These metrics confirm that the accelerated edge model maintains cloud-scale accuracy thresholds on complex multilingual tasks.

## 5. Phase 1: Ministry-Grade Evidence Management System
Beyond detection, SurakshaNet v2 implements a complete forensic evidence pipeline designed for legal and institutional use.

### 5.1. Cryptographic Storage & Chain of Custody
- **Secure Persistent Storage:** Evidence is saved natively via `chrome.storage.local` with configurable **AES Encryption** (`crypto-js` v4.2.0) and size monitoring up to a 50MB quota.
- **Forensic Tracking:** Implements millisecond-precision timestamps and explicit Chain of Custody logs (e.g., `CREATED`, `SCREENSHOT_ADDED`) for robust audit trails.

### 5.2. Visual & Document Export
- **DOM-to-Image:** Toxic interactions are visually captured retaining context (sender, UI styling, timestamp) using `html2canvas` (v1.4.1), heavily compressed via Base64.
- **Court-Admissible PDF Generation:** Utilizes `jsPDF` (v2.5.1) to dynamically generate professional legal documents featuring unique Report IDs (`SR-XXX-XXX`), severity breakdowns (HIGH/MEDIUM/LOW), category scoring, and embedded screenshot evidence.

## 6. Current Implementation Status
- ✅ **Cloud Infrastructure:** Modal-based automated multi-GPU training, evaluation, and ONNX conversion pipelines are fully operational.
- ✅ **Evidence Dashboard:** Ministry-grade UI equipped with severity badges, cryptographic data resets, and multi-format exports (PDF, JSON, CSV).
- ✅ **Zero-Latency Content Injection:** Content scripts accurately inject warning banners directly into the DOM of WhatsApp Web, Instagram, and Twitter using `MutationObserver` without degrading browser framerates.

## 7. Next Steps toward Production
- **Packaging:** Compile the final extension bundle ensuring Content Security Policies (`script-src 'self' 'wasm-unsafe-eval'`) are compliant with Chrome Web Store V3 manifests.
- **Phase 2 Expansion:** Expand annotation infrastructure for further domain-specific edge cases (e.g., misogyny, child safety) using the validated extraction tools.
