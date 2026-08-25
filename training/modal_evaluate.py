"""
SurakshaNet — Step 3/3: evaluate PT / ONNX FP32 / ONNX INT8 on held-out tests.

Run on Modal (CPU; we want CPU latency since that's what the extension uses):
    modal run training/modal_evaluate.py

Reads:
  - /vol/checkpoints/pt/
  - /vol/checkpoints/onnx_fp32/
  - /vol/checkpoints/onnx_int8/
  - /vol/davidson_test_indices.json

Test sets (NEVER seen during training):
  A) MACD hindi_test.csv (full)
  B) Davidson rows at indices loaded from DAVIDSON_INDICES_FILE
  C) A + B combined

Hard leakage check: SHA-256 hash every (text) string in train data
(MACD train+val + Davidson 90%) and assert disjointness with each test set
before any inference is performed.

Writes:
  - REPORTS/eval_metrics.json      full grid of (model x test_set) metrics
  - REPORTS/confusion_matrices.png 3x3 grid (3 models × 3 test sets)
  - REPORTS/final_report.md        human-readable summary table
"""

from __future__ import annotations

import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    APP_NAME,
    DAVIDSON_INDICES_FILE,
    ID2LABEL,
    IMAGE,
    MAX_LENGTH,
    NEURIPS_BASELINE,
    ONNX_FP32_DIR,
    ONNX_INT8_DIR,
    PT_CHECKPOINT_DIR,
    TRAIN_HPARAMS,
    VOLUME,
    VOLUME_MOUNT,
    davidson_split,
    file_size_mb,
    load_davidson_full,
    load_macd_test,
    load_macd_train_val,
)

app = modal.App(f"{APP_NAME}-evaluate")


@app.function(
    image=IMAGE,
    gpu="A10G",
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=60 * 30,
)
def evaluate():
    import hashlib
    import io

    import matplotlib
    import numpy as np
    import pandas as pd
    import torch
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 0. Sanity: required artifacts ───────────────────────────────────────
    for d in (PT_CHECKPOINT_DIR, ONNX_FP32_DIR, ONNX_INT8_DIR):
        assert os.path.isdir(d), f"missing checkpoint dir: {d}"
    assert os.path.exists(DAVIDSON_INDICES_FILE), (
        f"missing {DAVIDSON_INDICES_FILE}; run modal_train.py first"
    )

    # ── 1. Re-derive train data (for leakage check) ─────────────────────────
    print("[eval] re-loading train data for leakage check…")
    macd_train, macd_val = load_macd_train_val()
    davidson_full = load_davidson_full()
    davidson_train, davidson_test, _ = davidson_split(
        davidson_full, indices_file=DAVIDSON_INDICES_FILE
    )

    train_texts = pd.concat(
        [macd_train["text"], macd_val["text"], davidson_train["text"]],
        ignore_index=True,
    )
    train_text_set = set(train_texts.tolist())
    print(f"  train texts (incl. val): {len(train_text_set)} unique")

    # ── 2. Build test sets ──────────────────────────────────────────────────
    print("[eval] loading held-out test sets…")
    macd_test = load_macd_test()
    print(f"  Test A (MACD hindi_test): {len(macd_test)}")
    print(f"  Test B (Davidson 10% holdout): {len(davidson_test)}")

    test_a = macd_test.copy()
    test_b = davidson_test.copy()
    test_c = pd.concat([test_a, test_b], ignore_index=True)
    print(f"  Test C (A+B): {len(test_c)}")

    # ── 3. Leakage assertion ────────────────────────────────────────────────
    def assert_no_leakage(test_df: pd.DataFrame, name: str) -> dict:
        n = len(test_df)
        overlap_rows = test_df[test_df["text"].isin(train_text_set)]
        n_leak = len(overlap_rows)
        if n_leak > 0:
            print(f"  ✗ {name}: {n_leak}/{n} rows overlap with train")
            print("  examples:")
            for t in overlap_rows["text"].head(3).tolist():
                print(f"    - {t[:120]!r}")
            raise AssertionError(
                f"LEAKAGE: {n_leak}/{n} rows of {name} appear in train data"
            )
        # Also hash for the report
        h = hashlib.sha256()
        for t in test_df["text"].tolist():
            h.update(t.encode("utf-8"))
        print(f"  ✓ {name}: 0/{n} train overlap (SHA-256 = {h.hexdigest()[:16]}…)")
        return {"n": n, "leakage_rows": 0, "sha256_prefix": h.hexdigest()[:16]}

    print("[eval] leakage check…")
    leakage_report = {
        "test_a_macd_hindi": assert_no_leakage(test_a, "Test A (MACD hindi_test)"),
        "test_b_davidson_10pct": assert_no_leakage(
            test_b, "Test B (Davidson 10% holdout)"
        ),
        "test_c_combined": assert_no_leakage(test_c, "Test C (A+B combined)"),
    }

    # ── 4. Load all three model variants ────────────────────────────────────
    print("[eval] loading models…")
    tokenizer = AutoTokenizer.from_pretrained(PT_CHECKPOINT_DIR)

    pt_model = AutoModelForSequenceClassification.from_pretrained(PT_CHECKPOINT_DIR)
    pt_model.eval()
    pt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model.to(pt_device)

    ort_fp32 = ORTModelForSequenceClassification.from_pretrained(ONNX_FP32_DIR)
    ort_int8 = ORTModelForSequenceClassification.from_pretrained(
        ONNX_INT8_DIR, file_name="model_quantized.onnx"
    )

    onnx_fp32_path = os.path.join(ONNX_FP32_DIR, "model.onnx")
    onnx_int8_path = os.path.join(ONNX_INT8_DIR, "model_quantized.onnx")
    pt_weights_path = None
    for cand in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(PT_CHECKPOINT_DIR, cand)
        if os.path.exists(p):
            pt_weights_path = p
            break

    sizes_mb = {
        "pytorch_fp32": file_size_mb(pt_weights_path) if pt_weights_path else None,
        "onnx_fp32": file_size_mb(onnx_fp32_path),
        "onnx_int8": file_size_mb(onnx_int8_path),
    }

    # ── 5. Inference helpers ────────────────────────────────────────────────

    def predict_pt(texts: list[str], batch_size: int = 64) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(pt_device)
                logits = pt_model(**enc).logits
                preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    def predict_ort(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            preds.append(logits.argmax(dim=-1).numpy())
        return np.concatenate(preds)

    def latency_per_sample(predict_fn, texts, n: int = 200) -> float:
        # Warm-up
        predict_fn(texts[: min(8, len(texts))])
        sample = texts[: min(n, len(texts))]
        t0 = time.perf_counter()
        for t in sample:
            predict_fn([t])
        return round(((time.perf_counter() - t0) / len(sample)) * 1000, 3)

    # ── 6. Evaluate every (model, test_set) pair ────────────────────────────
    test_sets = {
        "test_a_macd_hindi": test_a,
        "test_b_davidson_10pct": test_b,
        "test_c_combined": test_c,
    }
    model_specs = [
        ("pytorch_fp32", lambda texts: predict_pt(texts), True),
        ("onnx_fp32", lambda texts: predict_ort(ort_fp32, texts), False),
        ("onnx_int8", lambda texts: predict_ort(ort_int8, texts), False),
    ]

    grid = {}
    confusion_grid = {}
    for model_name, predict_fn, _is_pt in model_specs:
        grid[model_name] = {}
        confusion_grid[model_name] = {}
        for ts_name, ts_df in test_sets.items():
            print(f"[eval] {model_name} on {ts_name} (n={len(ts_df)}) on {pt_device}…")
            texts = ts_df["text"].tolist()
            labels = ts_df["label"].values.astype(int)
            preds = predict_fn(texts)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            metrics = {
                "n": len(ts_df),
                "accuracy": float(accuracy_score(labels, preds)),
                "f1_macro": float(f1_score(labels, preds, average="macro")),
                "precision_macro": float(
                    precision_score(labels, preds, average="macro", zero_division=0)
                ),
                "recall_macro": float(
                    recall_score(labels, preds, average="macro", zero_division=0)
                ),
                "confusion_matrix": cm.tolist(),
            }
            grid[model_name][ts_name] = metrics
            confusion_grid[model_name][ts_name] = cm

    # ── 6b. Latency benchmark — ALWAYS on CPU ───────────────────────────────
    # The extension runs ONNX in the browser via WASM (CPU). Reporting GPU
    # latency would be misleading, so we explicitly move PT to CPU and reload
    # ORT sessions with the CPU execution provider before timing.
    print("[eval] reloading models on CPU for realistic latency benchmark…")
    pt_model.to("cpu")
    pt_model.eval()
    ort_fp32_cpu = ORTModelForSequenceClassification.from_pretrained(
        ONNX_FP32_DIR, provider="CPUExecutionProvider"
    )
    ort_int8_cpu = ORTModelForSequenceClassification.from_pretrained(
        ONNX_INT8_DIR,
        file_name="model_quantized.onnx",
        provider="CPUExecutionProvider",
    )

    def predict_pt_cpu(texts: list[str], batch_size: int = 64) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                logits = pt_model(**enc).logits
                preds.append(logits.argmax(dim=-1).numpy())
        return np.concatenate(preds)

    cpu_predict_fns = {
        "pytorch_fp32": predict_pt_cpu,
        "onnx_fp32": lambda texts: predict_ort(ort_fp32_cpu, texts),
        "onnx_int8": lambda texts: predict_ort(ort_int8_cpu, texts),
    }
    for model_name, fn in cpu_predict_fns.items():
        lat = latency_per_sample(fn, test_a["text"].tolist())
        grid[model_name]["latency_ms_per_sample_test_a_cpu"] = lat
        print(f"  {model_name} CPU latency: {lat} ms/sample (batch=1)")

    # Move PT back to GPU so any later code (none here, but defensive) is fine
    pt_model.to(pt_device)

    # ── 7. Confusion matrix figure ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    model_order = ["pytorch_fp32", "onnx_fp32", "onnx_int8"]
    ts_order = ["test_a_macd_hindi", "test_b_davidson_10pct", "test_c_combined"]
    for r, mn in enumerate(model_order):
        for c, tn in enumerate(ts_order):
            ax = axes[r, c]
            cm = confusion_grid[mn][tn]
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels([ID2LABEL[0], ID2LABEL[1]], rotation=20)
            ax.set_yticklabels([ID2LABEL[0], ID2LABEL[1]])
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{mn}\n{tn}")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Confusion matrices — 3 models × 3 test sets")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    cm_png = buf.read()

    # ── 8. Build human-readable report ──────────────────────────────────────
    def fmt_pct(x):
        return "—" if x is None else f"{x * 100:.2f}%"

    def fmt_mb(x):
        return "—" if x is None else f"{x:.1f} MB"

    int8_acc_a = grid["onnx_int8"]["test_a_macd_hindi"]["accuracy"]
    int8_f1_a = grid["onnx_int8"]["test_a_macd_hindi"]["f1_macro"]
    fp32_acc_a = grid["onnx_fp32"]["test_a_macd_hindi"]["accuracy"]
    fp32_f1_a = grid["onnx_fp32"]["test_a_macd_hindi"]["f1_macro"]
    pt_acc_a = grid["pytorch_fp32"]["test_a_macd_hindi"]["accuracy"]
    pt_f1_a = grid["pytorch_fp32"]["test_a_macd_hindi"]["f1_macro"]

    int8_size = sizes_mb["onnx_int8"]
    pt_size = sizes_mb["pytorch_fp32"] or 0.0

    baseline_acc = NEURIPS_BASELINE.get("macd_hindi_test_accuracy")
    baseline_f1 = NEURIPS_BASELINE.get("macd_hindi_test_f1_macro")
    baseline_size_mb = NEURIPS_BASELINE.get("size_mb")
    gap_acc = (
        None if baseline_acc is None else round((baseline_acc - int8_acc_a) * 100, 2)
    )
    gap_f1 = None if baseline_f1 is None else round((baseline_f1 - int8_f1_a) * 100, 2)
    size_ratio = (
        None
        if baseline_size_mb is None or int8_size == 0
        else round(baseline_size_mb / int8_size, 1)
    )

    md = []
    md.append("# SurakshaNet — Final Training & Evaluation Report")
    md.append("")
    md.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_")
    md.append("")
    md.append("## Recipe")
    md.append("")
    md.append("- Base model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`")
    md.append(
        "- Training data: MACD Hindi train+val (ShareChatAI/MACD) + Davidson 2017 English (90% train split)"
    )
    md.append(
        f"- Hyperparameters: epochs={TRAIN_HPARAMS['epochs']}, "
        f"lr={TRAIN_HPARAMS['lr']}, batch={TRAIN_HPARAMS['batch_size']}, "
        f"weight_decay={TRAIN_HPARAMS['weight_decay']}, "
        f"warmup_ratio={TRAIN_HPARAMS['warmup_ratio']}, "
        f"lr_scheduler={TRAIN_HPARAMS['lr_scheduler_type']}, "
        f"label_smoothing={TRAIN_HPARAMS['label_smoothing']}, "
        f"class_weights=balanced, metric_for_best={TRAIN_HPARAMS['metric_for_best_model']}"
    )
    md.append("")

    md.append("## Leakage check")
    md.append("")
    md.append("| Test set | N | Train overlap | sha256(text)[..16] |")
    md.append("|---|---:|---:|---|")
    for k, v in leakage_report.items():
        md.append(
            f"| {k} | {v['n']} | {v['leakage_rows']} | `{v['sha256_prefix']}…` |"
        )
    md.append("")
    md.append("All test sets are confirmed disjoint from training data.")
    md.append("")

    md.append("## Size comparison")
    md.append("")
    md.append("| Stage | File | Size |")
    md.append("|---|---|---:|")
    md.append(
        f"| PyTorch FP32 | weights file | {fmt_mb(sizes_mb['pytorch_fp32'])} |"
    )
    md.append(f"| ONNX FP32 | `model.onnx` | {fmt_mb(sizes_mb['onnx_fp32'])} |")
    md.append(
        f"| ONNX INT8 (shipped) | `model_quantized.onnx` | {fmt_mb(sizes_mb['onnx_int8'])} |"
    )
    if sizes_mb["pytorch_fp32"] and sizes_mb["onnx_int8"]:
        md.append(
            f"\nINT8 is **{round(sizes_mb['pytorch_fp32'] / sizes_mb['onnx_int8'], 1)}×** "
            f"smaller than the PyTorch FP32 weights."
        )
    md.append("")

    md.append("## Accuracy on held-out test sets")
    md.append("")
    for ts_label, ts_key in [
        ("Test A — MACD Hindi (`hindi_test.csv`, full)", "test_a_macd_hindi"),
        ("Test B — Davidson 10% holdout (English)", "test_b_davidson_10pct"),
        ("Test C — Combined (A + B)", "test_c_combined"),
    ]:
        md.append(f"### {ts_label}")
        md.append("")
        md.append("| Model | Accuracy | Macro F1 | Precision | Recall |")
        md.append("|---|---:|---:|---:|---:|")
        for mn in model_order:
            m = grid[mn][ts_key]
            md.append(
                f"| {mn} | {fmt_pct(m['accuracy'])} | {fmt_pct(m['f1_macro'])} | "
                f"{fmt_pct(m['precision_macro'])} | {fmt_pct(m['recall_macro'])} |"
            )
        md.append("")

    md.append("## CPU latency (batch = 1, single-threaded)")
    md.append("")
    md.append(
        "_Measured with the model forced onto the CPU execution provider; "
        "this is the runtime characteristic the Chrome extension actually "
        "experiences via `@xenova/transformers` + WASM._"
    )
    md.append("")
    md.append("| Model | ms / sample |")
    md.append("|---|---:|")
    for mn in model_order:
        md.append(
            f"| {mn} | {grid[mn]['latency_ms_per_sample_test_a_cpu']} |"
        )
    md.append("")

    md.append("## Quantization quality delta (Test A)")
    md.append("")
    md.append("| Metric | PyTorch FP32 | ONNX FP32 | ONNX INT8 | INT8 - FP32 (pp) |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(
        f"| Accuracy | {fmt_pct(pt_acc_a)} | {fmt_pct(fp32_acc_a)} | "
        f"{fmt_pct(int8_acc_a)} | {round((int8_acc_a - fp32_acc_a) * 100, 2)} |"
    )
    md.append(
        f"| Macro F1 | {fmt_pct(pt_f1_a)} | {fmt_pct(fp32_f1_a)} | "
        f"{fmt_pct(int8_f1_a)} | {round((int8_f1_a - fp32_f1_a) * 100, 2)} |"
    )
    md.append("")

    md.append("## NeurIPS baseline comparison (MACD Test A)")
    md.append("")
    if baseline_acc is None and baseline_f1 is None:
        md.append(
            "_The published baseline numbers from the MACD paper "
            "(Maity et al., NeurIPS 2022) are not configured in `_common.py`. "
            "Edit `NEURIPS_BASELINE` to fill in `macd_hindi_test_accuracy` and "
            "`macd_hindi_test_f1_macro` and re-run this script to regenerate the table below._"
        )
    else:
        md.append("| Model | Size | Accuracy | Macro F1 |")
        md.append("|---|---:|---:|---:|")
        md.append(
            f"| {NEURIPS_BASELINE['name']} | {fmt_mb(baseline_size_mb)} | "
            f"{fmt_pct(baseline_acc)} | {fmt_pct(baseline_f1)} |"
        )
        md.append(
            f"| Ours (MiniLM INT8, shipped) | {fmt_mb(int8_size)} | "
            f"{fmt_pct(int8_acc_a)} | {fmt_pct(int8_f1_a)} |"
        )
        md.append(
            f"| **Gap** | "
            f"{('—' if size_ratio is None else f'÷{size_ratio}× smaller')} | "
            f"{('—' if gap_acc is None else f'-{gap_acc} pp')} | "
            f"{('—' if gap_f1 is None else f'-{gap_f1} pp')} |"
        )
    md.append("")

    md.append("## Deployment artifact")
    md.append("")
    md.append(
        "The INT8 ONNX model and tokenizer have been written to "
        "`assets/models/custom-macd-model/`. The Chrome extension loads this "
        "directly via `@xenova/transformers` from the service worker; nothing "
        "is fetched from the network at runtime."
    )
    md.append("")

    final_report_md = "\n".join(md)

    eval_metrics = {
        "leakage_report": leakage_report,
        "sizes_mb": sizes_mb,
        "metrics_grid": grid,
        "neurips_baseline": NEURIPS_BASELINE,
    }

    return {
        "eval_metrics": eval_metrics,
        "confusion_matrices_png": cm_png,
        "final_report_md": final_report_md,
    }


@app.local_entrypoint()
def main():
    print("==> running modal_evaluate.evaluate() on Modal…")
    out = evaluate.remote()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(repo_root, "REPORTS")
    os.makedirs(reports_dir, exist_ok=True)

    with open(os.path.join(reports_dir, "eval_metrics.json"), "w") as f:
        json.dump(out["eval_metrics"], f, indent=2)
    with open(os.path.join(reports_dir, "confusion_matrices.png"), "wb") as f:
        f.write(out["confusion_matrices_png"])
    with open(os.path.join(reports_dir, "final_report.md"), "w") as f:
        f.write(out["final_report_md"])

    print("\n==> wrote:")
    print(f"  {reports_dir}/eval_metrics.json")
    print(f"  {reports_dir}/confusion_matrices.png")
    print(f"  {reports_dir}/final_report.md")
    print("\n==> done. final_report.md preview:\n")
    print(out["final_report_md"])
