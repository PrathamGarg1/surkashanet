"""
SurakshaNet — Step 1/3: train MiniLM on MACD Hindi + Davidson English.

Run on Modal A10G:
    modal run training/modal_train.py

Held-out test data is NEVER touched here:
  - MACD hindi_test.csv is not loaded
  - Davidson is split 90/10 with seed ``DAVIDSON_SEED``; the held-out 10%
    row indices are persisted to ``DAVIDSON_INDICES_FILE`` on the shared
    volume so modal_evaluate.py uses the exact same rows.

Outputs:
  - /vol/checkpoints/pt/                    PyTorch checkpoint + tokenizer
  - /vol/davidson_test_indices.json         held-out Davidson row indices
  - REPORTS/train_metrics.json              per-step + per-epoch metrics
  - REPORTS/loss_curves.png                 train + val loss
  - REPORTS/f1_curves.png                  val macro F1 per epoch (best checkpoint)
  - REPORTS/size_pt.json                    PyTorch FP32 file size + param count
"""

from __future__ import annotations

import json
import os
import sys

import modal

# Make `_common` importable when this file is run directly via `modal run`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    APP_NAME,
    DAVIDSON_INDICES_FILE,
    ID2LABEL,
    IMAGE,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_NAME,
    PT_CHECKPOINT_DIR,
    SEED,
    TRAIN_HPARAMS,
    VOLUME,
    VOLUME_MOUNT,
    davidson_split,
    dir_size_mb,
    file_size_mb,
    load_davidson_full,
    load_macd_train_val,
)

app = modal.App(f"{APP_NAME}-train")


@app.function(
    image=IMAGE,
    gpu="A10G",
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=60 * 60 * 3,
)
def train():
    import io

    import matplotlib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_seed(SEED)

    # ── 1. Load datasets ─────────────────────────────────────────────────────
    print("[train] loading MACD train+val (Hindi)…")
    macd_train, macd_val = load_macd_train_val()
    print(f"  MACD train: {len(macd_train)} rows")
    print(f"  MACD val  : {len(macd_val)} rows")

    print("[train] loading Davidson (English) and splitting 90/10…")
    davidson_df = load_davidson_full()
    print(f"  Davidson full: {len(davidson_df)} rows")
    # Always create a fresh split here. Test rows are persisted to the volume.
    if os.path.exists(DAVIDSON_INDICES_FILE):
        os.remove(DAVIDSON_INDICES_FILE)
    davidson_train, davidson_test, davidson_test_indices = davidson_split(
        davidson_df, indices_file=None
    )
    print(f"  Davidson train: {len(davidson_train)} rows")
    print(f"  Davidson test (held out): {len(davidson_test)} rows")

    os.makedirs(os.path.dirname(DAVIDSON_INDICES_FILE), exist_ok=True)
    with open(DAVIDSON_INDICES_FILE, "w") as f:
        json.dump(davidson_test_indices, f)
    VOLUME.commit()
    print(f"[train] persisted held-out indices -> {DAVIDSON_INDICES_FILE}")

    # ── 2. Build train / val frames ──────────────────────────────────────────
    train_df = pd.concat([macd_train, davidson_train], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_df = macd_val.copy()

    print(f"[train] combined train: {len(train_df)} rows")
    print(f"  label distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"[train] val: {len(val_df)} rows")
    print(f"  label distribution: {val_df['label'].value_counts().to_dict()}")

    # ── 3. Tokenizer + model ────────────────────────────────────────────────
    print(f"[train] loading tokenizer + base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        ds = ds.map(
            lambda b: tokenizer(
                b["text"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
        )
        ds = ds.remove_columns(["text"])
        ds.set_format("torch")
        return ds

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)

    # ── 4. Class weights ────────────────────────────────────────────────────
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    print(f"[train] class weights: {class_weights.tolist()}")

    # ── 5. Custom Trainer (class weights + label smoothing) ─────────────────
    label_smoothing = TRAIN_HPARAMS["label_smoothing"]

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weight = class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(
                weight=weight, label_smoothing=label_smoothing
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
        }

    # ── 6. Training arguments ───────────────────────────────────────────────
    if os.path.exists(PT_CHECKPOINT_DIR):
        import shutil

        shutil.rmtree(PT_CHECKPOINT_DIR)
    os.makedirs(PT_CHECKPOINT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=f"{VOLUME_MOUNT}/checkpoints/_runs",
        num_train_epochs=TRAIN_HPARAMS["epochs"],
        learning_rate=TRAIN_HPARAMS["lr"],
        per_device_train_batch_size=TRAIN_HPARAMS["batch_size"],
        per_device_eval_batch_size=TRAIN_HPARAMS["eval_batch_size"],
        weight_decay=TRAIN_HPARAMS["weight_decay"],
        warmup_ratio=TRAIN_HPARAMS["warmup_ratio"],
        lr_scheduler_type=TRAIN_HPARAMS["lr_scheduler_type"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=TRAIN_HPARAMS["metric_for_best_model"],
        greater_is_better=True,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ── 7. Train ────────────────────────────────────────────────────────────
    print("[train] starting…")
    trainer.train()
    print("[train] complete; saving best checkpoint…")
    trainer.save_model(PT_CHECKPOINT_DIR)
    tokenizer.save_pretrained(PT_CHECKPOINT_DIR)
    VOLUME.commit()

    # ── 8. Collect metrics + curves ─────────────────────────────────────────
    log_history = trainer.state.log_history
    train_steps = [
        {"step": e["step"], "loss": e["loss"], "epoch": e.get("epoch")}
        for e in log_history
        if "loss" in e and "eval_loss" not in e
    ]
    eval_epochs = [
        {
            "epoch": e["epoch"],
            "step": e.get("step"),
            "eval_loss": e["eval_loss"],
            "eval_accuracy": e.get("eval_accuracy"),
            "eval_f1_macro": e.get("eval_f1_macro"),
            "eval_precision_macro": e.get("eval_precision_macro"),
            "eval_recall_macro": e.get("eval_recall_macro"),
        }
        for e in log_history
        if "eval_loss" in e
    ]

    train_metrics = {
        "hparams": TRAIN_HPARAMS,
        "seed": SEED,
        "model_name": MODEL_NAME,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "class_weights": class_weights.tolist(),
        "train_label_distribution": (
            train_df["label"].value_counts().to_dict()
        ),
        "val_label_distribution": (
            val_df["label"].value_counts().to_dict()
        ),
        "per_step_train": train_steps,
        "per_epoch_eval": eval_epochs,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_steps:
        ax.plot(
            [e["step"] for e in train_steps],
            [e["loss"] for e in train_steps],
            label="train loss",
            alpha=0.6,
        )
    if eval_epochs:
        ax.plot(
            [e["step"] for e in eval_epochs],
            [e["eval_loss"] for e in eval_epochs],
            label="val loss",
            linestyle="--",
            marker="o",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("SurakshaNet — training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    loss_png = buf.read()
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    f1_epochs = [
        e["epoch"] for e in eval_epochs if e.get("eval_f1_macro") is not None
    ]
    f1_vals = [
        e["eval_f1_macro"] for e in eval_epochs if e.get("eval_f1_macro") is not None
    ]
    if f1_vals:
        ax2.plot(
            f1_epochs,
            f1_vals,
            marker="o",
            linestyle="-",
            color="C2",
            label="validation macro F1",
        )
        best_i = int(np.argmax(f1_vals))
        ax2.scatter(
            [f1_epochs[best_i]],
            [f1_vals[best_i]],
            s=140,
            zorder=5,
            color="crimson",
            edgecolors="white",
            linewidths=1.5,
            label=f"maximum ({f1_vals[best_i]:.4f} @ epoch {f1_epochs[best_i]:.0f})",
        )
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro F1 (validation)")
    ax2.set_title("SurakshaNet — validation macro F1 by epoch")
    if f1_epochs:
        ax2.set_xticks(f1_epochs)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(
        max(0.0, (min(f1_vals) if f1_vals else 0) - 0.02),
        min(1.0, (max(f1_vals) if f1_vals else 1) + 0.02),
    )
    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png", dpi=150)
    buf2.seek(0)
    f1_png = buf2.read()
    plt.close(fig2)

    # ── 9. Size + parameter count ───────────────────────────────────────────
    weights_file = None
    for cand in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(PT_CHECKPOINT_DIR, cand)
        if os.path.exists(p):
            weights_file = p
            break

    size_pt = {
        "checkpoint_dir": PT_CHECKPOINT_DIR,
        "checkpoint_dir_size_mb": dir_size_mb(PT_CHECKPOINT_DIR),
        "weights_file": os.path.basename(weights_file) if weights_file else None,
        "weights_file_size_mb": (
            file_size_mb(weights_file) if weights_file else None
        ),
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "params_trainable": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    print(f"[train] size_pt: {json.dumps(size_pt, indent=2)}")

    return {
        "train_metrics": train_metrics,
        "size_pt": size_pt,
        "loss_curves_png": loss_png,
        "f1_curves_png": f1_png,
    }


@app.local_entrypoint()
def main():
    print("==> running modal_train.train() on Modal…")
    out = train.remote()

    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "REPORTS"
    )
    os.makedirs(reports_dir, exist_ok=True)

    with open(os.path.join(reports_dir, "train_metrics.json"), "w") as f:
        json.dump(out["train_metrics"], f, indent=2, default=str)
    with open(os.path.join(reports_dir, "size_pt.json"), "w") as f:
        json.dump(out["size_pt"], f, indent=2)
    with open(os.path.join(reports_dir, "loss_curves.png"), "wb") as f:
        f.write(out["loss_curves_png"])
    with open(os.path.join(reports_dir, "f1_curves.png"), "wb") as f:
        f.write(out["f1_curves_png"])

    print("\n==> wrote:")
    print(f"  {reports_dir}/train_metrics.json")
    print(f"  {reports_dir}/size_pt.json")
    print(f"  {reports_dir}/loss_curves.png")
    print(f"  {reports_dir}/f1_curves.png")
    print("\n==> next: modal run training/modal_export.py")
