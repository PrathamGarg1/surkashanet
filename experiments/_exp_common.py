"""
Shared Hugging Face / Modal harness for SurakshaNet MiniLM experiments.

Invariants
----------
* Final backbone is always
  ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``.
* Validation is MACD ``hindi_val.csv`` only.
* ``hindi_test.csv`` is loaded only when ``allow_test=True`` and a
  ``test_reason`` is recorded. Search / ablation / pruning / threshold
  tuning never set that flag.
* Model selection objective is validation macro-F1.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

_COMMON_PATH = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(_COMMON_PATH))
EXPERIMENTS_DIR = os.path.dirname(_COMMON_PATH)
RESULTS_JSON = os.path.join(EXPERIMENTS_DIR, "results.json")
RESULTS_CSV = os.path.join(EXPERIMENTS_DIR, "results.csv")
CONFIGS_DIR = os.path.join(EXPERIMENTS_DIR, "configs")
LOGS_DIR = os.path.join(EXPERIMENTS_DIR, "logs")
PLOTS_DIR = os.path.join(EXPERIMENTS_DIR, "plots")

for _d in (CONFIGS_DIR, LOGS_DIR, PLOTS_DIR):
    os.makedirs(_d, exist_ok=True)

APP_NAME = "surakshanet-experiments"
VOLUME_NAME = "surakshanet-artifacts"
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_MOUNT = "/vol"

PT_CHECKPOINT_DIR = f"{VOLUME_MOUNT}/checkpoints/pt"
ONNX_FP32_DIR = f"{VOLUME_MOUNT}/checkpoints/onnx_fp32"
ONNX_INT8_DIR = f"{VOLUME_MOUNT}/checkpoints/onnx_int8"
PRUNED_DIR = f"{VOLUME_MOUNT}/checkpoints/pt_vocab_pruned"

# Keep train and export stacks on the same Transformers / tokenizers versions
# so the fast tokenizer JSON written at train time can be read by Optimum.
TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.2",
        "datasets==2.16.1",
        "accelerate==0.25.0",
        "evaluate==0.4.6",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "numpy==1.24.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.2",
        "matplotlib==3.8.2",
        "optuna==4.0.0",
    )
    .add_local_file(_COMMON_PATH, remote_path="/root/_exp_common.py")
)

EXPORT_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.2",
        "datasets==2.16.1",
        "optimum[onnxruntime]==1.16.1",
        "onnx==1.16.0",
        "onnxruntime==1.16.3",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "numpy==1.24.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.2",
    )
    .add_local_file(_COMMON_PATH, remote_path="/root/_exp_common.py")
)

app = modal.App(APP_NAME)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
DAVIDSON_SEED = 42
MAX_LENGTH = 96

ID2LABEL = {0: "abusive", 1: "non-abusive"}
LABEL2ID = {"abusive": 0, "non-abusive": 1}
DAVIDSON_LABEL_MAP = {0: 0, 1: 0, 2: 1}

MACD_BASE_URL = "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/"
DAVIDSON_URL = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)

BASE_CONFIG: dict[str, Any] = {
    "run_name": "unnamed",
    "model_name": MODEL_NAME,
    "seed": SEED,
    "data_mix": "macd+davidson",
    "davidson_fraction": 1.0,
    "drop_train_duplicates": False,
    "balance_train": False,
    "max_length": MAX_LENGTH,
    "padding": "dynamic",
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "linear",
    "max_grad_norm": 1.0,
    "fp16": True,
    "group_by_length": False,
    "class_weights": False,
    "label_smoothing": 0.0,
    "metric_for_best_model": "f1_macro",
    "early_stopping_patience": 0,
    "stage2": None,
    "distill": None,
    "save_to": None,
    "return_val_predictions": False,
    "allow_test": False,
    "test_reason": None,
}


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return str(obj)


def dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False)


def file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / (1024 * 1024), 3)


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return round(total / (1024 * 1024), 3)


def load_macd(split: str):
    import pandas as pd

    assert split in {"hindi_train", "hindi_val", "hindi_test"}
    url = f"{MACD_BASE_URL}{split}.csv"
    df = pd.read_csv(url)
    text_col = "text" if "text" in df.columns else "comment_text"
    label_col = "class" if "class" in df.columns else "label"
    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    out["text"] = out["text"].astype(str)
    out["label"] = out["label"].astype(int)
    return out.reset_index(drop=True)


def load_davidson_full():
    import pandas as pd

    df = pd.read_csv(DAVIDSON_URL)
    out = pd.DataFrame(
        {
            "text": df["tweet"].astype(str),
            "label": df["class"].map(DAVIDSON_LABEL_MAP).astype(int),
        }
    )
    return out.reset_index(drop=True)


def davidson_split(df, seed: int = DAVIDSON_SEED, holdout: float = 0.1):
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_hold = int(round(len(df) * holdout))
    hold_idx, train_idx = idx[:n_hold], idx[n_hold:]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[hold_idx].reset_index(drop=True),
        hold_idx.tolist(),
    )


def build_train_val(cfg: dict):
    import pandas as pd

    macd_train = load_macd("hindi_train")
    macd_val = load_macd("hindi_val")
    parts = [macd_train]
    dav_train_rows = 0
    if cfg["data_mix"] in {"macd+davidson", "davidson_only"}:
        dav_full = load_davidson_full()
        dav_train, _, _ = davidson_split(dav_full)
        frac = float(cfg.get("davidson_fraction", 1.0))
        if frac < 1.0:
            dav_train = dav_train.sample(
                frac=frac, random_state=cfg["seed"]
            ).reset_index(drop=True)
        dav_train_rows = len(dav_train)
        if cfg["data_mix"] == "davidson_only":
            parts = [dav_train]
        else:
            parts.append(dav_train)
    train_df = pd.concat(parts, ignore_index=True)
    if cfg.get("drop_train_duplicates"):
        train_df = train_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if cfg.get("balance_train"):
        counts = train_df["label"].value_counts()
        n_each = int(counts.min())
        train_df = pd.concat(
            [
                train_df[train_df["label"] == label].sample(
                    n=n_each, random_state=cfg["seed"]
                )
                for label in sorted(counts.index)
            ],
            ignore_index=True,
        )
    return train_df, macd_val, {
        "macd_train_rows": len(macd_train),
        "davidson_train_rows": dav_train_rows,
        "train_rows": len(train_df),
        "val_rows": len(macd_val),
        "train_label_counts": train_df["label"].value_counts().to_dict(),
        "val_label_counts": macd_val["label"].value_counts().to_dict(),
    }


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
    }


def full_report(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "per_class": {
            ID2LABEL[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in (0, 1)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
    }


def tune_threshold(probs, labels):
    import numpy as np
    from sklearn.metrics import f1_score

    best = {"threshold": 0.5, "f1_macro": -1.0}
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        # Label convention: 0=abusive. Convert "abusive probability" to class-0
        # decision: predict abusive when p_abusive >= threshold.
        preds = np.where(probs >= t, 0, 1)
        score = f1_score(labels, preds, average="macro")
        if score > best["f1_macro"]:
            best = {
                "threshold": float(t),
                "f1_macro": float(score),
                "accuracy": float((preds == labels).mean()),
            }
    return best


def package_versions():
    import sys

    import numpy as np
    import sklearn
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def train_texts_sha(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:32]


def run_training(cfg: dict) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    cfg = {**BASE_CONFIG, **cfg}
    set_seed(int(cfg["seed"]))
    started = time.time()

    train_df, val_df, data_meta = build_train_val(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    pad_kw = {}
    if cfg["padding"] == "max_length":
        pad_kw = {"padding": "max_length", "max_length": cfg["max_length"]}

    def tokenize_function(batch):
        if cfg["padding"] == "max_length":
            return tokenizer(batch["text"], truncation=True, **pad_kw)
        return tokenizer(
            batch["text"], truncation=True, max_length=cfg["max_length"]
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
    train_ds = train_ds.map(tokenize_function, batched=True).remove_columns(["text"])
    val_ds = val_ds.map(tokenize_function, batched=True).remove_columns(["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weight_tensor = None
    if cfg.get("class_weights"):
        weights = compute_class_weight(
            "balanced", classes=np.array([0, 1]), y=train_df["label"].to_numpy()
        )
        class_weight_tensor = torch.tensor(weights, dtype=torch.float)

    label_smoothing = float(cfg.get("label_smoothing") or 0.0)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(
                weight=class_weight_tensor.to(logits.device)
                if class_weight_tensor is not None
                else None,
                label_smoothing=label_smoothing,
            )
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"],
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    out_dir = f"/tmp/exp_runs/{cfg['run_name']}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        max_grad_norm=float(cfg["max_grad_norm"]),
        fp16=bool(cfg["fp16"]) and torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        save_total_limit=1,
        group_by_length=bool(cfg.get("group_by_length")),
        logging_steps=50,
        report_to="none",
        seed=int(cfg["seed"]),
    )

    callbacks = []
    if int(cfg.get("early_stopping_patience") or 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg["early_stopping_patience"])
            )
        )

    trainer = CustomTrainer(
        model_init=model_init,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()

    # Optional MACD-only stage-2 fine-tune.
    if cfg.get("stage2"):
        stage2 = cfg["stage2"]
        macd_only = load_macd("hindi_train")
        s2_ds = Dataset.from_pandas(macd_only[["text", "label"]], preserve_index=False)
        s2_ds = s2_ds.map(tokenize_function, batched=True).remove_columns(["text"])
        s2_args = TrainingArguments(
            output_dir=f"{out_dir}/stage2",
            learning_rate=float(stage2.get("learning_rate", 1e-5)),
            num_train_epochs=float(stage2.get("num_train_epochs", 2)),
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
            weight_decay=float(cfg["weight_decay"]),
            warmup_ratio=float(cfg["warmup_ratio"]),
            lr_scheduler_type=cfg["lr_scheduler_type"],
            fp16=bool(cfg["fp16"]) and torch.cuda.is_available(),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=cfg["metric_for_best_model"],
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=50,
            report_to="none",
            seed=int(cfg["seed"]),
        )
        trainer = CustomTrainer(
            model=trainer.model,
            args=s2_args,
            train_dataset=s2_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    pred = trainer.predict(val_ds)
    logits = pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 0]
    y_true = pred.label_ids
    y_pred = np.argmax(logits, axis=-1)
    val_report = full_report(y_true, y_pred)
    tuned = tune_threshold(probs, y_true)
    tuned_preds = np.where(probs >= tuned["threshold"], 0, 1)
    tuned_full = full_report(y_true, tuned_preds)
    tuned_full["threshold"] = tuned["threshold"]

    history = trainer.state.log_history
    per_step_train = [
        {"step": h.get("step"), "epoch": h.get("epoch"), "loss": h.get("loss"), "stage": 1}
        for h in history
        if "loss" in h and "eval_loss" not in h
    ]
    per_epoch_eval = [
        {
            "epoch": h.get("epoch"),
            "step": h.get("step"),
            "stage": 1,
            "eval_loss": h.get("eval_loss"),
            "eval_accuracy": h.get("eval_accuracy"),
            "eval_f1_macro": h.get("eval_f1_macro"),
            "eval_precision_macro": h.get("eval_precision_macro"),
            "eval_recall_macro": h.get("eval_recall_macro"),
        }
        for h in history
        if "eval_f1_macro" in h
    ]

    result = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "versions": package_versions(),
        "data": data_meta,
        "tokenizer": {
            "max_length": cfg["max_length"],
            "padding": cfg["padding"],
            "model_max_length": tokenizer.model_max_length,
            "vocab_size": len(tokenizer),
        },
        "train_texts_sha": train_texts_sha(train_df["text"].tolist()),
        "per_step_train": per_step_train,
        "per_epoch_eval": per_epoch_eval,
        "best_val_metric_stage1": max(
            (e.get("eval_f1_macro") or 0.0 for e in per_epoch_eval), default=0.0
        ),
        "val": val_report,
        "val_tuned_threshold": tuned_full,
        "runtime_sec": round(time.time() - started, 1),
    }

    if cfg.get("return_val_predictions"):
        result["val_predictions"] = {
            "preds": y_pred.tolist(),
            "p_abusive": probs.tolist(),
            "labels": y_true.tolist(),
        }

    if cfg.get("save_to"):
        save_to = cfg["save_to"]
        if os.path.exists(save_to):
            backup = f"{save_to}.backup-{int(time.time())}"
            shutil.move(save_to, backup)
            print(f"preserved previous checkpoint at {backup}")
        os.makedirs(save_to, exist_ok=True)
        trainer.save_model(save_to)
        tokenizer.save_pretrained(save_to)
        with open(os.path.join(save_to, "experiment_config.json"), "w") as f:
            json.dump(to_jsonable(cfg), f, indent=2)
        result["saved_to"] = save_to
        VOLUME.commit()

    if cfg.get("allow_test"):
        assert cfg.get("test_reason"), "allow_test requires test_reason"
        test_df = load_macd("hindi_test")
        test_ds = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False)
        test_ds = test_ds.map(tokenize_function, batched=True).remove_columns(["text"])
        test_pred = trainer.predict(test_ds)
        test_logits = test_pred.predictions
        test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()[:, 0]
        test_y = test_pred.label_ids
        test_hat = np.argmax(test_logits, axis=-1)
        result["macd_test"] = full_report(test_y, test_hat)
        thr = tuned_full["threshold"]
        result["macd_test_val_threshold"] = {
            "threshold": thr,
            **full_report(test_y, np.where(test_probs >= thr, 0, 1)),
        }
        result["test_reason"] = cfg["test_reason"]

    return result


@app.function(
    image=TRAIN_IMAGE,
    gpu="A10G",
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=60 * 120,
    retries=1,
)
def train_one(cfg: dict) -> str:
    print(f"[{cfg.get('run_name')}] starting")
    try:
        result = run_training(cfg)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return dumps(
            {
                "run_name": cfg.get("run_name"),
                "config": cfg,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    v = result["val"]
    print(
        f"[{cfg.get('run_name')}] val acc={v['accuracy']:.4f} "
        f"f1={v['f1_macro']:.4f} ({result['runtime_sec']}s)"
    )
    return dumps(result)


def save_config(name: str, cfg: dict) -> str:
    path = os.path.join(CONFIGS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(to_jsonable(cfg), f, indent=2)
    return path


def save_log(name: str, record: dict) -> str:
    path = os.path.join(LOGS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(to_jsonable(record), f, indent=2)
    return path


def append_results(records: list[dict], experiment: str, phase: str) -> None:
    existing = []
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            existing = json.load(f)
    by_name = {r.get("run_name"): r for r in existing if r.get("run_name")}
    for record in records:
        record = dict(record)
        record.setdefault("experiment", experiment)
        record.setdefault("phase", phase)
        record["recorded_at"] = datetime.now().isoformat(timespec="seconds")
        by_name[record["run_name"]] = record
    merged = list(by_name.values())
    with open(RESULTS_JSON, "w") as f:
        json.dump(to_jsonable(merged), f, indent=2)

    fieldnames = [
        "run_name",
        "experiment",
        "phase",
        "data_mix",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "weight_decay",
        "warmup_ratio",
        "lr_scheduler_type",
        "max_length",
        "padding",
        "class_weights",
        "label_smoothing",
        "seed",
        "train_rows",
        "val_accuracy",
        "val_f1_macro",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro_tuned",
        "val_threshold",
        "test_accuracy",
        "test_f1_macro",
        "model_mb",
        "tokenizer_mb",
        "total_artifact_mb",
        "runtime_sec",
        "notes",
    ]
    rows = []
    for r in merged:
        cfg = r.get("config") or {}
        val = r.get("val") or {}
        tuned = r.get("val_tuned_threshold") or {}
        test = r.get("macd_test") or {}
        sizes = r.get("sizes") or {}
        data = r.get("data") or {}
        rows.append(
            {
                "run_name": r.get("run_name"),
                "experiment": r.get("experiment"),
                "phase": r.get("phase"),
                "data_mix": cfg.get("data_mix"),
                "learning_rate": cfg.get("learning_rate"),
                "num_train_epochs": cfg.get("num_train_epochs"),
                "per_device_train_batch_size": cfg.get("per_device_train_batch_size"),
                "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps"),
                "weight_decay": cfg.get("weight_decay"),
                "warmup_ratio": cfg.get("warmup_ratio"),
                "lr_scheduler_type": cfg.get("lr_scheduler_type"),
                "max_length": cfg.get("max_length"),
                "padding": cfg.get("padding"),
                "class_weights": cfg.get("class_weights"),
                "label_smoothing": cfg.get("label_smoothing"),
                "seed": cfg.get("seed"),
                "train_rows": data.get("train_rows"),
                "val_accuracy": val.get("accuracy"),
                "val_f1_macro": val.get("f1_macro"),
                "val_precision_macro": val.get("precision_macro"),
                "val_recall_macro": val.get("recall_macro"),
                "val_f1_macro_tuned": tuned.get("f1_macro"),
                "val_threshold": tuned.get("threshold"),
                "test_accuracy": test.get("accuracy"),
                "test_f1_macro": test.get("f1_macro"),
                "model_mb": sizes.get("model_mb"),
                "tokenizer_mb": sizes.get("tokenizer_mb"),
                "total_artifact_mb": sizes.get("total_mb"),
                "runtime_sec": r.get("runtime_sec"),
                "notes": r.get("notes", ""),
            }
        )
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"==> results.json / results.csv updated ({len(merged)} runs)")
