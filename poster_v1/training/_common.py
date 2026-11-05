"""
Shared Modal Image, Volume, dataset loaders, and constants for the
SurakshaNet training pipeline.

The three top-level scripts in this folder
  - modal_train.py
  - modal_export.py
  - modal_evaluate.py
all import from this module so there is exactly one source of truth for
dependency versions, dataset URLs, label conventions, and the volume layout.

Strict invariant on dataset access:
  - load_macd_train_val()  may be called from train scripts
  - load_macd_test()       must ONLY be called from modal_evaluate.py
  - load_davidson_full()   loads the entire Davidson CSV. Train and evaluate
                           split it deterministically using DAVIDSON_SEED, and
                           the held-out indices are written to / read from
                           DAVIDSON_INDICES_FILE on the shared volume so train
                           and evaluate cannot disagree.
"""

from __future__ import annotations

import os as _os

import modal


# ── Modal app + image ────────────────────────────────────────────────────────

APP_NAME = "surakshanet"
VOLUME_NAME = "surakshanet-artifacts"

# Mount this file itself into the image so the three scripts can
# `from _common import ...` inside the container. Modal's automount only
# resolves imports it can statically follow from sys.path, and our scripts
# adjust sys.path at runtime, so we add this file explicitly.
_COMMON_PATH = _os.path.abspath(__file__)

IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.2",
        "datasets==2.16.1",
        "optimum[onnxruntime]==1.16.1",
        "accelerate==0.25.0",
        "scikit-learn==1.3.2",
        "pandas==2.1.4",
        "sentencepiece==0.1.99",
        "protobuf==4.25.2",
        "onnx==1.16.0",
        "onnxruntime==1.16.3",
        "numpy==1.24.4",
        "matplotlib==3.8.2",
    )
    .add_local_file(_COMMON_PATH, remote_path="/root/_common.py")
)

VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_MOUNT = "/vol"


# ── Volume layout ────────────────────────────────────────────────────────────

PT_CHECKPOINT_DIR = f"{VOLUME_MOUNT}/checkpoints/pt"
ONNX_FP32_DIR = f"{VOLUME_MOUNT}/checkpoints/onnx_fp32"
ONNX_INT8_DIR = f"{VOLUME_MOUNT}/checkpoints/onnx_int8"
DAVIDSON_INDICES_FILE = f"{VOLUME_MOUNT}/davidson_test_indices.json"


# ── Model + training hyperparameters ─────────────────────────────────────────

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
DAVIDSON_SEED = 42
MAX_LENGTH = 128

# "Improved" recipe (the one that produced the 1.28%-behind-NeurIPS result)
TRAIN_HPARAMS = {
    "epochs": 7,
    "lr": 3e-5,
    "batch_size": 32,
    "eval_batch_size": 64,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "label_smoothing": 0.1,
    "metric_for_best_model": "f1_macro",
}


# ── Label conventions ────────────────────────────────────────────────────────
#
# Our shipped model uses:
#     id2label = {0: "abusive", 1: "non-abusive"}
#     label2id = {"abusive": 0, "non-abusive": 1}
#
# These mappings agree with the binary `class` column in the MACD GitHub CSVs
# (ShareChatAI/MACD), where 0 means abusive and 1 means non-abusive. The
# extension's service worker (src/background/service-worker.js) is wired to
# this convention via the TOXICITY_CONFIG.labels block, so we MUST keep it.
#
# Davidson 2017 uses a 3-way label:
#     0 = hate speech
#     1 = offensive language
#     2 = neither
# We collapse this to our binary scheme below.

ID2LABEL = {0: "abusive", 1: "non-abusive"}
LABEL2ID = {"abusive": 0, "non-abusive": 1}

DAVIDSON_LABEL_MAP = {
    0: 0,  # hate -> abusive
    1: 0,  # offensive -> abusive
    2: 1,  # neither -> non-abusive
}


# ── Dataset URLs ─────────────────────────────────────────────────────────────

MACD_BASE_URL = "https://raw.githubusercontent.com/ShareChatAI/MACD/main/dataset/"
DAVIDSON_URL = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)


# ── NeurIPS baseline (for the comparison table in final_report.md) ───────────
#
# The MACD paper (Maity et al., NeurIPS Datasets & Benchmarks 2022) reports
# XLM-RoBERTa-Large fine-tuned on the multilingual MACD train set as the
# reference baseline. Fill in the published numbers here once you have the
# paper open; the report generator uses None to mean "not configured".

NEURIPS_BASELINE = {
    "name": "MACD paper baseline (XLM-R Large)",
    "size_mb": 1700.0,
    "macd_hindi_test_accuracy": None,  # e.g. 0.8780
    "macd_hindi_test_f1_macro": None,  # e.g. 0.8650
}


# ── Dataset loaders (all run inside the container) ───────────────────────────


def _download_csv(url: str, local_path: str) -> str:
    """Download a CSV with a User-Agent header (GitHub raw blocks empty UA)."""
    import os
    import urllib.request

    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(local_path, "wb") as f:
        f.write(r.read())
    return local_path


def _load_macd_split(split: str):
    """Load one MACD Hindi split as a DataFrame with columns ['text', 'label']."""
    import pandas as pd

    assert split in ("hindi_train", "hindi_val", "hindi_test"), split
    local = f"/tmp/{split}.csv"
    _download_csv(f"{MACD_BASE_URL}{split}.csv", local)
    df = pd.read_csv(local)
    for old, new in [("comment_text", "text"), ("tweet", "text"), ("class", "label")]:
        if old in df.columns:
            df = df.rename(columns={old: new})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").dropna().astype(int)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    return df


def load_macd_train_val():
    """Return ``(train_df, val_df)``. Never reads hindi_test."""
    return _load_macd_split("hindi_train"), _load_macd_split("hindi_val")


def load_macd_test():
    """Return MACD hindi_test DataFrame. ONLY call from evaluate."""
    return _load_macd_split("hindi_test")


def load_davidson_full():
    """Return full Davidson DataFrame with columns ['text', 'label'] (binarised)."""
    import pandas as pd

    local = "/tmp/davidson.csv"
    _download_csv(DAVIDSON_URL, local)
    df = pd.read_csv(local)
    text_col = "tweet" if "tweet" in df.columns else "text"
    label_col = "class" if "class" in df.columns else "label"
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").dropna().astype(int)
    df["label"] = df["label"].map(DAVIDSON_LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    return df


def davidson_split(df, indices_file: str | None = None):
    """
    Split Davidson into 90% train / 10% test deterministically.

    If ``indices_file`` exists on disk, the held-out indices are loaded from
    it (so evaluate sees the same rows that train held out). Otherwise a
    fresh split is created with seed ``DAVIDSON_SEED`` and the held-out
    indices are returned alongside the split for the caller to persist.

    Returns ``(train_df, test_df, test_indices: list[int])``.
    """
    import json
    import os

    import numpy as np

    n = len(df)
    if indices_file is not None and os.path.exists(indices_file):
        with open(indices_file) as f:
            test_indices = sorted(int(i) for i in json.load(f))
    else:
        rng = np.random.default_rng(DAVIDSON_SEED)
        perm = rng.permutation(n)
        test_n = int(round(n * 0.10))
        test_indices = sorted(int(i) for i in perm[:test_n])

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_indices] = True
    test_df = df.iloc[test_mask].reset_index(drop=True)
    train_df = df.iloc[~test_mask].reset_index(drop=True)
    return train_df, test_df, test_indices


# ── Misc helpers ─────────────────────────────────────────────────────────────


def file_size_mb(path: str) -> float:
    import os

    return round(os.path.getsize(path) / (1024 * 1024), 3)


def dir_size_mb(path: str) -> float:
    import os

    total = 0
    for dp, _, fn in os.walk(path):
        for f in fn:
            total += os.path.getsize(os.path.join(dp, f))
    return round(total / (1024 * 1024), 3)
