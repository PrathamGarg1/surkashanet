"""
Experiment 00 — train/val data audit (never loads hindi_test).
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _exp_common import (  # noqa: E402
    TRAIN_IMAGE,
    app,
    davidson_split,
    dumps,
    load_davidson_full,
    load_macd,
    save_log,
)


@app.function(image=TRAIN_IMAGE, timeout=60 * 30)
def audit() -> str:
    from collections import Counter

    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    macd_train = load_macd("hindi_train")
    macd_val = load_macd("hindi_val")
    dav_full = load_davidson_full()
    dav_train, dav_hold, _ = davidson_split(dav_full)

    def describe(name, df):
        texts = df["text"].astype(str).tolist()
        lengths = [len(x) for x in tokenizer(texts, truncation=False)["input_ids"]]
        lengths = np.asarray(lengths)
        labels = Counter(df["label"].tolist())
        duplicates = int(df["text"].duplicated().sum())
        return {
            "name": name,
            "rows": len(df),
            "label_counts": dict(labels),
            "duplicates": duplicates,
            "token_length": {
                "mean": float(lengths.mean()),
                "p50": float(np.percentile(lengths, 50)),
                "p95": float(np.percentile(lengths, 95)),
                "p99": float(np.percentile(lengths, 99)),
                "max": int(lengths.max()),
            },
            "devanagari_share": float(
                np.mean([bool(re.search(r"[\u0900-\u097F]", t)) for t in texts])
            ),
            "latin_share": float(
                np.mean([bool(re.search(r"[A-Za-z]", t)) for t in texts])
            ),
        }

    train_set = set(macd_train["text"].astype(str))
    val_set = set(macd_val["text"].astype(str))
    report = {
        "splits": [
            describe("macd_train", macd_train),
            describe("macd_val", macd_val),
            describe("davidson_train", dav_train),
            describe("davidson_holdout", dav_hold),
        ],
        "leakage_train_val_overlap": len(train_set & val_set),
        "note": "hindi_test was not loaded",
    }
    return dumps(report)


@app.local_entrypoint()
def main():
    report = json.loads(audit.remote())
    path = save_log("exp00_data_audit", report)
    print(json.dumps(report, indent=2)[:4000])
    print(f"==> wrote {path}")
