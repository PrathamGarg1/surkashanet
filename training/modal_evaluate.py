"""
Final held-out evaluation.

Requires the frozen checkpoint marker from ``training/modal_train.py`` and
loads MACD ``hindi_test.csv`` only after training/selection are complete.

    modal run training/modal_evaluate.py
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

from _exp_common import (  # noqa: E402
    EXPORT_IMAGE,
    ONNX_FP32_DIR,
    ONNX_INT8_DIR,
    PT_CHECKPOINT_DIR,
    VOLUME,
    VOLUME_MOUNT,
    app,
    dumps,
    full_report,
    load_macd,
)


@app.function(
    image=EXPORT_IMAGE,
    gpu="A10G",
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=60 * 30,
)
def evaluate_frozen() -> str:
    import numpy as np
    import torch
    from datasets import Dataset
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    marker = os.path.join(PT_CHECKPOINT_DIR, "experiment_config.json")
    assert os.path.exists(marker), "checkpoint is not marked as a frozen experiment"
    with open(marker) as f:
        max_length = int(json.load(f).get("max_length", 96))
    for path in (PT_CHECKPOINT_DIR, ONNX_FP32_DIR, ONNX_INT8_DIR):
        assert os.path.isdir(path), f"missing {path}; train and export first"

    df = load_macd("hindi_test")

    def make_dataset(tokenizer):
        ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
            batched=True,
        ).remove_columns(["text"])

    def predict(model, tokenizer):
        ds = make_dataset(tokenizer)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if hasattr(model, "eval"):
            model.eval()
        out = []
        for start in range(0, len(ds), 128):
            batch = collator([ds[i] for i in range(start, min(start + 128, len(ds)))])
            with torch.no_grad():
                out.extend(torch.argmax(model(**batch).logits, dim=-1).tolist())
        return full_report(df["label"].to_numpy(), np.asarray(out))

    pt_tokenizer = AutoTokenizer.from_pretrained(PT_CHECKPOINT_DIR)
    pt = AutoModelForSequenceClassification.from_pretrained(PT_CHECKPOINT_DIR)
    pt_score = predict(pt, pt_tokenizer)
    fp32_tokenizer = AutoTokenizer.from_pretrained(ONNX_FP32_DIR)
    fp32 = ORTModelForSequenceClassification.from_pretrained(ONNX_FP32_DIR)
    fp32_score = predict(fp32, fp32_tokenizer)
    int8_tokenizer = AutoTokenizer.from_pretrained(ONNX_INT8_DIR)
    int8 = ORTModelForSequenceClassification.from_pretrained(
        ONNX_INT8_DIR, file_name="model_quantized.onnx"
    )
    int8_score = predict(int8, int8_tokenizer)

    return dumps(
        {
            "dataset": "MACD hindi_test.csv",
            "n": len(df),
            "test_reason": "frozen validation-selected checkpoint; evaluation-only command",
            "pytorch": pt_score,
            "onnx_fp32": fp32_score,
            "onnx_int8": int8_score,
        }
    )


@app.local_entrypoint()
def main():
    result = json.loads(evaluate_frozen.remote())
    path = os.path.join(REPO_ROOT, "REPORTS", "eval_metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"==> wrote {path}")
