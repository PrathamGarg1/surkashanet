"""
Experiment 06 — ONNX FP32 + dynamic INT8 export and Chrome packaging.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _exp_common import (  # noqa: E402
    EXPORT_IMAGE,
    MODEL_NAME,
    ONNX_FP32_DIR,
    ONNX_INT8_DIR,
    PT_CHECKPOINT_DIR,
    VOLUME,
    VOLUME_MOUNT,
    app,
    append_results,
    dir_size_mb,
    dumps,
    file_size_mb,
    full_report,
    load_macd,
)

EXTENSION_FILES = [
    ("model_quantized.onnx", "onnx/model_quantized.onnx"),
    ("config.json", "config.json"),
    ("ort_config.json", "ort_config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
]


@app.function(image=EXPORT_IMAGE, volumes={VOLUME_MOUNT: VOLUME}, timeout=60 * 60)
def export_frozen(checkpoint_dir: str = PT_CHECKPOINT_DIR) -> str:
    import numpy as np
    import onnx
    import torch
    from datasets import Dataset
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer, DataCollatorWithPadding

    assert os.path.isdir(checkpoint_dir), f"missing {checkpoint_dir}"
    staging_root = f"{VOLUME_MOUNT}/experiments/compression/frozen_candidate"
    fp32_dir = f"{staging_root}/onnx_fp32"
    int8_dir = f"{staging_root}/onnx_int8"
    for d in (fp32_dir, int8_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    max_length = 96
    experiment_config = os.path.join(checkpoint_dir, "experiment_config.json")
    if os.path.exists(experiment_config):
        with open(experiment_config) as f:
            max_length = int(json.load(f).get("max_length", max_length))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        checkpoint_dir, export=True
    )
    ort_model.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(fp32_dir)
    fp32_file = os.path.join(fp32_dir, "model.onnx")
    assert os.path.exists(fp32_file)

    quantizer = ORTQuantizer.from_pretrained(fp32_dir)
    quantizer.quantize(
        save_dir=int8_dir,
        quantization_config=AutoQuantizationConfig.avx2(
            is_static=False, per_channel=False
        ),
    )
    tokenizer.save_pretrained(int8_dir)
    int8_file = os.path.join(int8_dir, "model_quantized.onnx")
    assert os.path.exists(int8_file)

    graph = onnx.load(int8_file)
    input_names = [x.name for x in graph.graph.input]
    output_names = [x.name for x in graph.graph.output]
    assert "input_ids" in input_names and "attention_mask" in input_names

    def eval_ort(model_dir, split, quantized=False):
        kwargs = {"file_name": "model_quantized.onnx"} if quantized else {}
        model = ORTModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
        df = load_macd(split)
        ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        ds = ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
            batched=True,
        ).remove_columns(["text"])
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        preds = []
        for start in range(0, len(ds), 128):
            batch = collator([ds[i] for i in range(start, min(start + 128, len(ds)))])
            with torch.no_grad():
                logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return full_report(df["label"].to_numpy(), np.asarray(preds))

    scores = {
        "fp32_val": eval_ort(fp32_dir, "hindi_val"),
        "int8_val": eval_ort(int8_dir, "hindi_val", quantized=True),
        "fp32_test": eval_ort(fp32_dir, "hindi_test"),
        "int8_test": eval_ort(int8_dir, "hindi_test", quantized=True),
    }
    sizes = {
        "pt_checkpoint_mb": dir_size_mb(checkpoint_dir),
        "onnx_fp32_mb": dir_size_mb(fp32_dir),
        "onnx_int8_mb": dir_size_mb(int8_dir),
        "onnx_fp32_model_mb": file_size_mb(fp32_file),
        "onnx_int8_model_mb": file_size_mb(int8_file),
        "compression_fp32_to_int8": round(
            file_size_mb(fp32_file) / file_size_mb(int8_file), 3
        ),
    }

    artifacts = {}
    for src_name, target in EXTENSION_FILES:
        path = os.path.join(int8_dir, src_name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            artifacts[target] = base64.b64encode(f.read()).decode("ascii")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = {}
    for target, staged in ((ONNX_FP32_DIR, fp32_dir), (ONNX_INT8_DIR, int8_dir)):
        backup = f"{target}.backup-{stamp}"
        if os.path.exists(target):
            shutil.move(target, backup)
            backups[target] = backup
        shutil.move(staged, target)
    shutil.rmtree(staging_root, ignore_errors=True)
    VOLUME.commit()
    return dumps(
        {
            "model_name": MODEL_NAME,
            "checkpoint": checkpoint_dir,
            "scores": scores,
            "sizes": sizes,
            "previous_exports": backups,
            "test_reason": "frozen winner selected by MACD hindi_val before export",
            "max_length": max_length,
            "artifacts": artifacts,
        }
    )


@app.local_entrypoint()
def main(checkpoint_dir: str = PT_CHECKPOINT_DIR):
    out = json.loads(export_frozen.remote(checkpoint_dir=checkpoint_dir))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    live_model = os.path.join(repo_root, "assets", "models", "custom-macd-model")
    candidate = os.path.join(repo_root, "assets", "models", "custom-macd-model-candidate")
    backup_dir = os.path.join(
        repo_root,
        "assets",
        "models",
        f"custom-macd-model-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    if os.path.exists(candidate):
        shutil.rmtree(candidate)
    os.makedirs(candidate, exist_ok=True)
    for rel, encoded in out["artifacts"].items():
        target = os.path.join(candidate, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(base64.b64decode(encoded))

    if os.path.exists(live_model):
        shutil.copytree(live_model, backup_dir)
        shutil.rmtree(live_model)
    shutil.copytree(candidate, live_model)
    try:
        subprocess.run(["npm", "run", "build"], cwd=repo_root, check=True)
    except Exception:
        if os.path.exists(live_model):
            shutil.rmtree(live_model)
        shutil.copytree(backup_dir, live_model)
        raise
    shutil.rmtree(candidate)
    out["local_extension_size_mb"] = round(
        sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(live_model)
            for fn in fns
        )
        / (1024 * 1024),
        3,
    )
    out.pop("artifacts", None)
    with open(os.path.join(repo_root, "experiments", "logs", "exp06_compression.json"), "w") as f:
        json.dump(out, f, indent=2)
    append_results(
        [
            {
                "run_name": "final_winner_int8",
                "config": {"model_name": out["model_name"], "run_name": "final_winner_int8"},
                "val": out["scores"]["int8_val"],
                "macd_test": out["scores"]["int8_test"],
                "sizes": {
                    "model_mb": out["sizes"]["onnx_int8_model_mb"],
                    "tokenizer_mb": round(
                        max(0.0, out["sizes"]["onnx_int8_mb"] - out["sizes"]["onnx_int8_model_mb"]),
                        3,
                    ),
                    "total_mb": out["local_extension_size_mb"],
                },
                "test_reason": out["test_reason"],
                "notes": "ONNX dynamic INT8; browser build passed",
            }
        ],
        experiment="exp06_compress",
        phase="compression",
    )
    print(json.dumps({"scores": out["scores"], "sizes": out["sizes"]}, indent=2))
    print(f"extension size: {out['local_extension_size_mb']} MB")
