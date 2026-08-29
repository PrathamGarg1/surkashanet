"""
Final export entry point.

Reads ``/vol/checkpoints/pt`` and writes staged ONNX FP32 + dynamic INT8.
Existing canonical directories are preserved as timestamped backups.

    modal run training/modal_export.py

For test evaluation + Chrome packaging use:

    modal run experiments/exp06_compress.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime

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
    dir_size_mb,
    dumps,
)


@app.function(image=EXPORT_IMAGE, volumes={VOLUME_MOUNT: VOLUME}, timeout=60 * 45)
def export_staged() -> str:
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    assert os.path.isdir(PT_CHECKPOINT_DIR), f"missing {PT_CHECKPOINT_DIR}"
    root = f"{VOLUME_MOUNT}/checkpoints/_export_staging"
    fp32 = os.path.join(root, "onnx_fp32")
    int8 = os.path.join(root, "onnx_int8")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(fp32, exist_ok=True)
    os.makedirs(int8, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(PT_CHECKPOINT_DIR)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        PT_CHECKPOINT_DIR, export=True
    )
    ort_model.save_pretrained(fp32)
    tokenizer.save_pretrained(fp32)
    quantizer = ORTQuantizer.from_pretrained(fp32)
    quantizer.quantize(
        save_dir=int8,
        quantization_config=AutoQuantizationConfig.avx2(
            is_static=False, per_channel=False
        ),
    )
    tokenizer.save_pretrained(int8)
    assert os.path.exists(os.path.join(fp32, "model.onnx"))
    assert os.path.exists(os.path.join(int8, "model_quantized.onnx"))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for target, staged in ((ONNX_FP32_DIR, fp32), (ONNX_INT8_DIR, int8)):
        backup = f"{target}.backup-{stamp}"
        if os.path.exists(target):
            shutil.move(target, backup)
        shutil.move(staged, target)
    shutil.rmtree(root, ignore_errors=True)
    VOLUME.commit()
    return dumps(
        {
            "onnx_fp32_dir": ONNX_FP32_DIR,
            "onnx_int8_dir": ONNX_INT8_DIR,
            "onnx_fp32_size_mb": dir_size_mb(ONNX_FP32_DIR),
            "onnx_int8_size_mb": dir_size_mb(ONNX_INT8_DIR),
            "backup_stamp": stamp,
            "test_evaluated": False,
        }
    )


@app.local_entrypoint()
def main():
    result = json.loads(export_staged.remote())
    reports = os.path.join(REPO_ROOT, "REPORTS")
    os.makedirs(reports, exist_ok=True)
    with open(os.path.join(reports, "size_comparison.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("Next: modal run experiments/exp06_compress.py")
