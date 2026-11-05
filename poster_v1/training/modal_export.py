"""
SurakshaNet — Step 2/3: export PyTorch checkpoint to ONNX, then int8 quantize.

Run on Modal (CPU is fine — quantization is CPU-bound):
    modal run training/modal_export.py

Reads:
  - /vol/checkpoints/pt/                    (written by modal_train.py)

Writes:
  - /vol/checkpoints/onnx_fp32/             FP32 ONNX
  - /vol/checkpoints/onnx_int8/             INT8 dynamic-quantised ONNX
  - REPORTS/size_comparison.json            PyTorch FP32 / ONNX FP32 / ONNX INT8 sizes
  - assets/models/custom-macd-model/        the artifacts the extension loads

The quantization config (avx2, dynamic, per-tensor) matches the existing
shipped ort_config.json so the deployed runtime characteristics don't change.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    APP_NAME,
    IMAGE,
    ONNX_FP32_DIR,
    ONNX_INT8_DIR,
    PT_CHECKPOINT_DIR,
    VOLUME,
    VOLUME_MOUNT,
    dir_size_mb,
    file_size_mb,
)

app = modal.App(f"{APP_NAME}-export")

# Files we ship inside the extension.
#
# `@xenova/transformers` (which the Chrome service worker uses to run the
# model) follows the HuggingFace convention of placing ONNX weights inside
# an "onnx/" subfolder. With `quantized: true` (the default in the browser),
# it loads "<model_dir>/onnx/<name>_quantized.onnx". So we place the
# `.onnx` file under "onnx/" while keeping config/tokenizer at the root.
#
# Each entry is (filename in /vol/checkpoints/onnx_int8/, target relative
# path inside assets/models/custom-macd-model/).
EXTENSION_FILES: list[tuple[str, str]] = [
    ("model_quantized.onnx", "onnx/model_quantized.onnx"),
    ("config.json", "config.json"),
    ("ort_config.json", "ort_config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("special_tokens_map.json", "special_tokens_map.json"),
]


@app.function(
    image=IMAGE,
    volumes={VOLUME_MOUNT: VOLUME},
    timeout=60 * 30,
)
def export():
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    assert os.path.isdir(PT_CHECKPOINT_DIR), (
        f"missing PyTorch checkpoint at {PT_CHECKPOINT_DIR}; "
        "run `modal run training/modal_train.py` first"
    )

    # ── 1. Sizes for the PyTorch checkpoint we're about to convert ──────────
    pt_weights_file = None
    for cand in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(PT_CHECKPOINT_DIR, cand)
        if os.path.exists(p):
            pt_weights_file = p
            break
    assert pt_weights_file is not None, "no weights file in PT checkpoint"

    pt_size = {
        "weights_file": os.path.basename(pt_weights_file),
        "weights_file_size_mb": file_size_mb(pt_weights_file),
        "checkpoint_dir_size_mb": dir_size_mb(PT_CHECKPOINT_DIR),
    }

    # ── 2. Export to FP32 ONNX ──────────────────────────────────────────────
    if os.path.exists(ONNX_FP32_DIR):
        shutil.rmtree(ONNX_FP32_DIR)
    os.makedirs(ONNX_FP32_DIR, exist_ok=True)

    print(f"[export] PyTorch -> ONNX FP32 from {PT_CHECKPOINT_DIR}")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        PT_CHECKPOINT_DIR, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(PT_CHECKPOINT_DIR)
    ort_model.save_pretrained(ONNX_FP32_DIR)
    tokenizer.save_pretrained(ONNX_FP32_DIR)
    onnx_fp32_file = os.path.join(ONNX_FP32_DIR, "model.onnx")
    assert os.path.exists(onnx_fp32_file), "FP32 ONNX export failed"

    fp32_size = {
        "onnx_file": "model.onnx",
        "onnx_file_size_mb": file_size_mb(onnx_fp32_file),
        "dir_size_mb": dir_size_mb(ONNX_FP32_DIR),
    }

    # ── 3. INT8 dynamic quantization (avx2, per-tensor) ─────────────────────
    if os.path.exists(ONNX_INT8_DIR):
        shutil.rmtree(ONNX_INT8_DIR)
    os.makedirs(ONNX_INT8_DIR, exist_ok=True)

    print(f"[export] ONNX FP32 -> ONNX INT8 (avx2 dynamic, per-tensor)")
    quantizer = ORTQuantizer.from_pretrained(ONNX_FP32_DIR)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=ONNX_INT8_DIR, quantization_config=qconfig)
    tokenizer.save_pretrained(ONNX_INT8_DIR)

    onnx_int8_file = os.path.join(ONNX_INT8_DIR, "model_quantized.onnx")
    assert os.path.exists(onnx_int8_file), "INT8 quantization failed"

    int8_size = {
        "onnx_file": "model_quantized.onnx",
        "onnx_file_size_mb": file_size_mb(onnx_int8_file),
        "dir_size_mb": dir_size_mb(ONNX_INT8_DIR),
    }

    VOLUME.commit()

    size_comparison = {
        "pytorch_fp32": pt_size,
        "onnx_fp32": fp32_size,
        "onnx_int8": int8_size,
        "compression_pt_to_int8": round(
            pt_size["weights_file_size_mb"] / int8_size["onnx_file_size_mb"], 3
        ),
        "compression_onnx_fp32_to_int8": round(
            fp32_size["onnx_file_size_mb"] / int8_size["onnx_file_size_mb"], 3
        ),
    }
    print(f"[export] size_comparison: {json.dumps(size_comparison, indent=2)}")

    # ── 4. Read INT8 artifacts as bytes so the local entrypoint can drop
    #       them into assets/models/custom-macd-model/ ──────────────────────
    # Key = relative path inside the extension folder (so the local
    # entrypoint just writes verbatim, including the "onnx/" subdir).
    artifacts: dict[str, bytes] = {}
    for src_fn, rel_target in EXTENSION_FILES:
        p = os.path.join(ONNX_INT8_DIR, src_fn)
        if not os.path.exists(p):
            print(f"[export] WARNING: missing expected file {src_fn}; skipping")
            continue
        with open(p, "rb") as f:
            artifacts[rel_target] = f.read()
    print(f"[export] packaged {len(artifacts)} files for the extension")

    return {
        "size_comparison": size_comparison,
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main():
    print("==> running modal_export.export() on Modal…")
    out = export.remote()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(repo_root, "REPORTS")
    ext_model_dir = os.path.join(repo_root, "assets", "models", "custom-macd-model")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(ext_model_dir, exist_ok=True)

    with open(os.path.join(reports_dir, "size_comparison.json"), "w") as f:
        json.dump(out["size_comparison"], f, indent=2)

    written = []
    for rel_target, blob in out["artifacts"].items():
        target = os.path.join(ext_model_dir, rel_target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(blob)
        written.append(target)

    print("\n==> wrote:")
    print(f"  {reports_dir}/size_comparison.json")
    for t in written:
        print(f"  {t}")
    print("\n==> next: modal run training/modal_evaluate.py")
