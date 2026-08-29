"""
Final reproducible training entry point.

Reads ``experiments/configs/final_winner.json`` when present, otherwise the
clean validation baseline. Training is delegated to the shared Hugging Face
``Trainer`` harness in ``experiments/_exp_common.py``.

    modal run training/modal_train.py

This command never loads ``hindi_test.csv``.
"""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

from _exp_common import (  # noqa: E402
    BASE_CONFIG,
    PT_CHECKPOINT_DIR,
    app,
    append_results,
    save_config,
    save_log,
    train_one,
)


def selected_config() -> dict:
    path = os.path.join(REPO_ROOT, "experiments", "configs", "final_winner.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        **BASE_CONFIG,
        "run_name": "pipeline_clean_baseline",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "max_length": 128,
    }


@app.local_entrypoint()
def main():
    cfg = {**BASE_CONFIG, **selected_config()}
    cfg.update(
        {
            "run_name": "pipeline_train",
            "save_to": PT_CHECKPOINT_DIR,
            "allow_test": False,
            "test_reason": None,
        }
    )
    save_config("pipeline_train", cfg)
    result = json.loads(train_one.remote(cfg))
    if "error" in result:
        raise RuntimeError(result["error"])
    save_log("pipeline_train", result)
    append_results([result], experiment="training/modal_train", phase="pipeline")
    print(f"saved PyTorch checkpoint: {PT_CHECKPOINT_DIR}")
    print(f"validation: {json.dumps(result['val'], indent=2)}")
