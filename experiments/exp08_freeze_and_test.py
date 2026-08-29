"""
Experiment 08 — freeze the best validation config and evaluate hindi_test once.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _exp_common import (  # noqa: E402
    BASE_CONFIG,
    PT_CHECKPOINT_DIR,
    RESULTS_JSON,
    app,
    append_results,
    save_config,
    save_log,
    train_one,
)

# Validation-selected winner from the prior Modal random search campaign.
# Re-confirmed by re-selecting max(val.f1_macro) when results.json is present.
WINNER = {
    **BASE_CONFIG,
    "run_name": "final_winner",
    "learning_rate": 5e-5,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 128,
    "max_length": 96,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "linear",
    "fp16": True,
}


@app.local_entrypoint()
def main():
    cfg = dict(WINNER)
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            records = json.load(f)
        # Only consider trainable search/ablation/freeze rows — never compression
        # artifacts, which lack full Trainer hyperparameters.
        trainable_keys = (
            "learning_rate",
            "num_train_epochs",
            "per_device_train_batch_size",
        )
        candidates = [
            r
            for r in records
            if "val" in r
            and "config" in r
            and "error" not in r
            and r.get("phase") not in {"compression"}
            and all(k in r["config"] for k in trainable_keys)
        ]
        if candidates:
            selected = max(candidates, key=lambda r: r["val"]["f1_macro"])
            cfg = {**BASE_CONFIG, **selected["config"]}
            print(f"selected from registry: {selected['run_name']}")
        else:
            print("no trainable registry candidates; using frozen WINNER defaults")
    cfg.update(
        {
            "run_name": "final_winner",
            "save_to": PT_CHECKPOINT_DIR,
            "allow_test": True,
            "test_reason": (
                "configuration frozen after validation-only model selection; "
                "no test metrics were available during selection"
            ),
        }
    )
    save_config("final_winner", cfg)
    result = json.loads(train_one.remote(cfg))
    if "error" in result:
        raise RuntimeError(result["error"])
    save_log("final_evaluation", result)
    append_results([result], experiment="exp08_freeze_and_test", phase="frozen_test")
    print("=== FROZEN FINAL EVALUATION ===")
    print(json.dumps(result["val"], indent=2))
    print(json.dumps(result["macd_test"], indent=2))
    print(f"checkpoint: {result.get('saved_to')}")
