"""
Experiment 01 — notebook reproduction + clean baseline.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _exp_common import (  # noqa: E402
    BASE_CONFIG,
    app,
    append_results,
    save_config,
    save_log,
    train_one,
)

CONFIGS = [
    {
        **BASE_CONFIG,
        "run_name": "b0_notebook_defaults",
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 64,
        "max_length": 512,
        "fp16": False,
    },
    {
        **BASE_CONFIG,
        "run_name": "b1_clean_baseline",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 128,
        "max_length": 128,
        "fp16": True,
    },
]


@app.local_entrypoint()
def main():
    for cfg in CONFIGS:
        save_config(cfg["run_name"], cfg)
    results = [json.loads(r) for r in train_one.map(CONFIGS)]
    ok = [r for r in results if "error" not in r]
    for r in results:
        save_log(f"exp01_{r['run_name']}", r)
    append_results(ok, experiment="exp01_baseline", phase="baseline")
    print("=== EXPERIMENT 01: BASELINES ===")
    for r in ok:
        print(
            f"{r['run_name']}: acc={r['val']['accuracy']*100:.2f}% "
            f"f1={r['val']['f1_macro']*100:.2f}%"
        )
