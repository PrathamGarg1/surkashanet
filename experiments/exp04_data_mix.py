"""
Experiment 04 — data-mix and two-stage studies (validation only).
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

CONTROL = {
    **BASE_CONFIG,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "max_length": 128,
}

CONFIGS = [
    {**CONTROL, "run_name": "d_macd_only", "data_mix": "macd_only"},
    {**CONTROL, "run_name": "d_dav25", "davidson_fraction": 0.25},
    {**CONTROL, "run_name": "d_dav50", "davidson_fraction": 0.5},
    {**CONTROL, "run_name": "d_dedup", "drop_train_duplicates": True},
    {
        **CONTROL,
        "run_name": "d_twostage_2p2",
        "num_train_epochs": 2,
        "stage2": {"num_train_epochs": 2, "learning_rate": 1e-5},
    },
    {
        **CONTROL,
        "run_name": "d_twostage_2p3_lr2e5",
        "num_train_epochs": 2,
        "stage2": {"num_train_epochs": 3, "learning_rate": 2e-5},
    },
    {
        **CONTROL,
        "run_name": "d_dav_then_macd",
        "data_mix": "davidson_only",
        "num_train_epochs": 2,
        "stage2": {"num_train_epochs": 2, "learning_rate": 1e-5},
    },
]


@app.local_entrypoint()
def main():
    for cfg in CONFIGS:
        save_config(cfg["run_name"], cfg)
    results = [json.loads(r) for r in train_one.map(CONFIGS)]
    ok = [r for r in results if "error" not in r]
    for r in results:
        save_log(f"exp04_{r['run_name']}", r)
    append_results(ok, experiment="exp04_data_mix", phase="data")
    print("=== EXPERIMENT 04: DATA MIX ===")
    for r in sorted(ok, key=lambda x: -x["val"]["f1_macro"]):
        print(
            f"{r['run_name']}: rows={r['data']['train_rows']} "
            f"acc={r['val']['accuracy']*100:.2f}% f1={r['val']['f1_macro']*100:.2f}%"
        )
