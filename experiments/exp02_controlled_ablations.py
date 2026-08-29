"""
Experiment 02 — one-variable ablations from the clean baseline.
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
    "per_device_eval_batch_size": 128,
    "max_length": 128,
    "padding": "dynamic",
}

VARIANTS = [
    ("a_lr1e-5", {"learning_rate": 1e-5}),
    ("a_lr3e-5", {"learning_rate": 3e-5}),
    ("a_lr5e-5", {"learning_rate": 5e-5}),
    ("a_ep2", {"num_train_epochs": 2}),
    ("a_ep4", {"num_train_epochs": 4}),
    ("a_ep5", {"num_train_epochs": 5}),
    ("a_bs16", {"per_device_train_batch_size": 16}),
    ("a_bs64", {"per_device_train_batch_size": 64}),
    ("a_ga2", {"gradient_accumulation_steps": 2}),
    ("a_ga4", {"gradient_accumulation_steps": 4}),
    ("a_wd0.01", {"weight_decay": 0.01}),
    ("a_warmup0.1", {"warmup_ratio": 0.1}),
    ("a_cosine", {"lr_scheduler_type": "cosine"}),
    ("a_len64", {"max_length": 64}),
    ("a_len96", {"max_length": 96}),
    ("a_len192", {"max_length": 192}),
    ("a_len256", {"max_length": 256}),
    ("a_padmax", {"padding": "max_length"}),
    ("a_groupbylen", {"group_by_length": True}),
    ("a_classweights", {"class_weights": True}),
    ("a_ls0.05", {"label_smoothing": 0.05}),
    ("a_ls0.1", {"label_smoothing": 0.1}),
]

CONFIGS = [{**CONTROL, "run_name": name, **updates} for name, updates in VARIANTS]


@app.local_entrypoint()
def main():
    for cfg in CONFIGS:
        save_config(cfg["run_name"], cfg)
    results = [json.loads(r) for r in train_one.map(CONFIGS)]
    ok = [r for r in results if "error" not in r]
    for r in results:
        save_log(f"exp02_{r['run_name']}", r)
    append_results(ok, experiment="exp02_controlled_ablations", phase="ablation")
    print("=== EXPERIMENT 02: CONTROLLED ABLATIONS ===")
    for r in sorted(ok, key=lambda x: -x["val"]["f1_macro"]):
        print(
            f"{r['run_name']:20s} acc={r['val']['accuracy']*100:.2f}% "
            f"f1={r['val']['f1_macro']*100:.2f}%"
        )
    for r in results:
        if "error" in r:
            print(f"{r['run_name']} FAILED: {r['error']}")
