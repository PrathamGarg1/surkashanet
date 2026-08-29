"""
Experiment 03 — random hyperparameter search (validation only).
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

SEARCH_SEED = 20260828


def sample_configs(n: int = 28):
    import numpy as np

    rng = np.random.default_rng(SEARCH_SEED)
    lrs = [1e-5, 2e-5, 3e-5, 5e-5]
    epochs = [2, 3, 4, 5]
    batches = [16, 32, 64]
    wds = [0.0, 0.01]
    warmups = [0.0, 0.1]
    schedulers = ["linear", "cosine"]
    lengths = [96, 128, 192]
    configs = []
    for i in range(n):
        lr = float(rng.choice(lrs))
        ep = int(rng.choice(epochs))
        bs = int(rng.choice(batches))
        wd = float(rng.choice(wds))
        wu = float(rng.choice(warmups))
        sch = str(rng.choice(schedulers))
        ml = int(rng.choice(lengths))
        name = f"h{i:02d}_lr{lr:g}_ep{ep}_bs{bs}_wd{wd}_wu{wu}_{sch[:3]}_len{ml}"
        configs.append(
            {
                **BASE_CONFIG,
                "run_name": name,
                "learning_rate": lr,
                "num_train_epochs": ep,
                "per_device_train_batch_size": bs,
                "weight_decay": wd,
                "warmup_ratio": wu,
                "lr_scheduler_type": sch,
                "max_length": ml,
                "fp16": True,
            }
        )
    return configs


@app.local_entrypoint()
def main(mode: str = "random", n_trials: int = 28):
    if mode != "random":
        raise SystemExit("Only --mode random is supported in this checkout")
    configs = sample_configs(n_trials)
    for cfg in configs:
        save_config(cfg["run_name"], cfg)
    results = [json.loads(r) for r in train_one.map(configs)]
    ok = [r for r in results if "error" not in r]
    for r in results:
        save_log(f"exp03_{r['run_name']}", r)
    append_results(ok, experiment="exp03_hparam_search", phase="search")
    print("=== EXPERIMENT 03: RANDOM SEARCH ===")
    for r in sorted(ok, key=lambda x: -x["val"]["f1_macro"])[:10]:
        print(
            f"{r['run_name']}: acc={r['val']['accuracy']*100:.2f}% "
            f"f1={r['val']['f1_macro']*100:.2f}%"
        )
