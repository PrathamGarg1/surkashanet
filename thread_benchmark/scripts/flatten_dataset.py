#!/usr/bin/env python3
"""Concatenate the per-cell files dataset/len_*/<script>/<class>/{train,val,test}.jsonl
into dataset/all/{train,val,test}.jsonl (the layout train_v2.py reads).

build_dataset.py produces the same files, but it rebuilds from the Bedrock generation cache
(not part of the repository) and deletes dataset/len_* first. Use this script instead.
"""
from __future__ import annotations

from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "dataset"
OUT = DATASET / "all"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        n = 0
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for f in sorted(DATASET.glob(f"len_*/*/*/{split}.jsonl")):
                for line in f.open(encoding="utf-8"):
                    out.write(line)
                    n += 1
        print(f"{split}: {n} threads")


if __name__ == "__main__":
    main()
