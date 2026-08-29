"""
Experiment 07 — training-only vocabulary pruning (validation only).
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _exp_common import (  # noqa: E402
    EXPORT_IMAGE,
    MODEL_NAME,
    PRUNED_DIR,
    PT_CHECKPOINT_DIR,
    VOLUME,
    VOLUME_MOUNT,
    app,
    dumps,
    full_report,
    load_davidson_full,
    load_macd,
)


@app.function(image=EXPORT_IMAGE, volumes={VOLUME_MOUNT: VOLUME}, timeout=60 * 60)
def prune_and_validate() -> str:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    assert os.path.isdir(PT_CHECKPOINT_DIR), f"missing {PT_CHECKPOINT_DIR}"
    tokenizer = AutoTokenizer.from_pretrained(PT_CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(PT_CHECKPOINT_DIR)
    old_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    max_length = 96
    config_path = os.path.join(PT_CHECKPOINT_DIR, "experiment_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            max_length = int(json.load(f).get("max_length", max_length))

    train_texts = load_macd("hindi_train")["text"].astype(str).tolist()
    train_texts += load_davidson_full()["text"].astype(str).tolist()
    keep_pieces = set(tokenizer.all_special_tokens)
    for text in train_texts:
        keep_pieces.update(
            tokenizer.convert_ids_to_tokens(
                tokenizer.encode(text, add_special_tokens=False)
            )
        )
    vocab = tokenizer.get_vocab()
    for piece in vocab:
        if len(piece) == 1:
            o = ord(piece)
            if 0x0900 <= o <= 0x097F or o <= 0x024F:
                keep_pieces.add(piece)
        if piece in ("<unk>", "<s>", "</s>", "<pad>", "▁") or piece.startswith("▁"):
            keep_pieces.add(piece)

    with open(os.path.join(PT_CHECKPOINT_DIR, "tokenizer.json")) as f:
        tokenizer_json = json.load(f)
    if tokenizer_json.get("model", {}).get("type") != "Unigram":
        raise RuntimeError("expected SentencePiece Unigram tokenizer")

    old_vocab = tokenizer_json["model"]["vocab"]
    new_vocab = [pair for pair in old_vocab if pair[0] in keep_pieces]
    old_piece_to_id = vocab
    new_piece_to_id = {piece: i for i, (piece, _) in enumerate(new_vocab)}
    old_embeddings = model.get_input_embeddings().weight.detach()
    new_embeddings = torch.zeros(
        len(new_vocab), old_embeddings.shape[1], dtype=old_embeddings.dtype
    )
    for new_id, (piece, _) in enumerate(new_vocab):
        new_embeddings[new_id] = old_embeddings[old_piece_to_id[piece]]
    model.get_input_embeddings().weight = torch.nn.Parameter(new_embeddings)
    model.config.vocab_size = len(new_vocab)

    if os.path.exists(PRUNED_DIR):
        shutil.rmtree(PRUNED_DIR)
    os.makedirs(PRUNED_DIR, exist_ok=True)
    model.save_pretrained(PRUNED_DIR)
    tokenizer_json["model"]["vocab"] = new_vocab
    for added in tokenizer_json.get("added_tokens", []):
        content = added.get("content")
        if content in new_piece_to_id:
            added["id"] = new_piece_to_id[content]
    with open(os.path.join(PRUNED_DIR, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    for name in ("tokenizer_config.json", "special_tokens_map.json", "config.json"):
        src = os.path.join(PT_CHECKPOINT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(PRUNED_DIR, name))
    with open(os.path.join(PRUNED_DIR, "config.json")) as f:
        model_cfg = json.load(f)
    model_cfg["vocab_size"] = len(new_vocab)
    with open(os.path.join(PRUNED_DIR, "config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)

    pruned_tokenizer = AutoTokenizer.from_pretrained(PRUNED_DIR)
    pruned_model = AutoModelForSequenceClassification.from_pretrained(PRUNED_DIR)
    pruned_model.eval()
    val = load_macd("hindi_val")
    ds = Dataset.from_pandas(val[["text", "label"]], preserve_index=False)
    ds = ds.map(
        lambda b: pruned_tokenizer(b["text"], truncation=True, max_length=max_length),
        batched=True,
    ).remove_columns(["text"])
    collator = DataCollatorWithPadding(tokenizer=pruned_tokenizer)
    preds = []
    for start in range(0, len(ds), 128):
        batch = collator([ds[i] for i in range(start, min(start + 128, len(ds)))])
        with torch.no_grad():
            preds.extend(torch.argmax(pruned_model(**batch).logits, dim=-1).tolist())
    val_report = full_report(val["label"].to_numpy(), np.asarray(preds))
    VOLUME.commit()
    return dumps(
        {
            "model_name": MODEL_NAME,
            "source_checkpoint": PT_CHECKPOINT_DIR,
            "pruned_checkpoint": PRUNED_DIR,
            "original_vocab": old_vocab_size,
            "pruned_vocab": len(new_vocab),
            "vocab_reduction_pct": round(100 * (1 - len(new_vocab) / old_vocab_size), 2),
            "val": val_report,
            "max_length": max_length,
            "test_loaded": False,
        }
    )


@app.local_entrypoint()
def main():
    out = json.loads(prune_and_validate.remote())
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs", "exp07_vocab_prune.json"
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"==> wrote {path}")
