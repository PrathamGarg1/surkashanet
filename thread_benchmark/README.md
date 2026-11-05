# Five-class thread benchmark

A synthetic dataset of 105,000 chat threads in English, Hindi (Devanagari) and Hinglish, and the models trained on it. The task is to classify the **last message given the earlier ones** into benign, friendly sarcasm, hostile abuse, sexual harassment or predatory grooming (annotation rules: [`LABELING_GUIDE.md`](LABELING_GUIDE.md)).

Results and open items: [`RESULTS.md`](RESULTS.md). Dataset description: [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md).

## Layout

| Path | Content |
|---|---|
| `dataset/len_*/<script>/<class>/{train,val,test}.jsonl` | the 105,000 threads (60 cells of 1,750), `MANIFEST.csv` with counts |
| `scripts/dataset_spec.py` | scenario space: 31 subtypes, 40 topics, names, audiences, same-last-line bank |
| `scripts/generate_threads.py`, `logged_bedrock.py`, `cost_ledger.py` | thread generation with Qwen3-32B on Amazon Bedrock, resumable, spend cap, per-call ledger |
| `scripts/build_dataset.py` | validators, de-duplication, split, manifest (rebuilds from the generation cache; see note below) |
| `scripts/flatten_dataset.py` | writes `dataset/all/{train,val,test}.jsonl` from the stored per-cell files |
| `scripts/train_v2.py`, `train_multiclass.py`, `run_training.sh` | fine-tuning of the thread model and of the last-message-only control |
| `gru/` | frozen MiniLM + GRU: line embedding, training, JSON export, plain-JS runtime, parity test |
| `quant/` | ONNX export, dynamic INT8 quantization, INT8 evaluation, latency measurement and their reports |
| `scripts/benchmark_threads.py`, `benchmark_v2.py` | accuracy and macro-F1 by length, script, class, and on same-last-line pairs |
| `external_eval/` | zero-shot check on real data (CAD, ConvAbuse, PRISM) |
| `cloud/` | EC2 (g4dn.xlarge) launch script and cost record of the GPU training run |
| `logs/` | cost ledger and run logs |
| `viewer/app.py` | local dataset browser (pick length, script, class; chat-style view) |

## Method in short

1. **Scenario sampling.** A seeded sampler picks a class-specific subtype, a topic, two speaker names, the audience and, where relevant, the target's gender and the child's age. About half of the sarcasm and hostile threads are assigned a shared last line from a bank of 36 lines per script.
2. **Generation.** One Qwen3-32B call per scenario returns exactly *k* alternating messages as JSON; two attempts per scenario.
3. **Validation.** Turn count, refusals, turn length, adjacent duplicates, last line repeated in the context, script, profanity or explicit words outside the hostile class, sexualised words in grooming threads; thread-hash de-duplication. In the last generation round 98.6% of attempts passed.
4. **Split.** 80/10/10 by scenario hash; same-last-line pair lines are split by line index, so no pair line appears in two splits.
5. **Models.** (A) MiniLM-L12 fine-tuned on `last [SEP] earlier turns (most recent first, [TURN]-separated)`, 2 epochs, learning rate 4e-5, batch 32, fp16, word-embedding table frozen; (B) the same recipe on the last message only; (C) frozen MiniLM encodes each line once (384-d), a one-layer GRU (hidden 128) reads the sequence, 198,021 trained weights.
6. **Export.** Dynamic INT8 quantization with ONNX Runtime for A and for the encoder of C.

## Reproducing

```bash
pip install -r ../requirements.txt
python scripts/flatten_dataset.py                          # dataset/all/*.jsonl
python scripts/train_v2.py --mode thread   --epochs 2      # writes artifacts_v2/model_thread
python scripts/train_v2.py --mode isolated --epochs 2      # last-message-only control
python gru/embed_lines.py && python gru/train_gru.py       # frozen encoder + GRU
node gru/parity_test.mjs
```

`quant/export_quantize.py`, `quant/eval_quant.py` and `external_eval/eval_external.py` read the cloud run's models from `artifacts_v2/cloud_results/model_{thread,isolated}`; copy the trained models there (or point the scripts at your own run). The GPU run used `cloud/launch_train_ec2.sh` (g4dn.xlarge, 19.8 minutes, about USD 0.18).

Notes:
- `scripts/build_dataset.py` rebuilds the dataset from the Bedrock generation cache (not part of this repository) and **deletes `dataset/len_*` first**. Do not run it without the cache; use `flatten_dataset.py` instead.
- Generation needs AWS credentials with Amazon Bedrock access (`us-east-1`) and costs about USD 22 for 105,000 threads.
- Model weights are not stored; the shipped binary models are never overwritten by these scripts.
