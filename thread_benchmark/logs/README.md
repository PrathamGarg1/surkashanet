# Cost and run logs

Spend for the five-class thread benchmark (dataset generation and the GPU training run). Start date 2026-11-03 (IST).
The earlier MACD-to-Hinglish conversion spend is not in this ledger.

| File | Content |
|---|---|
| `cost_ledger.csv` | USD estimate per day and job |
| `build_dataset_run1.log` | dataset build and validator counts |
| `train_isolated.log`, `train_thread.log` | local training logs of the last-message-only and thread models |

The per-call Bedrock log (`bedrock_calls.jsonl`, one line per call: timestamp, model, tokens, request id, estimated USD) is about 33 MB and is not included in this repository. The ledger is its daily summary.

List prices used for the estimates (us-east-1, standard on-demand):

| Model | Input / 1M tokens | Output / 1M tokens |
|---|---:|---:|
| `qwen.qwen3-32b` (Mantle) | $0.15 | $0.60 |
| `qwen.qwen3-32b-v1:0` (Converse) | $0.15 | $0.60 |
| `meta.llama3-8b-instruct-v1:0` (fallback) | $0.30 | $0.60 |

Estimates can differ from the invoice by a few percent (rounding, failed retries). Local INT8 scoring and MiniLM training on a laptop are logged as `local` with USD 0.
