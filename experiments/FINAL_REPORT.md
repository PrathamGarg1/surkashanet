# Final MiniLM Hindi abuse classifier report

## Verdict

**Approximately 88% MACD Hindi test accuracy was not achieved** with the fixed
backbone `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Best honest untouched-test result from the Modal campaign (Transformers 4.36.2
compatible stack):

| Artifact | MACD hindi_val | MACD hindi_test | Size |
|---|---|---|---|
| PyTorch FP32 winner | 84.71% acc / 84.68% macro-F1 | **84.97% acc / 84.94% macro-F1** | ~465 MB checkpoint |
| ONNX FP32 | same as PT within noise | ~84.97% | ~449 MB model |
| ONNX INT8 (full vocab) | ~84.65% | ~84.66% | ~113 MB model / ~129 MB artifact |
| Vocab-pruned INT8 (ship) | ~84.71% | **84.41% acc / 84.38% macro-F1** | **82.1 MB model / 93.3 MB Chrome artifact** |

Validation-selected configuration (`experiments/configs/final_winner.json`):

* lr `5e-5`, 5 epochs, batch 16, max length 96, linear schedule
* weight decay 0, warmup 0, no class weights, no label smoothing
* train = MACD hindi_train + full Davidson train split
* val = MACD hindi_val only
* threshold for the extension: `0.66` (tuned on validation only)

## Why 88% was not reached

1. **Backbone capacity.** MiniLM-L12 (~118M params, 384-d) is far smaller than
   the MACD paper's XLM-R Large reference. With the backbone frozen by design,
   head/optimizer tricks cannot add representation power.
2. **Domain gap.** Davidson is English and ~83% abusive; MACD Hindi is balanced.
   Mixing helps a little, but MACD-only and balanced-downsample ablations did
   not break into the high-80s on Hindi validation.
3. **Ceiling on Hindi validation.** Across 60+ validation-only trials
   (ablations, random search, data mixes), best val macro-F1 stayed near
   **85.6%** (older 4.44 stack) / **~85%** (export-compatible 4.36 stack).
   Test tracked validation and did not jump to 88%.
4. **Compression trade-off.** Vocab pruning (−33.6% tokens) and INT8 keep the
   Chrome package near **93 MB** with ~0.5 pp test drop vs FP32.

## Reproduce

Requires Modal credentials (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`):

```bash
modal run experiments/exp08_freeze_and_test.py
modal run experiments/exp07_vocab_prune.py
modal run experiments/exp06_compress.py --checkpoint-dir /vol/checkpoints/pt_vocab_pruned
modal run training/modal_evaluate.py
npm run build
node experiments/verify_browser_runtime.mjs
```

## Pipeline updates

* `training/_common.py` — frozen winner hyperparameters (`MAX_LENGTH=96`)
* `training/modal_train.py` — delegates to `experiments/_exp_common.run_training`
* `training/modal_export.py` — staged ONNX FP32/INT8 with backups
* `training/modal_evaluate.py` — evaluation-only MACD hindi_test access
* `src/background/service-worker.js` — validation-tuned threshold `0.66`
