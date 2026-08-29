# SurakshaNet MiniLM experiments

This directory is the reproducible experiment record for the Hindi abuse
classifier. The deployed backbone is fixed to
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

## Protocol

* Train: MACD `hindi_train.csv` + Davidson (labels 0/1 → abusive, 2 → non-abusive)
* Validate: MACD `hindi_val.csv` only
* Test: MACD `hindi_test.csv` only after a configuration is frozen
* Selection objective: validation macro-F1
* Never use `hindi_test` for hyperparameter, threshold, or vocabulary decisions

## Commands

```bash
export PATH="$HOME/.local/bin:$PATH"
modal run experiments/exp00_data_audit.py
modal run experiments/exp01_baseline.py
modal run experiments/exp02_controlled_ablations.py
modal run experiments/exp03_hparam_search.py --mode random
modal run experiments/exp04_data_mix.py
modal run experiments/exp08_freeze_and_test.py
modal run experiments/exp07_vocab_prune.py
modal run experiments/exp06_compress.py --checkpoint-dir /vol/checkpoints/pt_vocab_pruned
modal run training/modal_evaluate.py
npm run build
node experiments/verify_browser_runtime.mjs
```

## Honest target note

Approximately 88% MACD Hindi test accuracy is the stretch target. It is claimed
only if the untouched test evaluation actually reaches it. Prior Modal campaigns
with this fixed MiniLM backbone topped out near **~85% validation / ~85% test**
before compression; see `results.csv` and `logs/final_evaluation.json`.
