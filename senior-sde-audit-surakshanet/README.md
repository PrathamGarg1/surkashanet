# SurakshaNet FINAL (standalone)

This folder is self-contained. It is **not** nested under the old surakshanet repo.

## Model
- INT8 `model_quantized.onnx` ~**113 MB** (fetched from Modal `surakshanet-artifacts`)
- Plus tokenizer/config under `assets/models/custom-macd-model/`

## Build / load
```bash
cd ~/Desktop/senior-sde-audit-surakshanet   # or this folder
npm install
npm run build
# Chrome → Extensions → Developer mode → Load unpacked → select ./dist
```

## Behavior
- WhatsApp Web only, incoming only
- Incremental MutationObserver (addedNodes)
- Flag score > 0.5; high if >= 0.9 else medium
- Auto local evidence (hash dedupe, cap 200)
- Popup: export JSON / clear

See AUDIT.md and MANUAL_TEST.md.
