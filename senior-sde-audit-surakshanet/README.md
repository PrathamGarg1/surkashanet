# SurakshaNet FINAL (standalone)

Self-contained MV3 extension. Prefer this tree (or Desktop/`senior-sde-audit-surakshanet.zip`) over legacy `src/`.

## Model
- Full-vocab INT8 `model_quantized.onnx` ~**113 MB** (Modal `surakshanet-artifacts`)
- Tokenizer/config under `assets/models/custom-macd-model/`
- Note: GitHub rejects blobs >100 MB without LFS. If clone is missing the ONNX, copy from Desktop zip or Modal `/vol/checkpoints/onnx_int8*`.

## Build / load
```bash
cd ~/Desktop/senior-sde-audit-surakshanet   # or this folder
npm install
npm run build
# Chrome → Extensions → Developer mode → Load unpacked → select ./dist
```

## Behavior
- WhatsApp Web only, incoming only
- Incremental MutationObserver (`addedNodes` on chat pane)
- Flag `score > 0.5`; severity `high` if `>= 0.9` else `medium`
- Auto local evidence (SHA-256 id/dedupe, cap 200, no screenshots)
- Popup: export JSON / clear

See AUDIT.md and MANUAL_TEST.md.
