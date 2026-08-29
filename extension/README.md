# SurakshaNet minimal extension (ponytail)

WhatsApp Web only. Incremental DOM. Auto local evidence. No full-page poll.

## Load
```bash
# copy INT8 artifacts from notebook/Modal into:
#   assets/models/custom-macd-model/
#     config.json ort_config.json tokenizer*.json special_tokens_map.json
#     onnx/model_quantized.onnx

npm ci
npm run build
# Chrome → Load unpacked → dist/
```

## Rules
- Flag if abusive_score > 0.5; severity high if >= 0.9 else medium
- Save flagged incoming only; hash dedupe; cap 200; screenshot null
- Observer on `#main` addedNodes — not body-wide 1.2s rescan
