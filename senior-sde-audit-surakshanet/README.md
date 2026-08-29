# SurakshaNet FINAL (minimal)

Self-contained WhatsApp Web extension. Old `src/` is legacy — use this folder.

## Build / load
```bash
cd senior-sde-audit-surakshanet
npm install
npm run build
# Chrome → Extensions → Load unpacked → ./dist
```

## Model assets
Place INT8 package under `assets/models/custom-macd-model/`:
- `config.json`, `ort_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
- `onnx/model_quantized.onnx`

From notebook/Modal: copy `/vol/checkpoints/onnx_int8/*` (put onnx file under `onnx/model_quantized.onnx`).

## Manual test (WhatsApp Web)
1. Open a chat; send yourself / receive an abusive-looking Hindi/English line (incoming bubble).
2. Expect outline + banner (`high` if score≥0.9 else `medium`).
3. Dismiss → banner gone; same text should stay suppressed this session.
4. Safe message → no banner, no popup row.
5. Popup → see incident; Export JSON; Clear.
