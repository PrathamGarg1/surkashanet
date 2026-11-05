# SurakshaNet extension v2: gated conversational window

SurakshaNet is a Chrome Manifest V3 extension that detects abusive incoming
WhatsApp Web messages locally in the browser. Message text is never sent to a
server.

## Demo

[Watch the video demo on Google Drive](https://drive.google.com/file/d/1ElkdF0XA2bNzUCL6VWI95M_fqqFI4mmR/view?usp=sharing)

## Features

- On-device Transformers.js classification for incoming messages with the INT8 MiniLM retrained on Hindi, English and Hinglish
- Gated conversational window (`narrative_cache.js`, `sw.js`): the last up to 8 turns / 96 tokens of a chat are scored together with a short new message (fewer than 8 content tokens); the score is the maximum of the message and window scores; flag above 0.5, high from 0.9
- Contact-aware incident log with local SHA-256 deduplication
- Resilient WhatsApp DOM scanning and banner reattachment
- JSON and PDF export, including Hindi/Devanagari text
- No cloud sync, reporting, or automatic blocking

## Requirements

- Chrome or Chromium with Developer mode enabled
- Node.js `20.19+` or `22.12+`
- An authenticated WhatsApp Web session
- The trained model files, stored locally at:
  `assets/models/custom-macd-model/`

The model is intentionally excluded from source control. Its evaluation is in `../hinglish_context/` and `../thread_benchmark/quant/`; the original poster extension is in `../poster_v1/`.

## Build and load

```bash
npm ci
npm run build
```

Open `chrome://extensions`, enable **Developer mode**, choose **Load unpacked**,
and select `dist/`. Reload the extension and refresh WhatsApp Web.

The build bundles the extension, local model assets, ONNX Runtime WebAssembly,
and the PDF font. Runtime processing stays inside the browser.

## Debugging

Inspect the service worker from `chrome://extensions` or inspect the WhatsApp
Web page console. `DEBUG` controls general logs; `DEBUG_LOG_MESSAGES` controls
whether message text is printed. Disable both after debugging.
