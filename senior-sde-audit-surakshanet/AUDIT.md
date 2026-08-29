# Audit: why the old SurakshaNet DOM path is junior

Target files: `src/content/index.js`, `banner.js`, `service-worker.js`, `evidence_logger.js`, `storage_manager.js`.

## What a senior hates

1. **Full-page rescans as design** — every MutationObserver tick eventually `querySelectorAll`s the whole chat. WhatsApp virtualizes; you get churn, not correctness.
2. **`setTimeout(1200)` throttle as architecture** — debounce is fine; using it to hide a body-wide observer is not a scan strategy.
3. **`document.body` observer** — scroll/chrome chrome UI fires noise. Scope to the conversation pane.
4. **Text-only dedupe** — identical strings collapse; DOM rebuild clears `dataset` marks; edits never re-scan.
5. **Unbounded parallel `sendMessage`** — every bubble hits the SW at once; WASM model serializes badly under stampede.
6. **Outgoing + incoming** — classifying `.message-out` wastes inference and pollutes evidence.
7. **Evidence theater** — dual storage keys, unused fake AES (`crypto-js` + hardcoded key), “court-admissible” copy, screenshot stubs/`alert()` Block-Report-Save.
8. **Auto-log without a real schema** — hash used only as id, no cap/dedupe policy, popup/storage disagree.

## First-principles rewrite (this folder)

- Observe **addedNodes** in `#main` / conversation panel only.
- Incoming WhatsApp bubbles only.
- Mark nodes (`WeakSet` + `dataset`) + session text key; **one** SW queue.
- Flag `score > 0.5`; severity `high` if `>= 0.9` else `medium`.
- Auto-save `{id: sha256, ts, source, score, severity, text, screenshot: null}`; dedupe; cap 200.
- Banner: highlight + Dismiss. Popup: list / export JSON / clear. Honest on-device copy only.
