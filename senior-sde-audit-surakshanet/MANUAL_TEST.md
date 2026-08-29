# Manual test checklist

- [ ] Load `dist/` unpacked in Chrome
- [ ] Open https://web.whatsapp.com and an active chat
- [ ] Incoming abusive-ish text → banner + popup row (auto-saved)
- [ ] Incoming safe text → no save
- [ ] Outgoing `.message-out` → ignored
- [ ] Dismiss → suppressed for session
- [ ] Export JSON downloads; Clear empties list
- [ ] No network model fetch (remote models disabled)
