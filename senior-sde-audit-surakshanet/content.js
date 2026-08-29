import { flagBubble, isSuppressed } from './banner.js';

/** WhatsApp Web incoming selectable text only. */
const SEL = '.message-in span[data-testid="selectable-text"]';
const seen = new WeakSet();
const textSeen = new Set();
const queue = [];
let pumping = false;

function keyOf(text) {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) | 0;
  return String(h);
}

function enqueue(el) {
  if (seen.has(el) || el.dataset.surakshaScanned) return;
  const text = el.innerText?.trim();
  if (!text || text.length < 5) return;
  const key = keyOf(text);
  el.dataset.surakshaScanned = '1';
  seen.add(el);
  if (textSeen.has(key)) return;
  textSeen.add(key);
  queue.push({ el, text, key });
  pump();
}

async function pump() {
  if (pumping) return;
  pumping = true;
  while (queue.length) {
    const job = queue.shift();
    if (isSuppressed(job.key) || !chrome?.runtime?.sendMessage) continue;
    const result = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: 'ANALYZE', text: job.text, source: location.hostname },
        (r) => resolve(chrome.runtime.lastError ? null : r),
      );
    });
    if (result?.flagged && job.el.isConnected) {
      flagBubble(job.el, { score: result.score, severity: result.severity, id: job.key });
    }
  }
  pumping = false;
}

function onAdded(node) {
  if (node.nodeType !== 1) return;
  if (node.matches?.(SEL)) enqueue(node);
  node.querySelectorAll?.(SEL).forEach(enqueue);
}

function attach() {
  const pane =
    document.querySelector('#main') ||
    document.querySelector('[data-testid="conversation-panel-messages"]') ||
    document.body;
  if (!pane) return void setTimeout(attach, 500);

  pane.querySelectorAll(SEL).forEach(enqueue);
  new MutationObserver((muts) => {
    for (const m of muts) m.addedNodes.forEach(onAdded);
  }).observe(pane, { childList: true, subtree: true });
}

attach();
