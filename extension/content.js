import { flagBubble, isSuppressed } from './banner.js';

const seen = new WeakSet();
const textSeen = new Set();
const SEL = '.message-in span[data-testid="selectable-text"]';
const queue = [];
let pumping = false;

function hashLite(s) {
  // fast session key; SW does real SHA-256 for storage id
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return String(h);
}

function analyze(el) {
  if (seen.has(el) || el.dataset.surakshaScanned) return;
  const text = el.innerText?.trim();
  if (!text || text.length < 5) return;

  const key = hashLite(text);
  if (textSeen.has(key)) {
    el.dataset.surakshaScanned = '1';
    seen.add(el);
    return;
  }

  el.dataset.surakshaScanned = '1';
  seen.add(el);
  textSeen.add(key);
  queue.push({ el, text, key });
  pump();
}

async function pump() {
  if (pumping) return;
  pumping = true;
  while (queue.length) {
    const job = queue.shift();
    if (isSuppressed(job.key)) continue;
    if (!chrome?.runtime?.sendMessage) break;
    const result = await new Promise((resolve) => {
      chrome.runtime.sendMessage(
        { type: 'ANALYZE', text: job.text, source: location.hostname },
        (r) => resolve(chrome.runtime.lastError ? null : r),
      );
    });
    if (!result?.flagged || !job.el.isConnected) continue;
    flagBubble(job.el, { score: result.score, severity: result.severity, id: job.key });
  }
  pumping = false;
}

function walkAdded(node) {
  if (node.nodeType !== 1) return;
  if (node.matches?.(SEL)) analyze(node);
  node.querySelectorAll?.(SEL).forEach(analyze);
}

function attach() {
  const pane =
    document.querySelector('#main') ||
    document.querySelector('[data-testid="conversation-panel-messages"]') ||
    document.body;
  if (!pane) {
    setTimeout(attach, 500);
    return;
  }

  // one-time seed of currently visible incoming messages
  pane.querySelectorAll(SEL).forEach(analyze);

  new MutationObserver((muts) => {
    for (const m of muts) m.addedNodes.forEach(walkAdded);
  }).observe(pane, { childList: true, subtree: true });
}

attach();
