import { flagBubble, isSuppressed } from './banner.js';

/** Scope selectable text to real message containers, not chat-header metadata. */
const TEXT_SEL = 'span[data-testid="selectable-text"]';
const MSG_CONTAINER_SEL = '[data-testid="msg-container"]';
const SEL = `${MSG_CONTAINER_SEL} ${TEXT_SEL}`;
const CONTACT_SEL = '[data-testid="conversation-info-header-chat-title"]';
const DEBUG = true;
const DEBUG_LOG_MESSAGES = true;
const PREFIX = '[SurakshaNet]';
const processed = new WeakSet();
const pendingElements = new WeakSet();
const scheduledElements = new WeakSet();
const pendingKeys = new Set();
const textSeen = new Set();
const flaggedDetails = new Map();
const queue = [];
let pumping = false;
let attachedPane = null;
let paneObserver = null;
let attachTimer = null;
const MAX_RETRIES = 5;

function log(event, details = {}) {
  if (DEBUG) console.info(PREFIX, event, details);
}

function debugText(text) {
  return DEBUG_LOG_MESSAGES ? text : `[redacted:${text.length} chars]`;
}

function getContactName() {
  return document.querySelector(CONTACT_SEL)?.textContent?.trim() || 'Unknown chat';
}

log('content:loaded', { selector: SEL });

function keyOf(text, contact) {
  let h = 0;
  const value = `${contact}\n${text}`;
  for (let i = 0; i < value.length; i++) h = (h * 31 + value.charCodeAt(i)) | 0;
  return String(h);
}

function isOutgoing(el) {
  if (el.closest('.message-out')) return true;
  if (el.closest('.message-in')) return false;

  const container = el.closest('[data-testid="msg-container"]');
  if (!container) return false;
  const statusIcons = [...container.querySelectorAll('[data-icon]')];
  return statusIcons.some(
    (icon) =>
      !icon.closest('[data-testid="quoted-message"]') &&
      ['tail-out', 'msg-check', 'msg-dblcheck', 'msg-time', 'msg-error'].includes(icon.dataset.icon),
  );
}

function enqueue(el) {
  if (scheduledElements.has(el) || pendingElements.has(el)) return;
  scheduledElements.add(el);
  setTimeout(() => {
    scheduledElements.delete(el);
    enqueueNow(el);
  }, 75);
}

function enqueueNow(el) {
  if (isOutgoing(el)) {
    el.dataset.surakshaScanned = '1';
    processed.add(el);
    log('content:skipped:outgoing');
    return;
  }
  const text = el.innerText?.trim();
  if (!text || text.length < 5) return;
  const contact = getContactName();
  const key = keyOf(text, contact);
  if (isSuppressed(key)) {
    processed.add(el);
    el.dataset.surakshaScanned = '1';
    return;
  }
  const flagged = flaggedDetails.get(key);
  if (flagged) {
    processed.add(el);
    el.dataset.surakshaScanned = '1';
    el.dataset.surakshaText = text;
    if (el.isConnected) flagBubble(el, { ...flagged, id: key });
    log('content:banner:reapplied', { text: debugText(text), contact });
    return;
  }
  if (processed.has(el) && el.dataset.surakshaText === text) return;
  if (pendingElements.has(el)) return;
  if (textSeen.has(key)) {
    log('content:skipped:known-message', { text: debugText(text), contact });
    return;
  }
  if (pendingKeys.has(key)) return;
  pendingElements.add(el);
  pendingKeys.add(key);
  queue.push({ el, text, contact, key, attempts: 0 });
  log('content:queued', {
    text: debugText(text),
    pending: queue.length,
    textLength: text.length,
    contact,
  });
  pump();
}

async function pump() {
  if (pumping) return;
  pumping = true;
  log('content:pump:start', { pending: queue.length });
  try {
    while (queue.length) {
      const job = queue.shift();
      if (isSuppressed(job.key)) {
        pendingElements.delete(job.el);
        pendingKeys.delete(job.key);
        processed.add(job.el);
        job.el.dataset.surakshaScanned = '1';
        log('content:skipped:suppressed', {
          text: debugText(job.text),
          textLength: job.text.length,
          contact: job.contact,
        });
        continue;
      }
      if (!chrome?.runtime?.sendMessage) {
        retryJob(job, 'runtime unavailable');
        continue;
      }
      const result = await new Promise((resolve) => {
        try {
          chrome.runtime.sendMessage(
            { type: 'ANALYZE', text: job.text, contact: job.contact },
            (response) => {
              const error = chrome.runtime.lastError?.message || null;
              if (error) {
                log('content:message:failed', {
                  error,
                  text: debugText(job.text),
                  contact: job.contact,
                });
              }
              resolve(error ? { error } : response || { error: 'Empty worker response.' });
            },
          );
        } catch (error) {
          resolve({ error: String(error?.message || error) });
        }
      });
      log('content:message:complete', {
        text: debugText(job.text),
        textLength: job.text.length,
        contact: job.contact,
        flagged: Boolean(result?.flagged),
        score: typeof result?.score === 'number' ? Number(result.score.toFixed(4)) : null,
        error: result?.error || null,
      });
      if (result?.error) {
        retryJob(job, result.error);
        continue;
      }
      pendingElements.delete(job.el);
      pendingKeys.delete(job.key);
      textSeen.add(job.key);
      processed.add(job.el);
      job.el.dataset.surakshaScanned = '1';
      job.el.dataset.surakshaText = job.text;
      if (result?.flagged && job.el.isConnected) {
        flaggedDetails.set(job.key, { score: result.score, severity: result.severity });
        flagBubble(job.el, { score: result.score, severity: result.severity, id: job.key });
        log('content:banner:shown', { severity: result.severity });
      } else if (result?.flagged) {
        flaggedDetails.set(job.key, { score: result.score, severity: result.severity });
      }
    }
  } finally {
    pumping = false;
    log('content:pump:complete', { pending: queue.length });
  }
}

function retryJob(job, reason) {
  job.attempts += 1;
  log('content:retry', {
    attempt: job.attempts,
    maxAttempts: MAX_RETRIES,
    reason,
    text: debugText(job.text),
    contact: job.contact,
  });
  if (job.attempts > MAX_RETRIES || !job.el.isConnected) {
    pendingElements.delete(job.el);
    pendingKeys.delete(job.key);
    log('content:retry:exhausted', { reason, textLength: job.text.length });
    return;
  }
  const delay = Math.min(8000, 500 * 2 ** (job.attempts - 1));
  setTimeout(() => {
    queue.push(job);
    pump();
  }, delay);
}

function onAdded(node) {
  if (node.nodeType !== 1) return;
  if (node.matches?.(TEXT_SEL) && node.closest?.(MSG_CONTAINER_SEL)) enqueue(node);
  node.querySelectorAll?.(TEXT_SEL).forEach((el) => {
    if (el.closest(MSG_CONTAINER_SEL)) enqueue(el);
  });
}

function scheduleAttach() {
  if (attachTimer) return;
  attachTimer = setTimeout(() => {
    attachTimer = null;
    attach();
  }, 250);
}

function attach() {
  const pane =
    document.querySelector('#main') ||
    document.querySelector('[data-testid="conversation-panel-messages"]');
  if (!pane) return void setTimeout(attach, 500);
  if (pane === attachedPane && paneObserver) return;

  paneObserver?.disconnect();
  attachedPane = pane;

  const initial = pane.querySelectorAll(SEL);
  log('content:attached', {
    pane: pane.id || pane.dataset.testid || pane.tagName,
    candidates: initial.length,
    contact: getContactName(),
  });
  initial.forEach(enqueue);
  paneObserver = new MutationObserver((muts) => {
    for (const m of muts) {
      m.addedNodes.forEach(onAdded);
      const target = m.target.nodeType === 1 ? m.target : m.target.parentElement;
      const textEl = target?.closest?.(TEXT_SEL);
      if (textEl?.closest(MSG_CONTAINER_SEL)) enqueue(textEl);
      if (m.type === 'attributes') {
        const container = m.target.closest?.(MSG_CONTAINER_SEL);
        container?.querySelectorAll(TEXT_SEL).forEach(enqueue);
      }
    }
  });
  paneObserver.observe(pane, {
    childList: true,
    subtree: true,
    characterData: true,
    attributes: true,
    attributeFilter: ['class', 'data-icon'],
  });
}

new MutationObserver(() => {
  if (!attachedPane?.isConnected) scheduleAttach();
}).observe(document.documentElement, { childList: true, subtree: true });

attach();
