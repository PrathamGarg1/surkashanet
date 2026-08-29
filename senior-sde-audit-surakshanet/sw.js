import { pipeline, env } from '@xenova/transformers';
import { saveIncident } from './store.js';

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = chrome.runtime.getURL('assets/models/');
env.useBrowserCache = true;
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('assets/');

const FLAG = 0.5;
const HIGH = 0.9;

let classifier = null;
const queue = [];
let busy = false;

async function ensureModel() {
  if (!classifier) {
    classifier = await pipeline('text-classification', 'custom-macd-model', { top_k: null });
  }
}

function abusiveScore(results) {
  const rows = Array.isArray(results) ? results : [results];
  for (const r of rows) {
    const l = String(r.label).toLowerCase();
    if (l === 'abusive' || l === 'label_0') return r.score;
  }
  for (const r of rows) {
    const l = String(r.label).toLowerCase();
    if (l === 'non-abusive' || l === 'label_1') return 1 - r.score;
  }
  return 0;
}

async function classify(text) {
  await ensureModel();
  const score = abusiveScore(await classifier(text));
  const flagged = score > FLAG;
  return {
    flagged,
    score,
    severity: !flagged ? null : score >= HIGH ? 'high' : 'medium',
  };
}

async function drain() {
  if (busy) return;
  busy = true;
  while (queue.length) {
    const job = queue.shift();
    try {
      const result = await classify(job.text);
      if (result.flagged) {
        await saveIncident({
          text: job.text,
          source: job.source,
          score: result.score,
          severity: result.severity,
        });
      }
      job.resolve(result);
    } catch (e) {
      job.reject(e);
    }
  }
  busy = false;
}

function enqueue(text, source) {
  return new Promise((resolve, reject) => {
    queue.push({ text, source, resolve, reject });
    drain();
  });
}

chrome.runtime.onInstalled.addListener(() => ensureModel().catch(console.error));
chrome.runtime.onStartup.addListener(() => ensureModel().catch(console.error));

chrome.runtime.onMessage.addListener((msg, _s, sendResponse) => {
  if (msg?.type !== 'ANALYZE') return;
  enqueue(msg.text, msg.source)
    .then((r) => sendResponse(r))
    .catch((e) => sendResponse({ flagged: false, error: String(e.message || e) }));
  return true;
});
