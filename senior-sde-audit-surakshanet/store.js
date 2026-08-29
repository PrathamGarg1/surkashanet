/** Local incident log. On-device only. Hash = id/dedupe, not the whole product. */
const KEY = 'surakshanet_incidents';
const CAP = 200;

export async function sha256(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function saveIncident({ text, source, score, severity }) {
  const id = await sha256(`${source}\n${text.trim().toLowerCase()}`);
  const { [KEY]: list = [] } = await chrome.storage.local.get(KEY);
  if (list.some((x) => x.id === id)) return null;
  const row = { id, ts: new Date().toISOString(), source, score, severity, text, screenshot: null };
  list.unshift(row);
  await chrome.storage.local.set({ [KEY]: list.slice(0, CAP) });
  return row;
}

export async function listIncidents() {
  const { [KEY]: list = [] } = await chrome.storage.local.get(KEY);
  return list;
}

export async function clearIncidents() {
  await chrome.storage.local.set({ [KEY]: [] });
}

export async function exportJSON() {
  return JSON.stringify(
    { exportedAt: new Date().toISOString(), incidents: await listIncidents() },
    null,
    2,
  );
}
