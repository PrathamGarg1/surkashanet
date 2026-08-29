const suppressed = new Set();

export function flagBubble(el, { score, severity, id }) {
  if (el.dataset.surakshaFlagged) return;
  el.dataset.surakshaFlagged = '1';
  el.style.outline = severity === 'high' ? '2px solid #c62828' : '2px solid #ef6c00';

  const bar = document.createElement('div');
  bar.className = 'suraksha-banner';
  bar.style.cssText =
    'margin:4px 0;padding:6px 8px;font:12px/1.3 system-ui;border-left:3px solid ' +
    (severity === 'high' ? '#c62828' : '#ef6c00') +
    ';background:#fff8f6';
  bar.textContent = `${severity.toUpperCase()} · ${(score * 100).toFixed(0)}% `;

  const btn = document.createElement('button');
  btn.textContent = 'Dismiss';
  btn.style.cssText = 'margin-left:8px;font:11px system-ui;cursor:pointer';
  btn.onclick = (e) => {
    e.stopPropagation();
    if (id) suppressed.add(id);
    el.style.outline = '';
    bar.remove();
  };
  bar.appendChild(btn);
  el.parentElement?.appendChild(bar);
}

export function isSuppressed(id) {
  return Boolean(id && suppressed.has(id));
}
