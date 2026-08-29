import { listIncidents, clearIncidents, exportJSON } from './store.js';

const listEl = document.getElementById('list');

function row(item) {
  const div = document.createElement('div');
  div.className = 'row';
  div.innerHTML = `
    <div class="meta">
      <span class="${item.severity}">${item.severity}</span>
      · ${(item.score * 100).toFixed(0)}%
      · ${new Date(item.ts).toLocaleString()}
      · ${item.source}
    </div>
    <div></div>
  `;
  div.lastElementChild.textContent = item.text;
  return div;
}

async function render() {
  const items = await listIncidents();
  listEl.innerHTML = '';
  if (!items.length) {
    listEl.innerHTML = '<div class="empty">No flagged messages yet.</div>';
    return;
  }
  items.forEach((item) => listEl.appendChild(row(item)));
}

document.getElementById('export').onclick = async () => {
  const blob = new Blob([await exportJSON()], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), {
    href: url,
    download: `surakshanet-${Date.now()}.json`,
  });
  a.click();
  URL.revokeObjectURL(url);
};

document.getElementById('clear').onclick = async () => {
  if (!confirm('Clear all local incidents?')) return;
  await clearIncidents();
  render();
};

render();
