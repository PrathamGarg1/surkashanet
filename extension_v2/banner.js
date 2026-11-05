const suppressed = new Set();
const STYLE_ID = 'suraksha-banner-styles';
const SEVERITY_CONFIG = {
  high: {
    background: '#ffebee',
    border: '#d32f2f',
    text: '#8f1d1d',
    label: 'Critical threat',
    iconBackground: '#fff7f7',
  },
  medium: {
    background: '#fff3e0',
    border: '#f57c00',
    text: '#9a4d00',
    label: 'Harmful content',
    iconBackground: '#fffaf2',
  },
};

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    .suraksha-banner {
      align-items: center !important;
      box-sizing: border-box !important;
      display: flex !important;
      gap: 9px !important;
      float: none !important;
      margin: 7px 9px 5px !important;
      min-height: 51px !important;
      width: min(360px, calc(100% - 18px)) !important;
      padding: 8px 9px !important;
      border: 1px solid var(--sn-border) !important;
      border: 1px solid color-mix(in srgb, var(--sn-border) 30%, transparent) !important;
      border-left: 4px solid var(--sn-border) !important;
      border-radius: 10px !important;
      color: var(--sn-text) !important;
      background: var(--sn-bg) !important;
      background: linear-gradient(135deg, var(--sn-bg) 0%, color-mix(in srgb, var(--sn-bg) 86%, white) 100%) !important;
      box-shadow: 0 5px 14px rgba(0, 0, 0, .16) !important;
      font: 500 12px/1.2 "Avenir Next", "Segoe UI", sans-serif !important;
    }
    .suraksha-banner.suraksha-high {
      --sn-bg: #ffebee;
      --sn-border: #d32f2f;
      --sn-text: #8f1d1d;
    }
    .suraksha-banner__icon {
      align-items: center !important;
      background: var(--sn-icon-bg) !important;
      border-radius: 7px !important;
      color: var(--sn-border) !important;
      display: flex !important;
      flex: 0 0 27px !important;
      height: 27px !important;
      justify-content: center !important;
      width: 27px !important;
    }
    .suraksha-banner__copy {
      display: grid !important;
      flex: 1 1 auto !important;
      gap: 2px !important;
      min-width: 0 !important;
    }
    .suraksha-banner__title {
      color: var(--sn-text) !important;
      font-size: 11px !important;
      font-weight: 700 !important;
    }
    .suraksha-banner__detail {
      color: var(--sn-text) !important;
      color: color-mix(in srgb, var(--sn-text) 72%, transparent) !important;
      font-size: 10px !important;
    }
    .suraksha-banner__dismiss {
      min-height: 30px !important;
      padding: 5px 8px !important;
      border: 1px solid var(--sn-border) !important;
      border: 1px solid color-mix(in srgb, var(--sn-border) 48%, transparent) !important;
      border-radius: 7px !important;
      color: var(--sn-text) !important;
      background: transparent !important;
      cursor: pointer !important;
      font: 600 10px/1 "Avenir Next", "Segoe UI", sans-serif !important;
    }
    .suraksha-banner__dismiss:hover {
      background: rgba(255, 255, 255, .14) !important;
      background: color-mix(in srgb, var(--sn-border) 10%, transparent) !important;
    }
    .suraksha-banner__dismiss:focus-visible {
      outline: 2px solid #a9e1c9 !important;
      outline-offset: 2px !important;
    }
    .suraksha-banner.suraksha-medium {
      --sn-bg: #fff3e0;
      --sn-border: #f57c00;
      --sn-text: #9a4d00;
    }
  `;
  (document.head || document.documentElement).appendChild(style);
}

function makeIcon() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 26');
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '18');
  svg.setAttribute('fill', 'none');
  const shield = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  shield.setAttribute('d', 'M12 2.5 21 6v6.4c0 5.4-3.5 9.3-9 11.1-5.5-1.8-9-5.7-9-11.1V6l9-3.5Z');
  shield.setAttribute('stroke', 'currentColor');
  shield.setAttribute('stroke-width', '1.7');
  const mark = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  mark.setAttribute('d', 'm8.2 13 2.5 2.5 5.3-5.4');
  mark.setAttribute('stroke', 'currentColor');
  mark.setAttribute('stroke-width', '1.7');
  mark.setAttribute('stroke-linecap', 'round');
  mark.setAttribute('stroke-linejoin', 'round');
  svg.append(shield, mark);
  return svg;
}

export function flagBubble(el, { score, severity, id }) {
  const cfg = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.medium;
  const bubble = el.closest('[data-testid="msg-container"]') || el.parentElement;
  const bubbleParent = bubble?.parentElement;
  const messageRow =
    el.closest('[data-testid^="conv-msg-"]') || bubble?.parentElement || bubble;
  if (el.dataset.surakshaFlagged && messageRow?.querySelector('.suraksha-banner')) return;
  ensureStyles();
  el.dataset.surakshaFlagged = '1';
  bubble?.style.setProperty('border', `2px solid ${cfg.border}`, 'important');
  bubble?.style.setProperty('border-radius', '10px', 'important');
  bubble?.style.setProperty('background-color', `${cfg.border}12`, 'important');

  const bar = document.createElement('div');
  bar.className = `suraksha-banner suraksha-${severity}`;
  bar.style.setProperty('--sn-bg', cfg.background);
  bar.style.setProperty('--sn-border', cfg.border);
  bar.style.setProperty('--sn-text', cfg.text);
  bar.style.setProperty('--sn-icon-bg', cfg.iconBackground);
  const bubbleRect = bubble?.getBoundingClientRect();
  const parentRect = bubbleParent?.getBoundingClientRect();
  if (bubbleRect && parentRect) {
    const isRightAligned = bubbleRect.left + bubbleRect.width / 2 > window.innerWidth / 2;
    const bannerWidth = Math.min(360, Math.max(220, bubbleRect.width));
    const left = isRightAligned
      ? bubbleRect.right - bannerWidth - parentRect.left
      : bubbleRect.left - parentRect.left;
    bar.style.setProperty('margin-left', `${Math.max(9, left)}px`, 'important');
    bar.style.setProperty('margin-right', '0px', 'important');
    bar.style.setProperty('width', `${bannerWidth}px`, 'important');
  }
  bar.setAttribute('role', 'status');

  const icon = document.createElement('span');
  icon.className = 'suraksha-banner__icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.appendChild(makeIcon());

  const copy = document.createElement('span');
  copy.className = 'suraksha-banner__copy';
  const title = document.createElement('strong');
  title.className = 'suraksha-banner__title';
  title.textContent = cfg.label;
  const detail = document.createElement('span');
  detail.className = 'suraksha-banner__detail';
  detail.textContent = `${(Number(score) * 100).toFixed(0)}% confidence · saved locally`;
  copy.append(title, detail);

  const btn = document.createElement('button');
  btn.className = 'suraksha-banner__dismiss';
  btn.type = 'button';
  btn.textContent = 'Dismiss';
  btn.setAttribute('aria-label', 'Dismiss this SurakshaNet warning');
  btn.onclick = (e) => {
    e.stopPropagation();
    if (id) suppressed.add(id);
    bubble?.style.removeProperty('border');
    bubble?.style.removeProperty('border-radius');
    bubble?.style.removeProperty('background-color');
    bar.remove();
  };
  bar.append(icon, copy, btn);
  bubbleParent?.appendChild(bar);
}

export function isSuppressed(id) {
  return Boolean(id && suppressed.has(id));
}
