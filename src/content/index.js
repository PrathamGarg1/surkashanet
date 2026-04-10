import { showRedBanner } from './banner.js';

console.log("🛡️ SurakshaNet Content Script Loaded on:", window.location.hostname);

// Cache to remember exact texts we've already scanned so we never spam the SW
const scannedTexts = new Set();

// ─── SELECTORS ────────────────────────────────────────────────────────────────
function getMessageElements() {
    // Reverting to strict selectors because broad ones matched too many nested spans
    const selectors = [
        // WhatsApp Web – strict incoming and outgoing bubbles
        '.message-in span[data-testid="selectable-text"]',
        '.message-out span[data-testid="selectable-text"]',
        // Twitter/X tweets
        '[data-testid="tweetText"]',
        // Instagram DMs
        'div[role="row"] div[dir="auto"]',
    ];
    
    const elements = [];
    for (const sel of selectors) {
        try {
            document.querySelectorAll(sel).forEach(el => elements.push(el));
        } catch (e) { /* ignore */ }
    }
    return elements;
}

// ─── CORE SCAN ────────────────────────────────────────────────────────────────
function scanPage() {
    const elements = getMessageElements();
    
    elements.forEach(msg => {
        // Skip elements we've explicitly marked
        if (msg.dataset.surakshaScanned) return;

        const text = msg.innerText?.trim();
        if (!text || text.length < 5) return;

        // Mark DOM element so we skip it next time
        msg.dataset.surakshaScanned = "true";
        
        // CRITICAL SPAM PROTECTION: 
        // WhatsApp completely rebuilds the DOM on scroll.
        // If we've already scanned this exact text string, do not send to SW again!
        if (scannedTexts.has(text)) {
            return;
        }
        
        // Remember this text
        scannedTexts.add(text);
        
        console.log(`🔍 SurakshaNet Queuing: "${text.substring(0, 40)}"`);

        // ── Guard: extension context may be invalidated after a reload ───
        if (!chrome?.runtime?.sendMessage) {
            console.warn('⚠️ SurakshaNet: Extension context invalidated — please refresh the page.');
            delete msg.dataset.surakshaScanned;
            scannedTexts.delete(text);
            return;
        }

        // ── BENCHMARKING: round-trip latency (content-script side) ──────
        const _rtStart = performance.now();
        chrome.runtime.sendMessage(
            { type: 'ANALYZE_TEXT', text, source: window.location.hostname },
            (response) => {
                const _rtMs = performance.now() - _rtStart;
                console.log(`⏱️ SurakshaNet round-trip latency: ${_rtMs.toFixed(2)} ms`);

                // Guard again inside callback — runtime can become undefined between calls
                if (!chrome?.runtime) {
                    delete msg.dataset.surakshaScanned;
                    scannedTexts.delete(text);
                    return;
                }

                if (chrome.runtime.lastError) {
                    console.warn('⚠️ SurakshaNet SW not ready:', chrome.runtime.lastError.message);
                    delete msg.dataset.surakshaScanned;
                    scannedTexts.delete(text);
                    return;
                }
                
                if (!response || response.error) {
                    delete msg.dataset.surakshaScanned;
                    scannedTexts.delete(text);
                    return;
                }

                if (response.isToxic) {
                    console.warn('🚨 SurakshaNet TOXIC:', response.severity, (response.maxScore*100).toFixed(1)+'%');
                    showRedBanner(msg, {
                        severity: response.severity,
                        categories: response.categories,
                        maxScore: response.maxScore
                    });
                } else {
                    console.log(`✅ SurakshaNet Clean (${(response.maxScore*100).toFixed(1)}%):`, text.substring(0,20));
                }
            }
        );
    });
}

// ─── THROTTLED OBSERVER ───────────────────────────────────────────────────────
let scanTimeout = null;
function scheduleScan() {
    if (scanTimeout) return;
    // Increase throttle from 600ms to 1200ms to allow DOM to settle
    scanTimeout = setTimeout(() => { scanPage(); scanTimeout = null; }, 1200);
}

function startObserver() {
    const root = document.body || document.documentElement;
    if (!root) { setTimeout(startObserver, 200); return; }

    new MutationObserver(scheduleScan).observe(root, { childList: true, subtree: true });
    console.log('🛡️ SurakshaNet: Observer attached to', root.nodeName);

    scanPage();
    setTimeout(scanPage, 3000);
}

startObserver();

