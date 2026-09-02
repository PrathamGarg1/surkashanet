/**
 * Evidence Viewer & Parental Session Controller for SurakshaNet Popup
 * Manages view state transitions: Unconfigured -> Child Protection (Locked) -> Parent Dashboard (Unlocked)
 */

import {
    isAccountSetup,
    isParentSessionActive,
    loginParent,
    lockSession,
    logoutParent
} from '../utils/auth_manager.js';

import {
    getAllEvidence,
    getStorageStats,
    exportToJSON,
    downloadFile,
    clearAllEvidence
} from '../utils/storage_manager.js';

import { exportToPDF } from '../utils/pdf_exporter.js';

// DOM Elements
let viewUnconfigured, viewChildLocked, viewParentDashboard;
let btnLaunchSetup, unlockForm, popupPassword, btnUnlockPopup, unlockError;
let headerActions, footerLinks, btnLogoutReset;
let totalIncidentsEl, highSeverityEl, mediumSeverityEl;
let evidenceListEl, emptyStateEl, loadingEl;
let exportPDFBtn, exportJSONBtn, clearAllBtn;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    cacheDomElements();
    setupGlobalEventListeners();
    await updateViewState();
});

function cacheDomElements() {
    viewUnconfigured = document.getElementById('viewUnconfigured');
    viewChildLocked = document.getElementById('viewChildLocked');
    viewParentDashboard = document.getElementById('viewParentDashboard');

    btnLaunchSetup = document.getElementById('btnLaunchSetup');
    unlockForm = document.getElementById('unlockForm');
    popupPassword = document.getElementById('popupPassword');
    btnUnlockPopup = document.getElementById('btnUnlockPopup');
    unlockError = document.getElementById('unlockError');

    headerActions = document.getElementById('headerActions');
    footerLinks = document.getElementById('footerLinks');
    btnLogoutReset = document.getElementById('btnLogoutReset');

    totalIncidentsEl = document.getElementById('totalIncidents');
    highSeverityEl = document.getElementById('highSeverity');
    mediumSeverityEl = document.getElementById('mediumSeverity');
    evidenceListEl = document.getElementById('evidenceList');
    emptyStateEl = document.getElementById('emptyState');
    loadingEl = document.getElementById('loading');

    exportPDFBtn = document.getElementById('exportPDF');
    exportJSONBtn = document.getElementById('exportJSON');
    clearAllBtn = document.getElementById('clearAll');
}

function setupGlobalEventListeners() {
    if (btnLaunchSetup) {
        btnLaunchSetup.addEventListener('click', () => {
            const url = chrome.runtime.getURL('src/auth/auth.html');
            chrome.tabs.create({ url });
        });
    }

    if (unlockForm) {
        unlockForm.addEventListener('submit', handleUnlockSubmit);
    }

    if (exportPDFBtn) exportPDFBtn.addEventListener('click', handleExportPDF);
    if (exportJSONBtn) exportJSONBtn.addEventListener('click', handleExportJSON);
    if (clearAllBtn) clearAllBtn.addEventListener('click', handleClearAll);

    if (btnLogoutReset) {
        btnLogoutReset.addEventListener('click', handleLogoutReset);
    }
}

// ─── VIEW STATE ORCHESTRATION ───────────────────────────────────────────────

async function updateViewState() {
    const configured = await isAccountSetup();
    if (!configured) {
        showView('unconfigured');
        return;
    }

    const sessionActive = await isParentSessionActive();
    if (sessionActive) {
        showView('parentDashboard');
        await loadEvidence();
    } else {
        showView('childLocked');
    }
}

function showView(viewName) {
    viewUnconfigured.style.display = 'none';
    viewChildLocked.style.display = 'none';
    viewParentDashboard.style.display = 'none';
    headerActions.innerHTML = '';
    if (btnLogoutReset) btnLogoutReset.style.display = 'none';

    if (viewName === 'unconfigured') {
        viewUnconfigured.style.display = 'flex';
    } else if (viewName === 'childLocked') {
        viewChildLocked.style.display = 'flex';
        if (unlockError) unlockError.style.display = 'none';
        if (popupPassword) {
            popupPassword.value = '';
            popupPassword.focus();
        }
    } else if (viewName === 'parentDashboard') {
        viewParentDashboard.style.display = 'flex';
        if (btnLogoutReset) btnLogoutReset.style.display = 'inline-block';

        // Add "Lock Session" pill button to header
        const lockBtn = document.createElement('button');
        lockBtn.className = 'btn-pill lock-btn';
        lockBtn.innerHTML = '🔒 Lock Session';
        lockBtn.addEventListener('click', async () => {
            await lockSession();
            await updateViewState();
        });
        headerActions.appendChild(lockBtn);
    }
}

// ─── UNLOCK HANDLER ─────────────────────────────────────────────────────────

async function handleUnlockSubmit(e) {
    e.preventDefault();
    if (unlockError) unlockError.style.display = 'none';

    const password = popupPassword.value;
    if (!password) return;

    btnUnlockPopup.disabled = true;
    btnUnlockPopup.textContent = 'Verifying...';

    try {
        const success = await loginParent(password);
        if (success) {
            await updateViewState();
        } else {
            if (unlockError) {
                unlockError.textContent = 'Incorrect master password. Access denied.';
                unlockError.style.display = 'block';
            }
            popupPassword.value = '';
            popupPassword.focus();
        }
    } catch (err) {
        console.error('Unlock error:', err);
        if (unlockError) {
            unlockError.textContent = 'Error verifying password: ' + err.message;
            unlockError.style.display = 'block';
        }
    } finally {
        btnUnlockPopup.disabled = false;
        btnUnlockPopup.textContent = 'Unlock';
    }
}

// ─── LOGOUT HANDLER ─────────────────────────────────────────────────────────

async function handleLogoutReset(e) {
    e.preventDefault();
    const confirmPassword = prompt(
        '⚠️ Confirm Logout: Logging out will deactivate the current configuration and require parent setup on next use.\n\nEnter your Master Parent Password to confirm:'
    );

    if (!confirmPassword) return;

    try {
        await logoutParent(confirmPassword);
        alert('You have successfully logged out of SurakshaNet.');
        await updateViewState();
    } catch (error) {
        alert('Failed to log out: ' + error.message);
    }
}

// ─── EVIDENCE VIEWER DATA & ACTIONS ─────────────────────────────────────────

async function loadEvidence() {
    try {
        if (loadingEl) loadingEl.style.display = 'block';
        if (evidenceListEl) evidenceListEl.innerHTML = '';
        if (emptyStateEl) emptyStateEl.style.display = 'none';

        const evidence = await getAllEvidence();
        const stats = await getStorageStats();

        updateStats(stats);

        if (evidence.length === 0) {
            if (emptyStateEl) emptyStateEl.style.display = 'block';
        } else {
            displayEvidence(evidence);
        }

        if (loadingEl) loadingEl.style.display = 'none';
    } catch (error) {
        console.error('Failed to load evidence:', error);
        if (loadingEl) loadingEl.style.display = 'none';
        if (evidenceListEl) {
            evidenceListEl.innerHTML = '<p style="color: #ef4444; text-align: center; font-size: 11px;">Failed to load evidence</p>';
        }
    }
}

function updateStats(stats) {
    if (totalIncidentsEl) totalIncidentsEl.textContent = stats.totalIncidents;
    if (highSeverityEl) highSeverityEl.textContent = stats.severityCounts.HIGH;
    if (mediumSeverityEl) mediumSeverityEl.textContent = stats.severityCounts.MEDIUM;
}

function displayEvidence(evidenceList) {
    const sorted = [...evidenceList].sort((a, b) =>
        new Date(b.timestamp) - new Date(a.timestamp)
    );

    evidenceListEl.innerHTML = '';
    sorted.forEach(evidence => {
        const item = createEvidenceItem(evidence);
        evidenceListEl.appendChild(item);
    });
}

function createEvidenceItem(evidence) {
    const div = document.createElement('div');
    div.className = 'evidence-item';

    const header = document.createElement('div');
    header.className = 'evidence-header';

    const timestamp = document.createElement('span');
    timestamp.className = 'evidence-meta';
    timestamp.textContent = formatTimestamp(evidence.timestamp);

    const severityBadge = document.createElement('span');
    severityBadge.className = `severity-badge ${evidence.severity || 'UNKNOWN'}`;
    severityBadge.textContent = evidence.severity || 'UNKNOWN';

    header.appendChild(timestamp);
    header.appendChild(severityBadge);

    const meta = document.createElement('div');
    meta.className = 'evidence-meta';
    meta.style.marginBottom = '6px';
    meta.innerHTML = `
        <strong>Source:</strong> ${evidence.source || 'Unknown'} | 
        <strong>Score:</strong> ${(evidence.maxScore * 100).toFixed(1)}%
        ${evidence.screenshot ? ' | 📸 Visual Evidence' : ''}
    `;

    const categoriesDiv = document.createElement('div');
    categoriesDiv.className = 'category-tags';

    if (evidence.categories && evidence.categories.length > 0) {
        evidence.categories.forEach(cat => {
            const tag = document.createElement('span');
            tag.className = 'category-tag';
            tag.textContent = `${cat.category.toUpperCase()} (${(cat.score * 100).toFixed(0)}%)`;
            categoriesDiv.appendChild(tag);
        });
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'evidence-text';
    textDiv.textContent = evidence.text || 'No text captured';

    div.appendChild(header);
    div.appendChild(meta);
    if (evidence.categories && evidence.categories.length > 0) {
        div.appendChild(categoriesDiv);
    }
    div.appendChild(textDiv);

    return div;
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        dateStyle: 'short',
        timeStyle: 'short'
    });
}

async function handleExportPDF() {
    try {
        exportPDFBtn.disabled = true;
        exportPDFBtn.textContent = '⏳ Exporting...';

        const evidence = await getAllEvidence();
        if (evidence.length === 0) {
            alert('No evidence records to export.');
            return;
        }

        exportToPDF(evidence);
        exportPDFBtn.textContent = '✅ Exported!';
        setTimeout(() => {
            exportPDFBtn.textContent = '📄 Export PDF';
            exportPDFBtn.disabled = false;
        }, 2000);
    } catch (error) {
        console.error('PDF export failed:', error);
        alert('Failed to export PDF: ' + error.message);
        exportPDFBtn.textContent = '📄 Export PDF';
        exportPDFBtn.disabled = false;
    }
}

async function handleExportJSON() {
    try {
        exportJSONBtn.disabled = true;
        exportJSONBtn.textContent = '⏳ Exporting...';

        const jsonData = await exportToJSON();
        const timestamp = new Date().toISOString().split('T')[0];
        downloadFile(jsonData, `surakshanet_evidence_${timestamp}.json`, 'application/json');

        exportJSONBtn.textContent = '✅ Exported!';
        setTimeout(() => {
            exportJSONBtn.textContent = '💾 Export JSON';
            exportJSONBtn.disabled = false;
        }, 2000);
    } catch (error) {
        console.error('JSON export failed:', error);
        alert('Failed to export JSON: ' + error.message);
        exportJSONBtn.textContent = '💾 Export JSON';
        exportJSONBtn.disabled = false;
    }
}

async function handleClearAll() {
    const confirmed = confirm(
        '⚠️ WARNING: This will permanently delete ALL logged evidence records.\n\nAre you sure you want to continue?'
    );
    if (!confirmed) return;

    try {
        clearAllBtn.disabled = true;
        clearAllBtn.textContent = 'Clearing...';

        await clearAllEvidence();
        await loadEvidence();

        clearAllBtn.textContent = '✅ Cleared!';
        setTimeout(() => {
            clearAllBtn.textContent = '🗑️ Clear';
            clearAllBtn.disabled = false;
        }, 2000);
    } catch (error) {
        console.error('Failed to clear evidence:', error);
        alert('Failed to clear evidence: ' + error.message);
        clearAllBtn.textContent = '🗑️ Clear';
        clearAllBtn.disabled = false;
    }
}
