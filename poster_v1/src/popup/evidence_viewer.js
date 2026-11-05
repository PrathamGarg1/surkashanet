/**
 * Evidence Viewer - Popup UI Logic
 * Manages evidence display, statistics, and export functionality
 */

import { getAllEvidence, getStorageStats, exportToJSON, downloadFile, clearAllEvidence } from '../utils/storage_manager.js';
import { exportToPDF } from '../utils/pdf_exporter.js';

// DOM Elements
let totalIncidentsEl, highSeverityEl, mediumSeverityEl;
let evidenceListEl, emptyStateEl, loadingEl;
let exportPDFBtn, exportJSONBtn, clearAllBtn;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    // Get DOM elements
    totalIncidentsEl = document.getElementById('totalIncidents');
    highSeverityEl = document.getElementById('highSeverity');
    mediumSeverityEl = document.getElementById('mediumSeverity');
    evidenceListEl = document.getElementById('evidenceList');
    emptyStateEl = document.getElementById('emptyState');
    loadingEl = document.getElementById('loading');
    exportPDFBtn = document.getElementById('exportPDF');
    exportJSONBtn = document.getElementById('exportJSON');
    clearAllBtn = document.getElementById('clearAll');

    // Set up event listeners
    exportPDFBtn.addEventListener('click', handleExportPDF);
    exportJSONBtn.addEventListener('click', handleExportJSON);
    clearAllBtn.addEventListener('click', handleClearAll);

    // Load evidence
    await loadEvidence();
});

/**
 * Load and display evidence
 */
async function loadEvidence() {
    try {
        loadingEl.style.display = 'block';
        evidenceListEl.innerHTML = '';
        emptyStateEl.style.display = 'none';

        // Get evidence and stats
        const evidence = await getAllEvidence();
        const stats = await getStorageStats();

        // Update statistics
        updateStats(stats);

        // Display evidence
        if (evidence.length === 0) {
            emptyStateEl.style.display = 'block';
        } else {
            displayEvidence(evidence);
        }

        loadingEl.style.display = 'none';
    } catch (error) {
        console.error('Failed to load evidence:', error);
        loadingEl.style.display = 'none';
        evidenceListEl.innerHTML = '<p style="color: red; text-align: center;">Failed to load evidence</p>';
    }
}

/**
 * Update statistics display
 */
function updateStats(stats) {
    totalIncidentsEl.textContent = stats.totalIncidents;
    highSeverityEl.textContent = stats.severityCounts.HIGH;
    mediumSeverityEl.textContent = stats.severityCounts.MEDIUM;
}

/**
 * Display evidence list
 */
function displayEvidence(evidenceList) {
    // Sort by timestamp (newest first)
    const sorted = [...evidenceList].sort((a, b) =>
        new Date(b.timestamp) - new Date(a.timestamp)
    );

    evidenceListEl.innerHTML = '';

    sorted.forEach(evidence => {
        const item = createEvidenceItem(evidence);
        evidenceListEl.appendChild(item);
    });
}

/**
 * Create evidence item element
 */
function createEvidenceItem(evidence) {
    const div = document.createElement('div');
    div.className = 'evidence-item';

    // Header with severity badge
    const header = document.createElement('div');
    header.className = 'evidence-header';

    const timestamp = document.createElement('span');
    timestamp.className = 'evidence-meta';
    timestamp.textContent = formatTimestamp(evidence.timestamp);

    const severityBadge = document.createElement('span');
    severityBadge.className = `severity-badge ${evidence.severity}`;
    severityBadge.textContent = evidence.severity;

    header.appendChild(timestamp);
    header.appendChild(severityBadge);

    // Metadata
    const meta = document.createElement('div');
    meta.className = 'evidence-meta';
    meta.innerHTML = `
        <strong>Source:</strong> ${evidence.source} | 
        <strong>Score:</strong> ${(evidence.maxScore * 100).toFixed(1)}%
        ${evidence.screenshot ? ' | 📸 Screenshot' : ''}
    `;

    // Categories
    const categoriesDiv = document.createElement('div');
    categoriesDiv.className = 'evidence-categories';

    if (evidence.categories && evidence.categories.length > 0) {
        evidence.categories.forEach(cat => {
            const tag = document.createElement('span');
            tag.className = 'category-tag';
            tag.textContent = `${cat.category.toUpperCase()} (${(cat.score * 100).toFixed(0)}%)`;
            categoriesDiv.appendChild(tag);
        });
    }

    // Text content
    const textDiv = document.createElement('div');
    textDiv.className = 'evidence-text';
    textDiv.textContent = evidence.text;

    // Assemble
    div.appendChild(header);
    div.appendChild(meta);
    div.appendChild(categoriesDiv);
    div.appendChild(textDiv);

    return div;
}

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        dateStyle: 'short',
        timeStyle: 'short'
    });
}

/**
 * Handle PDF export
 */
async function handleExportPDF() {
    try {
        exportPDFBtn.disabled = true;
        exportPDFBtn.textContent = '⏳ Generating...';

        const evidence = await getAllEvidence();

        if (evidence.length === 0) {
            alert('No evidence to export');
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

/**
 * Handle JSON export
 */
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

/**
 * Handle clear all evidence
 */
async function handleClearAll() {
    const confirmed = confirm(
        '⚠️ WARNING: This will permanently delete ALL evidence.\n\n' +
        'Are you sure you want to continue?\n\n' +
        'Tip: Export evidence to PDF/JSON before clearing.'
    );

    if (!confirmed) return;

    try {
        clearAllBtn.disabled = true;
        clearAllBtn.textContent = '⏳ Clearing...';

        await clearAllEvidence();
        await loadEvidence();

        clearAllBtn.textContent = '✅ Cleared!';
        setTimeout(() => {
            clearAllBtn.textContent = '🗑️ Clear All';
            clearAllBtn.disabled = false;
        }, 2000);
    } catch (error) {
        console.error('Failed to clear evidence:', error);
        alert('Failed to clear evidence: ' + error.message);
        clearAllBtn.textContent = '🗑️ Clear All';
        clearAllBtn.disabled = false;
    }
}
