import { saveEvidence } from './storage_manager.js';

/**
 * Generates a SHA-256 hash of the input text.
 * @param {string} text - The content to hash.
 * @returns {Promise<string>} - The hex string of the hash.
 */
async function generateHash(text) {
    const encoder = new TextEncoder();
    const data = encoder.encode(text);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return hashHex;
}

/**
 * Logs an incident with enhanced metadata and persistent storage
 * @param {object} incidentData - Contains text, source, categories, severity, scores, screenshot
 */
async function logIncident(incidentData) {
    const timestamp = new Date().toISOString();
    const timestampMs = incidentData.timestamp || Date.now();
    const hash = await generateHash(incidentData.text);

    const record = {
        id: hash,
        timestamp: timestamp,
        timestampMs: timestampMs,
        source: incidentData.source,
        text: incidentData.text,

        // Toxicity detection metadata
        severity: incidentData.severity || 'UNKNOWN',
        maxScore: incidentData.maxScore || 0,
        categories: incidentData.categories || [],
        allScores: incidentData.allScores || [],

        // Visual evidence
        screenshot: incidentData.screenshot || null,

        // Chain of custody
        chainOfCustody: [
            {
                action: 'CREATED',
                timestamp: timestamp,
                actor: 'SYSTEM'
            }
        ],

        // Legacy fields
        score: incidentData.maxScore || incidentData.score || 0,
        label: incidentData.categories?.[0]?.category || incidentData.label || 'unknown'
    };

    // Save to chrome.storage.local
    const data = await chrome.storage.local.get("incident_logs");
    let logs = data.incident_logs || [];
    logs.push(record);
    await chrome.storage.local.set({ incident_logs: logs });

    // Save to persistent storage
    try {
        await saveEvidence(record);
    } catch (error) {
        console.error('⚠️ Failed to save to persistent storage:', error);
    }

    console.log("🚨 Incident logged:", {
        id: record.id.substring(0, 8) + '...',
        severity: record.severity,
        categories: record.categories.map(c => c.category).join(', '),
        maxScore: (record.maxScore * 100).toFixed(1) + '%',
        hasScreenshot: !!record.screenshot
    });

    return record;
}

/**
 * Update evidence with screenshot (called from content script)
 * @param {string} text - Original text to match
 * @param {string} screenshot - Base64 screenshot data
 */
async function updateEvidenceScreenshot(text, screenshot) {
    try {
        const hash = await generateHash(text);
        const data = await chrome.storage.local.get("incident_logs");
        let logs = data.incident_logs || [];

        // Find and update the evidence
        const index = logs.findIndex(log => log.id === hash);
        if (index !== -1) {
            logs[index].screenshot = screenshot;
            logs[index].chainOfCustody.push({
                action: 'SCREENSHOT_ADDED',
                timestamp: new Date().toISOString(),
                actor: 'SYSTEM'
            });

            await chrome.storage.local.set({ incident_logs: logs });

            // Update in persistent storage too
            await saveEvidence(logs[index]);

            console.log('📸 Screenshot added to evidence:', hash.substring(0, 8) + '...');
            return true;
        }
        return false;
    } catch (error) {
        console.error('❌ Failed to update screenshot:', error);
        return false;
    }
}

export { generateHash, logIncident, updateEvidenceScreenshot };
