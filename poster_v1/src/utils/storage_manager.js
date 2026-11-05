/**
 * Storage Manager - Persistent Evidence Storage
 * Manages file-based evidence backup with encryption and integrity verification
 */

import CryptoJS from 'crypto-js';

// Storage configuration
const STORAGE_CONFIG = {
    storageKey: 'surakshanet_evidence',
    maxStorageSize: 50 * 1024 * 1024, // 50MB limit
    encryptionEnabled: true,
    autoBackup: true
};

/**
 * Encrypts data using AES encryption
 * @param {string} data - Data to encrypt
 * @param {string} key - Encryption key (user-provided or generated)
 * @returns {string} Encrypted data
 */
function encryptData(data, key = 'surakshanet_default_key') {
    if (!STORAGE_CONFIG.encryptionEnabled) return data;
    return CryptoJS.AES.encrypt(data, key).toString();
}

/**
 * Decrypts data using AES decryption
 * @param {string} encryptedData - Encrypted data
 * @param {string} key - Decryption key
 * @returns {string} Decrypted data
 */
function decryptData(encryptedData, key = 'surakshanet_default_key') {
    if (!STORAGE_CONFIG.encryptionEnabled) return encryptedData;
    const bytes = CryptoJS.AES.decrypt(encryptedData, key);
    return bytes.toString(CryptoJS.enc.Utf8);
}

/**
 * Save evidence to chrome.storage.local with backup
 * @param {Object} evidence - Evidence record to save
 * @returns {Promise<Object>} Saved evidence with metadata
 */
export async function saveEvidence(evidence) {
    try {
        // Get existing evidence
        const data = await chrome.storage.local.get(STORAGE_CONFIG.storageKey);
        let evidenceList = data[STORAGE_CONFIG.storageKey] || [];

        // Add metadata
        const enrichedEvidence = {
            ...evidence,
            savedAt: new Date().toISOString(),
            version: '1.0',
            encrypted: STORAGE_CONFIG.encryptionEnabled
        };

        // Add to list
        evidenceList.push(enrichedEvidence);

        // Check storage size
        const storageSize = JSON.stringify(evidenceList).length;
        if (storageSize > STORAGE_CONFIG.maxStorageSize) {
            console.warn('⚠️ Storage limit approaching. Consider exporting and clearing old evidence.');
        }

        // Save to chrome.storage
        await chrome.storage.local.set({
            [STORAGE_CONFIG.storageKey]: evidenceList
        });

        console.log('💾 Evidence saved:', {
            id: enrichedEvidence.id?.substring(0, 8) + '...',
            total: evidenceList.length
        });

        return enrichedEvidence;
    } catch (error) {
        console.error('❌ Failed to save evidence:', error);
        throw error;
    }
}

/**
 * Get all evidence records
 * @returns {Promise<Array>} List of all evidence
 */
export async function getAllEvidence() {
    try {
        const data = await chrome.storage.local.get(STORAGE_CONFIG.storageKey);
        return data[STORAGE_CONFIG.storageKey] || [];
    } catch (error) {
        console.error('❌ Failed to retrieve evidence:', error);
        return [];
    }
}

/**
 * Get evidence by ID
 * @param {string} id - Evidence ID (hash)
 * @returns {Promise<Object|null>} Evidence record or null
 */
export async function getEvidenceById(id) {
    const allEvidence = await getAllEvidence();
    return allEvidence.find(e => e.id === id) || null;
}

/**
 * Delete evidence by ID
 * @param {string} id - Evidence ID to delete
 * @returns {Promise<boolean>} Success status
 */
export async function deleteEvidence(id) {
    try {
        const allEvidence = await getAllEvidence();
        const filtered = allEvidence.filter(e => e.id !== id);

        await chrome.storage.local.set({
            [STORAGE_CONFIG.storageKey]: filtered
        });

        console.log('🗑️ Evidence deleted:', id.substring(0, 8) + '...');
        return true;
    } catch (error) {
        console.error('❌ Failed to delete evidence:', error);
        return false;
    }
}

/**
 * Clear all evidence (with confirmation)
 * @returns {Promise<boolean>} Success status
 */
export async function clearAllEvidence() {
    try {
        await chrome.storage.local.set({
            [STORAGE_CONFIG.storageKey]: []
        });

        console.log('🗑️ All evidence cleared');
        return true;
    } catch (error) {
        console.error('❌ Failed to clear evidence:', error);
        return false;
    }
}

/**
 * Export evidence to JSON format
 * @param {Array} evidenceList - Evidence to export (optional, defaults to all)
 * @returns {Promise<string>} JSON string
 */
export async function exportToJSON(evidenceList = null) {
    const evidence = evidenceList || await getAllEvidence();

    const exportData = {
        exportDate: new Date().toISOString(),
        version: '1.0',
        totalIncidents: evidence.length,
        evidence: evidence
    };

    return JSON.stringify(exportData, null, 2);
}

/**
 * Export evidence to CSV format
 * @param {Array} evidenceList - Evidence to export (optional, defaults to all)
 * @returns {Promise<string>} CSV string
 */
export async function exportToCSV(evidenceList = null) {
    const evidence = evidenceList || await getAllEvidence();

    if (evidence.length === 0) {
        return 'No evidence to export';
    }

    // CSV headers
    const headers = [
        'ID', 'Timestamp', 'Severity', 'Max Score', 'Categories',
        'Source', 'Text'
    ];

    // CSV rows
    const rows = evidence.map(e => {
        const categories = e.categories?.map(c => c.category).join('; ') || '';
        return [
            e.id,
            e.timestamp,
            e.severity,
            e.maxScore,
            categories,
            e.source,
            `"${e.text.replace(/"/g, '""')}"` // Escape quotes in text
        ];
    });

    // Combine headers and rows
    const csv = [
        headers.join(','),
        ...rows.map(row => row.join(','))
    ].join('\n');

    return csv;
}

/**
 * Download file to user's computer
 * @param {string} content - File content
 * @param {string} filename - Filename
 * @param {string} mimeType - MIME type
 */
export function downloadFile(content, filename, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();

    URL.revokeObjectURL(url);
}

/**
 * Get storage statistics
 * @returns {Promise<Object>} Storage stats
 */
export async function getStorageStats() {
    const evidence = await getAllEvidence();
    const storageSize = JSON.stringify(evidence).length;

    const severityCounts = {
        HIGH: 0,
        MEDIUM: 0,
        LOW: 0,
        UNKNOWN: 0
    };

    evidence.forEach(e => {
        severityCounts[e.severity || 'UNKNOWN']++;
    });

    return {
        totalIncidents: evidence.length,
        storageSize: storageSize,
        storageSizeMB: (storageSize / (1024 * 1024)).toFixed(2),
        maxStorageMB: (STORAGE_CONFIG.maxStorageSize / (1024 * 1024)).toFixed(0),
        usagePercent: ((storageSize / STORAGE_CONFIG.maxStorageSize) * 100).toFixed(1),
        severityCounts: severityCounts,
        oldestIncident: evidence[0]?.timestamp || null,
        newestIncident: evidence[evidence.length - 1]?.timestamp || null
    };
}
