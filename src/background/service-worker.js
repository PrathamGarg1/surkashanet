import { pipeline, env } from '@xenova/transformers';
import { logIncident } from '../utils/evidence_logger.js';

// Configuration for Service Worker environment
// In Chrome Extensions, we must explicitly enable local models and block remote models.
env.allowLocalModels = true;
env.allowRemoteModels = false; 

// Tell Transformers.js where to look for the local model folder inside the extension
// Models should go inside /assets/models/
env.localModelPath = chrome.runtime.getURL('assets/models/');
env.useBrowserCache = true; // Enable caching for faster subsequent loads

// 1. Force WASM backend (prevent WebGL detection which fails in SW)
// 2. Disable multithreading (prevents spawning workers via Blob URLs which can fail)
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

// 3. Point WASM files to extension's assets folder
//    The WASM files are copied to dist/assets/ by the Vite build process
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('assets/');

let classifier = null;

// Toxicity category thresholds and severity mapping
// The MACD ShareChat model is a binary classifier:
// Label 0 is usually safe/non-abusive, Label 1 is abusive (or vice versa based on id2label metadata).
// In our pipeline test, it outputs either "abusive" or "non-abusive" directly.
const TOXICITY_CONFIG = {
    threshold: 0.65, // Confidence threshold for 'High Severity Abuse'
    labels: {
        abusive: ['abusive', 'label_1'], // Explicit matches for toxic strings
        safe: ['non-abusive', 'label_0']
    }
};

// Initialize the toxicity classification pipeline
async function initializeModel() {
    if (classifier) return;
    console.log('🛡️ SurakshaNet: Loading CUSTOM local MACD toxicity model...');
    console.log('⏳ Loading Quantized ONNX model from extension assets package...');

    try {
        // Load our custom fine-tuned, int8 quantized ONNX model
        // The path maps to chrome-extension://[ID]/assets/models/custom-macd-model/
        classifier = await pipeline('text-classification', 'custom-macd-model', {
            top_k: null // Return all labels with scores (in our case 'High Severity Abuse' vs 'Safe')
        });
        console.log('✅ Custom toxicity detection model loaded successfully.');
    } catch (error) {
        console.error('❌ Failed to load custom toxicity model:', error);
        throw error;
    }
}

/**
 * Analyzes toxicity results and determines if content should be flagged
 * @param {Array} results - Classification results from MACD local model
 * @returns {Object} Analysis with isToxic, categories, maxScore, severity
 */
function analyzeToxicity(results) {
    let isToxic = false;
    let maxScore = 0;
    let severity = 'LOW';
    let sumAbusiveScores = 0;

    for (const result of results) {
        // e.g. "abusive" or "non-abusive"
        const label = result.label.toLowerCase().trim();
        const score = result.score;
        
        // BUGFIX: MUST use exact matching because "non-abusive".includes("abusive") is true!
        const isAbusiveLabel = TOXICITY_CONFIG.labels.abusive.includes(label) || label === 'label_0' /* if 0=abusive */;
        const isSafeLabel = TOXICITY_CONFIG.labels.safe.includes(label) || label === 'label_1';

        if (isAbusiveLabel && !isSafeLabel) {
            sumAbusiveScores += score;
            if (score >= TOXICITY_CONFIG.threshold) {
                isToxic = true;
                maxScore = Math.max(maxScore, score);
                severity = maxScore > 0.85 ? 'HIGH' : 'MEDIUM'; 
            }
        }
    }

    return {
        isToxic: isToxic,
        categories: isToxic ? [{ category: 'Abusive (Hinglish MACD)', score: maxScore, severity: severity }] : [],
        maxScore: maxScore,
        severity: isToxic ? severity : null,
        allScores: results 
    };
}

// Initialize on install/startup
chrome.runtime.onInstalled.addListener(() => {
    console.log('🛡️ SurakshaNet extension installed. Initializing model...');
    initializeModel().catch(err => {
        console.error('Failed to initialize model on install:', err);
    });
});

// Also initialize on startup
chrome.runtime.onStartup.addListener(() => {
    console.log('🛡️ SurakshaNet extension started. Initializing model...');
    initializeModel().catch(err => {
        console.error('Failed to initialize model on startup:', err);
    });
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === 'ANALYZE_TEXT') {
        console.log("🔍 SurakshaNet: Analyzing text:", request.text.substring(0, 50) + '...');

        (async () => {
            // Ensure model is loaded
            if (!classifier) {
                try {
                    await initializeModel();
                } catch (error) {
                    console.error('Model initialization failed:', error);
                    sendResponse({
                        isToxic: false,
                        error: 'Model failed to load',
                        errorDetails: error.message
                    });
                    return;
                }
            }

            try {
                // Run toxicity classification
                const results = await classifier(request.text);

                // Analyze results
                const analysis = analyzeToxicity(results);

                console.log('📊 Analysis result:', {
                    isToxic: analysis.isToxic,
                    categories: analysis.categories.map(c => c.category),
                    severity: analysis.severity,
                    maxScore: analysis.maxScore
                });

                // Log incident if toxic content detected
                if (analysis.isToxic) {
                    await logIncident({
                        text: request.text,
                        source: request.source,
                        categories: analysis.categories,
                        severity: analysis.severity,
                        maxScore: analysis.maxScore,
                        allScores: analysis.allScores,
                        timestamp: Date.now()
                    });
                }

                // Send response back to content script
                sendResponse({
                    isToxic: analysis.isToxic,
                    categories: analysis.categories,
                    severity: analysis.severity,
                    maxScore: analysis.maxScore
                });

            } catch (err) {
                console.error("❌ Classification error:", err);
                sendResponse({
                    isToxic: false,
                    error: 'Classification failed',
                    errorDetails: err.message
                });
            }
        })();

        return true; // Keep channel open for async response
    }

    // Handle screenshot updates from content script
    if (request.type === 'UPDATE_EVIDENCE_SCREENSHOT') {
        (async () => {
            const { updateEvidenceScreenshot } = await import('../utils/evidence_logger.js');
            const success = await updateEvidenceScreenshot(request.text, request.screenshot);
            sendResponse({ success });
        })();
        return true;
    }
});
