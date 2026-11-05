/**
 * Screenshot Capture - Visual Evidence Collection
 * Captures message elements as images for evidence documentation
 */

import html2canvas from 'html2canvas';

/**
 * Capture screenshot of a DOM element
 * @param {HTMLElement} element - Element to capture
 * @param {Object} options - Capture options
 * @returns {Promise<string>} Base64 encoded image data URL
 */
export async function captureElement(element, options = {}) {
    try {
        const defaultOptions = {
            backgroundColor: '#ffffff',
            scale: 2, // Higher quality
            logging: false,
            useCORS: true,
            allowTaint: true,
            ...options
        };

        // Capture the element
        const canvas = await html2canvas(element, defaultOptions);

        // Convert to data URL
        const dataUrl = canvas.toDataURL('image/png');

        console.log('📸 Screenshot captured:', {
            width: canvas.width,
            height: canvas.height,
            size: (dataUrl.length / 1024).toFixed(2) + ' KB'
        });

        return dataUrl;
    } catch (error) {
        console.error('❌ Screenshot capture failed:', error);
        throw error;
    }
}

/**
 * Capture message with context (sender, timestamp, etc.)
 * @param {HTMLElement} messageElement - Message element to capture
 * @returns {Promise<Object>} Screenshot data with metadata
 */
export async function captureMessageWithContext(messageElement) {
    try {
        // Find the parent container that includes context
        let contextElement = messageElement;

        // Try to find a better parent that includes metadata
        const parentMessage = messageElement.closest('[data-id]') ||
            messageElement.closest('.message') ||
            messageElement.closest('[class*="message"]');

        if (parentMessage) {
            contextElement = parentMessage;
        }

        // Capture the element
        const screenshot = await captureElement(contextElement, {
            backgroundColor: '#e5ddd5', // WhatsApp background color
        });

        // Extract metadata
        const metadata = {
            timestamp: new Date().toISOString(),
            elementType: contextElement.tagName,
            elementClasses: contextElement.className,
            dimensions: {
                width: contextElement.offsetWidth,
                height: contextElement.offsetHeight
            }
        };

        return {
            screenshot: screenshot,
            metadata: metadata
        };
    } catch (error) {
        console.error('❌ Failed to capture message with context:', error);
        throw error;
    }
}

/**
 * Capture full page screenshot (for context)
 * @returns {Promise<string>} Base64 encoded image data URL
 */
export async function captureFullPage() {
    try {
        const screenshot = await captureElement(document.body, {
            backgroundColor: null, // Preserve original background
            windowWidth: document.documentElement.scrollWidth,
            windowHeight: document.documentElement.scrollHeight
        });

        return screenshot;
    } catch (error) {
        console.error('❌ Full page capture failed:', error);
        throw error;
    }
}

/**
 * Download screenshot as PNG file
 * @param {string} dataUrl - Base64 data URL
 * @param {string} filename - Filename (without extension)
 */
export function downloadScreenshot(dataUrl, filename = 'evidence') {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `${filename}_${Date.now()}.png`;
    link.click();
}

/**
 * Convert data URL to Blob
 * @param {string} dataUrl - Base64 data URL
 * @returns {Blob} Image blob
 */
export function dataUrlToBlob(dataUrl) {
    const arr = dataUrl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);

    while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
    }

    return new Blob([u8arr], { type: mime });
}

/**
 * Compress screenshot for storage optimization
 * @param {string} dataUrl - Base64 data URL
 * @param {number} quality - Compression quality (0-1)
 * @returns {Promise<string>} Compressed data URL
 */
export async function compressScreenshot(dataUrl, quality = 0.8) {
    return new Promise((resolve, reject) => {
        const img = new Image();

        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            // Convert to JPEG with compression
            const compressed = canvas.toDataURL('image/jpeg', quality);
            resolve(compressed);
        };

        img.onerror = reject;
        img.src = dataUrl;
    });
}
