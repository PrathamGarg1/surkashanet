/**
 * Displays a warning banner for toxic content
 * @param {HTMLElement} element - The message element to flag
 * @param {Object} detectionData - Contains severity, categories, maxScore
 */
export function showRedBanner(element, detectionData) {
    // Check if banner already exists for this element
    if (element.parentElement.querySelector('.suraksha-banner')) return;

    const { severity, categories, maxScore } = detectionData;

    // Severity-based styling
    const severityConfig = {
        'HIGH': {
            bgColor: '#ffebee',
            borderColor: '#d32f2f',
            textColor: '#b71c1c',
            icon: '🚨',
            label: 'CRITICAL THREAT'
        },
        'MEDIUM': {
            bgColor: '#fff3e0',
            borderColor: '#f57c00',
            textColor: '#e65100',
            icon: '⚠️',
            label: 'HARMFUL CONTENT'
        },
        'LOW': {
            bgColor: '#fff9c4',
            borderColor: '#fbc02d',
            textColor: '#f57f17',
            icon: '⚡',
            label: 'POTENTIALLY TOXIC'
        }
    };

    const config = severityConfig[severity] || severityConfig['MEDIUM'];

    // Highlight the message itself
    element.style.border = `2px solid ${config.borderColor}`;
    element.style.backgroundColor = `${config.borderColor}15`; // 15 = ~8% opacity in hex

    // Create the banner
    const banner = document.createElement('div');
    banner.className = 'suraksha-banner';
    banner.style.cssText = `
        background: linear-gradient(135deg, ${config.bgColor} 0%, ${config.bgColor}dd 100%);
        color: ${config.textColor};
        padding: 10px 14px;
        margin-top: 6px;
        border-radius: 6px;
        border-left: 4px solid ${config.borderColor};
        font-size: 12px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        z-index: 9999;
    `;

    // Format categories for display
    const categoryLabels = categories.map(c => {
        const label = c.category.replace('_', ' ').toUpperCase();
        return `${label} (${(c.score * 100).toFixed(0)}%)`;
    }).join(', ');

    // Main content container
    const contentDiv = document.createElement('div');
    contentDiv.style.cssText = 'display: flex; flex-direction: column; gap: 6px;';

    // Header with severity
    const headerSpan = document.createElement('div');
    headerSpan.style.cssText = 'display: flex; align-items: center; gap: 6px; font-weight: 600;';
    headerSpan.innerHTML = `
        <span style="font-size: 16px;">${config.icon}</span>
        <span>${config.label} DETECTED</span>
        <span style="background: ${config.borderColor}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-left: 4px;">
            ${severity}
        </span>
    `;

    // Categories display
    const categoriesSpan = document.createElement('div');
    categoriesSpan.style.cssText = 'font-size: 11px; opacity: 0.9;';
    categoriesSpan.innerHTML = `<strong>Detected:</strong> ${categoryLabels}`;

    // Action buttons container
    const actionsDiv = document.createElement('div');
    actionsDiv.style.cssText = 'display: flex; gap: 8px; margin-top: 4px;';

    const btnStyle = `
        border: 1px solid ${config.borderColor};
        background: white;
        color: ${config.textColor};
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        transition: all 0.2s;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    `;

    const blockBtn = document.createElement('button');
    blockBtn.innerText = '🚫 Block';
    blockBtn.style.cssText = btnStyle;
    blockBtn.onmouseover = () => blockBtn.style.background = config.borderColor;
    blockBtn.onmouseout = () => blockBtn.style.background = 'white';
    blockBtn.onclick = (e) => {
        e.stopPropagation();
        alert('Block action triggered (simulation)');
        // In real app, this would trigger platform specific block flow
    };

    const reportBtn = document.createElement('button');
    reportBtn.innerText = '📢 Report';
    reportBtn.style.cssText = btnStyle;
    reportBtn.onmouseover = () => reportBtn.style.background = config.borderColor;
    reportBtn.onmouseout = () => reportBtn.style.background = 'white';
    reportBtn.onclick = (e) => {
        e.stopPropagation();
        alert('Report action triggered (simulation)');
    };

    const saveBtn = document.createElement('button');
    saveBtn.innerText = '💾 Save Evidence';
    saveBtn.style.cssText = btnStyle;
    saveBtn.onmouseover = () => saveBtn.style.background = config.borderColor;
    saveBtn.onmouseout = () => saveBtn.style.background = 'white';
    saveBtn.onclick = (e) => {
        e.stopPropagation();
        alert('Evidence saved to Extension Storage for legal use.');
    };

    actionsDiv.appendChild(blockBtn);
    actionsDiv.appendChild(reportBtn);
    actionsDiv.appendChild(saveBtn);

    contentDiv.appendChild(headerSpan);
    contentDiv.appendChild(categoriesSpan);
    contentDiv.appendChild(actionsDiv);

    banner.appendChild(contentDiv);

    // Insert after the message element
    element.parentElement.appendChild(banner);
}
