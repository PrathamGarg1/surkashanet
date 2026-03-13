console.log("Popup opened");

async function loadLogs() {
    const logContainer = document.getElementById('log-container');
    const data = await chrome.storage.local.get("incident_logs");
    const logs = data.incident_logs || [];

    if (logs.length === 0) {
        logContainer.innerHTML = '<div class="log-item">No incidents detected yet.</div>';
        return;
    }

    logContainer.innerHTML = '';
    // Show latest first
    logs.slice().reverse().forEach(log => {
        const div = document.createElement('div');
        div.className = 'log-item';
        div.innerHTML = `
            <div><strong>Source:</strong> ${log.source}</div>
            <div><strong>Time:</strong> ${new Date(log.timestamp).toLocaleString()}</div>
            <div><strong>Score:</strong> ${(log.score * 100).toFixed(1)}%</div>
            <div style="font-size: 10px; color: #666; margin-top: 2px;">Hash: ${log.id.substring(0, 10)}...</div>
        `;
        logContainer.appendChild(div);
    });
}

document.addEventListener('DOMContentLoaded', loadLogs);
