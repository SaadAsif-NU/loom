// loom training dashboard frontend

const API_BASE = '/api';
let ws = null;
let isTraining = false;
let lossHistory = [];
let valLossHistory = [];
let currentStep = 0;
let totalSteps = 0;

// DOM elements
const configForm = document.getElementById('configForm');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const monitorPanel = document.getElementById('monitorPanel');
const generatePanel = document.getElementById('generatePanel');
const statusIndicator = document.getElementById('statusIndicator');
const metricStep = document.getElementById('metricStep');
const metricLoss = document.getElementById('metricLoss');
const metricValLoss = document.getElementById('metricValLoss');
const metricETA = document.getElementById('metricETA');
const lossCurve = document.getElementById('lossCurve');
const generateBtn = document.getElementById('generateBtn');
const messagePanel = document.getElementById('messagePanel');
const messageContent = document.getElementById('messageContent');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    startBtn.addEventListener('click', handleStartTraining);
    stopBtn.addEventListener('click', handleStopTraining);
    generateBtn.addEventListener('click', handleGenerate);

    // Load defaults
    fetch(`${API_BASE}/defaults`)
        .then(r => r.json())
        .then(defaults => {
            Object.entries(defaults).forEach(([key, value]) => {
                const el = document.getElementById(camelToHyphen(key));
                if (el) el.value = value;
            });
        });
});

function camelToHyphen(str) {
    return str.replace(/([a-z])([A-Z])/g, '$1-$2').toLowerCase();
}

function getFormConfig() {
    const formData = new FormData(configForm);
    const config = {};
    formData.forEach((value, key) => {
        const num = Number(value);
        config[key] = isNaN(num) ? value : num;
    });
    return config;
}

async function handleStartTraining() {
    const config = getFormConfig();

    try {
        const res = await fetch(`${API_BASE}/train/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });

        if (!res.ok) {
            const err = await res.json();
            showMessage(`Error: ${err.detail}`, 'error');
            return;
        }

        const data = await res.json();
        const runId = data.run_id;

        // Reset UI
        lossHistory = [];
        valLossHistory = [];
        currentStep = 0;
        totalSteps = config.steps;
        isTraining = true;

        // Update UI
        startBtn.disabled = true;
        stopBtn.disabled = false;
        configForm.style.pointerEvents = 'none';
        configForm.style.opacity = '0.5';
        monitorPanel.style.display = 'block';
        generatePanel.style.display = 'none';
        statusIndicator.textContent = 'training';
        statusIndicator.className = 'status-indicator running';

        // Update metrics immediately with correct total_steps
        metricStep.textContent = `0 / ${totalSteps}`;
        metricLoss.textContent = '–';
        metricValLoss.textContent = '–';
        metricETA.textContent = '–';

        // Connect WebSocket
        connectWebSocket(runId);

        showMessage(`Training started (${runId})`, 'info');
    } catch (e) {
        showMessage(`Error: ${e.message}`, 'error');
    }
}

function connectWebSocket(runId) {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://${window.location.host}${API_BASE}/train/stream`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        try {
            const evt = JSON.parse(event.data);
            handleTrainingEvent(evt);
        } catch (e) {
            console.error('Failed to parse event', e);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error', error);
        showMessage('WebSocket error. Connection lost.', 'error');
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
    };
}

function handleTrainingEvent(evt) {
    if (evt.event_type === 'step') {
        currentStep = evt.step;
        lossHistory.push(evt.loss);
        if (evt.val_loss !== null) {
            valLossHistory.push(evt.val_loss);
        }
        updateMetrics(evt);
        redrawLossCurve();
    } else if (evt.event_type === 'completed') {
        isTraining = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        configForm.style.pointerEvents = 'auto';
        configForm.style.opacity = '1';
        statusIndicator.textContent = 'completed';
        statusIndicator.className = 'status-indicator completed';
        generatePanel.style.display = 'block';
        generateBtn.disabled = false;
        showMessage('Training completed! Checkpoint saved.', 'success');
        if (ws) ws.close();
    } else if (evt.event_type === 'stopped') {
        isTraining = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        configForm.style.pointerEvents = 'auto';
        configForm.style.opacity = '1';
        statusIndicator.textContent = 'stopped';
        statusIndicator.className = 'status-indicator completed';
        generatePanel.style.display = 'block';
        generateBtn.disabled = false;
        showMessage('Training stopped. Checkpoint saved (you can resume later).', 'success');
        if (ws) ws.close();
    } else if (evt.event_type === 'failed') {
        isTraining = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        configForm.style.pointerEvents = 'auto';
        configForm.style.opacity = '1';
        statusIndicator.textContent = 'failed';
        statusIndicator.className = 'status-indicator failed';
        showMessage(`Training failed: ${evt.message}`, 'error');
        if (ws) ws.close();
    }
}

function updateMetrics(evt) {
    metricStep.textContent = `${evt.step} / ${totalSteps}`;
    metricLoss.textContent = evt.loss ? evt.loss.toFixed(4) : '–';
    metricValLoss.textContent = evt.val_loss ? evt.val_loss.toFixed(4) : '–';
    if (evt.eta_seconds) {
        metricETA.textContent = formatTime(evt.eta_seconds);
    }
}

function formatTime(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

function redrawLossCurve() {
    const svg = lossCurve;
    const width = 800;
    const height = 300;
    const padding = 40;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Clear paths
    let trainPath = svg.querySelector('.train-path');
    let valPath = svg.querySelector('.val-path');
    if (trainPath) trainPath.remove();
    if (valPath) valPath.remove();

    if (lossHistory.length === 0) return;

    const minLoss = Math.min(...lossHistory, ...(valLossHistory || [0]));
    const maxLoss = Math.max(...lossHistory, ...(valLossHistory || [0]));
    const range = Math.max(maxLoss - minLoss, 0.1);

    // Draw training loss curve
    if (lossHistory.length > 0) {
        const points = lossHistory.map((loss, i) => {
            const x = padding + (i / Math.max(lossHistory.length - 1, 1)) * plotWidth;
            const y = height - padding - ((loss - minLoss) / range) * plotHeight;
            return `${x},${y}`;
        });
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        path.setAttribute('points', points.join(' '));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', '#6366f1');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('vector-effect', 'non-scaling-stroke');
        path.className = 'train-path';
        svg.appendChild(path);
    }

    // Draw validation loss curve
    if (valLossHistory.length > 0) {
        const points = valLossHistory.map((loss, i) => {
            const x = padding + (i / Math.max(valLossHistory.length - 1, 1)) * plotWidth;
            const y = height - padding - ((loss - minLoss) / range) * plotHeight;
            return `${x},${y}`;
        });
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        path.setAttribute('points', points.join(' '));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', '#22d3ee');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('stroke-dasharray', '5,5');
        path.setAttribute('vector-effect', 'non-scaling-stroke');
        path.className = 'val-path';
        svg.appendChild(path);
    }

    // Draw axes and grid
    drawAxes(svg, width, height, padding, minLoss, maxLoss, range);
}

function drawAxes(svg, width, height, padding, minLoss, maxLoss, range) {
    // Remove old axes
    svg.querySelectorAll('line, text').forEach(el => {
        if (el.classList.contains('axis') || el.classList.contains('axis-label')) {
            el.remove();
        }
    });

    // X-axis
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', padding);
    xAxis.setAttribute('y1', height - padding);
    xAxis.setAttribute('x2', width - padding);
    xAxis.setAttribute('y2', height - padding);
    xAxis.setAttribute('stroke', '#475569');
    xAxis.setAttribute('stroke-width', '1');
    xAxis.className = 'axis';
    svg.appendChild(xAxis);

    // Y-axis
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', padding);
    yAxis.setAttribute('y1', padding);
    yAxis.setAttribute('x2', padding);
    yAxis.setAttribute('y2', height - padding);
    yAxis.setAttribute('stroke', '#475569');
    yAxis.setAttribute('stroke-width', '1');
    yAxis.className = 'axis';
    svg.appendChild(yAxis);

    // Y-axis labels
    const steps = 5;
    for (let i = 0; i <= steps; i++) {
        const loss = minLoss + (range * i) / steps;
        const y = height - padding - (i / steps) * (height - 2 * padding);
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', padding - 10);
        text.setAttribute('y', y + 4);
        text.setAttribute('text-anchor', 'end');
        text.setAttribute('font-size', '12');
        text.setAttribute('fill', '#94a3b8');
        text.textContent = loss.toFixed(2);
        text.className = 'axis-label';
        svg.appendChild(text);
    }

    // Legend
    const legendX = width - padding - 150;
    const legendY = padding + 10;

    const trainDot = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    trainDot.setAttribute('x1', legendX);
    trainDot.setAttribute('y1', legendY);
    trainDot.setAttribute('x2', legendX + 20);
    trainDot.setAttribute('y2', legendY);
    trainDot.setAttribute('stroke', '#6366f1');
    trainDot.setAttribute('stroke-width', '2');
    svg.appendChild(trainDot);

    const trainLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    trainLabel.setAttribute('x', legendX + 25);
    trainLabel.setAttribute('y', legendY + 4);
    trainLabel.setAttribute('font-size', '12');
    trainLabel.setAttribute('fill', '#cbd5e1');
    trainLabel.textContent = 'train';
    svg.appendChild(trainLabel);

    if (valLossHistory.length > 0) {
        const valDot = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        valDot.setAttribute('x1', legendX);
        valDot.setAttribute('y1', legendY + 20);
        valDot.setAttribute('x2', legendX + 20);
        valDot.setAttribute('y2', legendY + 20);
        valDot.setAttribute('stroke', '#22d3ee');
        valDot.setAttribute('stroke-width', '2');
        valDot.setAttribute('stroke-dasharray', '5,5');
        svg.appendChild(valDot);

        const valLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        valLabel.setAttribute('x', legendX + 25);
        valLabel.setAttribute('y', legendY + 24);
        valLabel.setAttribute('font-size', '12');
        valLabel.setAttribute('fill', '#cbd5e1');
        valLabel.textContent = 'val';
        svg.appendChild(valLabel);
    }
}

async function handleStopTraining() {
    try {
        const res = await fetch(`${API_BASE}/train/stop`, { method: 'POST' });
        if (res.ok) {
            showMessage('Stopping training...', 'info');
        }
    } catch (e) {
        showMessage(`Error: ${e.message}`, 'error');
    }
}

async function handleGenerate() {
    if (!isTraining) {
        generateBtn.disabled = false;
    }

    const prompt = document.getElementById('prompt').value;
    const tokens = parseInt(document.getElementById('genTokens').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const topK = parseInt(document.getElementById('topK').value);

    generateBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, tokens, temperature, top_k: topK }),
        });

        if (!res.ok) {
            const err = await res.json();
            showMessage(`Error: ${err.detail}`, 'error');
            generateBtn.disabled = false;
            return;
        }

        const data = await res.json();
        document.getElementById('generatedText').textContent = data.text;
        document.getElementById('generatedOutput').style.display = 'block';
        showMessage('Generated sample ready', 'success');
    } catch (e) {
        showMessage(`Error: ${e.message}`, 'error');
    } finally {
        generateBtn.disabled = !isTraining;
    }
}

function showMessage(text, type = 'info') {
    messagePanel.style.display = 'block';
    const timestamp = new Date().toLocaleTimeString();
    const icon = type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️';
    messageContent.innerHTML += `<div>${icon} [${timestamp}] ${text}</div>`;
    messageContent.scrollTop = messageContent.scrollHeight;
}
