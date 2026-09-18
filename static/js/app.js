/**
 * app.js — Day 9: Tactical Sonar Defense Dashboard Client
 * Handles all UI interactions, API calls, canvas rendering, and live state management.
 */

"use strict";

// ─── GLOBAL STATE ──────────────────────────────────────────────────────────────
const State = {
    features: null,           // Active 60-band signal (raw float array)
    threshold: 0.43,          // Current decision threshold
    lastPrediction: null,     // Most recent API prediction result
    refProfiles: null,        // Mean Mine / Mean Rock reference spectra
    curated: null,            // Pre-loaded curated test samples
    batchFile: null,          // CSV file object from dropzone
    activeTab: "presets",     // "presets" | "manual" | "batch"
    showMineRef: true,
    showRockRef: true,
    showActive: true,
};

// ─── DOM REFERENCES ────────────────────────────────────────────────────────────
const DOM = {
    // Tabs
    tabBtns: document.querySelectorAll(".tab-btn"),
    tabPanes: document.querySelectorAll(".tab-content"),

    // Preset buttons
    btnConfidentMine: document.getElementById("btn-preset-confident-mine"),
    btnBorderlineMine: document.getElementById("btn-preset-borderline-mine"),
    btnConfidentRock: document.getElementById("btn-preset-confident-rock"),
    btnBorderlineRock: document.getElementById("btn-preset-borderline-rock"),
    btnRandomSignal: document.getElementById("btn-random-signal"),

    // Manual input
    manualTextarea: document.getElementById("manual-feature-input"),
    featureCountDisplay: document.getElementById("feature-count-display"),
    btnFillSample: document.getElementById("btn-fill-sample"),
    btnClearManual: document.getElementById("btn-clear-manual"),

    // Batch CSV
    dropzone: document.getElementById("csv-dropzone"),
    batchFileInput: document.getElementById("batch-file-input"),
    dropzoneText: document.getElementById("dropzone-text"),
    btnRunBatch: document.getElementById("btn-run-batch"),

    // Threshold
    thresholdSlider: document.getElementById("threshold-slider"),
    thresholdDisplayBadge: document.getElementById("threshold-display-badge"),
    btnTauSafe: document.getElementById("btn-tau-safe"),
    btnTauDefault: document.getElementById("btn-tau-default"),
    btnTauPrecision: document.getElementById("btn-tau-precision"),

    // Primary classify button
    btnClassify: document.getElementById("btn-classify-signal"),

    // Threat assessment panel
    threatBanner: document.getElementById("threat-banner"),
    threatDefconBadge: document.getElementById("threat-defcon-badge"),
    threatStatusIcon: document.getElementById("threat-status-icon"),
    threatAlertLabel: document.getElementById("threat-alert-label"),
    threatClassificationName: document.getElementById("threat-classification-name"),
    threatMessage: document.getElementById("threat-message"),

    // Metrics
    metricMineProb: document.getElementById("metric-mine-prob"),
    metricRockProb: document.getElementById("metric-rock-prob"),
    metricConfidence: document.getElementById("metric-confidence"),
    metricLatency: document.getElementById("metric-latency"),
    metricMarginText: document.getElementById("metric-margin-text"),
    barMineProb: document.getElementById("bar-mine-prob"),
    barRockProb: document.getElementById("bar-rock-prob"),

    // Spectral canvas
    spectralCanvas: document.getElementById("spectralCanvas"),
    toggleActive: document.getElementById("toggle-active"),
    toggleMine: document.getElementById("toggle-mine"),
    toggleRock: document.getElementById("toggle-rock"),

    // Explainability
    btnRunExplain: document.getElementById("btn-run-explain"),
    explainBtnIcon: document.getElementById("explain-btn-icon"),
    explainBtnText: document.getElementById("explain-btn-text"),
    explainStatus: document.getElementById("explainability-status"),
    attributionColumns: document.getElementById("attribution-columns"),
    listMineDrv: document.getElementById("list-mine-drivers"),
    listRockDrv: document.getElementById("list-rock-drivers"),

    // Batch modal
    batchModal: document.getElementById("batch-modal"),
    modalBackdrop: document.getElementById("modal-backdrop"),
    btnCloseModal: document.getElementById("btn-close-modal"),
    btnModalDismiss: document.getElementById("btn-modal-dismiss"),
    batchSummaryCards: document.getElementById("batch-summary-cards"),
    batchTableBody: document.getElementById("batch-table-body"),
};

// ─── INIT ──────────────────────────────────────────────────────────────────────
async function init() {
    try {
        const resp = await fetch("/api/samples");
        const data = await resp.json();
        if (data.status === "success") {
            State.curated = data.samples;
            State.refProfiles = data.reference_profiles;
        }
    } catch (e) {
        console.warn("Could not load curated samples:", e);
    }

    // Load confident mine as default signal on page load
    if (State.curated?.confident_mine) {
        loadPreset("confident_mine");
    }

    attachEventListeners();
    renderSpectralCanvas();
}

// ─── TAB SWITCHING ─────────────────────────────────────────────────────────────
function switchTab(tabName) {
    State.activeTab = tabName;
    DOM.tabBtns.forEach(btn => btn.classList.toggle("active", btn.dataset.tab === tabName));
    DOM.tabPanes.forEach(pane => pane.classList.toggle("active", pane.id === `pane-${tabName}`));
}

// ─── PRESET LOADING ────────────────────────────────────────────────────────────
function loadPreset(presetKey) {
    if (!State.curated || !State.curated[presetKey]) return;
    const sample = State.curated[presetKey];
    State.features = sample.features;
    renderSpectralCanvas();
    flashHint(`Loaded: ${sample.label}`);
}

function loadRandomSignal() {
    if (!State.curated) return;
    const keys = Object.keys(State.curated);
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    loadPreset(randomKey);
}

// ─── MANUAL INPUT PARSING ──────────────────────────────────────────────────────
function parseManualInput() {
    const raw = DOM.manualTextarea.value.trim();
    if (!raw) return null;
    const values = raw.split(/[\s,]+/).map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    updateFeatureCount(values.length);
    if (values.length === 60) return values;
    return null;
}

function updateFeatureCount(count) {
    DOM.featureCountDisplay.textContent = `${count} / 60 Features`;
    DOM.featureCountDisplay.style.color = count === 60 ? "var(--green-safe)" : count > 60 ? "var(--red-hot)" : "var(--cyan-glow)";
}

// ─── THRESHOLD CONTROL ─────────────────────────────────────────────────────────
function setThreshold(val) {
    State.threshold = parseFloat(val);
    DOM.thresholdSlider.value = State.threshold.toFixed(2);
    DOM.thresholdDisplayBadge.textContent = `τ = ${State.threshold.toFixed(2)}`;

    // Update tau preset button active states
    document.querySelectorAll(".btn-preset-tau").forEach(btn => {
        btn.classList.toggle("active", parseFloat(btn.dataset.tau) === State.threshold);
    });

    // If we have a prior prediction, re-evaluate with new threshold (no API call - client-side)
    if (State.lastPrediction) {
        const prob = State.lastPrediction.probability_mine;
        const fakeResult = { ...State.lastPrediction, applied_threshold: State.threshold };
        const isThreat = prob >= State.threshold;
        fakeResult.is_threat = isThreat;
        fakeResult.classification = isThreat ? "Mine (Threat Detected)" : "Rock (Benign Object)";

        if (isThreat) {
            fakeResult.confidence_percent = State.threshold < 1.0
                ? ((prob - State.threshold) / (1.0 - State.threshold)) * 100
                : 100.0;
        } else {
            fakeResult.confidence_percent = State.threshold > 0.0
                ? ((State.threshold - prob) / State.threshold) * 100
                : 100.0;
        }

        renderThreatPanel(fakeResult);
    }
}

// ─── MAIN CLASSIFY FLOW ────────────────────────────────────────────────────────
async function classifyCurrentSignal() {
    // Determine signal source based on active tab
    let features = State.features;
    if (State.activeTab === "manual") {
        features = parseManualInput();
        if (!features) {
            flashHint("⚠ Provide exactly 60 comma-separated float values (0.0 - 1.0).", "error");
            return;
        }
        State.features = features;
        renderSpectralCanvas();
    }

    if (!features) {
        flashHint("⚠ No signal loaded. Select a preset or enter 60 frequency values.", "error");
        return;
    }

    // Show loading state
    DOM.btnClassify.disabled = true;
    DOM.btnClassify.innerHTML = `<span class="spinner-inline"></span>&nbsp; ANALYZING SIGNAL...`;

    try {
        const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features, threshold: State.threshold }),
        });

        const json = await resp.json();

        if (json.status !== "success") {
            flashHint(`Server error: ${json.message}`, "error");
            return;
        }

        State.lastPrediction = json.data;
        renderThreatPanel(json.data);
        DOM.explainStatus.innerHTML = 'Signal classified. Click <strong>"Run SHAP Attribution"</strong> to compute frequency attributions.';
        DOM.attributionColumns.style.display = "none";

    } catch (e) {
        flashHint(`Network error: ${e.message}`, "error");
    } finally {
        DOM.btnClassify.disabled = false;
        DOM.btnClassify.innerHTML = `<span class="btn-icon">⚡</span><span class="btn-text">ANALYZE &amp; CLASSIFY SONAR PING</span>`;
    }
}

// ─── RENDER THREAT PANEL ───────────────────────────────────────────────────────
function renderThreatPanel(result) {
    const prob = result.probability_mine;
    const probRock = 1.0 - prob;
    const tau = result.applied_threshold;
    const isThreat = prob >= tau;
    const conf = result.confidence_percent;

    // Determine threat class based on probability level
    let bannerClass, icon, alertLabel, classifLabel, message, defconCode;

    if (prob >= 0.75) {
        bannerClass = "threat-critical";
        icon = "💣";
        alertLabel = "CRITICAL — EXPLOSIVE THREAT";
        classifLabel = "NAVAL MINE CONFIRMED";
        message = "High-confidence acoustic signature consistent with an explosive underwater mine. Immediate evasive protocol and counter-measure deployment required.";
        defconCode = "DEFCON-1";
    } else if (isThreat) {
        bannerClass = "threat-elevated";
        icon = "⚠️";
        alertLabel = "ELEVATED — THREAT DETECTED";
        classifLabel = "MINE (THREAT DETECTED)";
        message = `Acoustic return (${(prob * 100).toFixed(1)}%) exceeds the ${(tau * 100).toFixed(0)}% safe decision threshold. Flagged under zero-missed-mine naval protocol.`;
        defconCode = "DEFCON-2";
    } else if (prob >= 0.25) {
        bannerClass = "threat-low";
        icon = "🔶";
        alertLabel = "MONITORED — AMBIGUOUS SIGNATURE";
        classifLabel = "LOW THREAT / POTENTIAL ROCK";
        message = `Probability (${(prob * 100).toFixed(1)}%) is below the decision threshold. Likely a geological rock formation. Continue passive monitoring.`;
        defconCode = "DEFCON-3";
    } else {
        bannerClass = "threat-clear";
        icon = "✅";
        alertLabel = "CLEAR — NO THREAT DETECTED";
        classifLabel = "GEOLOGICAL ROCK (BENIGN)";
        message = `Low acoustic energy return (${(prob * 100).toFixed(1)}%). Echo pattern consistent with natural seafloor rock formation. Navigation corridor confirmed clear.`;
        defconCode = "DEFCON-4";
    }

    // Update banner
    DOM.threatBanner.className = `threat-banner ${bannerClass} animate-fadein`;
    DOM.threatStatusIcon.textContent = icon;
    DOM.threatAlertLabel.textContent = alertLabel;
    DOM.threatClassificationName.textContent = classifLabel;
    DOM.threatMessage.textContent = message;

    // Update DEFCON badge
    DOM.threatDefconBadge.textContent = defconCode;
    DOM.threatDefconBadge.style.color = prob >= 0.75 ? "var(--red-hot)" : isThreat ? "#ff8c00" : prob >= 0.25 ? "var(--gold-bright)" : "var(--green-safe)";
    DOM.threatDefconBadge.style.borderColor = DOM.threatDefconBadge.style.color;

    // Probability metrics
    const probPercent = (prob * 100).toFixed(1);
    const rockPercent = (probRock * 100).toFixed(1);
    DOM.metricMineProb.textContent = `${probPercent}%`;
    DOM.metricRockProb.textContent = `${rockPercent}%`;
    DOM.metricConfidence.textContent = `${conf.toFixed(1)}%`;
    DOM.metricLatency.textContent = result.latency_ms ? `${result.latency_ms.toFixed(2)} ms` : "—";

    // Progress bars (animated)
    requestAnimationFrame(() => {
        DOM.barMineProb.style.width = `${probPercent}%`;
        DOM.barRockProb.style.width = `${rockPercent}%`;
    });

    // Margin text
    const margin = prob - tau;
    const sign = margin >= 0 ? "+" : "";
    DOM.metricMarginText.textContent = `Margin: ${sign}${(margin * 100).toFixed(1)}% vs τ = ${(tau * 100).toFixed(0)}%`;
    DOM.metricMarginText.style.color = margin >= 0 ? "var(--red-hot)" : "var(--green-safe)";
}

// ─── SPECTRAL CANVAS RENDERER ─────────────────────────────────────────────────
function renderSpectralCanvas() {
    const canvas = DOM.spectralCanvas;
    const ctx = canvas.getContext("2d");
    const W = canvas.parentElement.offsetWidth || 800;
    const H = 260;
    const DPR = window.devicePixelRatio || 1;

    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.scale(DPR, DPR);

    // Background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, H);
    bgGrad.addColorStop(0, "rgba(10, 17, 40, 0.0)");
    bgGrad.addColorStop(1, "rgba(0, 0, 0, 0.15)");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, W, H);

    const padL = 45, padR = 15, padT = 18, padB = 32;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // Determine global max for Y scaling
    const allArrays = [];
    if (State.refProfiles) {
        allArrays.push(State.refProfiles.mean_mine, State.refProfiles.mean_rock);
    }
    if (State.features) allArrays.push(State.features);
    const allVals = allArrays.flat().filter(v => typeof v === "number");
    const maxVal = allVals.length ? Math.max(...allVals) * 1.12 : 0.8;

    function toX(i) { return padL + (i / 59) * plotW; }
    function toY(v) { return padT + plotH - (v / maxVal) * plotH; }

    // Grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.045)";
    ctx.lineWidth = 0.5;
    for (let g = 0; g <= 5; g++) {
        const y = padT + (g / 5) * plotH;
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    }

    // Frequency zone background shading
    const zones = [
        { start: 0, end: 14, color: "rgba(0, 240, 255, 0.025)", label: "Low" },
        { start: 15, end: 34, color: "rgba(255, 0, 85, 0.025)", label: "Mid" },
        { start: 35, end: 59, color: "rgba(0, 232, 154, 0.025)", label: "High" },
    ];
    zones.forEach(z => {
        const x0 = toX(z.start), x1 = toX(z.end);
        ctx.fillStyle = z.color;
        ctx.fillRect(x0, padT, x1 - x0, plotH);
    });

    // Y-Axis labels
    ctx.fillStyle = "rgba(90, 110, 138, 0.8)";
    ctx.font = `${10 * DPR / DPR}px JetBrains Mono, monospace`;
    ctx.textAlign = "right";
    for (let g = 0; g <= 5; g++) {
        const v = maxVal - (g / 5) * maxVal;
        const y = padT + (g / 5) * plotH;
        ctx.fillText(v.toFixed(2), padL - 5, y + 3.5);
    }

    // Helper: draw filled line series
    function drawSeries(values, strokeColor, fillColor, lineW = 1.5) {
        if (!values || values.length < 2) return;
        ctx.beginPath();
        values.forEach((v, i) => {
            const x = toX(i), y = toY(v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        // Clone path for stroke then fill
        ctx.save();
        ctx.lineWidth = lineW;
        ctx.strokeStyle = strokeColor;
        ctx.lineJoin = "round";
        ctx.stroke();
        ctx.restore();

        // Filled area to baseline
        ctx.beginPath();
        values.forEach((v, i) => {
            const x = toX(i), y = toY(v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.lineTo(toX(values.length - 1), padT + plotH);
        ctx.lineTo(toX(0), padT + plotH);
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
    }

    // 1. Mean Rock Profile
    if (State.showRockRef && State.refProfiles) {
        drawSeries(
            State.refProfiles.mean_rock,
            "rgba(0, 232, 154, 0.55)",
            "rgba(0, 232, 154, 0.06)"
        );
    }

    // 2. Mean Mine Profile
    if (State.showMineRef && State.refProfiles) {
        drawSeries(
            State.refProfiles.mean_mine,
            "rgba(255, 0, 85, 0.55)",
            "rgba(255, 0, 85, 0.06)"
        );
    }

    // 3. Active Signal (on top, thicker, bright cyan)
    if (State.showActive && State.features) {
        drawSeries(
            State.features,
            "rgba(0, 240, 255, 0.92)",
            "rgba(0, 240, 255, 0.07)",
            2.2
        );

        // Draw individual frequency dots for active signal
        ctx.fillStyle = "rgba(0, 240, 255, 0.55)";
        State.features.forEach((v, i) => {
            ctx.beginPath();
            ctx.arc(toX(i), toY(v), 2, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // X-Axis tick labels (every 10 bands)
    ctx.fillStyle = "rgba(90, 110, 138, 0.8)";
    ctx.textAlign = "center";
    for (let i = 0; i < 60; i += 10) {
        const x = toX(i);
        ctx.fillText(`F${String(i + 1).padStart(2, "0")}`, x, padT + plotH + 16);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH + 4); ctx.stroke();
    }
}

// ─── EXPLAINABILITY ────────────────────────────────────────────────────────────
async function runExplain() {
    if (!State.features) {
        flashHint("⚠ Load a signal first.", "error");
        return;
    }

    DOM.btnRunExplain.classList.add("loading");
    DOM.explainBtnIcon.textContent = "";
    DOM.explainBtnText.innerHTML = `<span class="spinner-inline"></span> Computing SHAP...`;
    DOM.explainStatus.textContent = "Running SHAP KernelExplainer (nsamples=40) — this may take ~2–3 seconds...";
    DOM.attributionColumns.style.display = "none";

    try {
        const resp = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: State.features }),
        });

        const json = await resp.json();

        if (json.status !== "success") {
            DOM.explainStatus.textContent = `Error: ${json.message}`;
            return;
        }

        renderAttributions(json);

    } catch (e) {
        DOM.explainStatus.textContent = `Network error: ${e.message}`;
    } finally {
        DOM.btnRunExplain.classList.remove("loading");
        DOM.explainBtnIcon.textContent = "⚡";
        DOM.explainBtnText.textContent = "Run SHAP Attribution";
    }
}

function renderAttributions(data) {
    const maxShap = Math.max(
        ...data.top_mine_drivers.map(d => Math.abs(d.shap_value)),
        ...data.top_rock_drivers.map(d => Math.abs(d.shap_value)),
        0.001
    );

    DOM.listMineDrv.innerHTML = data.top_mine_drivers.map(d => {
        const pct = (Math.abs(d.shap_value) / maxShap * 100).toFixed(1);
        return `
            <div class="attr-item animate-fadein">
                <span class="attr-freq-name text-danger">${d.feature}</span>
                <div class="attr-bar-track">
                    <div class="attr-bar-fill mine-bar" style="width:${pct}%;"></div>
                </div>
                <span class="attr-shap-val text-danger">+${d.shap_value.toFixed(4)}</span>
                <span class="attr-raw-val">${d.raw_value.toFixed(3)}</span>
            </div>`;
    }).join("");

    DOM.listRockDrv.innerHTML = data.top_rock_drivers.map(d => {
        const pct = (Math.abs(d.shap_value) / maxShap * 100).toFixed(1);
        return `
            <div class="attr-item animate-fadein">
                <span class="attr-freq-name text-success">${d.feature}</span>
                <div class="attr-bar-track">
                    <div class="attr-bar-fill rock-bar" style="width:${pct}%;"></div>
                </div>
                <span class="attr-shap-val text-success">${d.shap_value.toFixed(4)}</span>
                <span class="attr-raw-val">${d.raw_value.toFixed(3)}</span>
            </div>`;
    }).join("");

    DOM.explainStatus.style.display = "none";
    DOM.attributionColumns.style.display = "grid";
    
    const method = data.attribution_type || "SHAP";
    // Restore status as note
    DOM.explainStatus.textContent = `${method} • Latency: ${data.latency_ms.toFixed(0)} ms`;
    DOM.explainStatus.style.display = "block";
    DOM.explainStatus.style.padding = "0.3rem 0";
    DOM.explainStatus.style.background = "none";
    DOM.explainStatus.style.border = "none";
    DOM.explainStatus.style.fontSize = "0.68rem";
    DOM.explainStatus.style.color = "var(--text-muted)";
    DOM.explainStatus.style.textAlign = "left";
}

// ─── BATCH CSV FLOW ────────────────────────────────────────────────────────────
async function runBatchInference() {
    if (!State.batchFile) {
        flashHint("⚠ No CSV file selected.", "error");
        return;
    }

    DOM.btnRunBatch.disabled = true;
    DOM.btnRunBatch.innerHTML = `<span class="spinner-inline"></span> Running batch inference...`;

    const formData = new FormData();
    formData.append("file", State.batchFile);
    formData.append("threshold", State.threshold.toString());

    try {
        const resp = await fetch("/api/predict_batch", { method: "POST", body: formData });
        const json = await resp.json();

        if (json.status !== "success") {
            flashHint(`Batch error: ${json.message}`, "error");
            return;
        }

        renderBatchModal(json);
        openModal();

    } catch (e) {
        flashHint(`Network error: ${e.message}`, "error");
    } finally {
        DOM.btnRunBatch.disabled = false;
        DOM.btnRunBatch.innerHTML = `<span>Run Batch Inference on CSV</span>`;
    }
}

function renderBatchModal(data) {
    const summary = data.summary;
    const totalSignals = summary.total_signals;
    const threats = summary.threats_detected;
    const benign = summary.benign_detected;
    const threatRate = totalSignals > 0 ? ((threats / totalSignals) * 100).toFixed(1) : "0.0";

    DOM.batchSummaryCards.innerHTML = `
        <div class="batch-stat-card">
            <div class="batch-stat-label">Total Signals</div>
            <div class="batch-stat-value text-accent">${totalSignals}</div>
        </div>
        <div class="batch-stat-card">
            <div class="batch-stat-label">Threats Detected</div>
            <div class="batch-stat-value text-danger">${threats}</div>
        </div>
        <div class="batch-stat-card">
            <div class="batch-stat-label">Clear / Benign</div>
            <div class="batch-stat-value text-success">${benign}</div>
        </div>
        <div class="batch-stat-card">
            <div class="batch-stat-label">Threat Rate</div>
            <div class="batch-stat-value text-gold">${threatRate}%</div>
        </div>
    `;

    DOM.batchTableBody.innerHTML = data.preview.map(row => {
        const isThreat = row.is_threat;
        const cls = isThreat ? "tbl-threat-mine" : "tbl-threat-rock";
        const statusIcon = isThreat ? "⚠ MINE" : "✓ CLEAR";
        return `
            <tr>
                <td>${row.signal_index}</td>
                <td class="${cls}">${(row.probability_mine * 100).toFixed(2)}%</td>
                <td>${(row.probability_rock * 100).toFixed(2)}%</td>
                <td class="${cls}">${row.classification}</td>
                <td class="${cls}">${statusIcon}</td>
            </tr>`;
    }).join("");
}

function openModal() {
    DOM.batchModal.style.display = "flex";
    document.body.style.overflow = "hidden";
}

function closeModal() {
    DOM.batchModal.style.display = "none";
    document.body.style.overflow = "";
}

// ─── HINT FLASH ───────────────────────────────────────────────────────────────
let hintTimeout;
function flashHint(msg, type = "info") {
    clearTimeout(hintTimeout);
    const el = DOM.explainStatus;
    el.style.display = "block";
    el.style.padding = "0.6rem 1rem";
    el.style.background = type === "error" ? "rgba(255,0,85,0.08)" : "rgba(0,240,255,0.06)";
    el.style.border = type === "error" ? "1px solid rgba(255,0,85,0.25)" : "1px solid rgba(0,240,255,0.15)";
    el.style.color = type === "error" ? "var(--red-hot)" : "var(--cyan-glow)";
    el.style.fontSize = "0.78rem";
    el.style.textAlign = "left";
    el.textContent = msg;

    hintTimeout = setTimeout(() => {
        el.style.background = "none";
        el.style.border = "none";
        el.style.color = "var(--text-muted)";
    }, 3500);
}

// ─── EVENT LISTENERS ──────────────────────────────────────────────────────────
function attachEventListeners() {
    // Tab switching
    DOM.tabBtns.forEach(btn => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

    // Preset buttons
    DOM.btnConfidentMine.addEventListener("click", () => { loadPreset("confident_mine"); switchTab("presets"); });
    DOM.btnBorderlineMine.addEventListener("click", () => { loadPreset("borderline_mine"); switchTab("presets"); });
    DOM.btnConfidentRock.addEventListener("click", () => { loadPreset("confident_rock"); switchTab("presets"); });
    DOM.btnBorderlineRock.addEventListener("click", () => { loadPreset("borderline_rock"); switchTab("presets"); });
    DOM.btnRandomSignal.addEventListener("click", loadRandomSignal);

    // Manual input
    DOM.manualTextarea.addEventListener("input", () => {
        const values = DOM.manualTextarea.value.split(/[\s,]+/).map(v => parseFloat(v)).filter(v => !isNaN(v));
        updateFeatureCount(values.length);
    });

    DOM.btnFillSample.addEventListener("click", () => {
        if (State.curated?.confident_mine) {
            DOM.manualTextarea.value = State.curated.confident_mine.features.join(", ");
            updateFeatureCount(60);
        }
    });

    DOM.btnClearManual.addEventListener("click", () => {
        DOM.manualTextarea.value = "";
        updateFeatureCount(0);
    });

    // Batch file
    DOM.dropzone.addEventListener("click", () => DOM.batchFileInput.click());
    DOM.dropzone.addEventListener("dragover", e => { e.preventDefault(); DOM.dropzone.classList.add("drag-over"); });
    DOM.dropzone.addEventListener("dragleave", () => DOM.dropzone.classList.remove("drag-over"));
    DOM.dropzone.addEventListener("drop", e => {
        e.preventDefault();
        DOM.dropzone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) handleFileSelect(file);
    });
    DOM.batchFileInput.addEventListener("change", e => {
        if (e.target.files[0]) handleFileSelect(e.target.files[0]);
    });
    DOM.btnRunBatch.addEventListener("click", () => { runBatchInference(); switchTab("batch"); });

    // Threshold slider
    DOM.thresholdSlider.addEventListener("input", e => setThreshold(e.target.value));

    // Tau preset buttons
    document.querySelectorAll(".btn-preset-tau").forEach(btn => {
        btn.addEventListener("click", () => setThreshold(btn.dataset.tau));
    });

    // Classify
    DOM.btnClassify.addEventListener("click", classifyCurrentSignal);

    // Explainability
    DOM.btnRunExplain.addEventListener("click", runExplain);

    // Modal
    DOM.btnCloseModal.addEventListener("click", closeModal);
    DOM.btnModalDismiss.addEventListener("click", closeModal);
    DOM.modalBackdrop.addEventListener("click", closeModal);

    // Chart toggles
    DOM.toggleActive.addEventListener("click", () => {
        State.showActive = !State.showActive;
        DOM.toggleActive.classList.toggle("active", State.showActive);
        renderSpectralCanvas();
    });

    DOM.toggleMine.addEventListener("click", () => {
        State.showMineRef = !State.showMineRef;
        DOM.toggleMine.classList.toggle("active", State.showMineRef);
        renderSpectralCanvas();
    });

    DOM.toggleRock.addEventListener("click", () => {
        State.showRockRef = !State.showRockRef;
        DOM.toggleRock.classList.toggle("active", State.showRockRef);
        renderSpectralCanvas();
    });

    // Resize handler for responsive canvas
    let resizeTimer;
    window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(renderSpectralCanvas, 100);
    });
}

function handleFileSelect(file) {
    State.batchFile = file;
    DOM.dropzoneText.innerHTML = `<strong>${file.name}</strong> selected (${(file.size / 1024).toFixed(1)} KB)`;
    DOM.btnRunBatch.disabled = false;
    DOM.dropzone.style.borderColor = "var(--green-safe)";
    DOM.dropzone.style.boxShadow = "0 0 14px rgba(0, 232, 154, 0.15)";
}

// ─── BOOTSTRAP ────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", init);
