import * as THREE from 'three';

const pageProtocol = window.location.protocol;
const apiHost = window.location.hostname || 'localhost';
const apiPort = '8000';
const apiHttpBase = `${pageProtocol === 'file:' ? 'http:' : pageProtocol}//${apiHost}:${apiPort}`;
const apiWsProtocol = pageProtocol === 'https:' ? 'wss:' : 'ws:';
const apiWsUrl = `${apiWsProtocol}//${apiHost}:${apiPort}/ws`;
const resultsApiUrl = `${apiHttpBase}/results`;

// --- Scene Setup ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c10);

const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 1, 5000);
const cameraInitialDirection = new THREE.Vector3(1, 0.8, 1).normalize();
const CAMERA_CORE_QUANTILE = 0.95;
const CAMERA_RADIUS_MARGIN = 1.35;
const CAMERA_MIN_DISTANCE = 720;
const CAMERA_START_DISTANCE = 1500;
const CAMERA_ORBIT_HEIGHT = cameraInitialDirection.y;
const CAMERA_ORBIT_SPEED = 0.035;
const CAMERA_TARGET_LERP = 0.025;
const CAMERA_DISTANCE_LERP = 0.025;
const CAMERA_POSITION_LERP = 0.018;
const NODE_POSITION_LERP = 0.38;
camera.position.copy(cameraInitialDirection).multiplyScalar(CAMERA_START_DISTANCE);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

// --- Swarm Visualization ---
const agentGroup = new THREE.Group();
scene.add(agentGroup);

const agents = new Map(); // id -> {mesh, data}

const sphereGeom = new THREE.SphereGeometry(1, 16, 16);
const lineMaterial = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.22, vertexColors: true });
const DEFAULT_BOND_COLOR = new THREE.Color(0xb7bfd1);
const bondGeometry = new THREE.BufferGeometry();
const bondSegments = new THREE.LineSegments(bondGeometry, lineMaterial);
bondSegments.frustumCulled = false;
let bondPositionArray = new Float32Array(0);
let bondColorArray = new Float32Array(0);
scene.add(bondSegments);
bondSegments.visible = false;

const AGENT_COLOR_STOPS = [
    { t: 0.00, color: new THREE.Color(0x8951d6) },
    { t: 0.34, color: new THREE.Color(0xa66af0) },
    { t: 0.58, color: new THREE.Color(0xaeb0ef) },
    { t: 0.78, color: new THREE.Color(0x8bdac4) },
    { t: 1.00, color: new THREE.Color(0x79f2a3) },
];
const AGENT_COLOR_HIGH = AGENT_COLOR_STOPS[AGENT_COLOR_STOPS.length - 1].color;
const PHASE_IDLE = 0;
const DEFAULT_TARGET_A_PLACEHOLDER = "e.g. cause";
const DEFAULT_TARGET_B_PLACEHOLDER = "e.g. outcome";

const phaseDebugCopy = {
    1: {
        kicker: "PHASE 01 / RESEARCH",
        title: "SEARCHLIGHTS ONLINE",
        detail: "The engine is gathering source terrain and pulling candidate evidence into view.",
    },
    2: {
        kicker: "PHASE 02 / PHYSICS",
        title: "GRAVITY ROOM",
        detail: "The graph is settling into 3D space so nearby ideas can become reachable paths.",
    },
    3: {
        kicker: "PHASE 03 / STABILIZED",
        title: "FIELD LOCKED",
        detail: "The swarm has cooled enough for thought agents to begin moving through it.",
    },
    4: {
        kicker: "PHASE 04 / THOUGHT",
        title: "ARGUMENT TRAINS RUNNING",
        detail: "Thought agents are chasing cited facts, specific figures, and bridges between the targets.",
    },
    5: {
        kicker: "PHASE 05 / SYNTHESIS",
        title: "PROMPT FURNACE",
        detail: "The strongest trains are being compressed into a readable synthesis.",
    },
};

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function quantile(values, q) {
    if (!values.length) return 1;
    const sorted = [...values].sort((a, b) => a - b);
    const index = (sorted.length - 1) * clamp01(q);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return sorted[lower];
    const mix = index - lower;
    return sorted[lower] * (1 - mix) + sorted[upper] * mix;
}

function connectivityScale(degree, softMax) {
    const capped = Math.min(Math.max(0, degree), Math.max(1, softMax));
    const norm = Math.log1p(capped) / Math.log1p(Math.max(1, softMax));
    return clamp01(Math.pow(norm, 0.78));
}

function getAgentColor(scale) {
    const t = clamp01(scale);
    for (let i = 1; i < AGENT_COLOR_STOPS.length; i += 1) {
        const previous = AGENT_COLOR_STOPS[i - 1];
        const next = AGENT_COLOR_STOPS[i];
        if (t <= next.t) {
            const span = Math.max(0.0001, next.t - previous.t);
            return previous.color.clone().lerp(next.color, (t - previous.t) / span);
        }
    }
    return AGENT_COLOR_HIGH.clone();
}

function escapeHTML(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderInlineMarkdown(value) {
    return escapeHTML(value)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
        .replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
}

function renderMarkdown(text) {
    const lines = String(text || "").trim().split(/\r?\n/);
    const html = [];
    let paragraph = [];
    let listItems = [];

    const flushParagraph = () => {
        if (!paragraph.length) return;
        html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
        paragraph = [];
    };

    const flushList = () => {
        if (!listItems.length) return;
        html.push(`<ul>${listItems.map(item => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
        listItems = [];
    };

    lines.forEach(rawLine => {
        const line = rawLine.trim();
        if (!line) {
            flushParagraph();
            flushList();
            return;
        }

        const heading = line.match(/^(#{1,4})\s+(.+)$/);
        if (heading) {
            flushParagraph();
            flushList();
            const level = heading[1].length;
            html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
            return;
        }

        const sourceRef = line.match(/^\[(\d+)\]:\s*(.+)$/);
        if (sourceRef) {
            flushParagraph();
            flushList();
            html.push(`<p class="source-ref"><strong>[${sourceRef[1]}]:</strong> ${renderInlineMarkdown(sourceRef[2])}</p>`);
            return;
        }

        const bullet = line.match(/^[-*]\s+(.+)$/);
        if (bullet) {
            flushParagraph();
            listItems.push(bullet[1]);
            return;
        }

        flushList();
        paragraph.push(line);
    });

    flushParagraph();
    flushList();
    return html.join("");
}

function renderReport(text, placeholder = "Awaiting research grounding...") {
    const reportEl = document.getElementById('report-content');
    const reportContainer = document.getElementById('report-container');
    reportContainer.classList.remove('phase-debug-mode');
    reportContainer.classList.remove('chat-mode');
    reportEl.className = "";

    const cleanText = String(text || "").trim();
    if (!cleanText) {
        reportEl.innerHTML = `<p class="placeholder">${escapeHTML(placeholder)}</p>`;
        return;
    }

    reportEl.innerHTML = renderMarkdown(cleanText);
}

function renderPhaseDebug(data) {
    const reportEl = document.getElementById('report-content');
    const reportContainer = document.getElementById('report-container');
    const phase = Number(data?.phase ?? PHASE_IDLE);
    const copy = phaseDebugCopy[phase] || {
        kicker: `PHASE ${phase}`,
        title: String(data?.status || "ENGINE ACTIVE").toUpperCase(),
        detail: "The system is moving through an active runtime phase.",
    };

    reportContainer.classList.remove('chat-mode');
    reportContainer.classList.add('phase-debug-mode');
    reportEl.className = "phase-debug";
    reportEl.innerHTML = `
        <div class="debug-orb" aria-hidden="true">
            <span></span>
            <span></span>
        </div>
        <div class="debug-kicker">${escapeHTML(copy.kicker)}</div>
        <div class="debug-title">${escapeHTML(copy.title)}</div>
        <div class="debug-status">${escapeHTML(data?.status || "Working")}</div>
        <div class="debug-detail">${escapeHTML(copy.detail)}</div>
        <div class="debug-dots" aria-hidden="true"><span></span><span></span><span></span></div>
    `;
}

function promptLabel(targetA, targetB) {
    const left = capitalizeFirst(targetA || "target A");
    const right = capitalizeFirst(targetB || "target B");
    return `${left} -> ${right}`;
}

function capitalizeFirst(value) {
    const text = String(value || "").trim();
    return text ? `${text.charAt(0).toUpperCase()}${text.slice(1)}` : "";
}

function scrollChatToBottom() {
    const reportContainer = document.getElementById('report-container');
    requestAnimationFrame(() => {
        reportContainer.scrollTop = reportContainer.scrollHeight;
    });
}

function renderChat() {
    const reportEl = document.getElementById('report-content');
    const reportContainer = document.getElementById('report-container');
    reportContainer.classList.remove('phase-debug-mode');
    reportContainer.classList.add('chat-mode');
    reportEl.className = "chat-feed";

    const parts = [];
    if (launchBlockedUntilNewPrompt()) {
        parts.push(`
            <div class="chat-cap-warning">
                This chat is complete. Click New Prompt before launching another research run.
            </div>
        `);
    }

    if (!chatTurns.length) {
        parts.push('<p class="placeholder">Start a new research prompt below. Each run is isolated.</p>');
    }

    chatTurns.forEach((turn, index) => {
        parts.push(`
            <div class="chat-row chat-row-user">
                <div class="chat-bubble chat-bubble-user">
                    <div class="chat-label">Prompt</div>
                    <div>${escapeHTML(promptLabel(turn.targetA, turn.targetB))}</div>
                </div>
            </div>
        `);

        const isLatest = index === chatTurns.length - 1;
        if (turn.response) {
            parts.push(`
                <div class="chat-row chat-row-assistant">
                    <div class="chat-bubble chat-bubble-assistant">
                        <div class="chat-label">Answer</div>
                        ${renderMarkdown(turn.response)}
                    </div>
                </div>
            `);
        } else {
            parts.push(`
                <div class="chat-row chat-row-assistant">
                    <div class="chat-bubble chat-bubble-assistant chat-bubble-pending">
                        <div class="chat-label">Answer</div>
                        <span>Thinking through the graph</span><span class="chat-thinking-dots">...</span>
                    </div>
                </div>
            `);
        }

        if (isLatest && turn.phaseLogs.length) {
            parts.push(`
                <div class="chat-debug-log">
                    <div class="chat-debug-title">Phase Debug</div>
                    ${turn.phaseLogs.slice(-8).map(log => `
                        <div class="chat-debug-line">
                            <span>${escapeHTML(log.phase)}</span>
                            ${escapeHTML(log.status)}
                        </div>
                    `).join("")}
                </div>
            `);
        }
    });

    reportEl.innerHTML = parts.join("");
    scrollChatToBottom();
}

function appendChatPrompt(targetA, targetB) {
    const turn = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        targetA,
        targetB,
        response: "",
        phaseLogs: [],
    };
    activeChatTurn = turn;
    chatTurns.push(turn);
    renderChat();
    return turn;
}

function addChatDebugLog(data) {
    if (!activeChatTurn) return;
    const phase = Number(data?.phase ?? PHASE_IDLE);
    const copy = phaseDebugCopy[phase] || { kicker: `PHASE ${phase}` };
    const status = String(data?.status || "Working");
    const signature = `${copy.kicker}:${status}`;
    const previous = activeChatTurn.phaseLogs[activeChatTurn.phaseLogs.length - 1];
    if (previous && previous.signature === signature) return;
    activeChatTurn.phaseLogs.push({
        phase: copy.kicker,
        status,
        signature,
    });
    renderChat();
}

function finalizeActiveChat(report) {
    if (!activeChatTurn) return;
    activeChatTurn.response = report || "No synthesis text was returned.";
    activeChatTurn = null;
    renderChat();
    updateLaunchControls();
}

function showChatError(message) {
    ignoredReportVersion = null;
    if (activeChatTurn && !activeChatTurn.response) {
        activeChatTurn.response = `**Error:** ${message}`;
        activeChatTurn = null;
    } else {
        chatTurns.push({
            id: `${Date.now()}-error`,
            targetA: "system",
            targetB: "error",
            response: `**Error:** ${message}`,
            phaseLogs: [],
        });
    }
    renderChat();
    updateLaunchControls();
}

function ensureBondCapacity(vertexCount) {
    const valueCount = vertexCount * 3;
    if (bondPositionArray.length >= valueCount) return;

    bondPositionArray = new Float32Array(valueCount);
    bondColorArray = new Float32Array(valueCount);

    const positionAttr = new THREE.BufferAttribute(bondPositionArray, 3);
    const colorAttr = new THREE.BufferAttribute(bondColorArray, 3);
    positionAttr.setUsage(THREE.DynamicDrawUsage);
    colorAttr.setUsage(THREE.DynamicDrawUsage);
    bondGeometry.setAttribute('position', positionAttr);
    bondGeometry.setAttribute('color', colorAttr);
}

function updateBonds(bonds) {
    const drawable = [];
    for (const bond of bonds || []) {
        const source = agents.get(Number(bond.s));
        const target = agents.get(Number(bond.d));
        if (!source || !target || source === target) continue;
        const utility = clamp01(Number(bond.utility) || 0);
        drawable.push({ source, target, utility });
    }

    drawable.sort((a, b) => b.utility - a.utility);
    const vertexCount = drawable.length * 2;
    if (!vertexCount) {
        bondSegments.visible = false;
        bondGeometry.setDrawRange(0, 0);
        return;
    }

    ensureBondCapacity(vertexCount);
    const color = new THREE.Color();
    for (let i = 0; i < drawable.length; i += 1) {
        const bond = drawable[i];
        const positionOffset = i * 6;
        const sourcePos = bond.source.targetPosition;
        const targetPos = bond.target.targetPosition;

        bondPositionArray[positionOffset] = sourcePos.x;
        bondPositionArray[positionOffset + 1] = sourcePos.y;
        bondPositionArray[positionOffset + 2] = sourcePos.z;
        bondPositionArray[positionOffset + 3] = targetPos.x;
        bondPositionArray[positionOffset + 4] = targetPos.y;
        bondPositionArray[positionOffset + 5] = targetPos.z;

        color.copy(DEFAULT_BOND_COLOR).lerp(AGENT_COLOR_HIGH, bond.utility);
        const colorOffset = i * 6;
        bondColorArray[colorOffset] = color.r;
        bondColorArray[colorOffset + 1] = color.g;
        bondColorArray[colorOffset + 2] = color.b;
        bondColorArray[colorOffset + 3] = color.r;
        bondColorArray[colorOffset + 4] = color.g;
        bondColorArray[colorOffset + 5] = color.b;
    }

    bondGeometry.attributes.position.needsUpdate = true;
    bondGeometry.attributes.color.needsUpdate = true;
    bondGeometry.setDrawRange(0, vertexCount);
    bondSegments.visible = true;
}

// --- WebSocket Connection ---
const socket = new WebSocket(apiWsUrl);

socket.onopen = () => {
    console.log(`[WEB] Connected to ${apiWsUrl}`);
};

socket.onerror = (event) => {
    console.error('[WEB] WebSocket connection failed.', event);
    renderReport("", "Connection error: dashboard could not reach the engine on port 8000.");
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateSwarm(data);
    updateUI(data);
};

// --- Camera Follow State ---
let cameraTarget = new THREE.Vector3(0, 0, 0);
let cameraDistanceTarget = CAMERA_MIN_DISTANCE;
let cameraDistanceCurrent = CAMERA_START_DISTANCE;
let cameraOrbitAngle = Math.atan2(cameraInitialDirection.z, cameraInitialDirection.x);
const animationClock = new THREE.Clock();

function cameraDistanceForRadius(radius) {
    const verticalFov = THREE.MathUtils.degToRad(camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
    const limitingFov = Math.min(verticalFov, horizontalFov);
    const safeRadius = Math.max(1, Number(radius) || 1) * CAMERA_RADIUS_MARGIN;
    const distance = Math.max(CAMERA_MIN_DISTANCE, safeRadius / Math.sin(limitingFov / 2));
    if (camera.far < distance * 2.5) {
        camera.far = distance * 3;
        camera.updateProjectionMatrix();
    }
    return distance;
}

function boundingSphereForPositions(positions) {
    if (!positions.length) return null;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    positions.forEach(([x, y, z]) => {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        minZ = Math.min(minZ, z);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        maxZ = Math.max(maxZ, z);
    });

    const center = new THREE.Vector3(
        (minX + maxX) * 0.5,
        (minY + maxY) * 0.5,
        (minZ + maxZ) * 0.5
    );

    let radius = 1;
    positions.forEach(([x, y, z]) => {
        radius = Math.max(
            radius,
            Math.hypot(x - center.x, y - center.y, z - center.z)
        );
    });
    return { center, radius };
}

function cameraSphereForPositions(positions) {
    const fullSphere = boundingSphereForPositions(positions);
    if (!fullSphere) return null;

    const distances = positions.map(([x, y, z]) => (
        Math.hypot(x - fullSphere.center.x, y - fullSphere.center.y, z - fullSphere.center.z)
    ));
    const coreCutoff = quantile(distances, CAMERA_CORE_QUANTILE);
    const corePositions = positions.filter((_position, index) => distances[index] <= coreCutoff);
    return boundingSphereForPositions(corePositions.length ? corePositions : positions);
}

function cameraOrbitDirection(deltaSeconds) {
    cameraOrbitAngle += deltaSeconds * CAMERA_ORBIT_SPEED;
    const vertical = clamp01(CAMERA_ORBIT_HEIGHT);
    const horizontal = Math.sqrt(Math.max(0.01, 1 - (vertical * vertical)));
    return new THREE.Vector3(
        Math.cos(cameraOrbitAngle) * horizontal,
        vertical,
        Math.sin(cameraOrbitAngle) * horizontal
    ).normalize();
}

function updateSwarm(data) {
    const bonds = data.connections || data.bonds || [];
    const activeIds = new Set();
    const degreeValues = data.agents.map(a => Number(a.degree ?? a.c ?? 0)).filter(v => Number.isFinite(v) && v > 0);
    const softDegreeMax = Math.max(8, quantile(degreeValues, 0.88));
    
    // Update Agents
    const nodePositions = [];
    data.agents.forEach(a => {
        activeIds.add(a.id);
        let entry = agents.get(a.id);
        
        const degree = Number(a.degree ?? a.c ?? 0);
        const t = connectivityScale(degree, softDegreeMax);

        if (!entry) {
            const material = new THREE.MeshBasicMaterial({ color: getAgentColor(t), transparent: true, opacity: 0.92 });
            const mesh = new THREE.Mesh(sphereGeom, material);
            agentGroup.add(mesh);
            entry = { mesh, material, targetPosition: new THREE.Vector3() };
            agents.set(a.id, entry);
        }

        const x = Number(a.x) || 0;
        const y = Number(a.y) || 0;
        const z = Number(a.z) || 0;

        entry.targetPosition.set(x, y, z);
        entry.material.color.copy(getAgentColor(t));
        entry.material.opacity = 0.72 + (t * 0.24);
        entry.mesh.renderOrder = 0;

        const scale = 1.15 + Math.pow(t, 0.8) * 8.5;
        entry.mesh.scale.set(scale, scale, scale);

        nodePositions.push([x, y, z]);
    });
    
    // Remove dead agents
    for (const [id, entry] of agents) {
        if (!activeIds.has(id)) {
            agentGroup.remove(entry.mesh);
            agents.delete(id);
        }
    }

    updateBonds(bonds);

    // Focus the camera on the geometric center of the 95% core sphere, not the average position.
    const cameraSphere = cameraSphereForPositions(nodePositions);
    if (cameraSphere) {
        cameraTarget.copy(cameraSphere.center);
        cameraDistanceTarget = cameraDistanceForRadius(cameraSphere.radius);
    }
}

let lastReportVersion = -1;
let latestLiveReport = "";
let selectedResultId = null;
let resultEntries = [];
let launchInFlight = false;
let lastPhaseDebugSignature = "";
let chatTurns = [];
let activeChatTurn = null;
let ignoredReportVersion = null;
let latestEngineState = { phase: PHASE_IDLE, status: "System Online" };

function launchBlockedUntilNewPrompt() {
    if (selectedResultId) return true;
    return Boolean(!activeChatTurn && chatTurns.some(turn => turn.response));
}

function engineIsReady(data = latestEngineState) {
    return Number(data?.phase ?? PHASE_IDLE) === PHASE_IDLE || data?.status === "System Online";
}

function updateLaunchControls(data = latestEngineState) {
    const targetAInput = document.getElementById('input-target-a');
    const targetBInput = document.getElementById('input-target-b');
    const launchBtn = document.getElementById('launch-btn');
    if (!targetAInput || !targetBInput || !launchBtn) return;

    const ready = engineIsReady(data);
    const blocked = launchBlockedUntilNewPrompt();

    if (!ready) {
        const status = data?.status || "Engine busy";
        targetAInput.placeholder = `[ ${status} ]`;
        targetBInput.placeholder = `[ ${status} ]`;
        launchBtn.innerText = "ENGINE BUSY...";
    } else if (blocked) {
        targetAInput.placeholder = "click New Prompt";
        targetBInput.placeholder = "click New Prompt";
        launchBtn.innerText = "NEW PROMPT REQUIRED";
    } else {
        targetAInput.placeholder = targetAInput.value ? "edit target A" : DEFAULT_TARGET_A_PLACEHOLDER;
        targetBInput.placeholder = targetBInput.value ? "edit target B" : DEFAULT_TARGET_B_PLACEHOLDER;
        launchBtn.innerText = "LAUNCH RESEARCH";
    }

    launchBtn.disabled = launchInFlight || !ready || blocked;
}

function resetForNewPrompt() {
    selectedResultId = null;
    chatTurns = [];
    activeChatTurn = null;
    latestLiveReport = "";
    if (lastReportVersion >= 0) {
        ignoredReportVersion = lastReportVersion;
    }
    lastPhaseDebugSignature = "";

    const targetAInput = document.getElementById('input-target-a');
    const targetBInput = document.getElementById('input-target-b');
    if (targetAInput) targetAInput.value = "";
    if (targetBInput) targetBInput.value = "";

    renderHistoryList();
    renderChat();
    updateLaunchControls();
    targetAInput?.focus();
}

function updateUI(data) {
    latestEngineState = {
        phase: Number(data?.phase ?? PHASE_IDLE),
        status: data?.status || "",
    };
    if (data.status) {
        updateLaunchControls(data);
    }

    const phase = Number(data.phase ?? PHASE_IDLE);
    const busy = phase !== PHASE_IDLE && data.status !== "System Online";
    const reportChanged = data.report_version !== lastReportVersion;

    if (reportChanged) {
        if (
            data.report
            && ignoredReportVersion !== null
            && data.report_version === ignoredReportVersion
        ) {
            lastReportVersion = data.report_version;
        } else {
            latestLiveReport = data.report || "";
            lastReportVersion = data.report_version;
            if (data.report) {
                ignoredReportVersion = null;
                setTimeout(loadResultsHistory, 1500);
            }
        }
    }

    if (selectedResultId) return;

    if (latestLiveReport) {
        if (activeChatTurn) {
            finalizeActiveChat(latestLiveReport);
            lastPhaseDebugSignature = "";
        } else if (!chatTurns.length && reportChanged) {
            chatTurns.push({
                id: `${Date.now()}-report`,
                targetA: "saved",
                targetB: "response",
                response: latestLiveReport,
                phaseLogs: [],
            });
            renderChat();
            lastPhaseDebugSignature = "";
        }
        return;
    }

    if (busy) {
        const signature = `${phase}:${data.status || ""}`;
        if (signature !== lastPhaseDebugSignature) {
            if (activeChatTurn) {
                addChatDebugLog(data);
            } else {
                renderPhaseDebug(data);
            }
            lastPhaseDebugSignature = signature;
        }
        return;
    }

    if (reportChanged) {
        renderChat();
        lastPhaseDebugSignature = "";
    }

}

function formatResultDate(value) {
    const text = String(value || "").trim();
    const match = text.match(/^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})(?:-\d{2})?/);
    if (!match) return text;

    const [, year, month, day, hour, minute] = match;
    return `${Number(hour)}:${minute} - ${Number(month)}/${Number(day)}/${year}`;
}

function setHistorySelection(resultId) {
    selectedResultId = resultId;
    renderHistoryList();
    updateLaunchControls();
}

function renderHistoryList() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;

    historyList.innerHTML = "";

    const newQueryBtn = document.createElement('button');
    newQueryBtn.className = `history-entry history-new-query${selectedResultId ? "" : " active"}`;
    newQueryBtn.type = "button";
    newQueryBtn.textContent = "New Prompt";
    newQueryBtn.onclick = resetForNewPrompt;
    historyList.appendChild(newQueryBtn);

    if (!resultEntries.length) {
        const empty = document.createElement('div');
        empty.className = "history-empty";
        empty.textContent = "No saved research results yet.";
        historyList.appendChild(empty);
        return;
    }

    resultEntries.forEach(result => {
        const button = document.createElement('button');
        button.className = `history-entry${selectedResultId === result.id ? " active" : ""}`;
        button.type = "button";
        button.onclick = () => loadHistoricalResult(result.id);

        const title = document.createElement('span');
        title.textContent = capitalizeFirst(result.title || result.id);
        button.appendChild(title);

        const date = document.createElement('span');
        date.className = "history-date";
        date.textContent = formatResultDate(result.created || result.id);
        button.appendChild(date);

        historyList.appendChild(button);
    });
}

async function loadResultsHistory() {
    try {
        const response = await fetch(resultsApiUrl, { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        resultEntries = Array.isArray(payload.results) ? payload.results : [];
        renderHistoryList();
    } catch (error) {
        console.warn("[WEB] Could not load result history.", error);
    }
}

async function loadHistoricalResult(resultId) {
    if (!resultId) return;
    setHistorySelection(resultId);
    chatTurns = [{
        id: `${resultId}-loading`,
        targetA: "saved",
        targetB: "result",
        response: "Loading saved result...",
        phaseLogs: [],
    }];
    activeChatTurn = null;
    renderChat();
    updateLaunchControls();

    try {
        const response = await fetch(`${resultsApiUrl}/${encodeURIComponent(resultId)}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        const title = result.title || result.id || "Saved result";
        chatTurns = [{
            id: result.id || resultId,
            targetA: result.target_a || "saved",
            targetB: result.target_b || "result",
            response: `## ${title}\n\n${result.synthesis || "No synthesis text was saved for this result."}`,
            phaseLogs: [],
        }];
        renderChat();
        updateLaunchControls();
    } catch (error) {
        console.error("[WEB] Could not load saved result.", error);
        showChatError("Could not load that saved result.");
    }
}

// --- Interaction ---
document.getElementById('launch-btn').onclick = async () => {
    if (launchInFlight) return;
    if (launchBlockedUntilNewPrompt()) {
        renderChat();
        return;
    }
    const targetA = document.getElementById('input-target-a').value.trim();
    const targetB = document.getElementById('input-target-b').value.trim();
    if (!targetA || !targetB) {
        showChatError("Enter both targets before launching research.");
        return;
    }

    setHistorySelection(null);
    const launchBtn = document.getElementById('launch-btn');
    launchInFlight = true;
    launchBtn.disabled = true;
    latestLiveReport = "";
    ignoredReportVersion = lastReportVersion;
    lastPhaseDebugSignature = "";
    appendChatPrompt(targetA, targetB);

    try {
        const response = await fetch(`${apiHttpBase}/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_a: targetA,
                target_b: targetB,
            })
        });

        if (!response.ok) {
            let detail = `HTTP ${response.status}`;
            try {
                const payload = await response.json();
                detail = payload.detail || detail;
            } catch (_) {
                // Ignore JSON parsing issues and keep the status fallback.
            }

            showChatError(`Launch failed: ${detail}`);
            console.error('[WEB] Launch failed:', detail);
            return;
        }
    } catch (error) {
        showChatError("Launch failed: could not reach the command endpoint on port 8000.");
        console.error('[WEB] Launch request failed.', error);
        return;
    } finally {
        launchInFlight = false;
        updateLaunchControls();
    }
};

renderChat();
updateLaunchControls();
loadResultsHistory();
setInterval(loadResultsHistory, 15000);

document.getElementById('copy-btn').onclick = () => {
    const text = document.getElementById('report-content').innerText;
    navigator.clipboard.writeText(text);
};

// --- Resize Handling ---
window.onresize = () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
};

// --- Animation Loop ---
const lookAtPoint = new THREE.Vector3(0, 0, 0);

function animate() {
    requestAnimationFrame(animate);
    const deltaSeconds = Math.min(animationClock.getDelta(), 1 / 30);

    for (const entry of agents.values()) {
        entry.mesh.position.lerp(entry.targetPosition, NODE_POSITION_LERP);
    }
    
    // Smoothly interpolate the camera's focus point toward the true sphere center.
    lookAtPoint.lerp(cameraTarget, CAMERA_TARGET_LERP);
    cameraDistanceCurrent = THREE.MathUtils.lerp(
        cameraDistanceCurrent,
        cameraDistanceTarget,
        CAMERA_DISTANCE_LERP
    );

    // Orbit around that same center while expanding/contracting to contain the 95% core.
    const orbitDirection = cameraOrbitDirection(deltaSeconds);
    const desiredPos = new THREE.Vector3()
        .copy(lookAtPoint)
        .addScaledVector(orbitDirection, cameraDistanceCurrent);
    camera.position.lerp(desiredPos, CAMERA_POSITION_LERP);
    camera.lookAt(lookAtPoint);

    renderer.render(scene, camera);
}
animate();
