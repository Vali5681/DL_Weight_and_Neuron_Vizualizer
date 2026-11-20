(() => {
  const networkCanvas = document.getElementById("networkCanvas");
  const boundaryCanvas = document.getElementById("boundaryCanvas");
  const lossCanvas = document.getElementById("lossCanvas");

  const btnPlayPause = document.getElementById("btnPlayPause");
  const btnStep1 = document.getElementById("btnStep1");
  const btnStep10 = document.getElementById("btnStep10");
  const btnReset = document.getElementById("btnReset");

  const metricEpoch = document.getElementById("metricEpoch");
  const metricLoss = document.getElementById("metricLoss");
  const metricAcc = document.getElementById("metricAcc");

  const selDataset = document.getElementById("selDataset");
  const selArch = document.getElementById("selArch");
  const selActivation = document.getElementById("selActivation");
  const selOptimizer = document.getElementById("selOptimizer");
  const rngLr = document.getElementById("rngLr");
  const lblLr = document.getElementById("lblLr");
  const selSpeed = document.getElementById("selSpeed");
  const chkLabels = document.getElementById("chkLabels");
  const chkDelta = document.getElementById("chkDelta");

  const inspectorBody = document.getElementById("inspectorBody");
  const updatesTbody = document.getElementById("updatesTbody");

  let appState = null;
  let isTraining = false;
  let stepInFlight = false;
  let trainTimer = null;

  // Layout geometry for hit-testing
  let renderedNodes = []; // { layerIdx, neuronIdx, x, y, r }
  let renderedEdges = []; // { layerIdx, fromIdx, toIdx, x1, y1, x2, y2 }
  let selection = null;   // { type: 'edge', layerIdx, fromIdx, toIdx } | { type: 'node', layerIdx, neuronIdx }

  const SPEED_MAP = {
    slow: { epochs: 1, intervalMs: 450 },
    medium: { epochs: 2, intervalMs: 120 },
    fast: { epochs: 5, intervalMs: 50 },
  };

  async function apiGet(url) {
    const res = await fetch(url);
    return res.json();
  }

  async function apiPost(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return res.json();
  }

  function formatNum(v, digits = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return (v >= 0 ? "+" : "") + Number(v).toFixed(digits);
  }

  function resizeNetworkCanvas() {
    const rect = networkCanvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    networkCanvas.width = Math.max(300, Math.floor(rect.width * dpr));
    networkCanvas.height = Math.max(380, Math.floor(rect.height * dpr));
    const ctx = networkCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function renderAll() {
    if (!appState) return;
    renderMetrics();
    renderNetwork();
    renderBoundary();
    renderLossChart();
    renderInspector();
    renderTopUpdates();
  }

  function renderMetrics() {
    const m = appState.model;
    metricEpoch.textContent = String(m.epoch);
    metricLoss.textContent = m.loss !== null ? m.loss.toFixed(4) : "—";
    metricAcc.textContent = m.accuracy !== null ? `${(m.accuracy * 100).toFixed(1)}%` : "—";
  }

  function renderNetwork() {
    resizeNetworkCanvas();
    const ctx = networkCanvas.getContext("2d");
    const rect = networkCanvas.parentElement.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);

    const model = appState.model;
    const sizes = model.layer_sizes;
    const numLayers = sizes.length;
    const padX = 76;
    const padY = 56;
    const usableW = Math.max(120, W - padX * 2);
    const usableH = Math.max(180, H - padY * 2);

    const nodePos = [];
    renderedNodes = [];
    renderedEdges = [];

    for (let l = 0; l < numLayers; l++) {
      const count = sizes[l];
      const x = numLayers === 1 ? W / 2 : padX + (l / (numLayers - 1)) * usableW;
      const layerNodes = [];
      for (let i = 0; i < count; i++) {
        const y = count === 1 ? H / 2 : padY + ((i + 0.5) / count) * usableH;
        const r = Math.max(14, Math.min(24, usableH / (count * 3.1)));
        const nodeObj = { layerIdx: l, neuronIdx: i, x, y, r };
        layerNodes.push(nodeObj);
        renderedNodes.push(nodeObj);
      }
      nodePos.push(layerNodes);

      // Layer title
      ctx.fillStyle = "#94a3b8";
      ctx.font = "600 11px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      let title = `Hidden ${l} (${count})`;
      if (l === 0) title = `Input (${count})`;
      else if (l === numLayers - 1) title = `Output (${count})`;
      ctx.fillText(title, x, 24);
    }

    // Find max |w| and max |delta| for scaling
    let maxW = 0.5;
    let maxDelta = 1e-5;
    for (const layer of model.layers) {
      for (let j = 0; j < layer.out_features; j++) {
        for (let i = 0; i < layer.in_features; i++) {
          maxW = Math.max(maxW, Math.abs(layer.weights[j][i]));
          maxDelta = Math.max(maxDelta, Math.abs(layer.last_delta_w[j][i]));
        }
      }
    }

    const showLabels = chkLabels.checked;
    const showDelta = chkDelta.checked;

    // Draw edges
    for (let l = 0; l < model.layers.length; l++) {
      const layer = model.layers[l];
      const fromNodes = nodePos[l];
      const toNodes = nodePos[l + 1];

      for (let j = 0; j < layer.out_features; j++) {
        for (let i = 0; i < layer.in_features; i++) {
          const p1 = fromNodes[i];
          const p2 = toNodes[j];
          const w = layer.weights[j][i];
          const dw = layer.last_delta_w[j][i];
          const normW = Math.min(1, Math.abs(w) / maxW);
          const normDelta = Math.min(1, Math.abs(dw) / maxDelta);

          const isSelected =
            selection &&
            selection.type === "edge" &&
            selection.layerIdx === l &&
            selection.fromIdx === i &&
            selection.toIdx === j;

          renderedEdges.push({
            layerIdx: l,
            fromIdx: i,
            toIdx: j,
            x1: p1.x,
            y1: p1.y,
            x2: p2.x,
            y2: p2.y,
          });

          // Optional Δw update glow
          if (showDelta && Math.abs(dw) > 1e-6) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(251, 191, 36, ${0.15 + normDelta * 0.65})`;
            ctx.lineWidth = 3 + normDelta * 8;
            ctx.stroke();
          }

          // Main weight line
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          const baseColor = w >= 0 ? "56, 189, 248" : "251, 113, 133";
          const alpha = isSelected ? 1.0 : 0.28 + normW * 0.72;
          ctx.strokeStyle = `rgba(${baseColor}, ${alpha})`;
          ctx.lineWidth = isSelected ? 4.5 : 1.0 + normW * 4.5;
          ctx.stroke();

          if (isSelected) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }

          // Weight value label
          if (showLabels) {
            const t = 0.34 + ((i + j) % 3) * 0.14;
            const lx = p1.x + (p2.x - p1.x) * t;
            const ly = p1.y + (p2.y - p1.y) * t;
            const label = w.toFixed(2);
            ctx.font = "600 10px ui-monospace, SFMono-Regular, monospace";
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = "rgba(11, 17, 32, 0.86)";
            ctx.fillRect(lx - tw / 2 - 3, ly - 7, tw + 6, 13);
            ctx.fillStyle = w >= 0 ? "#bae6fd" : "#fecdd3";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(label, lx, ly);
          }
        }
      }
    }

    // Draw neurons
    for (let l = 0; l < numLayers; l++) {
      const layerData = l > 0 ? model.layers[l - 1] : null;
      for (let i = 0; i < sizes[l]; i++) {
        const node = nodePos[l][i];
        const isSelected =
          selection &&
          selection.type === "node" &&
          selection.layerIdx === l &&
          selection.neuronIdx === i;

        let fill = "#1e293b";
        let border = "#64748b";
        let innerLabel = l === 0 ? (i === 0 ? "x₁" : "x₂") : "";

        if (layerData) {
          const act = layerData.mean_activations[i] ?? 0;
          const bias = layerData.biases[i] ?? 0;
          const intensity = Math.min(1, Math.abs(act));
          if (act >= 0) {
            fill = `rgba(56, 189, 248, ${0.18 + intensity * 0.55})`;
          } else {
            fill = `rgba(251, 113, 133, ${0.18 + intensity * 0.55})`;
          }
          border = bias >= 0 ? "#38bdf8" : "#fb7185";
          innerLabel = `b:${bias.toFixed(2)}`;
        }

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = isSelected ? 3.5 : 2;
        ctx.strokeStyle = isSelected ? "#fbbf24" : border;
        ctx.stroke();

        ctx.fillStyle = "#f8fafc";
        ctx.font = "600 10px ui-monospace, SFMono-Regular, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(innerLabel, node.x, node.y);
      }
    }
  }

  function renderBoundary() {
    const ctx = boundaryCanvas.getContext("2d");
    const W = boundaryCanvas.width;
    const H = boundaryCanvas.height;
    const grid = appState.grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const cellW = W / cols;
    const cellH = H / rows;

    ctx.clearRect(0, 0, W, H);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const p = grid[r][c];
        // Blend coral (0) -> dark neutral (0.5) -> cyan (1)
        let red, green, blue;
        if (p >= 0.5) {
          const t = (p - 0.5) * 2;
          red = Math.round(17 + t * (20 - 17));
          green = Math.round(24 + t * (145 - 24));
          blue = Math.round(39 + t * (215 - 39));
        } else {
          const t = (0.5 - p) * 2;
          red = Math.round(17 + t * (200 - 17));
          green = Math.round(24 + t * (65 - 24));
          blue = Math.round(39 + t * (95 - 39));
        }
        ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    const [lo, hi] = appState.grid_bounds;
    const span = hi - lo;
    for (const pt of appState.dataset) {
      const px = ((pt.x - lo) / span) * W;
      const py = ((hi - pt.y) / span) * H;
      ctx.beginPath();
      ctx.arc(px, py, 3.6, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 1 ? "#38bdf8" : "#fb7185";
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#0b1120";
      ctx.stroke();
    }
  }

  function renderLossChart() {
    const ctx = lossCanvas.getContext("2d");
    const W = lossCanvas.width;
    const H = lossCanvas.height;
    ctx.clearRect(0, 0, W, H);

    const losses = appState.model.loss_history || [];
    const accs = appState.model.acc_history || [];
    if (losses.length < 2) {
      ctx.fillStyle = "#64748b";
      ctx.font = "12px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Train to see loss & accuracy curve", W / 2, H / 2);
      return;
    }

    const pad = 14;
    const maxLoss = Math.max(0.8, ...losses);

    // Loss curve
    ctx.beginPath();
    losses.forEach((v, idx) => {
      const x = pad + (idx / (losses.length - 1)) * (W - pad * 2);
      const y = H - pad - (v / maxLoss) * (H - pad * 2);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#fb7185";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Accuracy curve
    ctx.beginPath();
    accs.forEach((v, idx) => {
      const x = pad + (idx / (accs.length - 1)) * (W - pad * 2);
      const y = H - pad - v * (H - pad * 2);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.font = "600 10px -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = "#fb7185";
    ctx.fillText("Loss", pad, 14);
    ctx.fillStyle = "#34d399";
    ctx.fillText("Accuracy", pad + 36, 14);
  }

  function renderInspector() {
    if (!selection) {
      inspectorBody.innerHTML = `<p class="muted">Click any neuron or weight edge in the graph to inspect its live values.</p>`;
      return;
    }

    const model = appState.model;
    if (selection.type === "edge") {
      const layer = model.layers[selection.layerIdx];
      if (!layer || selection.toIdx >= layer.out_features || selection.fromIdx >= layer.in_features) {
        selection = null;
        return renderInspector();
      }
      const w = layer.weights[selection.toIdx][selection.fromIdx];
      const gw = layer.grad_w[selection.toIdx][selection.fromIdx];
      const dw = layer.last_delta_w[selection.toIdx][selection.fromIdx];
      inspectorBody.innerHTML = `
        <div class="kv-grid">
          <span class="kv-key">Weight Edge</span>
          <span class="kv-val">L${selection.layerIdx}[${selection.fromIdx}] → L${selection.layerIdx + 1}[${selection.toIdx}]</span>
          <span class="kv-key">Weight (w)</span>
          <span class="kv-val ${w >= 0 ? "pos-txt" : "neg-txt"}">${formatNum(w, 5)}</span>
          <span class="kv-key">Gradient (∂L/∂w)</span>
          <span class="kv-val">${formatNum(gw, 5)}</span>
          <span class="kv-key">Last Update (Δw)</span>
          <span class="kv-val delta-txt">${formatNum(dw, 6)}</span>
        </div>
      `;
    } else if (selection.type === "node") {
      if (selection.layerIdx === 0) {
        const name = selection.neuronIdx === 0 ? "x₁ (horizontal)" : "x₂ (vertical)";
        inspectorBody.innerHTML = `
          <div class="kv-grid">
            <span class="kv-key">Input Neuron</span>
            <span class="kv-val">${name}</span>
          </div>
        `;
        return;
      }
      const layer = model.layers[selection.layerIdx - 1];
      if (!layer || selection.neuronIdx >= layer.out_features) {
        selection = null;
        return renderInspector();
      }
      const b = layer.biases[selection.neuronIdx];
      const gb = layer.grad_b[selection.neuronIdx];
      const db = layer.last_delta_b[selection.neuronIdx];
      const act = layer.mean_activations[selection.neuronIdx];
      inspectorBody.innerHTML = `
        <div class="kv-grid">
          <span class="kv-key">Neuron</span>
          <span class="kv-val">Layer ${selection.layerIdx} · #${selection.neuronIdx} (${layer.activation})</span>
          <span class="kv-key">Bias (b)</span>
          <span class="kv-val ${b >= 0 ? "pos-txt" : "neg-txt"}">${formatNum(b, 5)}</span>
          <span class="kv-key">Bias Grad (∂L/∂b)</span>
          <span class="kv-val">${formatNum(gb, 5)}</span>
          <span class="kv-key">Last Update (Δb)</span>
          <span class="kv-val delta-txt">${formatNum(db, 6)}</span>
          <span class="kv-key">Mean Activation</span>
          <span class="kv-val">${formatNum(act, 4)}</span>
        </div>
      `;
    }
  }

  function renderTopUpdates() {
    const items = [];
    for (let l = 0; l < appState.model.layers.length; l++) {
      const layer = appState.model.layers[l];
      for (let j = 0; j < layer.out_features; j++) {
        for (let i = 0; i < layer.in_features; i++) {
          items.push({
            label: `L${l}:${i}→${j}`,
            w: layer.weights[j][i],
            gw: layer.grad_w[j][i],
            dw: layer.last_delta_w[j][i],
          });
        }
      }
    }
    items.sort((a, b) => Math.abs(b.dw) - Math.abs(a.dw));
    const top = items.slice(0, 6);
    updatesTbody.innerHTML = top
      .map(
        (it) => `
      <tr>
        <td>${it.label}</td>
        <td class="${it.w >= 0 ? "pos-txt" : "neg-txt"}">${formatNum(it.w, 3)}</td>
        <td>${formatNum(it.gw, 3)}</td>
        <td class="delta-txt">${formatNum(it.dw, 4)}</td>
      </tr>`
      )
      .join("");
  }

  function distToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  networkCanvas.addEventListener("click", (evt) => {
    const rect = networkCanvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;

    for (const node of renderedNodes) {
      if (Math.hypot(x - node.x, y - node.y) <= node.r + 3) {
        selection = { type: "node", layerIdx: node.layerIdx, neuronIdx: node.neuronIdx };
        renderNetwork();
        renderInspector();
        return;
      }
    }

    let bestEdge = null;
    let bestDist = 8;
    for (const edge of renderedEdges) {
      const d = distToSegment(x, y, edge.x1, edge.y1, edge.x2, edge.y2);
      if (d < bestDist) {
        bestDist = d;
        bestEdge = edge;
      }
    }
    if (bestEdge) {
      selection = {
        type: "edge",
        layerIdx: bestEdge.layerIdx,
        fromIdx: bestEdge.fromIdx,
        toIdx: bestEdge.toIdx,
      };
      renderNetwork();
      renderInspector();
    }
  });

  async function doStep(epochs) {
    if (stepInFlight) return;
    stepInFlight = true;
    try {
      appState = await apiPost("/api/step", { epochs });
      renderAll();
    } finally {
      stepInFlight = false;
    }
  }

  function stopTraining() {
    isTraining = false;
    btnPlayPause.textContent = "▶ Train";
    if (trainTimer) {
      clearInterval(trainTimer);
      trainTimer = null;
    }
  }

  function startTraining() {
    if (isTraining) return;
    isTraining = true;
    btnPlayPause.textContent = "⏸ Pause";
    const cfg = SPEED_MAP[selSpeed.value] || SPEED_MAP.medium;
    trainTimer = setInterval(() => {
      doStep(cfg.epochs);
    }, cfg.intervalMs);
  }

  btnPlayPause.addEventListener("click", () => {
    if (isTraining) stopTraining();
    else startTraining();
  });

  btnStep1.addEventListener("click", () => {
    stopTraining();
    doStep(1);
  });

  btnStep10.addEventListener("click", () => {
    stopTraining();
    doStep(10);
  });

  btnReset.addEventListener("click", async () => {
    stopTraining();
    appState = await apiPost("/api/reset", {});
    renderAll();
  });

  async function pushConfig(patch) {
    appState = await apiPost("/api/config", patch);
    renderAll();
  }

  selDataset.addEventListener("change", () => {
    stopTraining();
    pushConfig({ dataset: selDataset.value });
  });

  selArch.addEventListener("change", () => {
    stopTraining();
    selection = null;
    const hidden = selArch.value.split(",").map((s) => parseInt(s.trim(), 10));
    pushConfig({ hidden_layers: hidden });
  });

  selActivation.addEventListener("change", () => {
    stopTraining();
    pushConfig({ activation: selActivation.value });
  });

  selOptimizer.addEventListener("change", () => {
    pushConfig({ optimizer: selOptimizer.value });
  });

  rngLr.addEventListener("input", () => {
    const val = parseFloat(rngLr.value);
    lblLr.textContent = val.toFixed(3);
  });

  rngLr.addEventListener("change", () => {
    pushConfig({ lr: parseFloat(rngLr.value) });
  });

  selSpeed.addEventListener("change", () => {
    if (isTraining) {
      stopTraining();
      startTraining();
    }
  });

  chkLabels.addEventListener("change", renderNetwork);
  chkDelta.addEventListener("change", renderNetwork);
  window.addEventListener("resize", renderNetwork);

  async function init() {
    appState = await apiGet("/api/state");
    renderAll();
  }

  init();
})();
