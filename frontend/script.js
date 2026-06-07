// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = "http://127.0.0.1:8000";

// ── Section switching ────────────────────────────────────────────────────────
function showSection(id) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  document.querySelectorAll(".nav-btn").forEach(b => {
    if (b.getAttribute("onclick").includes(id)) b.classList.add("active");
  });
  if (id === "dashboard") loadStats();
  if (id === "history")   loadHistory();
  if (id === "alerts")    loadAlerts();
}

// ── PCA input grid (V1–V28) ──────────────────────────────────────────────────
function buildPCAInputs() {
  const wrap = document.getElementById("pca-inputs");
  if (!wrap || wrap.children.length > 0) return;
  for (let i = 1; i <= 28; i++) {
    const inp = document.createElement("input");
    inp.type        = "number";
    inp.id          = `cc-v${i}`;
    inp.placeholder = `V${i}`;
    inp.step        = "0.000001";
    inp.value       = "0";
    wrap.appendChild(inp);
  }
}
document.addEventListener("DOMContentLoaded", () => {
  buildPCAInputs();
  loadStats();
});

// ── Stats ────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const res  = await fetch(`${API_BASE}/api/fraud/stats`);
    const data = await res.json();
    document.getElementById("stat-total").textContent  = data.total_transactions.toLocaleString();
    document.getElementById("stat-fraud").textContent  = data.fraud_count.toLocaleString();
    document.getElementById("stat-rate").textContent   = (data.fraud_rate * 100).toFixed(2) + "%";
    document.getElementById("stat-alerts").textContent = data.recent_alerts;

    const badge = document.getElementById("alert-badge");
    if (data.recent_alerts > 0) {
      badge.textContent = data.recent_alerts;
      badge.classList.remove("hidden");
    } else {
      badge.classList.add("hidden");
    }
  } catch {
    // API not running yet
  }
}

// ── Shared result renderer ───────────────────────────────────────────────────
function renderResult(containerId, data) {
  const el   = document.getElementById(containerId);
  const rl   = data.risk_level.toLowerCase();
  const pct  = (data.fraud_probability * 100).toFixed(2);
  const emoji = { safe: "✅", low: "🟡", medium: "🟠", high: "🚨" }[rl] || "❓";
  const label = { safe: "SAFE", low: "LOW RISK", medium: "MEDIUM RISK", high: "HIGH RISK — FRAUD" }[rl];

  el.className = `result-card ${rl}`;
  el.innerHTML = `
    <div class="result-header">
      <span class="result-emoji">${emoji}</span>
      <div>
        <div class="result-title">${label}</div>
        <div class="result-sub">Transaction #${data.transaction_id} · ${data.message}</div>
      </div>
    </div>
    <div class="prob-bar-wrap">
      <div class="prob-bar-label"><span>Fraud Probability</span><span>${pct}%</span></div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill ${rl}" style="width:${pct}%"></div>
      </div>
    </div>
    <div style="display:flex;gap:2rem;margin-top:.5rem;font-size:.9rem;color:var(--muted);">
      <span><b>Is Fraud:</b> ${data.is_fraud ? "YES" : "NO"}</span>
      <span><b>Risk Level:</b> ${data.risk_level}</span>
      <span><b>Probability:</b> ${data.fraud_probability}</span>
    </div>
  `;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Credit Card Prediction ───────────────────────────────────────────────────
function collectCCPayload() {
  const payload = {
    Amount: parseFloat(document.getElementById("cc-amount").value) || 0,
    Time:   parseFloat(document.getElementById("cc-time").value)   || 0,
  };
  for (let i = 1; i <= 28; i++) {
    payload[`V${i}`] = parseFloat(document.getElementById(`cc-v${i}`).value) || 0;
  }
  return payload;
}

async function predictCC() {
  const btn = event.target;
  btn.innerHTML = '<span class="spinner"></span> Analysing…';
  btn.disabled  = true;

  try {
    const res  = await fetch(`${API_BASE}/api/fraud/predict/creditcard`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(collectCCPayload()),
    });
    if (!res.ok) throw new Error(await res.text());
    renderResult("cc-result", await res.json());
    loadStats();
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    btn.innerHTML = "🔍 Check for Fraud";
    btn.disabled  = false;
  }
}

// Legit sample: average-looking transaction
function fillCCSample(isFraud) {
  document.getElementById("cc-amount").value = isFraud ? "2.69" : "149.62";
  document.getElementById("cc-time").value   = "0";

  // Known legit values from dataset row 1
  const legit = [-1.3598,-0.0728,2.5363,1.3782,-0.3383,0.4624,0.2396,0.0987,
                  0.3638,0.0908,-0.5516,-0.6178,-0.9914,-0.3112,1.4682,-0.4704,
                  0.2080,0.0258,0.4040,0.2514,-0.0183,0.2778,-0.1105,0.0669,
                  0.1285,-0.1891,0.1336,-0.0211];

  // Known fraud values (approximated from dataset)
  const fraud  = [-2.3122,1.9519,-1.6097,-0.0632,-1.1895,0.3618,-0.0516,0.6960,
                   0.0358,-0.6813,1.9459,-0.2127,-0.4557,0.3998,-0.5379,0.4164,
                   0.0774,-0.2953,0.2126,0.0027,0.6654,-0.0736,0.4476,-0.1070,
                   0.5665,-0.0657,-0.2656,-0.0428];

  const vals = isFraud ? fraud : legit;
  for (let i = 1; i <= 28; i++) {
    document.getElementById(`cc-v${i}`).value = vals[i - 1];
  }
}

// ── Transaction Prediction ───────────────────────────────────────────────────
async function predictTxn() {
  const btn = event.target;
  btn.innerHTML = '<span class="spinner"></span> Analysing…';
  btn.disabled  = true;

  const payload = {
    step:             parseInt(document.getElementById("txn-step").value)        || 1,
    transaction_type: document.getElementById("txn-type").value,
    amount:           parseFloat(document.getElementById("txn-amount").value)    || 0,
    oldbalanceOrg:    parseFloat(document.getElementById("txn-oldbalOrig").value)|| 0,
    newbalanceOrig:   parseFloat(document.getElementById("txn-newbalOrig").value)|| 0,
    oldbalanceDest:   parseFloat(document.getElementById("txn-oldbalDest").value)|| 0,
    newbalanceDest:   parseFloat(document.getElementById("txn-newbalDest").value)|| 0,
  };

  try {
    const res  = await fetch(`${API_BASE}/api/fraud/predict/transaction`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    renderResult("txn-result", await res.json());
    loadStats();
  } catch (e) {
    alert("Error: " + e.message);
  } finally {
    btn.innerHTML = "🔍 Check for Fraud";
    btn.disabled  = false;
  }
}

function fillTxnSample(isFraud) {
  document.getElementById("txn-type").value = isFraud ? "TRANSFER" : "PAYMENT";
  document.getElementById("txn-step").value = isFraud ? "1" : "239";
  document.getElementById("txn-amount").value      = isFraud ? "181.00" : "2363.31";
  document.getElementById("txn-oldbalOrig").value  = isFraud ? "181.00" : "79864.43";
  document.getElementById("txn-newbalOrig").value  = isFraud ? "0.00"   : "77501.12";
  document.getElementById("txn-oldbalDest").value  = isFraud ? "0.00"   : "0.00";
  document.getElementById("txn-newbalDest").value  = isFraud ? "0.00"   : "0.00";
}

// ── History ─────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res  = await fetch(`${API_BASE}/api/fraud/history?limit=30`);
    const data = await res.json();
    const wrap = document.getElementById("history-table-wrap");

    if (!data.length) {
      wrap.innerHTML = '<p class="empty-state">No predictions yet.</p>';
      return;
    }

    const rows = data.map(r => {
      const rl = r.risk_level.toLowerCase();
      return `<tr>
        <td>#${r.transaction_id}</td>
        <td>${r.dataset.toUpperCase()}</td>
        <td>$${r.amount.toFixed(2)}</td>
        <td>${(r.fraud_probability * 100).toFixed(2)}%</td>
        <td>${r.is_fraud ? "YES" : "NO"}</td>
        <td><span class="badge-${rl}">${r.risk_level}</span></td>
        <td style="color:var(--muted);font-size:.8rem">${r.created_at?.split(".")[0] ?? ""}</td>
      </tr>`;
    }).join("");

    wrap.innerHTML = `
      <table class="data-table">
        <thead><tr>
          <th>#ID</th><th>Dataset</th><th>Amount</th>
          <th>Fraud %</th><th>Fraud?</th><th>Risk</th><th>Time</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch {
    document.getElementById("history-table-wrap").innerHTML =
      '<p class="empty-state">Could not load history — is the API running?</p>';
  }
}

// ── Alerts ──────────────────────────────────────────────────────────────────
async function loadAlerts() {
  try {
    const res  = await fetch(`${API_BASE}/api/fraud/alerts?resolved=false`);
    const data = await res.json();
    const wrap = document.getElementById("alerts-list");

    if (!data.length) {
      wrap.innerHTML = '<p class="empty-state">No open alerts. 🎉</p>';
      return;
    }

    wrap.innerHTML = data.map(a => `
      <div class="alert-item ${a.risk_level.toLowerCase()}">
        <div>
          <strong>🚨 ${a.risk_level} RISK — Transaction #${a.transaction_id}</strong>
          <div>${a.message}</div>
          <div class="alert-meta">${a.created_at?.split(".")[0] ?? ""}</div>
        </div>
        <button class="btn btn-outline" onclick="resolveAlert(${a.id})">Resolve</button>
      </div>
    `).join("");
  } catch {
    document.getElementById("alerts-list").innerHTML =
      '<p class="empty-state">Could not load alerts — is the API running?</p>';
  }
}

async function resolveAlert(id) {
  await fetch(`${API_BASE}/api/fraud/alerts/${id}/resolve`, { method: "PATCH" });
  loadAlerts();
  loadStats();
}
