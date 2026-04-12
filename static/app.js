/* EduPilot — Frontend App */
'use strict';

// ── State ──────────────────────────────────────────────
const state = {
  sessionId: null,
  config: null,
  domainColors: {},
  pendingFiles: [],
};

// ── Helpers ────────────────────────────────────────────
const $ = id => document.getElementById(id);
const API = path => fetch(path).then(r => r.json());
const POST = (path, body) =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    .then(r => r.json());

function genUUID() {
  return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16));
}

function storageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function storageSet(k, v) { try { localStorage.setItem(k, v); } catch {} }

// ── Init ───────────────────────────────────────────────
async function init() {
  state.config = await API('/api/config');
  buildModelSelect();
  buildDomainChips();
  buildUploadDomainSelect();
  buildSamplePrompts();

  state.sessionId = storageGet('edupilot_session') || genUUID();
  storageSet('edupilot_session', state.sessionId);

  await loadSessions();
  await loadMessages(state.sessionId);
  loadKB();
  loadEvalCases();

  bindEvents();
}

// ── Config-driven UI ───────────────────────────────────
function buildModelSelect() {
  const sel = $('modelSelect');
  sel.innerHTML = state.config.available_models
    .map(m => `<option value="${m}" ${m === state.config.default_model ? 'selected' : ''}>${m}</option>`)
    .join('');
}

function buildDomainChips() {
  const list = $('domainsList');
  list.innerHTML = '';
  for (const [abbr, d] of Object.entries(state.config.domains)) {
    state.domainColors[abbr] = d.color;
    const el = document.createElement('div');
    el.className = 'domain-chip active';
    el.dataset.domain = abbr;
    el.innerHTML = `
      <span class="domain-dot" style="background:${d.color}"></span>
      <span class="domain-name">${d.name}</span>
      <span class="domain-count" id="dcount-${abbr}">—</span>`;
    list.appendChild(el);
  }
}

function buildUploadDomainSelect() {
  const sel = $('uploadDomain');
  sel.innerHTML = Object.entries(state.config.domains)
    .map(([abbr, d]) => `<option value="${abbr}">${abbr} — ${d.name}</option>`).join('');
}

const SAMPLES = [
  'What is the bias-variance tradeoff?',
  'Explain database normalization with examples.',
  'What is a p-value and how do I interpret it?',
  'How does attention work in transformers?',
  'What is RAG and how does it work?',
  'How does NL2SQL work and what ML does it use?',
];

function buildSamplePrompts() {
  $('samplePrompts').innerHTML = SAMPLES
    .map(s => `<button class="sample-btn" data-prompt="${s}">${s.slice(0, 40)}…</button>`)
    .join('');
}

// ── Sessions ───────────────────────────────────────────
async function loadSessions() {
  const data = await API('/api/sessions');
  const list = $('sessionsList');
  list.innerHTML = '';
  for (const s of data.sessions) {
    const label = s.title || `Session ${s.session_id.slice(0, 8)}`;
    const el = document.createElement('div');
    el.className = 'session-item' + (s.session_id === state.sessionId ? ' active' : '');
    el.dataset.sid = s.session_id;
    el.innerHTML = `<span class="session-label">${escHtml(label)}</span>
                    <button class="del-btn" title="Delete" data-sid="${s.session_id}">✕</button>`;
    el.addEventListener('click', async e => {
      if (e.target.classList.contains('del-btn')) {
        e.stopPropagation();
        if (!confirm('Delete this session?')) return;
        await fetch(`/api/sessions/${s.session_id}`, { method: 'DELETE' });
        if (s.session_id === state.sessionId) await newSession();
        else { await loadSessions(); }
        return;
      }
      await switchSession(s.session_id);
    });
    list.appendChild(el);
  }
}

async function switchSession(sid) {
  state.sessionId = sid;
  storageSet('edupilot_session', sid);
  await loadMessages(sid);
  await loadSessions();
}

async function newSession() {
  const res = await POST('/api/sessions', {});
  state.sessionId = res.session_id;
  storageSet('edupilot_session', state.sessionId);
  clearMessages();
  await loadSessions();
}

async function loadMessages(sid) {
  clearMessages();
  const data = await fetch(`/api/sessions/${sid}`).then(r => r.json());
  for (const m of data.messages) {
    if (m.role === 'user') appendUserMsg(m.content);
    else appendAssistantMsg(m.content, null, {});
  }
}

function clearMessages() {
  const c = $('messages');
  c.innerHTML = '';
}

// ── Chat ───────────────────────────────────────────────
async function sendMessage() {
  const input = $('queryInput');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  input.style.height = 'auto';
  $('sendBtn').disabled = true;

  // Remove welcome message
  const welcome = $('messages').querySelector('.welcome-msg');
  if (welcome) welcome.remove();

  appendUserMsg(query);
  const thinkingEl = appendThinking();

  const payload = {
    query,
    session_id: state.sessionId,
    model: $('modelSelect').value,
    top_k: parseInt($('topK').value),
    rerank_top_k: parseInt($('rerankK').value),
    confidence_threshold: parseFloat($('conf').value),
    enable_verification: $('verifyToggle').checked,
    manual_domains: null,
    chat_history: [],
  };

  try {
    const result = await POST('/api/chat', payload);
    thinkingEl.remove();
    appendAssistantMsg(result.final_answer, result, $('debugToggle').checked);
    await loadSessions();
    await updateDomainCounts();
  } catch (err) {
    thinkingEl.remove();
    appendAssistantMsg(`⚠️ Error: ${err.message}`, null, false);
  }

  $('sendBtn').disabled = false;
  input.focus();
}

function appendUserMsg(text) {
  const div = document.createElement('div');
  div.className = 'message user';
  div.innerHTML = `<div class="avatar">👤</div>
                   <div class="bubble">${escHtml(text)}</div>`;
  $('messages').appendChild(div);
  scrollBottom();
}

function appendAssistantMsg(text, result, showDebug) {
  const div = document.createElement('div');
  div.className = 'message assistant';

  let meta = '';
  if (result && result.detected_domains && result.detected_domains.length) {
    meta += result.detected_domains
      .map(d => `<span class="badge" style="background:${state.domainColors[d] || '#555'}">${d}</span>`)
      .join(' ');
  }
  if (result && result.quality_score) {
    const qs = Math.round(result.quality_score * 100);
    const icon = qs >= 70 ? '✅' : qs >= 40 ? '⚠️' : '❌';
    meta += `<span class="quality-chip">${icon} Quality ${qs}%</span>`;
    if (result.verification_revised)
      meta += `<span class="quality-chip">✏️ Revised</span>`;
  }

  const debugHtml = (showDebug && result && result.debug) ? buildDebugHtml(result.debug) : '';

  div.innerHTML = `
    <div class="avatar">🎓</div>
    <div style="flex:1;min-width:0">
      ${meta ? `<div class="msg-meta">${meta}</div>` : ''}
      <div class="bubble">${renderMd(text)}</div>
      ${debugHtml}
    </div>`;

  $('messages').appendChild(div);
  hljs.highlightAll();
  scrollBottom();
}

function appendThinking() {
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `<div class="avatar">🎓</div>
    <div class="thinking">
      <div class="dot-bounce"><span></span><span></span><span></span></div>
      Thinking…
    </div>`;
  $('messages').appendChild(div);
  scrollBottom();
  return div;
}

function buildDebugHtml(debug) {
  const rows = [];

  if (debug.router) {
    const r = debug.router;
    rows.push(`<div class="debug-section">Step 1 — Routing</div>
      <div class="debug-row">Intent: <b>${r.intent_type}</b> | Domains: <b>${r.domains}</b><br>
      ${escHtml(r.reasoning || '')}</div>`);
  }
  if (debug.sub_questions && debug.sub_questions.length) {
    const sqs = debug.sub_questions.map((sq, i) =>
      `${i + 1}. [${sq.domain}] ${escHtml(sq.question)}`).join('<br>');
    rows.push(`<div class="debug-section">Step 2 — Decomposition</div>
               <div class="debug-row">${sqs}</div>`);
  }
  if (debug.retrieval && debug.retrieval.length) {
    const ret = debug.retrieval.map(rd =>
      `<b>${rd.domain}</b>: ${rd.reranked_count}/${rd.raw_count} chunks kept`
    ).join('<br>');
    rows.push(`<div class="debug-section">Steps 3–4 — Retrieval & Reranking</div>
               <div class="debug-row">${ret}</div>`);
  }
  if (debug.verification && !debug.verification.skipped) {
    const v = debug.verification;
    rows.push(`<div class="debug-section">Step 7 — Verification</div>
      <div class="debug-row">Quality: ${pct(v.quality_score)} | Coverage: ${pct(v.coverage_score)} | Grounding: ${pct(v.grounding_score)}
      ${v.was_revised ? '<br>✏️ Answer was revised' : ''}</div>`);
  }

  if (!rows.length) return '';

  return `<div class="debug-panel">
    <div class="debug-header" onclick="this.nextElementSibling.classList.toggle('open')">
      🔧 Debug Panel <span>▾</span>
    </div>
    <div class="debug-body">${rows.join('')}</div>
  </div>`;
}

function pct(v) { return `${Math.round((v || 0) * 100)}%`; }

function renderMd(text) {
  marked.setOptions({ breaks: true, gfm: true });
  return marked.parse(text);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function scrollBottom() {
  const c = $('messages');
  requestAnimationFrame(() => c.scrollTop = c.scrollHeight);
}

// ── Knowledge Base ─────────────────────────────────────
async function loadKB() {
  const data = await API('/api/kb/status');
  const grid = $('kbContent');
  grid.innerHTML = '';

  for (const [domain, info] of Object.entries(data)) {
    const card = document.createElement('div');
    card.className = 'kb-card';
    card.innerHTML = `
      <div class="kb-card-header">
        <span class="domain-dot" style="background:${info.color}"></span>
        <span class="kb-card-title">${escHtml(info.name)}</span>
      </div>
      <div>
        <p class="kb-card-count">${info.chunk_count}</p>
        <p class="kb-card-label">chunks indexed</p>
      </div>
      <div class="kb-file-list">
        ${info.kb_files.map(f => `<div class="kb-file">📄 ${escHtml(f)}</div>`).join('')}
        ${info.uploaded_docs.map(d =>
          `<div class="kb-file">📎 ${escHtml(d.filename)} (${d.chunk_count} chunks)</div>`
        ).join('')}
      </div>`;
    grid.appendChild(card);

    // Also update sidebar count
    const badge = $(`dcount-${domain}`);
    if (badge) badge.textContent = info.chunk_count;
  }
}

async function updateDomainCounts() {
  const data = await API('/api/kb/status');
  for (const [domain, info] of Object.entries(data)) {
    const badge = $(`dcount-${domain}`);
    if (badge) badge.textContent = info.chunk_count;
  }
}

// ── Evaluation ─────────────────────────────────────────
async function loadEvalCases() {
  const data = await API('/api/evaluate/cases');
  const sel = $('singleTcSelect');
  sel.innerHTML = data.test_cases
    .map(tc => `<option value="${tc.id}">${tc.id}: ${escHtml(tc.name)}</option>`).join('');
}

async function runAllEvals() {
  const prog = $('evalProgress');
  const fill = $('progressFill');
  const label = $('progressLabel');
  prog.hidden = false;
  fill.style.width = '5%';
  label.textContent = 'Running all test cases…';
  $('runAllBtn').disabled = true;

  try {
    const data = await fetch('/api/evaluate', { method: 'POST' }).then(r => r.json());
    fill.style.width = '100%';
    label.textContent = 'Done';
    setTimeout(() => { prog.hidden = true; }, 1500);
    renderEvalSummary(data.stats);
    renderEvalResults(data.results);
  } catch (err) {
    label.textContent = `Error: ${err.message}`;
  }
  $('runAllBtn').disabled = false;
}

async function runSingleEval() {
  const tcId = $('singleTcSelect').value;
  if (!tcId) return;
  $('runSingleBtn').disabled = true;
  $('evalResults').innerHTML = `<div class="thinking"><div class="dot-bounce"><span></span><span></span><span></span></div>Running ${tcId}…</div>`;

  try {
    const data = await fetch(`/api/evaluate/${tcId}`, { method: 'POST' }).then(r => r.json());
    renderEvalResults([data]);
  } catch (err) {
    $('evalResults').innerHTML = `<div style="color:var(--error)">Error: ${escHtml(err.message)}</div>`;
  }
  $('runSingleBtn').disabled = false;
}

function renderEvalSummary(stats) {
  const el = $('evalSummary');
  el.hidden = false;
  el.innerHTML = `
    <div class="eval-stat">
      <div class="eval-stat-val">${stats.pass_rate}%</div>
      <div class="eval-stat-label">Pass Rate (${stats.passed}/${stats.total})</div>
    </div>
    <div class="eval-stat">
      <div class="eval-stat-val">${stats.intent_accuracy}%</div>
      <div class="eval-stat-label">Intent Accuracy</div>
    </div>
    <div class="eval-stat">
      <div class="eval-stat-val">${stats.domain_accuracy}%</div>
      <div class="eval-stat-label">Domain Accuracy</div>
    </div>
    <div class="eval-stat">
      <div class="eval-stat-val">${(stats.avg_quality_score || 0).toFixed(2)}</div>
      <div class="eval-stat-label">Avg Quality</div>
    </div>`;
}

function renderEvalResults(results) {
  const el = $('evalResults');
  el.innerHTML = '';
  for (const r of results) {
    const icon = r.passed ? '✅' : '❌';
    const card = document.createElement('div');
    card.className = 'eval-card';
    card.innerHTML = `
      <div class="eval-card-header" onclick="this.nextElementSibling.classList.toggle('open')">
        <span class="pass-icon">${icon}</span>
        <span class="eval-tc-name">${r.test_case_id || ''}: ${escHtml(r.name || '')}</span>
        <span class="quality-chip">${r.quality_score ? pct(r.quality_score) : ''}</span>
        <span>▾</span>
      </div>
      <div class="eval-card-body">
        <div class="eval-meta-row">
          <div class="eval-meta"><div class="eval-meta-key">Intent</div>
            <div class="eval-meta-val">${r.intent_match ? '✅' : '❌'} ${escHtml(r.actual_intent || '')}</div></div>
          <div class="eval-meta"><div class="eval-meta-key">Domains</div>
            <div class="eval-meta-val">${r.domain_match ? '✅' : '❌'} ${escHtml((r.actual_domains || []).join(', '))}</div></div>
          <div class="eval-meta"><div class="eval-meta-key">Quality</div>
            <div class="eval-meta-val">${r.quality_score ? pct(r.quality_score) : 'N/A'}</div></div>
        </div>
        ${r.expected_behavior ? `<div class="eval-answer"><b>Expected:</b> ${escHtml(r.expected_behavior)}</div>` : ''}
        ${r.answer_preview ? `<div class="eval-answer"><b>Answer:</b> ${escHtml(r.answer_preview)}…</div>` : ''}
        ${r.error ? `<div style="color:var(--error);font-size:12px">${escHtml(r.error)}</div>` : ''}
      </div>`;
    el.appendChild(card);
  }
}

// ── Uploads ────────────────────────────────────────────
function updateFileList() {
  const fl = $('fileList');
  fl.innerHTML = state.pendingFiles.map(f =>
    `<div class="file-item">📄 ${escHtml(f.name)} <span style="color:var(--text-muted)">(${(f.size/1024).toFixed(1)} KB)</span></div>`
  ).join('');
}

async function doUpload() {
  if (!state.pendingFiles.length) return;
  const domain = $('uploadDomain').value;
  const status = $('uploadStatus');
  status.textContent = '⏳ Indexing…';
  $('uploadBtn').disabled = true;

  const fd = new FormData();
  fd.append('domain', domain);
  for (const f of state.pendingFiles) fd.append('files', f);

  try {
    const res = await fetch('/api/kb/upload', { method: 'POST', body: fd }).then(r => r.json());
    const total = res.uploaded.reduce((s, u) => s + u.chunks_indexed, 0);
    status.style.color = 'var(--success)';
    status.textContent = `✅ ${res.uploaded.length} file(s), ${total} chunks indexed`;
    state.pendingFiles = [];
    updateFileList();
    loadKB();
  } catch (err) {
    status.style.color = 'var(--error)';
    status.textContent = `❌ ${err.message}`;
  }
  $('uploadBtn').disabled = false;
}

// ── Events ─────────────────────────────────────────────
function bindEvents() {
  // Tabs
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      $(`tab-${btn.dataset.tab}`).classList.add('active');
      if (btn.dataset.tab === 'kb') loadKB();
    });
  });

  // Send
  $('sendBtn').addEventListener('click', sendMessage);
  $('queryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  // Auto-resize textarea
  $('queryInput').addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 140) + 'px';
  });

  // Sample prompts
  $('samplePrompts').addEventListener('click', e => {
    if (e.target.classList.contains('sample-btn')) {
      $('queryInput').value = e.target.dataset.prompt;
      $('queryInput').dispatchEvent(new Event('input'));
      sendMessage();
    }
  });

  // Sidebar toggle
  $('sidebarToggle').addEventListener('click', () => {
    document.querySelector('.sidebar').classList.toggle('collapsed');
  });

  // New session
  $('newSessionBtn').addEventListener('click', newSession);

  // Sliders
  $('topK').addEventListener('input', () => $('topKVal').textContent = $('topK').value);
  $('rerankK').addEventListener('input', () => $('rerankKVal').textContent = $('rerankK').value);
  $('conf').addEventListener('input', () => $('confVal').textContent = parseFloat($('conf').value).toFixed(2));

  // Debug toggle — re-render last response not practical; just affects future msgs
  $('debugToggle').addEventListener('change', () => {});

  // File upload
  $('uploadLabel') && $('uploadLabel').addEventListener('click', () => $('fileInput').click());
  document.querySelector('.upload-label').addEventListener('click', () => $('fileInput').click());

  $('fileInput').addEventListener('change', e => {
    state.pendingFiles = Array.from(e.target.files);
    updateFileList();
  });

  // Drag-and-drop upload
  const uploadBox = $('uploadBox');
  uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.style.opacity = '.7'; });
  uploadBox.addEventListener('dragleave', () => { uploadBox.style.opacity = '1'; });
  uploadBox.addEventListener('drop', e => {
    e.preventDefault(); uploadBox.style.opacity = '1';
    state.pendingFiles = Array.from(e.dataTransfer.files)
      .filter(f => /\.(pdf|txt|md|docx)$/i.test(f.name));
    updateFileList();
  });

  $('uploadBtn').addEventListener('click', doUpload);

  // Evaluation
  $('runAllBtn').addEventListener('click', runAllEvals);
  $('runSingleBtn').addEventListener('click', runSingleEval);
}

// ── Start ──────────────────────────────────────────────
init().catch(console.error);
