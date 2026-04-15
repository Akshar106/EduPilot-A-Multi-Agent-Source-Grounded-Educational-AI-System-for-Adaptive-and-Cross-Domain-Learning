'use strict';

// ── State ──────────────────────────────────────────────
const S = {
  sessionId:    null,
  config:       null,
  domainColors: {},
  pendingFiles: [],   // files staged in the modal, not yet indexed
  attachedFiles: [],  // files already indexed, shown as chips
  previewFile:   null,
};

// ── Helpers ────────────────────────────────────────────
const $  = id  => document.getElementById(id);
const qs = sel => document.querySelector(sel);

async function apiFetch(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const txt = await r.text().catch(() => r.statusText);
    throw new Error(txt || r.statusText);
  }
  return r.json();
}

function post(path, body) {
  return apiFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function escHtml(s) {
  return String(s ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderMd(text) {
  marked.setOptions({ breaks: true, gfm: true });
  return marked.parse(text || '');
}

function pct(v) { return `${Math.round((v || 0) * 100)}%`; }

function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function storageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function storageSet(k, v) { try { localStorage.setItem(k, v); } catch {} }

function genUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

// ── Init ───────────────────────────────────────────────
async function init() {
  S.config = await apiFetch('/api/config');

  for (const [abbr, d] of Object.entries(S.config.domains)) {
    S.domainColors[abbr] = d.color;
  }

  buildSamplePrompts();
  buildUploadDomainSelect();
  await loadSessions();

  S.sessionId = storageGet('ep_session') || genUUID();
  storageSet('ep_session', S.sessionId);
  await loadSessionMessages(S.sessionId);
  highlightActiveSession();

  loadEvalCases();
  renderKBTab();
  bindEvents();
}

// ── Sample prompts ─────────────────────────────────────
const SAMPLES = [
  'Explain the bias-variance tradeoff.',
  'What is database normalization?',
  'What is a p-value?',
  'How does attention work in transformers?',
  'What is RAG and why does it help?',
  'How does NL2SQL work?',
];

function buildSamplePrompts() {
  $('samplePrompts').innerHTML = SAMPLES.map(s =>
    `<button class="sample-btn" data-prompt="${escHtml(s)}">${escHtml(s)}</button>`
  ).join('');
}

// ── Upload domain select ───────────────────────────────
function buildUploadDomainSelect() {
  $('uploadDomain').innerHTML = Object.entries(S.config.domains)
    .map(([a, d]) => `<option value="${a}">${a} — ${d.name}</option>`).join('');
}

async function refreshDomainCounts() {
  try {
    const data = await apiFetch('/api/kb/status');
    for (const [d, info] of Object.entries(data)) {
      const el = $(`dcount-${d}`);
      if (el) el.textContent = (info.chunk_count || 0).toLocaleString();
    }
  } catch {}
}

// ── Sessions ───────────────────────────────────────────
async function loadSessions() {
  const data = await apiFetch('/api/sessions');
  const list = $('sessionsList');
  list.innerHTML = '';
  for (const s of data.sessions) {
    const label = s.title || `Chat ${s.session_id.slice(0, 8)}`;
    const li = document.createElement('div');
    li.className = 'session-item' + (s.session_id === S.sessionId ? ' active' : '');
    li.dataset.sid = s.session_id;
    li.innerHTML = `<span class="session-label">${escHtml(label)}</span>
                    <button class="session-del" data-sid="${s.session_id}" title="Delete">✕</button>`;
    li.addEventListener('click', async e => {
      if (e.target.classList.contains('session-del')) {
        e.stopPropagation();
        if (!confirm('Delete this conversation?')) return;
        await fetch(`/api/sessions/${s.session_id}`, { method: 'DELETE' });
        if (s.session_id === S.sessionId) await startNewSession();
        else await loadSessions();
        return;
      }
      await switchSession(s.session_id);
    });
    list.appendChild(li);
  }
}

function highlightActiveSession() {
  document.querySelectorAll('.session-item').forEach(el => {
    el.classList.toggle('active', el.dataset.sid === S.sessionId);
  });
}

async function switchSession(sid) {
  S.sessionId = sid;
  storageSet('ep_session', sid);
  clearMessages();
  await loadSessionMessages(sid);
  await loadSessions();
}

async function startNewSession() {
  const res = await post('/api/sessions', {});
  S.sessionId = res.session_id;
  storageSet('ep_session', S.sessionId);
  clearMessages();
  S.attachedFiles = [];
  renderAttachChips();
  await loadSessions();
}

async function loadSessionMessages(sid) {
  clearMessages();
  try {
    const data = await apiFetch(`/api/sessions/${sid}`);
    if (data.messages && data.messages.length > 0) {
      for (const m of data.messages) {
        if (m.role === 'user') appendUserBubble(m.content);
        else appendAssistantBubble(m.content, null, false);
      }
    } else {
      showWelcome();
    }
  } catch { showWelcome(); }
}

// ── Messages ───────────────────────────────────────────
function clearMessages() { $('messages').innerHTML = ''; }

function showWelcome() {
  $('messages').innerHTML = `
    <div class="welcome-msg">
      <div class="welcome-icon">🎓</div>
      <h3>Welcome to EduPilot</h3>
      <p>Your AI tutor for <strong>Applied Machine Learning</strong>,
         <strong>Applied Database Technologies</strong>,
         <strong>Statistics</strong>, and
         <strong>Large Language Models</strong>.</p>
    </div>`;
}

function removeWelcome() {
  const w = qs('.welcome-msg');
  if (w) w.remove();
}

function appendUserBubble(text) {
  const d = document.createElement('div');
  d.className = 'message user';
  d.innerHTML = `<div class="avatar">U</div>
                 <div class="bubble">${escHtml(text)}</div>`;
  $('messages').appendChild(d);
  scrollBottom();
}

function appendAssistantBubble(text, result, showDebug) {
  const d = document.createElement('div');
  d.className = 'message assistant';

  let meta = '';
  if (result?.detected_domains?.length) {
    meta += result.detected_domains.map(dm =>
      `<span class="badge" style="background:${S.domainColors[dm] || '#999'}">${dm}</span>`
    ).join(' ');
  }
  if (result?.quality_score != null) {
    const q = Math.round(result.quality_score * 100);
    const icon = q >= 70 ? '✅' : q >= 40 ? '⚠️' : '❌';
    meta += ` <span class="quality-chip">${icon} Quality ${q}%</span>`;
    if (result.verification_revised) meta += ` <span class="quality-chip">✏️ Revised</span>`;
  }

  const debugHtml = showDebug && result?.debug ? buildDebugHtml(result.debug) : '';

  d.innerHTML = `
    <div class="avatar">🎓</div>
    <div style="flex:1;min-width:0">
      ${meta ? `<div class="msg-meta">${meta}</div>` : ''}
      <div class="bubble">${renderMd(text)}</div>
      ${debugHtml}
    </div>`;
  $('messages').appendChild(d);
  hljs.highlightAll();
  scrollBottom();
}

function appendThinking() {
  const d = document.createElement('div');
  d.className = 'message assistant';
  d.innerHTML = `<div class="avatar">🎓</div>
    <div class="thinking">
      <div class="dot-bounce"><span></span><span></span><span></span></div>
      Thinking…
    </div>`;
  $('messages').appendChild(d);
  scrollBottom();
  return d;
}

function buildDebugHtml(debug) {
  const rows = [];
  if (debug.router) {
    const r = debug.router;
    rows.push(`<div class="debug-section">Step 1 — Routing</div>
      <div class="debug-row">Intent: <b>${r.intent_type}</b> | Domains: <b>${(r.domains||[]).join(', ')}</b><br>${escHtml(r.reasoning||'')}</div>`);
  }
  if (debug.sub_questions?.length) {
    rows.push(`<div class="debug-section">Step 2 — Decomposition</div>
      <div class="debug-row">${debug.sub_questions.map((sq,i)=>`${i+1}. [${sq.domain}] ${escHtml(sq.question)}`).join('<br>')}</div>`);
  }
  if (debug.retrieval?.length) {
    rows.push(`<div class="debug-section">Steps 3–4 — Retrieval &amp; Reranking</div>
      <div class="debug-row">${debug.retrieval.map(rd=>`<b>${rd.domain}</b>: kept ${rd.reranked_count}/${rd.raw_count} chunks`).join('<br>')}</div>`);
  }
  if (debug.verification && !debug.verification.skipped) {
    const v = debug.verification;
    rows.push(`<div class="debug-section">Step 7 — Verification</div>
      <div class="debug-row">Quality ${pct(v.quality_score)} | Coverage ${pct(v.coverage_score)} | Grounding ${pct(v.grounding_score)}${v.was_revised?'<br>✏️ Revised':''}</div>`);
  }
  if (!rows.length) return '';
  return `<div class="debug-panel">
    <div class="debug-header" onclick="this.nextElementSibling.classList.toggle('open')">🔧 Pipeline Debug <span>▾</span></div>
    <div class="debug-body">${rows.join('')}</div>
  </div>`;
}

function scrollBottom() {
  const c = $('messages');
  requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
}

// ── Chat send ──────────────────────────────────────────
async function sendMessage() {
  const inp = $('queryInput');
  const query = inp.value.trim();
  if (!query) return;

  inp.value = '';
  inp.style.height = 'auto';
  $('sendBtn').disabled = true;
  removeWelcome();
  appendUserBubble(query);
  const thinking = appendThinking();

  try {
    const result = await post('/api/chat', {
      query,
      session_id: S.sessionId,
      model: S.config.model,
      top_k: parseInt($('topK').value),
      rerank_top_k: parseInt($('rerankK').value),
      confidence_threshold: parseFloat($('conf').value),
      enable_verification: $('verifyToggle').checked,
      manual_domains: S.attachedFiles.length
        ? [...new Set(S.attachedFiles.map(a => a.domain))]
        : null,
      attached_filenames: S.attachedFiles.length
        ? S.attachedFiles.map(a => a.file.name)
        : null,
      chat_history: [],
    });
    thinking.remove();
    appendAssistantBubble(result.final_answer, result, $('debugToggle').checked);
    await loadSessions();
    highlightActiveSession();
    refreshDomainCounts();
  } catch (err) {
    thinking.remove();
    appendAssistantBubble(`⚠️ Error: ${escHtml(err.message)}`, null, false);
  }

  $('sendBtn').disabled = false;
  inp.focus();
}

// ── Attach chips ───────────────────────────────────────
function renderAttachChips() {
  const container = $('inputAttachments');
  if (!S.attachedFiles.length) { container.hidden = true; container.innerHTML = ''; return; }
  container.hidden = false;
  container.innerHTML = S.attachedFiles.map((a, i) =>
    `<div class="attach-chip" data-idx="${i}">
       <span>📄 ${escHtml(a.file.name)} (${a.domain})</span>
       <button class="attach-chip-remove" data-idx="${i}" title="Remove">✕</button>
     </div>`
  ).join('');
}

// ── Preview panel ──────────────────────────────────────
function openPreview(file) {
  S.previewFile = file;
  $('previewTitle').textContent = file.name;
  $('previewMeta').textContent = `${fmtSize(file.size)} · ${file.name.split('.').pop().toUpperCase()}`;

  const ext = file.name.split('.').pop().toLowerCase();
  const body = $('previewBody');

  if (ext === 'pdf') {
    const url = URL.createObjectURL(file);
    body.innerHTML = `<iframe src="${url}" style="width:100%;height:100%;border:none;"></iframe>`;
    body.style.padding = '0';
    body.style.whiteSpace = 'normal';
  } else if (['txt', 'md'].includes(ext)) {
    const reader = new FileReader();
    reader.onload = e => {
      body.textContent = e.target.result;
      body.style.padding = '16px';
      body.style.whiteSpace = 'pre-wrap';
    };
    reader.readAsText(file);
  } else {
    body.textContent = `[${ext.toUpperCase()} — preview not available]\n\nFile: ${file.name}\nSize: ${fmtSize(file.size)}`;
    body.style.padding = '16px';
    body.style.whiteSpace = 'pre-wrap';
  }

  $('previewPanel').classList.add('open');
}

function closePreview() {
  $('previewPanel').classList.remove('open');
  $('previewBody').style.padding = '16px';
  S.previewFile = null;
}

// ── Upload modal ───────────────────────────────────────
function openModal() {
  $('uploadModal').hidden = false;
  S.pendingFiles = [];
  renderModalFileList();
  $('modalStatus').textContent = '';
  $('modalFileInput').value = '';
}

function closeModal() {
  $('uploadModal').hidden = true;
  S.pendingFiles = [];
}

function renderModalFileList() {
  $('modalFileList').innerHTML = S.pendingFiles.map(f =>
    `<div class="modal-file-item">
       <span>📄 ${escHtml(f.name)}</span>
       <span>${fmtSize(f.size)}</span>
     </div>`
  ).join('');
}

async function doIndexAndAttach() {
  if (!S.pendingFiles.length) {
    $('modalStatus').style.color = 'var(--error)';
    $('modalStatus').textContent = 'Please select at least one file.';
    return;
  }
  const domain = $('uploadDomain').value;
  $('modalIndexBtn').disabled = true;
  $('modalStatus').style.color = 'var(--text-muted)';
  $('modalStatus').textContent = '⏳ Indexing into knowledge base…';

  const fd = new FormData();
  fd.append('domain', domain);
  for (const f of S.pendingFiles) fd.append('files', f);

  try {
    const res = await fetch('/api/kb/upload', { method: 'POST', body: fd }).then(r => r.json());
    const total = res.uploaded.reduce((s, u) => s + u.chunks_indexed, 0);
    $('modalStatus').style.color = 'var(--success)';
    $('modalStatus').textContent = `✅ ${res.uploaded.length} file(s) indexed — ${total} chunks added.`;

    for (const f of S.pendingFiles) S.attachedFiles.push({ file: f, domain: domain });
    renderAttachChips();

    if (S.pendingFiles.length > 0) {
      openPreview(S.pendingFiles[0]);
    }
    refreshDomainCounts();
    setTimeout(() => closeModal(), 1500);
  } catch (err) {
    $('modalStatus').style.color = 'var(--error)';
    $('modalStatus').textContent = `❌ ${err.message}`;
  }
  $('modalIndexBtn').disabled = false;
}

// ── Knowledge Base Tab ─────────────────────────────────
const KB_INFO = {
  AML: {
    summary: 'Covers the full applied machine learning curriculum — from foundational ML concepts to advanced deep learning architectures. Content is drawn from 21 lecture slide decks and a comprehensive ML textbook.',
    toc: [
      'Introduction to ML & Course Overview',
      'Classification & Regression',
      'Linear & Logistic Regression',
      'Softmax & Multi-class Classification',
      'Support Vector Machines (SVM)',
      'Decision Trees & Random Forests',
      'Dimensionality Reduction (PCA, t-SNE)',
      'Unsupervised Learning: K-Means & EM',
      'Deep Neural Networks — Fundamentals',
      'Training Deep Networks (Backprop, Optimizers)',
      'Convolutional Neural Networks (CNNs)',
      'Transformers & NLP',
      'Autoencoders, GANs & Diffusion Models',
      'Regularization & Polynomial Regression',
      'Model Evaluation & ML Project Workflow',
    ],
  },
  ADT: {
    summary: 'Covers relational databases, SQL, NoSQL, normalization, query optimization, and natural language to SQL (NL2SQL). Content is sourced from curated knowledge base documents.',
    toc: [
      'Relational Model & Database Design',
      'SQL — DDL, DML, JOINs, Aggregation',
      'Normalization (1NF → BCNF)',
      'ACID Properties & Transactions',
      'Indexing & Query Optimization',
      'NoSQL: Key-Value, Document, Column, Graph',
      'CAP Theorem & Distributed Databases',
      'Entity-Relationship (ER) Modeling',
      'Stored Procedures & Triggers',
      'NL2SQL — LLM-based Query Generation',
      'Concurrency Control & MVCC',
    ],
  },
  STAT: {
    summary: 'Covers both introductory and applied statistics — from descriptive stats and probability to hypothesis testing and inference. Content draws from 7 lecture decks and a full-length textbook on statistical inference with R.',
    toc: [
      'Descriptive Statistics: Mean, Median, Variance',
      'Probability Theory & Bayes\' Theorem',
      'Probability Distributions (Normal, Binomial, t, χ²)',
      'Central Limit Theorem',
      'Confidence Intervals',
      'Hypothesis Testing Framework',
      'p-Values & Statistical Significance',
      'Type I & Type II Errors',
      'z-Tests, t-Tests (one/two sample, paired)',
      'Chi-Square Tests & ANOVA',
      'Effect Size & Power Analysis',
      'Multiple Testing (Bonferroni, FDR)',
      'Non-Parametric Tests & Bootstrap Methods',
      'Simple & Multiple Linear Regression',
    ],
  },
  LLM: {
    summary: 'Covers the full modern LLM curriculum — transformer internals, pretraining, alignment, fine-tuning, prompting, RAG, and agents. Content is drawn from 11 graduate-level lecture decks and curated knowledge base documents.',
    toc: [
      'Introduction to Large Language Models',
      'Word Representations & Embeddings',
      'Language Modeling & Next-Token Prediction',
      'Attention Mechanism & Transformer Architecture',
      'Pretraining at Scale (Data, Compute, Scaling Laws)',
      'Instruction Tuning & Supervised Fine-Tuning (SFT)',
      'Prompting Techniques (Zero-Shot, Few-Shot, CoT)',
      'RLHF — Reinforcement Learning from Human Feedback',
      'DPO — Direct Preference Optimization',
      'Retrieval-Augmented Generation (RAG)',
      'LLM Agents, Tool Use & Multi-Agent Systems',
      'Hallucination, Safety & Prompt Injection',
      'Quantization & Efficient Inference',
      'LLM Evaluation (MMLU, RAGAS, LLM-as-Judge)',
    ],
  },
};

async function renderKBTab() {
  const container = $('kbContent');
  container.innerHTML = '<div style="color:var(--text-muted);padding:20px">Loading…</div>';

  let chunkCounts = {};
  try {
    const data = await apiFetch('/api/kb/status');
    for (const [d, info] of Object.entries(data)) chunkCounts[d] = info.chunk_count;
  } catch {}

  const domains = S.config?.domains || {};
  container.innerHTML = Object.entries(KB_INFO).map(([abbr, info]) => {
    const d = domains[abbr] || {};
    const color = d.color || '#999';
    const count = chunkCounts[abbr] || 0;
    return `
    <div class="kb-subject-card">
      <div class="kb-card-header">
        <span class="kb-domain-pill" style="background:${color}">${abbr}</span>
        <span class="kb-card-name">${d.name || abbr}</span>
        <span class="kb-chunk-badge">${count.toLocaleString()} chunks indexed</span>
      </div>
      <div class="kb-card-body">
        <div class="kb-summary">
          <div class="kb-section-label">About this Knowledge Base</div>
          <p class="kb-summary-text">${escHtml(info.summary)}</p>
        </div>
        <div class="kb-toc">
          <div class="kb-section-label">Topics Covered</div>
          <ul class="kb-toc-list">
            ${info.toc.map(t => `<li>${escHtml(t)}</li>`).join('')}
          </ul>
        </div>
      </div>
    </div>`;
  }).join('');
}

// ── Evaluation ─────────────────────────────────────────
async function loadEvalCases() {
  try {
    const data = await apiFetch('/api/evaluate/cases');
    $('singleTcSelect').innerHTML = data.test_cases
      .map(tc => `<option value="${tc.id}">${tc.id}: ${escHtml(tc.name)}</option>`).join('');
  } catch {}
}

async function runAllEvals() {
  const prog = $('evalProgress');
  prog.hidden = false;
  $('progressFill').style.width = '5%';
  $('progressLabel').textContent = 'Running all test cases…';
  $('runAllBtn').disabled = true;
  $('evalResults').innerHTML = '';
  $('evalSummary').hidden = true;

  try {
    const data = await fetch('/api/evaluate', { method: 'POST' }).then(r => r.json());
    $('progressFill').style.width = '100%';
    $('progressLabel').textContent = '✅ Complete';
    setTimeout(() => { prog.hidden = true; }, 2000);
    renderEvalSummary(data.stats);
    renderEvalResults(data.results);
  } catch (err) {
    $('progressLabel').textContent = `❌ ${err.message}`;
  }
  $('runAllBtn').disabled = false;
}

async function runSingleEval() {
  const tcId = $('singleTcSelect').value;
  if (!tcId) return;
  $('runSingleBtn').disabled = true;
  $('evalResults').innerHTML = `<div class="thinking"><div class="dot-bounce"><span></span><span></span><span></span></div> Running ${escHtml(tcId)}…</div>`;
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
    <div class="eval-stat"><div class="eval-stat-val">${stats.pass_rate}%</div><div class="eval-stat-label">Pass Rate (${stats.passed}/${stats.total})</div></div>
    <div class="eval-stat"><div class="eval-stat-val">${stats.intent_accuracy}%</div><div class="eval-stat-label">Intent Accuracy</div></div>
    <div class="eval-stat"><div class="eval-stat-val">${stats.domain_accuracy}%</div><div class="eval-stat-label">Domain Accuracy</div></div>
    <div class="eval-stat"><div class="eval-stat-val">${(stats.avg_quality_score||0).toFixed(2)}</div><div class="eval-stat-label">Avg Quality</div></div>`;
}

function renderEvalResults(results) {
  const el = $('evalResults');
  el.innerHTML = '';
  for (const r of results) {
    const card = document.createElement('div');
    card.className = 'eval-card';
    card.innerHTML = `
      <div class="eval-card-header" onclick="this.nextElementSibling.classList.toggle('open')">
        <span>${r.passed ? '✅' : '❌'}</span>
        <span class="eval-tc-name">${escHtml(r.test_case_id||'')} — ${escHtml(r.name||'')}</span>
        <span class="quality-chip">${r.quality_score ? pct(r.quality_score) : ''}</span>
        <span style="color:var(--text-muted)">▾</span>
      </div>
      <div class="eval-card-body">
        <div class="eval-meta-row">
          <div class="eval-meta"><div class="eval-meta-key">Intent</div><div class="eval-meta-val">${r.intent_match?'✅':'❌'} ${escHtml(r.actual_intent||'')}</div></div>
          <div class="eval-meta"><div class="eval-meta-key">Domain</div><div class="eval-meta-val">${r.domain_match?'✅':'❌'} ${escHtml((r.actual_domains||[]).join(', '))}</div></div>
          <div class="eval-meta"><div class="eval-meta-key">Quality</div><div class="eval-meta-val">${r.quality_score ? pct(r.quality_score) : 'N/A'}</div></div>
        </div>
        ${r.expected_behavior ? `<div class="eval-answer"><b>Expected:</b> ${escHtml(r.expected_behavior)}</div>` : ''}
        ${r.answer_preview ? `<div class="eval-answer"><b>Answer:</b> ${escHtml(r.answer_preview)}…</div>` : ''}
        ${r.error ? `<div style="color:var(--error);font-size:12px">${escHtml(r.error)}</div>` : ''}
      </div>`;
    el.appendChild(card);
  }
}

// ── Events ─────────────────────────────────────────────
function bindEvents() {
  // Tabs
  document.querySelectorAll('.tab[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      $(`tab-${btn.dataset.tab}`).classList.add('active');
      if (btn.dataset.tab === 'kb') renderKBTab();
    });
  });

  // Sidebar toggle
  $('sidebarToggle').addEventListener('click', () => $('sidebar').classList.toggle('collapsed'));

  // Send
  $('sendBtn').addEventListener('click', sendMessage);
  $('queryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  $('queryInput').addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 140) + 'px';
  });

  // Sample prompts
  $('samplePrompts').addEventListener('click', e => {
    if (e.target.classList.contains('sample-btn')) {
      $('queryInput').value = e.target.dataset.prompt;
      sendMessage();
    }
  });

  // New session buttons
  $('newSessionBtn').addEventListener('click', startNewSession);
  $('newSessionBtn2').addEventListener('click', startNewSession);

  // Sliders
  $('topK').addEventListener('input', () => $('topKVal').textContent = $('topK').value);
  $('rerankK').addEventListener('input', () => $('rerankKVal').textContent = $('rerankK').value);
  $('conf').addEventListener('input', () => $('confVal').textContent = parseFloat($('conf').value).toFixed(2));

  // Attach chips — remove or click to preview
  $('inputAttachments').addEventListener('click', async e => {
    const removeBtn = e.target.closest('.attach-chip-remove');
    if (removeBtn) {
      const idx = parseInt(removeBtn.dataset.idx);
      S.attachedFiles.splice(idx, 1);
      renderAttachChips();
      if (!S.attachedFiles.length) closePreview();
      return;
    }
    const chip = e.target.closest('.attach-chip');
    if (chip) {
      const idx = parseInt(chip.dataset.idx);
      const a = S.attachedFiles[idx];
      if (a) openPreview(a.file);
    }
  });

  // Attach button → open modal
  $('attachBtn').addEventListener('click', openModal);

  // Modal
  $('modalClose').addEventListener('click', closeModal);
  $('modalCancelBtn').addEventListener('click', closeModal);
  $('uploadModal').addEventListener('click', e => { if (e.target === $('uploadModal')) closeModal(); });
  $('browseLink').addEventListener('click', () => $('modalFileInput').click());
  $('uploadDropZone').addEventListener('click', () => $('modalFileInput').click());
  $('modalFileInput').addEventListener('change', e => {
    S.pendingFiles = Array.from(e.target.files);
    renderModalFileList();
  });

  const dz = $('uploadDropZone');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragover');
    S.pendingFiles = Array.from(e.dataTransfer.files).filter(f => /\.(pdf|txt|md|docx)$/i.test(f.name));
    renderModalFileList();
  });
  $('modalIndexBtn').addEventListener('click', doIndexAndAttach);

  // Preview close
  $('previewClose').addEventListener('click', closePreview);

  // Evaluation
  $('runAllBtn').addEventListener('click', runAllEvals);
  $('runSingleBtn').addEventListener('click', runSingleEval);
}

// ── Boot ───────────────────────────────────────────────
init().catch(console.error);
