'use strict';

// ── State ──────────────────────────────────────────────
const S = {
  sessionId:          null,
  config:             null,
  domainColors:       {},
  pendingFiles:       [],   // files staged in the modal, not yet indexed
  attachedFiles:      [],  // files already indexed, shown as chips
  previewFile:        null,
  chatHistory:        [],   // [{role, content}] for current session context
  editingMsgId:       null, // DB message id of the user message being edited
  editingRemovedEls:  [],   // DOM elements removed during edit (for cancel restore)
  editingHistorySnap: [],   // chatHistory snapshot before edit (for cancel restore)
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

function applyKatex(el) {
  if (!el || !window.renderMathInElement) return;
  renderMathInElement(el, {
    delimiters: [
      { left: '$$', right: '$$', display: true  },
      { left: '$',  right: '$',  display: false },
      { left: '\\[', right: '\\]', display: true  },
      { left: '\\(', right: '\\)', display: false },
    ],
    throwOnError: false,
    errorColor: '#c00',
  });
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
  ssLoadSessions();
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
  S.chatHistory = [];
  try {
    const data = await apiFetch(`/api/sessions/${sid}`);
    if (data.messages && data.messages.length > 0) {
      for (const m of data.messages) {
        if (m.role === 'user') appendUserBubble(m.content, m.id);
        else appendAssistantBubble(m.content, null, false);
        S.chatHistory.push({ role: m.role, content: m.content });
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

function appendUserBubble(text, msgId = null) {
  const d = document.createElement('div');
  d.className = 'message user';
  if (msgId) d.dataset.msgId = msgId;
  d.innerHTML = `
    <div class="avatar">U</div>
    <div class="bubble-wrap">
      <div class="bubble">${escHtml(text)}</div>
      <button class="edit-msg-btn" title="Edit message" data-text="${escHtml(text)}"${msgId ? ` data-msg-id="${msgId}"` : ''}>✏️</button>
    </div>`;
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
  const sourcesHtml = result?.sources?.length ? buildSourcesHtml(result.sources) : '';

  d.innerHTML = `
    <div class="avatar">🎓</div>
    <div style="flex:1;min-width:0">
      ${meta ? `<div class="msg-meta">${meta}</div>` : ''}
      <div class="bubble">${renderMd(text)}</div>
      ${sourcesHtml}
      ${debugHtml}
    </div>`;
  $('messages').appendChild(d);
  hljs.highlightAll();
  applyKatex(d);
  scrollBottom();
}

function buildSourcesHtml(sources) {
  if (!sources || !sources.length) return '';

  // Group by domain so multi-domain responses are clear
  const byDomain = {};
  for (const s of sources) {
    if (!byDomain[s.domain]) byDomain[s.domain] = [];
    byDomain[s.domain].push(s);
  }

  const domainColors = S.domainColors || {};
  const items = sources.map(s => {
    const color = domainColors[s.domain] || '#888';
    const downloadUrl = `/api/documents/${encodeURIComponent(s.domain)}/${encodeURIComponent(s.filename)}`;
    // citation_label already includes page number — don't add it again
    const label = s.citation_label;
    return `
      <div class="source-item">
        <div class="source-item-header" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
          <span class="source-num">Source ${s.source_num}</span>
          <span class="source-domain-tag" style="background:${color}20;color:${color};border:1px solid ${color}40">${escHtml(s.domain)}</span>
          <span class="source-label" title="${escHtml(label)}">${escHtml(label)}</span>
          <a class="source-download" href="${downloadUrl}" download title="Download ${escHtml(s.filename)}" onclick="event.stopPropagation()">⬇ PDF</a>
          <span class="source-expand-arrow">▾</span>
        </div>
        <div class="source-item-text">${escHtml(s.text)}</div>
      </div>`;
  }).join('');

  const total = sources.length;
  return `
    <div class="sources-panel">
      <div class="sources-toggle" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
        📎 <span>${total} source${total !== 1 ? 's' : ''} retrieved</span>
        <span style="font-size:10px;color:var(--text-muted);margin-left:4px">— click to expand and verify grounding</span>
        <span class="sources-toggle-arrow">▾</span>
      </div>
      <div class="sources-body">${items}</div>
    </div>`;
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
  // If editing, truncate DB from that message onward (DOM already cleaned on edit click)
  const truncateMsgId = S.editingMsgId;
  S.editingMsgId = null;
  S.editingRemovedEls = [];
  S.editingHistorySnap = [];
  const indicator = $('editIndicator');
  if (indicator) indicator.remove();

  removeWelcome();

  if (truncateMsgId) {
    try {
      await fetch(`/api/sessions/${S.sessionId}/messages/${truncateMsgId}`, { method: 'DELETE' });
    } catch { /* best-effort */ }
  }

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
      chat_history: S.chatHistory.slice(-10),  // last 5 pairs
    });
    thinking.remove();

    // Attach returned DB ids to the just-appended user bubble
    if (result.user_message_id) {
      const lastUser = $('messages').querySelector('.message.user:last-of-type');
      if (lastUser) {
        lastUser.dataset.msgId = result.user_message_id;
        const editBtn = lastUser.querySelector('.edit-msg-btn');
        if (editBtn) editBtn.dataset.msgId = result.user_message_id;
      }
    }

    appendAssistantBubble(result.final_answer, result, $('debugToggle').checked);

    // Update in-memory chat history
    S.chatHistory.push({ role: 'user', content: query });
    S.chatHistory.push({ role: 'assistant', content: result.final_answer });

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

function cancelEditMode() {
  // Restore removed DOM elements
  if (S.editingRemovedEls.length) {
    const container = $('messages');
    for (const el of S.editingRemovedEls) container.appendChild(el);
    S.editingRemovedEls = [];
  }
  // Restore chat history snapshot
  if (S.editingHistorySnap.length) {
    S.chatHistory = S.editingHistorySnap;
    S.editingHistorySnap = [];
  }
  S.editingMsgId = null;
  const indicator = $('editIndicator');
  if (indicator) indicator.remove();
  const inp = $('queryInput');
  inp.value = '';
  inp.style.height = 'auto';
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

const FILE_ICONS = { '.pdf': '📄', '.md': '📝', '.txt': '📃', '.docx': '📋' };

function fmtBytes(b) {
  if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' KB';
  return b + ' B';
}

async function renderKBTab() {
  const container = $('kbContent');
  container.innerHTML = '<div style="color:var(--text-muted);padding:20px">Loading…</div>';

  let chunkCounts = {};
  let kbDocs = {};
  try {
    const [statusData, docsData] = await Promise.all([
      apiFetch('/api/kb/status'),
      apiFetch('/api/kb/documents'),
    ]);
    for (const [d, info] of Object.entries(statusData)) chunkCounts[d] = info.chunk_count;
    kbDocs = docsData;
  } catch {}

  const domains = S.config?.domains || {};
  container.innerHTML = Object.entries(KB_INFO).map(([abbr, info]) => {
    const d = domains[abbr] || {};
    const color = d.color || '#999';
    const count = chunkCounts[abbr] || 0;
    const docs = (kbDocs[abbr] || {}).documents || [];

    const docsHtml = docs.length
      ? docs.map(doc => {
          const icon = FILE_ICONS[doc.extension] || '📄';
          const url = `/api/documents/${encodeURIComponent(abbr)}/${encodeURIComponent(doc.filename)}`;
          return `
            <div class="kb-doc-item">
              <span class="kb-doc-icon">${icon}</span>
              <span class="kb-doc-name" title="${escHtml(doc.filename)}">${escHtml(doc.filename)}</span>
              <span class="kb-doc-meta">${fmtBytes(doc.size_bytes)}</span>
              <a class="kb-doc-download" href="${url}" download title="Download ${escHtml(doc.filename)}">⬇ Download</a>
            </div>`;
        }).join('')
      : '<div style="color:var(--text-muted);font-size:13px;padding:6px 0">No documents found in this knowledge base folder.</div>';

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
      <div class="kb-docs-section">
        <div class="kb-section-label">Source Documents (${docs.length} file${docs.length !== 1 ? 's' : ''})</div>
        <div class="kb-docs-list">${docsHtml}</div>
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
  $('progressLabel').textContent = 'Running all 10 test cases… (this may take 1–2 min)';
  $('runAllBtn').disabled = true;
  $('evalResults').innerHTML = '';
  $('evalSummary').hidden = true;

  try {
    const res = await fetch('/api/evaluate', { method: 'POST' });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Server error ${res.status}: ${text.slice(0, 200)}`);
    }
    const data = await res.json();
    $('progressFill').style.width = '100%';
    $('progressLabel').textContent = `✅ Complete — ${data.stats.passed}/${data.stats.total} passed`;
    setTimeout(() => { prog.hidden = true; }, 3000);
    renderEvalSummary(data.stats);
    renderEvalResults(data.results);
  } catch (err) {
    $('progressFill').style.width = '100%';
    $('progressFill').style.background = 'var(--error)';
    $('progressLabel').textContent = `❌ ${err.message}`;
    setTimeout(() => { $('progressFill').style.background = ''; prog.hidden = true; }, 5000);
  }
  $('runAllBtn').disabled = false;
}

async function runSingleEval() {
  const tcId = $('singleTcSelect').value;
  if (!tcId) return;
  $('runSingleBtn').disabled = true;
  $('evalResults').innerHTML = `<div class="thinking"><div class="dot-bounce"><span></span><span></span><span></span></div> Running ${escHtml(tcId)}…</div>`;
  try {
    const res = await fetch(`/api/evaluate/${tcId}`, { method: 'POST' });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Server error ${res.status}: ${text.slice(0, 200)}`);
    }
    const data = await res.json();
    renderEvalResults([data]);
  } catch (err) {
    $('evalResults').innerHTML = `<div style="color:var(--error);padding:12px">Error: ${escHtml(err.message)}</div>`;
  }
  $('runSingleBtn').disabled = false;
}

function renderEvalSummary(stats) {
  const el = $('evalSummary');
  el.hidden = false;

  // Category breakdown rows
  const catRows = Object.entries(stats.by_category || {}).map(([cat, s]) => {
    const catPct = s.total ? Math.round(s.passed / s.total * 100) : 0;
    const color = catPct === 100 ? 'var(--success,#4CAF50)' : catPct >= 50 ? 'var(--iu-crimson)' : 'var(--error,#F44336)';
    return `<span style="background:#F3F4F6;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;color:${color}">${cat}: ${s.passed}/${s.total}</span>`;
  }).join('');

  const aqScore = stats.avg_answer_quality || stats.avg_quality_score || 0;
  const aqColor = aqScore >= 0.80 ? 'var(--success,#4CAF50)' : aqScore >= 0.65 ? '#FF9800' : 'var(--iu-crimson)';
  const aqLabel = stats.answer_tests_count != null
    ? `Answer Quality (${stats.answer_tests_count} graded)`
    : 'Answer Quality';

  el.innerHTML = `
    <div class="eval-stat"><div class="eval-stat-val" style="color:${stats.pass_rate>=80?'var(--success,#4CAF50)':'var(--iu-crimson)'}">${stats.pass_rate}%</div><div class="eval-stat-label">Pass Rate (${stats.passed}/${stats.total})</div></div>
    <div class="eval-stat"><div class="eval-stat-val">${stats.intent_accuracy}%</div><div class="eval-stat-label">Intent Accuracy</div></div>
    <div class="eval-stat"><div class="eval-stat-val">${stats.domain_accuracy}%</div><div class="eval-stat-label">Domain Accuracy</div></div>
    <div class="eval-stat"><div class="eval-stat-val" style="color:${aqColor}">${Math.round(aqScore*100)}%</div><div class="eval-stat-label">${aqLabel}</div></div>
    ${catRows ? `<div style="grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px;align-items:center"><span style="font-size:11px;color:var(--text-muted);margin-right:4px">By category:</span>${catRows}</div>` : ''}`;
}

function renderEvalResults(results) {
  const el = $('evalResults');
  el.innerHTML = '';
  const catColors = {
    'single-domain': '#4CAF50', 'multi-domain': '#2196F3', 'edge-case': '#FF9800',
    'verification': '#9C27B0', 'citation': '#00BCD4', 'multi-turn': '#F44336',
  };
  for (const r of results) {
    const card = document.createElement('div');
    card.className = 'eval-card';
    const catColor = catColors[r.category] || '#888';
    const catBadge = r.category
      ? `<span style="font-size:10px;font-weight:600;background:${catColor}18;color:${catColor};border:1px solid ${catColor}40;border-radius:3px;padding:1px 6px">${escHtml(r.category)}</span>`
      : '';
    const qualColor = r.quality_score >= 0.75 ? 'var(--success,#4CAF50)' : r.quality_score >= 0.5 ? '#FF9800' : 'var(--error,#F44336)';
    // Mismatch highlights
    const mismatches = [];
    if (r.intent_match === false) mismatches.push(`Intent mismatch: got <b>${escHtml(r.actual_intent||'?')}</b>`);
    if (r.domain_match === false) mismatches.push(`Domain mismatch: got <b>[${escHtml((r.actual_domains||[]).join(', '))}]</b>`);
    const mismatchHtml = mismatches.length
      ? `<div style="background:#FFF3E0;border:1px solid #FFB74D;border-radius:4px;padding:8px 10px;font-size:12px;color:#E65100">${mismatches.join(' &nbsp;|&nbsp; ')}</div>`
      : '';

    card.innerHTML = `
      <div class="eval-card-header" onclick="this.nextElementSibling.classList.toggle('open')">
        <span style="font-size:15px">${r.passed ? '✅' : '❌'}</span>
        <span class="eval-tc-name">${escHtml(r.test_case_id||'')} — ${escHtml(r.name||'')}</span>
        ${catBadge}
        <span style="font-size:13px;font-weight:700;color:${qualColor}">${r.quality_score != null ? pct(r.quality_score) : ''}</span>
        <span style="color:var(--text-muted)">▾</span>
      </div>
      <div class="eval-card-body">
        <div class="eval-meta-row">
          <div class="eval-meta"><div class="eval-meta-key">Intent Check</div><div class="eval-meta-val">${r.intent_match?'✅ Pass':'❌ Fail'} &nbsp;<span style="font-weight:400;font-size:12px;color:var(--text-muted)">${escHtml(r.actual_intent||'—')}</span></div></div>
          <div class="eval-meta"><div class="eval-meta-key">Domain Check</div><div class="eval-meta-val">${r.domain_match?'✅ Pass':'❌ Fail'} &nbsp;<span style="font-weight:400;font-size:12px;color:var(--text-muted)">${escHtml((r.actual_domains||[]).join(', ')||'—')}</span></div></div>
          <div class="eval-meta"><div class="eval-meta-key">Quality Score</div><div class="eval-meta-val" style="color:${qualColor}">${r.quality_score != null ? pct(r.quality_score) : 'N/A'}</div></div>
        </div>
        ${mismatchHtml}
        ${r.expected_behavior ? `<div class="eval-answer"><b style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px">Expected Behavior</b><br>${escHtml(r.expected_behavior)}</div>` : ''}
        ${r.answer_preview ? `<div class="eval-answer"><b style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px">Answer Preview</b><br>${escHtml(r.answer_preview)}${r.answer_preview.length >= 500 ? '…' : ''}</div>` : ''}
        ${r.error ? `<div style="background:#FFEBEE;border:1px solid #EF9A9A;border-radius:4px;padding:8px 10px;color:#C62828;font-size:12px"><b>Error:</b> ${escHtml(r.error)}</div>` : ''}
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
      if (btn.dataset.tab === 'ss') ssLoadSessions();
    });
  });

  // Sidebar toggle
  $('sidebarToggle').addEventListener('click', () => $('sidebar').classList.toggle('collapsed'));

  // Edit message (delegated on message container)
  $('messages').addEventListener('click', e => {
    const btn = e.target.closest('.edit-msg-btn');
    if (!btn) return;
    const text = btn.dataset.text;
    const msgId = btn.dataset.msgId ? parseInt(btn.dataset.msgId) : null;

    // Snapshot history before removing anything (for cancel restore)
    S.editingHistorySnap = S.chatHistory.slice();

    // Remove the edited bubble and everything after it from the DOM immediately
    const allMsgs = Array.from($('messages').querySelectorAll('.message'));
    const editedEl = btn.closest('.message');
    let removing = false;
    S.editingRemovedEls = [];
    for (const el of allMsgs) {
      if (el === editedEl) removing = true;
      if (removing) {
        S.editingRemovedEls.push(el);
        el.remove();
      }
    }

    // Trim chatHistory to remove the edited turn and everything after
    // Find how many messages remain in DOM and keep that many turns
    const remaining = $('messages').querySelectorAll('.message').length;
    S.chatHistory = S.chatHistory.slice(0, remaining);

    S.editingMsgId = msgId;

    // Fill input
    const inp = $('queryInput');
    inp.value = text;
    inp.style.height = 'auto';
    inp.style.height = Math.min(inp.scrollHeight, 140) + 'px';
    inp.focus();

    // Show editing indicator
    let indicator = $('editIndicator');
    if (!indicator) {
      indicator = document.createElement('div');
      indicator.id = 'editIndicator';
      indicator.className = 'edit-indicator';
      inp.parentNode.insertBefore(indicator, inp);
    }
    indicator.innerHTML = `✏️ Editing message &nbsp;<button onclick="cancelEditMode()" style="background:none;border:none;cursor:pointer;color:var(--text-muted);font-size:12px">✕ Cancel</button>`;
  });

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

  // Self Study
  bindSSEvents();
}

// ══════════════════════════════════════════════════════
// SELF STUDY
// ══════════════════════════════════════════════════════

const SS = {
  activeSessionId: null,
  chatHistory: [],
  docs: [],              // current session's document list
  activeFilters: null,   // null = all docs; array of filenames = filtered
  pendingQuery: null,    // query waiting for clarification
};

// ── Session management ─────────────────────────────────
async function ssLoadSessions() {
  const data = await apiFetch('/api/self-study/sessions');
  const list = $('ssSessionsList');
  if (!data.sessions.length) {
    list.innerHTML = '<div class="ss-sidebar-empty">No sessions yet.<br>Create one to get started.</div>';
    return;
  }
  list.innerHTML = '';
  for (const s of data.sessions) {
    const isActive = s.ss_session_id === SS.activeSessionId;
    const card = document.createElement('div');
    card.className = 'ss-session-card' + (isActive ? ' active' : '');
    card.dataset.sid = s.ss_session_id;
    const age = ssRelativeTime(s.updated_at);
    card.innerHTML = `
      <div class="ss-session-card-name">
        ${isActive ? '<div class="ss-session-active-dot"></div>' : ''}
        ${escHtml(s.name)}
      </div>
      <div class="ss-session-card-meta">
        <span>${s.doc_count} file${s.doc_count !== 1 ? 's' : ''}</span>
        <span>·</span>
        <span>${(s.total_chunks || 0).toLocaleString()} chunks</span>
        <span>·</span>
        <span>${age}</span>
      </div>`;
    card.addEventListener('click', () => ssSelectSession(s.ss_session_id, s.name));
    list.appendChild(card);
  }
}

function ssRelativeTime(isoStr) {
  if (!isoStr) return '';
  const now = new Date();
  const then = new Date(isoStr.replace(' ', 'T') + 'Z');
  const diffMs = now - then;
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1)  return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)  return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

async function ssSelectSession(sid, name) {
  SS.activeSessionId = sid;
  SS.chatHistory = [];
  SS.docs = [];
  SS.activeFilters = null;
  SS.pendingQuery = null;

  $('ssEmpty').hidden = true;
  $('ssActive').removeAttribute('hidden');
  $('ssActiveName').textContent = name;

  await ssLoadSessions();
  await ssLoadSessionData(sid);
}

async function ssLoadSessionData(sid) {
  const data = await apiFetch(`/api/self-study/sessions/${sid}`);

  // Update header meta
  const s = data.session;
  const docs = data.documents || [];
  const totalChunks = docs.reduce((n, d) => n + (d.chunk_count || 0), 0);
  $('ssActiveMeta').textContent =
    `${docs.length} file${docs.length !== 1 ? 's' : ''} · ${totalChunks.toLocaleString()} chunks indexed` +
    (s.description ? ` · ${s.description}` : '');
  $('ssDocCount').textContent = `${docs.length} file${docs.length !== 1 ? 's' : ''}`;

  // Render documents
  ssRenderDocs(docs);

  // Render messages
  const msgs = data.messages || [];
  $('ssMessages').innerHTML = '';
  if (!msgs.length) {
    ssShowChatWelcome();
  } else {
    for (const m of msgs) {
      if (m.role === 'user') ssAppendUserBubble(m.content);
      else ssAppendAssistantBubble(m.content, null);
      SS.chatHistory.push({ role: m.role, content: m.content });
    }
  }
}

function ssShowChatWelcome() {
  $('ssMessages').innerHTML = `
    <div class="ss-chat-welcome" id="ssChatWelcome">
      <div class="welcome-icon" style="font-size:36px">📖</div>
      <h4>Ready to study</h4>
      <p>Upload documents on the left, then ask questions about them here.</p>
    </div>`;
}

// ── Documents ──────────────────────────────────────────
function ssGetFileIcon(ext) {
  const icons = { pdf: '📄', txt: '📝', md: '📋', docx: '📘' };
  return icons[(ext || '').replace('.', '')] || '📄';
}

function ssRenderDocs(docs) {
  SS.docs = docs;
  const list = $('ssDocsList');
  if (!docs.length) { list.innerHTML = ''; ssRebuildFilterBar(); return; }

  list.innerHTML = docs.map(d => {
    const ext = (d.file_type || '').replace('.', '').toUpperCase() || 'FILE';
    const size = d.file_size_bytes ? fmtSize(d.file_size_bytes) : '';
    const hasChunks = d.chunk_count > 0;
    const statusHtml = hasChunks
      ? `<div class="ss-doc-status ready">✓ Ready — ${d.chunk_count} chunks</div>`
      : `<div class="ss-doc-status warn">⚠️ No text extracted — file may be image-based or encrypted</div>`;
    return `
      <div class="ss-doc-card" data-doc-id="${d.id}">
        <div class="ss-doc-card-top">
          <span class="ss-doc-icon">${ssGetFileIcon(d.file_type)}</span>
          <div class="ss-doc-info">
            <div class="ss-doc-name" title="${escHtml(d.filename)}">${escHtml(d.filename)}</div>
            <div class="ss-doc-details">${ext}${size ? ' · ' + size : ''}</div>
          </div>
          <button class="ss-doc-remove" data-doc-id="${d.id}" data-filename="${escHtml(d.filename)}" title="Remove document">✕</button>
        </div>
        ${statusHtml}
      </div>`;
  }).join('');

  ssRebuildFilterBar();
}

function ssRebuildFilterBar() {
  const docsWithChunks = (SS.docs || []).filter(d => d.chunk_count > 0);
  const bar = $('ssFilterBar');
  const chips = $('ssFilterChips');

  if (docsWithChunks.length < 2) {
    bar.hidden = true;
    SS.activeFilters = null;
    return;
  }

  bar.hidden = false;

  // Initialise filters to all-selected if not set
  if (SS.activeFilters === null) {
    SS.activeFilters = docsWithChunks.map(d => d.filename);
  } else {
    // Prune any filters for docs that no longer exist
    const names = new Set(docsWithChunks.map(d => d.filename));
    SS.activeFilters = SS.activeFilters.filter(f => names.has(f));
    if (!SS.activeFilters.length) SS.activeFilters = docsWithChunks.map(d => d.filename);
  }

  chips.innerHTML = docsWithChunks.map(d => {
    const active = SS.activeFilters.includes(d.filename);
    return `<div class="ss-filter-chip ${active ? 'active' : ''}" data-fname="${escHtml(d.filename)}" title="${escHtml(d.filename)}">
      <span>${escHtml(d.filename)}</span>
    </div>`;
  }).join('');
}

async function ssRemoveDocument(docId, filename) {
  if (!confirm(`Remove "${filename}" from this session?`)) return;
  try {
    await fetch(`/api/self-study/sessions/${SS.activeSessionId}/documents/${docId}`, { method: 'DELETE' });
    await ssLoadSessionData(SS.activeSessionId);
    await ssLoadSessions();
  } catch (err) {
    alert(`Error removing document: ${err.message}`);
  }
}

// ── Upload ─────────────────────────────────────────────
async function ssHandleUpload(files) {
  if (!SS.activeSessionId || !files.length) return;
  const valid = Array.from(files).filter(f => /\.(pdf|txt|md|docx)$/i.test(f.name));
  if (!valid.length) { alert('Please upload PDF, TXT, MD, or DOCX files.'); return; }

  const prog = $('ssUploadProgress');
  const fill = $('ssProgressFill');
  const label = $('ssProgressLabel');

  prog.removeAttribute('hidden');
  fill.style.width = '15%';
  label.textContent = `Uploading ${valid.length} file${valid.length > 1 ? 's' : ''}…`;

  // Show pending cards
  const list = $('ssDocsList');
  const pendingIds = valid.map((f, i) => `ss-pending-${i}`);
  valid.forEach((f, i) => {
    const el = document.createElement('div');
    el.className = 'ss-pending-card';
    el.id = pendingIds[i];
    el.innerHTML = `<div class="ss-pending-spinner"></div><span>${escHtml(f.name)} — Indexing…</span>`;
    list.prepend(el);
  });

  const fd = new FormData();
  for (const f of valid) fd.append('files', f);

  try {
    fill.style.width = '40%';
    label.textContent = 'Embedding & indexing…';

    const res = await fetch(`/api/self-study/sessions/${SS.activeSessionId}/upload`, {
      method: 'POST', body: fd,
    }).then(r => r.json());

    fill.style.width = '100%';
    const total = res.uploaded.reduce((n, u) => n + u.chunks_indexed, 0);
    label.textContent = `✅ ${res.uploaded.length} file(s) indexed — ${total} chunks added`;

    setTimeout(async () => {
      prog.hidden = true;
      fill.style.width = '0%';
      await ssLoadSessionData(SS.activeSessionId);
      await ssLoadSessions();
    }, 1800);

  } catch (err) {
    fill.style.width = '0%';
    label.textContent = `❌ ${err.message}`;
    setTimeout(() => { prog.hidden = true; }, 2500);
  }

  // Clean up pending cards
  pendingIds.forEach(id => { const el = $(id); if (el) el.remove(); });
  $('ssFileInput').value = '';
}

// ── Chat ───────────────────────────────────────────────
function ssAppendUserBubble(text) {
  const d = document.createElement('div');
  d.className = 'message user';
  d.innerHTML = `
    <div class="avatar">U</div>
    <div class="bubble-wrap">
      <div class="bubble">${escHtml(text)}</div>
    </div>`;
  $('ssMessages').appendChild(d);
  ssScrollBottom();
}

function ssAppendAssistantBubble(text, result) {
  const d = document.createElement('div');
  d.className = 'message assistant';

  let meta = '';
  if (result?.quality_score != null) {
    const q = Math.round(result.quality_score * 100);
    const icon = q >= 70 ? '✅' : q >= 40 ? '⚠️' : '❌';
    meta += `<span class="quality-chip">${icon} Quality ${q}%</span>`;
    if (result.verification_revised) meta += ` <span class="quality-chip">✏️ Revised</span>`;
  }

  let sourcesHtml = '';
  if (result?.sources?.length) {
    sourcesHtml = `<div class="ss-source-chips">
      ${result.sources.map(s => `<span class="ss-source-chip">📄 ${escHtml(s)}</span>`).join('')}
    </div>`;
  }

  d.innerHTML = `
    <div class="avatar">📖</div>
    <div style="flex:1;min-width:0">
      ${meta ? `<div class="msg-meta">${meta}</div>` : ''}
      <div class="bubble">${renderMd(text)}</div>
      ${sourcesHtml}
    </div>`;
  $('ssMessages').appendChild(d);
  hljs.highlightAll();
  applyKatex(d);
  ssScrollBottom();
}

function ssAppendThinking() {
  const d = document.createElement('div');
  d.className = 'message assistant';
  d.innerHTML = `<div class="avatar">📖</div>
    <div class="thinking">
      <div class="dot-bounce"><span></span><span></span><span></span></div>
      Thinking…
    </div>`;
  $('ssMessages').appendChild(d);
  ssScrollBottom();
  return d;
}

function ssScrollBottom() {
  const c = $('ssMessages');
  requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
}

function ssIsGeneralQuery(query, docs) {
  const q = query.toLowerCase();
  // Check if any filename is explicitly referenced
  const mentionsFile = docs.some(d => q.includes(d.filename.toLowerCase().replace(/\.[^.]+$/, '')));
  if (mentionsFile) return false;
  // General indicator keywords
  const generalKws = ['all', 'any', 'every', 'across', 'materials', 'documents', 'files',
    'uploaded', 'everything', 'overall', 'summary', 'overview', 'important', 'key'];
  return generalKws.some(kw => q.includes(kw)) || query.split(' ').length < 8;
}

async function ssSendMessage() {
  const inp = $('ssQueryInput');
  const query = inp.value.trim();
  if (!query || !SS.activeSessionId) return;

  const docsWithChunks = (SS.docs || []).filter(d => d.chunk_count > 0);

  // Show clarification card if: 2+ distinct docs with chunks, general query, no specific filter active
  const allSelected = SS.activeFilters === null || SS.activeFilters.length === docsWithChunks.length;
  if (docsWithChunks.length >= 2 && allSelected && ssIsGeneralQuery(query, docsWithChunks)) {
    inp.value = '';
    inp.style.height = 'auto';
    SS.pendingQuery = query;
    const welcome = document.getElementById('ssChatWelcome');
    if (welcome) welcome.remove();
    ssAppendUserBubble(query);
    ssShowClarificationCard(query, docsWithChunks);
    return;
  }

  await ssRunQuery(query, SS.activeFilters && SS.activeFilters.length < docsWithChunks.length ? SS.activeFilters : null);
}

function ssShowClarificationCard(query, docs) {
  const card = document.createElement('div');
  card.className = 'message assistant';
  card.id = 'ssClarifyCard';
  const btnAll = `<button class="ss-clarify-btn all-docs" data-filter="__all__">All Documents</button>`;
  const btnDocs = docs.map(d =>
    `<button class="ss-clarify-btn" data-filter="${escHtml(d.filename)}" title="${escHtml(d.filename)}">📄 ${escHtml(d.filename)}</button>`
  ).join('');
  card.innerHTML = `
    <div class="avatar">📖</div>
    <div style="flex:1;min-width:0">
      <div class="ss-clarify-card">
        <div class="ss-clarify-title">🤔 Multiple Documents Detected</div>
        <div class="ss-clarify-desc">
          Your session has <strong>${docs.length} documents</strong> covering different topics.
          Which would you like me to focus on for: <em>"${escHtml(query)}"</em>
        </div>
        <div class="ss-clarify-options">
          ${btnAll}
          ${btnDocs}
        </div>
      </div>
    </div>`;
  $('ssMessages').appendChild(card);
  ssScrollBottom();
}

async function ssRunQuery(query, sourceFilter) {
  const inp = $('ssQueryInput');
  inp.value = '';
  inp.style.height = 'auto';
  $('ssSendBtn').disabled = true;

  const welcome = document.getElementById('ssChatWelcome');
  if (welcome) welcome.remove();

  const clarifyCard = $('ssClarifyCard');
  if (clarifyCard) clarifyCard.remove();

  // Show user bubble on direct send. Clarification path already showed it in ssSendMessage.
  if (!SS.pendingQuery) ssAppendUserBubble(query);

  const thinking = ssAppendThinking();

  try {
    const result = await post('/api/self-study/chat', {
      query,
      ss_session_id: SS.activeSessionId,
      model: S.config?.model || S.config?.default_model || 'gemini-2.5-flash',
      top_k: parseInt($('topK').value),
      rerank_top_k: parseInt($('rerankK').value),
      confidence_threshold: parseFloat($('conf').value),
      enable_verification: $('ssVerifyToggle').checked,
      chat_history: SS.chatHistory.slice(-10),
      source_filter: sourceFilter || null,
    });
    thinking.remove();
    ssAppendAssistantBubble(result.final_answer, result);
    SS.chatHistory.push({ role: 'user', content: query });
    SS.chatHistory.push({ role: 'assistant', content: result.final_answer });
  } catch (err) {
    thinking.remove();
    ssAppendAssistantBubble(`⚠️ Error: ${escHtml(err.message)}`, null);
  }

  $('ssSendBtn').disabled = false;
  inp.focus();
  SS.pendingQuery = null;
}

// ── Create session modal ───────────────────────────────
function ssOpenCreateModal() {
  $('ssCreateModal').hidden = false;
  $('ssSessionName').value = '';
  $('ssSessionDesc').value = '';
  $('ssModalStatus').textContent = '';
  setTimeout(() => $('ssSessionName').focus(), 50);
}

function ssCloseCreateModal() {
  $('ssCreateModal').hidden = true;
}

async function ssDoCreateSession() {
  const name = $('ssSessionName').value.trim();
  if (!name) {
    $('ssModalStatus').style.color = 'var(--error)';
    $('ssModalStatus').textContent = 'Please enter a session name.';
    return;
  }
  const desc = $('ssSessionDesc').value.trim() || null;
  $('ssModalCreateBtn').disabled = true;
  $('ssModalStatus').style.color = 'var(--text-muted)';
  $('ssModalStatus').textContent = 'Creating…';

  try {
    const res = await post('/api/self-study/sessions', { name, description: desc });
    ssCloseCreateModal();
    await ssLoadSessions();
    await ssSelectSession(res.ss_session_id, res.name);
  } catch (err) {
    $('ssModalStatus').style.color = 'var(--error)';
    $('ssModalStatus').textContent = `Error: ${err.message}`;
  }
  $('ssModalCreateBtn').disabled = false;
}

// ── Delete session ─────────────────────────────────────
async function ssDeleteActiveSession() {
  if (!SS.activeSessionId) return;
  const name = $('ssActiveName').textContent;
  if (!confirm(`Delete study session "${name}"?\n\nAll documents and chat history will be permanently removed.`)) return;

  try {
    await fetch(`/api/self-study/sessions/${SS.activeSessionId}`, { method: 'DELETE' });
    SS.activeSessionId = null;
    SS.chatHistory = [];
    $('ssActive').hidden = true;
    $('ssEmpty').hidden = false;
    await ssLoadSessions();
  } catch (err) {
    alert(`Error deleting session: ${err.message}`);
  }
}

// ── Bind Self Study events ─────────────────────────────
function bindSSEvents() {
  // New session buttons
  $('ssNewBtn').addEventListener('click', ssOpenCreateModal);
  $('ssCreateFirstBtn').addEventListener('click', ssOpenCreateModal);

  // Modal
  $('ssModalClose').addEventListener('click', ssCloseCreateModal);
  $('ssModalCancelBtn').addEventListener('click', ssCloseCreateModal);
  $('ssCreateModal').addEventListener('click', e => { if (e.target === $('ssCreateModal')) ssCloseCreateModal(); });
  $('ssModalCreateBtn').addEventListener('click', ssDoCreateSession);
  $('ssSessionName').addEventListener('keydown', e => { if (e.key === 'Enter') ssDoCreateSession(); });

  // Delete session
  $('ssDeleteSessionBtn').addEventListener('click', ssDeleteActiveSession);

  // Upload zone
  const dz = $('ssUploadZone');
  $('ssBrowseLink').addEventListener('click', e => { e.stopPropagation(); $('ssFileInput').click(); });
  dz.addEventListener('click', () => $('ssFileInput').click());
  $('ssFileInput').addEventListener('change', e => ssHandleUpload(e.target.files));
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('dragover');
    ssHandleUpload(e.dataTransfer.files);
  });

  // Document remove (delegated)
  $('ssDocsList').addEventListener('click', e => {
    const btn = e.target.closest('.ss-doc-remove');
    if (btn) ssRemoveDocument(parseInt(btn.dataset.docId), btn.dataset.filename);
  });

  // Filter chip toggles
  $('ssFilterChips').addEventListener('click', e => {
    const chip = e.target.closest('.ss-filter-chip');
    if (!chip) return;
    const fname = chip.dataset.fname;
    const docsWithChunks = (SS.docs || []).filter(d => d.chunk_count > 0);
    if (!SS.activeFilters) SS.activeFilters = docsWithChunks.map(d => d.filename);

    if (SS.activeFilters.includes(fname)) {
      // Deselect — keep at least one selected
      const next = SS.activeFilters.filter(f => f !== fname);
      SS.activeFilters = next.length ? next : SS.activeFilters;
    } else {
      SS.activeFilters = [...SS.activeFilters, fname];
    }
    // Re-render chip states
    document.querySelectorAll('.ss-filter-chip').forEach(c => {
      c.classList.toggle('active', SS.activeFilters.includes(c.dataset.fname));
    });
  });

  // Clarification card option clicks (delegated on messages container)
  $('ssMessages').addEventListener('click', e => {
    const btn = e.target.closest('.ss-clarify-btn');
    if (!btn || !SS.pendingQuery) return;
    const filter = btn.dataset.filter;
    const sourceFilter = filter === '__all__' ? null : [filter];
    ssRunQuery(SS.pendingQuery, sourceFilter);
  });

  // Chat
  $('ssSendBtn').addEventListener('click', ssSendMessage);
  $('ssQueryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ssSendMessage(); }
  });
  $('ssQueryInput').addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 140) + 'px';
  });
}

// ── Boot ───────────────────────────────────────────────
init().catch(console.error);
