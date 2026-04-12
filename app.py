"""
EduPilot — Multi-Agent Educational RAG System
==============================================
Streamlit entry point. Run with:

    streamlit run app.py

Environment variables required (add to .env or Streamlit Cloud secrets):
    GROQ_API_KEY=gsk_...
    PINECONE_API_KEY=...
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import streamlit as st

# Load .env before anything else (database/config imports read env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPilot — AI Educational Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Main container */
.block-container { padding-top: 1rem; max-width: 1200px; }

/* Domain badges */
.badge-AML  { background:#4CAF50; color:white; padding:2px 8px;
               border-radius:10px; font-size:11px; font-weight:bold; }
.badge-ADT  { background:#2196F3; color:white; padding:2px 8px;
               border-radius:10px; font-size:11px; font-weight:bold; }
.badge-STAT { background:#FF9800; color:white; padding:2px 8px;
               border-radius:10px; font-size:11px; font-weight:bold; }

/* Debug box */
.debug-box { background:#f8f9fa; border:1px solid #dee2e6;
              border-radius:6px; padding:12px; margin:4px 0;
              font-size:13px; font-family:monospace; }

/* Score bars */
.score-bar { height:8px; border-radius:4px; background:#4CAF50;
              display:inline-block; }

/* Citation */
.citation-text { font-size:12px; color:#6c757d; font-style:italic; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Lazy imports of pipeline modules (keeps startup fast)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline modules…")
def _import_pipeline():
    from config import DOMAINS, AVAILABLE_MODELS, DEFAULT_MODEL
    from retriever import get_retriever, initialize_all_retrievers
    from router import classify_query, should_ask_for_clarification, get_clarification_message, get_out_of_domain_message
    from query_splitter import split_query
    from reranker import rerank, score_summary
    from synthesizer import generate_domain_answer, synthesize_answers
    from verifier import verify_answer, get_final_answer
    from evaluation import TEST_CASES, run_evaluation, run_all_evaluations, summary_stats
    from utils import PipelineResult
    return {
        "DOMAINS": DOMAINS,
        "AVAILABLE_MODELS": AVAILABLE_MODELS,
        "DEFAULT_MODEL": DEFAULT_MODEL,
        "get_retriever": get_retriever,
        "initialize_all_retrievers": initialize_all_retrievers,
        "classify_query": classify_query,
        "should_ask_for_clarification": should_ask_for_clarification,
        "get_clarification_message": get_clarification_message,
        "get_out_of_domain_message": get_out_of_domain_message,
        "split_query": split_query,
        "rerank": rerank,
        "score_summary": score_summary,
        "generate_domain_answer": generate_domain_answer,
        "synthesize_answers": synthesize_answers,
        "verify_answer": verify_answer,
        "get_final_answer": get_final_answer,
        "TEST_CASES": TEST_CASES,
        "run_evaluation": run_evaluation,
        "run_all_evaluations": run_all_evaluations,
        "summary_stats": summary_stats,
        "PipelineResult": PipelineResult,
    }


@st.cache_resource(show_spinner="Initializing knowledge bases…")
def _init_retrievers():
    pipeline = _import_pipeline()
    counts = pipeline["initialize_all_retrievers"]()
    return counts


# ---------------------------------------------------------------------------
# Full pipeline function
# ---------------------------------------------------------------------------
def run_pipeline(
    query: str,
    model: str,
    top_k: int,
    rerank_top_k: int,
    enable_verification: bool,
    confidence_threshold: float,
    manual_domains: list[str] | None = None,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Execute the full EduPilot pipeline and return a results dict.
    This dict drives both the UI display and the evaluation module.
    """
    p = _import_pipeline()
    debug: dict = {}

    # ---- Step 1: Route & classify ----
    router_result = p["classify_query"](query, model=model, chat_history=chat_history)
    debug["router"] = {
        "intent_type": router_result.intent_type,
        "domains": router_result.domains,
        "is_course_related": router_result.is_course_related,
        "needs_clarification": router_result.needs_clarification,
        "reasoning": router_result.reasoning,
    }

    # Override with manually selected domains if provided
    effective_domains = manual_domains if manual_domains else router_result.domains

    # ---- Early exits ----
    if not router_result.is_course_related:
        return {
            "query": query,
            "final_answer": p["get_out_of_domain_message"](),
            "intent_type": router_result.intent_type,
            "detected_domains": [],
            "sub_questions": [],
            "domain_answers": [],
            "is_course_related": False,
            "needs_clarification": False,
            "clarification_hint": None,
            "quality_score": 0.0,
            "verification_issues": [],
            "debug": debug,
        }

    if p["should_ask_for_clarification"](router_result) and not manual_domains:
        return {
            "query": query,
            "final_answer": p["get_clarification_message"](router_result),
            "intent_type": router_result.intent_type,
            "detected_domains": [],
            "sub_questions": [],
            "domain_answers": [],
            "is_course_related": True,
            "needs_clarification": True,
            "clarification_hint": router_result.clarification_hint,
            "quality_score": 0.0,
            "verification_issues": [],
            "debug": debug,
        }

    # ---- Step 2: Decompose query ----
    sub_questions = p["split_query"](
        query=query,
        intent_type=router_result.intent_type,
        detected_domains=effective_domains,
        model=model,
    )
    debug["sub_questions"] = sub_questions

    # ---- Steps 3–5: Retrieve → Rerank → Generate per domain ----
    domain_answers = []
    debug["retrieval"] = []

    for sq in sub_questions:
        domain = sq["domain"]
        question = sq["question"]

        # Retrieve
        retriever = p["get_retriever"](domain)
        raw_chunks = retriever.retrieve(question, top_k=top_k)

        # Rerank
        reranked = p["rerank"](
            query=question,
            chunks=raw_chunks,
            top_k=rerank_top_k,
            confidence_threshold=confidence_threshold,
        )

        debug["retrieval"].append({
            "domain": domain,
            "question": question,
            "raw_count": len(raw_chunks),
            "reranked_count": len(reranked),
            "score_summary": p["score_summary"](reranked),
            "chunks": [
                {
                    "text": c.text[:300],
                    "source": c.citation_label(),
                    "rerank_score": round(c.rerank_score, 4),
                    "semantic_score": round(c.semantic_score, 4),
                    "bm25_score": round(c.bm25_score, 4),
                }
                for c in reranked
            ],
        })

        # Domain agent answer
        da = p["generate_domain_answer"](
            sub_question=question,
            domain=domain,
            retrieved_chunks=reranked,
            model=model,
        )
        domain_answers.append(da)

    debug["domain_answers"] = [
        {"domain": da.domain, "question": da.sub_question, "preview": da.answer[:300]}
        for da in domain_answers
    ]

    # ---- Step 6: Cross-domain synthesis ----
    synthesized = p["synthesize_answers"](
        original_query=query,
        domain_answers=domain_answers,
        model=model,
    )
    debug["synthesized_preview"] = synthesized[:500]

    # ---- Step 7: Verification ----
    verification = p["verify_answer"](
        original_query=query,
        sub_questions=sub_questions,
        domain_answers=domain_answers,
        synthesized_answer=synthesized,
        model=model,
        enabled=enable_verification,
    )
    debug["verification"] = {
        "is_satisfactory": verification.is_satisfactory,
        "quality_score": verification.quality_score,
        "coverage_score": verification.coverage_score,
        "grounding_score": verification.grounding_score,
        "issues": verification.issues,
        "missing_topics": verification.missing_topics,
        "was_revised": verification.revised_answer is not None,
        "skipped": verification.skipped,
    }

    final_answer = p["get_final_answer"](synthesized, verification)

    return {
        "query": query,
        "final_answer": final_answer,
        "synthesized_answer": synthesized,
        "intent_type": router_result.intent_type,
        "detected_domains": effective_domains,
        "sub_questions": sub_questions,
        "domain_answers": domain_answers,
        "is_course_related": True,
        "needs_clarification": False,
        "clarification_hint": None,
        "quality_score": verification.quality_score,
        "verification_issues": verification.issues,
        "verification_revised": verification.revised_answer is not None,
        "debug": debug,
    }


# ---------------------------------------------------------------------------
# Database init (SQLite — creates tables if they don't exist)
# ---------------------------------------------------------------------------
import database as db
db.init_db()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    db.ensure_session(st.session_state.session_id)

if "chat_history" not in st.session_state:
    # Restore this session's messages from SQLite on first load
    saved = db.get_session_messages(st.session_state.session_id)
    st.session_state.chat_history = [
        {"role": m["role"], "content": m["content"]} for m in saved
    ]
if "debug_results" not in st.session_state:
    st.session_state.debug_results = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

# ---------------------------------------------------------------------------
# Check API keys
# ---------------------------------------------------------------------------
api_key = os.environ.get("GROQ_API_KEY", "")
pinecone_key = os.environ.get("PINECONE_API_KEY", "")
if not api_key:
    st.error(
        "⚠️ **GROQ_API_KEY not set.**  "
        "Add it to your `.env` file or Streamlit Cloud secrets:  \n"
        "`GROQ_API_KEY=gsk_...`"
    )
    st.stop()
if not pinecone_key:
    st.error(
        "⚠️ **PINECONE_API_KEY not set.**  "
        "Add it to your `.env` file:  \n"
        "`PINECONE_API_KEY=your-pinecone-key`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load pipeline (cached)
# ---------------------------------------------------------------------------
with st.spinner("Initializing EduPilot…"):
    try:
        pipeline = _import_pipeline()
        chunk_counts = _init_retrievers()
        DOMAINS = pipeline["DOMAINS"]
        AVAILABLE_MODELS = pipeline["AVAILABLE_MODELS"]
        DEFAULT_MODEL = pipeline["DEFAULT_MODEL"]
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# ████  SIDEBAR  ████
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("EduPilot")
    st.caption("Multi-Agent Educational AI Tutor")
    st.divider()

    # --- Model settings ---
    st.subheader("⚙️ Model Settings")
    selected_model = st.selectbox(
        "Claude Model",
        AVAILABLE_MODELS,
        index=0,
        help="Select the Claude model for all pipeline stages.",
    )
    top_k = st.slider("Retrieval Top-K", 2, 10, 5, help="Chunks fetched per domain.")
    rerank_top_k = st.slider(
        "Rerank Top-K", 1, 5, 3,
        help="Final chunks passed to the domain agent after reranking."
    )
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.20, 0.05,
        help="Minimum rerank score to include a chunk."
    )
    enable_verification = st.checkbox(
        "Enable Verification Step", value=True,
        help="Run a second LLM pass to verify and optionally revise the answer."
    )
    show_debug = st.checkbox("Show Debug Panel", value=False)

    st.divider()

    # --- Domain selection ---
    st.subheader("🗂️ Domain Routing")
    domain_mode = st.radio(
        "Routing Mode",
        ["Auto-Route (recommended)", "Manual Domain Selection"],
        index=0,
    )
    manual_domains: list[str] | None = None
    if domain_mode == "Manual Domain Selection":
        manual_domains = st.multiselect(
            "Force Domain(s)",
            list(DOMAINS.keys()),
            format_func=lambda d: f"{d} — {DOMAINS[d]['name']}",
        )

    st.divider()

    # --- Document upload ---
    st.subheader("📂 Upload Documents")
    upload_domain = st.selectbox(
        "Target Domain",
        list(DOMAINS.keys()),
        format_func=lambda d: f"{d} — {DOMAINS[d]['name']}",
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs, TXT, MD, or DOCX",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx"],
    )
    if st.button("📥 Index Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("No files selected.")
        else:
            with st.spinner(f"Indexing {len(uploaded_files)} file(s) into {upload_domain}…"):
                domain_cfg = DOMAINS[upload_domain]
                kb_path = Path(domain_cfg["knowledge_base_path"])
                kb_path.mkdir(parents=True, exist_ok=True)
                saved_paths = []
                file_infos = []
                for uf in uploaded_files:
                    raw = uf.read()
                    dest = kb_path / uf.name
                    dest.write_bytes(raw)
                    saved_paths.append(str(dest))
                    file_infos.append({
                        "name": uf.name,
                        "size": len(raw),
                        "type": Path(uf.name).suffix.lower(),
                    })
                retriever = pipeline["get_retriever"](upload_domain)
                n = retriever.add_documents(saved_paths)
                # Persist upload metadata to SQLite
                chunks_per_file = max(1, n // len(saved_paths)) if saved_paths else 0
                for fi in file_infos:
                    db.save_uploaded_doc(
                        filename=fi["name"],
                        domain=upload_domain,
                        file_type=fi["type"],
                        chunk_count=chunks_per_file,
                        file_size_bytes=fi["size"],
                    )
                st.success(f"✅ Indexed {n} chunks into {upload_domain}.")

    st.divider()

    # --- Knowledge base status ---
    st.subheader("📊 Knowledge Base Status")
    for domain, cfg in DOMAINS.items():
        r = pipeline["get_retriever"](domain)
        count = r.document_count()
        color = cfg["color"]
        st.markdown(
            f'<span style="color:{color}; font-weight:bold;">{domain}</span> '
            f'— {cfg["name"][:20]}…  \n'
            f'**{count}** chunks indexed',
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Session management ---
    st.subheader("💬 Session History")
    sessions = db.list_sessions(limit=10)
    if sessions:
        for s in sessions:
            sid = s["session_id"]
            label = s["title"] or f"Session {sid[:8]}…"
            msg_count = s.get("message_count", 0)
            is_current = sid == st.session_state.session_id
            btn_label = f"{'▶ ' if is_current else ''}{label} ({msg_count} msgs)"
            if st.button(btn_label, key=f"ses_{sid}", use_container_width=True,
                         disabled=is_current):
                # Load selected session
                st.session_state.session_id = sid
                saved = db.get_session_messages(sid)
                st.session_state.chat_history = [
                    {"role": m["role"], "content": m["content"]} for m in saved
                ]
                st.session_state.debug_results = []
                st.rerun()

    col_new, col_clear = st.columns(2)
    if col_new.button("➕ New Chat", use_container_width=True):
        new_sid = str(uuid.uuid4())
        db.ensure_session(new_sid)
        st.session_state.session_id = new_sid
        st.session_state.chat_history = []
        st.session_state.debug_results = []
        st.rerun()
    if col_clear.button("🗑️ Clear", use_container_width=True):
        db.delete_session(st.session_state.session_id)
        new_sid = str(uuid.uuid4())
        db.ensure_session(new_sid)
        st.session_state.session_id = new_sid
        st.session_state.chat_history = []
        st.session_state.debug_results = []
        st.rerun()

# ---------------------------------------------------------------------------
# Helper render functions (defined before tabs so they are available below)
# ---------------------------------------------------------------------------

def _render_debug_panel(debug: dict) -> None:
    """Render the full debug panel inside an expander."""
    with st.expander("🔧 Debug Panel — Internal Pipeline Steps", expanded=False):
        # Router
        if "router" in debug:
            r = debug["router"]
            st.markdown("**Step 1: Query Understanding**")
            st.markdown(
                f'<div class="debug-box">'
                f'Intent: <b>{r["intent_type"]}</b> | '
                f'Domains: <b>{r["domains"]}</b> | '
                f'Course-related: {r["is_course_related"]} | '
                f'Reasoning: {r["reasoning"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Sub-questions
        if "sub_questions" in debug and debug["sub_questions"]:
            st.markdown("**Step 2: Query Decomposition**")
            for i, sq in enumerate(debug["sub_questions"], 1):
                st.markdown(
                    f'<div class="debug-box">'
                    f'Sub-Q {i}: <b>{sq["question"]}</b>  →  '
                    f'<span class="badge-{sq["domain"]}">{sq["domain"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Retrieval
        if "retrieval" in debug:
            st.markdown("**Steps 3–4: Retrieval & Reranking**")
            for rd in debug["retrieval"]:
                with st.expander(
                    f"{rd['domain']} — {rd['reranked_count']} chunks after reranking",
                    expanded=False,
                ):
                    for j, chunk in enumerate(rd["chunks"], 1):
                        st.markdown(
                            f"**[Src {j}]** `{chunk['source']}` "
                            f"(rerank={chunk['rerank_score']:.3f}, "
                            f"sem={chunk['semantic_score']:.3f}, "
                            f"bm25={chunk['bm25_score']:.3f})\n\n"
                            f"> {chunk['text'][:250]}…"
                        )

        # Domain answers preview
        if "domain_answers" in debug:
            st.markdown("**Step 5: Domain Agent Answers**")
            for da in debug["domain_answers"]:
                with st.expander(f"[{da['domain']}] {da['question'][:60]}…", expanded=False):
                    st.write(da["preview"] + "…")

        # Synthesized draft
        if "synthesized_preview" in debug:
            st.markdown("**Step 6: Synthesized Draft (before verification)**")
            st.info(debug["synthesized_preview"] + "…")

        # Verification
        if "verification" in debug:
            vd = debug["verification"]
            st.markdown("**Step 7: Verification**")
            if vd.get("skipped"):
                st.success("Verification was disabled.")
            else:
                cols = st.columns(3)
                cols[0].metric("Quality", f"{vd.get('quality_score', 0):.0%}")
                cols[1].metric("Coverage", f"{vd.get('coverage_score', 0):.0%}")
                cols[2].metric("Grounding", f"{vd.get('grounding_score', 0):.0%}")
                if vd.get("issues"):
                    st.markdown("**Issues found:**")
                    for iss in vd["issues"]:
                        st.markdown(f"- {iss}")
                if vd.get("was_revised"):
                    st.success("✅ Answer was revised by the verifier.")


def _render_eval_result(result) -> None:
    """Render a single EvalResult."""
    tc = result.test_case

    col_pass, col_intent, col_domain = st.columns(3)
    col_pass.metric("Result", "✅ PASS" if result.passed else "❌ FAIL")
    col_intent.metric(
        "Intent",
        f"{'✅' if result.intent_match else '❌'} {result.actual_intent}",
        f"Expected: {tc.expected_intent}",
    )
    col_domain.metric(
        "Domains",
        f"{'✅' if result.domain_match else '❌'} {result.actual_domains}",
        f"Expected: {tc.expected_domains}",
    )

    if result.quality_score:
        st.metric("Quality Score", f"{result.quality_score:.0%}")

    st.markdown("**Expected behavior:**")
    st.info(tc.expected_behavior)

    if result.answer_preview:
        st.markdown("**Answer preview:**")
        st.write(result.answer_preview + ("…" if len(result.answer_preview) >= 500 else ""))

    if result.error:
        st.error(f"Error: {result.error}")


# ---------------------------------------------------------------------------
# ████  MAIN CONTENT  ████
# ---------------------------------------------------------------------------
tab_chat, tab_eval, tab_kb = st.tabs(["💬 Chat", "🔬 Evaluation", "📚 Knowledge Base"])

# ============================================================
# TAB 1: CHAT
# ============================================================
with tab_chat:
    st.title("🎓 EduPilot — AI Educational Tutor")
    st.caption(
        "Ask anything about **Applied Machine Learning**, "
        "**Applied Database Technologies**, or **Statistics**."
    )

    # ---- Sample prompt buttons ----
    st.markdown("**Quick examples:**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    sample_prompts = [
        ("Bias-Variance Tradeoff", "What is the bias and variance tradeoff?"),
        ("Database Normalization", "What is normalization in databases?"),
        ("P-Value", "What is a p-value and how do I use it?"),
        ("Multi-Domain", "What is machine learning and how does NL2SQL work?"),
        ("Confidence Interval", "Explain confidence intervals with an example."),
        ("Overfitting", "What is overfitting and how can it be prevented?"),
    ]
    cols = [sample_col1, sample_col2, sample_col3, sample_col1, sample_col2, sample_col3]
    for col, (label, prompt) in zip(cols, sample_prompts):
        if col.button(label, use_container_width=True, key=f"sp_{label}"):
            st.session_state._pending_prompt = prompt

    st.divider()

    # ---- Chat history ----
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show debug expander for assistant messages
                if msg["role"] == "assistant" and show_debug and i < len(st.session_state.debug_results):
                    debug_idx = i // 2  # one debug result per exchange
                    if debug_idx < len(st.session_state.debug_results):
                        _render_debug_panel(st.session_state.debug_results[debug_idx])

    # ---- Chat input ----
    pending = st.session_state.pop("_pending_prompt", None)
    user_input = st.chat_input("Ask your question here…") or pending

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        db.save_message(st.session_state.session_id, "user", user_input)
        # Auto-title the session from the first user message
        if len(st.session_state.chat_history) == 1:
            db.update_session_title(st.session_state.session_id, user_input[:60])

        # Run pipeline
        with st.chat_message("assistant"):
            with st.status("Thinking…", expanded=True) as status:
                st.write("🔍 Classifying intent and domain(s)…")
                t_start = time.time()

                try:
                    result = run_pipeline(
                        query=user_input,
                        model=selected_model,
                        top_k=top_k,
                        rerank_top_k=rerank_top_k,
                        enable_verification=enable_verification,
                        confidence_threshold=confidence_threshold,
                        manual_domains=manual_domains if manual_domains else None,
                        chat_history=[
                            m for m in st.session_state.chat_history[:-1]
                            if m["role"] == "user"
                        ][-4:],
                    )
                    debug = result["debug"]

                    # Show intermediate steps in status
                    st.write(
                        f"✅ Intent: **{result['intent_type']}** | "
                        f"Domains: **{', '.join(result['detected_domains']) or 'N/A'}**"
                    )
                    if result.get("sub_questions"):
                        st.write(f"✂️ Decomposed into **{len(result['sub_questions'])}** sub-question(s)")
                    st.write("📚 Retrieving and reranking sources…")
                    st.write("✍️ Generating grounded answer…")
                    if enable_verification and result.get("is_course_related"):
                        st.write("🔎 Verifying answer quality…")

                    elapsed = time.time() - t_start
                    status.update(
                        label=f"Done in {elapsed:.1f}s",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:
                    status.update(label="Error", state="error")
                    st.error(f"Pipeline error: {exc}")
                    st.stop()

            # ---- Domain badges ----
            if result["detected_domains"]:
                badges = " ".join(
                    f'<span class="badge-{d}">{d}</span>'
                    for d in result["detected_domains"]
                )
                st.markdown(badges, unsafe_allow_html=True)

            # ---- Final answer ----
            st.markdown(result["final_answer"])

            # ---- Verification indicator ----
            if not result.get("needs_clarification") and result.get("is_course_related"):
                vd = debug.get("verification", {})
                if not vd.get("skipped"):
                    qs = vd.get("quality_score", 0)
                    color = "green" if qs >= 0.7 else "orange" if qs >= 0.4 else "red"
                    indicator = "✅" if vd.get("is_satisfactory") else "⚠️"
                    revised_note = " (answer was revised)" if vd.get("was_revised") else ""
                    st.caption(
                        f"{indicator} Quality score: **{qs:.0%}**{revised_note} "
                        f"| Coverage: {vd.get('coverage_score', 0):.0%} "
                        f"| Grounding: {vd.get('grounding_score', 0):.0%}"
                    )
                    if vd.get("issues"):
                        with st.expander("⚠️ Verification notes"):
                            for issue in vd["issues"]:
                                st.markdown(f"- {issue}")

            # ---- Debug panel ----
            if show_debug:
                _render_debug_panel(debug)

        # Save to in-memory history and persist to SQLite
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["final_answer"]}
        )
        st.session_state.debug_results.append(debug)
        db.save_message(
            session_id=st.session_state.session_id,
            role="assistant",
            content=result["final_answer"],
            intent_type=result.get("intent_type"),
            detected_domains=result.get("detected_domains"),
            quality_score=result.get("quality_score"),
            pipeline_meta=result.get("debug"),
        )


# ============================================================
# TAB 2: EVALUATION
# ============================================================
with tab_eval:
    st.title("🔬 Evaluation Suite")
    st.markdown(
        "Run the 10 built-in test cases to validate routing, retrieval, "
        "synthesis, verification, and edge-case handling."
    )

    TEST_CASES = pipeline["TEST_CASES"]
    run_all_fn = pipeline["run_all_evaluations"]
    summary_fn = pipeline["summary_stats"]
    run_single_fn = pipeline["run_evaluation"]

    # Show test case table
    with st.expander("📋 View All Test Cases", expanded=False):
        for tc in TEST_CASES:
            st.markdown(
                f"**{tc.id}** — {tc.name}  \n"
                f"*Query:* `{tc.query}`  \n"
                f"*Expected:* intent=`{tc.expected_intent}`, "
                f"domains=`{tc.expected_domains}`  \n"
                f"*Category:* `{tc.category}`"
            )
            st.divider()

    col_run, col_single = st.columns([2, 1])
    with col_run:
        run_all = st.button("▶️ Run All Test Cases", type="primary", use_container_width=True)
    with col_single:
        tc_id_options = [f"{tc.id}: {tc.name}" for tc in TEST_CASES]
        selected_tc_str = st.selectbox("Run single test", tc_id_options, label_visibility="collapsed")
        run_single = st.button("▶️ Run Selected", use_container_width=True)

    # Run single
    if run_single and selected_tc_str:
        tc_idx = tc_id_options.index(selected_tc_str)
        tc = TEST_CASES[tc_idx]
        with st.spinner(f"Running {tc.id}…"):
            result = run_single_fn(
                test_case=tc,
                pipeline_fn=lambda query, **kw: run_pipeline(
                    query=query,
                    model=kw.get("model", selected_model),
                    top_k=kw.get("top_k", top_k),
                    rerank_top_k=kw.get("rerank_top_k", rerank_top_k),
                    enable_verification=kw.get("enable_verification", enable_verification),
                    confidence_threshold=confidence_threshold,
                ),
                model=selected_model,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_verification=enable_verification,
            )
        _render_eval_result(result)

    # Run all
    if run_all:
        progress_bar = st.progress(0, text="Starting evaluation…")
        results: list = []

        def on_progress(name: str, current: int, total: int):
            pct = current / total
            progress_bar.progress(pct, text=f"Running {current}/{total}: {name}")

        with st.spinner("Running all test cases…"):
            results = run_all_fn(
                pipeline_fn=lambda query, **kw: run_pipeline(
                    query=query,
                    model=kw.get("model", selected_model),
                    top_k=kw.get("top_k", top_k),
                    rerank_top_k=kw.get("rerank_top_k", rerank_top_k),
                    enable_verification=kw.get("enable_verification", enable_verification),
                    confidence_threshold=confidence_threshold,
                ),
                model=selected_model,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_verification=enable_verification,
                on_progress=on_progress,
            )
            st.session_state.eval_results = results

        progress_bar.empty()

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        stats = summary_fn(results)

        # Summary metrics
        st.subheader("📊 Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pass Rate", f"{stats['pass_rate']}%", f"{stats['passed']}/{stats['total']}")
        m2.metric("Intent Accuracy", f"{stats['intent_accuracy']}%")
        m3.metric("Domain Accuracy", f"{stats['domain_accuracy']}%")
        m4.metric("Avg Quality", f"{stats['avg_quality_score']:.2f}")

        # By category
        if stats.get("by_category"):
            st.subheader("By Category")
            cat_cols = st.columns(len(stats["by_category"]))
            for col, (cat, cat_stats) in zip(cat_cols, stats["by_category"].items()):
                pct = round(cat_stats["passed"] / cat_stats["total"] * 100)
                col.metric(
                    cat.replace("-", " ").title(),
                    f"{pct}%",
                    f"{cat_stats['passed']}/{cat_stats['total']} passed",
                )

        # Detailed results
        st.subheader("Detailed Results")
        for r in results:
            icon = "✅" if r.passed else "❌"
            with st.expander(f"{icon} {r.test_case.id}: {r.test_case.name}", expanded=False):
                _render_eval_result(r)


# ============================================================
# TAB 3: KNOWLEDGE BASE
# ============================================================
with tab_kb:
    st.title("📚 Knowledge Base")
    st.markdown(
        "Overview of indexed documents per domain. "
        "Upload new documents via the **sidebar**."
    )

    for domain, cfg in DOMAINS.items():
        color = cfg["color"]
        st.markdown(
            f"### <span style='color:{color}'>{cfg['name']} ({domain})</span>",
            unsafe_allow_html=True,
        )
        st.caption(cfg["description"])

        r = pipeline["get_retriever"](domain)
        chunk_count = r.document_count()

        col_a, col_b = st.columns([1, 3])
        col_a.metric("Chunks Indexed", chunk_count)

        kb_path = Path(cfg["knowledge_base_path"])
        docs = []
        for ext in [".pdf", ".txt", ".md", ".docx"]:
            docs.extend(kb_path.glob(f"*{ext}"))

        if docs:
            col_b.markdown("**Knowledge base files:**")
            for doc in docs:
                col_b.markdown(f"- 📄 `{doc.name}`")
        else:
            col_b.info(
                f"No documents found in `{cfg['knowledge_base_path']}`. "
                "Upload files via the sidebar."
            )

        # Show user-uploaded documents from SQLite
        uploaded = db.list_uploaded_docs(domain=domain)
        if uploaded:
            col_b.markdown("**User-uploaded files:**")
            for u in uploaded:
                size_kb = round(u.get("file_size_bytes", 0) / 1024, 1)
                col_b.markdown(
                    f"- 📎 `{u['filename']}` "
                    f"({u['chunk_count']} chunks, {size_kb} KB) "
                    f"— {u['upload_timestamp'][:10]}"
                )
        st.divider()

    st.markdown("""
    ### Adding New Domains

    To add a new domain (e.g., **NLP**), edit `config.py`:

    ```python
    DOMAINS["NLP"] = {
        "name": "Natural Language Processing",
        "abbr": "NLP",
        "color": "#9C27B0",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "nlp"),
        "vector_store_path": str(VECTOR_STORE_DIR / "nlp"),
        "collection_name": "nlp_docs",
        "description": "Text processing, transformers, embeddings, etc.",
        "keywords": ["NLP", "transformer", "BERT", "tokenization", ...],
    }
    ```

    Then create `knowledge_base/nlp/` and upload documents. No other code changes needed.
    """)
