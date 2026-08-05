"""
Agent orchestration
===================
The five agents and the pipeline that runs them.

    Router -> Planner -> [Retriever -> Answerer] x N -> Synthesizer -> Guardrails

Behavioural changes from the previous `_run_pipeline` in main.py:

  * **No answer without evidence.** Empty retrieval produces a refusal. The
    old pipeline routed to a "no context" prompt that answered from the
    model's own memory, which is the opposite of the system's stated contract.
  * **Sub-questions run concurrently.** They were sequential, so a
    three-domain question paid three round-trips end to end for work that has
    no ordering dependency.
  * **The verifier can act.** Its verdict now flows into the output policy,
    which annotates or replaces the answer. Previously `revised_answer` was
    hard-coded null in the prompt *and* nulled again in code, so verification
    could never change what a student saw.
  * **Quality scores are measured or absent.** No floors. `main.py:714`
    applied `max(quality, 0.75)`, which meant the displayed number was a
    constant for any answer scoring below it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from edupilot.guardrails import scan_chunks
from edupilot.guardrails.output import OutputVerdict, apply_output_guardrails
from edupilot.retrieval import HybridRetriever, RetrievalConfig, RetrievalResult

from . import prompts
from .contracts import (
    EvidenceBlock,
    build_evidence,
    build_evidence_summary,
    format_history,
    is_refusal,
    make_fence,
    parse_json_response,
)

logger = logging.getLogger(__name__)

LLMCallable = Callable[..., str]
"""(messages, system, model, max_tokens) -> completion text."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RouterDecision:
    intent_type: str = "single"
    domains: list[str] = field(default_factory=list)
    is_course_related: bool = True
    needs_clarification: bool = False
    clarification_hint: str | None = None
    reasoning: str = ""
    used_fallback: bool = False


@dataclass
class SubAnswer:
    """One domain's grounded answer."""

    domain: str
    question: str
    answer: str
    evidence: EvidenceBlock
    retrieval: RetrievalResult
    refused: bool = False
    latency_ms: int = 0

    @property
    def has_evidence(self) -> bool:
        return not self.retrieval.is_empty


@dataclass
class PipelineResult:
    """Everything the API layer needs to build a response."""

    final_answer: str
    intent_type: str = "single"
    domains: list[str] = field(default_factory=list)
    sub_answers: list[SubAnswer] = field(default_factory=list)
    verdict: OutputVerdict | None = None
    is_course_related: bool = True
    needs_clarification: bool = False
    refused: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def grounding_score(self) -> float | None:
        """Measured grounding, or None when it was not measured. Never a default."""
        return self.verdict.grounding_score if self.verdict else None

    @property
    def sources(self) -> list[dict]:
        """Flat, de-duplicated citation list for the UI."""
        out: list[dict] = []
        seen: set[str] = set()
        for sub in self.sub_answers:
            for i, (label, chunk) in enumerate(
                zip(sub.evidence.labels, sub.retrieval.chunks), start=1
            ):
                key = f"{sub.domain}:{label}"
                if key in seen:
                    continue
                seen.add(key)
                meta = chunk.metadata
                out.append(
                    {
                        "source_num": i,
                        "domain": sub.domain,
                        "citation_label": label,
                        "filename": meta.get("filename"),
                        "page_number": meta.get("page_number"),
                        "section_path": meta.get("section_path"),
                        "relevance": round(chunk.relevance, 4),
                        "text": chunk.text[:1200],
                    }
                )
        return out


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class Router:
    """Classifies intent and domains. Never answers."""

    def __init__(self, llm: LLMCallable, domains: dict[str, dict]) -> None:
        self.llm = llm
        self.domains = domains

    def classify(
        self, query: str, *, model: str, history: Sequence[dict] | None = None
    ) -> RouterDecision:
        user_prompt = prompts.ROUTER_USER.format(
            query=query, history_block=format_history(history)
        )
        try:
            raw = self.llm(
                messages=[{"role": "user", "content": user_prompt}],
                system=prompts.ROUTER_SYSTEM,
                model=model,
                max_tokens=400,
            )
            data = parse_json_response(raw)
        except Exception as exc:
            logger.warning("router LLM failed (%s) — using keyword fallback", exc)
            return self._keyword_fallback(query)

        if not data:
            logger.warning("router returned unparseable JSON — using keyword fallback")
            return self._keyword_fallback(query)

        domains = [d for d in data.get("domains", []) if d in self.domains]
        if not domains and data.get("is_course_related", True):
            domains = self._keyword_fallback(query).domains

        intent = data.get("intent_type", "single")
        return RouterDecision(
            intent_type=intent if intent in ("single", "multi") else "single",
            domains=domains,
            is_course_related=bool(data.get("is_course_related", True)),
            needs_clarification=bool(data.get("needs_clarification", False)) and not domains,
            clarification_hint=data.get("clarification_hint"),
            reasoning=str(data.get("reasoning", "")),
        )

    def _keyword_fallback(self, query: str) -> RouterDecision:
        """Keyword classification for when the LLM is unavailable."""
        lowered = query.lower()
        hits = [
            domain
            for domain, cfg in self.domains.items()
            if any(kw.lower() in lowered for kw in cfg.get("keywords", []))
        ]
        return RouterDecision(
            intent_type="multi" if len(hits) > 1 else "single",
            domains=hits,
            is_course_related=bool(hits),
            needs_clarification=not hits and len(query.split()) < 4,
            reasoning="keyword fallback (router unavailable)",
            used_fallback=True,
        )


class Planner:
    """Decomposes a multi-topic question into per-domain sub-questions."""

    def __init__(self, llm: LLMCallable, domains: dict[str, dict]) -> None:
        self.llm = llm
        self.domains = domains

    def plan(
        self, query: str, decision: RouterDecision, *, model: str
    ) -> list[dict[str, str]]:
        if decision.intent_type == "single" or len(decision.domains) <= 1:
            domain = decision.domains[0] if decision.domains else next(iter(self.domains))
            return [{"question": query, "domain": domain}]

        user_prompt = prompts.PLANNER_USER.format(
            query=query, domains=", ".join(decision.domains)
        )
        try:
            raw = self.llm(
                messages=[{"role": "user", "content": user_prompt}],
                system=prompts.PLANNER_SYSTEM,
                model=model,
                max_tokens=700,
            )
            data = parse_json_response(raw)
        except Exception as exc:
            logger.warning("planner failed (%s) — one sub-question per domain", exc)
            data = {}

        subs = [
            {"question": sq["question"].strip(), "domain": sq.get("domain", "").upper()}
            for sq in data.get("sub_questions", [])
            if isinstance(sq, dict) and sq.get("question", "").strip()
        ]
        subs = [s for s in subs if s["domain"] in self.domains]

        if not subs:
            # Fall back to asking the original question of each detected domain
            # rather than inventing a split the model did not produce.
            return [{"question": query, "domain": d} for d in decision.domains]
        return subs


class Answerer:
    """Generates one grounded answer from retrieved evidence, or refuses."""

    def __init__(self, llm: LLMCallable, domains: dict[str, dict]) -> None:
        self.llm = llm
        self.domains = domains

    def answer(
        self,
        question: str,
        domain: str,
        retrieval: RetrievalResult,
        *,
        model: str,
        max_tokens: int,
        history: Sequence[dict] | None = None,
        self_study: bool = False,
    ) -> SubAnswer:
        started = time.perf_counter()
        fence = make_fence()
        evidence = build_evidence(retrieval.chunks, fence)

        # No evidence means refuse. There is deliberately no path that answers
        # from the model's own knowledge — that path is what the previous
        # DOMAIN_AGENT_USER_NO_CONTEXT prompt provided.
        if retrieval.is_empty:
            logger.info("no evidence for %r in %s — refusing", question[:60], domain)
            return SubAnswer(
                domain=domain,
                question=question,
                answer=prompts.NO_EVIDENCE_RESPONSE,
                evidence=evidence,
                retrieval=retrieval,
                refused=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        cfg = self.domains.get(domain, {})
        if self_study:
            system = prompts.SELF_STUDY_SYSTEM
            template = prompts.SELF_STUDY_USER
        else:
            system = prompts.ANSWERER_SYSTEM.format(
                domain_name=cfg.get("name", domain), domain_abbr=cfg.get("abbr", domain)
            )
            template = prompts.ANSWERER_USER

        user_prompt = template.format(
            question=question,
            history_block=format_history(history),
            fence_open=fence.open,
            fence_close=fence.close,
            retrieved_chunks=evidence.text,
        )

        try:
            text = self.llm(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            from edupilot.security.errors import classify_upstream

            error = classify_upstream(exc)
            error.log()
            text = f"⚠️ {error.client_message}"

        return SubAnswer(
            domain=domain,
            question=question,
            answer=(text or "").strip(),
            evidence=evidence,
            retrieval=retrieval,
            refused=is_refusal(text or ""),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class Synthesizer:
    """Merges per-domain answers. Reorganizes only; never adds facts."""

    def __init__(self, llm: LLMCallable, domains: dict[str, dict]) -> None:
        self.llm = llm
        self.domains = domains

    def synthesize(
        self, query: str, sub_answers: list[SubAnswer], *, model: str, max_tokens: int
    ) -> str:
        answered = [s for s in sub_answers if not s.refused]

        if not answered:
            return prompts.NO_EVIDENCE_RESPONSE
        if len(answered) == 1:
            return answered[0].answer

        blocks = "\n\n".join(
            f"### [{s.domain}] {s.question}\n\n{s.answer}" for s in answered
        )
        # A domain that refused is reported as a gap rather than dropped, so
        # the student knows the question was only partly covered.
        refused = [s for s in sub_answers if s.refused]
        if refused:
            gaps = "\n".join(f"- {s.domain}: {s.question}" for s in refused)
            blocks += f"\n\n### Not covered by the course materials\n{gaps}"

        user_prompt = prompts.SYNTHESIZER_USER.format(
            original_query=query,
            num_parts=len(answered),
            num_domains=len({s.domain for s in answered}),
            sub_answers=blocks,
        )
        try:
            return self.llm(
                messages=[{"role": "user", "content": user_prompt}],
                system=prompts.SYNTHESIZER_SYSTEM,
                model=model,
                max_tokens=max_tokens,
            ).strip()
        except Exception as exc:
            logger.warning("synthesis failed (%s) — concatenating sub-answers", exc)
            # Concatenation preserves grounding; a failed merge must not lose
            # answers that were already produced and cited.
            return "\n\n".join(
                f"## {self.domains.get(s.domain, {}).get('name', s.domain)}\n\n{s.answer}"
                for s in answered
            )


class Verifier:
    """LLM audit of an answer against its evidence. Scores only; never rewrites."""

    def __init__(self, llm: LLMCallable) -> None:
        self.llm = llm

    def verify(
        self,
        query: str,
        sub_answers: list[SubAnswer],
        answer: str,
        *,
        model: str,
    ) -> dict:
        chunks = [c for s in sub_answers for c in s.retrieval.chunks]
        if not chunks:
            return {"skipped": True, "reason": "no evidence to verify against"}

        fence = make_fence()
        user_prompt = prompts.VERIFIER_USER.format(
            original_query=query,
            sub_questions="\n".join(f"- [{s.domain}] {s.question}" for s in sub_answers),
            fence_open=fence.open,
            fence_close=fence.close,
            evidence_summary=fence.scrub(build_evidence_summary(chunks)),
            answer=answer,
        )
        try:
            raw = self.llm(
                messages=[{"role": "user", "content": user_prompt}],
                system=prompts.VERIFIER_SYSTEM,
                model=model,
                max_tokens=1000,
            )
            data = parse_json_response(raw)
        except Exception as exc:
            logger.warning("verifier unavailable: %s", exc)
            return {"skipped": True, "reason": f"verifier unavailable: {exc}"}

        if not data:
            return {"skipped": True, "reason": "verifier returned unparseable output"}

        def clamp(key: str) -> float | None:
            value = data.get(key)
            if not isinstance(value, (int, float)):
                return None
            return max(0.0, min(1.0, float(value)))

        return {
            "skipped": False,
            "grounding_score": clamp("grounding_score"),
            "coverage_score": clamp("coverage_score"),
            "quality_score": clamp("quality_score"),
            "is_satisfactory": bool(data.get("is_satisfactory", False)),
            "unsupported_claims": [str(c) for c in data.get("unsupported_claims", [])][:10],
            "miscited_claims": [str(c) for c in data.get("miscited_claims", [])][:10],
            "missing_topics": [str(t) for t in data.get("missing_topics", [])][:10],
            "issues": [str(i) for i in data.get("issues", [])][:10],
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Per-request pipeline settings."""

    model: str = ""
    verify_model: str = ""
    max_tokens_answer: int = 2000
    max_tokens_synth: int = 2500
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    enable_verification: bool = True
    enable_grounding_check: bool = True
    max_parallel_domains: int = 4


class EduPilotPipeline:
    """
    Runs the full multi-agent pipeline for one question.

    Args:
        llm: Shared LLM caller.
        domains: Domain registry from config.
        retriever_factory: Maps a domain key to a `HybridRetriever`.
    """

    def __init__(
        self,
        llm: LLMCallable,
        domains: dict[str, dict],
        retriever_factory: Callable[[str], HybridRetriever],
    ) -> None:
        self.domains = domains
        self.retriever_factory = retriever_factory
        self.router = Router(llm, domains)
        self.planner = Planner(llm, domains)
        self.answerer = Answerer(llm, domains)
        self.synthesizer = Synthesizer(llm, domains)
        self.verifier = Verifier(llm)

    def run(
        self,
        query: str,
        cfg: PipelineConfig,
        *,
        history: Sequence[dict] | None = None,
        manual_domains: Sequence[str] | None = None,
        filenames: Sequence[str] | None = None,
    ) -> PipelineResult:
        started = time.perf_counter()
        diagnostics: dict[str, Any] = {}

        # --- route --------------------------------------------------------
        decision = self.router.classify(query, model=cfg.model, history=history)
        diagnostics["router"] = {
            "intent_type": decision.intent_type,
            "domains": decision.domains,
            "is_course_related": decision.is_course_related,
            "reasoning": decision.reasoning,
            "used_fallback": decision.used_fallback,
        }

        if manual_domains:
            decision.domains = [d for d in manual_domains if d in self.domains]
            decision.is_course_related = True
            decision.needs_clarification = False

        if not decision.is_course_related:
            return PipelineResult(
                final_answer=prompts.OUT_OF_DOMAIN_RESPONSE,
                intent_type=decision.intent_type,
                is_course_related=False,
                refused=True,
                diagnostics=diagnostics,
            )

        if decision.needs_clarification:
            return PipelineResult(
                final_answer=prompts.CLARIFICATION_RESPONSE,
                intent_type=decision.intent_type,
                needs_clarification=True,
                diagnostics=diagnostics,
            )

        # --- plan ---------------------------------------------------------
        sub_questions = self.planner.plan(query, decision, model=cfg.model)
        diagnostics["sub_questions"] = sub_questions

        # --- retrieve + answer, one task per sub-question -------------------
        def handle(sub: dict[str, str]) -> SubAnswer:
            retriever = self.retriever_factory(sub["domain"])
            retrieval = retriever.retrieve(
                sub["question"], config=cfg.retrieval, filenames=filenames
            )
            return self.answerer.answer(
                sub["question"],
                sub["domain"],
                retrieval,
                model=cfg.model,
                max_tokens=cfg.max_tokens_answer,
                history=history,
            )

        if len(sub_questions) == 1:
            sub_answers = [handle(sub_questions[0])]
        else:
            # Sub-questions are independent; running them serially made a
            # three-domain question take three times as long for no reason.
            with ThreadPoolExecutor(max_workers=cfg.max_parallel_domains) as pool:
                sub_answers = list(pool.map(handle, sub_questions))

        diagnostics["retrieval"] = [
            {
                "domain": s.domain,
                "question": s.question,
                "chunks": len(s.retrieval.chunks),
                "refused": s.refused,
                "latency_ms": s.latency_ms,
                **s.retrieval.diagnostics,
            }
            for s in sub_answers
        ]

        injection_reports = [
            report
            for s in sub_answers
            for report in scan_chunks(s.retrieval.chunks)[0]
        ]
        if injection_reports:
            diagnostics["injection_flags"] = injection_reports

        # --- synthesize -----------------------------------------------------
        merged = self.synthesizer.synthesize(
            query, sub_answers, model=cfg.model, max_tokens=cfg.max_tokens_synth
        )

        # --- verify ---------------------------------------------------------
        if cfg.enable_verification:
            diagnostics["verifier"] = self.verifier.verify(
                query, sub_answers, merged, model=cfg.verify_model or cfg.model
            )

        # --- guardrails ------------------------------------------------------
        all_chunks = [c for s in sub_answers for c in s.retrieval.chunks]
        all_labels = [label for s in sub_answers for label in s.evidence.labels]
        verdict = apply_output_guardrails(
            merged,
            [c.text for c in all_chunks],
            all_labels,
            check_claims=cfg.enable_grounding_check and bool(all_chunks),
        )
        diagnostics["guardrails"] = verdict.as_dict()
        diagnostics["total_latency_ms"] = int((time.perf_counter() - started) * 1000)

        return PipelineResult(
            final_answer=verdict.answer,
            intent_type=decision.intent_type,
            domains=[s.domain for s in sub_answers],
            sub_answers=sub_answers,
            verdict=verdict,
            refused=verdict.is_refusal or all(s.refused for s in sub_answers),
            diagnostics=diagnostics,
        )
