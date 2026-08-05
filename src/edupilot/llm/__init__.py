"""
LLM client
==========
One `call_llm` entry point across Groq and Gemini, with retry, provider
fallback, and per-request token accounting::

    from edupilot.llm import call_llm, start_usage, get_usage

    start_usage()
    answer = call_llm(messages, model="llama-3.3-70b-versatile")
    tokens = get_usage().as_dict()
"""

from .client import (
    LLMError,
    Usage,
    build_fallback_chain,
    call_llm,
    get_usage,
    is_gemini_model,
    is_groq_model,
    parse_json_response,
    start_usage,
)

__all__ = [
    "LLMError",
    "Usage",
    "build_fallback_chain",
    "call_llm",
    "get_usage",
    "is_gemini_model",
    "is_groq_model",
    "parse_json_response",
    "start_usage",
]
