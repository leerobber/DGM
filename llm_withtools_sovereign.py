"""
llm_withtools_sovereign.py — DGM LLM Adapter for Sovereign Core

Drop-in replacement for DGM's llm_withtools.py that routes all
inference through the Sovereign Core gateway instead of Anthropic/OpenAI.

Usage — in DGM_outer.py, coding_agent.py, coding_agent_polyglot.py:
  # Replace:
  from llm_withtools import CLAUDE_MODEL, OPENAI_MODEL, chat_with_agent
  # With:
  from llm_withtools_sovereign import CLAUDE_MODEL, OPENAI_MODEL, chat_with_agent

The gateway handles:
  1. RTX 5050 — Qwen2.5-32B-AWQ (primary brain)
  2. Radeon 780M — DeepSeek-Coder-33B (code verification)
  3. Ryzen 7 CPU — Mistral-7B (fallback)

Falls back to direct Ollama then to the original llm_withtools if gateway
is unreachable (keeps DGM working even when TatorTot is offline).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

# ── Model aliases (preserve DGM's original constants) ────────────────────────
CLAUDE_MODEL = os.environ.get("SOVEREIGN_PRIMARY_MODEL", "qwen2.5:14b")
OPENAI_MODEL = os.environ.get("SOVEREIGN_CODER_MODEL",   "deepseek-coder:6.7b")

# ── Gateway config ────────────────────────────────────────────────────────────
GATEWAY_URL = os.environ.get("SOVEREIGN_GATEWAY_URL", "http://localhost:8000")
DIRECT_OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
REQUEST_TIMEOUT = int(os.environ.get("SOVEREIGN_TIMEOUT", "120"))
MAX_TOKENS = int(os.environ.get("SOVEREIGN_MAX_TOKENS", "16384"))

# ── HTTP session with retries ─────────────────────────────────────────────────
_session = requests.Session()
_session.mount("http://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1.0,
    status_forcelist=[502, 503, 504],
)))


def _is_gateway_alive() -> bool:
    try:
        r = _session.get(f"{GATEWAY_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_gateway(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    """Call /v1/chat/completions on the Sovereign Core gateway."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = _session.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _call_ollama_direct(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    """Direct Ollama fallback — used when gateway is unreachable."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = _session.post(
        f"{DIRECT_OLLAMA}/api/chat",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def chat_with_agent(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> tuple[str, list[dict]]:
    """
    Drop-in replacement for DGM's chat_with_agent.

    Args:
        messages:       Conversation history (OpenAI format).
        model:          Model ID. None = use CLAUDE_MODEL (primary brain).
        temperature:    Sampling temperature.
        max_tokens:     Max tokens to generate.
        tools:          Tool definitions (ignored — tools handled by SAGE loop).
        tool_choice:    Tool choice override (ignored).
        system_prompt:  Optional system prompt prepended to messages.
        **kwargs:       Absorbed for compatibility.

    Returns:
        (response_text, updated_messages)
    """
    active_model = model or CLAUDE_MODEL
    all_messages = list(messages)

    # Prepend system prompt if provided and not already present
    if system_prompt:
        if not all_messages or all_messages[0].get("role") != "system":
            all_messages = [{"role": "system", "content": system_prompt}] + all_messages

    t0 = time.time()
    response_text = ""
    backend_used = "unknown"

    try:
        if _is_gateway_alive():
            response_text = _call_gateway(all_messages, active_model, temperature, max_tokens)
            backend_used = "sovereign_gateway"
        else:
            logger.warning("Gateway unreachable — trying direct Ollama")
            response_text = _call_ollama_direct(all_messages, active_model, temperature, max_tokens)
            backend_used = "ollama_direct"
    except Exception as exc:
        logger.warning("Sovereign call failed (%s) — trying original llm_withtools", exc)
        try:
            from llm_withtools import chat_with_agent as _original_chat
            response_text, all_messages = _original_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                system_prompt=system_prompt,
                **kwargs,
            )
            return response_text, all_messages
        except Exception as fallback_exc:
            logger.error("All fallbacks failed: %s", fallback_exc)
            raise

    latency = time.time() - t0
    logger.debug("chat_with_agent: model=%s backend=%s latency=%.2fs tokens≈%d",
                 active_model, backend_used, latency, len(response_text) // 4)

    # Append assistant response to history
    updated_messages = all_messages + [{"role": "assistant", "content": response_text}]
    return response_text, updated_messages


# ── Self-improvement integration ──────────────────────────────────────────────

def get_response_with_tools(
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history: list[dict] | None = None,
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    **kwargs: Any,
) -> tuple[str, list[dict], dict]:
    """
    Alias matching DGM's get_response_withtools signature.
    Returns (response_text, updated_history, usage_dict)
    """
    history = list(msg_history or [])
    history.append({"role": "user", "content": user_message})

    response, updated = chat_with_agent(
        messages=history,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        system_prompt=system_prompt,
        **kwargs,
    )

    usage = {
        "prompt_tokens": sum(len(m.get("content", "")) // 4 for m in history),
        "completion_tokens": len(response) // 4,
        "backend": "sovereign",
    }
    return response, updated, usage
