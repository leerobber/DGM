# Adapted from jennyzzt/dgm — Sovereign Core rewiring.
# All LLM calls route to the local vLLM endpoint (Qwen2.5-32B-AWQ).
# No Anthropic / Bedrock / Vertex / OpenRouter dependencies required.
import json
import os
import re

import backoff
import openai

MAX_OUTPUT_TOKENS = 4096

# ── Sovereign Core endpoint ───────────────────────────────────────────────────
SOVEREIGN_API_BASE: str = os.environ.get("SOVEREIGN_API_BASE", "http://localhost:8001/v1")
SOVEREIGN_MODEL: str = os.environ.get("SOVEREIGN_MODEL", "qwen2.5-32b-awq")
SOVEREIGN_API_KEY: str = os.environ.get("SOVEREIGN_API_KEY", "sovereign")

AVAILABLE_LLMS = [
    # Sovereign Core (local vLLM — primary)
    "sovereign/qwen2.5-32b-awq",
    # Kept for backward-compat; will route via openai-compatible endpoint
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
]


def create_client(model: str):
    """
    Create and return an (client, model_name) tuple for the given model.

    Sovereign prefix ``sovereign/<name>`` routes every call to the local vLLM
    endpoint regardless of the model name that follows the slash.
    """
    if model.startswith("sovereign/"):
        client_model = model.split("/", 1)[1]
        print(f"Using Sovereign Core endpoint with model {client_model}.")
        client = openai.OpenAI(
            base_url=SOVEREIGN_API_BASE,
            api_key=SOVEREIGN_API_KEY,
        )
        return client, client_model

    # ── Legacy OpenAI cloud paths (only active when sovereign prefix absent) ──
    if "gpt" in model or model.startswith("o1-") or model.startswith("o3-"):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model

    raise ValueError(
        f"Model {model!r} not supported. "
        "Use 'sovereign/<model-name>' to route to the local Sovereign Core endpoint, "
        "or set SOVEREIGN_MODEL env var."
    )


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.75,
    n_responses=1,
):
    """Get N independent responses for a single message (ensemble / multi-sample)."""
    if msg_history is None:
        msg_history = []

    content, new_msg_history = [], []
    for _ in range(n_responses):
        c, hist = get_response_from_llm(
            msg,
            client,
            model,
            system_message,
            print_debug=False,
            msg_history=None,
            temperature=temperature,
        )
        content.append(c)
        new_msg_history.append(hist)

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, m in enumerate(new_msg_history[0]):
            print(f'{j}, {m["role"]}: {m["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError),
    max_time=120,
)
def get_response_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
):
    """Single-turn LLM call using the OpenAI-compatible chat completions API."""
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_tokens=MAX_OUTPUT_TOKENS,
        n=1,
        stop=None,
    )
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f'User: {new_msg_history[-2]["content"]}')
        print(f'Assistant: {new_msg_history[-1]["content"]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output):
    """Extract the first JSON block from LLM output (```json ... ```)."""
    inside_json_block = False
    json_lines = []

    for line in llm_output.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```json"):
            inside_json_block = True
            continue
        if inside_json_block and stripped.startswith("```"):
            inside_json_block = False
            break
        if inside_json_block:
            json_lines.append(line)

    if not json_lines:
        fallback_pattern = r"\{.*?\}"
        for candidate in re.findall(fallback_pattern, llm_output, re.DOTALL):
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(clean)
                    except json.JSONDecodeError:
                        continue
        return None

    json_string = "\n".join(json_lines).strip()
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return None
