# Code adapted from https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py.
import json
import os
import re

import backoff
import openai

# ---------------------------------------------------------------------------
# Sovereign Core configuration
# ---------------------------------------------------------------------------
SOVEREIGN_API_BASE = os.getenv("SOVEREIGN_API_BASE", "http://localhost:8001/v1")
SOVEREIGN_MODEL = os.getenv("SOVEREIGN_MODEL", "qwen2.5-32b-awq")

MAX_OUTPUT_TOKENS = 8192

AVAILABLE_LLMS = [
    "sovereign",       # Qwen2.5-32B-AWQ via local vLLM endpoint (default)
    "qwen2.5-32b-awq", # Same as above — use directly by model name
]


def create_client(model: str):
    """
    Create and return an LLM client based on the specified model.

    For Sovereign Core, all models route to the local vLLM/litellm endpoint
    at SOVEREIGN_API_BASE (default: http://localhost:8001/v1).

    Returns:
        Tuple[openai.OpenAI, str]: (client, resolved_model_name)
    """
    resolved_model = SOVEREIGN_MODEL if model in ("sovereign",) else model
    client = openai.OpenAI(
        base_url=SOVEREIGN_API_BASE,
        api_key=os.getenv("SOVEREIGN_API_KEY", "sovereign"),  # vLLM ignores key
        timeout=120.0,
    )
    print(f"Using Sovereign Core endpoint {SOVEREIGN_API_BASE} with model {resolved_model}.")
    return client, resolved_model


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError),
    max_time=120,
)
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
    """Get N responses from a single message; used for ensembling."""
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
        n=n_responses,
        stop=None,
    )
    content = [r.message.content for r in response.choices]
    new_msg_history = [
        new_msg_history + [{"role": "assistant", "content": c}] for c in content
    ]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, entry in enumerate(new_msg_history[0]):
            print(f'{j}, {entry["role"]}: {entry["content"]}')
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
    """Get a single response from the Sovereign Core LLM endpoint."""
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
    inside_json_block = False
    json_lines = []

    for line in llm_output.split('\n'):
        stripped_line = line.strip()

        if stripped_line.startswith("```json"):
            inside_json_block = True
            continue

        if inside_json_block and stripped_line.startswith("```"):
            inside_json_block = False
            break

        if inside_json_block:
            json_lines.append(line)

    if not json_lines:
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    json_string = "\n".join(json_lines).strip()
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None
