# Adapted from jennyzzt/dgm — Sovereign Core rewiring.
# Tool-calling agent: all model calls now route to the local vLLM endpoint.
import ast
import copy
import json
import os
import re

import backoff
import openai

from llm import create_client, get_response_from_llm
from prompts.tooluse_prompt import get_tooluse_prompt
from tools import load_all_tools

# ── Default models (env-overridable) ─────────────────────────────────────────
SOVEREIGN_MODEL: str = f"sovereign/{os.environ.get('SOVEREIGN_MODEL', 'qwen2.5-32b-awq')}"
# Legacy constants retained so external imports don't break, but point to sovereign.
CLAUDE_MODEL = SOVEREIGN_MODEL
OPENAI_MODEL = SOVEREIGN_MODEL


# ── Tool format helpers ───────────────────────────────────────────────────────

def _to_openai_tool(tool_info: dict) -> dict:
    """Convert Anthropic-style tool info to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool_info["name"],
            "description": tool_info.get("description", ""),
            "parameters": tool_info.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert a list of tool-info dicts to OpenAI format."""
    return [_to_openai_tool(t) for t in tools]


def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]["function"](**tool_input)
        return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


# ── Core tool-calling loop ────────────────────────────────────────────────────

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    client,
    model,
    messages,
    tools,
    tool_choice,
    logging=None,
    max_retry=3,
):
    """
    Single LLM call with tool definitions.

    All models now use the OpenAI-compatible chat completions API.
    Anthropic-format tool definitions are converted automatically.
    """
    if logging is None:
        logging = print

    # Convert Anthropic → OpenAI tool format if needed
    if tools and isinstance(tools[0], dict) and "input_schema" in tools[0]:
        tools = _to_openai_tools(tools)

    # Normalise tool_choice: Anthropic uses {"type": "auto"} — OpenAI accepts "auto"
    if isinstance(tool_choice, dict) and "type" in tool_choice:
        tool_choice = tool_choice["type"]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        logging(f"Error in get_response_withtools: {str(e)}")
        if max_retry > 0:
            return get_response_withtools(
                client, model, messages, tools, tool_choice, logging, max_retry - 1
            )
        raise


def _extract_tool_calls_from_response(response) -> tuple[str | None, list[dict]]:
    """
    Extract text content and tool call list from an OpenAI chat completion response.
    Returns (text_content, tool_calls_list).
    """
    choice = response.choices[0]
    msg = choice.message
    text = msg.content or ""
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            })
    return text, tool_calls


def chat_with_agent(
    instruction: str,
    model: str = SOVEREIGN_MODEL,
    msg_history: list | None = None,
    logging=None,
    system_message: str | None = None,
    max_iterations: int = 50,
) -> list[dict]:
    """
    Agentic chat loop: drives the model in a ReAct-style tool-use loop until
    the model produces a final text-only response or max_iterations is reached.
    """
    if logging is None:
        logging = print
    if msg_history is None:
        msg_history = []

    client_tuple = create_client(model)
    client, client_model = client_tuple

    all_tools = load_all_tools(logging=logging)
    tools_dict = {t["name"]: t for t in all_tools}
    tool_defs = [t["info"] for t in all_tools]

    if system_message is None:
        system_message = get_tooluse_prompt()

    messages = [{"role": "system", "content": system_message}]
    messages += msg_history
    messages.append({"role": "user", "content": instruction})

    for _ in range(max_iterations):
        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=messages,
            tools=tool_defs,
            tool_choice="auto",
            logging=logging,
        )

        text, tool_calls = _extract_tool_calls_from_response(response)

        if not tool_calls:
            # Final text response — loop ends
            messages.append({"role": "assistant", "content": text})
            logging(f"[Agent] Final response: {text[:200]}")
            break

        # Record assistant turn with tool calls
        messages.append(response.choices[0].message)

        # Execute each tool and feed results back
        for tc in tool_calls:
            result = process_tool_call(tools_dict, tc["name"], tc["input"])
            logging(f"[Tool] {tc['name']}({tc['input']}) → {str(result)[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result),
            })

    return messages
