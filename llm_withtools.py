import json
import re
import backoff
import openai
import copy

from llm import create_client, get_response_from_llm, SOVEREIGN_MODEL
from prompts.tooluse_prompt import get_tooluse_prompt
from tools import load_all_tools

# ---------------------------------------------------------------------------
# Sovereign Core: single model for all agent interactions
# ---------------------------------------------------------------------------
CLAUDE_MODEL = SOVEREIGN_MODEL   # backward-compat alias used by coding_agent.py
OPENAI_MODEL = SOVEREIGN_MODEL   # backward-compat alias


def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError),
    max_time=600,
    max_value=60,
)
def get_response_withtools(
    client, model, messages, tools, tool_choice,
    logging=None, max_retry=3
):
    """
    Call the Sovereign Core endpoint with OpenAI-compatible tool calling.

    vLLM + Qwen2.5 supports the OpenAI function-calling API natively.
    tool_choice can be "auto", "none", or {"type": "function", "function": {"name": "..."}}.
    """
    try:
        # Normalise tool_choice from Anthropic format to OpenAI format
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "auto":
            tool_choice = "auto"

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            tool_choice=tool_choice,
            tools=tools,
        )
        return response
    except Exception as e:
        if logging:
            logging(f"Error in get_response_withtools: {str(e)}")
        if max_retry > 0:
            return get_response_withtools(
                client, model, messages, tools, tool_choice, logging, max_retry - 1
            )
        raise


def check_for_tool_use(response, model=''):
    """
    Check if the OpenAI-compatible response contains a tool call.

    Returns a dict with 'tool_id', 'tool_name', 'tool_input', or None.
    """
    # OpenAI-compatible response (covers sovereign / vLLM)
    if hasattr(response, 'choices') and response.choices:
        msg = response.choices[0].message
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tc = msg.tool_calls[0]
            return {
                'tool_id': tc.id,
                'tool_name': tc.function.name,
                'tool_input': json.loads(tc.function.arguments),
            }
        return None

    # Fallback: plain string response with <tool_use> tag (manual tool calling)
    if isinstance(response, str):
        pattern = r'<tool_use>(.*?)</tool_use>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            tool_use_str = match.group(1).strip()
            try:
                import ast
                tool_use_dict = ast.literal_eval(tool_use_str)
                if isinstance(tool_use_dict, dict) and 'tool_name' in tool_use_dict:
                    return tool_use_dict
            except Exception:
                pass

    return None


def convert_tool_info(tool_info, model=None):
    """
    Convert tool_info (in Anthropic/Claude format) to OpenAI function-tool format.

    vLLM's Qwen2.5 endpoint accepts OpenAI function tools natively.
    """
    return {
        'type': 'function',
        'function': {
            'name': tool_info['name'],
            'description': tool_info['description'],
            'parameters': tool_info['input_schema'],
        }
    }


def convert_block_claude(block):
    """Convert a single Claude content block to generic format."""
    if isinstance(block, dict):
        block_type = block.get('type')
        text = block.get('text', '')
        tool_name = block.get('name')
        tool_input = block.get('input')
        tool_result = block.get('content')
    else:
        block_type = getattr(block, 'type', None)
        text = getattr(block, 'text', '') or ''
        tool_name = getattr(block, 'name', None)
        tool_input = getattr(block, 'input', None)
        tool_result = getattr(block, 'content', None)

    if block_type == "text":
        return {"type": "text", "text": text}
    elif block_type == "tool_use":
        return {
            "type": "text",
            "text": f"<tool_use>\n{{'tool_name': {tool_name}, 'tool_input': {tool_input}}}\n</tool_use>"
        }
    elif block_type == "tool_result":
        return {"type": "text", "text": f"Tool Result: {tool_result}"}
    else:
        return {"type": "text", "text": str(block)}


def convert_msg_history_claude(msg_history):
    new_msg_history = []
    for msg in msg_history:
        role = msg.get('role', '')
        content_blocks = msg.get('content', [])
        new_content = [convert_block_claude(b) for b in content_blocks]
        new_msg_history.append({"role": role, "content": new_content})
    return new_msg_history


def convert_msg_history_openai(msg_history):
    """Convert OpenAI-style message history (including tool_calls) to generic format."""
    new_msg_history = []
    for msg in msg_history:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'tool':
                new_msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Tool Result: {content}"}],
                }
            else:
                new_msg = {"role": role, "content": content}
        else:
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', None)
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                tc = tool_calls[0]
                new_msg = {
                    "role": role,
                    "content": [{
                        "type": "text",
                        "text": f"<tool_use>\n{{'tool_name': {tc.function.name}, "
                                f"'tool_input': {tc.function.arguments}}}\n</tool_use>"
                    }],
                }
            else:
                new_msg = {"role": role, "content": [{"type": "text", "text": content}]}
        new_msg_history.append(new_msg)
    return new_msg_history


def convert_msg_history(msg_history, model=None):
    """Convert model-specific message history to generic format."""
    return convert_msg_history_openai(msg_history)


def chat_with_agent_manualtools(msg, model, msg_history=None, logging=print):
    """Fallback: tool calling via <tool_use> text tags (no native tool API)."""
    if msg_history is None:
        msg_history = []
    system_message = f'You are a coding agent.\n\n{get_tooluse_prompt()}'
    new_msg_history = msg_history

    try:
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        client, client_model = create_client(model)

        logging(f"Input: {msg}")
        response, new_msg_history = get_response_from_llm(
            msg=msg, client=client, model=client_model,
            system_message=system_message, print_debug=False,
            msg_history=new_msg_history,
        )
        logging(f"Output: {response}")

        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)
            tool_msg = f'Tool Used: {tool_name}\nTool Input: {tool_input}\nTool Result: {tool_result}'
            logging(tool_msg)
            response, new_msg_history = get_response_from_llm(
                msg=tool_msg, client=client, model=client_model,
                system_message=system_message, print_debug=False,
                msg_history=new_msg_history,
            )
            logging(f"Output: {response}")
            tool_use = check_for_tool_use(response, model=client_model)

    except Exception:
        pass

    return new_msg_history


def chat_with_agent_sovereign(
        msg,
        model=None,
        msg_history=None,
        logging=print,
):
    """
    Chat with the Sovereign Core agent using OpenAI-compatible tool calling.

    vLLM serving Qwen2.5-32B-AWQ supports function-calling via the
    standard OpenAI /v1/chat/completions endpoint.
    """
    if model is None:
        model = SOVEREIGN_MODEL
    if msg_history is None:
        msg_history = []

    separator = '=' * 10
    logging(f"\n{separator} User Instruction {separator}\n{msg}")

    new_msg_history = [{"role": "user", "content": msg}]

    try:
        client, client_model = create_client(model)
        all_tools = load_all_tools(logging=logging)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        tools = [convert_tool_info(tool['info'], model=client_model) for tool in all_tools]

        response = get_response_withtools(
            client=client,
            model=client_model,
            messages=msg_history + new_msg_history,
            tool_choice="auto",
            tools=tools,
            logging=logging,
        )
        logging(f"\n{separator} Agent Response {separator}\n{response}")

        tool_use = check_for_tool_use(response, model=client_model)
        while tool_use:
            tool_name = tool_use['tool_name']
            tool_input = tool_use['tool_input']
            tool_result = process_tool_call(tools_dict, tool_name, tool_input)

            logging(f"Tool Used: {tool_name}")
            logging(f"Tool Input: {tool_input}")
            logging(f"Tool Result: {tool_result}")

            # Append assistant message with tool_calls, then tool result
            assistant_msg = response.choices[0].message
            new_msg_history.append({
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tool_use['tool_id'],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input),
                        }
                    }
                ],
            })
            new_msg_history.append({
                "role": "tool",
                "tool_call_id": tool_use['tool_id'],
                "content": str(tool_result),
            })

            response = get_response_withtools(
                client=client,
                model=client_model,
                messages=msg_history + new_msg_history,
                tool_choice="auto",
                tools=tools,
                logging=logging,
            )
            tool_use = check_for_tool_use(response, model=client_model)
            logging(f"Tool Response: {response}")

        # Append final assistant message
        final_content = response.choices[0].message.content or ""
        new_msg_history.append({"role": "assistant", "content": final_content})

    except Exception:
        pass

    return msg_history + new_msg_history


def chat_with_agent(
    msg,
    model=SOVEREIGN_MODEL,
    msg_history=None,
    logging=print,
    convert=False,
):
    """
    Unified chat dispatcher. Routes all calls to the Sovereign Core backend.
    """
    if msg_history is None:
        msg_history = []

    new_msg_history = chat_with_agent_sovereign(
        msg, model=model, msg_history=msg_history, logging=logging
    )
    if convert:
        new_msg_history = convert_msg_history(new_msg_history, model=model)

    return new_msg_history


if __name__ == "__main__":
    msg = "hello!"
    chat_with_agent(msg)
