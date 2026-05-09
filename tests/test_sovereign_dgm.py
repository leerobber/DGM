"""
tests/test_sovereign_dgm.py

Unit tests for the Sovereign Core rewiring of jennyzzt/dgm.
All tests run without a live LLM connection (mocks only).
"""
import importlib
import json
import os
import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to import the rewired modules without importing docker/SWE-bench
# ---------------------------------------------------------------------------

REWRITE_DIR = os.path.join(os.path.dirname(__file__), "..")
if REWRITE_DIR not in sys.path:
    sys.path.insert(0, REWRITE_DIR)

# Stub heavy optional imports so the modules load in CI without full deps
for _mod in [
    "docker", "datasets", "unidiff", "ghapi", "ghapi.all",
    "rich", "pre_commit", "git",
    "swe_bench", "swe_bench.harness", "swe_bench.report",
    "polyglot", "polyglot.harness",
    "utils", "utils.common_utils", "utils.docker_utils",
    "utils.evo_utils", "utils.git_utils", "utils.eval_utils",
    "prompts", "prompts.self_improvement_prompt",
    "prompts.diagnose_improvement_prompt", "prompts.testrepo_prompt",
    "prompts.tooluse_prompt",
    "tools",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# tooluse_prompt stub
sys.modules["prompts.tooluse_prompt"].get_tooluse_prompt = lambda: "Use the tools."

# tools stub — returns one fake bash tool
_fake_tool = {
    "info": {
        "name": "bash",
        "description": "Run bash commands.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    "function": lambda command="": f"$ {command}\nok",
    "name": "bash",
}
sys.modules["tools"].load_all_tools = lambda logging=print: [_fake_tool]


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
import llm as llm_mod
import llm_withtools as lwt_mod


# ===========================================================================
# TestLLMConfig
# ===========================================================================
class TestLLMConfig:
    def test_sovereign_env_vars_have_defaults(self):
        assert llm_mod.SOVEREIGN_API_BASE == os.environ.get(
            "SOVEREIGN_API_BASE", "http://localhost:8001/v1"
        )
        assert llm_mod.SOVEREIGN_MODEL == os.environ.get(
            "SOVEREIGN_MODEL", "qwen2.5-32b-awq"
        )
        assert llm_mod.SOVEREIGN_API_KEY == os.environ.get(
            "SOVEREIGN_API_KEY", "sovereign"
        )

    def test_sovereign_model_in_available_llms(self):
        assert "sovereign/qwen2.5-32b-awq" in llm_mod.AVAILABLE_LLMS

    def test_no_anthropic_in_available_llms(self):
        for m in llm_mod.AVAILABLE_LLMS:
            assert "claude" not in m.lower(), f"Unexpected Claude model: {m}"
            assert "bedrock" not in m.lower(), f"Unexpected Bedrock model: {m}"
            assert "vertex" not in m.lower(), f"Unexpected Vertex model: {m}"

    def test_max_output_tokens(self):
        assert llm_mod.MAX_OUTPUT_TOKENS == 4096


# ===========================================================================
# TestCreateClient
# ===========================================================================
class TestCreateClient:
    def test_sovereign_prefix_creates_openai_client(self, monkeypatch):
        captured = {}

        def fake_openai(base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            return MagicMock(name="sovereign_client")

        monkeypatch.setattr(llm_mod.openai, "OpenAI", fake_openai)
        client, model = llm_mod.create_client("sovereign/qwen2.5-32b-awq")
        assert model == "qwen2.5-32b-awq"
        assert captured["base_url"] == llm_mod.SOVEREIGN_API_BASE

    def test_sovereign_strips_prefix_from_model_name(self, monkeypatch):
        monkeypatch.setattr(llm_mod.openai, "OpenAI", lambda **kw: MagicMock())
        _, model = llm_mod.create_client("sovereign/custom-model-name")
        assert model == "custom-model-name"

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            llm_mod.create_client("anthropic/claude-3")

    def test_gpt_routes_to_openai_cloud(self, monkeypatch):
        captured = {}

        def fake_openai(**kw):
            captured.update(kw)
            return MagicMock()

        monkeypatch.setattr(llm_mod.openai, "OpenAI", fake_openai)
        client, model = llm_mod.create_client("gpt-4o-2024-05-13")
        assert model == "gpt-4o-2024-05-13"
        # cloud path: no base_url / api_key override
        assert "base_url" not in captured


# ===========================================================================
# TestGetResponseFromLLM
# ===========================================================================
class TestGetResponseFromLLM:
    def _mock_response(self, text: str):
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_returns_content_string(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._mock_response("hello")
        content, history = llm_mod.get_response_from_llm(
            "ping", fake_client, "qwen2.5-32b-awq", "system msg"
        )
        assert content == "hello"

    def test_appends_user_and_assistant_to_history(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._mock_response("resp")
        _, history = llm_mod.get_response_from_llm(
            "user msg", fake_client, "qwen2.5-32b-awq", "sys"
        )
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles

    def test_system_message_in_api_call(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._mock_response("ok")
        llm_mod.get_response_from_llm("q", fake_client, "m", "SYSTEM_CONTENT")
        call_kwargs = fake_client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"]
        assert messages[0] == {"role": "system", "content": "SYSTEM_CONTENT"}

    def test_existing_history_preserved(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = self._mock_response("ans")
        prior = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "prev_ans"}]
        _, history = llm_mod.get_response_from_llm("new", fake_client, "m", "sys", msg_history=prior)
        assert history[0]["content"] == "prev"


# ===========================================================================
# TestExtractJSON
# ===========================================================================
class TestExtractJSON:
    def test_extract_from_json_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nmore'
        result = llm_mod.extract_json_between_markers(text)
        assert result == {"key": "value"}

    def test_fallback_bare_json(self):
        text = 'No block but here: {"a": 1}'
        result = llm_mod.extract_json_between_markers(text)
        assert result == {"a": 1}

    def test_returns_none_for_no_json(self):
        result = llm_mod.extract_json_between_markers("just plain text, no json here")
        assert result is None

    def test_nested_json_object(self):
        text = '```json\n{"outer": {"inner": 42}}\n```'
        result = llm_mod.extract_json_between_markers(text)
        assert result["outer"]["inner"] == 42


# ===========================================================================
# TestLLMWithTools
# ===========================================================================
class TestLLMWithTools:
    def test_sovereign_model_constant_has_prefix(self):
        assert lwt_mod.SOVEREIGN_MODEL.startswith("sovereign/")

    def test_claude_model_alias_points_to_sovereign(self):
        assert lwt_mod.CLAUDE_MODEL == lwt_mod.SOVEREIGN_MODEL

    def test_openai_model_alias_points_to_sovereign(self):
        assert lwt_mod.OPENAI_MODEL == lwt_mod.SOVEREIGN_MODEL


# ===========================================================================
# TestToOpenAITool
# ===========================================================================
class TestToOpenAITool:
    def test_converts_anthropic_format(self):
        anthropic_tool = {
            "name": "bash",
            "description": "Run bash.",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
        }
        oa = lwt_mod._to_openai_tool(anthropic_tool)
        assert oa["type"] == "function"
        assert oa["function"]["name"] == "bash"
        assert "parameters" in oa["function"]
        assert "input_schema" not in oa["function"]

    def test_batch_conversion(self):
        tools = [
            {"name": "t1", "description": "d1", "input_schema": {"type": "object", "properties": {}}},
            {"name": "t2", "description": "d2", "input_schema": {"type": "object", "properties": {}}},
        ]
        converted = lwt_mod._to_openai_tools(tools)
        assert len(converted) == 2
        for t in converted:
            assert t["type"] == "function"

    def test_missing_description_defaults_to_empty(self):
        tool = {"name": "x", "input_schema": {"type": "object", "properties": {}}}
        oa = lwt_mod._to_openai_tool(tool)
        assert oa["function"]["description"] == ""


# ===========================================================================
# TestGetResponseWithtools
# ===========================================================================
class TestGetResponseWithtools:
    def _build_mock_client(self, content="ok", tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls or []
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        return client, resp

    def test_returns_response_object(self):
        client, expected = self._build_mock_client()
        result = lwt_mod.get_response_withtools(
            client, "qwen2.5-32b-awq", [], [], "auto"
        )
        assert result is expected

    def test_converts_anthropic_tool_format(self):
        client, _ = self._build_mock_client()
        anthropic_tools = [
            {"name": "bash", "description": "d", "input_schema": {"type": "object", "properties": {}}}
        ]
        lwt_mod.get_response_withtools(client, "m", [], anthropic_tools, "auto")
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["tools"][0]["type"] == "function"

    def test_normalises_anthropic_tool_choice(self):
        client, _ = self._build_mock_client()
        lwt_mod.get_response_withtools(client, "m", [], [], {"type": "auto"})
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "auto"


# ===========================================================================
# TestExtractToolCallsFromResponse
# ===========================================================================
class TestExtractToolCallsFromResponse:
    def _make_response(self, content=None, tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_no_tool_calls(self):
        resp = self._make_response(content="hello", tool_calls=None)
        text, calls = lwt_mod._extract_tool_calls_from_response(resp)
        assert text == "hello"
        assert calls == []

    def test_extracts_tool_call(self):
        tc = MagicMock()
        tc.id = "tc_1"
        tc.function.name = "bash"
        tc.function.arguments = json.dumps({"command": "ls"})
        resp = self._make_response(content=None, tool_calls=[tc])
        text, calls = lwt_mod._extract_tool_calls_from_response(resp)
        assert calls[0]["name"] == "bash"
        assert calls[0]["input"] == {"command": "ls"}
        assert calls[0]["id"] == "tc_1"


# ===========================================================================
# TestChatWithAgent (integration-style, fully mocked)
# ===========================================================================
class TestChatWithAgent:
    def test_returns_message_list(self, monkeypatch):
        # Patch create_client to return mock
        client = MagicMock()
        monkeypatch.setattr(lwt_mod, "create_client", lambda m: (client, "qwen2.5"))

        # Response with no tool calls → immediate finish
        msg = MagicMock()
        msg.content = "Done."
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        client.chat.completions.create.return_value = resp

        result = lwt_mod.chat_with_agent("do something")
        assert isinstance(result, list)
        assert any(m.get("content") == "Done." for m in result if isinstance(m, dict))

    def test_tool_execution_loop(self, monkeypatch):
        """Model calls bash tool once, then returns text response."""
        client = MagicMock()
        monkeypatch.setattr(lwt_mod, "create_client", lambda m: (client, "qwen2.5"))

        # First response: tool call
        tc = MagicMock()
        tc.id = "tc_1"
        tc.function.name = "bash"
        tc.function.arguments = json.dumps({"command": "echo hello"})
        msg1 = MagicMock()
        msg1.content = None
        msg1.tool_calls = [tc]
        choice1 = MagicMock()
        choice1.message = msg1
        resp1 = MagicMock()
        resp1.choices = [choice1]

        # Second response: final text
        msg2 = MagicMock()
        msg2.content = "All done."
        msg2.tool_calls = None
        choice2 = MagicMock()
        choice2.message = msg2
        resp2 = MagicMock()
        resp2.choices = [choice2]

        client.chat.completions.create.side_effect = [resp1, resp2]

        result = lwt_mod.chat_with_agent("run echo")
        assert client.chat.completions.create.call_count == 2


# ===========================================================================
# TestEnvDocumented
# ===========================================================================
class TestEnvDocumented:
    def test_env_example_exists(self):
        env_path = os.path.join(REWRITE_DIR, ".env.example")
        assert os.path.exists(env_path), ".env.example not found"

    def test_sovereign_api_base_documented(self):
        env_path = os.path.join(REWRITE_DIR, ".env.example")
        content = open(env_path).read()
        assert "SOVEREIGN_API_BASE" in content

    def test_sovereign_model_documented(self):
        env_path = os.path.join(REWRITE_DIR, ".env.example")
        content = open(env_path).read()
        assert "SOVEREIGN_MODEL" in content
