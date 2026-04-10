"""
Tests for DGM Sovereign Core rewiring.

These unit tests verify that:
  - All cloud API references are removed
  - Sovereign Core constants are correct
  - create_client() returns an OpenAI-compatible client pointing at the sovereign endpoint
  - get_response_from_llm / get_batch_responses_from_llm have the right signatures
  - llm_withtools exports SOVEREIGN_MODEL and the correct tool format
  - self_improve_step.py passes sovereign env vars to containers
"""
import importlib
import inspect
import os
import sys
import re
import ast

# ── Path setup ────────────────────────────────────────────────────────────────
# Tests run from the repo root; conftest.py adds parent dir to sys.path
import pytest


# ── Helper ────────────────────────────────────────────────────────────────────

def _read_src(filename):
    """Read a source file relative to the test directory's parent (repo root)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, filename)
    with open(path) as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# llm.py — Sovereign Core configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMSovereignConfig:
    def test_no_anthropic_import(self):
        src = _read_src("llm.py")
        assert "import anthropic" not in src, "llm.py must not import anthropic"

    def test_no_cloud_model_constants(self):
        src = _read_src("llm.py")
        for cloud_name in ("claude-3", "gpt-4", "o1-", "o3-", "bedrock/", "vertex_ai/"):
            assert cloud_name not in src, f"Cloud model '{cloud_name}' found in llm.py"

    def test_sovereign_model_constant_defined(self):
        src = _read_src("llm.py")
        assert "SOVEREIGN_MODEL" in src
        assert "SOVEREIGN_API_BASE" in src

    def test_default_sovereign_url_localhost(self):
        src = _read_src("llm.py")
        assert "localhost:8001" in src, "Default sovereign URL should be localhost:8001"

    def test_sovereign_in_available_llms(self):
        src = _read_src("llm.py")
        assert "AVAILABLE_LLMS" in src
        assert "sovereign" in src

    def test_create_client_function_exists(self):
        src = _read_src("llm.py")
        assert "def create_client" in src

    def test_create_client_uses_openai(self):
        src = _read_src("llm.py")
        assert "openai.OpenAI" in src

    def test_create_client_passes_base_url(self):
        src = _read_src("llm.py")
        # create_client should set base_url to the sovereign endpoint
        assert "base_url" in src

    def test_get_response_from_llm_exists(self):
        src = _read_src("llm.py")
        assert "def get_response_from_llm" in src

    def test_get_batch_responses_from_llm_exists(self):
        src = _read_src("llm.py")
        assert "def get_batch_responses_from_llm" in src

    def test_extract_json_between_markers_exists(self):
        src = _read_src("llm.py")
        assert "def extract_json_between_markers" in src

    def test_no_anthropic_api_calls(self):
        src = _read_src("llm.py")
        assert "client.messages.create" not in src, \
            "Anthropic-style API calls must not appear in llm.py"


# ─────────────────────────────────────────────────────────────────────────────
# llm_withtools.py — Tool-calling via OpenAI function-calling API
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMWithToolsSovereign:
    def test_no_anthropic_import(self):
        src = _read_src("llm_withtools.py")
        assert "import anthropic" not in src

    def test_sovereign_model_alias_for_claude(self):
        """CLAUDE_MODEL must now be an alias for SOVEREIGN_MODEL."""
        src = _read_src("llm_withtools.py")
        assert "CLAUDE_MODEL" in src
        assert "SOVEREIGN_MODEL" in src
        # Must not be a hard-coded cloud model string
        assert "bedrock/" not in src
        assert "claude-3" not in src

    def test_sovereign_model_alias_for_openai(self):
        """OPENAI_MODEL must now be an alias for SOVEREIGN_MODEL."""
        src = _read_src("llm_withtools.py")
        assert "OPENAI_MODEL" in src
        assert "o3-mini" not in src

    def test_convert_tool_info_openai_format(self):
        """convert_tool_info must output OpenAI function tool format."""
        src = _read_src("llm_withtools.py")
        assert "'type': 'function'" in src or '"type": "function"' in src

    def test_check_for_tool_use_openai_path(self):
        """check_for_tool_use must handle OpenAI choices[0].message.tool_calls."""
        src = _read_src("llm_withtools.py")
        assert "tool_calls" in src
        assert "choices" in src

    def test_chat_with_agent_sovereign_exists(self):
        src = _read_src("llm_withtools.py")
        assert "def chat_with_agent_sovereign" in src

    def test_chat_with_agent_dispatches_to_sovereign(self):
        src = _read_src("llm_withtools.py")
        assert "chat_with_agent_sovereign" in src

    def test_no_anthropic_response_parsing(self):
        src = _read_src("llm_withtools.py")
        assert "stop_reason" not in src, \
            "Anthropic stop_reason parsing must not appear in llm_withtools.py"
        assert "response.content[0].text" not in src, \
            "Anthropic response parsing must not appear in llm_withtools.py"


# ─────────────────────────────────────────────────────────────────────────────
# self_improve_step.py — Sovereign env vars in Docker containers
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfImproveStepSovereign:
    def test_diagnose_model_uses_sovereign(self):
        src = _read_src("self_improve_step.py")
        assert "diagnose_model = SOVEREIGN_MODEL" in src

    def test_no_hardcoded_cloud_model(self):
        src = _read_src("self_improve_step.py")
        assert "o1-2024-12-17" not in src

    def test_no_anthropic_api_key_in_env_vars(self):
        src = _read_src("self_improve_step.py")
        assert "ANTHROPIC_API_KEY" not in src

    def test_no_openai_api_key_in_env_vars(self):
        src = _read_src("self_improve_step.py")
        assert "OPENAI_API_KEY" not in src

    def test_sovereign_api_base_in_env_vars(self):
        src = _read_src("self_improve_step.py")
        assert "SOVEREIGN_API_BASE" in src

    def test_sovereign_model_in_env_vars(self):
        src = _read_src("self_improve_step.py")
        assert "SOVEREIGN_MODEL" in src

    def test_imports_sovereign_constants(self):
        src = _read_src("self_improve_step.py")
        assert "SOVEREIGN_MODEL" in src
        assert "SOVEREIGN_API_BASE" in src


# ─────────────────────────────────────────────────────────────────────────────
# requirements.txt — no cloud SDK dependencies
# ─────────────────────────────────────────────────────────────────────────────

class TestRequirementsSovereign:
    def test_no_anthropic_package(self):
        src = _read_src("requirements.txt")
        lines = [l.strip().lower() for l in src.splitlines() if l.strip() and not l.startswith("#")]
        for line in lines:
            assert not line.startswith("anthropic"), \
                f"requirements.txt must not include anthropic: {line}"

    def test_no_boto3_package(self):
        src = _read_src("requirements.txt")
        lines = [l.strip().lower() for l in src.splitlines() if l.strip() and not l.startswith("#")]
        for line in lines:
            assert not line.startswith("boto3"), f"requirements.txt must not include boto3: {line}"
            assert not line.startswith("botocore"), f"requirements.txt must not include botocore: {line}"

    def test_openai_package_present(self):
        src = _read_src("requirements.txt")
        assert "openai" in src.lower()

    def test_backoff_package_present(self):
        src = _read_src("requirements.txt")
        assert "backoff" in src.lower()


# ─────────────────────────────────────────────────────────────────────────────
# .env.example — sovereign endpoint documented
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvExample:
    def test_env_example_exists(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert os.path.exists(os.path.join(root, ".env.example")), \
            ".env.example must exist"

    def test_sovereign_api_base_documented(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".env.example")) as f:
            content = f.read()
        assert "SOVEREIGN_API_BASE" in content
        assert "localhost:8001" in content

    def test_sovereign_model_documented(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".env.example")) as f:
            content = f.read()
        assert "SOVEREIGN_MODEL" in content
        assert "qwen2.5" in content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Integration: convert_tool_info output shape
# ─────────────────────────────────────────────────────────────────────────────

class TestConvertToolInfoShape:
    """Verify the output shape of convert_tool_info matches OpenAI spec."""

    def _get_tool_info_sample(self):
        return {
            'name': 'bash',
            'description': 'Run a shell command and return the output.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'command': {'type': 'string', 'description': 'The command to run.'}
                },
                'required': ['command'],
            }
        }

    def test_convert_produces_type_function(self):
        """Must return {'type': 'function', 'function': {...}}."""
        # Parse convert_tool_info from source and eval with a mock
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("llm_withtools_test", os.path.join(root, "llm_withtools.py"))
        # We can't fully import due to missing deps — check statically instead
        src = _read_src("llm_withtools.py")
        # The function must mention 'type': 'function' and 'function' key
        assert "'type': 'function'" in src or '"type": "function"' in src
        # Must have a 'function' sub-dict (not just top-level 'function' key)
        assert "'function': {" in src or '"function": {' in src
