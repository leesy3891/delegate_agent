"""Test that the hf_local transport is registered and works correctly."""

import pytest


class TestHFLocalTransportRegistry:
    def test_get_transport_returns_instance(self):
        from agent.transports import get_transport
        transport = get_transport("hf_local")
        assert transport is not None

    def test_api_mode_property(self):
        from agent.transports import get_transport
        transport = get_transport("hf_local")
        assert transport.api_mode == "hf_local"

    def test_convert_messages_passthrough(self):
        from agent.transports import get_transport
        transport = get_transport("hf_local")
        msgs = [{"role": "user", "content": "hello"}]
        result = transport.convert_messages(msgs)
        assert result is msgs

    def test_convert_tools_passthrough(self):
        from agent.transports import get_transport
        transport = get_transport("hf_local")
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        result = transport.convert_tools(tools)
        assert result is tools

    def test_build_kwargs(self):
        from agent.transports import get_transport
        transport = get_transport("hf_local")
        kwargs = transport.build_kwargs("Qwen/Qwen3.5-9B", [{"role": "user", "content": "hi"}])
        assert "model"    in kwargs
        assert "messages" in kwargs

    def test_normalize_hf_response(self):
        from agent.transports import get_transport
        from agent.transports.types import NormalizedResponse
        transport = get_transport("hf_local")

        class _FakeResp:
            _hf_local  = True
            text       = "hello world"
            usage      = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
            timings    = {"e2e_s": 0.5, "compute_s": 0.4}
            probe_rows = []

        nr = transport.normalize_response(_FakeResp())
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == "hello world"
        assert nr.finish_reason == "stop"
        assert nr.usage.prompt_tokens == 10

    def test_normalize_tool_call_parsing(self):
        import json
        from agent.transports import get_transport
        transport = get_transport("hf_local")

        tool_call_json = json.dumps({
            "name": "read_file",
            "arguments": {"path": "/tmp/x.txt"},
        })

        class _FakeResp:
            _hf_local  = True
            text       = f"<tool_call>\n{tool_call_json}\n</tool_call>"
            usage      = {}
            timings    = {}
            probe_rows = []

        nr = transport.normalize_response(_FakeResp())
        assert nr.finish_reason == "tool_calls"
        assert nr.tool_calls is not None
        assert len(nr.tool_calls) == 1
        tc = nr.tool_calls[0]
        assert tc.name == "read_file"
        assert '"path"' in tc.arguments

    def test_normalize_multiple_tool_calls(self):
        import json
        from agent.transports import get_transport
        transport = get_transport("hf_local")

        tc1 = json.dumps({"name": "tool_a", "arguments": {"x": 1}})
        tc2 = json.dumps({"name": "tool_b", "arguments": {"y": 2}})

        class _FakeResp:
            _hf_local  = True
            text       = f"<tool_call>{tc1}</tool_call>\n<tool_call>{tc2}</tool_call>"
            usage      = {}
            timings    = {}
            probe_rows = []

        nr = transport.normalize_response(_FakeResp())
        assert len(nr.tool_calls) == 2
        names = {tc.name for tc in nr.tool_calls}
        assert names == {"tool_a", "tool_b"}
