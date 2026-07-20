"""HF-local provider transport.

ProviderTransport for api_mode='hf_local'. Used when sub-agents or aux tasks
are routed to in-process HuggingFace models via HFWorkerPool.

The transport converts OpenAI-format messages/tools into the form that the
pool's worker can consume (already OpenAI-format; HF chat templates are
applied inside the worker's LocalModel.generate()).

normalize_response handles text -> NormalizedResponse, including:
  - Qwen3 <tool_call>...</tool_call> tool-call parsing
  - Usage extraction

This module is imported at startup by _discover_transports() so it must NOT
import torch/transformers at module level.  All heavy imports are deferred to
method bodies or to the pool (which lives in worker processes).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports import register_transport
from agent.transports.types import NormalizedResponse, ToolCall, Usage

# Qwen3/Qwen3.5 tool-call output format
# <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _parse_tool_calls(text: str) -> tuple:
    """Extract <tool_call> blocks from model output.

    Returns (remaining_text_without_blocks, tool_calls).
    """
    tool_calls: List[ToolCall] = []

    for m in _TOOL_CALL_RE.finditer(text):
        raw_json = m.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            name = parsed.get("name") or parsed.get("function") or ""
            arguments = parsed.get("arguments") or parsed.get("parameters") or {}
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            tc = ToolCall(
                id=f"hf-{len(tool_calls)}",
                name=str(name),
                arguments=str(arguments),
            )
            tool_calls.append(tc)
        except (json.JSONDecodeError, KeyError):
            pass

    if tool_calls:
        stripped = _TOOL_CALL_RE.sub("", text).strip()
    else:
        stripped = text

    return stripped, tool_calls


class HFLocalTransport(ProviderTransport):
    """Transport for api_mode='hf_local'.

    Converts messages/tools to/from OpenAI format (near-identity since HF
    chat templates accept OpenAI-style message dicts).  Actual generation
    is delegated to HFWorkerPool; the shim client (HFLocalClientShim) calls
    the pool and wraps the result in a fake response object.
    """

    @property
    def api_mode(self) -> str:
        return "hf_local"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        # OpenAI-format messages are passed through to the worker directly.
        return messages

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        # Tools are passed through; the worker injects them into the chat template.
        return tools

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build kwargs dict for HFLocalClientShim.create()."""
        return {
            "model":    model,
            "messages": messages,
            "tools":    tools,
            **params,
        }

    def normalize_response(
        self,
        response: Any,
        **kwargs,
    ) -> NormalizedResponse:
        """Normalize a response from HFLocalClientShim.

        The shim sets response._hf_local=True and attaches:
            response.text, response.usage, response.timings, response.probe_rows
        """
        if getattr(response, "_hf_local", False):
            text       = response.text or ""
            raw_usage  = response.usage or {}
            probe_rows = response.probe_rows or []
            timings    = response.timings or {}
        elif hasattr(response, "choices") and response.choices:
            msg  = response.choices[0].message
            text = getattr(msg, "content", "") or ""
            raw_usage  = {}
            probe_rows = []
            timings    = {}
        else:
            text       = str(response)
            raw_usage  = {}
            probe_rows = []
            timings    = {}

        content, tool_calls = _parse_tool_calls(text)
        finish_reason = "tool_calls" if tool_calls else "stop"

        usage = Usage(
            prompt_tokens     = raw_usage.get("prompt_tokens",     0),
            completion_tokens = raw_usage.get("completion_tokens", 0),
            total_tokens      = raw_usage.get("total_tokens",      0),
        )

        provider_data: Dict[str, Any] = {}
        if timings:
            provider_data["hf_timings"]    = timings
        if probe_rows:
            provider_data["hf_probe_rows"] = probe_rows

        return NormalizedResponse(
            content       = content or None,
            tool_calls    = tool_calls or None,
            finish_reason = finish_reason,
            usage         = usage,
            provider_data = provider_data or None,
        )


register_transport("hf_local", HFLocalTransport)
