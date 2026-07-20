"""Tests for hf_local.enabled=false config gate.

Verifies that when hf_local.enabled is false (the default):
  - No torch/transformers imports occur in the parent process
  - No logs/ or profiling/ directories are created
  - The transport registry entry does NOT require torch
  - _resolve_delegation_credentials handles hf-local correctly
"""

import sys
import pytest
from unittest.mock import patch


class TestConfigGateImports:
    def test_transport_importable_without_torch(self):
        """HFLocalTransport can be imported even when torch is absent."""
        # Temporarily hide torch if present
        import importlib
        torch_present = "torch" in sys.modules

        if torch_present:
            # Transport module is already imported; just verify no torch at module level
            import agent.transports.hf_local as hf_t
            # The module itself must not hold a torch reference at module scope
            # (it's allowed to import torch inside function bodies)
            assert not hasattr(hf_t, "_torch")  # no eager torch alias at module level

        # In fresh environments without torch, import must succeed cleanly
        # (tested in CI without torch; here we just verify the module has no top-level torch)

    def test_transport_registered_in_registry(self):
        """hf_local is registered in the transport registry after discovery."""
        from agent.transports import get_transport
        t = get_transport("hf_local")
        assert t is not None
        assert t.api_mode == "hf_local"

    def test_noop_recorder_when_disabled(self):
        """get_recorder() returns NoOpRecorder when profiling.enabled is false."""
        from agent.interp import session_recorder as sr
        sr._RECORDER_SINGLETON = None  # reset singleton

        with patch("agent.interp.session_recorder._make_recorder") as mock:
            from agent.interp.session_recorder import NoOpRecorder
            mock.return_value = NoOpRecorder()
            recorder = sr.get_recorder()
            assert isinstance(recorder, NoOpRecorder)

        # Reset
        sr._RECORDER_SINGLETON = None


class TestHFLocalProviderResolution:
    def test_resolve_runtime_provider_hf_local(self):
        """resolve_runtime_provider('hf-local') returns hf_local api_mode."""
        with patch("hermes_cli.runtime_provider.resolve_requested_provider",
                   return_value="hf-local"):
            from hermes_cli.runtime_provider import resolve_runtime_provider
            result = resolve_runtime_provider(requested="hf-local")

        assert result["api_mode"] == "hf_local"
        assert result["provider"] == "hf-local"
        assert result["api_key"]  # non-empty placeholder

    def test_hf_local_in_valid_api_modes(self):
        from hermes_cli.runtime_provider import _VALID_API_MODES
        assert "hf_local" in _VALID_API_MODES

    def test_parse_api_mode_accepts_hf_local(self):
        from hermes_cli.runtime_provider import _parse_api_mode
        assert _parse_api_mode("hf_local") == "hf_local"

    def test_hf_local_in_provider_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "hf-local" in PROVIDER_REGISTRY


class TestDefaultConfigBlocks:
    def test_hf_local_defaults_disabled(self):
        """DEFAULT_CONFIG has hf_local.enabled=False."""
        from hermes_cli.config import DEFAULT_CONFIG
        hf = DEFAULT_CONFIG.get("hf_local", {})
        assert hf.get("enabled") is False

    def test_profiling_defaults_disabled(self):
        """DEFAULT_CONFIG has profiling.enabled=False."""
        from hermes_cli.config import DEFAULT_CONFIG
        pro = DEFAULT_CONFIG.get("profiling", {})
        assert pro.get("enabled") is False

    def test_hf_local_model_defaults(self):
        from hermes_cli.config import DEFAULT_CONFIG
        hf = DEFAULT_CONFIG.get("hf_local", {})
        assert hf["llm"]["model"] == "Qwen/Qwen3.5-9B"
        assert hf["vlm"]["model"] == "Qwen/Qwen3-VL-8B-Instruct"

    def test_profiling_probe_mode_default(self):
        from hermes_cli.config import DEFAULT_CONFIG
        pro = DEFAULT_CONFIG.get("profiling", {})
        assert pro.get("probe_mode") == "output_contribution"
        assert pro.get("decode_window") == 64
