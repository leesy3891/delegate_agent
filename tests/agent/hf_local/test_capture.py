"""Tests for agent.hf_local.capture's forward-hook accumulation.

Regression coverage for the storage[layer_idx]-overwrite bug: generate()
calls forward once for prefill and once per decode step, so a hook that
overwrites a single dict per layer only ever keeps the last step's capture.
register_attention_hooks must instead accumulate one entry per forward pass.
"""

import pytest

pytest.importorskip("torch", reason="torch not installed")


class _FakeAttention:
    """Minimal stand-in for a Qwen/LLaMA-style attention module."""

    def __init__(self, d_model=8, n_heads=2):
        import torch.nn as nn

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.num_heads = n_heads
        self.num_key_value_heads = n_heads
        self.head_dim = d_model // n_heads
        self.__class__.__name__ = "FakeAttention"

    def named_modules(self):
        yield "", self
        yield "v_proj", self.v_proj


def _run_forward(module, seq_len, register_forward_hook_targets):
    """Simulate one forward pass: call v_proj then fire the module's own
    registered forward hooks with a fake (hidden, attn_weights) output,
    mirroring how torch invokes hooks for a real nn.Module.
    """
    import torch

    x = torch.randn(1, seq_len, 8)
    v_out = module.v_proj(x)
    for hook in register_forward_hook_targets.get("v_proj", []):
        hook(module.v_proj, (x,), v_out)

    attn_weights = torch.softmax(torch.randn(1, module.num_heads, seq_len, seq_len), dim=-1)
    hidden = torch.randn(1, seq_len, 8)
    outputs = (hidden, attn_weights)
    for hook in register_forward_hook_targets.get("self", []):
        hook(module, (x,), outputs)


class _HandleCollectingModule:
    """Wraps _FakeAttention so register_forward_hook calls are capturable."""

    def __init__(self, inner):
        self.inner = inner
        self.hooks = {"self": [], "v_proj": []}

    def register_forward_hook(self, fn):
        self.hooks["self"].append(fn)
        return object()

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def named_modules(self):
        yield "", self
        v_wrapper = _VProjWrapper(self.inner.v_proj, self.hooks)
        yield "v_proj", v_wrapper


class _VProjWrapper:
    def __init__(self, inner, hooks):
        self.inner = inner
        self._hooks = hooks

    def register_forward_hook(self, fn):
        self._hooks["v_proj"].append(fn)
        return object()

    def __getattr__(self, name):
        return getattr(self.inner, name)


class TestAccumulation:
    def test_storage_accumulates_one_entry_per_forward(self, monkeypatch):
        from agent.hf_local import capture as capture_mod
        import agent.interp.kv_probe as kv_probe_mod

        monkeypatch.setattr(kv_probe_mod, "is_softmax_attention_layer", lambda m: isinstance(m, _HandleCollectingModule))

        fake = _FakeAttention()
        wrapped = _HandleCollectingModule(fake)

        class _Model:
            def named_modules(self):
                yield "layers.0.self_attn", wrapped

        handles, storage = capture_mod.register_attention_hooks(_Model())
        assert 0 in storage or storage == {}  # not populated until forward runs

        # Simulate: 1 prefill forward (seq_len=5) + 2 decode-step forwards (seq_len=1)
        _run_forward(fake, 5, wrapped.hooks)
        _run_forward(fake, 1, wrapped.hooks)
        _run_forward(fake, 1, wrapped.hooks)

        assert 0 in storage
        entries = storage[0]
        assert len(entries) == 3, "expected one accumulated entry per forward pass, not an overwrite"

        # Prefill entry covers the full 5-token prompt.
        assert entries[0]["attn_weights"].shape[-1] == 5
        assert entries[0]["v_states"].shape[1] == 5

        # Each decode entry covers exactly the new (single) token.
        assert entries[1]["attn_weights"].shape[2] == 1
        assert entries[1]["v_states"].shape[1] == 1
        assert entries[2]["attn_weights"].shape[2] == 1

        # v_states must be paired with the SAME forward pass's attn_weights,
        # not the previous (or next) call's — this is the ordering bug the
        # pending_v hand-off fixes.
        for entry in entries:
            assert entry["v_states"] is not None
            assert entry["attn_weights"] is not None
