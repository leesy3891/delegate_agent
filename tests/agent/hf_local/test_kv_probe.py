"""Tests for agent.interp.kv_probe.

Uses synthetic tensors with known GQA groupings to verify:
  - Correct per-KV-head cosine/l2 values
  - Correct shapes and number of probe rows
  - Prefill vs decode window aggregation
  - output_contribution vs value_projection probe modes
"""

import math
import pytest

pytest.importorskip("torch", reason="torch not installed")


def _make_tensors(n_q=4, n_kv=2, head_dim=8, d_model=32, seq_len=6):
    """Return synthetic (attn_weights, v_states, o_proj, block_output) tensors."""
    import torch

    torch.manual_seed(42)
    attn_weights = torch.softmax(torch.randn(1, n_q, seq_len, seq_len), dim=-1)
    v_states     = torch.randn(1, n_kv, seq_len, head_dim)
    o_proj       = torch.randn(d_model, n_q * head_dim)
    block_output = torch.randn(seq_len, d_model)

    return attn_weights, v_states, o_proj, block_output


class TestBuildGQAGroups:
    def test_equal_heads(self):
        from agent.interp.kv_probe import _build_gqa_groups
        groups = _build_gqa_groups(4, 4)
        assert groups == [[0], [1], [2], [3]]

    def test_gqa_2x(self):
        from agent.interp.kv_probe import _build_gqa_groups
        groups = _build_gqa_groups(4, 2)
        assert groups == [[0, 1], [2, 3]]

    def test_gqa_4x(self):
        from agent.interp.kv_probe import _build_gqa_groups
        groups = _build_gqa_groups(8, 2)
        assert groups == [[0, 1, 2, 3], [4, 5, 6, 7]]


class TestComputeProbeRows:
    def _run(self, phase="prefill", probe_mode="output_contribution", n_kv=2, seq_len=6):
        from agent.interp.kv_probe import compute_probe_rows

        attn_weights, v_states, o_proj, block_output = _make_tensors(
            n_q=4, n_kv=n_kv, head_dim=8, d_model=32, seq_len=seq_len
        )
        capture = {
            "attn_weights":  attn_weights,
            "v_states":      v_states,
            "o_proj_weight": o_proj,
            "num_q_heads":   4,
            "num_kv_heads":  n_kv,
            "head_dim":      8,
        }
        rows = compute_probe_rows(
            layer_idx=0,
            capture=capture,
            block_output=block_output,
            request_id="test",
            call_order=0,
            parallel_group="g0",
            is_parallel=False,
            tool="test_tool",
            model="test_model",
            phase=phase,
            decode_window=3,
            probe_mode=probe_mode,
        )
        return rows

    def test_prefill_row_count(self):
        rows = self._run(phase="prefill", n_kv=2)
        # One row per KV head for prefill (single aggregate)
        assert len(rows) == 2

    def test_decode_row_count(self):
        # seq_len=6, decode_window=3 → 2 windows, 2 kv_heads → 4 rows
        rows = self._run(phase="decode", n_kv=2, seq_len=6)
        assert len(rows) == 4

    def test_row_schema(self):
        rows = self._run(phase="prefill")
        row = rows[0]
        assert "cosine"   in row
        assert "l2"       in row
        assert "layer"    in row
        assert "kv_head"  in row
        assert "phase"    in row
        assert "seq_pos"  in row

    def test_cosine_bounded(self):
        rows = self._run(phase="prefill")
        for row in rows:
            assert -1.0 <= row["cosine"] <= 1.0, f"cosine={row['cosine']} out of range"

    def test_l2_nonnegative(self):
        rows = self._run(phase="prefill")
        for row in rows:
            assert row["l2"] >= 0.0

    def test_kv_head_indices(self):
        rows = self._run(phase="prefill", n_kv=3)
        kv_heads = {r["kv_head"] for r in rows}
        assert kv_heads == {0, 1, 2}

    def test_value_projection_mode(self):
        rows_oc  = self._run(phase="prefill", probe_mode="output_contribution")
        rows_vp  = self._run(phase="prefill", probe_mode="value_projection")
        # Both should produce the same number of rows
        assert len(rows_oc) == len(rows_vp)
        # But values will differ (different computation)
        cosines_oc = [r["cosine"] for r in rows_oc]
        cosines_vp = [r["cosine"] for r in rows_vp]
        assert cosines_oc != cosines_vp

    def test_missing_v_states_returns_empty(self):
        from agent.interp.kv_probe import compute_probe_rows
        import torch

        _, _, o_proj, block_output = _make_tensors()
        capture = {
            "attn_weights":  None,
            "v_states":      None,   # missing
            "o_proj_weight": o_proj,
            "num_q_heads":   4,
            "num_kv_heads":  2,
            "head_dim":      8,
        }
        rows = compute_probe_rows(
            layer_idx=0, capture=capture,
            block_output=block_output,
            request_id="x", call_order=0,
            parallel_group="", is_parallel=False,
            tool="t", model="m", phase="prefill",
        )
        assert rows == []

    def test_prefill_seq_pos_zero(self):
        rows = self._run(phase="prefill")
        for row in rows:
            assert row["seq_pos"] == 0

    def test_decode_seq_pos_windows(self):
        rows = self._run(phase="decode", seq_len=6)
        seq_positions = sorted({r["seq_pos"] for r in rows})
        # decode_window=3, seq_len=6 → windows starting at 0 and 3
        assert seq_positions == [0, 3]


class TestIsSoftmaxAttentionLayer:
    def test_standard_attention(self):
        from agent.interp.kv_probe import is_softmax_attention_layer

        class FakeSelfAttention:
            pass

        m = FakeSelfAttention()
        # Add q_proj and o_proj attributes (standard layout)
        import torch.nn as nn
        m.q_proj = nn.Linear(32, 32)
        m.o_proj = nn.Linear(32, 32)
        assert is_softmax_attention_layer(m)

    def test_deltanet_excluded(self):
        from agent.interp.kv_probe import is_softmax_attention_layer

        class FakeDeltaNetAttention:
            pass

        m = FakeDeltaNetAttention()
        import torch.nn as nn
        m.q_proj = nn.Linear(32, 32)
        m.o_proj = nn.Linear(32, 32)
        # Name contains "deltanet" → should be excluded
        m.__class__.__name__ = "DeltaNetAttention"
        assert not is_softmax_attention_layer(m)

    def test_no_projections_excluded(self):
        from agent.interp.kv_probe import is_softmax_attention_layer

        class PlainModule:
            pass

        assert not is_softmax_attention_layer(PlainModule())
