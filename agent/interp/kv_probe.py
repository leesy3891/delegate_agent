"""KV-head residual-stream influence probe.

Computes, per attention layer, per KV head (GQA basis), per phase:

  probe_mode=output_contribution (default):
    o_k = Σ_{q∈group(k)}  softmax(Q_q Kᵀ/√d) · V_k · W_O^{(q)}
    (aggregated across the sequence with the scheme in §5 of the spec)

  probe_mode=value_projection:
    o_k = mean_q( V_k · W_O^{(q)} )
    (unweighted V→O projection, ignoring attention weights)

Then for each KV head k: cosine(o_k, B) and l2(o_k, B) where B is the
block residual output (output_hidden_states of the (Attn+FFN) block).

All heavy computation stays on-GPU.  Only compact scalar rows cross IPC.

Public API:
    compute_probe_rows(capture_data, block_output, config) -> List[ProbeRow]

``capture_data`` is the dict assembled by ``agent.hf_local.capture`` for one
layer; ``block_output`` is the corresponding output_hidden_states tensor.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# NOTE: this module is imported inside worker processes only (the parent
# never calls compute_probe_rows directly).  torch import is deferred to
# function bodies so the module itself is importable without torch.


def _softmax(x):
    """torch.softmax along the last dim, avoids a local import at module level."""
    import torch
    return torch.softmax(x, dim=-1)


def _cosine_l2(a, b) -> Tuple[float, float]:
    """Return (cosine_similarity, l2_distance) for two 1-D tensors."""
    import torch
    a = a.float()
    b = b.float()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    l2 = torch.dist(a, b, p=2).item()
    return cos, l2


def _build_gqa_groups(num_q_heads: int, num_kv_heads: int) -> List[List[int]]:
    """Return groups[kv_head] = list of q_head indices in that group."""
    ratio = num_q_heads // num_kv_heads
    return [list(range(k * ratio, k * ratio + ratio)) for k in range(num_kv_heads)]


def compute_probe_rows(
    *,
    layer_idx: int,
    capture: Dict[str, Any],
    block_output: "torch.Tensor",
    request_id: str,
    call_order: int,
    parallel_group: str,
    is_parallel: bool,
    tool: str,
    model: str,
    phase: str,
    decode_window: int = 64,
    probe_mode: str = "output_contribution",
) -> List[Any]:
    """Compute per-KV-head probe rows for one attention layer.

    Args:
        layer_idx: Layer index (0-based, original model numbering).
        capture: Dict from capture.py: keys "attn_weights", "v_states", "o_proj_weight",
                 "num_q_heads", "num_kv_heads", "head_dim".
        block_output: shape (1, seq_len, d_model) or (seq_len, d_model) —
                      the residual-stream output of this (Attn+FFN) block,
                      from output_hidden_states.
        phase: "prefill" | "decode"
        decode_window: Size of the decode-phase aggregation window.
        probe_mode: "output_contribution" | "value_projection"

    Returns a list of ProbeRow-like dicts (avoids importing schemas inside worker).
    """
    import torch

    attn_weights = capture.get("attn_weights")    # (1, n_q_heads, seq_q, seq_k) or None
    v_states     = capture.get("v_states")         # (1, n_kv_heads, seq, head_dim)
    o_proj       = capture.get("o_proj_weight")    # (d_model, n_q_heads * head_dim)
    n_q_heads    = capture.get("num_q_heads", 0)
    n_kv_heads   = capture.get("num_kv_heads", 0)
    head_dim     = capture.get("head_dim", 0)

    if v_states is None or o_proj is None or n_kv_heads == 0 or head_dim == 0:
        return []

    # Ensure float32 for numerical stability
    v_states = v_states.float()
    o_proj   = o_proj.float()

    # block_output: (1, seq, d_model) or (seq, d_model)
    if block_output.dim() == 3:
        block_output = block_output.squeeze(0)  # → (seq, d_model)
    block_output = block_output.float()

    seq_len   = v_states.shape[2]
    d_model   = o_proj.shape[0]
    groups    = _build_gqa_groups(n_q_heads, n_kv_heads)

    rows = []

    # Determine sequence windows for aggregation
    if phase == "prefill":
        windows = [(0, seq_len, 0)]  # (start, end, seq_pos_label)
    else:
        # Decode: average over 64-token windows
        windows = []
        pos = 0
        while pos < seq_len:
            end = min(pos + decode_window, seq_len)
            windows.append((pos, end, pos))
            pos = end

    for win_start, win_end, seq_pos in windows:
        # block residual aggregated over the window
        B_win = block_output[win_start:win_end].mean(dim=0)  # (d_model,)

        for kv_head_idx in range(n_kv_heads):
            q_indices = groups[kv_head_idx]

            # V_k_full: full sequence (seq_len, head_dim) — needed for attn matmul
            # because causal attention at query pos i attends over all keys [0..i].
            V_k_full = v_states[0, kv_head_idx, :, :]
            # V_k_win: window slice for value_projection mode
            V_k_win = v_states[0, kv_head_idx, win_start:win_end, :]

            o_k_sum = torch.zeros(d_model, device=o_proj.device)

            for q in q_indices:
                # W_O^{(q)}: (d_model, head_dim)
                W_O_q = o_proj[:, q * head_dim : (q + 1) * head_dim]

                if probe_mode == "value_projection":
                    # Unweighted: mean over window positions
                    c_q = V_k_win.mean(dim=0)          # (head_dim,)
                    o_q = W_O_q @ c_q                  # (d_model,)
                else:
                    # output_contribution (default)
                    if attn_weights is not None:
                        # attn_weights: (1, n_q_heads, seq_q, seq_k)
                        # Row-slice to windowed queries; keep all keys (causal).
                        A_q = attn_weights[0, q, win_start:win_end, :]  # (win, seq_len)
                        # context per query position: (win, head_dim)
                        c_q_seq = A_q @ V_k_full  # (win_len, head_dim)
                    else:
                        c_q_seq = V_k_win  # fallback: no attention weights available

                    # Per-query-head write aggregated over window
                    o_q_seq = c_q_seq @ W_O_q.T  # (win_len, d_model)
                    o_q = o_q_seq.mean(dim=0)     # (d_model,)

                o_k_sum += o_q

            # o_k = sum over q group (output_contribution) or mean (already averaged)
            if probe_mode == "value_projection":
                o_k = o_k_sum / max(len(q_indices), 1)
            else:
                o_k = o_k_sum

            cosine, l2 = _cosine_l2(o_k, B_win)

            rows.append({
                "request_id":     request_id,
                "call_order":     call_order,
                "parallel_group": parallel_group,
                "is_parallel":    is_parallel,
                "tool":           tool,
                "model":          model,
                "phase":          phase,
                "seq_pos":        seq_pos,
                "layer":          layer_idx,
                "kv_head":        kv_head_idx,
                "cosine":         cosine,
                "l2":             l2,
            })

    return rows


def is_softmax_attention_layer(module: Any) -> bool:
    """Return True if this module performs standard softmax attention.

    Detection heuristic:
    - Has q_proj and o_proj children (standard MHA/GQA layout), AND
    - Class name does NOT indicate linear/DeltaNet attention.

    DeltaNet/linear attention modules are excluded because they have no
    softmax KV cache and no useful per-head attention weights.
    """
    cls_name = type(module).__name__.lower()
    # Exclude known linear-attention / DeltaNet classes
    _LINEAR_ATTENTION_PATTERNS = (
        "deltanet",
        "linearattn",
        "linear_attn",
        "linearnorm",
        "retnet",
        "mamba",
        "rwkv",
        "ssm",
        "s4",
        "gla",  # Gated Linear Attention
    )
    for pat in _LINEAR_ATTENTION_PATTERNS:
        if pat in cls_name:
            return False

    # Must have canonical projection attributes
    has_q = hasattr(module, "q_proj") or hasattr(module, "q_proj_weight")
    has_o = hasattr(module, "o_proj") or hasattr(module, "out_proj")
    return has_q and has_o
