"""Forward hooks for per-head attention capture.

Registers hooks on every softmax-attention submodule in the model.
Only softmax attention layers (those with q_proj/o_proj and no DeltaNet
class-name marker) are hooked; linear-attention/DeltaNet layers are skipped.

The hook captures, per layer:
  - attention_weights: softmax(QKᵀ/√d)·V  (from output_attentions=True under eager)
  - v_states: per-KV-head value states
  - o_proj_weight: the o_proj weight slice for the output projection

All tensors stay on-GPU; the worker calls compute_probe_rows and passes
only scalar rows across IPC.

Usage::

    handles, storage = register_attention_hooks(model)
    # ... run forward pass with output_attentions=True ...
    # storage[layer_idx] = {"attn_weights": ..., "v_states": ..., ...}
    for h in handles:
        h.remove()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _detect_head_dims(module: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (num_q_heads, num_kv_heads, head_dim) from the attention module.

    Works for standard Qwen / LLaMA-style attention modules.
    """
    # Qwen / LLaMA style
    num_q  = getattr(module, "num_heads",        None) or getattr(module, "num_attention_heads", None)
    num_kv = getattr(module, "num_key_value_heads", None) or getattr(module, "num_kv_heads", None) or num_q
    head_d = getattr(module, "head_dim", None)

    if num_q is None or head_d is None:
        # Try to infer from q_proj weight
        qp = getattr(module, "q_proj", None)
        if qp is not None and hasattr(qp, "weight"):
            d_model = qp.weight.shape[1]
            q_out   = qp.weight.shape[0]
            if num_q:
                head_d = q_out // num_q
            elif d_model > 0 and q_out > 0:
                # assume head_dim = d_model / some power-of-2
                pass

    return num_q, num_kv, head_d


def register_attention_hooks(
    model: Any,
    *,
    storage: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Tuple[List[Any], Dict[int, List[Dict[str, Any]]]]:
    """Register forward hooks on all softmax-attention layers.

    ``generate()`` runs one prefill forward followed by N decode-step
    forwards (one per generated token). Each forward triggers this hook once
    per layer, so a single-dict-per-layer ``storage`` would have each step
    overwrite the previous one — by the time generation finishes, only the
    last decode step's capture would survive. Instead, every forward's
    capture is *appended* to a per-layer list: ``storage[layer_idx][0]`` is
    always the prefill capture, ``storage[layer_idx][1:]`` are the decode
    steps in order. The caller (``runtime.py``) uses this to build both the
    prefill probe and the decode-window probe from real, distinct tensors.

    Args:
        model: A loaded HuggingFace model.
        storage: Optional dict to reuse (cleared on call).

    Returns:
        (handles, storage) where handles is a list of hook handles to remove
        later, and storage maps layer_idx -> list of per-forward-pass capture
        dicts (index 0 = prefill, index >=1 = decode steps in order).
    """
    from agent.interp.kv_probe import is_softmax_attention_layer

    if storage is None:
        storage = {}
    else:
        storage.clear()

    handles: List[Any] = []

    # Build ordered list of (layer_idx, module) for attention layers
    # Walk all named modules and find attention layers.
    layer_counter = [0]  # use list for closure mutation

    # v_proj's forward hook fires *before* its owning attention module's own
    # post-forward hook (v_proj is called from inside attention.forward()),
    # for every single forward pass. So each layer's v-states land here first
    # and get consumed (popped) by that same forward's attention-module hook
    # immediately after — this pairs the two hooks correctly per forward pass
    # without relying on dict-overwrite timing.
    pending_v: Dict[int, Any] = {}

    def _make_hook(layer_idx: int, module: Any):
        """Return a forward hook that captures attention data for this layer."""
        num_q, num_kv, head_d = _detect_head_dims(module)

        def hook_fn(mod, inputs, outputs):
            data: Dict[str, Any] = {
                "num_q_heads":  num_q,
                "num_kv_heads": num_kv,
                "head_dim":     head_d,
                "attn_weights": None,
                "v_states":     pending_v.pop(layer_idx, None),
                "o_proj_weight": None,
            }

            # capture o_proj weight (detach — we only need it for the probe).
            # Static across forwards, but cheap enough to recapture each time.
            o_proj = getattr(mod, "o_proj", None) or getattr(mod, "out_proj", None)
            if o_proj is not None and hasattr(o_proj, "weight"):
                data["o_proj_weight"] = o_proj.weight.detach()

            # output_attentions=True surfaces attention weights as a tuple
            # element when using attn_implementation="eager"
            if isinstance(outputs, tuple):
                # Standard shape: (hidden_state, attn_weights, past_key_value)
                # attn_weights is usually the second element when present
                for elem in outputs[1:]:
                    if elem is not None:
                        try:
                            import torch
                            if isinstance(elem, torch.Tensor) and elem.dim() == 4:
                                # (batch, n_q_heads, seq_q, seq_k) shape
                                data["attn_weights"] = elem.detach()
                                break
                        except Exception:
                            pass

            storage.setdefault(layer_idx, []).append(data)

        return hook_fn

    # First pass: register post-forward hooks on attention modules
    for name, module in model.named_modules():
        if not is_softmax_attention_layer(module):
            continue

        layer_idx = layer_counter[0]
        layer_counter[0] += 1

        h = module.register_forward_hook(_make_hook(layer_idx, module))
        handles.append(h)

        # Also hook v_proj to capture value states for this same forward pass.
        v_proj = getattr(module, "v_proj", None)
        if v_proj is not None:
            _layer_idx = layer_idx  # capture in closure

            def _make_v_hook(lidx: int):
                def v_hook_fn(mod, inputs, output):
                    pending_v[lidx] = output.detach()
                return v_hook_fn

            vh = v_proj.register_forward_hook(_make_v_hook(_layer_idx))
            handles.append(vh)

    return handles, storage


def get_layer_inventory(model: Any) -> List[Dict[str, Any]]:
    """Return per-layer head inventory for all layers in the model.

    Returns list of dicts with keys:
        layer_idx, layer_type, q_heads, k_heads, v_heads, head_dim, rope_dim
    """
    from agent.interp.kv_probe import is_softmax_attention_layer

    inventory = []
    layer_idx = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__

        is_softmax = is_softmax_attention_layer(module)
        has_attn_like = (
            "attention" in cls_name.lower()
            or "attn" in cls_name.lower()
        ) and hasattr(module, "q_proj")

        if not has_attn_like:
            continue

        layer_type = "attention" if is_softmax else "linear_attn"
        num_q, num_kv, head_d = _detect_head_dims(module)
        rope_dim = getattr(module, "rope_dim", None) or getattr(module, "rotary_dim", None)

        inventory.append({
            "layer_idx":  layer_idx,
            "layer_type": layer_type,
            "q_heads":    num_q,
            "k_heads":    num_kv,
            "v_heads":    num_kv,
            "head_dim":   head_d,
            "rope_dim":   rope_dim,
        })
        layer_idx += 1

    return inventory
