"""Dataclasses for HF-local profiling log/CSV/txt rows.

All types are plain dataclasses with no torch/transformers dependency so they
can be imported in the parent process even with profiling.enabled=false (they
carry no data; instantiation is gated separately).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProbeRow:
    """One row in the profiling CSV.

    Fields match the CSV column spec:
        request_id, call_order, parallel_group, is_parallel, tool, model,
        phase, seq_pos, layer, kv_head, cosine, l2
    """

    request_id: str
    call_order: int
    parallel_group: str
    is_parallel: bool
    tool: str
    model: str
    phase: str        # "prefill" | "decode"
    seq_pos: int      # token index (0 for prefill aggregate; window_start for decode)
    layer: int        # original layer index (0-based)
    kv_head: int      # KV-head index (GQA basis)
    cosine: float
    l2: float


@dataclass
class LayerInventoryRow:
    """Per-layer head inventory for the .txt profiling report."""

    layer_idx: int
    layer_type: str           # "attention" | "linear_attn"
    q_heads: Optional[int]
    k_heads: Optional[int]
    v_heads: Optional[int]
    head_dim: Optional[int]
    rope_dim: Optional[int]


@dataclass
class InferenceRecord:
    """Per-request record for the .json log and .txt report."""

    request_id: str
    call_order: int
    parallel_group: str
    is_parallel: bool
    tool: str                 # which tool triggered this request
    model: str
    role: str                 # "main" | "subagent" | "aux"
    subagent_id: Optional[str]
    parent_subagent_id: Optional[str]
    turn_id: str
    # Message delta (new context added this turn, not full history)
    input_delta: List[Dict[str, Any]]
    output_text: str
    reasoning_text: Optional[str]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    e2e_latency_s: Optional[float]
    compute_latency_s: Optional[float]   # None for remote API calls
    probe_rows: List[ProbeRow] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)


@dataclass
class TurnBundle:
    """All records for one top-level user turn."""

    turn_id: str
    timestamp_kst: str        # YYYYMMDD-HHMMSS.ffffff
    llm_model_id: str
    vlm_model_id: Optional[str]
    records: List[InferenceRecord] = field(default_factory=list)
    layer_inventory: List[LayerInventoryRow] = field(default_factory=list)
    # Config snapshots
    llm_config: Dict[str, Any] = field(default_factory=dict)
    vlm_config: Dict[str, Any] = field(default_factory=dict)
