"""Turn-scoped session recorder.

Buffers all inference records for one top-level user turn (main agent +
descendant sub-agents + aux tasks) under a shared turn_id.  Holds the
file open until the turn AND all spawned children/aux calls complete
(completion barrier), then flushes json/txt/csv to <repo>/logs/ and
<repo>/profiling/.

When profiling.enabled=false the recorder is a no-op: no imports of
torch/transformers, no FS writes.

Usage (parent agent turn start)::

    recorder = get_recorder()      # singleton, no-op when disabled
    turn_id = recorder.start_turn(llm_model_id, config)

Usage (per inference record)::

    recorder.add_record(InferenceRecord(...))
    recorder.add_aux_record(...)

Usage (turn end / child completion)::

    recorder.complete_turn(turn_id)   # waits for barrier, then flushes

The recorder is process-local (parent process only).  Worker processes
send compact probe_rows back via IPC; the parent attaches them to the
InferenceRecord before calling add_record().
"""

from __future__ import annotations

import contextvars
import csv
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_KST = timezone(timedelta(hours=9))

_RECORDER_LOCK = threading.Lock()
_RECORDER_SINGLETON: Optional["SessionRecorder"] = None

# Process-local "current turn" context. Set by the top-level turn entry point
# (run_agent.py's AIAgent.run_conversation) and read by every HF-local call
# site so InferenceRecord.turn_id can be filled in without threading a turn_id
# parameter through every call. Delegation sub-agents run as threads in the
# same process, so the value must be explicitly propagated across the thread
# boundary (contextvars are not inherited by new threads automatically) —
# see tools/delegate_tool.py's child-thread entry point.
_current_turn_id_var: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "hf_local_current_turn_id", default=""
)


def set_current_turn_id(turn_id: str) -> None:
    """Bind the current turn id for this thread's contextvar chain."""
    _current_turn_id_var.set(turn_id or "")


def get_current_turn_id() -> str:
    """Return the turn id bound for this thread, or "" if none is active."""
    return _current_turn_id_var.get()


def clear_current_turn_id() -> None:
    """Clear the current turn id (call when a top-level turn completes)."""
    _current_turn_id_var.set("")


def get_recorder() -> "SessionRecorder":
    """Return the global SessionRecorder, creating it if needed."""
    global _RECORDER_SINGLETON
    if _RECORDER_SINGLETON is not None:
        return _RECORDER_SINGLETON
    with _RECORDER_LOCK:
        if _RECORDER_SINGLETON is None:
            _RECORDER_SINGLETON = _make_recorder()
    return _RECORDER_SINGLETON


def _make_recorder() -> "SessionRecorder":
    """Create the appropriate recorder based on config."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        pro = cfg.get("profiling") or {}
        enabled = bool(pro.get("enabled", False))
    except Exception:
        enabled = False

    if not enabled:
        return NoOpRecorder()

    # Find repo root (directory containing run_agent.py)
    repo_root = _find_repo_root()
    return ActiveRecorder(repo_root=repo_root)


def _find_repo_root() -> Path:
    """Find the repo root (parent of agent/)."""
    try:
        import agent
        return Path(agent.__file__).parent.parent
    except Exception:
        return Path.cwd()


class SessionRecorder:
    """Abstract base."""

    def start_turn(
        self,
        turn_id: str,
        llm_model_id: str,
        *,
        vlm_model_id: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        vlm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    def add_record(self, record: Any) -> None:
        pass

    def register_pending(self, turn_id: str, count: int = 1) -> None:
        """Register N pending child completions for the turn barrier."""
        pass

    def complete_pending(self, turn_id: str, count: int = 1) -> None:
        """Signal that N pending children have completed."""
        pass

    def complete_turn(self, turn_id: str) -> None:
        """Signal that the main turn is done; flush when barrier clears."""
        pass

    def set_layer_inventory(self, turn_id: str, inventory: List[Any]) -> None:
        pass

    def next_call_order(self, turn_id: str) -> int:
        """Return the next monotonic call_order for this turn (thread-safe)."""
        return 0


class NoOpRecorder(SessionRecorder):
    """Recorder used when profiling.enabled=false. Completely inert."""
    pass


class ActiveRecorder(SessionRecorder):
    """Recorder that writes json/txt/csv when the turn completes."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._turns: Dict[str, "_TurnState"] = {}
        self._lock = threading.Lock()

    def start_turn(
        self,
        turn_id: str,
        llm_model_id: str,
        *,
        vlm_model_id: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        vlm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.now(_KST)
        ts  = now.strftime("%Y%m%d-%H%M%S.%f")
        state = _TurnState(
            turn_id        = turn_id,
            timestamp_kst  = ts,
            llm_model_id   = llm_model_id,
            vlm_model_id   = vlm_model_id,
            llm_config     = llm_config or {},
            vlm_config     = vlm_config or {},
        )
        with self._lock:
            self._turns[turn_id] = state

    def add_record(self, record: Any) -> None:
        turn_id = getattr(record, "turn_id", None) or ""
        with self._lock:
            state = self._turns.get(turn_id)
        if state is None:
            return
        with state.lock:
            state.records.append(record)

    def set_layer_inventory(self, turn_id: str, inventory: List[Any]) -> None:
        with self._lock:
            state = self._turns.get(turn_id)
        if state is None:
            return
        with state.lock:
            state.layer_inventory = list(inventory)

    def next_call_order(self, turn_id: str) -> int:
        with self._lock:
            state = self._turns.get(turn_id)
        if state is None:
            return 0
        with state.lock:
            state.call_order_counter += 1
            return state.call_order_counter

    def register_pending(self, turn_id: str, count: int = 1) -> None:
        with self._lock:
            state = self._turns.get(turn_id)
        if state is None:
            return
        with state.lock:
            state.pending += count

    def complete_pending(self, turn_id: str, count: int = 1) -> None:
        with self._lock:
            state = self._turns.get(turn_id)
        if state is None:
            return
        flush = False
        with state.lock:
            state.pending = max(0, state.pending - count)
            if state.main_done and state.pending == 0:
                flush = True
        if flush:
            self._flush(state)

    def complete_turn(self, turn_id: str) -> None:
        with self._lock:
            state = self._turns.get(turn_id)
        if state is None:
            return
        flush = False
        with state.lock:
            state.main_done = True
            if state.pending == 0:
                flush = True
        if flush:
            self._flush(state)

    def _flush(self, state: "_TurnState") -> None:
        """Write json/txt/csv for the completed turn."""
        try:
            self._write_files(state)
        except Exception as exc:
            logger.error("SessionRecorder: flush failed for turn %s: %s", state.turn_id, exc)
        finally:
            with self._lock:
                self._turns.pop(state.turn_id, None)

    def _write_files(self, state: "_TurnState") -> None:
        logs_dir    = self._repo_root / "logs"
        prof_dir    = self._repo_root / "profiling"
        logs_dir.mkdir(parents=True, exist_ok=True)
        prof_dir.mkdir(parents=True, exist_ok=True)

        ts = state.timestamp_kst
        # Collision avoidance: append short id if files already exist
        base = ts
        if (logs_dir / f"{base}.json").exists() or (prof_dir / f"{base}.csv").exists():
            base = f"{ts}-{uuid.uuid4().hex[:6]}"

        self._write_json(state, logs_dir / f"{base}.json")
        self._write_txt(state,  prof_dir / f"{base}.txt")
        self._write_csv(state,  prof_dir / f"{base}.csv")
        logger.info("SessionRecorder: flushed turn %s → %s.{json,txt,csv}", state.turn_id, base)

    def _write_json(self, state: "_TurnState", path: Path) -> None:
        records_out = []
        for rec in state.records:
            records_out.append({
                "request_id":       rec.request_id,
                "call_order":       rec.call_order,
                "parallel_group":   rec.parallel_group,
                "is_parallel":      rec.is_parallel,
                "tool":             rec.tool,
                "model":            rec.model,
                "role":             rec.role,
                "subagent_id":      rec.subagent_id,
                "parent_subagent_id": rec.parent_subagent_id,
                "input_delta":      rec.input_delta,
                "output_text":      rec.output_text,
                "reasoning_text":   rec.reasoning_text,
                "input_tokens":     rec.input_tokens,
                "output_tokens":    rec.output_tokens,
                "reasoning_tokens": rec.reasoning_tokens,
                "e2e_latency_s":    rec.e2e_latency_s,
                "compute_latency_s":rec.compute_latency_s,
                "tools_called":     rec.tools_called,
            })
        payload = {
            "turn_id":       state.turn_id,
            "timestamp_kst": state.timestamp_kst,
            "llm_model":     state.llm_model_id,
            "vlm_model":     state.vlm_model_id,
            "records":       records_out,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_txt(self, state: "_TurnState", path: Path) -> None:
        lines = []

        lines.append(f"=== Hermes HF-Local Profiling Report ===")
        lines.append(f"Turn:      {state.turn_id}")
        lines.append(f"Timestamp: {state.timestamp_kst} KST")
        lines.append("")

        lines.append("## Models")
        lines.append(f"  LLM: {state.llm_model_id}")
        if state.llm_config:
            for k, v in state.llm_config.items():
                lines.append(f"    {k}: {v}")
        if state.vlm_model_id:
            lines.append(f"  VLM: {state.vlm_model_id}")
            for k, v in (state.vlm_config or {}).items():
                lines.append(f"    {k}: {v}")
        lines.append("")

        if state.layer_inventory:
            lines.append("## Layer Inventory")
            lines.append(f"  {'idx':>4}  {'type':<14}  {'Q':>4}  {'K':>4}  {'V':>4}  {'head_dim':>8}  {'rope_dim':>8}")
            for row in state.layer_inventory:
                if isinstance(row, dict):
                    li = row
                else:
                    li = vars(row)
                lines.append(
                    f"  {li.get('layer_idx', '?'):>4}  "
                    f"{li.get('layer_type', '?'):<14}  "
                    f"{str(li.get('q_heads', 'N/A')):>4}  "
                    f"{str(li.get('k_heads', 'N/A')):>4}  "
                    f"{str(li.get('v_heads', 'N/A')):>4}  "
                    f"{str(li.get('head_dim', 'N/A')):>8}  "
                    f"{str(li.get('rope_dim', 'N/A')):>8}"
                )
            lines.append("")

        lines.append("## Inference Records")
        for rec in state.records:
            lines.append(f"  [{rec.call_order}] role={rec.role} model={rec.model}")
            lines.append(f"    tool:             {rec.tool or '(main)'}")
            lines.append(f"    subagent_id:      {rec.subagent_id or 'N/A'}")
            lines.append(f"    input_tokens:     {rec.input_tokens}")
            lines.append(f"    output_tokens:    {rec.output_tokens}")
            lines.append(f"    reasoning_tokens: {rec.reasoning_tokens}")
            if rec.e2e_latency_s is not None:
                lines.append(f"    e2e_latency_s:    {rec.e2e_latency_s:.3f}")
            if rec.compute_latency_s is not None:
                lines.append(f"    compute_latency_s:{rec.compute_latency_s:.3f}")
            else:
                lines.append(f"    compute_latency_s:N/A (remote API)")
            lines.append(f"    tools_called:     {', '.join(rec.tools_called) or 'none'}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_csv(self, state: "_TurnState", path: Path) -> None:
        fieldnames = [
            "request_id", "call_order", "parallel_group", "is_parallel",
            "tool", "model", "phase", "seq_pos", "layer", "kv_head",
            "cosine", "l2",
        ]
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        for rec in state.records:
            for row in rec.probe_rows:
                if isinstance(row, dict):
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
                else:
                    writer.writerow({k: getattr(row, k, "") for k in fieldnames})

        path.write_text(buf.getvalue(), encoding="utf-8")


class _TurnState:
    """Mutable state for one in-progress turn."""

    def __init__(
        self,
        turn_id: str,
        timestamp_kst: str,
        llm_model_id: str,
        vlm_model_id: Optional[str],
        llm_config: Dict[str, Any],
        vlm_config: Dict[str, Any],
    ) -> None:
        self.turn_id         = turn_id
        self.timestamp_kst   = timestamp_kst
        self.llm_model_id    = llm_model_id
        self.vlm_model_id    = vlm_model_id
        self.llm_config      = llm_config
        self.vlm_config      = vlm_config
        self.records: List[Any] = []
        self.layer_inventory: List[Any] = []
        self.pending: int    = 0      # pending child/aux completions
        self.main_done: bool = False
        self.call_order_counter: int = 0
        self.lock            = threading.Lock()
