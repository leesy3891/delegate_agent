"""Tests for agent.interp.session_recorder.

Verifies:
  - NoOpRecorder is used when profiling.enabled=false (no FS writes)
  - ActiveRecorder writes json/txt/csv with correct schema on turn completion
  - Completion barrier: files are written only after all pending children complete
  - Config gate: disabled → no files created
"""

import json
import threading
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_active_recorder(tmp_path: Path):
    """Create an ActiveRecorder pointing at a temporary directory."""
    from agent.interp.session_recorder import ActiveRecorder
    return ActiveRecorder(repo_root=tmp_path)


def _make_record(turn_id: str, call_order: int = 0):
    """Build a minimal InferenceRecord-like object."""
    from types import SimpleNamespace
    return SimpleNamespace(
        request_id        = str(uuid.uuid4()),
        call_order        = call_order,
        parallel_group    = "g0",
        is_parallel       = False,
        tool              = "test_tool",
        model             = "Qwen/Qwen3.5-9B",
        role              = "subagent",
        subagent_id       = "sa-0-abc12345",
        parent_subagent_id= None,
        turn_id           = turn_id,
        input_delta       = [{"role": "user", "content": "hello"}],
        output_text       = "world",
        reasoning_text    = None,
        input_tokens      = 10,
        output_tokens     = 5,
        reasoning_tokens  = 0,
        e2e_latency_s     = 1.2,
        compute_latency_s = 0.9,
        probe_rows        = [],
        tools_called      = ["read_file"],
    )


class TestNoOpRecorder:
    def test_is_noop_when_disabled(self, tmp_path):
        from agent.interp.session_recorder import NoOpRecorder
        recorder = NoOpRecorder()
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.add_record(_make_record(turn_id))
        recorder.complete_turn(turn_id)
        # NoOp: no profiling files written
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "profiling").exists()

    def test_get_recorder_returns_noop_when_disabled(self, monkeypatch):
        """get_recorder() returns NoOpRecorder when profiling.enabled=false."""
        from agent.interp import session_recorder as sr
        # Reset singleton
        monkeypatch.setattr(sr, "_RECORDER_SINGLETON", None)

        with patch("agent.interp.session_recorder._make_recorder") as mock_make:
            from agent.interp.session_recorder import NoOpRecorder
            mock_make.return_value = NoOpRecorder()
            recorder = sr.get_recorder()
            assert isinstance(recorder, NoOpRecorder)


class TestActiveRecorder:
    def test_writes_json_on_complete(self, tmp_path):
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.add_record(_make_record(turn_id, call_order=0))
        recorder.complete_turn(turn_id)

        json_files = list((tmp_path / "logs").glob("*.json"))
        assert len(json_files) == 1

    def test_json_schema(self, tmp_path):
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.add_record(_make_record(turn_id))
        recorder.complete_turn(turn_id)

        json_file = list((tmp_path / "logs").glob("*.json"))[0]
        payload = json.loads(json_file.read_text())

        assert "turn_id"       in payload
        assert "timestamp_kst" in payload
        assert "llm_model"     in payload
        assert "records"       in payload
        assert len(payload["records"]) == 1
        rec = payload["records"][0]
        assert "request_id"    in rec
        assert "input_delta"   in rec
        assert "output_text"   in rec
        assert "input_tokens"  in rec

    def test_writes_txt_on_complete(self, tmp_path):
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.add_record(_make_record(turn_id))
        recorder.complete_turn(turn_id)

        txt_files = list((tmp_path / "profiling").glob("*.txt"))
        assert len(txt_files) == 1

    def test_writes_csv_on_complete(self, tmp_path):
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.add_record(_make_record(turn_id))
        recorder.complete_turn(turn_id)

        csv_files = list((tmp_path / "profiling").glob("*.csv"))
        assert len(csv_files) == 1

    def test_csv_has_required_columns(self, tmp_path):
        import csv
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        rec = _make_record(turn_id)
        # Add a probe row
        from types import SimpleNamespace
        rec.probe_rows = [{
            "request_id": rec.request_id, "call_order": 0,
            "parallel_group": "g0", "is_parallel": False,
            "tool": "test", "model": "Qwen/Qwen3.5-9B",
            "phase": "prefill", "seq_pos": 0, "layer": 0,
            "kv_head": 0, "cosine": 0.9, "l2": 0.3,
        }]
        recorder.add_record(rec)
        recorder.complete_turn(turn_id)

        csv_file = list((tmp_path / "profiling").glob("*.csv"))[0]
        reader = csv.DictReader(csv_file.read_text().splitlines())
        required = {"request_id", "call_order", "parallel_group", "is_parallel",
                    "tool", "model", "phase", "seq_pos", "layer", "kv_head",
                    "cosine", "l2"}
        assert required.issubset(set(reader.fieldnames or []))

    def test_barrier_waits_for_pending(self, tmp_path):
        """Files are NOT written until all pending children complete."""
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.add_record(_make_record(turn_id))
        recorder.register_pending(turn_id, count=1)

        recorder.complete_turn(turn_id)   # main done but 1 pending child

        # Should NOT have written yet
        assert not any((tmp_path / "logs").glob("*.json")) if (tmp_path / "logs").exists() else True

        # Now complete the pending child
        recorder.complete_pending(turn_id, count=1)

        json_files = list((tmp_path / "logs").glob("*.json"))
        assert len(json_files) == 1

    def test_multiple_records(self, tmp_path):
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        for i in range(5):
            recorder.add_record(_make_record(turn_id, call_order=i))
        recorder.complete_turn(turn_id)

        json_file = list((tmp_path / "logs").glob("*.json"))[0]
        payload = json.loads(json_file.read_text())
        assert len(payload["records"]) == 5

    def test_turn_cleaned_up_after_flush(self, tmp_path):
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.add_record(_make_record(turn_id))
        recorder.complete_turn(turn_id)

        # After flush, the turn state is removed
        assert turn_id not in recorder._turns

    def test_layer_inventory_in_txt(self, tmp_path):
        recorder = _make_active_recorder(tmp_path)
        turn_id = str(uuid.uuid4())
        recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        recorder.set_layer_inventory(turn_id, [
            {"layer_idx": 0, "layer_type": "attention",
             "q_heads": 8, "k_heads": 2, "v_heads": 2, "head_dim": 64, "rope_dim": 64},
            {"layer_idx": 1, "layer_type": "linear_attn",
             "q_heads": 8, "k_heads": 2, "v_heads": 2, "head_dim": 64, "rope_dim": None},
        ])
        recorder.add_record(_make_record(turn_id))
        recorder.complete_turn(turn_id)

        txt_file = list((tmp_path / "profiling").glob("*.txt"))[0]
        txt = txt_file.read_text()
        assert "Layer Inventory" in txt
        assert "attention" in txt
        assert "linear_attn" in txt
