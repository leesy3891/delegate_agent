"""Integration tests for the recorder <-> HF-local call-site wiring.

Verifies the seams added on top of agent.interp.session_recorder:
  - agent.agent_init._HFLocalClientShim.create() builds an InferenceRecord
    per call and hands it to the active recorder, sharing request_id/
    call_order/parallel_group/is_parallel with the meta passed to
    pool.generate() (so probe_rows carry the same identifiers).
  - agent.auxiliary_client._call_hf_local_aux() does the same for aux tasks.
  - The recorder flush actually writes json/csv containing both a
    phase="prefill" and a phase="decode" probe row once decode probe rows
    are present (proving the plumbing carries decode rows through to disk,
    independent of whether the real GPU capture path produced them).
  - When profiling is disabled, get_recorder() is a NoOp and none of this
    writes any files or reaches for torch/transformers.
"""

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _fake_pool(probe_rows):
    """A stand-in for HFWorkerPool exposing only .generate()."""
    pool = SimpleNamespace()

    def generate(messages, *, model_type="llm", sampling=None, capture=False, meta=None, tools=None, timeout=600.0):
        return (
            "hello from hf-local",
            {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15, "reasoning_tokens": 0},
            {"e2e_s": 0.42, "compute_s": 0.31},
            probe_rows,
        )

    pool.generate = generate
    return pool


def _probe_rows(request_id: str, call_order: int):
    common = dict(
        request_id=request_id, call_order=call_order,
        parallel_group="", is_parallel=False, tool="", model="Qwen/Qwen3.5-9B",
        layer=0, kv_head=0, cosine=0.5, l2=1.0,
    )
    return [
        {**common, "phase": "prefill", "seq_pos": 0},
        {**common, "phase": "decode", "seq_pos": 0},
        {**common, "phase": "decode", "seq_pos": 64},
    ]


@pytest.fixture
def active_recorder(tmp_path, monkeypatch):
    """Point the global recorder singleton at an ActiveRecorder in tmp_path."""
    from agent.interp import session_recorder as sr

    recorder = sr.ActiveRecorder(repo_root=tmp_path)
    monkeypatch.setattr(sr, "_RECORDER_SINGLETON", recorder)
    yield recorder
    monkeypatch.setattr(sr, "_RECORDER_SINGLETON", None)
    sr.clear_current_turn_id()


@pytest.fixture
def noop_recorder(monkeypatch):
    """Point the global recorder singleton at a NoOpRecorder."""
    from agent.interp import session_recorder as sr

    recorder = sr.NoOpRecorder()
    monkeypatch.setattr(sr, "_RECORDER_SINGLETON", recorder)
    yield recorder
    monkeypatch.setattr(sr, "_RECORDER_SINGLETON", None)
    sr.clear_current_turn_id()


class TestShimWiring:
    def test_create_adds_inference_record(self, active_recorder, tmp_path):
        from agent.agent_init import _HFLocalClientShim
        from agent.interp.session_recorder import set_current_turn_id

        turn_id = uuid.uuid4().hex
        active_recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        set_current_turn_id(turn_id)

        agent = SimpleNamespace(model="Qwen/Qwen3.5-9B", _subagent_id=None, _parent_subagent_id=None)
        pool = _fake_pool(_probe_rows("will-be-overwritten", 0))
        shim = _HFLocalClientShim(pool, "Qwen/Qwen3.5-9B", agent=agent)

        shim.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

        assert len(active_recorder._turns[turn_id].records) == 1
        rec = active_recorder._turns[turn_id].records[0]
        assert rec.role == "main"
        assert rec.turn_id == turn_id
        assert rec.output_text == "hello from hf-local"
        assert rec.input_tokens == 12

        active_recorder.complete_turn(turn_id)

        json_files = list((tmp_path / "logs").glob("*.json"))
        assert len(json_files) == 1

    def test_subagent_role_and_parallel_group(self, active_recorder):
        from agent.agent_init import _HFLocalClientShim
        from agent.interp.session_recorder import set_current_turn_id

        turn_id = uuid.uuid4().hex
        active_recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        set_current_turn_id(turn_id)

        agent = SimpleNamespace(
            model="Qwen/Qwen3.5-9B",
            _subagent_id="sa-0-abcd1234",
            _parent_subagent_id=None,
            _hf_parallel_group="batch-xyz",
            _hf_is_parallel=True,
        )
        pool = _fake_pool([])
        shim = _HFLocalClientShim(pool, "Qwen/Qwen3.5-9B", agent=agent)
        shim.chat.completions.create(messages=[{"role": "user", "content": "go"}])

        rec = active_recorder._turns[turn_id].records[0]
        assert rec.role == "subagent"
        assert rec.subagent_id == "sa-0-abcd1234"
        assert rec.parallel_group == "batch-xyz"
        assert rec.is_parallel is True

        active_recorder.complete_turn(turn_id)

    def test_input_delta_excludes_prior_messages(self, active_recorder):
        """Only messages new since the previous call on this shim are recorded."""
        from agent.agent_init import _HFLocalClientShim
        from agent.interp.session_recorder import set_current_turn_id

        turn_id = uuid.uuid4().hex
        active_recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        set_current_turn_id(turn_id)

        agent = SimpleNamespace(model="m", _subagent_id=None, _parent_subagent_id=None)
        pool = _fake_pool([])
        shim = _HFLocalClientShim(pool, "m", agent=agent)

        first_messages = [{"role": "user", "content": "hi"}]
        shim.chat.completions.create(messages=first_messages)
        second_messages = first_messages + [
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ]
        shim.chat.completions.create(messages=second_messages)

        records = active_recorder._turns[turn_id].records
        assert records[0].input_delta == first_messages
        assert records[1].input_delta == second_messages[1:]

        active_recorder.complete_turn(turn_id)

    def test_call_order_monotonic_across_calls(self, active_recorder):
        from agent.agent_init import _HFLocalClientShim
        from agent.interp.session_recorder import set_current_turn_id

        turn_id = uuid.uuid4().hex
        active_recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        set_current_turn_id(turn_id)

        agent = SimpleNamespace(model="m", _subagent_id=None, _parent_subagent_id=None)
        pool = _fake_pool([])
        shim = _HFLocalClientShim(pool, "m", agent=agent)

        shim.chat.completions.create(messages=[{"role": "user", "content": "a"}])
        shim.chat.completions.create(messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])

        records = active_recorder._turns[turn_id].records
        assert records[0].call_order < records[1].call_order

        active_recorder.complete_turn(turn_id)


class TestAuxWiring:
    def test_call_hf_local_aux_adds_inference_record(self, active_recorder):
        from agent.interp.session_recorder import set_current_turn_id

        turn_id = uuid.uuid4().hex
        active_recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        set_current_turn_id(turn_id)

        fake_pool = _fake_pool(_probe_rows("x", 1))
        with patch("agent.hf_local.pool.get_pool", return_value=fake_pool), \
             patch("agent.hf_local.pool._load_hf_config", return_value={
                 "profiling_enabled": True, "llm_model": "Qwen/Qwen3.5-9B", "vlm_model": None,
             }):
            from agent.auxiliary_client import _call_hf_local_aux
            resp = _call_hf_local_aux(
                task="compression", messages=[{"role": "user", "content": "summarize"}],
                model=None, temperature=0.0, max_tokens=256,
            )

        assert resp.choices[0].message.content == "hello from hf-local"
        records = active_recorder._turns[turn_id].records
        assert len(records) == 1
        assert records[0].role == "aux"
        assert records[0].tool == "compression"

        active_recorder.complete_turn(turn_id)


class TestFlushIncludesDecodeRows:
    def test_csv_has_prefill_and_decode_rows(self, active_recorder, tmp_path):
        from agent.agent_init import _HFLocalClientShim
        from agent.interp.session_recorder import set_current_turn_id
        import csv

        turn_id = uuid.uuid4().hex
        active_recorder.start_turn(turn_id, "Qwen/Qwen3.5-9B")
        set_current_turn_id(turn_id)

        agent = SimpleNamespace(model="Qwen/Qwen3.5-9B", _subagent_id=None, _parent_subagent_id=None)
        pool = _fake_pool(_probe_rows("req-1", 0))
        shim = _HFLocalClientShim(pool, "Qwen/Qwen3.5-9B", agent=agent)
        shim.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

        active_recorder.complete_turn(turn_id)

        csv_file = list((tmp_path / "profiling").glob("*.csv"))[0]
        rows = list(csv.DictReader(csv_file.read_text().splitlines()))
        phases = {r["phase"] for r in rows}
        assert "prefill" in phases
        assert "decode" in phases
        decode_positions = {r["seq_pos"] for r in rows if r["phase"] == "decode"}
        assert "64" in decode_positions


class TestNoOpPathIsInert:
    def test_shim_create_writes_nothing_when_disabled(self, noop_recorder, tmp_path, monkeypatch):
        from agent.agent_init import _HFLocalClientShim
        from agent.interp.session_recorder import clear_current_turn_id

        clear_current_turn_id()  # no ambient turn, matches a real disabled-profiling run

        agent = SimpleNamespace(model="m", _subagent_id=None, _parent_subagent_id=None)
        pool = _fake_pool([])
        shim = _HFLocalClientShim(pool, "m", agent=agent)
        shim.chat.completions.create(messages=[{"role": "user", "content": "hi"}])

        # NoOpRecorder never touches the filesystem.
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "profiling").exists()

    def test_aux_writes_nothing_when_disabled(self, noop_recorder):
        from agent.interp.session_recorder import clear_current_turn_id
        clear_current_turn_id()

        fake_pool = _fake_pool([])
        with patch("agent.hf_local.pool.get_pool", return_value=fake_pool), \
             patch("agent.hf_local.pool._load_hf_config", return_value={
                 "profiling_enabled": False, "llm_model": "m", "vlm_model": None,
             }):
            from agent.auxiliary_client import _call_hf_local_aux
            resp = _call_hf_local_aux(
                task="compression", messages=[{"role": "user", "content": "x"}],
                model=None, temperature=0.0, max_tokens=64,
            )
        assert resp.choices[0].message.content == "hello from hf-local"
        # NoOpRecorder.add_record is a true no-op; nothing to assert on disk
        # since no ActiveRecorder is involved in this fixture.

    def test_recorder_modules_do_not_import_torch(self):
        """The call-site wiring modules (session_recorder, schemas, the tool-
        call parser) must not pull in torch/transformers at import time —
        those stay confined to capture.py/runtime.py inside worker
        processes. This guarantees the parent process stays light when
        hf_local/profiling are disabled.
        """
        import subprocess

        code = (
            "import sys\n"
            "import agent.interp.session_recorder\n"
            "import agent.interp.schemas\n"
            "import agent.transports.hf_local\n"
            "assert 'torch' not in sys.modules, sys.modules.keys()\n"
            "assert 'transformers' not in sys.modules, sys.modules.keys()\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout
