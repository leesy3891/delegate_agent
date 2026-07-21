"""Tests for two hf-local routing bugs:

  1. async_call_llm() had no "hf-local" branch, so vision tasks (which
     always go through async_call_llm via tools/vision_tools.py) could
     never reach the VLM slot — resolve_vision_provider_client("hf-local")
     doesn't recognize hf-local as a vision provider and the client build
     fails, silently falling back to the main/auto provider.

  2. `tools` was accepted by _HFLocalClientShim.create() but dropped before
     it reached apply_chat_template(), so Qwen never saw its available
     tools and could not emit <tool_call> blocks — delegation sub-agents
     routed to hf-local could not use tools at all.

Both are exercised here without loading a real model: HFWorkerPool /
LocalModel are stood in with lightweight fakes/mocks so these run fast and
without a GPU.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class TestAsyncHFLocalRouting:
    def test_async_call_llm_routes_hf_local_for_text_task(self):
        """provider='hf-local' must short-circuit to _call_hf_local_aux,
        run off the event loop thread, before the vision-provider branch.
        """
        from agent.auxiliary_client import async_call_llm

        fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        async def _run():
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("hf-local", "Qwen/Qwen3.5-9B", "", "", None),
            ), patch(
                "agent.auxiliary_client._call_hf_local_aux", return_value=fake_resp
            ) as mock_aux, patch(
                "agent.auxiliary_client.resolve_vision_provider_client"
            ) as mock_vision:
                result = await async_call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=None,
                )

            mock_vision.assert_not_called()
            mock_aux.assert_called_once()
            assert mock_aux.call_args.kwargs["task"] == "compression"
            return result

        result = asyncio.run(_run())
        assert result is fake_resp

    def test_async_call_llm_routes_vision_to_vlm_slot(self):
        """A vision task with provider='hf-local' must reach
        _call_hf_local_aux (model_type='vlm' internally) instead of
        resolve_vision_provider_client, which doesn't know 'hf-local'.
        """
        from agent.auxiliary_client import async_call_llm

        fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="a photo"))])

        async def _run():
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("hf-local", "Qwen/Qwen3-VL-8B-Instruct", "", "", None),
            ), patch(
                "agent.auxiliary_client._call_hf_local_aux", return_value=fake_resp
            ) as mock_aux, patch(
                "agent.auxiliary_client.resolve_vision_provider_client"
            ) as mock_vision:
                result = await async_call_llm(
                    task="vision",
                    messages=[{"role": "user", "content": "describe this image"}],
                )

            mock_vision.assert_not_called()
            mock_aux.assert_called_once()
            assert mock_aux.call_args.kwargs["task"] == "vision"
            return result

        result = asyncio.run(_run())
        assert result is fake_resp

    def test_async_call_llm_hf_local_runs_off_event_loop(self):
        """_call_hf_local_aux is a blocking call; it must not run inline on
        the event loop (would freeze concurrent async work) — verified by
        asserting it executes via asyncio.to_thread (different thread than
        the event loop's).
        """
        from agent.auxiliary_client import async_call_llm

        seen_thread = {}

        def _blocking_aux(**kwargs):
            import threading
            seen_thread["ident"] = threading.get_ident()
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        async def _run():
            import threading
            main_thread_ident = threading.get_ident()
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("hf-local", "m", "", "", None),
            ), patch("agent.auxiliary_client._call_hf_local_aux", side_effect=_blocking_aux):
                await async_call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
            return main_thread_ident

        main_thread_ident = asyncio.run(_run())
        assert seen_thread["ident"] != main_thread_ident

    def test_call_llm_and_async_call_llm_pass_tools_through(self):
        """Both sync and async hf-local branches must forward `tools`
        unchanged to _call_hf_local_aux.
        """
        from agent.auxiliary_client import call_llm, async_call_llm

        tools = [{"type": "function", "function": {"name": "read_file"}}]
        fake_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("hf-local", "m", "", "", None),
        ), patch("agent.auxiliary_client._call_hf_local_aux", return_value=fake_resp) as mock_aux:
            call_llm(task=None, messages=[{"role": "user", "content": "hi"}], tools=tools)
        assert mock_aux.call_args.kwargs["tools"] == tools

        async def _run():
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("hf-local", "m", "", "", None),
            ), patch("agent.auxiliary_client._call_hf_local_aux", return_value=fake_resp) as mock_aux2:
                await async_call_llm(task=None, messages=[{"role": "user", "content": "hi"}], tools=tools)
            assert mock_aux2.call_args.kwargs["tools"] == tools

        asyncio.run(_run())


class TestToolsReachChatTemplate:
    def _make_local_model(self, *, is_vlm=False):
        """Build a LocalModel instance without running __init__ (which
        loads a real HF model) — only the attributes generate() touches
        are set.
        """
        import torch
        from agent.hf_local.runtime import LocalModel

        model = object.__new__(LocalModel)
        model.model_id = "Qwen/Qwen3.5-9B"
        model.device = "cpu"
        model.max_new_tokens = 8
        model._dtype_str = "bfloat16"
        model.is_vlm = is_vlm

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="PROMPT_TEXT")
        tokenizer.decode = MagicMock(return_value="hi there")
        tokenizer.side_effect = None

        def _tokenizer_call(text, return_tensors="pt"):
            return _FakeBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})

        tokenizer.__call__ = MagicMock(side_effect=_tokenizer_call)
        model.tokenizer = tokenizer

        fake_generated = torch.tensor([[1, 2, 3, 4, 5]])
        underlying_model = MagicMock()
        underlying_model.generate = MagicMock(return_value=SimpleNamespace(sequences=fake_generated))
        model.model = underlying_model

        if is_vlm:
            processor = MagicMock()
            processor.apply_chat_template = MagicMock(
                return_value=_FakeBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
            )
            model.processor = processor
        else:
            model.processor = None

        return model

    def test_tools_forwarded_to_apply_chat_template(self):
        model = self._make_local_model(is_vlm=False)
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

        model.generate(
            [{"role": "user", "content": "list files"}],
            capture=False,
            tools=tools,
        )

        model.tokenizer.apply_chat_template.assert_called_once()
        _, kwargs = model.tokenizer.apply_chat_template.call_args
        assert kwargs.get("tools") == tools

    def test_no_tools_kwarg_when_tools_not_given(self):
        """Backward compatibility: omitting tools must not inject a
        tools=None kwarg into apply_chat_template (some templates would
        choke on an explicit tools=None)."""
        model = self._make_local_model(is_vlm=False)

        model.generate([{"role": "user", "content": "hi"}], capture=False)

        _, kwargs = model.tokenizer.apply_chat_template.call_args
        assert "tools" not in kwargs

    def test_vlm_ignores_tools_with_warning(self, caplog):
        model = self._make_local_model(is_vlm=True)
        tools = [{"type": "function", "function": {"name": "read_file"}}]

        with caplog.at_level(logging.WARNING, logger="agent.hf_local.runtime"):
            model.generate(
                [{"role": "user", "content": "describe"}],
                capture=False,
                tools=tools,
            )

        model.processor.apply_chat_template.assert_called_once()
        _, kwargs = model.processor.apply_chat_template.call_args
        assert "tools" not in kwargs
        assert any("tools" in rec.message.lower() for rec in caplog.records)


class TestToolsReachWorkerAndPool:
    def test_shim_forwards_tools_to_pool_generate(self):
        from agent.agent_init import _HFLocalClientShim

        agent = SimpleNamespace(model="m", _subagent_id=None, _parent_subagent_id=None)
        pool = MagicMock()
        pool.generate.return_value = ("out", {"prompt_tokens": 1, "completion_tokens": 1,
                                               "total_tokens": 2, "reasoning_tokens": 0},
                                      {"e2e_s": 0.1, "compute_s": 0.05}, [])
        shim = _HFLocalClientShim(pool, "m", agent=agent)

        tools = [{"type": "function", "function": {"name": "read_file"}}]
        shim.chat.completions.create(messages=[{"role": "user", "content": "hi"}], tools=tools)

        assert pool.generate.call_args.kwargs["tools"] == tools

    def test_pool_generate_puts_tools_on_request_queue(self):
        from agent.hf_local.pool import HFWorkerPool, _DeviceSlot

        pool = object.__new__(HFWorkerPool)
        pool._slot_lock = __import__("threading").Lock()
        slot = _DeviceSlot(0)
        slot.ready.set()
        pool._slots = [slot]

        tools = [{"type": "function", "function": {"name": "search"}}]

        def _respond():
            req = slot.req_queue.get(timeout=5)
            assert req["tools"] == tools
            slot.resp_queue.put({
                "ok": True, "request_id": req["request_id"], "text": "hi",
                "usage": {}, "timings": {}, "probe_rows": [],
            })

        import threading
        t = threading.Thread(target=_respond, daemon=True)
        t.start()
        pool.generate([{"role": "user", "content": "hi"}], tools=tools, timeout=5)
        t.join(timeout=5)

    def test_worker_forwards_tools_to_local_model_generate(self):
        """worker_main reads msg['tools'] and passes it through to
        active_model.generate(tools=...)."""
        import agent.hf_local.worker as worker_mod

        fake_llm = MagicMock()
        fake_llm.generate.return_value = ("hi", {}, {}, [])

        req_queue = __import__("queue").Queue()
        resp_queue = __import__("queue").Queue()
        tools = [{"type": "function", "function": {"name": "search"}}]
        req_queue.put({
            "type": "generate", "request_id": "r1", "model_type": "llm",
            "messages": [{"role": "user", "content": "hi"}],
            "sampling": {}, "capture": False, "meta": {}, "tools": tools,
        })
        req_queue.put({"type": "shutdown"})

        with patch("agent.hf_local.runtime.LocalModel", return_value=fake_llm):
            worker_mod.worker_main(
                device_idx=0, llm_model_id="m", vlm_model_id=None,
                llm_config={}, req_queue=req_queue, resp_queue=resp_queue,
            )

        # First message on resp_queue is the "ready" ack from LocalModel load.
        ready = resp_queue.get(timeout=5)
        assert ready["type"] == "ready"
        resp = resp_queue.get(timeout=5)
        assert resp["ok"] is True

        fake_llm.generate.assert_called_once()
        assert fake_llm.generate.call_args.kwargs["tools"] == tools
