"""HFWorkerPool — singleton pool of per-GPU worker processes.

Spawns exactly len(CUDA_VISIBLE_DEVICES) workers (one per device).
Dispatches each request to a device so concurrent sub-agents land on
distinct GPUs; ensures ≤1 in-flight generate per device.
Aux-task requests share the same pool/queue.

Usage::

    pool = get_pool()      # creates singleton on first call
    result = pool.generate(messages, model_type="llm", ...)
    # result = (text, usage, timings, probe_rows)

Shutdown::

    pool.shutdown()        # sends sentinel to each worker, joins processes

The pool is created lazily and only when hf_local.enabled=true.
No torch/transformers is imported in the parent process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_POOL_LOCK   = threading.Lock()
_POOL_SINGLETON: Optional["HFWorkerPool"] = None


def get_pool() -> "HFWorkerPool":
    """Return the global HFWorkerPool, creating it if needed."""
    global _POOL_SINGLETON
    if _POOL_SINGLETON is not None:
        return _POOL_SINGLETON
    with _POOL_LOCK:
        if _POOL_SINGLETON is None:
            _POOL_SINGLETON = HFWorkerPool()
            _POOL_SINGLETON.start()
    return _POOL_SINGLETON


def _load_hf_config() -> Dict[str, Any]:
    """Load hf_local and profiling config from config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    hf  = cfg.get("hf_local") or {}
    pro = cfg.get("profiling") or {}
    return {
        "enabled":              bool(hf.get("enabled", False)),
        "dtype":                str(hf.get("dtype", "bfloat16")),
        "attn_implementation":  str(hf.get("attn_implementation", "eager")),
        "llm_model":            str((hf.get("llm") or {}).get("model", "Qwen/Qwen3.5-9B")),
        "llm_max_new_tokens":   int((hf.get("llm") or {}).get("max_new_tokens", 2048)),
        "vlm_model":            str((hf.get("vlm") or {}).get("model", "Qwen/Qwen3-VL-8B-Instruct")),
        "vlm_lazy":             bool((hf.get("vlm") or {}).get("lazy", True)),
        "profiling_enabled":    bool(pro.get("enabled", False)),
        "probe_mode":           str(pro.get("probe_mode", "output_contribution")),
        "decode_window":        int(pro.get("decode_window", 64)),
    }


class _DeviceSlot:
    """State for one GPU device slot in the pool."""

    def __init__(self, device_idx: int) -> None:
        self.device_idx  = device_idx
        self.req_queue:  mp.Queue = mp.Queue()
        self.resp_queue: mp.Queue = mp.Queue()
        self.process:    Optional[mp.Process] = None
        self.in_flight   = threading.Semaphore(1)   # ≤1 generate at a time per device
        self.ready       = threading.Event()
        self.error:      Optional[str] = None


class HFWorkerPool:
    """Pool of per-GPU worker processes."""

    def __init__(self) -> None:
        cfg = _load_hf_config()

        # Detect available CUDA devices
        raw_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if raw_devs and raw_devs.lower() not in ("", "nodevfiles"):
            self._device_ids: List[str] = [d.strip() for d in raw_devs.split(",") if d.strip()]
        else:
            # Default: try to detect from torch (but don't import in parent)
            self._device_ids = ["0"]  # single device fallback

        self._config      = cfg
        self._slots: List[_DeviceSlot] = [
            _DeviceSlot(i) for i in range(len(self._device_ids))
        ]
        self._slot_lock   = threading.Lock()
        self._started     = False

    def start(self) -> None:
        """Spawn worker processes for each device slot."""
        if self._started:
            return
        self._started = True

        cfg = self._config
        llm_worker_cfg = {
            "dtype":                cfg["dtype"],
            "attn_implementation":  cfg["attn_implementation"],
            "max_new_tokens":       cfg["llm_max_new_tokens"],
            "probe_mode":           cfg["probe_mode"],
            "decode_window":        cfg["decode_window"],
        }

        ctx = mp.get_context("spawn")

        from agent.hf_local.worker import worker_main

        for i, slot in enumerate(self._slots):
            # Each worker sees only its own GPU
            phys_device = self._device_ids[i]
            env_override = {"CUDA_VISIBLE_DEVICES": phys_device}

            proc = ctx.Process(
                target=_worker_entry,
                args=(
                    0,                      # within the worker, always device 0
                    cfg["llm_model"],
                    cfg["vlm_model"] if not cfg["vlm_lazy"] else cfg["vlm_model"],
                    llm_worker_cfg,
                    slot.req_queue,
                    slot.resp_queue,
                    env_override,
                ),
                daemon=True,
                name=f"hf-worker-{i}",
            )
            proc.start()
            slot.process = proc

        # Wait for all workers to signal ready (with timeout)
        _WORKER_READY_TIMEOUT = 300  # seconds
        for slot in self._slots:
            deadline = time.time() + _WORKER_READY_TIMEOUT
            while time.time() < deadline:
                try:
                    msg = slot.resp_queue.get(timeout=5)
                    if msg.get("type") == "ready":
                        slot.ready.set()
                        logger.info("HF worker %d ready", slot.device_idx)
                        break
                    elif msg.get("type") == "startup_error":
                        slot.error = msg.get("error", "unknown")
                        logger.error("HF worker %d failed to start: %s",
                                     slot.device_idx, slot.error)
                        break
                except Exception:
                    pass
            if not slot.ready.is_set() and not slot.error:
                slot.error = f"Worker {slot.device_idx} timed out during startup"
                logger.error("HF worker %d startup timeout", slot.device_idx)

    def _pick_slot(self) -> Optional[_DeviceSlot]:
        """Return a slot that is ready and not currently in-flight, or None."""
        with self._slot_lock:
            # Round-robin among ready slots
            for slot in self._slots:
                if slot.ready.is_set() and not slot.error:
                    if slot.in_flight.acquire(blocking=False):
                        return slot
        return None

    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        model_type: str = "llm",
        sampling: Optional[Dict[str, Any]] = None,
        capture: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 600.0,
    ) -> Tuple[str, Dict[str, int], Dict[str, float], List[Dict[str, Any]]]:
        """Submit a generate request to an available slot and wait for result.

        Blocks until a slot is available (polling with 0.1s intervals up to
        the timeout), then waits for generation to complete.

        Args:
            messages:   OpenAI-format messages list.
            model_type: "llm" | "vlm"
            sampling:   Dict with temperature, max_new_tokens.
            capture:    Whether to run KV/attention capture.
            meta:       Metadata for probe rows.
            tools:      OpenAI-format tool definitions, forwarded to the
                        worker's apply_chat_template(tools=...) call. Ignored
                        for model_type="vlm" (see runtime.py).
            timeout:    Total timeout in seconds.

        Returns (text, usage, timings, probe_rows).
        """
        deadline = time.time() + timeout
        slot: Optional[_DeviceSlot] = None

        # Wait for an available slot
        while time.time() < deadline:
            slot = self._pick_slot()
            if slot is not None:
                break
            time.sleep(0.1)

        if slot is None:
            raise RuntimeError("HFWorkerPool: no GPU slot available within timeout")

        try:
            request_id = str(uuid.uuid4())
            slot.req_queue.put({
                "type":       "generate",
                "request_id": request_id,
                "messages":   messages,
                "model_type": model_type,
                "sampling":   sampling or {},
                "capture":    capture,
                "meta":       meta or {},
                "tools":      tools,
            })

            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("HFWorkerPool: timed out waiting for slot response")

            resp = slot.resp_queue.get(timeout=remaining)

            if not resp.get("ok"):
                raise RuntimeError(f"HF worker error: {resp.get('error', 'unknown')}")

            return (
                resp.get("text", ""),
                resp.get("usage", {}),
                resp.get("timings", {}),
                resp.get("probe_rows", []),
            )
        finally:
            slot.in_flight.release()

    @property
    def num_devices(self) -> int:
        return len(self._slots)

    def shutdown(self) -> None:
        """Send shutdown sentinel to each worker and join processes."""
        for slot in self._slots:
            try:
                slot.req_queue.put({"type": "shutdown"})
            except Exception:
                pass
        for slot in self._slots:
            if slot.process is not None:
                slot.process.join(timeout=10)
                if slot.process.is_alive():
                    slot.process.terminate()


def _worker_entry(
    device_idx: int,
    llm_model_id: str,
    vlm_model_id: Optional[str],
    llm_config: Dict[str, Any],
    req_queue: Any,
    resp_queue: Any,
    env_override: Dict[str, str],
) -> None:
    """Top-level function called in the spawned process.

    Sets environment variables before any CUDA import.
    Must be a top-level picklable function (no lambda/closure) for spawn.
    """
    for k, v in env_override.items():
        os.environ[k] = v

    from agent.hf_local.worker import worker_main
    worker_main(
        device_idx=device_idx,
        llm_model_id=llm_model_id,
        vlm_model_id=vlm_model_id,
        llm_config=llm_config,
        req_queue=req_queue,
        resp_queue=resp_queue,
    )
