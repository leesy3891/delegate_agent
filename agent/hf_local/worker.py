"""Per-GPU worker process for HF model inference.

Each worker:
  - Pins to one CUDA device (CUDA_VISIBLE_DEVICES set by pool.py before spawn)
  - Preloads the LLM at startup
  - Lazy-loads the VLM on the first vision request
  - Loops on a request queue, returns (text, usage, timings, probe_rows)

Only compact probe rows + text + usage cross IPC (heavy tensors stay on-GPU).

Message protocol (multiprocessing.Queue):
  Request:  {"type": "generate", "messages": [...], "model_type": "llm"|"vlm",
             "sampling": {...}, "capture": bool, "meta": {...}, "tools": [...]|None,
             "request_id": str}
  Response: {"ok": True, "text": str, "usage": {...}, "timings": {...}, "probe_rows": [...]}
            {"ok": False, "error": str}

Sentinel:  {"type": "shutdown"}
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def worker_main(
    device_idx: int,
    llm_model_id: str,
    vlm_model_id: Optional[str],
    llm_config: Dict[str, Any],
    req_queue: Any,     # multiprocessing.Queue
    resp_queue: Any,    # multiprocessing.Queue
) -> None:
    """Worker entrypoint. Called in a spawned process.

    Args:
        device_idx:   The index within the worker's CUDA_VISIBLE_DEVICES (always 0
                      since pool.py uses per-process env vars).
        llm_model_id: HuggingFace model id for the LLM.
        vlm_model_id: HuggingFace model id for the VLM (None → VLM disabled).
        llm_config:   Dict with keys: dtype, attn_implementation, max_new_tokens.
        req_queue:    Queue to read requests from.
        resp_queue:   Queue to write responses to.
    """
    # Configure logging in worker
    logging.basicConfig(
        level=logging.INFO,
        format=f"[worker-{device_idx}] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    device_str = f"cuda:{device_idx}"

    from agent.hf_local.runtime import LocalModel

    # Preload LLM
    try:
        llm = LocalModel(
            llm_model_id,
            device=device_str,
            dtype_str=llm_config.get("dtype", "bfloat16"),
            attn_implementation=llm_config.get("attn_implementation", "eager"),
            max_new_tokens=int(llm_config.get("max_new_tokens", 2048)),
        )
        resp_queue.put({"type": "ready", "device_idx": device_idx})
    except Exception as exc:
        resp_queue.put({"type": "startup_error", "device_idx": device_idx, "error": str(exc)})
        return

    vlm: Optional[LocalModel] = None
    vlm_lazy = True  # VLM is loaded on first vision request

    probe_mode  = llm_config.get("probe_mode",    "output_contribution")
    decode_window = int(llm_config.get("decode_window", 64))

    while True:
        try:
            msg = req_queue.get()
        except Exception as exc:
            logger.error("Worker %d: queue read error: %s", device_idx, exc)
            continue

        if msg is None or msg.get("type") == "shutdown":
            logger.info("Worker %d: shutting down", device_idx)
            break

        if msg.get("type") != "generate":
            resp_queue.put({"ok": False, "error": f"Unknown message type: {msg.get('type')}"})
            continue

        request_id  = msg.get("request_id", "")
        model_type  = msg.get("model_type", "llm")
        messages    = msg.get("messages", [])
        sampling    = msg.get("sampling", {})
        capture     = bool(msg.get("capture", False))
        meta        = msg.get("meta", {})
        tools       = msg.get("tools")

        try:
            if model_type == "vlm":
                if vlm is None:
                    if vlm_model_id is None:
                        raise RuntimeError("VLM not configured (hf_local.vlm.model is not set)")
                    logger.info("Worker %d: lazy-loading VLM %s", device_idx, vlm_model_id)
                    vlm = LocalModel(
                        vlm_model_id,
                        device=device_str,
                        dtype_str=llm_config.get("dtype", "bfloat16"),
                        attn_implementation=llm_config.get("attn_implementation", "eager"),
                        max_new_tokens=int(llm_config.get("max_new_tokens", 2048)),
                    )
                active_model = vlm
            else:
                active_model = llm

            text, usage, timings, probe_rows = active_model.generate(
                messages,
                temperature=float(sampling.get("temperature", 0.0)),
                max_new_tokens=sampling.get("max_new_tokens"),
                capture=capture,
                request_meta=meta,
                probe_mode=probe_mode,
                decode_window=decode_window,
                tools=tools,
            )

            resp_queue.put({
                "ok":         True,
                "request_id": request_id,
                "text":       text,
                "usage":      usage,
                "timings":    timings,
                "probe_rows": probe_rows,
            })

        except Exception as exc:
            logger.exception("Worker %d: generate failed: %s", device_idx, exc)
            resp_queue.put({
                "ok":         False,
                "request_id": request_id,
                "error":      str(exc),
            })
