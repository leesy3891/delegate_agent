"""Single-device HuggingFace model wrapper.

Loads one model on one CUDA device with eager attention (required for
per-head weight capture) and exposes a generate() method that returns
(text, usage, timings, probe_rows).

Thinks/reasoning tokens split: text in <think>…</think> at the start of
the output is treated as the reasoning trace; everything after is the
completion.

This module is imported ONLY inside worker processes spawned by pool.py.
It is never imported in the parent process.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regex to strip the leading <think>…</think> block from Qwen3 outputs
_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def _split_reasoning(text: str) -> Tuple[str, str]:
    """Split '<think>...</think> completion' → (reasoning, completion)."""
    m = _THINK_RE.match(text)
    if m:
        reasoning = m.group(1).strip()
        completion = text[m.end():].strip()
        return reasoning, completion
    return "", text


class LocalModel:
    """Wrapper around a single HF model on one CUDA device."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        dtype_str: str = "bfloat16",
        attn_implementation: str = "eager",
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._dtype_str = dtype_str

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        logger.info("Loading model %s on %s (dtype=%s, attn=%s)",
                    model_id, device, dtype_str, attn_implementation)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        # Try to load as causal LM; VLMs need AutoProcessor + AutoModel
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
                device_map=device,
            )
            self.is_vlm = False
            self.processor = None
        except Exception:
            # Fall back to VLM (e.g. Qwen3-VL)
            from transformers import AutoModel
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    attn_implementation=attn_implementation,
                    trust_remote_code=True,
                    device_map=device,
                )
            except ImportError:
                self.model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    attn_implementation=attn_implementation,
                    trust_remote_code=True,
                    device_map=device,
                )
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.is_vlm = True

        self.model.eval()
        logger.info("Model %s loaded successfully", model_id)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
        capture: bool = False,
        request_meta: Optional[Dict[str, Any]] = None,
        probe_mode: str = "output_contribution",
        decode_window: int = 64,
    ) -> Tuple[str, Dict[str, int], Dict[str, float], List[Dict[str, Any]]]:
        """Run the model on `messages` and return (text, usage, timings, probe_rows).

        Args:
            messages: OpenAI-format messages list.
            temperature: Sampling temperature (0 = greedy).
            max_new_tokens: Override model default.
            capture: If True, run KV/attention capture and compute probe rows.
            request_meta: Metadata for probe rows (request_id, call_order, etc.)
            probe_mode: "output_contribution" | "value_projection"
            decode_window: Decode-phase aggregation window size.

        Returns:
            text:        Generated text (completion only, without <think> block).
            usage:       {prompt_tokens, completion_tokens, total_tokens, reasoning_tokens}
            timings:     {e2e_s, compute_s}
            probe_rows:  List of probe row dicts (empty when capture=False).
        """
        import torch
        from agent.hf_local.capture import register_attention_hooks
        from agent.interp.kv_probe import compute_probe_rows

        max_new_tokens = max_new_tokens or self.max_new_tokens
        t_start = time.perf_counter()

        # Build input
        if self.is_vlm and self.processor is not None:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            text_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        # Set up capture hooks if needed
        hook_handles: List[Any] = []
        hook_storage: Dict[int, Dict[str, Any]] = {}
        if capture:
            hook_handles, hook_storage = register_attention_hooks(self.model, storage=hook_storage)

        t_compute_start = time.perf_counter()

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "output_hidden_states": capture,
            "output_attentions": capture,
            "return_dict_in_generate": True,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["temperature"] = None  # greedy

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        t_compute_end = time.perf_counter()

        # Clean up hooks
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass

        # Decode output
        generated_ids = output.sequences[0][prompt_len:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        completion_tokens = len(generated_ids)
        total_tokens = prompt_len + completion_tokens

        reasoning_text, completion_text = _split_reasoning(raw_text)
        reasoning_tokens = 0
        if reasoning_text:
            # Approximate reasoning token count from the raw text
            think_match = _THINK_RE.match(raw_text)
            if think_match:
                think_ids = self.tokenizer.encode(think_match.group(0), add_special_tokens=False)
                reasoning_tokens = len(think_ids)
                completion_tokens = max(0, completion_tokens - reasoning_tokens)

        usage = {
            "prompt_tokens":     prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
            "reasoning_tokens":  reasoning_tokens,
        }

        t_end = time.perf_counter()
        timings = {
            "e2e_s":     t_end - t_start,
            "compute_s": t_compute_end - t_compute_start,
        }

        # Compute probe rows
        probe_rows: List[Dict[str, Any]] = []
        if capture and hook_storage:
            meta = request_meta or {}
            # Extract block hidden states for each layer
            hidden_states_list: List[Any] = []
            if hasattr(output, "hidden_states") and output.hidden_states:
                # output_hidden_states returns tuple of tuples (one per generated token)
                # We want the prefill + decode blocks
                # Shape: tuple[token_step] of tuple[layer] of (batch, seq, dim)
                # We'll use the first step (prefill) and all decode steps
                all_steps = output.hidden_states  # tuple of steps
                # all_steps[0] = prefill hidden states (tuple of layers)
                # all_steps[1:] = per-decode-step hidden states
                if all_steps:
                    prefill_hs = all_steps[0]  # tuple of layer tensors
                    hidden_states_list = list(prefill_hs)

            for layer_idx, capture_data in sorted(hook_storage.items()):
                # Block output: hidden_states[layer_idx + 1] (after the block)
                block_out = None
                if layer_idx + 1 < len(hidden_states_list):
                    block_out = hidden_states_list[layer_idx + 1]

                if block_out is None:
                    continue

                # Prefill probe
                rows = compute_probe_rows(
                    layer_idx=layer_idx,
                    capture=capture_data,
                    block_output=block_out,
                    request_id=meta.get("request_id", ""),
                    call_order=meta.get("call_order", 0),
                    parallel_group=meta.get("parallel_group", ""),
                    is_parallel=meta.get("is_parallel", False),
                    tool=meta.get("tool", ""),
                    model=self.model_id,
                    phase="prefill",
                    decode_window=decode_window,
                    probe_mode=probe_mode,
                )
                probe_rows.extend(rows)

                # Decode probe (using stored v_states if available)
                # For decode, we'd need to accumulate hidden states across steps
                # Simplified: skip decode probe when hidden_states not available

        return completion_text, usage, timings, probe_rows
