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


def _concat_v_states(entries: List[Dict[str, Any]]) -> Optional[Any]:
    """Concatenate per-forward-pass v_states along the sequence dim.

    Each decode-step forward only projects the *new* token (KV caching means
    v_proj only ever sees the incremental input), so the full key/value
    sequence a later step attends over has to be reassembled by concatenating
    every prior forward's v_states — prefill (full prompt) followed by each
    single-token decode step, in order.
    """
    import torch
    parts = [e.get("v_states") for e in entries if e.get("v_states") is not None]
    if not parts:
        return None
    return torch.cat(parts, dim=2)


def _build_square_attn(entries: List[Dict[str, Any]]) -> Optional[Any]:
    """Build a (1, n_q_heads, full_seq_len, full_seq_len) causal attention
    matrix from per-forward-pass attn_weights.

    The prefill entry contributes a (1, n_q, prompt_len, prompt_len) block;
    each decode-step entry contributes a single (1, n_q, 1, seq_k_so_far)
    row. Rows are placed at their absolute sequence position and the
    remaining (not-yet-existing key) columns are left at zero — which is
    exactly what causal masking already implies, so this is not a fudge,
    just materializing the same causal attention compute_probe_rows expects
    from a single "prefill-shaped" capture.
    """
    import torch
    aws = [e.get("attn_weights") for e in entries]
    if not aws or aws[-1] is None:
        return None
    final_seq_len = aws[-1].shape[-1]
    n_q_heads = aws[-1].shape[1]
    batch = aws[-1].shape[0]

    combined = torch.zeros(
        batch, n_q_heads, final_seq_len, final_seq_len,
        dtype=aws[-1].dtype, device=aws[-1].device,
    )
    row = 0
    for aw in aws:
        if aw is None:
            continue
        q_len = aw.shape[2]
        k_len = aw.shape[-1]
        combined[:, :, row:row + q_len, :k_len] = aw
        row += q_len
    return combined


def _concat_block_outputs(all_steps: Any, layer_idx: int) -> Optional[Any]:
    """Concatenate per-step block hidden states (prefill + decode) along seq dim.

    ``all_steps`` is ``output.hidden_states``: a tuple of per-generation-step
    tuples of per-layer tensors. ``all_steps[0]`` is the prefill step (shape
    (1, prompt_len, d_model) per layer); ``all_steps[1:]`` are decode steps
    (shape (1, 1, d_model) per layer). Concatenating along dim=1 in order
    reproduces the residual-stream output for the full generated sequence.
    """
    import torch
    parts = []
    for step in all_steps:
        if step and layer_idx + 1 < len(step):
            parts.append(step[layer_idx + 1])
    if not parts:
        return None
    return torch.cat(parts, dim=1)


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
        tools: Optional[List[Dict[str, Any]]] = None,
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
            tools: OpenAI-format tool definitions, forwarded to
                ``tokenizer.apply_chat_template(tools=...)`` so Qwen3/3.5 can
                see and emit ``<tool_call>`` blocks. Ignored (with a warning)
                for VLM requests — the processor's chat template does not
                support a ``tools`` kwarg; tool use is LLM-slot only.

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
            if tools:
                logger.warning(
                    "LocalModel.generate: tools were provided for a VLM request "
                    "(model=%s) but the processor's chat template does not "
                    "support tools — ignoring them. Tool use is LLM-slot only.",
                    self.model_id,
                )
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            template_kwargs: Dict[str, Any] = {}
            if tools:
                template_kwargs["tools"] = tools
            text_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        # Set up capture hooks if needed
        hook_handles: List[Any] = []
        hook_storage: Dict[int, List[Dict[str, Any]]] = {}
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

        # Compute probe rows.
        # hook_storage[layer_idx] is a list of per-forward-pass captures:
        # index 0 = prefill, index >=1 = decode steps in order (see
        # capture.py's register_attention_hooks docstring).
        probe_rows: List[Dict[str, Any]] = []
        if capture and hook_storage:
            meta = request_meta or {}
            all_steps: Tuple[Any, ...] = ()
            if hasattr(output, "hidden_states") and output.hidden_states:
                # output_hidden_states returns tuple of tuples (one per
                # generation step): all_steps[0] = prefill (tuple of
                # per-layer tensors), all_steps[1:] = one tuple per decode
                # step.
                all_steps = output.hidden_states

            for layer_idx, entries in sorted(hook_storage.items()):
                if not entries:
                    continue

                prefill_entry = entries[0]
                decode_entries = entries[1:]

                # ── Prefill probe: block output = all_steps[0][layer_idx+1] ──
                prefill_block_out = None
                if all_steps and all_steps[0] and layer_idx + 1 < len(all_steps[0]):
                    prefill_block_out = all_steps[0][layer_idx + 1]

                if prefill_block_out is not None:
                    rows = compute_probe_rows(
                        layer_idx=layer_idx,
                        capture=prefill_entry,
                        block_output=prefill_block_out,
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

                # ── Decode probe: reassemble the full causal capture from
                # every forward pass (prefill + each decode step) so the
                # attention matmul in kv_probe.py has the real, full key
                # sequence to attend over, then window it in decode_window
                # chunks. ──
                if decode_entries and len(all_steps) > 1:
                    full_v = _concat_v_states(entries)
                    full_attn = _build_square_attn(entries)
                    decode_block_out = _concat_block_outputs(all_steps, layer_idx)

                    if full_v is not None and full_attn is not None and decode_block_out is not None:
                        decode_capture = {
                            "attn_weights":   full_attn,
                            "v_states":       full_v,
                            "o_proj_weight":  prefill_entry.get("o_proj_weight"),
                            "num_q_heads":    prefill_entry.get("num_q_heads"),
                            "num_kv_heads":   prefill_entry.get("num_kv_heads"),
                            "head_dim":       prefill_entry.get("head_dim"),
                        }
                        rows = compute_probe_rows(
                            layer_idx=layer_idx,
                            capture=decode_capture,
                            block_output=decode_block_out,
                            request_id=meta.get("request_id", ""),
                            call_order=meta.get("call_order", 0),
                            parallel_group=meta.get("parallel_group", ""),
                            is_parallel=meta.get("is_parallel", False),
                            tool=meta.get("tool", ""),
                            model=self.model_id,
                            phase="decode",
                            decode_window=decode_window,
                            probe_mode=probe_mode,
                        )
                        probe_rows.extend(rows)

        return completion_text, usage, timings, probe_rows
