"""Local-only named worker diagnostics and evaluation-only teacher forcing.

No installed vLLM sources are changed. No pickle RPC is enabled.
"""

import json
import math


class ScaleWorkerExtension:
    def _ail_layers(self):
        return {
            name: layer
            for name, layer in self.model_runner.get_model().named_modules()
            if all(hasattr(layer, key) for key in ("_k_scale", "_v_scale", "calculate_kv_scales"))
        }

    def ail_snapshot(self):
        rows = []
        for name, layer in self._ail_layers().items():
            row = {
                "layer": name,
                "calculate": bool(layer.calculate_kv_scales),
                "dtype": layer.kv_cache_dtype,
                "backend": type(layer.impl).__name__,
            }
            for key in ("_q_scale", "_k_scale", "_v_scale"):
                row[key] = getattr(layer, key).detach().float().cpu().reshape(-1).tolist()
            rows.append(row)
        return {"rows": rows, "runner_calculate": bool(self.model_runner.calculate_kv_scales)}

    def ail_arm_scales(self):
        for layer in self._ail_layers().values():
            if not str(layer.kv_cache_dtype).startswith("fp8"):
                raise ValueError("Can only arm FP8 layers")
            layer.calculate_kv_scales = True
        self.model_runner.calculate_kv_scales = True
        return self.ail_snapshot()

    def ail_load_scales(self, encoded):
        import torch

        source = {r["layer"]: r for r in json.loads(encoded)["rows"]}
        layers = self._ail_layers()
        if source.keys() != layers.keys() or len(layers) != 24:
            raise ValueError("Expected matching 24-layer scale set")
        # Validate everything before modifying any layer.
        for name, layer in layers.items():
            if not str(layer.kv_cache_dtype).startswith("fp8"):
                raise ValueError("Scale loading is for FP8 only")
            for key in ("_q_scale", "_k_scale", "_v_scale"):
                vals = source[name][key]
                if len(vals) != 1 or not all(math.isfinite(x) and x > 0 for x in vals):
                    raise ValueError("Expected finite positive scalar scales")
        for name, layer in layers.items():
            for key in ("_q_scale", "_k_scale", "_v_scale"):
                tensor = getattr(layer, key)
                tensor.copy_(
                    torch.tensor(source[name][key], device=tensor.device).reshape_as(tensor)
                )
                setattr(layer, key + "_float", float(source[name][key][0]))
            layer.calculate_kv_scales = False
        self.model_runner.calculate_kv_scales = False
        return self.ail_snapshot()


# Imported by vLLM only in separate conditional-perplexity runs, not performance runs.
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class GoldContinuationProcessor(LogitsProcessor):
    """Force gold tokens while vLLM returns *raw*, pre-processor logprobs.

    This ensures token 2 onward reads the actual quantized KV cache. Use only
    for quality scoring; never include this processor in performance runs.
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        if vllm_config.model_config.logprobs_mode != "raw_logprobs":
            raise ValueError("Teacher-forced NLL requires raw_logprobs")
        self.requests = {}

    @classmethod
    def validate_params(cls, params):
        targets = (params.extra_args or {}).get("gold_tokens")
        if targets is not None and (
            not targets or not all(isinstance(t, int) and t >= 0 for t in targets)
        ):
            raise ValueError("gold_tokens must contain nonnegative token IDs")

    def is_argmax_invariant(self):
        return False

    def update_state(self, update):
        if update is None:
            return
        for index in update.removed:
            self.requests.pop(index, None)
        for index, params, _prompt, output in update.added:
            targets = (params.extra_args or {}).get("gold_tokens")
            if targets is not None:
                self.requests[index] = (targets, output)
            else:
                self.requests.pop(index, None)
        for src, dst, direction in update.moved:
            if direction.name == "SWAP":
                a, b = self.requests.pop(src, None), self.requests.pop(dst, None)
                if a is not None:
                    self.requests[dst] = a
                if b is not None:
                    self.requests[src] = b
            else:
                value = self.requests.pop(src, None)
                self.requests.pop(dst, None)
                if value is not None:
                    self.requests[dst] = value

    def apply(self, logits):
        for index, (targets, outputs) in self.requests.items():
            position = len(outputs)
            if position >= len(targets):
                raise RuntimeError("Teacher forcing exceeded target length")
            target = targets[position]
            logits[index].fill_(float("-inf"))
            logits[index, target] = 0.0
        return logits
