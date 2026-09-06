"""Evaluation-only gold-continuation processor for raw conditional NLL."""

from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class GoldContinuationProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        if vllm_config.model_config.logprobs_mode != "raw_logprobs":
            raise ValueError("Teacher forcing requires raw_logprobs")
        self.requests = {}

    @classmethod
    def validate_params(cls, params):
        targets = (params.extra_args or {}).get("gold_tokens")
        if targets is not None and (
            not targets or not all(isinstance(t, int) and t >= 0 for t in targets)
        ):
            raise ValueError("Invalid gold_tokens")

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
                raise RuntimeError("Teacher forcing exceeded target")
            logits[index].fill_(float("-inf"))
            logits[index, targets[position]] = 0.0
        return logits
