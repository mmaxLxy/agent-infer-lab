"""CPU-only tests for the evaluation logits processor, using server packages."""

import unittest
from types import SimpleNamespace as NS

import torch
from fp8_runtime import GoldContinuationProcessor


class RuntimeTests(unittest.TestCase):
    def processor(self):
        return GoldContinuationProcessor(
            NS(model_config=NS(logprobs_mode="raw_logprobs")), "cpu", False
        )

    def update(self, added=(), removed=(), moved=()):
        return NS(added=added, removed=removed, moved=moved)

    def test_continuation(self):
        p = self.processor()
        output = []
        p.update_state(
            self.update(added=[(0, NS(extra_args={"gold_tokens": [2, 4]}), [1], output)])
        )
        self.assertEqual(p.apply(torch.zeros(1, 6)).argmax(-1).item(), 2)
        output.append(2)
        self.assertEqual(p.apply(torch.zeros(1, 6)).argmax(-1).item(), 4)
        output.append(4)
        with self.assertRaises(RuntimeError):
            p.apply(torch.zeros(1, 6))

    def test_swap_move_remove(self):
        p = self.processor()
        p.update_state(
            self.update(
                added=[
                    (0, NS(extra_args={"gold_tokens": [2]}), [], []),
                    (1, NS(extra_args={"gold_tokens": [3]}), [], []),
                ]
            )
        )
        p.update_state(self.update(moved=[(0, 1, NS(name="SWAP"))]))
        self.assertEqual(p.apply(torch.zeros(2, 6)).argmax(-1).tolist(), [3, 2])
        p.update_state(self.update(removed=[0], moved=[(1, 0, NS(name="UNIDIRECTIONAL"))]))
        self.assertEqual(list(p.requests), [0])
        self.assertEqual(p.apply(torch.zeros(1, 6)).argmax(-1).item(), 2)
        p.update_state(self.update(removed=[0]))
        self.assertFalse(p.requests)

    def test_regular_request_untouched(self):
        p = self.processor()
        p.update_state(self.update(added=[(0, NS(extra_args=None), [], [])]))
        original = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.equal(original, p.apply(original.clone())))


if __name__ == "__main__":
    unittest.main()
