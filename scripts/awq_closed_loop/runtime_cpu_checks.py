import unittest
from types import SimpleNamespace as NS

import torch
from awq_runtime import GoldContinuationProcessor


class Tests(unittest.TestCase):
    def test_positions_and_regular_request(self):
        p = GoldContinuationProcessor(
            NS(model_config=NS(logprobs_mode="raw_logprobs")), "cpu", False
        )
        output = []
        p.update_state(
            NS(
                removed=[],
                moved=[],
                added=[(0, NS(extra_args={"gold_tokens": [2, 4]}), [], output)],
            )
        )
        self.assertEqual(p.apply(torch.zeros(1, 6)).argmax(-1).item(), 2)
        output.append(2)
        self.assertEqual(p.apply(torch.zeros(1, 6)).argmax(-1).item(), 4)
        p.update_state(NS(removed=[0], moved=[], added=[(0, NS(extra_args=None), [], [])]))
        original = torch.tensor([[1.0, 3.0, 2.0]])
        self.assertTrue(torch.equal(original, p.apply(original.clone())))


if __name__ == "__main__":
    unittest.main()
