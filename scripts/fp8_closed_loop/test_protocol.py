"""CPU tests; no vLLM/GPU import required."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from run_suite import extract, fingerprint, numeric, save, validate_snapshot


class ProtocolTests(unittest.TestCase):
    def test_numeric_scoring(self):
        self.assertEqual(extract("reason 3, final #### -1,250.0"), numeric("-1250"))
        self.assertIsNone(extract("The answer is 5"))
        self.assertEqual(extract("The answer is 5", False), "5")
        self.assertEqual(extract("#### 10\n#### 20"), "2E+1")

    def test_scale_validation(self):
        data = {
            "rows": [
                {"layer": str(i), "_q_scale": [1.0], "_k_scale": [0.1], "_v_scale": [0.2]}
                for i in range(24)
            ]
        }
        validate_snapshot(data)
        self.assertEqual(fingerprint(data), fingerprint({"rows": list(reversed(data["rows"]))}))
        other = copy.deepcopy(data)
        other["rows"][0]["_k_scale"] = [0.3]
        self.assertNotEqual(fingerprint(data), fingerprint(other))
        other["rows"][0]["_k_scale"] = [float("nan")]
        with self.assertRaises(RuntimeError):
            validate_snapshot(other)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            save(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                save(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})


if __name__ == "__main__":
    unittest.main()
