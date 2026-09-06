import json
import tempfile
import unittest
from pathlib import Path

from run_suite import extract, numeric, save


class Tests(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(extract("work #### -1,250.0"), numeric("-1250"))
        self.assertIsNone(extract("answer 5"))
        self.assertEqual(extract("answer 5", False), "5")

    def test_exclusive_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            save(p, {"a": 1})
            with self.assertRaises(FileExistsError):
                save(p, {"a": 2})
            self.assertEqual(json.loads(p.read_text()), {"a": 1})


if __name__ == "__main__":
    unittest.main()
