from __future__ import annotations

import unittest


class ImportTest(unittest.TestCase):
    def test_package_imports(self) -> None:
        import rag_benchmark

        self.assertEqual(rag_benchmark.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
