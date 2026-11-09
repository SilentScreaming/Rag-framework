from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]

from rag_benchmark.generation.backends import resolve_openai_ca_bundle


class GenerationTransportTest(unittest.TestCase):
    def test_resolves_configured_relative_ca_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            cert_path = repo_root / ".certs" / "bundle.pem"
            cert_path.parent.mkdir()
            cert_path.write_text("test cert", encoding="utf-8")

            resolved = resolve_openai_ca_bundle(
                repo_root=repo_root,
                config={"ca_bundle": ".certs/bundle.pem"},
            )

            self.assertEqual(resolved, str(cert_path.resolve()))

    def test_unconfigured_local_ca_bundle_is_not_automatically_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            cert_path = repo_root / ".certs" / "fixture-bundle.pem"
            cert_path.parent.mkdir()
            cert_path.write_text("test cert", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "OPENAI_CA_BUNDLE": "",
                    "SSL_CERT_FILE": "",
                    "REQUESTS_CA_BUNDLE": "",
                },
            ):
                resolved = resolve_openai_ca_bundle(repo_root=repo_root, config={})

            self.assertIsNone(resolved)

    def test_missing_configured_ca_bundle_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                resolve_openai_ca_bundle(
                    repo_root=Path(tmp),
                    config={"ca_bundle": ".certs/missing.pem"},
                )


if __name__ == "__main__":
    unittest.main()
