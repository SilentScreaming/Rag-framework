from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.embed_kilt_candidate_cache import finalize_manifest


class KiltEmbeddingManifestTest(unittest.TestCase):
    def test_finalize_manifest_hashes_vectors_and_combined_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vectors_path = root / "vectors.npy"
            vectors_path.write_bytes(b"frozen vectors")
            manifest_path = root / "manifest.json"

            manifest = finalize_manifest(
                manifest={"cache_hash": "candidate-hash"},
                manifest_path=manifest_path,
                vectors_path=vectors_path,
                embedding_model="model@revision",
                dimensions=1024,
                document_count=2,
                embedding_seconds=1.5,
            )

            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["vectors_sha256"],
            hashlib.sha256(b"frozen vectors").hexdigest(),
        )
        self.assertEqual(persisted["frozen_artifact_hash"], manifest["frozen_artifact_hash"])
        self.assertEqual(persisted["embedded_documents"], 2)


if __name__ == "__main__":
    unittest.main()
