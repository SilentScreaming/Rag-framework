"""Inspect the active Python environment; GPU support is optional."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-models", action="store_true", help="Require optional neural-model dependencies.")
    parser.add_argument("--require-cuda", action="store_true", help="Import PyTorch and require an available CUDA device.")
    args = parser.parse_args()
    required = ("numpy", "yaml", "pydantic", "rank_bm25", "requests", "openai", "httpx", "tiktoken")
    optional = ("torch", "transformers", "sentence_transformers", "sentencepiece", "accelerate",
                "datasets", "huggingface_hub", "faiss")
    available = {name: importlib.util.find_spec(name) is not None for name in (*required, *optional)}
    available["rag_benchmark"] = importlib.util.find_spec("rag_benchmark") is not None
    errors = []
    if sys.version_info < (3, 11):
        errors.append("Python 3.11 or newer is required")
    errors.extend(f"Missing required package: {name}" for name in required if not available[name])
    if not available["rag_benchmark"]:
        errors.append("The framework package is not installed. Run python -m pip install -e . from the checkout.")
    if args.require_models:
        errors.extend(f"Missing optional model package: {name}" for name in optional if name != "faiss" and not available[name])
    result = {
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "isolated_environment": sys.prefix != sys.base_prefix,
        "packages": available,
        "cuda_checked": args.require_cuda,
    }
    if args.require_cuda:
        try:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            if not result["cuda_available"]:
                errors.append("CUDA was explicitly required but is unavailable")
        except ImportError:
            errors.append("CUDA was explicitly required but PyTorch is unavailable")
    result.update(status="failed" if errors else "ok", errors=errors)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
