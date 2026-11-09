"""Path configuration helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _project_root_candidate(repo_root: Path | None = None) -> Path | None:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    configured = os.environ.get("RAG_FRAMEWORK_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    # Editable installations retain this source layout. A wheel installation
    # must never mistake site-packages or its parent for a writable project.
    checkout = Path(__file__).resolve().parents[2]
    if ((checkout / "pyproject.toml").is_file()
            and (checkout / "src" / "rag_benchmark" / "__init__.py").is_file()
            and (checkout / "configs").is_dir()):
        return checkout
    return None


def resolve_project_root(repo_root: Path | None = None) -> Path:
    """Find a configuration workspace, never an installed package directory.

    Library callers can pass a workspace explicitly or set RAG_FRAMEWORK_ROOT.
    Editable installs additionally discover their source checkout.
    """
    root = _project_root_candidate(repo_root)
    if root is None or not (root / "configs").is_dir():
        raise FileNotFoundError(
            "A configuration workspace is required. Pass repo_root or set "
            "RAG_FRAMEWORK_ROOT to a directory containing configs/. "
            "Editable installations can discover their source checkout."
        )
    return root


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def resolve_repo_path(value: str, repo_root: Path) -> Path:
    """Interpret relative paths from the checkout, regardless of the shell cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()


def path_config_file(repo_root: Path, *, use_local: bool = True) -> Path:
    local = repo_root / "configs" / "paths.local.json"
    return local if use_local and local.is_file() else repo_root / "configs" / "paths.example.json"


def load_local_paths(repo_root: Path | None = None, *, use_local: bool = True) -> dict[str, Any]:
    """Load optional local overrides, falling back to repository-local examples."""
    root = resolve_project_root(repo_root)
    path = path_config_file(root, use_local=use_local)
    data = load_json(path)
    for key in ("datasets_root", "models_root", "outputs_root"):
        value = data.get(key)
        if not value:
            raise ValueError(f"Missing required path key: {key}")
        data[key] = str(resolve_repo_path(str(value), root))
    for name, entry in data.get("datasets", {}).items():
        if not isinstance(entry, dict) or not entry.get("root"):
            raise ValueError(f"Dataset {name!r} must provide a root path")
        entry["root"] = str(resolve_repo_path(str(entry["root"]), root))
    return data


def configured_datasets_root(repo_root: Path | None = None) -> Path:
    return Path(load_local_paths(repo_root)["datasets_root"])


def configure_cache_environment(repo_root: Path | None = None) -> None:
    """Default model caches to this checkout without creating files on import.

    Explicit environment variables remain user-controlled. Set them before
    starting a command to choose a different cache location.
    """
    root = _project_root_candidate(repo_root)
    override = os.environ.get("RAG_FRAMEWORK_CACHE_ROOT")
    if override:
        cache = resolve_repo_path(override, root or Path.cwd())
    elif root is not None:
        cache = root / ".cache"
    else:
        # A bare wheel import uses a user cache, never a package installation.
        user_cache = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
        cache = user_cache / "rag-framework"
    defaults = {
        "XDG_CACHE_HOME": cache,
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "SENTENCE_TRANSFORMERS_HOME": cache / "sentence-transformers",
        "TIKTOKEN_CACHE_DIR": cache / "tiktoken",
    }
    for name, location in defaults.items():
        os.environ.setdefault(name, str(location))
