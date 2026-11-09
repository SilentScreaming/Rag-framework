"""Create optional local settings and directories without installing anything."""

from __future__ import annotations

import json
from pathlib import Path

from rag_benchmark.paths import load_local_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config_dir = REPO_ROOT / "configs"
    local = config_dir / "paths.local.json"
    created = False
    if not local.exists():
        example = json.loads((config_dir / "paths.example.json").read_text(encoding="utf-8"))
        with local.open("x", encoding="utf-8") as handle:
            json.dump(example, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        created = True
    paths = load_local_paths(REPO_ROOT)
    # Initialization only creates project-local locations. A custom external
    # dataset/model/output path is left for the user to manage explicitly.
    managed = [REPO_ROOT / ".cache"]
    for key in ("datasets_root", "models_root", "outputs_root"):
        candidate = Path(paths[key])
        if candidate.is_relative_to(REPO_ROOT):
            managed.append(candidate)
    for location in managed:
        location.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "paths_config": "created" if created else "preserved",
        "config_file": "configs/paths.local.json",
        "created_or_existing_directories": [path.relative_to(REPO_ROOT).as_posix() for path in managed],
        "environment_created": False,
        "packages_installed": False,
        "downloads_started": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
