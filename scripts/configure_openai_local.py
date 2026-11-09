"""Save non-secret OpenAI-compatible model settings; use OPENAI_API_KEY for credentials."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model ID supported by your configured endpoint.")
    parser.add_argument("--input-price", type=float, default=None, help="Optional USD per million input tokens.")
    parser.add_argument("--output-price", type=float, default=None, help="Optional USD per million output tokens.")
    parser.add_argument("--ca-bundle", default=None, help="Optional explicit CA bundle; TLS verification remains enabled.")
    args = parser.parse_args()
    if (args.input_price is None) != (args.output_price is None):
        parser.error("Provide both --input-price and --output-price, or neither")
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in (args.input_price, args.output_price)):
        parser.error("Prices must be finite and nonnegative")
    return args


def main() -> int:
    args = parse_args()
    path = REPO_ROOT / "configs" / "openai.local.json"
    # Rebuild only the supported non-secret fields. Existing unknown fields
    # are neither read nor copied to the new local configuration.
    config = {
        "model": args.model,
        "input_cost_per_1m_tokens": args.input_price,
        "output_cost_per_1m_tokens": args.output_price,
    }
    if args.ca_bundle:
        ca_bundle = Path(args.ca_bundle).expanduser()
        if not ca_bundle.is_absolute():
            ca_bundle = REPO_ROOT / ca_bundle
        if not ca_bundle.is_file():
            raise FileNotFoundError("The explicitly configured CA bundle does not exist")
        config["ca_bundle"] = str(ca_bundle.resolve())
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "saved",
        "path": "configs/openai.local.json",
        "model": args.model,
        "prices_configured": args.input_price is not None,
        "credential_source": "Set OPENAI_API_KEY in your environment",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
