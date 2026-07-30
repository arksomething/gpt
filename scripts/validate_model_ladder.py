#!/usr/bin/env python3
"""Validate model-ladder configs using the training model implementation.

Models are constructed on PyTorch's ``meta`` device, so even the 1B config has
no parameter storage allocated. Calling ``tie_weights`` explicitly is important
because Transformers cannot preserve aliases while initially constructing some
models on the meta device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = tuple(
    ROOT / "configs" / f"model_{name}.yaml"
    for name in ("25m", "60m", "150m", "350m", "1b")
)
MODEL_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "hidden_act",
    "attention_bias",
    "mlp_bias",
    "tie_word_embeddings",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
)


def load_model_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("model"), dict):
        raise ValueError(f"{path}: expected a top-level 'model' mapping")
    return document


def build_meta_model(model_config: dict[str, Any]) -> LlamaForCausalLM:
    """Mirror ``scripts/train.py`` construction without allocating weights."""
    missing = [key for key in MODEL_KEYS if key not in model_config]
    if missing:
        raise ValueError(f"missing model keys: {', '.join(missing)}")
    config = LlamaConfig(**{key: model_config[key] for key in MODEL_KEYS})
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    model.tie_weights()
    return model


def validate_config(path: Path, tolerance: float = 0.02) -> dict[str, Any]:
    document = load_model_file(path)
    ladder = document.get("ladder") or {}
    target = int(ladder["target_parameters"])
    model = build_meta_model(document["model"])

    total = sum(parameter.numel() for parameter in model.parameters())
    embedding = model.get_input_embeddings().weight.numel()
    non_embedding = total - embedding
    delta = (total - target) / target

    result = {
        "name": ladder.get("name", path.stem.removeprefix("model_")),
        "path": str(path),
        "target_parameters": target,
        "total_parameters": total,
        "non_embedding_parameters": non_embedding,
        "embedding_parameters": embedding,
        "relative_delta": delta,
        "within_tolerance": abs(delta) <= tolerance,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "max_position_embeddings": model.config.max_position_embeddings,
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "parameter_device": str(next(model.parameters()).device),
    }
    if result["parameter_device"] != "meta":
        raise AssertionError(f"{path}: validator allocated real parameter storage")
    return result


def _format_table(results: list[dict[str, Any]]) -> str:
    header = (
        f"{'model':<7} {'target':>11} {'total':>11} {'non-embed':>11} "
        f"{'delta':>8} {'shape':>24} {'GQA':>7}"
    )
    rows = [header, "-" * len(header)]
    for result in results:
        shape = (
            f"{result['num_hidden_layers']}x{result['hidden_size']}"
            f"/{result['intermediate_size']}"
        )
        gqa = (
            f"{result['num_attention_heads']}:"
            f"{result['num_key_value_heads']}"
        )
        rows.append(
            f"{result['name']:<7} "
            f"{result['target_parameters'] / 1e6:>10.1f}M "
            f"{result['total_parameters'] / 1e6:>10.3f}M "
            f"{result['non_embedding_parameters'] / 1e6:>10.3f}M "
            f"{result['relative_delta']:>+7.2%} "
            f"{shape:>24} {gqa:>7}"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_CONFIGS),
        help="Model YAML files (defaults to the complete ladder)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Allowed fractional difference from target (default: 0.02)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    results = [validate_config(path, args.tolerance) for path in args.configs]
    print(json.dumps(results, indent=2) if args.json else _format_table(results))
    failures = [result["name"] for result in results if not result["within_tolerance"]]
    if failures:
        raise SystemExit(
            "Outside parameter-count tolerance: " + ", ".join(failures)
        )


if __name__ == "__main__":
    main()
