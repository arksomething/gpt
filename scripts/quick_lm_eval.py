#!/usr/bin/env python3
"""Run quick, reproducible lm-eval benchmarks.

This script wraps the exact "quick check" recipe used in this repo:
- model backend: hf
- tasks: hellaswag, arc_easy, arc_challenge, piqa, winogrande, lambada_openai
- 0-shot
- limit=200
- batch_size=1

Examples:
    uv run quick-lm-eval --models local
    uv run quick-lm-eval --models local,gpt2,opt125m,smollm2
    uv run quick-lm-eval --models local --local_pretrained runs/llama-100m-v4/hf-eval --local_tokenizer runs/llama-100m-v4/hf-eval
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_TASKS = "hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai"
KNOWN_MODELS = ("local", "gpt2", "opt125m", "smollm2")


def _build_eval_spec(
    model_name: str,
    local_pretrained: str,
    local_tokenizer: str,
    output_root: str,
) -> tuple[str, str, str]:
    if model_name == "local":
        return (
            "local",
            (
                f"pretrained={local_pretrained},"
                f"tokenizer={local_tokenizer},"
                "use_fast_tokenizer=False"
            ),
            "runs/llama-100m-v3/lm-eval-sample",
        )
    if model_name == "gpt2":
        return ("gpt2", "pretrained=gpt2", f"{output_root}/gpt2")
    if model_name == "opt125m":
        return ("facebook/opt-125m", "pretrained=facebook/opt-125m", f"{output_root}/opt-125m")
    if model_name == "smollm2":
        return (
            "HuggingFaceTB/SmolLM2-135M",
            "pretrained=HuggingFaceTB/SmolLM2-135M",
            f"{output_root}/smollm2-135m",
        )
    raise ValueError(f"Unknown model preset: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quick lm-eval benchmarks for local and baseline models."
    )
    parser.add_argument(
        "--models",
        default="local",
        help=(
            "Comma-separated model presets. "
            f"Supported: {', '.join(KNOWN_MODELS)}. Default: local"
        ),
    )
    parser.add_argument(
        "--lm_eval_bin",
        default=".venv/bin/lm_eval",
        help="Path to lm_eval executable. Default: .venv/bin/lm_eval",
    )
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS,
        help=f"Comma-separated lm-eval tasks. Default: {DEFAULT_TASKS}",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Evaluation device passed to lm_eval (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--batch_size",
        default="1",
        help="Batch size for lm_eval. Use 'auto' if desired.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples. Default: 0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help=(
            "Per-task sample cap for fast checks. "
            "Use a larger value or remove this flag for final metrics."
        ),
    )
    parser.add_argument(
        "--output_root",
        default="runs/lm-eval-baselines",
        help="Root output directory for baseline runs.",
    )
    parser.add_argument(
        "--local_pretrained",
        default="runs/llama-100m-v3/hf-eval",
        help="HF-format local model path for the 'local' preset.",
    )
    parser.add_argument(
        "--local_tokenizer",
        default="runs/llama-100m-v3/hf-eval",
        help="Tokenizer path for the 'local' preset.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lm_eval_bin = Path(args.lm_eval_bin)
    if not lm_eval_bin.exists():
        raise SystemExit(
            f"lm_eval executable not found at {args.lm_eval_bin}. "
            "Install lm-eval-harness or pass --lm_eval_bin."
        )

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested_models if m not in KNOWN_MODELS]
    if unknown:
        raise SystemExit(
            f"Unknown --models entries: {unknown}. "
            f"Supported presets: {', '.join(KNOWN_MODELS)}"
        )

    for model_key in requested_models:
        display_name, model_args, output_path = _build_eval_spec(
            model_name=model_key,
            local_pretrained=args.local_pretrained,
            local_tokenizer=args.local_tokenizer,
            output_root=args.output_root,
        )
        Path(output_path).mkdir(parents=True, exist_ok=True)

        cmd = [
            args.lm_eval_bin,
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--device",
            args.device,
            "--tasks",
            args.tasks,
            "--batch_size",
            str(args.batch_size),
            "--num_fewshot",
            str(args.num_fewshot),
            "--limit",
            str(args.limit),
            "--output_path",
            output_path,
        ]

        print(f"\n==> {display_name}")
        print(" ".join(shlex.quote(part) for part in cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
