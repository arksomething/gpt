#!/usr/bin/env python3
"""Export a training checkpoint to a standalone Hugging Face directory.

The training model is a stock transformers LlamaForCausalLM, so export is a
re-serialization, not a conversion. What this script adds over copying files
is verification: the exported directory is reloaded via AutoModel/AutoTokenizer
(the way lm-eval and third parties will load it) and its logits are compared
against the training-path loader on fixed probe texts. A verification report
with hashes is written next to the export; mismatched logits abort.

Usage:
  uv run python scripts/export_hf.py \
    --checkpoint runs/probes/25m-aws-20260730/runs/probes/25m-placeholder/final \
    --model_config configs/model_25m.yaml \
    --tokenizer tokenizer/spm.model \
    --out runs/hf/25m-base
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)

PROBE_TEXTS = [
    "The history of the United States began",
    "Paris is the capital of France, and",
    "def add(a, b):\n    return",
]
LOGIT_TOLERANCE = 1e-4


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_training_model(checkpoint: Path, model_config: Path) -> LlamaForCausalLM:
    model_cfg = yaml.safe_load(model_config.read_text())["model"]
    config = LlamaConfig(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        intermediate_size=model_cfg["intermediate_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg.get(
            "num_key_value_heads", model_cfg["num_attention_heads"]
        ),
        max_position_embeddings=model_cfg["max_position_embeddings"],
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        hidden_act=model_cfg.get("hidden_act", "silu"),
        attention_bias=model_cfg.get("attention_bias", False),
        mlp_bias=model_cfg.get("mlp_bias", False),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        bos_token_id=model_cfg.get("bos_token_id", 1),
        eos_token_id=model_cfg.get("eos_token_id", 2),
        pad_token_id=model_cfg.get("pad_token_id", 3),
    )
    model = LlamaForCausalLM(config)
    state_dict = torch.load(checkpoint / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model_config", required=True, type=Path)
    parser.add_argument("--tokenizer", default=Path("tokenizer/spm.model"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    source = build_training_model(args.checkpoint, args.model_config)

    args.out.mkdir(parents=True, exist_ok=True)
    source.save_pretrained(args.out, safe_serialization=True)

    tokenizer = LlamaTokenizerFast(
        vocab_file=str(args.tokenizer),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_bos_token=True,
        add_eos_token=False,
        legacy=False,
    )
    tokenizer.save_pretrained(args.out)

    # --- verification through the third-party loading path
    reloaded = AutoModelForCausalLM.from_pretrained(args.out, torch_dtype=torch.float32)
    reloaded.eval()
    re_tok = AutoTokenizer.from_pretrained(args.out)

    worst = 0.0
    for text in PROBE_TEXTS:
        ids = re_tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            a = source(ids).logits
            b = reloaded(ids).logits
        worst = max(worst, float((a - b).abs().max()))
    passed = worst <= LOGIT_TOLERANCE

    report = {
        "source_checkpoint": str(args.checkpoint),
        "source_model_pt_sha256": _sha256(args.checkpoint / "model.pt"),
        "tokenizer_sha256": _sha256(args.tokenizer),
        "probe_texts": PROBE_TEXTS,
        "max_abs_logit_diff": worst,
        "tolerance": LOGIT_TOLERANCE,
        "passed": passed,
    }
    (args.out / "export_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("EXPORT VERIFICATION FAILED: logit mismatch")
    print(f"Export verified: {args.out}")


if __name__ == "__main__":
    main()
