#!/usr/bin/env python3
"""Interactive chat REPL for SFT checkpoints.

Wraps the gpt-chatml template (<|user|>...<|end|><|assistant|>), keeps a short
rolling history, stops generation at the model's own <|end|> marker, and
applies sampling defaults suited to small models (repetition penalty on).

Usage:
  uv run python scripts/chat_repl.py \
    --checkpoint runs/sft/25m-dolly-rehearsal-20260730/runs/sft/25m-dolly-rehearsal/final

Commands inside the REPL: /reset clears history, /quit exits.
"""

from __future__ import annotations

import argparse

import sentencepiece as spm
import torch

from scripts.infer import load_model

END_MARKER = "<|end|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"

import re

# Marker or marker debris: "<|end|>", "|end|>", "<|end", "<|user|>", "<|" ...
_MARKER_RE = re.compile(r"<\|[a-z]*\|?>?|\|(?:end|user|assistant)\|>?")


def _clean_marker_prefix(text: str) -> str:
    """Strip marker debris (and stray whitespace) from the start of a reply."""
    while True:
        stripped = text.lstrip()
        m = _MARKER_RE.match(stripped)
        if not m:
            return stripped
        text = stripped[m.end():]


def generate_reply(
    model,
    sp,
    history: list[tuple[str, str]],
    user_text: str,
    device: str,
    max_new_tokens: int = 120,
    temperature: float = 0.45,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
) -> str:
    prompt = ""
    for u, a in history[-3:]:  # short rolling window; it's a 25M model
        prompt += f"{USER}{u}{END_MARKER}{ASSISTANT}{a}{END_MARKER}"
    prompt += f"{USER}{user_text}{END_MARKER}{ASSISTANT}"

    ids = [sp.bos_id()] + sp.encode(prompt)
    input_ids = torch.tensor([ids], device=device)

    generated: list[int] = []
    past = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = (
                model(input_ids, use_cache=True)
                if past is None
                else model(input_ids[:, -1:], past_key_values=past, use_cache=True)
            )
            past = out.past_key_values
            logits = out.logits[0, -1].float()

            for token in set(generated[-64:]):
                if logits[token] > 0:
                    logits[token] /= repetition_penalty
                else:
                    logits[token] *= repetition_penalty

            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = probs.sort(descending=True)
            keep = sorted_probs.cumsum(0) - sorted_probs < top_p
            keep[0] = True
            choice = sorted_idx[keep][torch.multinomial(sorted_probs[keep] / sorted_probs[keep].sum(), 1)]
            token = int(choice)

            if token == sp.eos_id():
                break
            generated.append(token)
            input_ids = torch.cat([input_ids, torch.tensor([[token]], device=device)], dim=1)

            # String-level stop. Small models emit imperfect markers
            # ("|end|>", "<|end", a bare "<|user|>" turn start), so match
            # marker debris by pattern, drop any that prefixes the reply,
            # and cut at the first one that appears mid-text.
            text = _clean_marker_prefix(sp.decode(generated))
            m = _MARKER_RE.search(text)
            if m:
                return text[: m.start()].strip()

    return _clean_marker_prefix(sp.decode(generated)).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/sft/25m-dolly-rehearsal-20260730/runs/sft/25m-dolly-rehearsal/final",
    )
    parser.add_argument("--model_config", default="configs/model_25m.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/spm.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    args = parser.parse_args()

    print(f"loading model on {args.device} ...")
    model, _ = load_model(args.checkpoint, args.model_config, args.device)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print("ready. /reset clears history, /quit exits. (25M params: expect confident nonsense)\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        reply = generate_reply(
            model,
            sp,
            history,
            user_text,
            args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"bot> {reply}\n")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
