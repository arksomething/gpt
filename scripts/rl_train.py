"""GRPO post-training on top of an SFT checkpoint.

Uses TRL's GRPOTrainer rather than a hand-written loop -- the algorithm is not
where our risk is. What we supply is the part that is specific to this project:
the reward functions, the prompt distributions, and the diagnostic that says
whether the run learned anything at all.

That diagnostic is the point of the exercise at 25M. GRPO's advantage is
(r_i - mean) / std within a rollout group, so a group whose rollouts all score
alike contributes exactly zero gradient. All-wrong and all-right are equally
dead, and mean reward cannot tell either apart from healthy learning. Every
reward call here records the fraction of degenerate groups, so a run can prove
it measured "no gain" rather than silently measuring nothing.

Run under transformers==4.49.0: 5.x diverges from the SentencePiece tokenizer
on any string starting with whitespace (~6% of real documents).

    uv run --with trl --with 'transformers==4.49.0' --with protobuf \
        python scripts/rl_train.py --model runs/.../hf-chat --env format_constraint
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.rl_rewards import (  # noqa: E402
    group_advantages,
    reward_calibrated_answer,
    reward_constraints,
    reward_length,
    reward_no_repetition,
)

USER, ASSISTANT = "<|user|>", "<|assistant|>"


def _prompt(text: str) -> str:
    """Match the SFT template so RL does not shift the input distribution."""
    return f"{USER}\n{text}\n{ASSISTANT}\n"


# --------------------------------------------------------------------------
# Prompt distributions. Each environment is a (prompts, reward) pair; the
# trainer never learns which one it is running.
# --------------------------------------------------------------------------

_TOPICS = [
    "why leaves change color", "how bicycles stay upright", "what causes tides",
    "why bread rises", "how magnets work", "why the sky is blue",
    "what makes music sound sad", "how seeds become plants",
    "why ice floats", "how rainbows form", "what clouds are made of",
    "why onions make you cry", "how birds navigate", "what causes thunder",
    "why coffee smells good", "how soap cleans", "what makes popcorn pop",
    "why cats purr", "how bridges hold weight", "what makes rust",
]

_CHITCHAT = [
    "Hi there!", "How are you doing today?", "What can you help me with?",
    "I'm feeling a bit tired.", "Tell me something interesting.",
    "Good morning!", "What's your name?", "Thanks for the help!",
    "I had a long day.", "Do you like music?",
]

_TRIVIA = [
    ("What is the capital of France?", ["paris"]),
    ("How many days are in a week?", ["seven", "7"]),
    ("What color is a banana when ripe?", ["yellow"]),
    ("What planet do we live on?", ["earth"]),
    ("How many legs does a spider have?", ["eight", "8"]),
    ("What is frozen water called?", ["ice"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What gas do plants absorb?", ["carbon dioxide", "co2"]),
    # Deliberately obscure: a 25M model cannot know these, so the only way to
    # score above the wrong-answer penalty is to decline. Without such items the
    # abstention reward has nothing to teach.
    ("What was the population of Ulaanbaatar in 1974?", ["__unanswerable__"]),
    ("What is the middle name of the 14th mayor of Bristol?", ["__unanswerable__"]),
]


def build_dataset(env: str, n: int, seed: int):
    from datasets import Dataset

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    if env == "format_constraint":
        specs = [
            ({"max_words": 15}, "in 15 words or fewer"),
            ({"max_sentences": 2}, "in at most two sentences"),
            ({"requires_question": True}, "and end with a question"),
            ({"must_include_all": ["because"]}, "using the word 'because'"),
            ({"max_words": 25, "must_include_all": ["water"]},
             "in under 25 words, mentioning water"),
        ]
        for i in range(n):
            topic = _TOPICS[i % len(_TOPICS)]
            checks, phrasing = specs[i % len(specs)]
            rows.append({
                "prompt": _prompt(f"Explain {topic} {phrasing}."),
                "checks": json.dumps(checks),
            })
    elif env == "length_control":
        for i in range(n):
            target = [10, 20, 30, 40][i % 4]
            rows.append({
                "prompt": _prompt(f"Explain {_TOPICS[i % len(_TOPICS)]} in about {target} words."),
                "target_words": target,
            })
    elif env == "no_repetition":
        pool = _TOPICS + _CHITCHAT
        for i in range(n):
            rows.append({"prompt": _prompt(pool[i % len(pool)])})
    elif env == "calibrated_abstention":
        for i in range(n):
            q, gold = _TRIVIA[i % len(_TRIVIA)]
            rows.append({"prompt": _prompt(q), "gold": json.dumps(gold)})
    else:
        raise SystemExit(f"unknown env: {env}")

    rng.shuffle(rows)
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------
# Reward adapters. TRL calls reward_funcs(completions=..., **columns) and wants
# one float per completion.
# --------------------------------------------------------------------------


class RewardWithDiagnostics:
    """Wrap a reward so every group's health is recorded, not just its mean."""

    def __init__(self, env: str, num_generations: int):
        self.env = env
        self.num_generations = num_generations
        self.groups = Counter()  # "usable" / "degenerate"
        self.reward_sum = 0.0
        self.reward_n = 0
        self.__name__ = f"reward_{env}"

    def _score_one(self, completion: str, row: Dict[str, Any]) -> float:
        if self.env == "format_constraint":
            return reward_constraints(completion, json.loads(row["checks"]), dense=True)
        if self.env == "length_control":
            return reward_length(completion, int(row["target_words"]))
        if self.env == "no_repetition":
            return reward_no_repetition(completion)
        if self.env == "calibrated_abstention":
            gold = json.loads(row["gold"])
            if gold == ["__unanswerable__"]:
                # Nothing counts as correct; declining is the only positive move.
                from scripts.rl_rewards import is_abstention

                return 0.0 if is_abstention(completion) else -1.0
            return reward_calibrated_answer(completion, gold)
        raise SystemExit(f"unknown env: {self.env}")

    def __call__(self, completions, **kwargs) -> List[float]:
        texts = [c if isinstance(c, str) else c[0]["content"] for c in completions]
        n = len(texts)
        rows = [{k: v[i] for k, v in kwargs.items() if isinstance(v, list) and len(v) == n}
                for i in range(n)]
        scores = [self._score_one(t, r) for t, r in zip(texts, rows)]

        g = self.num_generations
        for i in range(0, n - g + 1, g):
            stats = group_advantages(scores[i : i + g])
            self.groups["degenerate" if stats.degenerate else "usable"] += 1
        self.reward_sum += sum(scores)
        self.reward_n += n
        return scores

    def report(self) -> Dict[str, Any]:
        total = sum(self.groups.values())
        return {
            "environment": self.env,
            "groups_total": total,
            "groups_usable": self.groups["usable"],
            "groups_degenerate": self.groups["degenerate"],
            "degenerate_fraction": (self.groups["degenerate"] / total) if total else None,
            "mean_reward": (self.reward_sum / self.reward_n) if self.reward_n else None,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF-format SFT checkpoint")
    ap.add_argument("--tokenizer", default="tokenizer/fast")
    ap.add_argument("--env", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--prompts", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_completion_length", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from scripts.build_fast_tokenizer import load_verified_fast_tokenizer

    try:
        tok = load_verified_fast_tokenizer(args.tokenizer)
    except Exception as e:  # parity failure must stop the run, not warn
        raise SystemExit(f"tokenizer parity check failed: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    )
    dataset = build_dataset(args.env, args.prompts, args.seed)
    reward = RewardWithDiagnostics(args.env, args.num_generations)

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        max_completion_length=args.max_completion_length,
        max_prompt_length=128,
        learning_rate=args.learning_rate,
        beta=args.beta,
        temperature=args.temperature,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
    )
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)

    report = reward.report()
    # A run whose groups were nearly all degenerate did not measure "no gain";
    # it measured nothing, and saying so is the whole point of running at 25M.
    frac = report["degenerate_fraction"]
    report["verdict"] = (
        "no-signal: nearly every rollout group scored uniformly"
        if frac is not None and frac > 0.9
        else "harness live: groups showed reward variance"
    )
    with open(os.path.join(args.output_dir, "rl_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
