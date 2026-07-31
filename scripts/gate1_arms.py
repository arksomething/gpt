"""Gate 1 mixture arms.

Every arm trains on the same union corpus (``data/gate1/union-v2``) and differs
only in ``data.indexed.source_weights``. Identical documents, identical
tokenization, identical filtering: the only variable in the experiment is the
sampling mixture, so corpus-construction variance cannot be mistaken for a
mixture effect.

Weights are percentages of each run's 250M-token budget and are checked to sum
to 100 at import time. Run ``uv run gate1-arms`` to emit the train configs.
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Dict

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "train_25m_probe.yaml")
OUT_DIR = os.path.join(REPO_ROOT, "configs", "gate1")
UNION_CORPUS = "data/gate1/union-v2"
G6_XLARGE_HOURLY = 0.81  # us-east-1 on-demand; spot runs cost less, never more

# The conversation sub-corpus. OASST's planned 0.5 point is folded into
# stackexchange: the chat lane (data/chat/v1, 0.6M tokens) is too small to
# supply 0.5% of a 250M-token run, and inventing tokens it does not have would
# misstate the mixture.
CONVERSATION = {
    "stackexchange": 5.5,
    "youtube": 2.5,
    "ubuntu_irc": 1.5,
    "github_archive": 1.5,
    "uk_hansard": 1.0,
}
# wikiteam was designed at 2.0 and cannot supply it. Its content is dominated by
# near-duplicate template pages (`Template:Cite web/doc` and friends), so dedup
# removes most of it: the whole filtered source yields ~0.5M usable tokens, and a
# 2% slice of a 250M-token run would have meant repeating it ten times over. That
# is memorization, not a reference register. The mass moves to Wikipedia and
# LibreTexts, which keeps the reference slice at 5 and keeps it de-monopolized.
REFERENCE = {"wiki": 3.0, "wikiteam": 0.45, "libretexts": 1.55}


def _mix(**overrides: float) -> Dict[str, float]:
    """B0.1 with named substitutions; keys set to 0 are dropped."""
    mix: Dict[str, float] = {
        "finepdfs": 20.0,
        "fineweb": 28.0,
        "dclm": 15.0,
        **CONVERSATION,
        "finemath": 8.0,
        "code": 9.0,
        "narrative": 3.0,
        **REFERENCE,
    }
    mix.update(overrides)
    return {k: v for k, v in mix.items() if v > 0}


def _scale_non_conversation(mix: Dict[str, float], target: float) -> Dict[str, float]:
    """Rescale every non-conversation source so they sum to ``target``."""
    others = {k: v for k, v in mix.items() if k not in CONVERSATION}
    factor = target / sum(others.values())
    return {k: (v * factor if k in others else v) for k, v in mix.items()}


# P2 doubles the whole conversation sub-corpus, keeping its internal ratios, and
# pays for it proportionally from every other source rather than from one.
_P2 = _scale_non_conversation(
    {**_mix(), **{k: v * 2 for k, v in CONVERSATION.items()}}, 100.0 - 24.0
)

ARMS: Dict[str, Dict[str, object]] = {
    "b0-seed1": {"seed": 1337, "mix": _mix(), "note": "baseline B0.1"},
    "b0-seed2": {"seed": 2024, "mix": _mix(), "note": "baseline B0.1"},
    "b0-seed3": {"seed": 7331, "mix": _mix(), "note": "baseline B0.1"},
    "p1": {
        "seed": 1337,
        "mix": _mix(fineweb=43.0, dclm=5.0, finepdfs=15.0),
        "note": "harder edu filtering: Edu 28->43, DCLM 15->5, FinePDFs 20->15",
    },
    "p2": {
        "seed": 1337,
        "mix": _P2,
        "note": "conversation dose 12->24, sub-corpus ratios preserved",
    },
    "p3": {
        "seed": 1337,
        "mix": _mix(finemath=12.71, code=14.29, dclm=10.0, fineweb=23.0),
        "note": "STEM 17->27, paid from DCLM and Edu",
    },
    "p4": {
        "seed": 1337,
        "mix": _mix(cosmopedia=15.0, fineweb=13.0),
        "note": "synthetic textbooks: Cosmopedia 15 in, Edu 28->13",
    },
    "p5": {
        "seed": 1337,
        "mix": _mix(dclm=30.0, fineweb=13.0),
        "note": "anti-P1: DCLM 15->30, Edu 28->13",
    },
}

for _name, _arm in ARMS.items():
    _total = sum(_arm["mix"].values())  # type: ignore[union-attr]
    if abs(_total - 100.0) > 1e-6:
        raise ValueError(f"arm {_name} weights sum to {_total}, not 100")


def build_config(name: str, arm: Dict[str, object], base: dict) -> dict:
    cfg = copy.deepcopy(base)
    indexed = cfg["data"]["indexed"]
    indexed["enabled"] = True
    indexed["train_dir"] = f"{UNION_CORPUS}/train"
    indexed["val_dir"] = f"{UNION_CORPUS}/validation"
    # Per-source validation loss must cover every source in the union corpus,
    # not just the ones this arm trains on: decision rule 2 asks whether an arm
    # sacrificed a register, which is unanswerable if that register is unscored.
    indexed["source_weights"] = {k: round(v / 100.0, 6) for k, v in arm["mix"].items()}  # type: ignore[union-attr]
    indexed["validation_source_weights"] = None
    cfg["training"]["seed"] = arm["seed"]
    cfg["training"]["output_dir"] = f"runs/gate1/{name}"
    cfg["budget"]["throughput_path"] = f"runs/gate1/{name}/throughput.json"
    # The template ships zeroes, which silently disables the budget guard
    # entirely (train.py skips the check when any value is <= 0). A g6.xlarge
    # at ~60k tok/s should finish 250M tokens for about $0.95, so a $5 ceiling
    # leaves room for a slow box while still catching a throughput collapse
    # before it runs to the 8h kill-switch.
    cfg["budget"]["hourly_rate"] = G6_XLARGE_HOURLY
    cfg["budget"]["max_cost"] = 5.0
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit Gate 1 arm train configs.")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, arm in ARMS.items():
        cfg = build_config(name, arm, base)
        path = os.path.join(args.out_dir, f"{name}.yaml")
        with open(path, "w") as f:
            f.write(f"# Gate 1 arm {name}: {arm['note']}\n")
            f.write("# Generated by scripts/gate1_arms.py -- edit the table there.\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"{name:9s} seed={arm['seed']:<5} {arm['note']}")
    print(f"\nWrote {len(ARMS)} arm configs to {args.out_dir}")


if __name__ == "__main__":
    main()
