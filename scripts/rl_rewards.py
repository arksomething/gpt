"""Reward functions and group-advantage math for GRPO.

An RL environment here is just a pair: a prompt distribution and a reward
function. The trainer never needs to know which environment it is running, the
same way prepare_data does not need to know which corpus source it is reading.
Everything in this module is a pure function so it can be tested without a GPU,
a model, or a network.

The governing constraint, and the reason some environments are worth building
and others are not:

    GRPO learns from *differences between rollouts of the same prompt*. The
    advantage is (r_i - mean(r)) / std(r). If every rollout in a group earns the
    same reward, std is zero, every advantage is zero, and the update is a
    no-op no matter how many GPU-hours it burns.

So a reward is useful when the current policy already produces a spread of
outcomes. A reward that is always 0 (math for a 25M model) and a reward that is
always 1 (a constraint too easy to violate) are equally worthless, and they look
identical to a training curve that only logs the mean.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

# Phrases that count as an explicit refusal to guess. Kept small and literal:
# a fuzzy abstention detector would let the model earn the abstain reward with
# hedged answers that are really still guesses.
ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i cannot answer",
    "i can't answer",
    "unsure",
)


@dataclass(frozen=True)
class GroupStats:
    """Advantages for one prompt's rollout group, plus a health verdict."""

    advantages: tuple[float, ...]
    mean: float
    std: float
    degenerate: bool  # every rollout scored alike -> zero gradient

    @property
    def usable(self) -> bool:
        return not self.degenerate


def group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> GroupStats:
    """Group-relative advantages, with the dead-group case made explicit.

    Reporting `degenerate` is the point. A run whose groups are nearly all
    degenerate is not "learning slowly", it is not learning at all, and the mean
    reward alone will not say so.
    """
    n = len(rewards)
    if n == 0:
        return GroupStats((), 0.0, 0.0, True)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < eps:
        return GroupStats(tuple(0.0 for _ in rewards), mean, 0.0, True)
    return GroupStats(
        tuple((r - mean) / std for r in rewards), mean, std, False
    )


# --------------------------------------------------------------------------
# Mechanical rewards. These work at 25M because the behaviours are ones the
# base policy already produces sometimes -- which is exactly the condition for
# a non-degenerate group.
# --------------------------------------------------------------------------


def reward_termination(terminated: bool) -> float:
    """1.0 if the rollout emitted EOS instead of running to the token cap."""
    return 1.0 if terminated else 0.0


def reward_no_repetition(text: str, n: int = 4) -> float:
    """1.0 for no repeated n-gram, falling toward 0.0 as looping increases.

    Small models loop, and a loop is cheap to detect exactly: the fraction of
    distinct n-grams over total n-grams.
    """
    words = text.split()
    if len(words) < n + 1:
        return 1.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def reward_length(text: str, target_words: int, tolerance: float = 0.5) -> float:
    """1.0 at the target length, decaying linearly to 0.0 outside tolerance."""
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    actual = len(text.split())
    rel = abs(actual - target_words) / target_words
    return max(0.0, 1.0 - rel / tolerance) if tolerance > 0 else float(rel == 0)


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def reward_sentence_count(text: str, target: int) -> float:
    return 1.0 if _sentence_count(text) == target else 0.0


def reward_forbidden_substring(text: str, forbidden: str) -> float:
    return 0.0 if forbidden.lower() in text.lower() else 1.0


def reward_must_contain(text: str, required: str) -> float:
    return 1.0 if required.lower() in text.lower() else 0.0


# --------------------------------------------------------------------------
# Calibrated abstention. This is the useful form of "factual accuracy" at small
# scale: the model cannot know many facts, but it can learn when to decline.
# --------------------------------------------------------------------------


def is_abstention(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in ABSTENTION_MARKERS)


def reward_calibrated_answer(
    text: str,
    gold_answers: Sequence[str],
    abstain_credit: float = 0.0,
    wrong_penalty: float = -1.0,
) -> float:
    """Ternary reward: correct / abstained / confidently wrong.

    Deliberately not binary. A binary correct-or-not reward pays off for any
    guess whose success probability is above zero, so it trains a model to
    guess confidently -- which is hallucination with extra steps. Giving
    abstention a value strictly between wrong and correct is what makes
    declining rational when the model does not know.
    """
    if not gold_answers:
        raise ValueError("gold_answers must not be empty")
    low = text.lower()
    if any(g.lower() in low for g in gold_answers):
        return 1.0
    if is_abstention(text):
        return abstain_credit
    return wrong_penalty


# --------------------------------------------------------------------------
# Environment registry. Adding an environment is a table entry, not a new code
# path through the trainer.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Environment:
    name: str
    description: str
    reward: Callable[..., float]
    works_below_100m: bool
    needs_external_judge: bool = False


ENVIRONMENTS: Dict[str, Environment] = {
    e.name: e
    for e in (
        Environment(
            "termination",
            "Emit EOS rather than running to the token cap.",
            reward_termination,
            works_below_100m=True,
        ),
        Environment(
            "no_repetition",
            "Avoid n-gram loops.",
            reward_no_repetition,
            works_below_100m=True,
        ),
        Environment(
            "length_control",
            "Answer at a requested length.",
            reward_length,
            works_below_100m=True,
        ),
        Environment(
            "format_constraint",
            "IFEval-style verifiable constraints (sentence counts, "
            "required/forbidden strings).",
            reward_sentence_count,
            works_below_100m=True,
        ),
        Environment(
            "calibrated_abstention",
            "Ternary correct/abstain/wrong on short-answer QA.",
            reward_calibrated_answer,
            works_below_100m=True,
        ),
    )
}


def environments_for_scale(params: int) -> List[Environment]:
    """Environments worth running at a given parameter count."""
    if params < 100_000_000:
        return [e for e in ENVIRONMENTS.values() if e.works_below_100m]
    return list(ENVIRONMENTS.values())
