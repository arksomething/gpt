"""Reward functions and the dead-group detector."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.rl_rewards import (  # noqa: E402
    ENVIRONMENTS,
    environments_for_scale,
    group_advantages,
    is_abstention,
    reward_calibrated_answer,
    reward_length,
    reward_no_repetition,
    reward_sentence_count,
    reward_termination,
)


class GroupAdvantageTests(unittest.TestCase):
    def test_spread_group_is_usable_and_centred(self):
        stats = group_advantages([0.0, 1.0, 0.5, 0.25])
        self.assertTrue(stats.usable)
        self.assertAlmostEqual(sum(stats.advantages), 0.0, places=9)
        self.assertGreater(stats.std, 0.0)

    def test_all_zero_group_is_degenerate(self):
        """The 25M math case: nothing solved, so nothing to learn from."""
        stats = group_advantages([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(stats.degenerate)
        self.assertEqual(set(stats.advantages), {0.0})

    def test_all_correct_group_is_also_degenerate(self):
        """A constraint too easy to fail teaches exactly as little."""
        stats = group_advantages([1.0, 1.0, 1.0])
        self.assertTrue(stats.degenerate)

    def test_mean_reward_cannot_distinguish_dead_from_healthy(self):
        # Same mean, opposite learning value -- which is why `degenerate` is
        # reported separately rather than inferred from the reward curve.
        dead = group_advantages([0.5, 0.5, 0.5, 0.5])
        live = group_advantages([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(dead.mean, live.mean)
        self.assertTrue(dead.degenerate)
        self.assertTrue(live.usable)

    def test_empty_group(self):
        self.assertTrue(group_advantages([]).degenerate)


class MechanicalRewardTests(unittest.TestCase):
    def test_termination(self):
        self.assertEqual(reward_termination(True), 1.0)
        self.assertEqual(reward_termination(False), 0.0)

    def test_repetition_penalises_loops(self):
        clean = "the cat sat on a warm mat beside the quiet window today"
        looped = " ".join(["hello there friend now"] * 8)
        self.assertGreater(reward_no_repetition(clean), reward_no_repetition(looped))
        self.assertLess(reward_no_repetition(looped), 0.5)

    def test_repetition_short_text_is_neutral(self):
        self.assertEqual(reward_no_repetition("hi there"), 1.0)

    def test_length_peaks_at_target(self):
        exact = " ".join(["w"] * 10)
        near = " ".join(["w"] * 12)
        far = " ".join(["w"] * 40)
        self.assertEqual(reward_length(exact, 10), 1.0)
        self.assertGreater(reward_length(near, 10), reward_length(far, 10))
        self.assertEqual(reward_length(far, 10), 0.0)

    def test_length_rejects_bad_target(self):
        with self.assertRaises(ValueError):
            reward_length("x", 0)

    def test_sentence_count(self):
        self.assertEqual(reward_sentence_count("One. Two.", 2), 1.0)
        self.assertEqual(reward_sentence_count("One. Two.", 3), 0.0)


class CalibratedAbstentionTests(unittest.TestCase):
    def test_correct_answer_scores_highest(self):
        self.assertEqual(reward_calibrated_answer("It is Paris.", ["Paris"]), 1.0)

    def test_abstention_beats_being_wrong(self):
        abstain = reward_calibrated_answer("I don't know.", ["Paris"])
        wrong = reward_calibrated_answer("It is Berlin.", ["Paris"])
        self.assertGreater(abstain, wrong)

    def test_ordering_is_correct_gt_abstain_gt_wrong(self):
        correct = reward_calibrated_answer("Paris", ["Paris"])
        abstain = reward_calibrated_answer("I'm not sure", ["Paris"])
        wrong = reward_calibrated_answer("Berlin", ["Paris"])
        self.assertGreater(correct, abstain)
        self.assertGreater(abstain, wrong)

    def test_guessing_is_negative_expected_value_when_unlikely(self):
        """The property that makes abstention rational rather than merely allowed.

        With correct=+1, abstain=0, wrong=-1, guessing only pays when the model
        is right more than half the time. A binary reward (wrong=0) would make
        guessing free, which is what trains confident hallucination.
        """
        p_correct = 0.25
        ternary_ev = p_correct * 1.0 + (1 - p_correct) * (-1.0)
        self.assertLess(ternary_ev, 0.0)  # abstaining (0.0) is better
        binary_ev = p_correct * 1.0 + (1 - p_correct) * 0.0
        self.assertGreater(binary_ev, 0.0)  # binary would reward the guess

    def test_abstention_detection(self):
        self.assertTrue(is_abstention("I do not know the answer"))
        self.assertFalse(is_abstention("The answer is clearly Paris"))

    def test_requires_gold(self):
        with self.assertRaises(ValueError):
            reward_calibrated_answer("x", [])


class RegistryTests(unittest.TestCase):
    def test_small_scale_excludes_judge_environments(self):
        for env in environments_for_scale(25_000_000):
            self.assertFalse(env.needs_external_judge)
            self.assertTrue(env.works_below_100m)

    def test_registry_names_match_keys(self):
        for name, env in ENVIRONMENTS.items():
            self.assertEqual(name, env.name)


if __name__ == "__main__":
    unittest.main()
