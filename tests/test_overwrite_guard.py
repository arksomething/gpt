"""The guard that stops a retrain from destroying a finished model."""

import os
import sys
import tempfile
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import _guard_completed_run  # noqa: E402


def _args(**kw):
    base = dict(
        overwrite=False, resume_from=None, resume_from_slot=None, resume_from_hf=None
    )
    base.update(kw)
    return Namespace(**base)


class GuardCompletedRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _complete(self):
        os.makedirs(os.path.join(self.tmp, "final"), exist_ok=True)

    def test_empty_dir_is_allowed(self):
        _guard_completed_run(self.tmp, _args())

    def test_missing_dir_is_allowed(self):
        _guard_completed_run(os.path.join(self.tmp, "nope"), _args())

    def test_in_progress_run_without_final_is_allowed(self):
        os.makedirs(os.path.join(self.tmp, "step_0000100"), exist_ok=True)
        _guard_completed_run(self.tmp, _args())

    def test_completed_run_is_refused(self):
        self._complete()
        with self.assertRaises(SystemExit) as ctx:
            _guard_completed_run(self.tmp, _args())
        self.assertIn("already holds a completed run", str(ctx.exception))

    def test_explicit_overwrite_is_allowed(self):
        self._complete()
        _guard_completed_run(self.tmp, _args(overwrite=True))

    def test_resume_paths_are_allowed(self):
        self._complete()
        for field in ("resume_from", "resume_from_slot", "resume_from_hf"):
            with self.subTest(field=field):
                _guard_completed_run(self.tmp, _args(**{field: "something"}))


if __name__ == "__main__":
    unittest.main()
