"""Focused tests for best-validation checkpoint selection."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ML.diffusion.best_validation import (
    best_validation_checkpoint_path,
    is_strictly_better_validation_loss,
    replace_checkpoint_directory,
    run_checkpoint_path,
    select_evaluation_checkpoint,
)


class BestValidationSelectionTests(unittest.TestCase):
    def test_first_finite_loss_is_best(self):
        self.assertTrue(is_strictly_better_validation_loss(0.5, None))

    def test_only_strict_improvements_replace_the_checkpoint(self):
        self.assertTrue(is_strictly_better_validation_loss(0.4, 0.5))
        self.assertFalse(is_strictly_better_validation_loss(0.5, 0.5))
        self.assertFalse(is_strictly_better_validation_loss(0.6, 0.5))

    def test_non_finite_candidate_never_replaces_best(self):
        self.assertFalse(is_strictly_better_validation_loss(math.nan, None))
        self.assertFalse(is_strictly_better_validation_loss(math.inf, 0.5))
        self.assertFalse(is_strictly_better_validation_loss(-math.inf, 0.5))

    def test_finite_candidate_replaces_non_finite_recorded_value(self):
        self.assertTrue(is_strictly_better_validation_loss(0.5, math.nan))

    def test_wandb_best_checkpoint_is_run_isolated(self):
        root = Path("checkpoints") / "ddpm"
        self.assertEqual(
            best_validation_checkpoint_path(root, "abc123"),
            root / "wandb_runs" / "abc123" / "best_val",
        )
        self.assertEqual(best_validation_checkpoint_path(root), root / "best_val")
        self.assertEqual(
            run_checkpoint_path(root, "unet2d", "abc123"),
            root / "wandb_runs" / "abc123" / "unet2d",
        )

    def test_evaluation_ignores_stale_or_non_finite_best_checkpoint(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stale_best = root / "best_val"
            stale_best.mkdir()
            final = root / "unet2d"
            final.mkdir()

            self.assertEqual(select_evaluation_checkpoint(stale_best, None, final), final)
            self.assertEqual(select_evaluation_checkpoint(stale_best, math.nan, final), final)
            self.assertEqual(select_evaluation_checkpoint(stale_best, 0.5, final), stale_best)

    def test_staged_replacement_restores_previous_checkpoint_on_install_failure(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            destination = root / "best_val"
            destination.mkdir()
            (destination / "marker.txt").write_text("old", encoding="utf-8")
            staged = root / ".best_val.tmp"
            staged.mkdir()
            (staged / "marker.txt").write_text("new", encoding="utf-8")

            call_count = 0

            def fail_on_install(source: Path, target: Path) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("simulated staged-install failure")
                source.replace(target)

            with self.assertRaisesRegex(OSError, "simulated"):
                replace_checkpoint_directory(staged, destination, move=fail_on_install)

            self.assertEqual((destination / "marker.txt").read_text(encoding="utf-8"), "old")
            self.assertTrue(staged.is_dir())

    def test_staged_replacement_removes_old_checkpoint_after_success(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            destination = root / "best_val"
            destination.mkdir()
            (destination / "marker.txt").write_text("old", encoding="utf-8")
            staged = root / ".best_val.tmp"
            staged.mkdir()
            (staged / "marker.txt").write_text("new", encoding="utf-8")

            replace_checkpoint_directory(staged, destination)

            self.assertEqual((destination / "marker.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(staged.exists())
            self.assertFalse(list(root.glob(".best_val.backup-*")))


if __name__ == "__main__":
    unittest.main()
