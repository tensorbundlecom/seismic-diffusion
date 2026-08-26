"""Focused validation tests for W&B-sweepable diffusion CLI parameters."""

import argparse
import unittest

from ML.diffusion.sweep_hyperparameters import (
    groupnorm_base_channels,
    nonnegative_finite_float,
    positive_finite_float,
    positive_int,
)


class SweepHyperparameterValidationTests(unittest.TestCase):
    def test_valid_values_preserve_current_defaults(self):
        self.assertEqual(positive_int("32"), 32)
        self.assertEqual(positive_finite_float("1e-4"), 1e-4)
        self.assertEqual(nonnegative_finite_float("1e-2"), 1e-2)
        self.assertEqual(groupnorm_base_channels("64"), 64)

    def test_learning_rate_must_be_positive_and_finite(self):
        for value in ("0", "-0.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_finite_float(value)

    def test_weight_decay_must_be_nonnegative_and_finite(self):
        for value in ("-0.1", "nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                nonnegative_finite_float(value)

    def test_batch_size_and_depth_must_be_positive_integers(self):
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_int(value)

    def test_base_width_must_be_groupnorm_compatible(self):
        self.assertEqual(groupnorm_base_channels("32"), 32)
        for value in ("0", "48", "65"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                groupnorm_base_channels(value)


if __name__ == "__main__":
    unittest.main()
