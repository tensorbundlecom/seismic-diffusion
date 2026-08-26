"""Regression tests for waveform-domain PhaseNet input contracts."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalization_contract import resolve_amplitude_epsilon
from perceptual import (
    PHASENET_LEGACY_COUNTS_MODE,
    PHASENET_PHYSICAL_INVERSE_MODE,
    resolve_phasenet_input_mode,
    spectrogram_to_linear_magnitude,
)


class PhaseNetContractTest(unittest.TestCase):
    def test_count_domain_reproduces_legacy_expm1_even_with_global_bounds(self):
        spectrogram = torch.tensor([[[[-2.0, 0.0, 0.25, 1.0]]]])
        actual = spectrogram_to_linear_magnitude(
            spectrogram,
            input_mode=resolve_phasenet_input_mode(
                "instrument_counts", global_min=0.0, global_max=15.3
            ),
            global_min=0.0,
            global_max=15.3,
            amplitude_epsilon=1.0,
        )
        expected = torch.expm1(spectrogram.clamp_min(0.0))
        torch.testing.assert_close(actual, expected)

    def test_physical_domain_uses_exact_global_inverse(self):
        global_min = math.log(2.0)
        global_max = math.log(10.0)
        normalized = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        actual = spectrogram_to_linear_magnitude(
            normalized,
            input_mode=resolve_phasenet_input_mode(
                "physical_acceleration", global_min, global_max
            ),
            global_min=global_min,
            global_max=global_max,
            amplitude_epsilon=1.0,
        )
        expected = torch.exp(normalized * (global_max - global_min) + global_min) - 1.0
        torch.testing.assert_close(actual, expected)

    def test_physical_domain_requires_global_bounds(self):
        with self.assertRaisesRegex(ValueError, "requires exact global-normalization bounds"):
            resolve_phasenet_input_mode("physical_acceleration")

    def test_domain_aware_amplitude_defaults_preserve_count_legacy_contract(self):
        self.assertEqual(resolve_amplitude_epsilon("instrument_counts", None), 1.0)
        self.assertEqual(resolve_amplitude_epsilon("physical_acceleration", None), 1e-12)
        self.assertEqual(resolve_amplitude_epsilon("instrument_counts", 0.25), 0.25)

    def test_mode_names_are_explicit(self):
        self.assertEqual(
            resolve_phasenet_input_mode("instrument_counts"),
            PHASENET_LEGACY_COUNTS_MODE,
        )
        self.assertEqual(
            resolve_phasenet_input_mode("physical_acceleration", 0.0, 1.0),
            PHASENET_PHYSICAL_INVERSE_MODE,
        )


if __name__ == "__main__":
    unittest.main()
