"""Tests for waveform-domain provenance and physical-unit conversions."""

import unittest

import numpy as np

from ML.diffusion.reconstruction import ReconstructionSpec, resolve_reconstruction_spec
from ML.diffusion.waveform_domain import (
    INSTRUMENT_COUNTS,
    PHYSICAL_ACCELERATION,
    physical_acceleration_to_motion,
    resolve_waveform_domain,
)


class WaveformDomainTests(unittest.TestCase):
    def test_legacy_provenance_defaults_to_instrument_counts(self):
        self.assertEqual(resolve_waveform_domain(None), INSTRUMENT_COUNTS)

    def test_explicit_override_wins(self):
        self.assertEqual(
            resolve_waveform_domain(INSTRUMENT_COUNTS, PHYSICAL_ACCELERATION),
            PHYSICAL_ACCELERATION,
        )

    def test_physical_acceleration_is_not_response_corrected_again(self):
        acc = np.asarray([0.1, -0.2, 0.3])
        converted = physical_acceleration_to_motion(acc, "ACC", 100.0)
        np.testing.assert_array_equal(converted, acc)
        self.assertIsNot(converted, acc)

    def test_frequency_domain_velocity_integration(self):
        fs = 100.0
        seconds = 4.0
        frequency = 2.0
        t = np.arange(int(fs * seconds)) / fs
        acc = np.cos(2.0 * np.pi * frequency * t)
        expected_velocity = np.sin(2.0 * np.pi * frequency * t) / (
            2.0 * np.pi * frequency
        )
        velocity = physical_acceleration_to_motion(acc, "VEL", fs)
        np.testing.assert_allclose(velocity, expected_velocity, atol=1e-12)

    def test_reconstruction_provenance_and_cache_identity_include_domain(self):
        config = {
            "embedding_provenance": {
                "normalization_mode": "global",
                "global_min": -10.0,
                "global_max": 1.0,
                "amplitude_epsilon": 1e-12,
                "waveform_domain": PHYSICAL_ACCELERATION,
            }
        }
        spec = resolve_reconstruction_spec(config)
        self.assertEqual(spec.waveform_domain, PHYSICAL_ACCELERATION)
        overridden = resolve_reconstruction_spec(
            config, waveform_domain_override=INSTRUMENT_COUNTS
        )
        self.assertEqual(overridden.waveform_domain, INSTRUMENT_COUNTS)

        counts = ReconstructionSpec(
            mode="global", ae_checkpoint=None, waveform_domain=INSTRUMENT_COUNTS
        )
        physical = ReconstructionSpec(
            mode="global", ae_checkpoint=None, waveform_domain=PHYSICAL_ACCELERATION
        )
        self.assertNotEqual(counts.cache_tag, physical.cache_tag)


if __name__ == "__main__":
    unittest.main()
