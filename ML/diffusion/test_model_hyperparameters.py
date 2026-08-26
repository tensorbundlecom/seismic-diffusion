"""Focused CPU-only checks for sweepable Diffusion U-Net architecture knobs."""

import unittest

from ML.diffusion.model import DiffusionUNet2D


class DiffusionUNetHyperparameterTests(unittest.TestCase):
    def _make_model(self, **overrides):
        config = {
            "in_channels": 16,
            "out_channels": 16,
            "num_stations": 4,
            "num_channels": 4,
        }
        config.update(overrides)
        return DiffusionUNet2D(**config)

    def test_defaults_preserve_current_architecture(self):
        model = self._make_model()

        self.assertEqual(tuple(model.model.config.block_out_channels), (64, 128, 256))
        self.assertEqual(model.model.config.layers_per_block, 2)
        self.assertEqual(model.base_channels, 64)
        self.assertEqual(model.layers_per_block, 2)

    def test_sweepable_width_and_depth_are_used(self):
        model = self._make_model(base_channels=32, layers_per_block=1)

        self.assertEqual(tuple(model.model.config.block_out_channels), (32, 64, 128))
        self.assertEqual(model.model.config.layers_per_block, 1)

    def test_groupnorm_incompatible_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            self._make_model(base_channels=48)

    def test_nonpositive_depth_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layers_per_block must be positive"):
            self._make_model(layers_per_block=0)


if __name__ == "__main__":
    unittest.main()
