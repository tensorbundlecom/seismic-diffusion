"""Tests for checkpoint-bound reconstruction/cache identities."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ML.diffusion.reconstruction import (
    ReconstructionSpec,
    diffusion_cache_tag,
    scipy_spectrum_to_librosa_magnitude,
)


class STFTAmplitudeContractTests(unittest.TestCase):
    def test_hann_window_restores_scipy_spectrum_scaling(self):
        magnitude = np.ones((33, 701), dtype=np.float32)

        converted = scipy_spectrum_to_librosa_magnitude(
            magnitude,
            window="hann",
            win_length=50,
        )

        np.testing.assert_allclose(converted, 25.0)
        np.testing.assert_allclose(magnitude, 1.0)


class DiffusionCacheTagTests(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        checkpoint = root / "unet2d"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"architecture":"unet"}')
        (checkpoint / "scheduler_config.json").write_text('{"beta_end":0.02}')
        (checkpoint / "station_embedding.pt").write_bytes(b"station-v1")
        (checkpoint / "diffusion_pytorch_model.safetensors").write_bytes(b"weights-v1")
        return checkpoint

    @staticmethod
    def _reconstruction() -> ReconstructionSpec:
        return ReconstructionSpec(mode="per_event", ae_checkpoint="autoencoder.pt")

    def test_weight_bytes_change_the_cache_tag_for_a_reused_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._checkpoint(Path(directory))
            config = {"data_mode": "latent", "training_type": "ddpm"}
            before = diffusion_cache_tag(checkpoint, config, self._reconstruction())

            # Same output directory and same training config, new learned model.
            (checkpoint / "diffusion_pytorch_model.safetensors").write_bytes(b"weights-v2")
            after = diffusion_cache_tag(checkpoint, config, self._reconstruction())

            self.assertNotEqual(before, after)

    def test_irrelevant_checkpoint_sidecars_do_not_change_the_cache_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._checkpoint(Path(directory))
            config = {"data_mode": "latent", "training_type": "ddpm"}
            before = diffusion_cache_tag(checkpoint, config, self._reconstruction())

            # Training logs/provenance must not force expensive waveform
            # resampling when the actual sampling artifacts are unchanged.
            (checkpoint / "training.log").write_text("finished")
            (checkpoint / "notes.txt").write_text("human annotation")
            after = diffusion_cache_tag(checkpoint, config, self._reconstruction())

            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
