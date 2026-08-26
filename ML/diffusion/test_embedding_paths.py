"""Tests for checkpoint-namespaced diffusion embedding exports."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ML.diffusion.embedding_paths import (
    embedding_artifact_paths,
    named_embedding_dir,
    project_relative_path,
    resolve_project_path,
)


class EmbeddingPathTests(unittest.TestCase):
    def test_project_checkpoint_path_round_trips_as_project_relative(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "checkout"
            checkpoint = root / "ML" / "autoencoder" / "checkpoints" / "physical" / "best_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"ae")

            recorded = project_relative_path(checkpoint, root=root)

            self.assertEqual(recorded, "ML/autoencoder/checkpoints/physical/best_model.pt")
            self.assertEqual(
                resolve_project_path(recorded, root=root, require_exists=True),
                checkpoint.resolve(),
            )

    def test_legacy_absolute_checkpoint_relocates_to_another_checkout(self):
        with TemporaryDirectory() as tmp_dir:
            new_root = Path(tmp_dir) / "new-checkout"
            checkpoint = (
                new_root / "ML" / "autoencoder" / "checkpoints" /
                "vae-global-v1" / "best_model.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"ae")
            old_machine_path = (
                "/mnt/data/seismic-diffusion/ML/autoencoder/checkpoints/"
                "vae-global-v1/best_model.pt"
            )

            self.assertEqual(
                resolve_project_path(old_machine_path, root=new_root, require_exists=True),
                checkpoint.resolve(),
            )

    def test_legacy_checkpoint_can_use_recorded_ae_name_as_a_safe_fallback(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "new-checkout"
            checkpoint = (
                root / "ML" / "autoencoder" / "checkpoints" /
                "vae-global-v1" / "best_model.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"ae")

            self.assertEqual(
                resolve_project_path(
                    "/old/location/best_model.pt",
                    root=root,
                    ae_name="vae-global-v1",
                    require_exists=True,
                ),
                checkpoint.resolve(),
            )

    def test_missing_checkpoint_error_explains_portable_layout(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "project-relative checkpoint path"):
                resolve_project_path(
                    "/old/checkout/ML/autoencoder/checkpoints/missing/best_model.pt",
                    root=Path(tmp_dir),
                    require_exists=True,
                )

    def test_export_directory_is_namespaced_by_autoencoder_name(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ae_checkpoint = root / "autoencoder" / "checkpoints" / "physical_20260824" / "best_model.pt"

            result = named_embedding_dir(root / "embeddings", ae_checkpoint)

            self.assertEqual(result, root / "embeddings" / "physical_20260824")

    def test_named_directory_artifacts_carry_ae_and_domain_provenance(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ae_checkpoint = root / "autoencoder" / "checkpoints" / "physical_20260824" / "best_model.pt"
            embedding_dir = named_embedding_dir(root / "embeddings", ae_checkpoint)
            paths = embedding_artifact_paths(embedding_dir)
            embedding_dir.mkdir(parents=True)
            paths["embeddings"].write_bytes(b"latent-tensor")
            paths["metadata"].write_text("[]", encoding="utf-8")
            paths["source"].write_text(
                json.dumps(
                    {
                        "ae_checkpoint": str(ae_checkpoint),
                        "ae_name": "physical_20260824",
                        "waveform_domain": "physical_acceleration",
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(paths["embeddings"], embedding_dir / "embeddings.pt")
            self.assertEqual(paths["metadata"], embedding_dir / "metadata.json")
            source = json.loads(paths["source"].read_text(encoding="utf-8"))
            self.assertEqual(source["ae_name"], ae_checkpoint.parent.name)
            self.assertEqual(source["waveform_domain"], "physical_acceleration")

    def test_explicit_legacy_flat_embedding_directory_is_still_usable(self):
        with TemporaryDirectory() as tmp_dir:
            legacy_dir = Path(tmp_dir) / "embeddings"
            paths = embedding_artifact_paths(legacy_dir)
            legacy_dir.mkdir()
            paths["embeddings"].write_bytes(b"legacy-latents")
            paths["metadata"].write_text("[]", encoding="utf-8")
            paths["source"].write_text("{}", encoding="utf-8")

            self.assertEqual(paths["embeddings"], legacy_dir / "embeddings.pt")
            self.assertTrue(all(path.exists() for path in paths.values()))

    def test_preview_decoder_selects_and_caches_the_recorded_ae_checkpoint(self):
        from ML.diffusion import utils

        class FakeModel:
            def eval(self):
                return self

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "physical" / "best_model.pt"
            second = root / "counts" / "best_model.pt"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            utils._ae_models.clear()
            loaded = []

            def fake_load(path, device):
                loaded.append(Path(path))
                return FakeModel(), {}

            with patch.object(utils, "load_model", side_effect=fake_load):
                first_model = utils._get_ae_model(first)
                self.assertIs(first_model, utils._get_ae_model(first))
                second_model = utils._get_ae_model(second)

            self.assertIsNot(first_model, second_model)
            self.assertEqual(loaded, [first.resolve(), second.resolve()])

    def test_eval_never_falls_back_to_parent_artifacts(self):
        from eval.embedding_artifacts import artifact_path

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "embeddings"
            selected = root / "physical"
            selected.mkdir(parents=True)
            (root / "source.json").write_text('{"legacy": true}', encoding="utf-8")
            (root / "station_locations.json").write_text("{}", encoding="utf-8")

            self.assertEqual(artifact_path(selected, "source.json"), selected / "source.json")
            self.assertEqual(
                artifact_path(selected, "station_locations.json"),
                selected / "station_locations.json",
            )

    def test_evaluation_fraction_is_deterministic_and_approximately_one_tenth(self):
        from eval.embedding_artifacts import deterministic_fraction_subset

        values = list(range(103))
        first = deterministic_fraction_subset(values, fraction=0.1)
        second = deterministic_fraction_subset(values, fraction=0.1)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 11)
        self.assertEqual((first[0], first[-1]), (0, 102))

    def test_evaluation_fraction_keeps_one_record_for_small_sets(self):
        from eval.embedding_artifacts import deterministic_fraction_subset

        self.assertEqual(deterministic_fraction_subset([10, 20, 30], 0.1), [20])
        with self.assertRaises(ValueError):
            deterministic_fraction_subset([1, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
