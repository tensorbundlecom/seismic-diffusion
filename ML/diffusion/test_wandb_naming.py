"""Focused unit tests for W&B display-name construction."""

from argparse import Namespace
import unittest

from ML.diffusion.wandb_naming import build_wandb_run_name, parse_name_params


class WandbNamingTests(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            prediction_target="epsilon",
            include_station_id=True,
            optional_value=None,
            num_epochs=20,
        )

    def test_appends_requested_values_in_supplied_order(self):
        name = build_wandb_run_name(
            "prediction-target-sweep",
            " prediction_target, include_station_id ",
            self.args,
            "legacy-default",
        )

        self.assertEqual(
            name,
            "prediction-target-sweep-prediction_target=epsilon-include_station_id=true",
        )

    def test_none_and_boolean_values_are_stable(self):
        name = build_wandb_run_name(
            "run", "optional_value,include_station_id", self.args, "legacy-default"
        )

        self.assertEqual(name, "run-optional_value=none-include_station_id=true")

    def test_omitted_options_preserve_the_legacy_default(self):
        self.assertEqual(
            build_wandb_run_name(None, None, self.args, "diffusion_latent_ddpm"),
            "diffusion_latent_ddpm",
        )

    def test_empty_and_unknown_parameter_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_name_params("prediction_target,,num_epochs", self.args)
        with self.assertRaisesRegex(ValueError, "Unknown.*missing"):
            parse_name_params("prediction_target,missing", self.args)

    def test_duplicate_parameter_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_name_params("prediction_target,prediction_target", self.args)

    def test_empty_base_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wandb_run_name"):
            build_wandb_run_name("  ", None, self.args, "legacy-default")


if __name__ == "__main__":
    unittest.main()
