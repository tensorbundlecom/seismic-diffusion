"""Focused tests for the final W&B evaluation orchestration."""

from pathlib import Path
import io
import os
import tempfile
import unittest
from unittest.mock import patch

from ML.diffusion.evaluation import (
    EvaluationJob,
    EVAL_OUTPUT_ROOT_ENV,
    _run_job,
    _paths_reported_by_evaluator,
    build_evaluation_commands,
    build_first_order_command,
    changed_pngs,
    evaluation_run_output_root,
    run_final_evaluation,
    snapshot_pngs,
)


class _FakeRun:
    def __init__(self):
        self.logs = []

    def log(self, values):
        self.logs.append(values)


class _StreamingProcess:
    """Minimal Popen stand-in whose output can be consumed incrementally."""

    def __init__(self, output: str, returncode: int = 0):
        self.stdout = io.StringIO(output)
        self.returncode = returncode

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


class _FakeProgress:
    def __init__(self):
        self.descriptions = []
        self.updates = []
        self.closed = False

    def set_description(self, description):
        self.descriptions.append(description)

    def update(self, amount=1):
        self.updates.append(amount)

    def close(self):
        self.closed = True


class EvaluationOrchestrationTests(unittest.TestCase):
    def test_wandb_run_output_roots_are_distinct_and_stay_under_eval(self):
        """Concurrent W&B runs must never share evaluator files or caches."""
        root = Path("/project")
        epsilon_root = evaluation_run_output_root(
            root, "unet2d", "prediction-target-epsilon", "run-epsilon"
        )
        x0_root = evaluation_run_output_root(
            root, "unet2d", "prediction-target-x0", "run-x0"
        )

        self.assertNotEqual(epsilon_root, x0_root)
        for output_root, run_id in (
            (epsilon_root, "run-epsilon"),
            (x0_root, "run-x0"),
        ):
            self.assertTrue(output_root.is_relative_to(root / "eval"))
            self.assertIn("runs", output_root.parts)
            self.assertEqual(output_root.name, run_id)
            self.assertEqual(output_root.parent.parent.name, "unet2d")

    def test_output_root_is_passed_to_evaluator_subprocess(self):
        """The child process receives its run-specific root, not a shared cwd."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "eval" / "runs" / "unet2d" / "demo" / "run-1"
            run = _FakeRun()
            process = _StreamingProcess("")
            with patch("ML.diffusion.evaluation.subprocess.Popen", return_value=process) as popen:
                _run_job(
                    EvaluationJob("scaling", ("python", "scaling.py")),
                    root,
                    run,
                    lambda path, **_: path,
                    output_root=output_root,
                )

            child_env = popen.call_args.kwargs["env"]
            self.assertEqual(child_env[EVAL_OUTPUT_ROOT_ENV], str(output_root.resolve()))

    def test_legacy_standalone_output_root_defaults_to_eval(self):
        """Without training's environment override, evaluator CLIs keep eval/*."""
        from eval.output_paths import evaluation_output_dir, evaluation_output_root

        project_root = Path(__file__).resolve().parents[2]
        with patch.dict(os.environ, {EVAL_OUTPUT_ROOT_ENV: ""}, clear=False):
            self.assertEqual(evaluation_output_root(), project_root / "eval")
            self.assertEqual(
                evaluation_output_dir("first_order"), project_root / "eval" / "first_order"
            )
            self.assertEqual(
                evaluation_output_dir("residual_distributions", "peak_amplitudes"),
                project_root / "eval" / "peak_amplitudes",
            )
        scoped = project_root / "eval" / "runs" / "ddpm" / "demo" / "run-1"
        with patch.dict(os.environ, {EVAL_OUTPUT_ROOT_ENV: str(scoped)}, clear=False):
            self.assertEqual(
                evaluation_output_dir("residual_distributions", "peak_amplitudes"),
                scoped / "residual_distributions",
            )

    def test_run_scoped_peak_and_residual_caches_never_escape_output_root(self):
        """Dependent evaluators use this run's first-order and peak artifacts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "eval" / "runs" / "unet2d" / "demo" / "run-1"
            first_cache = output_root / "first_order" / "cache_demo.npz"
            expected_peaks = output_root / "peak_amplitudes" / "peaks_demo.npz"
            seen = []

            def record_job(job, *_args, **_kwargs):
                seen.append(job)
                if job.name == "first_order":
                    first_cache.parent.mkdir(parents=True)
                    first_cache.write_bytes(b"cache")
                    return True, f"[eval] cache written: {first_cache}\n"
                if job.name == "peak_amplitudes":
                    expected_peaks.parent.mkdir(parents=True)
                    expected_peaks.write_bytes(b"cache")
                return True, ""

            with patch("ML.diffusion.evaluation._run_job", side_effect=record_job):
                run_final_evaluation(
                    root,
                    root / "checkpoint",
                    _FakeRun(),
                    lambda path, **_: path,
                    "python-test",
                    output_root=output_root,
                )

            commands = {job.name: job.command for job in seen}
            self.assertTrue(first_cache.is_relative_to(output_root))
            self.assertTrue(expected_peaks.is_relative_to(output_root))
            self.assertIn(str(first_cache), commands["peak_amplitudes"])
            self.assertIn(str(first_cache), commands["distributions"])
            self.assertIn(str(first_cache), commands["shake_duration"])
            self.assertIn(str(expected_peaks), commands["residual_distributions"])

    def test_commands_bind_all_dependents_to_the_final_checkpoint_and_caches(self):
        root = Path("/project")
        checkpoint = root / "ML/diffusion/checkpoints/ddpm/demo/unet2d"
        first = root / "eval/first_order/cache_demo.npz"
        peaks = root / "eval/peak_amplitudes/peaks_demo.npz"

        jobs = build_evaluation_commands(root, checkpoint, first, peaks, "python-test")
        commands = {job.name: job.command for job in jobs}

        self.assertEqual(commands["peak_amplitudes"][0], "python-test")
        self.assertIn(str(first), commands["peak_amplitudes"])
        self.assertIn("edwardsfah13", commands["peak_amplitudes"])
        self.assertIn(str(peaks), commands["residual_distributions"])
        for name in ("distributions", "shake_duration", "attenuation", "scaling", "model_probabilities"):
            self.assertIn(str(checkpoint), commands[name])
        for name in ("distributions", "shake_duration"):
            self.assertIn(str(first), commands[name])

    def test_commands_propagate_the_selected_embedding_export(self):
        root = Path("/project")
        checkpoint = root / "ML/diffusion/checkpoints/ddpm/demo/unet2d"
        embeddings = root / "ML/diffusion/embeddings/ae-physical"
        jobs = build_evaluation_commands(
            root, checkpoint, root / "first.npz", root / "peaks.npz",
            "python-test", embeddings,
        )
        commands = {job.name: job.command for job in jobs}
        self.assertIn("--embeddings_dir", build_first_order_command(
            root, checkpoint, "python-test", embeddings
        ).command)
        for name in (
            "peak_amplitudes", "distributions", "shake_duration", "attenuation",
            "scaling", "model_probabilities",
        ):
            self.assertIn("--embeddings_dir", commands[name])
            self.assertIn(str(embeddings), commands[name])
        self.assertNotIn("--embeddings_dir", commands["residual_distributions"])

    def test_automatic_evaluation_uses_one_deterministic_tenth_once(self):
        """Only source-data evaluators reduce records; cache consumers reuse them."""
        root = Path("/project")
        checkpoint = root / "ML/diffusion/checkpoints/ddpm/demo/unet2d"
        first = build_first_order_command(root, checkpoint, "python-test").command
        jobs = build_evaluation_commands(
            root,
            checkpoint,
            root / "eval/first_order/cache_demo.npz",
            root / "eval/peak_amplitudes/peaks_demo.npz",
            "python-test",
        )
        commands = {job.name: job.command for job in jobs}

        self.assertEqual(first[first.index("--fraction") + 1], "0.1")
        self.assertEqual(first[first.index("--seed") + 1], "0")
        for name in (
            "peak_amplitudes",
            "residual_distributions",
            "distributions",
            "shake_duration",
        ):
            self.assertNotIn("--fraction", commands[name])
            self.assertNotIn("--seed", commands[name])

    def test_independent_evaluators_receive_the_same_deterministic_fraction(self):
        """The scenario evaluators sample their observed records consistently."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.resolve()
            run = _FakeRun()
            commands = []

            def record_job(job, *_args):
                commands.append(job.command)
                return False, "first-order failed"

            with patch("ML.diffusion.evaluation._run_job", side_effect=record_job):
                run_final_evaluation(
                    root, root / "checkpoint", run, lambda path, **_: path, "python-test"
                )

            self.assertEqual(len(commands), 4)
            self.assertEqual(commands[0][2], "all")
            self.assertEqual(commands[0][commands[0].index("--fraction") + 1], "0.1")
            self.assertEqual(commands[0][commands[0].index("--seed") + 1], "0")
            for command in commands[1:]:
                self.assertEqual(command[command.index("--fraction") + 1], "0.1")
                self.assertEqual(command[command.index("--seed") + 1], "0")

    def test_suite_progress_accounts_for_all_eight_tasks(self):
        """The top-level bar advances for executed and skipped evaluator tasks."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _FakeRun()
            progress = _FakeProgress()
            with patch("ML.diffusion.evaluation.tqdm", return_value=progress) as make_bar, patch(
                "ML.diffusion.evaluation._run_job",
                side_effect=[(False, "first failure"), (True, ""), (True, ""), (True, "")],
            ):
                run_final_evaluation(
                    root, root / "checkpoint", run, lambda path, **_: path, "python-test"
                )

            self.assertEqual(make_bar.call_args.kwargs["total"], 8)
            self.assertEqual(sum(progress.updates), 8)
            self.assertTrue(progress.closed)
            self.assertEqual(
                progress.descriptions,
                [
                    "Final evaluation: first_order",
                    "Final evaluation: attenuation",
                    "Final evaluation: scaling",
                    "Final evaluation: model_probabilities",
                ],
            )

    def test_only_new_or_modified_pngs_are_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            eval_root = Path(directory) / "eval"
            eval_root.mkdir()
            old = eval_root / "old.png"
            old.write_bytes(b"old")
            before = snapshot_pngs(eval_root)
            fresh = eval_root / "fresh.png"
            fresh.write_bytes(b"fresh")

            self.assertEqual(changed_pngs(eval_root, before), [fresh.resolve()])

    def test_reported_figure_paths_must_be_inside_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure = root / "eval" / "scaling" / "figure.png"
            figure.parent.mkdir(parents=True)
            figure.write_bytes(b"png")
            outside = root / "outside.png"
            outside.write_bytes(b"png")
            output = (
                f"[eval] figure saved: {figure}\n"
                f"[eval] figure saved: {outside}\n"
            )

            self.assertEqual(_paths_reported_by_evaluator(output, root), [figure.resolve()])

    def test_image_logging_uses_only_paths_reported_by_the_current_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reported = root / "eval" / "scaling" / "reported.png"
            stale = root / "eval" / "scaling" / "stale.png"
            reported.parent.mkdir(parents=True)
            reported.write_bytes(b"png")
            stale.write_bytes(b"png")
            run = _FakeRun()
            output = f"[eval] figure saved: {reported}\n"
            completed = _StreamingProcess(output)
            with patch("ML.diffusion.evaluation.subprocess.Popen", return_value=completed):
                _run_job(
                    EvaluationJob("scaling", ("python", "scaling.py")), root, run,
                    lambda path, **_: path,
                )

            image_log = next(log for log in run.logs if "Evaluation/scaling/images" in log)
            self.assertEqual(image_log["Evaluation/scaling/images"], [str(reported.resolve())])

    def test_job_streams_subprocess_output_and_preserves_the_full_transcript(self):
        """Progress from an evaluator must remain visible before W&B parsing it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = "phase one\\nphase two\\n"
            process = _StreamingProcess(output)
            run = _FakeRun()
            with patch("ML.diffusion.evaluation.subprocess.Popen", return_value=process), patch(
                "builtins.print"
            ) as printed:
                ok, transcript = _run_job(
                    EvaluationJob("scaling", ("python", "scaling.py")), root, run,
                    lambda path, **_: path,
                )

            self.assertTrue(ok)
            self.assertEqual(transcript, output)
            emitted = "".join(
                str(call.args[0]) for call in printed.call_args_list if call.args
            )
            self.assertIn("phase one", emitted)
            self.assertIn("phase two", emitted)

    def test_failed_first_order_skips_dependents_but_runs_independent_evaluators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _FakeRun()
            # first_order fails; attenuation, scaling, and model_probabilities
            # still get a chance to provide independent diagnostics.
            with patch(
                "ML.diffusion.evaluation._run_job",
                side_effect=[(False, "first failure"), (True, ""), (True, ""), (True, "")],
            ) as run_job:
                status = run_final_evaluation(
                    root, root / "checkpoint", run, lambda path, **_: path, "python-test"
                )

            self.assertEqual(status, "partial")
            self.assertEqual(
                [call.args[0].name for call in run_job.call_args_list],
                ["first_order", "attenuation", "scaling", "model_probabilities"],
            )
            skipped = [
                next(iter(log)) for log in run.logs
                if next(iter(log)).endswith("/status") and next(iter(log)).startswith("Evaluation/")
                and log[next(iter(log))] == "skipped"
            ]
            self.assertEqual(len(skipped), 4)
            self.assertEqual(run.logs[-1]["Evaluation/status"], "partial")


if __name__ == "__main__":
    unittest.main()
