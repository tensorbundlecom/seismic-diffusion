"""Run the end-of-training evaluation suite and collect its figures.

The evaluators are intentionally executed as separate processes.  They remain
usable from the command line, while training can pass its exact final
checkpoint and keep all W&B logging on the already-open training run.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class EvaluationJob:
    """One evaluator invocation in the end-of-training suite."""

    name: str
    command: Tuple[str, ...]


_CACHE_RE = re.compile(r"^\[eval\] cache written: (?P<path>.+\.npz)\s*$", re.MULTILINE)
_FIGURE_RE = re.compile(r"^\[eval\] figure saved: (?P<path>.+\.png)\s*$", re.MULTILINE)
FINAL_EVALUATION_FRACTION = 0.1
FINAL_EVALUATION_SEED = 0
EVAL_OUTPUT_ROOT_ENV = "SEISMIC_EVAL_OUTPUT_ROOT"
_SAFE_OUTPUT_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _output_slug(value: Optional[str], fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("._-")
    return (slug or fallback)[:120]


def evaluation_run_output_root(
    project_root: Path,
    training_type: str,
    experiment_name: Optional[str],
    wandb_run_id: str,
) -> Path:
    """Return a stable, collision-free output root for one tracked run."""
    experiment_slug = _output_slug(experiment_name, "default")
    components = (("training_type", training_type), ("wandb_run_id", wandb_run_id))
    for label, value in components:
        if not _SAFE_OUTPUT_COMPONENT.fullmatch(value):
            raise ValueError(f"Unsafe {label} for evaluation output path: {value!r}")
    return (
        project_root.resolve()
        / "eval"
        / "runs"
        / training_type
        / experiment_slug
        / wandb_run_id
    )


def build_evaluation_commands(
    project_root: Path,
    checkpoint: Path,
    first_order_cache: Path,
    peaks_cache: Path,
    python_executable: Optional[str] = None,
    embeddings_dir: Optional[Path] = None,
    fraction: float = FINAL_EVALUATION_FRACTION,
    seed: int = FINAL_EVALUATION_SEED,
) -> List[EvaluationJob]:
    """Build dependent evaluation commands for a checkpoint.

    ``first_order_cache`` and ``peaks_cache`` are explicit rather than
    "latest" lookups, preventing this run from accidentally consuming output
    created by a different checkpoint.
    """
    executable = python_executable or sys.executable
    eval_dir = project_root / "eval"
    checkpoint_arg = str(checkpoint)
    first_cache_arg = str(first_order_cache)
    peaks_cache_arg = str(peaks_cache)
    embeddings_args = (
        ("--embeddings_dir", str(embeddings_dir)) if embeddings_dir is not None else ()
    )
    sample_args = ("--fraction", f"{fraction:g}", "--seed", str(seed))

    def command(script: str, *args: str) -> Tuple[str, ...]:
        return (executable, str(eval_dir / script), *args)

    return [
        EvaluationJob(
            "peak_amplitudes",
            command(
                "evaluate_peak_amplitudes.py", "all", "--cache", first_cache_arg,
                "--gmm", "edwardsfah13", "--set_mode", "matched", *embeddings_args,
            ),
        ),
        EvaluationJob(
            "residual_distributions",
            command("evaluate_residual_distributions.py", "--peaks_cache", peaks_cache_arg,
                    "--gmm", "edwardsfah13"),
        ),
        EvaluationJob(
            "distributions",
            command(
                "evaluate_distributions.py", "all", "--cache", first_cache_arg,
                "--checkpoint", checkpoint_arg, *embeddings_args,
            ),
        ),
        EvaluationJob(
            "shake_duration",
            command(
                "evaluate_shake_duration.py", "all", "--cache", first_cache_arg,
                "--checkpoint", checkpoint_arg, *embeddings_args,
            ),
        ),
        EvaluationJob(
            "attenuation",
            command("evaluate_attenuation.py", "all", "--checkpoint", checkpoint_arg,
                    *sample_args, *embeddings_args),
        ),
        EvaluationJob(
            "scaling",
            command("evaluate_scaling.py", "all", "--checkpoint", checkpoint_arg,
                    *sample_args, *embeddings_args),
        ),
        EvaluationJob(
            "model_probabilities",
            command("evaluate_model_probabilities.py", "all", "--checkpoint", checkpoint_arg,
                    *sample_args, *embeddings_args),
        ),
    ]


def build_first_order_command(
    project_root: Path,
    checkpoint: Path,
    python_executable: Optional[str] = None,
    embeddings_dir: Optional[Path] = None,
    fraction: float = FINAL_EVALUATION_FRACTION,
    seed: int = FINAL_EVALUATION_SEED,
) -> EvaluationJob:
    executable = python_executable or sys.executable
    embeddings_args = (
        ("--embeddings_dir", str(embeddings_dir)) if embeddings_dir is not None else ()
    )
    return EvaluationJob(
        "first_order",
        (
            executable,
            str(project_root / "eval" / "evaluate_first_order.py"),
            "all",
            "--checkpoint",
            str(checkpoint),
            "--fraction",
            f"{fraction:g}",
            "--seed",
            str(seed),
            *embeddings_args,
        ),
    )


def snapshot_pngs(eval_root: Path) -> Dict[Path, Tuple[int, int]]:
    """Return file signatures for evaluator figures currently on disk."""
    if not eval_root.exists():
        return {}
    signatures: Dict[Path, Tuple[int, int]] = {}
    for path in eval_root.rglob("*.png"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signatures[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
    return signatures


def changed_pngs(
    eval_root: Path, before: Mapping[Path, Tuple[int, int]]
) -> List[Path]:
    """Return only new or modified evaluator figures, never stale PNGs."""
    after = snapshot_pngs(eval_root)
    return sorted(path for path, signature in after.items() if before.get(path) != signature)


def _paths_reported_by_evaluator(output: str, root: Path) -> List[Path]:
    """Find figures reported by an evaluator and reject paths outside ``eval``."""
    eval_root = (root / "eval").resolve()
    paths: List[Path] = []
    for match in _FIGURE_RE.finditer(output):
        path = Path(match.group("path")).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        try:
            path.relative_to(eval_root)
        except ValueError:
            continue
        if path.suffix.lower() == ".png" and path.is_file():
            paths.append(path)
    return paths


def _cache_from_output(output: str, root: Path) -> Optional[Path]:
    matches = list(_CACHE_RE.finditer(output))
    if not matches:
        return None
    path = Path(matches[-1].group("path")).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return path.resolve() if path.is_file() else None


def _run_job(
    job: EvaluationJob,
    project_root: Path,
    run: object,
    image_factory: Callable[..., object],
    output_root: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Run one job, emit status/images to W&B, and return its console output."""
    print(f"[train] final evaluation: {job.name}")
    try:
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if output_root is not None:
            child_env[EVAL_OUTPUT_ROOT_ENV] = str(output_root.resolve())
        completed = subprocess.Popen(
            job.command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )
        chunks: List[str] = []
        display_buffer: List[str] = []
        if completed.stdout is not None:
            while True:
                chunk = completed.stdout.read(1)
                if chunk == "":
                    break
                chunks.append(chunk)
                display_buffer.append(chunk)
                if chunk in {"\r", "\n"}:
                    print("".join(display_buffer), end="", flush=True)
                    display_buffer.clear()
        if display_buffer:
            print("".join(display_buffer), end="", flush=True)
        completed.wait()
        output = "".join(chunks)
    except OSError as exc:
        output = f"Could not start evaluator: {exc}"
        completed = None

    if output and not output.endswith(("\n", "\r")):
        print()
    success = completed is not None and completed.returncode == 0
    status: Dict[str, object] = {
        f"Evaluation/{job.name}/status": "success" if success else "failed",
    }
    if completed is not None:
        status[f"Evaluation/{job.name}/return_code"] = completed.returncode
    if not success:
        # Keep the W&B record compact while retaining the useful failure context.
        status[f"Evaluation/{job.name}/output_tail"] = output[-4000:]
    run.log(status)

    # Evaluators explicitly report each figure they save.  Use those paths
    # rather than scanning changed files: several figure names are shared by
    # models, and another process may legitimately update its own PNG nearby.
    figures = set(_paths_reported_by_evaluator(output, project_root))
    if figures:
        paths = sorted(figures)
        run.log(
            {
                f"Evaluation/{job.name}/images": [
                    image_factory(str(path), caption=f"{job.name}: {path.name}")
                    for path in paths
                ],
                f"Evaluation/{job.name}/image_count": len(paths),
            }
        )
    return success, output


def _log_skipped(run: object, name: str, reason: str) -> None:
    print(f"[train] final evaluation: skipping {name} ({reason})")
    run.log({f"Evaluation/{name}/status": "skipped", f"Evaluation/{name}/reason": reason})


def _log_suite_summary(run: object, succeeded: int, failed: int, skipped: int) -> str:
    """Make an incomplete suite obvious in the W&B run overview."""
    if failed == 0 and skipped == 0:
        status = "success"
    elif succeeded:
        status = "partial"
    else:
        status = "failed"
    run.log(
        {
            "Evaluation/status": status,
            "Evaluation/succeeded": succeeded,
            "Evaluation/failed": failed,
            "Evaluation/skipped": skipped,
        }
    )
    print(
        f"[train] final evaluation complete: {status} "
        f"({succeeded} succeeded, {failed} failed, {skipped} skipped)"
    )
    return status


def run_final_evaluation(
    project_root: Path,
    checkpoint: Path,
    run: object,
    image_factory: Callable[..., object],
    python_executable: Optional[str] = None,
    embeddings_dir: Optional[Path] = None,
    fraction: float = FINAL_EVALUATION_FRACTION,
    seed: int = FINAL_EVALUATION_SEED,
    output_root: Optional[Path] = None,
) -> str:
    """Run the full evaluation suite without allowing one failed plot to stop it."""
    root = project_root.resolve()
    ckpt = checkpoint.resolve()
    embeddings = embeddings_dir.resolve() if embeddings_dir is not None else None
    suite_root = output_root.resolve() if output_root is not None else root / "eval"
    eval_root = (root / "eval").resolve()
    try:
        suite_root.relative_to(eval_root)
    except ValueError as exc:
        raise ValueError(f"Evaluation output root must be below {eval_root}: {suite_root}") from exc
    suite_root.mkdir(parents=True, exist_ok=True)
    print(f"[train] final evaluation output: {suite_root}")
    run.log({"Evaluation/output_root": str(suite_root)})
    progress = tqdm(total=8, desc="Final evaluation", unit="task", dynamic_ncols=True)

    def run_job(job: EvaluationJob) -> Tuple[bool, str]:
        progress.set_description(f"Final evaluation: {job.name}")
        try:
            result = _run_job(job, root, run, image_factory, suite_root)
        except BaseException:
            progress.update(1)
            progress.close()
            raise
        progress.update(1)
        return result

    first_order = build_first_order_command(
        root, ckpt, python_executable, embeddings, fraction, seed
    )
    first_ok, first_output = run_job(first_order)
    succeeded = int(first_ok)
    failed = int(not first_ok)
    skipped = 0
    first_cache = _cache_from_output(first_output, root) if first_ok else None
    if first_cache is None:
        reason = "first_order did not produce a usable waveform cache"
        for name in (
            "peak_amplitudes",
            "residual_distributions",
            "distributions",
            "shake_duration",
        ):
            _log_skipped(run, name, reason)
            skipped += 1
            progress.update(1)
        # The scenario sweep evaluators do not depend on first-order output.
        independent = [
            EvaluationJob(
                name,
                (
                    python_executable or sys.executable,
                    str(root / "eval" / script),
                    "all",
                    "--checkpoint",
                    str(ckpt),
                    "--fraction",
                    f"{fraction:g}",
                    "--seed",
                    str(seed),
                    *(("--embeddings_dir", str(embeddings)) if embeddings is not None else ()),
                ),
            )
            for name, script in (
                ("attenuation", "evaluate_attenuation.py"),
                ("scaling", "evaluate_scaling.py"),
                ("model_probabilities", "evaluate_model_probabilities.py"),
            )
        ]
        for job in independent:
            ok, _ = run_job(job)
            succeeded += int(ok)
            failed += int(not ok)
        progress.close()
        return _log_suite_summary(run, succeeded, failed, skipped)

    # The peak cache name is determined by the first-order cache.  The peak
    # evaluator writes it deterministically and prints it even when reusing it.
    peaks_cache = suite_root / "peak_amplitudes" / (
        f"peaks_{first_cache.stem.removeprefix('cache_')}.npz"
    )
    peak_succeeded = False
    for job in build_evaluation_commands(
        root, ckpt, first_cache, peaks_cache, python_executable, embeddings,
        fraction, seed,
    ):
        if job.name == "residual_distributions" and (
            not peak_succeeded or not peaks_cache.is_file()
        ):
            _log_skipped(run, job.name, "peak_amplitudes did not produce a usable cache")
            skipped += 1
            progress.update(1)
            continue
        job_ok, _ = run_job(job)
        succeeded += int(job_ok)
        failed += int(not job_ok)
        if job.name == "peak_amplitudes":
            peak_succeeded = job_ok
    progress.close()
    return _log_suite_summary(run, succeeded, failed, skipped)
