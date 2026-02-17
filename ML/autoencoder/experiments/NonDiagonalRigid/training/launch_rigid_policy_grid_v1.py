#!/usr/bin/env python3
"""
Generate and optionally launch NonDiagonalRigid Phase-1 policy grid commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _parse_scalar(v: str) -> Any:
    t = v.strip().strip('"').strip("'")
    if not t:
        return t
    if t.lower() in {"true", "false"}:
        return t.lower() == "true"
    try:
        if "." in t:
            return float(t)
        return int(t)
    except ValueError:
        return t


def load_yaml_like(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        pass

    data: Dict[str, Any] = {}
    current_key = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.startswith("  - ") and current_key is not None:
                data.setdefault(current_key, []).append(_parse_scalar(line[4:]))
            elif ":" in line and not line.startswith("  "):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                current_key = key
                if not val:
                    data[key] = []
                else:
                    data[key] = _parse_scalar(val)
    return data


def load_budget_config(path: Path) -> Dict[str, Any]:
    data = load_yaml_like(path)
    if isinstance(data, dict) and "phases" in data and isinstance(data["phases"], dict):
        return data

    phases: Dict[str, Dict[str, Any]] = {}
    in_phases = False
    current_phase = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue
            if line.strip() == "phases:":
                in_phases = True
                continue
            if not in_phases:
                continue
            if line.startswith("  ") and line.strip().endswith(":") and not line.startswith("    "):
                current_phase = line.strip()[:-1]
                phases[current_phase] = {}
                continue
            if current_phase is None:
                continue
            if line.startswith("    ") and ":" in line:
                k, v = line.strip().split(":", 1)
                phases[current_phase][k] = _parse_scalar(v)
    return {"phases": phases}


def scale_tag(scale: float) -> str:
    return str(scale).replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch/generate commands for NonDiagonalRigid Phase-1 policy grid.")
    parser.add_argument(
        "--policy-grid-config",
        default="ML/autoencoder/experiments/NonDiagonalRigid/configs/backbone_policy_grid_v1.yaml",
    )
    parser.add_argument(
        "--training-budget-config",
        default="ML/autoencoder/experiments/NonDiagonalRigid/configs/training_budget_v1.yaml",
    )
    parser.add_argument(
        "--script",
        default="ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_policy_single.py",
    )
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument(
        "--mode",
        choices=["print", "script", "launch"],
        default="script",
    )
    parser.add_argument(
        "--out-script",
        default="ML/autoencoder/experiments/NonDiagonalRigid/training/run_rigid_policy_grid_v1.sh",
    )
    parser.add_argument(
        "--master-log",
        default="ML/autoencoder/experiments/NonDiagonalRigid/logs/run_rigid_policy_grid_v1.master.log",
    )
    parser.add_argument(
        "--phase",
        choices=["pilot", "final", "custom"],
        default="pilot",
    )
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=-1, help="Override fixed latent for policy stage.")
    parser.add_argument("--base-channels", default="32,64,128,256")

    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--val-check-every-steps", type=int, default=2000)
    parser.add_argument("--patience-evals", type=int, default=9999)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-steps-before-early-stop", type=int, default=0)
    args = parser.parse_args()

    grid = load_yaml_like(Path(args.policy_grid_config))
    families: List[str] = list(grid.get("model_families", []))
    policies: List[str] = list(grid.get("policies", []))
    scales: List[float] = [float(x) for x in list(grid.get("scales", []))]
    seeds: List[int] = [int(x) for x in list(grid.get("seeds", []))]
    latent_dim = int(grid.get("latent_fixed_for_policy_stage", 128)) if args.latent_dim <= 0 else args.latent_dim

    if args.phase == "custom":
        max_steps = args.max_steps
        val_check = args.val_check_every_steps
        patience = args.patience_evals
        min_delta = args.min_delta
        min_steps_before_es = args.min_steps_before_early_stop
    else:
        budget = load_budget_config(Path(args.training_budget_config))
        phase_cfg = budget.get("phases", {}).get(args.phase, {})
        if not phase_cfg:
            raise ValueError(f"Phase '{args.phase}' not found in {args.training_budget_config}")
        max_steps = int(phase_cfg.get("max_steps", args.max_steps))
        val_check = int(phase_cfg.get("val_check_every_steps", args.val_check_every_steps))
        patience = int(phase_cfg.get("patience_evals", args.patience_evals))
        min_delta = float(phase_cfg.get("min_delta", args.min_delta))
        min_steps_before_es = int(phase_cfg.get("min_steps_before_early_stop", args.min_steps_before_early_stop))

    cmds = []
    for fam in families:
        for pol in policies:
            for sc in scales:
                for sd in seeds:
                    run_name = f"rigid_policy_{args.phase}_{fam}_{pol}_sc{scale_tag(sc)}_ld{latent_dim}_s{sd}"
                    cmd = [
                        args.python_bin,
                        args.script,
                        "--model_family",
                        str(fam),
                        "--policy",
                        str(pol),
                        "--scale",
                        str(sc),
                        "--latent_dim",
                        str(latent_dim),
                        "--seed",
                        str(sd),
                        "--base_channels",
                        str(args.base_channels),
                        "--batch_size",
                        str(args.batch_size),
                        "--num_workers",
                        str(args.num_workers),
                        "--max_steps",
                        str(max_steps),
                        "--val_check_every_steps",
                        str(val_check),
                        "--patience_evals",
                        str(patience),
                        "--min_delta",
                        str(min_delta),
                        "--min_steps_before_early_stop",
                        str(min_steps_before_es),
                        "--run_name",
                        run_name,
                    ]
                    cmds.append(cmd)

    if args.max_jobs > 0:
        cmds = cmds[: args.max_jobs]

    if args.mode == "print":
        for cmd in cmds:
            print(" ".join(shlex.quote(x) for x in cmd))
        print(
            f"# total_jobs={len(cmds)} phase={args.phase} latent_dim={latent_dim} "
            f"max_steps={max_steps} val_check={val_check} patience={patience}"
        )
        return 0

    if args.mode == "script":
        out = Path(args.out_script)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# generated_at={datetime.now().isoformat()}",
            f"# total_jobs={len(cmds)} phase={args.phase} latent_dim={latent_dim}",
            f"# budget: max_steps={max_steps} val_check={val_check} patience={patience}",
            "",
            "ROOT_LOG_DIR=\"ML/autoencoder/experiments/NonDiagonalRigid/logs\"",
            "mkdir -p \"$ROOT_LOG_DIR\"",
            "",
        ]
        for i, cmd in enumerate(cmds, start=1):
            run_name = cmd[-1]
            log_file = f"$ROOT_LOG_DIR/{run_name}.launch.log"
            cmd_str = " ".join(shlex.quote(x) for x in cmd)
            lines.append(f"echo \"[{i}/{len(cmds)}] running {run_name}\"")
            lines.append(f"{cmd_str} >> {log_file} 2>&1")
            lines.append("")

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(out, 0o755)
        print(f"[OK] Script written: {out} (jobs={len(cmds)})")
        return 0

    # launch mode: detach a single master script that runs jobs sequentially.
    script_cmd = [
        "python3",
        "ML/autoencoder/experiments/NonDiagonalRigid/training/launch_rigid_policy_grid_v1.py",
        "--mode",
        "script",
        "--phase",
        args.phase,
        "--policy-grid-config",
        args.policy_grid_config,
        "--training-budget-config",
        args.training_budget_config,
        "--script",
        args.script,
        "--python-bin",
        args.python_bin,
        "--out-script",
        args.out_script,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--latent-dim",
        str(args.latent_dim),
        "--base-channels",
        args.base_channels,
        "--max-jobs",
        str(args.max_jobs),
        "--max-steps",
        str(args.max_steps),
        "--val-check-every-steps",
        str(args.val_check_every_steps),
        "--patience-evals",
        str(args.patience_evals),
        "--min-delta",
        str(args.min_delta),
        "--min-steps-before-early-stop",
        str(args.min_steps_before_early_stop),
    ]
    subprocess.check_call(script_cmd)

    master_log = Path(args.master_log)
    master_log.parent.mkdir(parents=True, exist_ok=True)
    detached_cmd = f"{args.out_script} >> {str(master_log)} 2>&1"
    proc = subprocess.Popen(["setsid", "bash", "-lc", detached_cmd])

    launch_meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": args.phase,
        "latent_dim": latent_dim,
        "total_jobs": len(cmds),
        "master_pid": proc.pid,
        "script": args.out_script,
        "master_log": str(master_log),
    }
    log_dir = Path("ML/autoencoder/experiments/NonDiagonalRigid/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / f"launch_policy_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    meta_path.write_text(json.dumps(launch_meta, indent=2), encoding="utf-8")
    print(f"[OK] Detached master launched pid={proc.pid}")
    print(f"[OK] Script: {args.out_script}")
    print(f"[OK] Master log: {master_log}")
    print(f"[OK] Manifest: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
