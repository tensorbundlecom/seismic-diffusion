#!/usr/bin/env python3
"""
Generate and optionally launch NonDiagonalRigid V1 training commands.
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


def load_yaml_like(path: Path) -> Dict[str, Any]:
    # Prefer PyYAML if available.
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        pass

    # Minimal fallback parser for this project's simple list-based config.
    data: Dict[str, Any] = {}
    current_key = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line.startswith("  - ") and current_key is not None:
                val = line[4:].strip().strip('"').strip("'")
                if val.isdigit():
                    val = int(val)
                data.setdefault(current_key, []).append(val)
            elif ":" in line and not line.startswith("  "):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                current_key = key
                if not val:
                    data[key] = []
                else:
                    v = val.strip('"').strip("'")
                    if v.isdigit():
                        v = int(v)
                    data[key] = v
            else:
                continue
    return data


def load_budget_config(path: Path) -> Dict[str, Any]:
    # Try generic YAML loader first.
    data = load_yaml_like(path)
    if isinstance(data, dict) and "phases" in data and isinstance(data["phases"], dict):
        return data

    # Fallback parser for nested "phases" block.
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

            # Phase header, e.g. "  pilot:"
            if line.startswith("  ") and line.strip().endswith(":") and not line.startswith("    "):
                current_phase = line.strip()[:-1]
                phases[current_phase] = {}
                continue

            if current_phase is None:
                continue

            # Key-value inside phase, e.g. "    max_steps: 12000"
            if line.startswith("    ") and ":" in line:
                k, v = line.strip().split(":", 1)
                vv = v.strip().strip('"').strip("'")
                if vv.lower() in {"true", "false"}:
                    parsed: Any = vv.lower() == "true"
                else:
                    try:
                        parsed = int(vv)
                    except ValueError:
                        try:
                            parsed = float(vv)
                        except ValueError:
                            parsed = vv
                phases[current_phase][k] = parsed

    return {"phases": phases}


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch/generate commands for NonDiagonalRigid V1 grid.")
    parser.add_argument(
        "--grid-config",
        default="ML/autoencoder/experiments/NonDiagonalRigid/configs/model_grid_v1.yaml",
    )
    parser.add_argument(
        "--script",
        default="ML/autoencoder/experiments/NonDiagonalRigid/training/train_rigid_single.py",
    )
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument(
        "--mode",
        choices=["print", "script", "launch"],
        default="script",
        help="print: stdout only, script: write shell script, launch: detached launch now",
    )
    parser.add_argument(
        "--out-script",
        default="ML/autoencoder/experiments/NonDiagonalRigid/training/run_rigid_grid_v1.sh",
    )
    parser.add_argument(
        "--master-log",
        default="ML/autoencoder/experiments/NonDiagonalRigid/logs/run_rigid_grid_v1.master.log",
    )
    parser.add_argument(
        "--training-budget-config",
        default="ML/autoencoder/experiments/NonDiagonalRigid/configs/training_budget_v1.yaml",
    )
    parser.add_argument(
        "--phase",
        choices=["pilot", "final", "custom"],
        default="final",
        help="training budget profile: pilot/final from config, or custom from CLI args",
    )
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means all jobs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=60000)
    parser.add_argument("--val-check-every-steps", type=int, default=1000)
    parser.add_argument("--patience-evals", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-steps-before-early-stop", type=int, default=0)
    args = parser.parse_args()

    grid = load_yaml_like(Path(args.grid_config))
    families: List[str] = list(grid.get("model_families", []))
    backbones: List[str] = list(grid.get("backbones", []))
    latent_dims: List[int] = list(grid.get("latent_dims", []))
    seeds: List[int] = list(grid.get("seeds", []))

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
        for bb in backbones:
            for ld in latent_dims:
                for sd in seeds:
                    run_name = f"rigid_{args.phase}_{fam}_{bb}_ld{ld}_s{sd}"
                    cmd = [
                        args.python_bin,
                        args.script,
                        "--model_family",
                        str(fam),
                        "--backbone",
                        str(bb),
                        "--latent_dim",
                        str(ld),
                        "--seed",
                        str(sd),
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
        print(f"# total_jobs={len(cmds)} phase={args.phase} max_steps={max_steps} val_check={val_check} patience={patience}")
        return 0

    if args.mode == "script":
        out = Path(args.out_script)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# generated_at={datetime.now().isoformat()}",
            f"# total_jobs={len(cmds)}",
            f"# phase={args.phase} max_steps={max_steps} val_check={val_check} patience={patience}",
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
        "ML/autoencoder/experiments/NonDiagonalRigid/training/launch_rigid_grid_v1.py",
        "--mode",
        "script",
        "--phase",
        args.phase,
        "--grid-config",
        args.grid_config,
        "--script",
        args.script,
        "--python-bin",
        args.python_bin,
        "--out-script",
        args.out_script,
        "--training-budget-config",
        args.training_budget_config,
        "--max-jobs",
        str(args.max_jobs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
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
        "total_jobs": len(cmds),
        "master_pid": proc.pid,
        "script": args.out_script,
        "master_log": str(master_log),
    }
    log_dir = Path("ML/autoencoder/experiments/NonDiagonalRigid/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / f"launch_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    meta_path.write_text(json.dumps(launch_meta, indent=2), encoding="utf-8")
    print(f"[OK] Detached master launched pid={proc.pid}")
    print(f"[OK] Script: {args.out_script}")
    print(f"[OK] Master log: {master_log}")
    print(f"[OK] Manifest: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
