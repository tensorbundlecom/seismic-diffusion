import argparse
import copy
import os
import subprocess
from datetime import datetime
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


from ML.autoencoder.experiments.LegacyCondDiffusion.core.config_utils import load_config, save_json


def parse_args():
    p = argparse.ArgumentParser(description="Prepare/launch diffusion ablation grid runs.")
    p.add_argument(
        "--grid_config",
        default="ML/autoencoder/experiments/LegacyCondDiffusion/configs/ablation_grid_v1.json",
    )
    p.add_argument("--launch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    grid = load_config(args.grid_config)
    exp_root = Path("ML/autoencoder/experiments/LegacyCondDiffusion")
    generated_cfg_dir = exp_root / "configs" / "generated"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(grid["base_config"])
    launches = []
    for denoiser in grid["denoisers"]:
        for cond_mode in grid["cond_modes"]:
            cfg = copy.deepcopy(base_cfg)
            run_name = f"diff_{denoiser}_{cond_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cfg["run"]["name"] = run_name
            cfg["model"]["denoiser"] = denoiser
            cfg["model"]["cond_mode"] = cond_mode
            cfg_path = generated_cfg_dir / f"{run_name}.json"
            save_json(cfg, str(cfg_path))
            launches.append((run_name, cfg_path))

    print("[INFO] Planned runs:")
    for run_name, cfg_path in launches:
        print(f"  - {run_name}: {cfg_path}")

    if not args.launch:
        return

    logs_dir = exp_root / "logs" / "ablation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for run_name, cfg_path in launches:
        log_path = logs_dir / f"{run_name}.log"
        cmd = (
            "setsid bash -lc '/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/LegacyCondDiffusion/training/"
            f"train_latent_diffusion.py --config {cfg_path} > {log_path} 2>&1 < /dev/null & disown'"
        )
        subprocess.run(cmd, shell=True, check=False)
        print(f"[LAUNCH] {run_name} -> {log_path}")


if __name__ == "__main__":
    main()
