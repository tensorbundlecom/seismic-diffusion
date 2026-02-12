import argparse
import itertools
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Launch detached beta/lambda sweep for CVAE_MRLoss')
    p.add_argument('--betas', type=str, default='0.02,0.05,0.1,0.2')
    p.add_argument('--lambdas', type=str, default='0.2,0.4,0.6,0.8')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--launch', action='store_true', help='Actually launch jobs (otherwise dry-run)')
    args = p.parse_args()

    betas = [x.strip() for x in args.betas.split(',') if x.strip()]
    lambdas = [x.strip() for x in args.lambdas.split(',') if x.strip()]

    base_dir = Path('ML/autoencoder/experiments/CVAE_MRLoss')
    log_dir = base_dir / 'logs' / 'train'
    result_dir = base_dir / 'results'
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    runbook = []

    for beta, lam in itertools.product(betas, lambdas):
        run_name = f'b{beta}_l{lam}'.replace('.', 'p')
        log_file = log_dir / f'train_{run_name}.log'
        pid_file = log_dir / f'train_{run_name}.pid'

        cmd = (
            "setsid bash -lc '"
            f"/home/gms/.pyenv/shims/python -u ML/autoencoder/experiments/CVAE_MRLoss/training/train_cvae_mrloss_external.py "
            f"--run_name {run_name} --epochs {args.epochs} --batch_size {args.batch_size} "
            f"--lr {args.lr} --beta {beta} --lambda_mr {lam} --lambda_img {1.0 - float(lam):.6f} "
            f"--num_workers {args.num_workers} "
            f"> {log_file} 2>&1 < /dev/null & echo $! > {pid_file}; disown' >/dev/null 2>&1"
        )

        runbook.append({
            'run_name': run_name,
            'beta': float(beta),
            'lambda_mr': float(lam),
            'lambda_img': 1.0 - float(lam),
            'log_file': str(log_file),
            'pid_file': str(pid_file),
            'cmd': cmd,
        })

        if args.launch:
            subprocess.run(cmd, shell=True, check=False)

    runbook_file = result_dir / 'sweep_runbook.json'
    import json
    with open(runbook_file, 'w') as f:
        json.dump(runbook, f, indent=2)

    print(f'[INFO] Sweep runbook saved: {runbook_file}')
    print(f'[INFO] Total runs: {len(runbook)} | launch={args.launch}')


if __name__ == '__main__':
    main()
