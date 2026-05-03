from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.edm import Stage2EDM
from core.frozen_config import default_config_path, load_frozen_config
from core.latent_cache_dataset import build_station_mapping
from core.stage2_loaders import build_stage2_dataloaders
from core.stage2_unet import build_stage2_unet_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PaperRepro Stage-2 latent EDM.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--cache-root", type=str, required=True, help="Stage-2 latent cache directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Explicit run name.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override workers.")
    parser.add_argument("--dry-run", action="store_true", help="Run a single train/val batch only.")
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def should_fallback_num_workers(exc: Exception) -> bool:
    text = str(exc)
    return "torch_shm_manager" in text or "Operation not permitted" in text


def build_stage2_run_name(cfg: dict) -> str:
    template = cfg["operations"]["run_name_templates"]["stage2"]
    version_token = str(cfg["version"]).split("_")[-1]
    tag_parts = [
        "hh100",
        "ori4064",
        "logspec128",
        "lat8x32x32",
        "evt801010",
        "s42",
        version_token,
    ]
    station_cfg = cfg["stage2"]["conditioning"].get("station_embedding", {})
    if bool(station_cfg.get("enabled", False)):
        tag_parts.insert(-1, f"semb{int(station_cfg.get('embedding_dim', 0))}")
    tag = "_".join(tag_parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return template.replace("YYYYMMDD_HHMM", timestamp).replace("<tag>", tag)


def save_checkpoint(path: Path, *, model: Stage2EDM, optimizer, scheduler, epoch: int, global_step: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def evaluate(model: Stage2EDM, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            latent = batch["latent"].to(device)
            cond = batch["condition_normalized"].to(device)
            station_index = batch["station_index"].to(device)
            metrics = model.compute_loss(latent, cond, station_index=station_index)
            batch_size = latent.shape[0]
            total_loss += float(metrics["loss"].item()) * batch_size
            count += batch_size
    return {"validation/loss": total_loss / max(count, 1)}


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    train_cfg = cfg["stage2"]["training"]
    seed = int(train_cfg["seed"])
    set_seed(seed)

    run_name = args.run_name or build_stage2_run_name(cfg)
    run_root = EXPERIMENT_ROOT / "runs" / run_name
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = run_root / "config_snapshot.yaml"
    if not config_snapshot_path.exists():
        shutil.copyfile(args.config or default_config_path(), config_snapshot_path)
    station_mapping = build_station_mapping(args.cache_root)
    save_json(run_root / "upstream_checkpoint_ref.json", {"cache_root": str(Path(args.cache_root).resolve()), "num_stations": len(station_mapping)})

    device = detect_device(args.device)
    loader_num_workers = 0 if args.dry_run and args.num_workers is None else args.num_workers
    effective_num_workers = loader_num_workers
    log(f"Starting Stage-2 run: {run_name}")
    log(f"Device={device} epochs={args.epochs or train_cfg['epochs']} batch_size={args.batch_size or train_cfg['batch_size']} num_workers={effective_num_workers if effective_num_workers is not None else train_cfg['num_workers']}")

    train_loader, val_loader = build_stage2_dataloaders(
        cfg,
        cache_root=args.cache_root,
        num_workers_override=effective_num_workers,
        batch_size_override=args.batch_size,
    )

    denoiser = build_stage2_unet_from_config(cfg, num_stations=len(station_mapping)).to(device)
    edm_cfg = cfg["stage2"]["edm"]
    model = Stage2EDM(
        denoiser,
        sigma_min=float(edm_cfg["sigma_min"]),
        sigma_max=float(edm_cfg["sigma_max"]),
        sigma_data=float(edm_cfg["sigma_data"]),
        p_mean=float(edm_cfg["P_mean"]),
        p_std=float(edm_cfg["P_std"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(args.epochs or train_cfg["epochs"])
    max_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=float(train_cfg["eta_min"]))

    if args.dry_run:
        try:
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
        except RuntimeError as exc:
            if effective_num_workers and should_fallback_num_workers(exc):
                log("Shared-memory worker startup failed; retrying with num_workers=0.")
                effective_num_workers = 0
                train_loader, val_loader = build_stage2_dataloaders(
                    cfg,
                    cache_root=args.cache_root,
                    num_workers_override=0,
                    batch_size_override=args.batch_size,
                )
                train_batch = next(iter(train_loader))
                val_batch = next(iter(val_loader))
            else:
                raise

        with torch.no_grad():
            train_metrics = model.compute_loss(
                train_batch["latent"].to(device),
                train_batch["condition_normalized"].to(device),
                station_index=train_batch["station_index"].to(device),
            )
            val_metrics = model.compute_loss(
                val_batch["latent"].to(device),
                val_batch["condition_normalized"].to(device),
                station_index=val_batch["station_index"].to(device),
            )
        save_json(
            run_root / "dry_run_summary.json",
            {
                "run_name": run_name,
                "device": str(device),
                "num_workers": effective_num_workers,
                "train_latent_shape": list(train_batch["latent"].shape),
                "train_cond_shape": list(train_batch["condition_normalized"].shape),
                "val_latent_shape": list(val_batch["latent"].shape),
                "val_cond_shape": list(val_batch["condition_normalized"].shape),
                "train_loss": float(train_metrics["loss"].item()),
                "val_loss": float(val_metrics["loss"].item()),
            },
        )
        log(f"Dry-run finished. Summary written to {run_root / 'dry_run_summary.json'}")
        return

    history = []
    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()
    warmed_up = False
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_iterator = iter(train_loader)
        if not warmed_up:
            try:
                first_batch = next(train_iterator)
            except RuntimeError as exc:
                if effective_num_workers and should_fallback_num_workers(exc):
                    log("Shared-memory worker startup failed during warmup; retrying with num_workers=0.")
                    effective_num_workers = 0
                    train_loader, val_loader = build_stage2_dataloaders(
                        cfg,
                        cache_root=args.cache_root,
                        num_workers_override=0,
                        batch_size_override=args.batch_size,
                    )
                    train_iterator = iter(train_loader)
                    first_batch = next(train_iterator)
                else:
                    raise
            batches = itertools.chain([first_batch], train_iterator)
            warmed_up = True
        else:
            batches = train_iterator

        for batch in batches:
            latent = batch["latent"].to(device)
            cond = batch["condition_normalized"].to(device)
            station_index = batch["station_index"].to(device)
            metrics = model.compute_loss(latent, cond, station_index=station_index)
            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            optimizer.step()
            scheduler.step()
            batch_size_actual = latent.shape[0]
            train_loss_sum += float(metrics["loss"].item()) * batch_size_actual
            train_count += batch_size_actual
            global_step += 1

        validation_metrics = evaluate(model, val_loader, device)
        train_metrics = {"training/loss": train_loss_sum / max(train_count, 1)}
        epoch_metrics = {"epoch": epoch, "global_step": global_step, **train_metrics, **validation_metrics}
        history.append(epoch_metrics)
        save_json(logs_dir / "history.json", {"epochs": history})
        log(f"Epoch {epoch}/{epochs} train_loss={train_metrics['training/loss']:.6f} val_loss={validation_metrics['validation/loss']:.6f}")

        current_val_loss = float(validation_metrics["validation/loss"])
        save_checkpoint(
            checkpoints_dir / "last.ckpt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=min(best_val_loss, current_val_loss),
        )
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_checkpoint(
                checkpoints_dir / "best_val_loss.ckpt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )
            log(f"New best validation checkpoint: {checkpoints_dir / 'best_val_loss.ckpt'}")

    save_json(
        run_root / "resource_summary.json",
        {
            "device": str(device),
            "num_workers": effective_num_workers,
            "epochs": epochs,
            "global_step": global_step,
            "train_wall_clock_sec": time.time() - start_time,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "gpu_peak_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else None
            ),
        },
    )
    log(f"Training finished. Artifacts written under {run_root}")


if __name__ == "__main__":
    main()
