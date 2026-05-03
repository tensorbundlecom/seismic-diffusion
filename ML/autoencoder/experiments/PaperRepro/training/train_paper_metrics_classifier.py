from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.binning import distance_bin_edges_from_config, magnitude_bin_edges_from_config, num_joint_classes
from core.classifier_datasets import PaperReproClassifierDataset, save_classifier_bins
from core.frozen_config import default_config_path, load_frozen_config
from core.paper_metrics_classifier import build_classifier_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train the PaperRepro classifier required by paper-style metrics.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num workers.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name.")
    parser.add_argument("--dry-run", action="store_true", help="Build one batch and exit.")
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def detect_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_run_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"run_{timestamp}_classifier_hh100_ori4064_logspec128_evt801010_s42_v1"


def run_dir_from_name(name: str) -> Path:
    return EXPERIMENT_ROOT / "runs" / name


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def evaluate(model, loader: DataLoader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            total_loss += float(loss.item()) * x.shape[0]
            total_correct += int((pred == y).sum().item())
            total_count += int(x.shape[0])
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "count": total_count,
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg_path = args.config or default_config_path()
    cfg = load_frozen_config(cfg_path)

    train_cfg = cfg["evaluation"]["classifier"]["training"]
    device = detect_device(args.device)
    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_name = args.run_name or default_run_name()
    run_dir = run_dir_from_name(run_name)
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, run_dir / "config_snapshot.yaml")

    train_dataset = PaperReproClassifierDataset(cfg, splits=("train",))
    validation_dataset = PaperReproClassifierDataset(cfg, splits=("validation",))
    test_dataset = PaperReproClassifierDataset(cfg, splits=("test",))
    ood_dataset = PaperReproClassifierDataset(cfg, splits=("ood",))

    class_weights = train_dataset.class_weights()
    num_classes = int(num_joint_classes(magnitude_bin_edges_from_config(cfg), distance_bin_edges_from_config(cfg)))

    batch_size = int(args.batch_size or train_cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg["num_workers"])
    learning_rate = float(args.learning_rate or train_cfg["learning_rate"])
    epochs = int(args.epochs or train_cfg["epochs"])

    train_loader = build_loader(train_dataset, batch_size, num_workers, True, device)
    validation_loader = build_loader(validation_dataset, batch_size, num_workers, False, device)
    test_loader = build_loader(test_dataset, batch_size, num_workers, False, device)
    ood_loader = build_loader(ood_dataset, batch_size, num_workers, False, device)

    model = build_classifier_from_config(cfg, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(len(train_loader) * epochs, 1),
        eta_min=float(train_cfg["eta_min"]),
    )

    save_classifier_bins(
        run_dir / "classifier_bins.json",
        magnitude_edges=magnitude_bin_edges_from_config(cfg),
        distance_edges=distance_bin_edges_from_config(cfg),
    )

    dry_summary = {
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_classes": num_classes,
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_size": len(test_dataset),
        "ood_size": len(ood_dataset),
    }
    if args.dry_run:
        batch = next(iter(train_loader))
        x = batch["signal"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        dry_summary.update(
            {
                "batch_signal_shape": list(x.shape),
                "batch_label_shape": list(y.shape),
                "logits_shape": list(logits.shape),
            }
        )
        save_json(run_dir / "dry_run_summary.json", dry_summary)
        log(f"Dry-run complete. Artifacts written under {run_dir}")
        return

    log(f"Starting classifier run: {run_name}")
    log(f"Device={device} epochs={epochs} batch_size={batch_size} num_workers={num_workers}")

    history: list[dict] = []
    best_val_accuracy = -math.inf
    start_time = time.time()
    gpu_peak_memory_mb = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for batch in train_loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred = logits.argmax(dim=1)
            running_loss += float(loss.item()) * x.shape[0]
            running_correct += int((pred == y).sum().item())
            running_count += int(x.shape[0])

            if running_count % max(batch_size * 100, 1) == 0:
                log(
                    "Epoch {epoch}/{epochs} progress samples={samples} train_loss_so_far={loss:.6f} train_acc_so_far={acc:.4f}".format(
                        epoch=epoch,
                        epochs=epochs,
                        samples=running_count,
                        loss=running_loss / max(running_count, 1),
                        acc=running_correct / max(running_count, 1),
                    )
                )

        train_loss = running_loss / max(running_count, 1)
        train_accuracy = running_correct / max(running_count, 1)
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        ood_metrics = evaluate(model, ood_loader, criterion, device)

        if device.type == "cuda":
            gpu_peak_memory_mb = max(
                gpu_peak_memory_mb,
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "test_accuracy_real": test_metrics["accuracy"],
            "ood_accuracy_real": ood_metrics["accuracy"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_record)
        save_json(logs_dir / "history.json", {"epochs": history})

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_accuracy": validation_metrics["accuracy"],
            "num_classes": num_classes,
        }
        torch.save(checkpoint_payload, checkpoints_dir / "last.pt")
        if validation_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = validation_metrics["accuracy"]
            torch.save(checkpoint_payload, checkpoints_dir / "best_val_accuracy.pt")
            log(f"New best classifier checkpoint: {checkpoints_dir / 'best_val_accuracy.pt'}")

        log(
            "Epoch {epoch}/{epochs} train_loss={train_loss:.6f} train_acc={train_accuracy:.4f} "
            "val_loss={val_loss:.6f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} ood_acc={ood_acc:.4f}".format(
                epoch=epoch,
                epochs=epochs,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=validation_metrics["loss"],
                val_acc=validation_metrics["accuracy"],
                test_acc=test_metrics["accuracy"],
                ood_acc=ood_metrics["accuracy"],
            )
        )

    resource_summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "training_wall_clock_sec": time.time() - start_time,
        "num_workers": num_workers,
        "batch_size": batch_size,
    }
    save_json(run_dir / "resource_summary.json", resource_summary)

    final_checkpoint = torch.load(checkpoints_dir / "best_val_accuracy.pt", map_location=device)
    model.load_state_dict(final_checkpoint["model_state_dict"])
    final_test = evaluate(model, test_loader, criterion, device)
    final_ood = evaluate(model, ood_loader, criterion, device)
    save_json(
        run_dir / "real_data_eval.json",
        {
            "validation_best_accuracy": best_val_accuracy,
            "test_accuracy_real": final_test["accuracy"],
            "ood_accuracy_real": final_ood["accuracy"],
            "num_classes": num_classes,
            "train_class_counts": train_dataset.class_counts().tolist(),
        },
    )
    log(f"Classifier training finished. Artifacts written under {run_dir}")


if __name__ == "__main__":
    main()
