"""Training entry point for experiments2/exp001."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import (
    ExternalHHComplexSTFTDataset,
    collate_exp001,
    make_magnitude_bin_indices,
    make_weighted_sampler_weights,
    prepare_exp001_artifacts,
)
from .evaluate import (
    HIGHER_BETTER_METRICS,
    LOWER_BETTER_METRICS,
    evaluate_condition_only,
    evaluate_reconstruction,
    total_vae_loss,
)
from .model import CVAEComplexSTFT
from .utils import (
    EarlyStopping,
    RunningMean,
    beta_linear_warmup,
    build_seed_bank,
    configure_logger,
    create_run_tree,
    load_config,
    load_json,
    robust_median_mad,
    resolve_device,
    save_json,
    save_yaml_compatible,
    set_seed,
    zscore_robust,
)


class CondgenCompositeCalibrator:
    """
    Implements D013: pre-gate + robust-z family composite.
    """

    SPEC_KEYS = ("complex_l1", "mr_lsd")
    TIME_KEYS = ("abs_xcorr_lag_s", "onset_mae_p_s", "onset_mae_s_s", "onset_mae_dtps_s")
    SHAPE_KEYS = ("xcorr_max", "envelope_corr", "band_energy_ratio_error")

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.calib_required = int(cfg["evaluation"]["condgen_calibration_runs"])
        self.calib_stats: Optional[Dict[str, Dict[str, float]]] = None

    def _passes_gate(self, metrics_mean: Mapping[str, float]) -> tuple[bool, Dict[str, bool]]:
        gate_cfg = self.cfg["evaluation"]["condgen_pre_gate"]
        quality_cfg = self.cfg["evaluation"].get("condgen_quality_gate", {"enabled": False})
        lag_abs_mean = float(
            metrics_mean.get(
                "abs_xcorr_lag_s",
                abs(float(metrics_mean.get("xcorr_lag_s", 1e9))),
            )
        )
        checks = {
            "onset_evaluable_rate": float(metrics_mean.get("onset_evaluable_rate", 0.0))
            >= float(gate_cfg["min_onset_evaluable_rate"]),
            "onset_failure_rate_p": float(metrics_mean.get("onset_failure_rate_p", 1.0))
            <= float(gate_cfg["max_onset_failure_p"]),
            "onset_failure_rate_s": float(metrics_mean.get("onset_failure_rate_s", 1.0))
            <= float(gate_cfg["max_onset_failure_s"]),
            "abs_xcorr_lag_s": lag_abs_mean <= float(gate_cfg["max_abs_xcorr_lag_s"]),
        }
        if bool(quality_cfg.get("enabled", False)):
            # Stage-2 quality gate: optional thresholds (only applied if present in config).
            if "min_xcorr_max" in quality_cfg:
                checks["quality_xcorr_max"] = float(metrics_mean.get("xcorr_max", -1.0)) >= float(
                    quality_cfg["min_xcorr_max"]
                )
            if "min_envelope_corr" in quality_cfg:
                checks["quality_envelope_corr"] = float(metrics_mean.get("envelope_corr", -1.0)) >= float(
                    quality_cfg["min_envelope_corr"]
                )
            if "max_mr_lsd" in quality_cfg:
                checks["quality_mr_lsd"] = float(metrics_mean.get("mr_lsd", 1e9)) <= float(
                    quality_cfg["max_mr_lsd"]
                )
            if "max_onset_mae_dtps_s" in quality_cfg:
                checks["quality_onset_mae_dtps_s"] = float(metrics_mean.get("onset_mae_dtps_s", 1e9)) <= float(
                    quality_cfg["max_onset_mae_dtps_s"]
                )
        return all(checks.values()), checks

    def maybe_build_calibration(self, history_means: Sequence[Mapping[str, float]]) -> None:
        if self.calib_stats is not None:
            return
        if len(history_means) < self.calib_required:
            return
        calib_rows = history_means[: self.calib_required]
        metric_names = sorted({k for row in calib_rows for k in row.keys()})
        stats: Dict[str, Dict[str, float]] = {}
        for key in metric_names:
            vals = [float(row[key]) for row in calib_rows if key in row and np.isfinite(row[key])]
            if not vals:
                continue
            med, scale = robust_median_mad(vals)
            stats[key] = {"median": med, "scale": scale}
        self.calib_stats = stats

    def _z_for_key(self, key: str, value: float) -> float:
        assert self.calib_stats is not None
        st = self.calib_stats.get(key)
        if st is None:
            return 4.0
        med = st["median"]
        scale = st["scale"]
        if key in LOWER_BETTER_METRICS:
            return zscore_robust(value, med, scale)
        if key in HIGHER_BETTER_METRICS:
            return zscore_robust(med - value, 0.0, scale)
        # default lower-better
        return zscore_robust(value, med, scale)

    def composite_from_metrics(self, metrics_mean: Mapping[str, float]) -> Optional[Dict[str, float]]:
        if self.calib_stats is None:
            return None
        z_spec = np.mean([self._z_for_key(k, float(metrics_mean.get(k, np.nan))) for k in self.SPEC_KEYS])
        z_time = np.mean([self._z_for_key(k, float(metrics_mean.get(k, np.nan))) for k in self.TIME_KEYS])
        z_shape = np.mean([self._z_for_key(k, float(metrics_mean.get(k, np.nan))) for k in self.SHAPE_KEYS])
        w = self.cfg["evaluation"]["condgen_composite_weights"]
        z_comp = float(w["spec"]) * float(z_spec) + float(w["time"]) * float(z_time) + float(w["shape"]) * float(z_shape)
        return {
            "z_spec": float(z_spec),
            "z_time": float(z_time),
            "z_shape": float(z_shape),
            "z_comp": float(z_comp),
        }

    def score(
        self,
        metrics_mean: Mapping[str, float],
        history_means: Sequence[Mapping[str, float]],
    ) -> Dict[str, Any]:
        gate_ok, gate_checks = self._passes_gate(metrics_mean)
        self.maybe_build_calibration(history_means)

        out = {
            "gate_ok": gate_ok,
            "gate_checks": gate_checks,
            "calibrated": self.calib_stats is not None,
            "z_spec": None,
            "z_time": None,
            "z_shape": None,
            "z_comp": None,
        }
        if not gate_ok:
            return out
        comp_vals = self.composite_from_metrics(metrics_mean)
        if comp_vals is None:
            return out

        out["z_spec"] = comp_vals["z_spec"]
        out["z_time"] = comp_vals["z_time"]
        out["z_shape"] = comp_vals["z_shape"]
        out["z_comp"] = comp_vals["z_comp"]
        return out


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    cfg: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": float(val_loss),
        "config": cfg,
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, path)


def _build_dataloaders(
    cfg: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    norm_stats: Mapping[str, Any],
) -> Dict[str, DataLoader]:
    train_ds = ExternalHHComplexSTFTDataset(cfg, manifest, split["train"]["indices"], norm_stats)
    val_ds = ExternalHHComplexSTFTDataset(cfg, manifest, split["val"]["indices"], norm_stats)
    test_ds = ExternalHHComplexSTFTDataset(cfg, manifest, split["test"]["indices"], norm_stats)
    ood_ds = ExternalHHComplexSTFTDataset(cfg, manifest, split["ood"]["indices"], norm_stats)

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    use_weighted = bool(cfg["train"]["use_weighted_sampler"])

    if use_weighted:
        weights = make_weighted_sampler_weights(
            rows=train_ds.rows,
            alpha=float(cfg["train"]["weighted_sampler_alpha"]),
            w_max=float(cfg["train"]["weighted_sampler_wmax"]),
        )
        sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_exp001,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_exp001,
        )

    common = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_exp001,
    }
    return {
        "train": train_loader,
        "val": DataLoader(val_ds, **common),
        "test": DataLoader(test_ds, **common),
        "ood": DataLoader(ood_ds, **common),
    }


def _train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
    device: torch.device,
    epoch_1based: int,
) -> Dict[str, float]:
    model.train()
    m_total = RunningMean()
    m_rec = RunningMean()
    m_c = RunningMean()
    m_m = RunningMean()
    m_b = RunningMean()
    m_kl_raw = RunningMean()
    m_kl_fb = RunningMean()

    beta_t = beta_linear_warmup(
        epoch_idx_1based=epoch_1based,
        beta_max=float(cfg["loss"]["beta_max"]),
        warmup_epochs=int(cfg["loss"]["beta_warmup_epochs"]),
    )
    grad_clip = float(cfg["train"]["grad_clip_norm"])

    for batch in dataloader:
        x = batch["x"].to(device)
        cond = batch["cond"].to(device)
        station = batch["station_idx"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred, mu, logvar = model(x, cond, station)
        losses = total_vae_loss(
            pred=pred,
            target=x,
            mu=mu,
            logvar=logvar,
            lambda_complex=float(cfg["loss"]["lambda_complex"]),
            lambda_logmag=float(cfg["loss"]["lambda_logmag"]),
            lambda_band=float(cfg["loss"].get("lambda_band", 0.0)),
            beta_t=beta_t,
            free_bits_per_dim=float(cfg["loss"]["free_bits_per_dim"]),
            cfg=cfg,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bsz = int(x.size(0))
        m_total.update(float(losses["total"].item()), bsz)
        m_rec.update(float(losses["recon_total"].item()), bsz)
        m_c.update(float(losses["loss_complex"].item()), bsz)
        m_m.update(float(losses["loss_logmag"].item()), bsz)
        m_b.update(float(losses["loss_band"].item()), bsz)
        m_kl_raw.update(float(losses["kl_raw"].item()), bsz)
        m_kl_fb.update(float(losses["kl_fb"].item()), bsz)

    return {
        "beta_t": beta_t,
        "train_total": m_total.avg,
        "train_recon": m_rec.avg,
        "train_complex": m_c.avg,
        "train_logmag": m_m.avg,
        "train_band": m_b.avg,
        "train_kl_raw": m_kl_raw.avg,
        "train_kl_fb": m_kl_fb.avg,
    }


def _validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    cfg: Mapping[str, Any],
    device: torch.device,
    epoch_1based: int,
) -> Dict[str, float]:
    model.eval()
    m_total = RunningMean()
    m_rec = RunningMean()
    m_c = RunningMean()
    m_m = RunningMean()
    m_b = RunningMean()
    m_kl_raw = RunningMean()
    m_kl_fb = RunningMean()
    beta_t = beta_linear_warmup(
        epoch_idx_1based=epoch_1based,
        beta_max=float(cfg["loss"]["beta_max"]),
        warmup_epochs=int(cfg["loss"]["beta_warmup_epochs"]),
    )
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            cond = batch["cond"].to(device)
            station = batch["station_idx"].to(device)
            pred, mu, logvar = model(x, cond, station)
            losses = total_vae_loss(
                pred=pred,
                target=x,
                mu=mu,
                logvar=logvar,
                lambda_complex=float(cfg["loss"]["lambda_complex"]),
                lambda_logmag=float(cfg["loss"]["lambda_logmag"]),
                lambda_band=float(cfg["loss"].get("lambda_band", 0.0)),
                beta_t=beta_t,
                free_bits_per_dim=float(cfg["loss"]["free_bits_per_dim"]),
                cfg=cfg,
            )
            bsz = int(x.size(0))
            m_total.update(float(losses["total"].item()), bsz)
            m_rec.update(float(losses["recon_total"].item()), bsz)
            m_c.update(float(losses["loss_complex"].item()), bsz)
            m_m.update(float(losses["loss_logmag"].item()), bsz)
            m_b.update(float(losses["loss_band"].item()), bsz)
            m_kl_raw.update(float(losses["kl_raw"].item()), bsz)
            m_kl_fb.update(float(losses["kl_fb"].item()), bsz)
    return {
        "val_total": m_total.avg,
        "val_recon": m_rec.avg,
        "val_complex": m_c.avg,
        "val_logmag": m_m.avg,
        "val_band": m_b.avg,
        "val_kl_raw": m_kl_raw.avg,
        "val_kl_fb": m_kl_fb.avg,
    }


def _write_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_valid_by_bin_mean(obj: Any) -> bool:
    return isinstance(obj, Mapping) and all(k in obj for k in ("all", "m3to5", "ge5"))


def _extract_by_bin_mean_from_payload(payload: Mapping[str, Any]) -> Optional[Dict[str, Dict[str, float]]]:
    candidates = [
        payload.get("by_bin_mean"),
        payload.get("condition_only", {}).get("by_bin_mean") if isinstance(payload.get("condition_only"), Mapping) else None,
        payload.get("cond_eval", {}).get("by_bin_mean") if isinstance(payload.get("cond_eval"), Mapping) else None,
        payload.get("rerank", {}).get("selected", {}).get("cond_eval", {}).get("by_bin_mean")
        if isinstance(payload.get("rerank"), Mapping)
        else None,
        payload.get("splits", {}).get("val", {}).get("condition_only", {}).get("by_bin_mean")
        if isinstance(payload.get("splits"), Mapping)
        else None,
    ]
    for cand in candidates:
        if not _is_valid_by_bin_mean(cand):
            continue
        out: Dict[str, Dict[str, float]] = {}
        for bname in ("all", "m3to5", "ge5"):
            metrics = cand[bname]
            if not isinstance(metrics, Mapping):
                break
            parsed: Dict[str, float] = {}
            for k, v in metrics.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    parsed[str(k)] = fv
            out[bname] = parsed
        if all(b in out for b in ("all", "m3to5", "ge5")):
            return out
    return None


def _load_imbalance_reference_by_bin(path: str | Path) -> Dict[str, Dict[str, float]]:
    payload = load_json(path)
    by_bin = _extract_by_bin_mean_from_payload(payload)
    if by_bin is None:
        raise ValueError(
            "Could not locate by_bin_mean in imbalance guardrail reference file. "
            "Expected one of: by_bin_mean, condition_only.by_bin_mean, cond_eval.by_bin_mean, "
            "rerank.selected.cond_eval.by_bin_mean, splits.val.condition_only.by_bin_mean."
        )
    return by_bin


def _pct_improvement_lower_better(reference: float, candidate: float, eps: float = 1e-8) -> float:
    denom = max(abs(float(reference)), float(eps))
    return (float(reference) - float(candidate)) / denom * 100.0


def _evaluate_imbalance_guardrails(
    calibrator: CondgenCompositeCalibrator,
    candidate_by_bin_mean: Mapping[str, Mapping[str, float]],
    reference_by_bin_mean: Mapping[str, Mapping[str, float]],
    guard_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "ready": calibrator.calib_stats is not None,
        "pass": False,
    }
    if calibrator.calib_stats is None:
        out["reason"] = "calibration_not_ready"
        return out

    comp_ref: Dict[str, float] = {}
    comp_cand: Dict[str, float] = {}
    for bname in ("all", "m3to5", "ge5"):
        ref_metrics = reference_by_bin_mean.get(bname, {})
        cand_metrics = candidate_by_bin_mean.get(bname, {})
        ref_comp = calibrator.composite_from_metrics(ref_metrics)
        cand_comp = calibrator.composite_from_metrics(cand_metrics)
        if ref_comp is None or cand_comp is None:
            out["reason"] = f"missing_composite_for_bin:{bname}"
            return out
        comp_ref[bname] = float(ref_comp["z_comp"])
        comp_cand[bname] = float(cand_comp["z_comp"])

    imp_ge5 = _pct_improvement_lower_better(comp_ref["ge5"], comp_cand["ge5"])
    imp_all = _pct_improvement_lower_better(comp_ref["all"], comp_cand["all"])
    imp_mid = _pct_improvement_lower_better(comp_ref["m3to5"], comp_cand["m3to5"])
    deg_all = max(0.0, -imp_all)
    deg_mid = max(0.0, -imp_mid)

    tail_min = float(guard_cfg.get("tail_bin_min_improvement_pct", 5.0))
    all_max = float(guard_cfg.get("all_bin_max_degradation_pct", 8.0))
    mid_max = float(guard_cfg.get("mid_bin_max_degradation_pct", 10.0))

    checks = {
        "tail_improvement_ge5": imp_ge5 >= tail_min,
        "all_degradation": deg_all <= all_max,
        "mid_degradation_m3to5": deg_mid <= mid_max,
    }
    out.update(
        {
            "pass": all(checks.values()),
            "checks": checks,
            "reference_z_comp_by_bin": comp_ref,
            "candidate_z_comp_by_bin": comp_cand,
            "improvement_pct": {
                "ge5": float(imp_ge5),
                "all": float(imp_all),
                "m3to5": float(imp_mid),
            },
            "degradation_pct": {
                "all": float(deg_all),
                "m3to5": float(deg_mid),
            },
            "thresholds_pct": {
                "tail_bin_min_improvement_pct": tail_min,
                "all_bin_max_degradation_pct": all_max,
                "mid_bin_max_degradation_pct": mid_max,
            },
        }
    )
    return out


def _rerank_condgen_candidates(
    candidates: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: Mapping[str, Any],
    device: torch.device,
    calibrator: CondgenCompositeCalibrator,
    cond_history_means: Sequence[Mapping[str, float]],
    imbalance_reference_by_bin: Optional[Mapping[str, Mapping[str, float]]] = None,
    imbalance_guard_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not candidates:
        return {"selected": None, "results": []}

    k_final = int(cfg["evaluation"]["k_samples_final"])
    seed_bank = build_seed_bank(int(cfg["evaluation"]["seed_bank_base"]), k_final)
    results = []
    for c in candidates:
        ckpt_path = Path(c["checkpoint_path"])
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        cond_eval = evaluate_condition_only(
            model=model,
            dataloader=val_loader,
            cfg=cfg,
            device=device,
            k_samples=k_final,
            seed_bank=seed_bank,
            max_samples=None,
        )
        comp = calibrator.score(cond_eval["mean"], cond_history_means)
        if imbalance_reference_by_bin is not None and imbalance_guard_cfg is not None:
            imbalance_guard = _evaluate_imbalance_guardrails(
                calibrator=calibrator,
                candidate_by_bin_mean=cond_eval["by_bin_mean"],
                reference_by_bin_mean=imbalance_reference_by_bin,
                guard_cfg=imbalance_guard_cfg,
            )
        else:
            imbalance_guard = {"enabled": False, "pass": True}
        results.append(
            {
                "epoch": int(c["epoch"]),
                "checkpoint_path": str(ckpt_path),
                "z_comp": comp["z_comp"],
                "gate_ok": comp["gate_ok"],
                "imbalance_guardrail": imbalance_guard,
                "cond_eval": cond_eval,
            }
        )
    valid = [
        r
        for r in results
        if r["gate_ok"] and r["z_comp"] is not None and bool(r["imbalance_guardrail"].get("pass", False))
    ]
    if not valid:
        return {"selected": None, "results": results}
    selected = min(valid, key=lambda r: float(r["z_comp"]))
    return {"selected": selected, "results": results}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train experiments2/exp001 CVAE pipeline.")
    p.add_argument(
        "--config",
        default="ML/autoencoder/experiments2/configs/exp001_base.json",
        help="Config JSON path.",
    )
    p.add_argument("--run-tag", default=None, help="Optional run tag override.")
    p.add_argument("--force-manifest", action="store_true", help="Rebuild manifest even if cached.")
    p.add_argument("--force-split", action="store_true", help="Rebuild frozen split even if cached.")
    p.add_argument("--force-stats", action="store_true", help="Rebuild normalization stats even if cached.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.run_tag:
        cfg["experiment"]["run_tag"] = args.run_tag

    set_seed(int(cfg["experiment"]["seed"]))
    device = resolve_device()

    manifest, split, norm_stats = prepare_exp001_artifacts(
        cfg=cfg,
        force_rebuild_manifest=bool(args.force_manifest),
        force_rebuild_split=bool(args.force_split),
        force_rebuild_stats=bool(args.force_stats),
    )

    run_paths = create_run_tree(
        run_root=cfg["artifacts"]["run_root"],
        run_tag=str(cfg["experiment"]["run_tag"]),
    )
    save_yaml_compatible(run_paths["run_dir"] / "config_resolved.yaml", cfg)
    logger = configure_logger(run_paths["run_dir"] / "train.log")
    logger.info("Run dir: %s", run_paths["run_dir"])
    logger.info("Device: %s", str(device))
    logger.info("Manifest rows: %d", len(manifest))
    logger.info(
        "Split sizes train/val/test/ood: %d/%d/%d/%d",
        len(split["train"]["indices"]),
        len(split["val"]["indices"]),
        len(split["test"]["indices"]),
        len(split["ood"]["indices"]),
    )

    imbalance_guard_cfg = dict(cfg.get("imbalance_guardrails", {}))
    imbalance_guard_enabled = bool(imbalance_guard_cfg.get("enabled", True))
    imbalance_reference_by_bin: Optional[Dict[str, Dict[str, float]]] = None
    if bool(cfg["train"]["use_weighted_sampler"]) and imbalance_guard_enabled:
        ref_path = imbalance_guard_cfg.get("reference_cond_eval_json")
        if not ref_path:
            raise ValueError(
                "use_weighted_sampler=true requires imbalance_guardrails.reference_cond_eval_json "
                "for D015 baseline comparison."
            )
        imbalance_reference_by_bin = _load_imbalance_reference_by_bin(ref_path)
        logger.info("Imbalance guardrails enabled with reference: %s", ref_path)
    else:
        logger.info(
            "Imbalance guardrails inactive in this run (use_weighted_sampler=%s, enabled=%s).",
            bool(cfg["train"]["use_weighted_sampler"]),
            imbalance_guard_enabled,
        )

    loaders = _build_dataloaders(cfg, manifest, split, norm_stats)
    split_bin_counts: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "val", "test", "ood"):
        ds_rows = loaders[split_name].dataset.rows  # type: ignore[attr-defined]
        bins = make_magnitude_bin_indices(ds_rows)
        split_bin_counts[split_name] = {k: len(v) for k, v in bins.items()}
    logger.info(
        "Magnitude-bin sample counts train/val/test/ood: %s / %s / %s / %s",
        split_bin_counts["train"],
        split_bin_counts["val"],
        split_bin_counts["test"],
        split_bin_counts["ood"],
    )
    if split_bin_counts["val"].get("ge5", 0) == 0:
        logger.warning(
            "VAL split has zero ge5 samples; ge5 is excluded from model selection and used as TEST holdout check only."
        )

    num_stations = len(load_json(cfg["data"]["station_list_file"]))

    model = CVAEComplexSTFT(
        numeric_cond_dim=len(cfg["conditions"]["numeric_feature_order"]),
        num_stations=num_stations,
        latent_dim=int(cfg["model"]["latent_dim"]),
        station_embedding_dim=int(cfg["model"]["station_embedding_dim"]),
        condition_hidden_dim=int(cfg["model"]["condition_hidden_dim"]),
        encoder_channels=tuple(cfg["model"]["encoder_channels"]),
        decoder_channels=tuple(cfg["model"]["decoder_channels"]),
        input_shape=(2, int(cfg["stft"]["target_freq_bins"]), int(cfg["stft"]["target_time_frames"])),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["train"]["scheduler_factor"]),
        patience=int(cfg["train"]["scheduler_patience"]),
    )
    early_stop = EarlyStopping(
        min_epochs=int(cfg["train"]["early_stop_min_epochs"]),
        patience=int(cfg["train"]["early_stop_patience"]),
        min_delta=float(cfg["train"]["early_stop_min_delta"]),
    )

    best_val_legacy = float("inf")
    best_val_epoch_legacy: Optional[int] = None
    best_val_fair = float("inf")
    best_val_epoch_fair: Optional[int] = None
    best_cond_score = float("inf")
    topk_cond_candidates: List[Dict[str, Any]] = []
    cond_history_means: List[Dict[str, float]] = []
    calibrator = CondgenCompositeCalibrator(cfg)

    history_rows: List[Dict[str, Any]] = []

    max_epochs = int(cfg["train"]["epochs"])
    eval_every = int(cfg["train"]["eval_every_epochs"])
    cond_subset = cfg["train"].get("cond_eval_subset_size")
    topk = int(cfg["train"]["save_topk_cond_candidates"])

    seed_bank_sweep = build_seed_bank(int(cfg["evaluation"]["seed_bank_base"]), int(cfg["evaluation"]["k_samples_sweep"]))
    seed_bank_final = build_seed_bank(int(cfg["evaluation"]["seed_bank_base"]), int(cfg["evaluation"]["k_samples_final"]))

    for epoch in range(1, max_epochs + 1):
        train_stats = _train_one_epoch(model, loaders["train"], optimizer, cfg, device, epoch)
        val_stats = _validate_one_epoch(model, loaders["val"], cfg, device, epoch)
        beta_max = float(cfg["loss"]["beta_max"])
        val_fair = float(val_stats["val_recon"]) + beta_max * float(val_stats["val_kl_fb"])
        scheduler.step(val_fair)

        row: Dict[str, Any] = {"epoch": epoch, **train_stats, **val_stats, "val_fair": val_fair}

        if val_stats["val_total"] < best_val_legacy:
            best_val_legacy = float(val_stats["val_total"])
            best_val_epoch_legacy = int(epoch)
            _save_checkpoint(
                path=run_paths["checkpoints"] / "best_val_loss.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_val_legacy,
                cfg=cfg,
                extra={"criterion": "best_val_loss_legacy"},
            )
            logger.info("[E%03d] new best_val_loss_legacy=%.6f", epoch, best_val_legacy)

        if val_fair < best_val_fair:
            best_val_fair = float(val_fair)
            best_val_epoch_fair = int(epoch)
            _save_checkpoint(
                path=run_paths["checkpoints"] / "best_val_fair.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_val_fair,
                cfg=cfg,
                extra={
                    "criterion": "best_val_fair",
                    "val_total": float(val_stats["val_total"]),
                    "val_recon": float(val_stats["val_recon"]),
                    "val_kl_fb": float(val_stats["val_kl_fb"]),
                    "beta_t": float(train_stats["beta_t"]),
                    "beta_max": beta_max,
                },
            )
            logger.info("[E%03d] new best_val_fair=%.6f", epoch, best_val_fair)

        if epoch % eval_every == 0:
            cond_eval = evaluate_condition_only(
                model=model,
                dataloader=loaders["val"],
                cfg=cfg,
                device=device,
                k_samples=int(cfg["evaluation"]["k_samples_sweep"]),
                seed_bank=seed_bank_sweep,
                max_samples=None if cond_subset is None else int(cond_subset),
            )
            cond_mean = dict(cond_eval["mean"])
            cond_history_means.append(cond_mean)
            comp = calibrator.score(cond_mean, cond_history_means)
            if imbalance_reference_by_bin is not None:
                imbalance_guard = _evaluate_imbalance_guardrails(
                    calibrator=calibrator,
                    candidate_by_bin_mean=cond_eval["by_bin_mean"],
                    reference_by_bin_mean=imbalance_reference_by_bin,
                    guard_cfg=imbalance_guard_cfg,
                )
            else:
                imbalance_guard = {"enabled": False, "pass": True}

            row["cond_eval_k"] = int(cond_eval["k_samples"])
            row["cond_z_comp"] = comp["z_comp"]
            row["cond_gate_ok"] = int(bool(comp["gate_ok"]))
            row["cond_calibrated"] = int(bool(comp["calibrated"]))
            row["cond_imbalance_guard_pass"] = int(bool(imbalance_guard.get("pass", False)))

            save_json(
                run_paths["metrics"] / f"cond_eval_epoch_{epoch:03d}.json",
                {
                    "epoch": epoch,
                    "cond_eval": cond_eval,
                    "composite": comp,
                    "imbalance_guardrail": imbalance_guard,
                },
            )

            if comp["gate_ok"] and comp["z_comp"] is not None and bool(imbalance_guard.get("pass", False)):
                score = float(comp["z_comp"])
                if score < best_cond_score:
                    best_cond_score = score
                    logger.info("[E%03d] provisional best_condgen z_comp=%.6f", epoch, score)

                ckpt_path = run_paths["tmp"] / f"cond_candidate_epoch_{epoch:03d}.pt"
                _save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_stats["val_total"],
                    cfg=cfg,
                    extra={"criterion": "condgen_composite", "z_comp": score},
                )
                topk_cond_candidates.append(
                    {
                        "epoch": epoch,
                        "z_comp": score,
                        "checkpoint_path": str(ckpt_path),
                    }
                )
                topk_cond_candidates = sorted(topk_cond_candidates, key=lambda x: float(x["z_comp"]))[:topk]

                keep_paths = {Path(c["checkpoint_path"]) for c in topk_cond_candidates}
                for stale in run_paths["tmp"].glob("cond_candidate_epoch_*.pt"):
                    if stale not in keep_paths:
                        stale.unlink(missing_ok=True)
            elif comp["gate_ok"] and comp["z_comp"] is not None:
                logger.info("[E%03d] condgen candidate rejected by imbalance guardrail.", epoch)

        history_rows.append(row)
        logger.info(
            "[E%03d] train=%.6f val=%.6f beta=%.4f",
            epoch,
            train_stats["train_total"],
            val_stats["val_total"],
            train_stats["beta_t"],
        )

        if early_stop.step(epoch_idx_1based=epoch, value=val_fair):
            logger.info("Early stop triggered at epoch %d", epoch)
            break

    # Final condgen rerank with K_final across top candidates.
    rerank = _rerank_condgen_candidates(
        candidates=topk_cond_candidates,
        model=model,
        val_loader=loaders["val"],
        cfg=cfg,
        device=device,
        calibrator=calibrator,
        cond_history_means=cond_history_means,
        imbalance_reference_by_bin=imbalance_reference_by_bin,
        imbalance_guard_cfg=imbalance_guard_cfg if imbalance_reference_by_bin is not None else None,
    )

    selected = rerank["selected"]
    if selected is not None:
        src_ckpt = Path(selected["checkpoint_path"])
        dst_ckpt = run_paths["checkpoints"] / "best_condgen_composite.pt"
        shutil.copy2(src_ckpt, dst_ckpt)
        selected_payload = torch.load(src_ckpt, map_location=device)
        model.load_state_dict(selected_payload["model_state_dict"])
        logger.info("Selected best_condgen checkpoint from epoch %s", selected["epoch"])
    elif imbalance_reference_by_bin is not None:
        logger.error(
            "No condgen candidate passed D015 imbalance guardrail in weighted run; "
            "stage-2 run is rejected."
        )
        raise RuntimeError(
            "D015 guardrail rejection: no candidate passed imbalance guardrail. "
            "Inspect metrics/cond_eval_epoch_*.json and rerank results."
        )
    else:
        # Fallback: copy fair-val checkpoint when no valid condgen candidate exists.
        src_ckpt = run_paths["checkpoints"] / "best_val_fair.pt"
        dst_ckpt = run_paths["checkpoints"] / "best_condgen_composite.pt"
        shutil.copy2(src_ckpt, dst_ckpt)
        fallback_payload = torch.load(src_ckpt, map_location=device)
        model.load_state_dict(fallback_payload["model_state_dict"])
        logger.warning("No valid condgen candidate; fallback to best_val_fair.pt")

    # Final reconstruction reports on val/test/ood for quick sanity.
    recon_val = evaluate_reconstruction(model, loaders["val"], cfg, device)
    recon_test = evaluate_reconstruction(model, loaders["test"], cfg, device)
    recon_ood = evaluate_reconstruction(model, loaders["ood"], cfg, device)

    # Q1-D policy: ge5 is a test-holdout diagnostic only (not a selection gate).
    cond_eval_test_final = evaluate_condition_only(
        model=model,
        dataloader=loaders["test"],
        cfg=cfg,
        device=device,
        k_samples=int(cfg["evaluation"]["k_samples_final"]),
        seed_bank=seed_bank_final,
        max_samples=None,
    )
    holdout_tail_test = {
        "policy": "test_holdout_only_not_used_for_selection",
        "bin_counts_test": split_bin_counts["test"],
        "ge5_available": bool(split_bin_counts["test"].get("ge5", 0) > 0),
        "ge5_metrics": dict(cond_eval_test_final.get("by_bin_mean", {}).get("ge5", {})),
    }
    save_json(
        run_paths["metrics"] / "cond_eval_holdout_test_final.json",
        {
            "selection_policy": "ge5 excluded from model selection (Q1-D), test-only holdout check",
            "split_bin_counts": split_bin_counts,
            "cond_eval_test_final": cond_eval_test_final,
        },
    )

    _write_history_csv(run_paths["metrics"] / "train_history.csv", history_rows)
    summary = {
        "best_val_loss": best_val_legacy,  # backward-compatible key
        "best_val_loss_legacy": best_val_legacy,
        "best_val_epoch_legacy": best_val_epoch_legacy,
        "best_val_fair": best_val_fair,
        "best_val_epoch_fair": best_val_epoch_fair,
        "best_val_fair_formula": "val_recon + beta_max * val_kl_fb",
        "best_condgen_score_sweep": None if not np.isfinite(best_cond_score) else best_cond_score,
        "num_epochs_finished": len(history_rows),
        "cond_history_len": len(cond_history_means),
        "cond_eval_selection": {
            "split": "val",
            "subset_size": cond_subset,
            "full_val_used": cond_subset is None,
            "val_bin_counts": split_bin_counts["val"],
            "test_bin_counts": split_bin_counts["test"],
            "ge5_policy": "test_holdout_only_not_used_for_selection",
        },
        "imbalance_guardrail": {
            "active": imbalance_reference_by_bin is not None,
            "reference_file": imbalance_guard_cfg.get("reference_cond_eval_json") if imbalance_reference_by_bin is not None else None,
            "thresholds_pct": {
                "tail_bin_min_improvement_pct": imbalance_guard_cfg.get("tail_bin_min_improvement_pct"),
                "all_bin_max_degradation_pct": imbalance_guard_cfg.get("all_bin_max_degradation_pct"),
                "mid_bin_max_degradation_pct": imbalance_guard_cfg.get("mid_bin_max_degradation_pct"),
            },
        },
        "rerank": rerank,
        "recon_eval": {
            "val": recon_val,
            "test": recon_test,
            "ood": recon_ood,
        },
        "holdout_tail_test": holdout_tail_test,
    }
    save_json(run_paths["metrics"] / "summary.json", summary)
    logger.info("Training completed. Summary written to %s", run_paths["metrics"] / "summary.json")


if __name__ == "__main__":
    main()
