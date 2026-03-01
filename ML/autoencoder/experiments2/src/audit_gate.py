"""Audit and calibrate two-stage D013 gate thresholds against labels/proxy quality."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass
class EpochRecord:
    run_name: str
    run_dir: str
    epoch: int
    metrics: Dict[str, float]


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _load_epoch_records(run_dirs: Sequence[str]) -> List[EpochRecord]:
    out: List[EpochRecord] = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        run_name = run_path.name
        files = sorted(glob.glob(str(run_path / "metrics" / "cond_eval_epoch_*.json")))
        for fp in files:
            payload = json.loads(Path(fp).read_text(encoding="utf-8"))
            epoch = int(payload.get("epoch", -1))
            mean = payload.get("cond_eval", {}).get("mean", {})
            if epoch <= 0 or not mean:
                continue
            out.append(
                EpochRecord(
                    run_name=run_name,
                    run_dir=str(run_path),
                    epoch=epoch,
                    metrics={k: float(v) for k, v in mean.items() if isinstance(v, (int, float))},
                )
            )
    return out


def _label_key_candidates(run_name: str, run_dir: str, epoch: int) -> List[Tuple[str, int]]:
    return [
        (run_name, epoch),
        (run_dir, epoch),
    ]


def _load_manual_labels(labels_csv: str) -> Dict[Tuple[str, int], int]:
    """
    CSV columns:
    - run
    - epoch
    - label   (good|bad|1|0|true|false)
    """
    labels: Dict[Tuple[str, int], int] = {}
    with Path(labels_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = str(row.get("run", "")).strip()
            if not run:
                continue
            epoch = int(row.get("epoch", "0"))
            raw = str(row.get("label", "")).strip().lower()
            if raw in ("good", "1", "true", "pass"):
                y = 1
            elif raw in ("bad", "0", "false", "fail"):
                y = 0
            else:
                continue
            labels[(run, epoch)] = y
    return labels


def _label_from_proxy(rec: EpochRecord, proxy_cfg: Mapping[str, float]) -> int:
    """
    Returns 1=good, 0=bad.
    Proxy is intentionally based on quality metrics outside D013 gate checks.
    """
    m = rec.metrics
    bad = False
    bad |= float(m.get("xcorr_max", 0.0)) < float(proxy_cfg["min_xcorr_max"])
    bad |= float(m.get("envelope_corr", 0.0)) < float(proxy_cfg["min_envelope_corr"])
    bad |= float(m.get("mr_lsd", 1e9)) > float(proxy_cfg["max_mr_lsd"])
    bad |= float(m.get("onset_mae_dtps_s", 1e9)) > float(proxy_cfg["max_onset_mae_dtps_s"])
    return 0 if bad else 1


def _gate_pass(metrics: Mapping[str, float], th: Mapping[str, float]) -> bool:
    ok = (
        float(metrics.get("onset_evaluable_rate", 0.0)) >= float(th["min_onset_evaluable_rate"])
        and float(metrics.get("onset_failure_rate_p", 1.0)) <= float(th["max_onset_failure_p"])
        and float(metrics.get("onset_failure_rate_s", 1.0)) <= float(th["max_onset_failure_s"])
        and float(metrics.get("abs_xcorr_lag_s", 1e9)) <= float(th["max_abs_xcorr_lag_s"])
    )
    if not ok:
        return False
    if not bool(th.get("quality_enabled", False)):
        return True
    if "min_xcorr_max" in th and float(metrics.get("xcorr_max", -1.0)) < float(th["min_xcorr_max"]):
        return False
    if "min_envelope_corr" in th and float(metrics.get("envelope_corr", -1.0)) < float(th["min_envelope_corr"]):
        return False
    if "max_mr_lsd" in th and float(metrics.get("mr_lsd", 1e9)) > float(th["max_mr_lsd"]):
        return False
    if "max_onset_mae_dtps_s" in th and float(metrics.get("onset_mae_dtps_s", 1e9)) > float(th["max_onset_mae_dtps_s"]):
        return False
    return True


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _evaluate_thresholds(
    records: Sequence[EpochRecord],
    labels: Mapping[Tuple[str, int], int],
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    n_labeled = 0
    for rec in records:
        y: Optional[int] = None
        for key in _label_key_candidates(rec.run_name, rec.run_dir, rec.epoch):
            if key in labels:
                y = int(labels[key])
                break
        if y is None:
            continue
        n_labeled += 1
        pred_accept = 1 if _gate_pass(rec.metrics, thresholds) else 0
        if pred_accept == 1 and y == 1:
            tp += 1
        elif pred_accept == 1 and y == 0:
            fp += 1
        elif pred_accept == 0 and y == 0:
            tn += 1
        else:
            fn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    bad_accept_rate = _safe_div(fp, fp + tn)  # among bad epochs, how many were accepted
    good_reject_rate = _safe_div(fn, tp + fn)  # among good epochs, how many were rejected
    pass_rate = _safe_div(tp + fp, n_labeled)

    return {
        "n_labeled": n_labeled,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_accept_is_good": precision,
        "recall_good_accepted": recall,
        "f1": f1,
        "bad_accept_rate": bad_accept_rate,
        "good_reject_rate": good_reject_rate,
        "pass_rate": pass_rate,
    }


def _sweep_thresholds(
    records: Sequence[EpochRecord],
    labels: Mapping[Tuple[str, int], int],
    grids: Mapping[str, Sequence[float]],
    quality_enabled: bool,
    topk: int,
    min_pass_rate: float,
    max_pass_rate: float,
    min_recall_good: float,
) -> List[Dict[str, Any]]:
    rows_feasible: List[Dict[str, Any]] = []
    rows_all: List[Dict[str, Any]] = []
    for min_eval in grids["min_onset_evaluable_rate"]:
        for max_fp in grids["max_onset_failure_p"]:
            for max_fs in grids["max_onset_failure_s"]:
                for max_lag in grids["max_abs_xcorr_lag_s"]:
                    for min_xc in grids["min_xcorr_max"]:
                        for min_env in grids["min_envelope_corr"]:
                            for max_mr in grids["max_mr_lsd"]:
                                for max_dtps in grids["max_onset_mae_dtps_s"]:
                                    th = {
                                        "min_onset_evaluable_rate": float(min_eval),
                                        "max_onset_failure_p": float(max_fp),
                                        "max_onset_failure_s": float(max_fs),
                                        "max_abs_xcorr_lag_s": float(max_lag),
                                        "quality_enabled": bool(quality_enabled),
                                    }
                                    if bool(quality_enabled):
                                        th.update(
                                            {
                                                "min_xcorr_max": float(min_xc),
                                                "min_envelope_corr": float(min_env),
                                                "max_mr_lsd": float(max_mr),
                                                "max_onset_mae_dtps_s": float(max_dtps),
                                            }
                                        )
                                    perf = _evaluate_thresholds(records, labels, th)
                                    row = {
                                        "thresholds": th,
                                        "performance": perf,
                                    }
                                    rows_all.append(row)
                                    feasible = (
                                        (perf["pass_rate"] >= float(min_pass_rate))
                                        and (perf["pass_rate"] <= float(max_pass_rate))
                                        and (perf["recall_good_accepted"] >= float(min_recall_good))
                                    )
                                    if feasible:
                                        rows_feasible.append(
                                            {
                                                **row,
                                                "_rank": (
                                                    perf["bad_accept_rate"],
                                                    -perf["f1"],
                                                    perf["good_reject_rate"],
                                                    abs(perf["pass_rate"] - 0.35),
                                                ),
                                            }
                                        )

    if rows_feasible:
        rows = sorted(rows_feasible, key=lambda x: x["_rank"])[: int(topk)]
        for r in rows:
            r.pop("_rank", None)
        return rows

    # Fallback ranking if no threshold set satisfies practical constraints.
    rows_fallback = []
    for row in rows_all:
        perf = row["performance"]
        rows_fallback.append(
            {
                **row,
                "_rank": (
                    abs(perf["pass_rate"] - 0.35),
                    -perf["f1"],
                    perf["bad_accept_rate"],
                    perf["good_reject_rate"],
                ),
            }
        )
    rows = sorted(rows_fallback, key=lambda x: x["_rank"])[: int(topk)]
    for r in rows:
        r.pop("_rank", None)
    return rows


def _build_disagreements(
    records: Sequence[EpochRecord],
    labels: Mapping[Tuple[str, int], int],
    thresholds: Mapping[str, float],
    limit: int = 200,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        y: Optional[int] = None
        for key in _label_key_candidates(rec.run_name, rec.run_dir, rec.epoch):
            if key in labels:
                y = int(labels[key])
                break
        if y is None:
            continue
        pred = 1 if _gate_pass(rec.metrics, thresholds) else 0
        if pred == y:
            continue
        out.append(
            {
                "run": rec.run_name,
                "epoch": rec.epoch,
                "label": "good" if y == 1 else "bad",
                "gate_accept": bool(pred == 1),
                "metrics": {
                    "onset_evaluable_rate": rec.metrics.get("onset_evaluable_rate"),
                    "onset_failure_rate_p": rec.metrics.get("onset_failure_rate_p"),
                    "onset_failure_rate_s": rec.metrics.get("onset_failure_rate_s"),
                    "abs_xcorr_lag_s": rec.metrics.get("abs_xcorr_lag_s"),
                    "xcorr_max": rec.metrics.get("xcorr_max"),
                    "envelope_corr": rec.metrics.get("envelope_corr"),
                    "mr_lsd": rec.metrics.get("mr_lsd"),
                    "onset_mae_dtps_s": rec.metrics.get("onset_mae_dtps_s"),
                },
            }
        )
        if len(out) >= int(limit):
            break
    return out


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    cur = payload["current"]
    rec = payload["recommended"]
    lines = [
        "# Condgen Gate Audit",
        "",
        f"- Label source: `{payload['label_source']}`",
        f"- Runs: `{len(payload['run_dirs'])}`",
        f"- Epoch records: `{payload['num_records']}`",
        f"- Labeled records used: `{cur['performance']['n_labeled']}`",
        "",
        "## Current Gate",
        "",
        f"- Thresholds: `{cur['thresholds']}`",
        f"- bad_accept_rate: `{cur['performance']['bad_accept_rate']:.4f}`",
        f"- good_reject_rate: `{cur['performance']['good_reject_rate']:.4f}`",
        f"- f1: `{cur['performance']['f1']:.4f}`",
        f"- pass_rate: `{cur['performance']['pass_rate']:.4f}`",
        "",
        "## Recommended Gate (Sweep Top-1)",
        "",
        f"- Thresholds: `{rec['thresholds']}`",
        f"- bad_accept_rate: `{rec['performance']['bad_accept_rate']:.4f}`",
        f"- good_reject_rate: `{rec['performance']['good_reject_rate']:.4f}`",
        f"- f1: `{rec['performance']['f1']:.4f}`",
        f"- pass_rate: `{rec['performance']['pass_rate']:.4f}`",
        "",
        "## Notes",
        "",
        "- Proxy-label mode is provisional; finalize thresholds with manual labels.",
        "- Review disagreement list before freezing gate thresholds.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit D013 condgen pre-gate thresholds.")
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories (e.g., runs/exp001/run_... ).",
    )
    p.add_argument(
        "--labels-csv",
        default=None,
        help="Optional manual labels CSV with columns: run,epoch,label.",
    )
    p.add_argument(
        "--out-json",
        required=True,
        help="Output audit JSON path.",
    )
    p.add_argument(
        "--out-md",
        default=None,
        help="Optional markdown summary path.",
    )
    p.add_argument("--min-onset-evaluable-rate-grid", default="0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--max-onset-failure-p-grid", default="0.05,0.10,0.20,0.30")
    p.add_argument("--max-onset-failure-s-grid", default="0.15,0.20,0.25,0.30,0.35")
    p.add_argument("--max-abs-xcorr-lag-s-grid", default="1.5,2.0,2.5,3.0,4.0")
    p.add_argument("--min-xcorr-max-grid", default="0.74,0.76,0.78")
    p.add_argument("--min-envelope-corr-grid", default="0.69,0.71,0.73")
    p.add_argument("--max-mr-lsd-grid", default="0.0195,0.0200,0.0205")
    p.add_argument("--max-onset-mae-dtps-s-grid", default="1.9,2.0,2.1")
    p.add_argument("--quality-enabled", action="store_true", help="Enable stage-2 quality gate in sweep/current gate.")
    p.add_argument("--current-min-onset-evaluable-rate", type=float, default=0.60)
    p.add_argument("--current-max-onset-failure-p", type=float, default=0.05)
    p.add_argument("--current-max-onset-failure-s", type=float, default=0.35)
    p.add_argument("--current-max-abs-xcorr-lag-s", type=float, default=3.0)
    p.add_argument("--current-min-xcorr-max", type=float, default=0.74)
    p.add_argument("--current-min-envelope-corr", type=float, default=0.69)
    p.add_argument("--current-max-mr-lsd", type=float, default=0.0195)
    p.add_argument("--current-max-onset-mae-dtps-s", type=float, default=2.0)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--proxy-min-xcorr-max", type=float, default=0.70)
    p.add_argument("--proxy-min-envelope-corr", type=float, default=0.65)
    p.add_argument("--proxy-max-mr-lsd", type=float, default=0.022)
    p.add_argument("--proxy-max-onset-mae-dtps-s", type=float, default=2.0)
    p.add_argument("--min-pass-rate", type=float, default=0.20)
    p.add_argument("--max-pass-rate", type=float, default=0.80)
    p.add_argument("--min-recall-good", type=float, default=0.40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_epoch_records(args.run_dirs)
    if not records:
        raise RuntimeError("No cond_eval_epoch_*.json records found in provided run directories.")

    current_thresholds = {
        "min_onset_evaluable_rate": float(args.current_min_onset_evaluable_rate),
        "max_onset_failure_p": float(args.current_max_onset_failure_p),
        "max_onset_failure_s": float(args.current_max_onset_failure_s),
        "max_abs_xcorr_lag_s": float(args.current_max_abs_xcorr_lag_s),
        "quality_enabled": bool(args.quality_enabled),
    }
    if bool(args.quality_enabled):
        current_thresholds.update(
            {
                "min_xcorr_max": float(args.current_min_xcorr_max),
                "min_envelope_corr": float(args.current_min_envelope_corr),
                "max_mr_lsd": float(args.current_max_mr_lsd),
                "max_onset_mae_dtps_s": float(args.current_max_onset_mae_dtps_s),
            }
        )

    if args.labels_csv:
        labels = _load_manual_labels(args.labels_csv)
        label_source = "manual"
    else:
        proxy_cfg = {
            "min_xcorr_max": float(args.proxy_min_xcorr_max),
            "min_envelope_corr": float(args.proxy_min_envelope_corr),
            "max_mr_lsd": float(args.proxy_max_mr_lsd),
            "max_onset_mae_dtps_s": float(args.proxy_max_onset_mae_dtps_s),
        }
        labels = {}
        for rec in records:
            labels[(rec.run_name, rec.epoch)] = _label_from_proxy(rec, proxy_cfg)
        label_source = "proxy"

    grids = {
        "min_onset_evaluable_rate": _parse_float_list(args.min_onset_evaluable_rate_grid),
        "max_onset_failure_p": _parse_float_list(args.max_onset_failure_p_grid),
        "max_onset_failure_s": _parse_float_list(args.max_onset_failure_s_grid),
        "max_abs_xcorr_lag_s": _parse_float_list(args.max_abs_xcorr_lag_s_grid),
        "min_xcorr_max": _parse_float_list(args.min_xcorr_max_grid),
        "min_envelope_corr": _parse_float_list(args.min_envelope_corr_grid),
        "max_mr_lsd": _parse_float_list(args.max_mr_lsd_grid),
        "max_onset_mae_dtps_s": _parse_float_list(args.max_onset_mae_dtps_s_grid),
    }

    current_perf = _evaluate_thresholds(records, labels, current_thresholds)
    sweep = _sweep_thresholds(
        records=records,
        labels=labels,
        grids=grids,
        quality_enabled=bool(args.quality_enabled),
        topk=int(args.topk),
        min_pass_rate=float(args.min_pass_rate),
        max_pass_rate=float(args.max_pass_rate),
        min_recall_good=float(args.min_recall_good),
    )
    recommended = sweep[0] if sweep else {"thresholds": current_thresholds, "performance": current_perf}
    disagreements_current = _build_disagreements(records, labels, current_thresholds)
    disagreements_recommended = _build_disagreements(records, labels, recommended["thresholds"])

    payload: Dict[str, Any] = {
        "label_source": label_source,
        "run_dirs": [str(Path(p)) for p in args.run_dirs],
        "num_records": len(records),
        "current": {
            "thresholds": current_thresholds,
            "performance": current_perf,
        },
        "recommended": recommended,
        "sweep_topk": sweep,
        "disagreements": {
            "current": disagreements_current,
            "recommended": disagreements_recommended,
        },
        "notes": [
            "Proxy-label mode is provisional; freeze thresholds only after manual label audit.",
            "Use disagreement lists to build owner review set.",
        ],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[INFO] Gate audit JSON written: {out_json}")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, payload)
        print(f"[INFO] Gate audit markdown written: {out_md}")


if __name__ == "__main__":
    main()
