from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from obspy import read

EARTH_RADIUS_KM = 6371.0
EARTH_KM_PER_DEG = math.pi * EARTH_RADIUS_KM / 180.0
PHASE_PREFIX_P = ("P",)
PHASE_PREFIX_S = ("S",)


@dataclass(frozen=True)
class WaveformHeadInfo:
    file_path: str
    event_id: str
    station_code: str
    start_time: str
    end_time: str
    start_offset_s: float
    end_offset_s: float
    trace_count: int
    sample_rate_hz: float
    num_samples: int
    component_signature: str
    waveform_contract_ok: bool
    waveform_contract_issues: tuple[str, ...]


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _parse_origin_time(date_text: str, time_text: str) -> datetime:
    return datetime.strptime(f"{date_text.strip()} {time_text.strip()}", "%Y.%m.%d %H:%M:%S.%f")


def extract_event_station(file_path: str | Path) -> tuple[str, str]:
    stem = Path(file_path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"unexpected waveform file name: {file_path}")
    return parts[0], parts[1]


def parse_event_catalog(catalog_path: str | Path) -> dict[str, dict[str, Any]]:
    catalog_path = Path(catalog_path)
    out: dict[str, dict[str, Any]] = {}
    with catalog_path.open("r", encoding="latin1", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            event_id = str(row["Deprem Kodu"]).strip()
            ml = _to_float(row.get("ML", ""))
            if not math.isfinite(ml):
                continue
            depth_km = _to_float(row.get("Der(km)", ""))
            latitude = _to_float(row.get("Enlem", ""))
            longitude = _to_float(row.get("Boylam", ""))
            origin_date = str(row.get("Olus tarihi", "")).strip()
            origin_clock = str(row.get("Olus zamani", "")).strip()
            origin_dt = _parse_origin_time(origin_date, origin_clock)
            out[event_id] = {
                "event_id": event_id,
                "origin_time": origin_dt.isoformat(),
                "origin_date": origin_date,
                "origin_clock": origin_clock,
                "magnitude": ml,
                "depth_km": depth_km,
                "latitude": latitude,
                "longitude": longitude,
                "place": str(row.get("Yer", "")).strip(),
            }
    return out


def inspect_waveform_head(
    file_path: str | Path,
    *,
    origin_time_iso: str,
    expected_components: Iterable[str],
    expected_sample_rate_hz: float,
) -> WaveformHeadInfo:
    file_path = Path(file_path)
    stream = read(str(file_path), headonly=True)
    event_id, station = extract_event_station(file_path)
    origin_dt = datetime.fromisoformat(origin_time_iso)
    starts = [trace.stats.starttime.datetime.replace(tzinfo=None) for trace in stream]
    ends = [trace.stats.endtime.datetime.replace(tzinfo=None) for trace in stream]
    sample_rates = {round(float(trace.stats.sampling_rate), 6) for trace in stream}
    npts_values = {int(trace.stats.npts) for trace in stream}
    components = sorted(trace.stats.channel[-1] for trace in stream)

    issues: list[str] = []
    if len(stream) != len(tuple(expected_components)):
        issues.append(f"trace_count={len(stream)}")
    if components != sorted(expected_components):
        issues.append(f"components={'/'.join(components)}")
    if len(sample_rates) != 1:
        issues.append(f"sample_rates={sorted(sample_rates)}")
    elif not math.isclose(next(iter(sample_rates)), float(expected_sample_rate_hz), rel_tol=0.0, abs_tol=1e-6):
        issues.append(f"sample_rate={next(iter(sample_rates))}")
    if len(npts_values) != 1:
        issues.append(f"npts={sorted(npts_values)}")

    start_dt = min(starts)
    end_dt = max(ends)
    start_offset_s = (start_dt - origin_dt).total_seconds()
    end_offset_s = (end_dt - origin_dt).total_seconds()

    return WaveformHeadInfo(
        file_path=str(file_path),
        event_id=event_id,
        station_code=station,
        start_time=start_dt.isoformat(),
        end_time=end_dt.isoformat(),
        start_offset_s=start_offset_s,
        end_offset_s=end_offset_s,
        trace_count=len(stream),
        sample_rate_hz=next(iter(sample_rates)) if len(sample_rates) == 1 else float("nan"),
        num_samples=next(iter(npts_values)) if len(npts_values) == 1 else -1,
        component_signature="/".join(components),
        waveform_contract_ok=not issues,
        waveform_contract_issues=tuple(issues),
    )


def parse_phase_picks_targeted(
    phase_pick_dir: str | Path,
    *,
    needed_pairs: set[tuple[str, str]],
    needed_events: set[str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, float]]:
    phase_pick_dir = Path(phase_pick_dir)
    pair_out: dict[tuple[str, str], dict[str, Any]] = {}
    event_gap_out: dict[str, float] = {}
    current_event: str | None = None

    for file_path in sorted(phase_pick_dir.glob("*.txt")):
        with file_path.open("r", encoding="latin1", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.startswith("EVENT "):
                    parts = line.strip().split()
                    current_event = parts[1] if len(parts) >= 2 else None
                    continue
                if current_event is None or current_event not in needed_events:
                    continue
                if not line.strip():
                    continue
                if line.startswith(("BEGIN", "MSG", "DATA", "Date", "Sta")):
                    continue

                toks = line.split()
                if line[:4].isdigit() and len(toks) >= 8:
                    gap = _to_float(toks[7])
                    if math.isfinite(gap):
                        event_gap_out[current_event] = gap
                    continue

                station = line[:5].strip()
                if not station:
                    continue
                key = (current_event, station)
                if key not in needed_pairs:
                    continue
                if len(toks) < 4:
                    continue
                dist_deg = _to_float(toks[1])
                azimuth_deg = _to_float(toks[2])
                if not math.isfinite(dist_deg) or not math.isfinite(azimuth_deg):
                    continue
                phase_tok = None
                for tok in toks[3:8]:
                    upper = tok.upper()
                    if upper.startswith(PHASE_PREFIX_P) or upper.startswith(PHASE_PREFIX_S):
                        phase_tok = upper
                        break
                if phase_tok is None:
                    continue
                record = pair_out.setdefault(
                    key,
                    {
                        "dist_deg": float(dist_deg),
                        "azimuth_deg": float(azimuth_deg),
                        "has_p": False,
                        "has_s": False,
                    },
                )
                record["dist_deg"] = float(dist_deg)
                record["azimuth_deg"] = float(azimuth_deg)
                if phase_tok.startswith(PHASE_PREFIX_P):
                    record["has_p"] = True
                if phase_tok.startswith(PHASE_PREFIX_S):
                    record["has_s"] = True
    return pair_out, event_gap_out


def velocity_from_1d_model(depth_km: float, model_cfg: Mapping[str, Any]) -> tuple[float, float]:
    depths = [float(v) for v in model_cfg["Depths"]]
    vp_vals = [float(v) for v in model_cfg["Vp"]]
    vs_vals = [float(v) for v in model_cfg["Vs"]]
    if not (len(depths) == len(vp_vals) == len(vs_vals)) or not depths:
        raise ValueError("velocity_model_1d knot profile has invalid lengths")

    depth_unit = str(model_cfg.get("depth_unit", "")).lower().strip() or "km"
    velocity_unit = str(model_cfg.get("velocity_unit", "")).lower().strip() or "km/s"

    if depth_unit in {"m", "meter", "meters"}:
        depth_nodes_km = [abs(value) / 1000.0 for value in depths]
    elif depth_unit in {"km", "kilometer", "kilometers"}:
        depth_nodes_km = [abs(value) for value in depths]
    else:
        raise ValueError(f"unsupported depth_unit: {depth_unit}")

    if velocity_unit in {"m/s", "mps"}:
        vp_nodes = [value / 1000.0 for value in vp_vals]
        vs_nodes = [value / 1000.0 for value in vs_vals]
    elif velocity_unit in {"km/s", "kmps"}:
        vp_nodes = list(vp_vals)
        vs_nodes = list(vs_vals)
    else:
        raise ValueError(f"unsupported velocity_unit: {velocity_unit}")

    nodes = sorted(zip(depth_nodes_km, vp_nodes, vs_nodes), key=lambda item: item[0])
    node_depths = [item[0] for item in nodes]
    node_vp = [item[1] for item in nodes]
    node_vs = [item[2] for item in nodes]

    if depth_km <= node_depths[0]:
        return node_vp[0], node_vs[0]
    if depth_km >= node_depths[-1]:
        return node_vp[-1], node_vs[-1]

    for index in range(1, len(node_depths)):
        z0, z1 = node_depths[index - 1], node_depths[index]
        if z0 <= depth_km <= z1:
            frac = 0.0 if math.isclose(z0, z1) else (depth_km - z0) / (z1 - z0)
            vp = node_vp[index - 1] + frac * (node_vp[index] - node_vp[index - 1])
            vs = node_vs[index - 1] + frac * (node_vs[index] - node_vs[index - 1])
            return vp, vs

    return node_vp[-1], node_vs[-1]


def compute_travel_times(
    *,
    repi_km: float,
    depth_km: float,
    pre_origin_sec: float,
    velocity_model_cfg: Mapping[str, Any],
) -> dict[str, float]:
    vp_km_s, vs_km_s = velocity_from_1d_model(depth_km=depth_km, model_cfg=velocity_model_cfg)
    hypocentral_distance_km = math.sqrt(max(repi_km, 0.0) ** 2 + max(depth_km, 0.0) ** 2)
    t_p_origin = hypocentral_distance_km / max(vp_km_s, 1e-6)
    t_s_origin = hypocentral_distance_km / max(vs_km_s, 1e-6)
    t_p_ref = t_p_origin + float(pre_origin_sec)
    t_s_ref = t_s_origin + float(pre_origin_sec)
    return {
        "vp_km_s": vp_km_s,
        "vs_km_s": vs_km_s,
        "hypocentral_distance_km": hypocentral_distance_km,
        "tP_ref_s": t_p_ref,
        "tS_ref_s": t_s_ref,
        "dtPS_ref_s": t_s_ref - t_p_ref,
    }


def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dump_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
