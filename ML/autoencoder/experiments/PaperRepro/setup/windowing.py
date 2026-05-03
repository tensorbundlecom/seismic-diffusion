from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
from obspy import read


@dataclass(frozen=True)
class WindowExtractionInfo:
    file_path: str
    target_start_time: str
    target_end_time: str
    requires_left_pad: bool
    requires_right_pad: bool
    filled_samples: int
    total_samples: int


def load_origin_window(
    file_path: str | Path,
    *,
    origin_time_iso: str,
    num_samples: int,
    sample_rate_hz: float,
    components: Iterable[str],
    pre_origin_sec: float = 0.0,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, WindowExtractionInfo]:
    stream = read(str(file_path))
    target_components = tuple(components)
    traces_by_component = {trace.stats.channel[-1]: trace for trace in stream}
    missing = [component for component in target_components if component not in traces_by_component]
    if missing:
        raise ValueError(f"missing components in {file_path}: {missing}")

    origin_dt = datetime.fromisoformat(origin_time_iso)
    target_start = origin_dt - timedelta(seconds=pre_origin_sec)
    target_end = target_start + timedelta(seconds=(num_samples - 1) / sample_rate_hz)

    window = np.full((len(target_components), num_samples), float(padding_value), dtype=np.float32)
    total_filled = 0
    requires_left_pad = False
    requires_right_pad = False

    for channel_index, component in enumerate(target_components):
        trace = traces_by_component[component]
        trace_start = trace.stats.starttime.datetime.replace(tzinfo=None)
        trace_end = trace.stats.endtime.datetime.replace(tzinfo=None)
        trace_fs = float(trace.stats.sampling_rate)
        if abs(trace_fs - sample_rate_hz) > 1.0e-6:
            raise ValueError(f"sample rate mismatch for {file_path}: {trace_fs} vs {sample_rate_hz}")

        trace_data = np.asarray(trace.data, dtype=np.float32)
        start_offset_samples = int(round((trace_start - target_start).total_seconds() * sample_rate_hz))
        end_offset_samples = start_offset_samples + len(trace_data)

        dst_start = max(0, start_offset_samples)
        dst_end = min(num_samples, end_offset_samples)
        src_start = max(0, -start_offset_samples)
        src_end = src_start + max(0, dst_end - dst_start)

        if dst_start > 0:
            requires_left_pad = True
        if dst_end < num_samples:
            requires_right_pad = True
        if dst_end > dst_start:
            window[channel_index, dst_start:dst_end] = trace_data[src_start:src_end]
            total_filled += int(dst_end - dst_start)

        if trace_start > target_start:
            requires_left_pad = True
        if trace_end < target_end:
            requires_right_pad = True

    info = WindowExtractionInfo(
        file_path=str(file_path),
        target_start_time=target_start.isoformat(),
        target_end_time=target_end.isoformat(),
        requires_left_pad=requires_left_pad,
        requires_right_pad=requires_right_pad,
        filled_samples=total_filled,
        total_samples=len(target_components) * num_samples,
    )
    return window, info
