"""
Shaking duration via cumulative Arias Intensity (GWM paper, Fig. 6).

  a) Cumulative Arias Intensity (cAI) curves for one example record (black)
     and --n_realizations fresh diffusion synthetics generated with the same
     conditioning vector (colored), with markers at the 5% and 95% cAI
     thresholds. This panel re-samples the diffusion model, so `compute`
     needs the checkpoints (and realistically a GPU).
  b) Significant duration D5-95 (time between 5% and 95% of the final cAI)
     versus magnitude for every test-split record: real data vs the cached
     one-synthetic-per-record waveforms from evaluate_first_order.py, with
     per-magnitude-bin mean +/- std overlays.

The paper's figure contains no GMM; published duration models exist, and an
optional overlay is available through a registry so the (always tentative)
model choice stays a one-flag swap: --duration_gmm afshari_stewart16 adds
Afshari & Stewart (2016) medians from openquake.hazardlib, binned like the
data. Caveat: below M~5.35 its source term saturates, so at this dataset's
magnitudes it predicts a nearly magnitude-independent, path-dominated
duration - treat the overlay as a reference line, not a benchmark.

Durations are measured on response-deconvolved acceleration (E component),
identically for real and synthetic waveforms. All outputs are cached in
eval/shake_duration/ and `compute` is resumable; `plot` only needs the cache.

Run from the project root:
    python eval/evaluate_shake_duration.py compute [--limit N] [--n_realizations 100] [--example_mag 3.0]
    python eval/evaluate_shake_duration.py plot    [--mag_bin 0.25] [--duration_gmm afshari_stewart16]
    python eval/evaluate_shake_duration.py all     [...]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_peak_amplitudes import (  # noqa: E402
    CHANNEL_NAMES, DIFF_DIR, FS, ROOT, TARGET_SAMPLES,
    GWM_COLOR, GMM_COLOR, INK, MUTED, GRID,
    _crop_pad, _deconvolve, find_synth_cache,
)

OUT_DIR = ROOT / "eval" / "shake_duration"
REAL_COLOR = "#63615c"
GRAVITY = 9.80665
CAI_DECIMATE = 10  # stored example-curve resolution: FS/10


# ── Arias intensity ───────────────────────────────────────────────────────────
def arias_curve(acc: np.ndarray) -> np.ndarray:
    """Cumulative Arias Intensity [m/s] of an acceleration trace [m/s^2]."""
    return np.cumsum(np.asarray(acc, dtype=np.float64) ** 2) * np.pi / (2 * GRAVITY * FS)


def d595(cai: np.ndarray):
    """(duration, t5, t95): time between 5% and 95% of the final cAI."""
    total = float(cai[-1])
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan, np.nan
    t5 = float(np.searchsorted(cai, 0.05 * total)) / FS
    t95 = float(np.searchsorted(cai, 0.95 * total)) / FS
    return t95 - t5, t5, t95


# ── Workers (panel b) ─────────────────────────────────────────────────────────
def _duration_real(args):
    """Worker: real mseed -> D5-95 of the deconvolved E-component acceleration."""
    path_str, station, channel_type, event_id, channel_idx = args
    try:
        from obspy import read as obspy_read

        stream = obspy_read(path_str)
        if len(stream) != 3:
            return np.nan
        stream.sort(keys=["channel"])
        trace = stream[channel_idx]
        if abs(trace.stats.sampling_rate - FS) > 1e-6:
            trace.resample(FS)
        data = _crop_pad(trace.data.astype(np.float64))
        code = f"{channel_type}{CHANNEL_NAMES[channel_idx]}"
        acc = _deconvolve(data, station, code, event_id, "ACC")
        if acc is None:
            return np.nan
        return d595(arias_curve(acc))[0]
    except Exception:
        return np.nan


def _duration_synth(args):
    """Worker: cached counts-domain synthetic -> D5-95."""
    wave, station, channel_code, event_id = args
    try:
        acc = _deconvolve(np.asarray(wave, dtype=np.float64),
                          station, channel_code, event_id, "ACC")
        if acc is None:
            return np.nan
        return d595(arias_curve(acc))[0]
    except Exception:
        return np.nan


# ── Duration GMMs (optional overlay, swappable) ───────────────────────────────
def _predict_afshari_stewart16(mag, r_hyp, vs30):
    """Afshari & Stewart (2016) median D5-95 [s]; R_rup ~ R_hyp, rake 0,
    z1pt0 from the CY14 California Vs30 relation."""
    from openquake.hazardlib.contexts import simple_cmaker
    from openquake.hazardlib.gsim import get_available_gsims

    gsim = get_available_gsims()["AfshariStewart2016"]()
    cmaker = simple_cmaker([gsim], ["RSD595"])
    ctx = cmaker.new_ctx(mag.shape[0])
    ctx["mag"] = mag
    ctx["rrup"] = r_hyp
    ctx["vs30"] = vs30
    ctx["rake"] = 0.0
    ctx["z1pt0"] = np.exp(-7.15 / 4.0 * np.log(
        (vs30 ** 4 + 570.94 ** 4) / (1360.0 ** 4 + 570.94 ** 4)))
    mean, _, _, _ = cmaker.get_mean_stds([ctx])
    return np.exp(mean[0, 0])


DURATION_GMM_REGISTRY = {
    "afshari_stewart16": {
        "label": "Afshari & Stewart (2016)",
        "predict": _predict_afshari_stewart16,
    },
}


# ── Cache ─────────────────────────────────────────────────────────────────────
def cache_path(synth_cache: Path) -> Path:
    return OUT_DIR / f"durations_{synth_cache.stem.removeprefix('cache_')}.npz"


def save_cache(path: Path, cache: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez(tmp, **cache)
    tmp.replace(path)


# ── compute stage ─────────────────────────────────────────────────────────────
def run_compute(args):
    from obspy.geodetics import gps2dist_azimuth

    metadatas = json.load(open(DIFF_DIR / "embeddings" / "metadata.json"))
    station_locations = json.load(open(DIFF_DIR / "embeddings" / "station_locations.json"))
    station_vs30 = json.load(open(DIFF_DIR / "embeddings" / "station_vs30.json"))

    synth_cache_path = Path(args.cache) if args.cache else find_synth_cache()
    with np.load(synth_cache_path) as sc:
        synth_indices = sc["indices"].astype(int).tolist()
        wave_synth = sc["wave_synth"]
    channel = synth_cache_path.stem.rsplit("_ch", 1)[-1].split("_")[0]
    channel_idx = CHANNEL_NAMES.index(channel)
    print(f"[eval] synthetics: {synth_cache_path.name} "
          f"({len(synth_indices)} samples, channel {channel})")

    indices = list(synth_indices)
    if args.limit and args.limit < len(indices):
        picks = np.linspace(0, len(indices) - 1, num=args.limit, dtype=int)
        indices = [indices[i] for i in picks]

    path = cache_path(synth_cache_path)
    n = len(indices)
    cache = None
    if path.exists():
        old = dict(np.load(path))
        if old["indices"].tolist() == indices:
            cache = old
        else:
            print(f"[eval] {path.name} covers a different record set; rebuilding.")
    if cache is None:
        cache = {
            "indices": np.asarray(indices, dtype=np.int64),
            "mag": np.full(n, np.nan), "r_hyp": np.full(n, np.nan),
            "vs30": np.full(n, np.nan),
            "dur_real": np.full(n, np.nan), "dur_synth": np.full(n, np.nan),
            "done_real": np.zeros(n, dtype=bool),
            "done_synth": np.zeros(n, dtype=bool),
        }

    for j, idx in enumerate(indices):
        m = metadatas[idx]
        sta = station_locations[m["station_name"]]
        dist_m, _, _ = gps2dist_azimuth(
            m["latitude"], m["longitude"], sta["latitude"], sta["longitude"]
        )
        cache["mag"][j] = float(m["magnitude"])
        cache["r_hyp"][j] = float(np.hypot(dist_m / 1000.0, float(m["depth"])))
        cache["vs30"][j] = float(station_vs30[m["station_name"]])

    def resolve(m):
        p = Path(m["file_path"])
        return str(p if p.is_absolute() else (DIFF_DIR / p).resolve())

    def run_pool(todo, jobs, done_key, out_key, worker, label):
        t0, n_done = time.time(), 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for start in range(0, len(todo), args.chunk):
                rows = todo[start:start + args.chunk]
                for j, out in zip(rows, pool.map(worker, [jobs[j] for j in rows],
                                                 chunksize=8)):
                    cache[out_key][j] = out
                    cache[done_key][j] = True
                n_done += len(rows)
                save_cache(path, cache)
                rate = n_done / max(time.time() - t0, 1e-9)
                eta_min = (len(todo) - n_done) / max(rate, 1e-9) / 60.0
                print(f"[eval] {label} {n_done}/{len(todo)} "
                      f"({rate:.1f} rec/s, ETA {eta_min:.0f} min)")

    todo = [j for j in range(n) if not cache["done_real"][j]]
    if todo:
        print(f"[eval] real pass: {len(todo)} waveforms ({args.num_workers} workers)")
        jobs = {
            j: (resolve(metadatas[indices[j]]),
                metadatas[indices[j]]["station_name"],
                metadatas[indices[j]].get("channel_type", "HH"),
                str(metadatas[indices[j]].get("event_id", "")), channel_idx)
            for j in todo
        }
        run_pool(todo, jobs, "done_real", "dur_real", _duration_real, "real")

    synth_row = {idx: j for j, idx in enumerate(synth_indices)}
    todo = [j for j in range(n) if not cache["done_synth"][j]]
    if todo:
        print(f"[eval] synthetic pass: {len(todo)} waveforms")
        jobs = {}
        for j in todo:
            m = metadatas[indices[j]]
            jobs[j] = (wave_synth[synth_row[indices[j]]], m["station_name"],
                       f"{m.get('channel_type', 'HH')}{channel}",
                       str(m.get("event_id", "")))
        run_pool(todo, jobs, "done_synth", "dur_synth", _duration_synth, "synth")

    print(f"[eval] cache written: {path}")

    if args.n_realizations > 0:
        run_example(args, metadatas, station_locations, synth_cache_path,
                    indices, channel, channel_idx)
    return path


# ── Example realizations (panel a) ────────────────────────────────────────────
def run_example(args, metadatas, station_locations, synth_cache_path,
                indices, channel, channel_idx):
    """Generate --n_realizations synthetics for one record's conditioning and
    store the cAI curves. Reuses the model stack of evaluate_first_order.py."""
    import evaluate_first_order as fo
    from gwm_sampling import GwmSampler

    if args.example_index >= 0:
        ex_idx = args.example_index
    elif args.example_mag is not None:
        # Closest magnitude in the test set; ties broken by higher SNR so the
        # real reference curve is clean.
        diff = {i: abs(float(metadatas[i]["magnitude"]) - args.example_mag)
                for i in indices}
        best = min(diff.values())
        tied = [i for i, d in diff.items() if d <= best + 1e-9]
        ex_idx = max(tied, key=lambda i: float(metadatas[i].get("snr", 0.0)))
    else:  # default: largest-magnitude test-split record
        ex_idx = max(indices, key=lambda i: float(metadatas[i]["magnitude"]))
    meta = metadatas[ex_idx]
    sampler = GwmSampler(args.checkpoint, args.ae_checkpoint)
    out_path = OUT_DIR / (f"example_{synth_cache_path.stem.removeprefix('cache_')}"
                          f"_idx{ex_idx}_model{sampler.cache_tag}"
                          f"_n{args.n_realizations}.npz")
    if out_path.exists():
        print(f"[eval] example cache exists: {out_path.name}")
        return out_path
    print(f"[eval] example record {ex_idx}: M{meta['magnitude']} "
          f"station {meta['station_name']}, {args.n_realizations} realizations")

    station = meta["station_name"]
    code = f"{meta.get('channel_type', 'HH')}{channel}"
    event_id = str(meta.get("event_id", ""))

    n_curve = int(np.ceil(TARGET_SAMPLES / CAI_DECIMATE))
    cai_synth = np.full((args.n_realizations, n_curve), np.nan)
    done = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for start in range(0, args.n_realizations, args.batch_size):
            b = min(args.batch_size, args.n_realizations - start)
            waves = sampler.generate([meta] * b, args.steps, args.gl_iters,
                                     args.seed + start, pool, channel_idx=channel_idx)
            for k, wave in zip(range(start, start + b), waves):
                if wave is None:
                    continue
                acc = _deconvolve(wave, station, code, event_id, "ACC")
                if acc is not None:
                    cai_synth[k] = arias_curve(acc)[::CAI_DECIMATE]
            done += b
            print(f"[eval] example realizations {done}/{args.n_realizations}")

    real = _duration_real((str((DIFF_DIR / meta["file_path"]).resolve()),
                           station, meta.get("channel_type", "HH"),
                           event_id, channel_idx))
    from obspy import read as obspy_read

    stream = obspy_read(str((DIFF_DIR / meta["file_path"]).resolve()))
    stream.sort(keys=["channel"])
    trace = stream[channel_idx]
    if abs(trace.stats.sampling_rate - FS) > 1e-6:
        trace.resample(FS)
    acc = _deconvolve(_crop_pad(trace.data.astype(np.float64)), station, code,
                      event_id, "ACC")
    cai_real = (arias_curve(acc)[::CAI_DECIMATE] if acc is not None
                else np.full(n_curve, np.nan))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, cai_real=cai_real, cai_synth=cai_synth,
             mag=float(meta["magnitude"]), example_index=ex_idx,
             station=station, dur_real=real, decimate=CAI_DECIMATE)
    print(f"[eval] example cache written: {out_path}")
    return out_path


# ── plot stage ────────────────────────────────────────────────────────────────
def _bin_stats(mag, dur, bin_width, min_count):
    edges = np.arange(np.floor(mag.min()), mag.max() + bin_width, bin_width)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mag >= lo) & (mag < hi) & np.isfinite(dur)
        if sel.sum() < min_count:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(np.mean(dur[sel]))
        stds.append(np.std(dur[sel]))
    return np.asarray(centers), np.asarray(means), np.asarray(stds)


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caches = sorted(OUT_DIR.glob("durations_*.npz"), key=lambda p: p.stat().st_mtime)
    if not caches:
        raise FileNotFoundError(f"No cache in {OUT_DIR}; run `compute` first.")
    path = Path(args.durations_cache) if args.durations_cache else caches[-1]
    cache = dict(np.load(path))
    tag = path.stem.removeprefix("durations_")

    examples = sorted(OUT_DIR.glob(f"example_{tag}_*.npz"),
                      key=lambda p: p.stat().st_mtime)
    example = dict(np.load(examples[-1], allow_pickle=True)) if examples else None

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.patch.set_facecolor("white")

    # ── a) example cAI curves ────────────────────────────────────────────────
    if example is not None:
        t = np.arange(example["cai_real"].shape[0]) * float(example["decimate"]) / FS
        for k, cai in enumerate(example["cai_synth"]):
            if not np.isfinite(cai[-1]) or cai[-1] <= 0:
                continue
            ax_a.semilogy(t, np.maximum(cai, 1e-14), color=GWM_COLOR,
                          lw=0.6, alpha=0.25, zorder=1,
                          label="GWM synthetics" if k == 0 else None)
            for frac, marker in ((0.05, ">"), (0.95, "v")):
                i = int(np.searchsorted(cai, frac * cai[-1]))
                i = min(i, len(cai) - 1)
                ax_a.plot(t[i], cai[i], marker, color=GWM_COLOR, ms=3,
                          alpha=0.5, zorder=2)
        cai = example["cai_real"]
        if np.isfinite(cai[-1]) and cai[-1] > 0:
            ax_a.semilogy(t, np.maximum(cai, 1e-14), color=INK, lw=1.8,
                          zorder=3, label="Real data")
            for frac, marker in ((0.05, ">"), (0.95, "v")):
                i = min(int(np.searchsorted(cai, frac * cai[-1])), len(cai) - 1)
                ax_a.plot(t[i], cai[i], marker, mfc="white", mec=INK, ms=6, zorder=4)
        ax_a.set_title(f"M{float(example['mag']):.1f}, station {example['station']}"
                       f"  ({example['cai_synth'].shape[0]} realizations)",
                       color=INK, fontsize=10)
        ax_a.legend(fontsize=8, loc="lower right", frameon=True, edgecolor=GRID)
    else:
        ax_a.text(0.5, 0.5, "no example cache\n(run compute with "
                  "--n_realizations > 0)", ha="center", va="center",
                  transform=ax_a.transAxes, color=MUTED)
    ax_a.set_xlabel("Time [s]", color=INK)
    ax_a.set_ylabel("Cumulative Arias Intensity [m/s]", color=INK)
    ax_a.text(-0.1, 1.04, "a)", transform=ax_a.transAxes, fontsize=12, color=INK)

    # ── b) duration vs magnitude ─────────────────────────────────────────────
    mag = cache["mag"]
    ok = np.isfinite(cache["dur_real"]) & np.isfinite(cache["dur_synth"])
    print(f"[eval] panel b: {int(ok.sum())} records "
          f"(real median {np.nanmedian(cache['dur_real'][ok]):.1f} s, "
          f"synth median {np.nanmedian(cache['dur_synth'][ok]):.1f} s)")
    ax_b.scatter(cache["dur_real"][ok], mag[ok], s=4, color=REAL_COLOR,
                 alpha=0.18, lw=0, rasterized=True, label="Real data")
    ax_b.scatter(cache["dur_synth"][ok], mag[ok], s=5, marker="^",
                 color=GWM_COLOR, alpha=0.18, lw=0, rasterized=True,
                 label="GWM synthetics")
    for key, color, label in (("dur_real", INK, "mean ± std, real"),
                              ("dur_synth", GWM_COLOR, "mean ± std, GWM")):
        centers, means, stds = _bin_stats(mag[ok], cache[key][ok],
                                          args.mag_bin, args.min_bin_count)
        ax_b.errorbar(means, centers, xerr=stds, fmt="o" if key == "dur_real" else "^",
                      ms=3.5, color=color, ecolor=color, elinewidth=1.2,
                      capsize=2, lw=1.2, label=label, zorder=3)

    if args.duration_gmm != "none":
        gmm = DURATION_GMM_REGISTRY[args.duration_gmm]
        pred = gmm["predict"](mag[ok], cache["r_hyp"][ok], cache["vs30"][ok])
        centers, means, _ = _bin_stats(mag[ok], pred, args.mag_bin,
                                       args.min_bin_count)
        ax_b.plot(means, centers, color=GMM_COLOR, lw=1.6, ls="--",
                  label=f"{gmm['label']} (median)", zorder=3)
        print(f"[eval] {args.duration_gmm}: median prediction "
              f"{np.nanmedian(pred):.1f} s")

    ax_b.set_xlabel("Shaking Duration D5-95 [s]", color=INK)
    ax_b.set_ylabel("Magnitude", color=INK)
    ax_b.set_xlim(0, args.max_duration)
    ax_b.legend(fontsize=8, loc="upper right", frameon=True, edgecolor=GRID)
    ax_b.text(-0.1, 1.04, "b)", transform=ax_b.transAxes, fontsize=12, color=INK)

    for ax in (ax_a, ax_b):
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)

    fig.suptitle("Shaking duration from cumulative Arias Intensity — "
                 "test split, E component", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUT_DIR / f"fig_duration_{tag}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stage", choices=["compute", "plot", "all"], nargs="?",
                        default="all")
    parser.add_argument("--cache", type=str, default=None,
                        help="Synthetic-waveform cache from evaluate_first_order.py "
                             "(default: most recent in eval/first_order).")
    parser.add_argument("--durations_cache", type=str, default=None,
                        help="Durations .npz for `plot` (default: most recent).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only N evenly-spaced records (0 = all).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=512)
    # Panel a (example realizations; requires model checkpoints).
    parser.add_argument("--n_realizations", type=int, default=100,
                        help="Synthetics for the example record (0 = skip panel a).")
    parser.add_argument("--example_index", type=int, default=-1,
                        help="Metadata index of the example record "
                             "(-1 = pick by --example_mag or largest magnitude).")
    parser.add_argument("--example_mag", type=float, default=None,
                        help="Pick the test record with magnitude closest to "
                             "this value (ties broken by higher SNR). "
                             "Ignored when --example_index is set.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gl_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    # Panel b / plot.
    parser.add_argument("--duration_gmm", choices=["none", *sorted(DURATION_GMM_REGISTRY)],
                        default="none",
                        help="Optional duration-model overlay for panel b.")
    parser.add_argument("--mag_bin", type=float, default=0.25)
    parser.add_argument("--min_bin_count", type=int, default=5)
    parser.add_argument("--max_duration", type=float, default=70.0,
                        help="Panel b x-axis limit [s].")
    args = parser.parse_args()

    if args.stage in ("compute", "all"):
        run_compute(args)
    if args.stage in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
