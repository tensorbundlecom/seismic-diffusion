"""
Statistics of GWM realizations for one scenario (GWM paper, Fig. 7).

For a single conditioning vector, generate --n_realizations synthetics and
study the peak-amplitude ensemble as a function of the number of draws:

  a) running median +/- std of the peak amplitude for n = 1..N realizations
     (how many draws are needed before the ensemble statistics stabilize);
  b) Shapiro-Wilk test statistic of log10 peak amplitudes for n = 3..N
     (values near 1: the peaks are log-normally distributed, the shape
     assumed by ground motion models and hazard analysis).

The scenario defaults to the test-split record closest to the test split's
median magnitude, median hypocentral distance, and median Vs30 (z-scored
nearest neighbour); override with --example_index.

ARCHITECTURAL NOTE: legacy per-event-normalized AEs rescale each synthetic to
the AmplitudeMLP's deterministic per-channel amplitude prediction. Their
realizations can therefore share one counts-domain peak. Global-normalized AEs
recover amplitude from the decoded spectrogram and do not apply that rescale.

Run from the project root:
    python eval/evaluate_distributions.py compute [--n_realizations 100]
    python eval/evaluate_distributions.py plot    [--im pga]
    python eval/evaluate_distributions.py all     [...]

Outputs (realization cache and figure) go to eval/distributions/.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_peak_amplitudes import (  # noqa: E402
    CHANNEL_NAMES, DIFF_DIR, ROOT,
    GWM_COLOR, INK, MUTED, GRID,
    _crop_pad, _deconvolve, find_synth_cache,
)

OUT_DIR = ROOT / "eval" / "distributions"


# ── Scenario selection ────────────────────────────────────────────────────────
def pick_median_scenario(indices, metadatas, station_locations, station_vs30):
    """Test record closest to the split's median (M, R_hyp, Vs30), z-scored."""
    from obspy.geodetics import gps2dist_azimuth

    feats = np.empty((len(indices), 3))
    for j, idx in enumerate(indices):
        m = metadatas[idx]
        sta = station_locations[m["station_name"]]
        dist_m, _, _ = gps2dist_azimuth(
            m["latitude"], m["longitude"], sta["latitude"], sta["longitude"]
        )
        feats[j] = (float(m["magnitude"]),
                    float(np.hypot(dist_m / 1000.0, float(m["depth"]))),
                    float(station_vs30[m["station_name"]]))
    med = np.median(feats, axis=0)
    z = (feats - med) / feats.std(axis=0)
    j = int(np.argmin(np.sum(z ** 2, axis=1)))
    return indices[j], feats[j], med


# ── compute stage ─────────────────────────────────────────────────────────────
def run_compute(args):
    import evaluate_first_order as fo
    from gwm_sampling import GwmSampler

    metadatas = json.load(open(DIFF_DIR / "embeddings" / "metadata.json"))
    station_locations = json.load(open(DIFF_DIR / "embeddings" / "station_locations.json"))
    station_vs30 = json.load(open(DIFF_DIR / "embeddings" / "station_vs30.json"))

    synth_cache_path = Path(args.cache) if args.cache else find_synth_cache()
    with np.load(synth_cache_path) as sc:
        indices = sc["indices"].astype(int).tolist()
    channel = synth_cache_path.stem.rsplit("_ch", 1)[-1].split("_")[0]
    channel_idx = CHANNEL_NAMES.index(channel)
    tag = synth_cache_path.stem.removeprefix("cache_")

    if args.example_index >= 0:
        ex_idx = args.example_index
        meta = metadatas[ex_idx]
        from obspy.geodetics import gps2dist_azimuth

        sta = station_locations[meta["station_name"]]
        dist_m, _, _ = gps2dist_azimuth(meta["latitude"], meta["longitude"],
                                        sta["latitude"], sta["longitude"])
        feats = (float(meta["magnitude"]),
                 float(np.hypot(dist_m / 1000.0, float(meta["depth"]))),
                 float(station_vs30[meta["station_name"]]))
    else:
        ex_idx, feats, med = pick_median_scenario(
            indices, metadatas, station_locations, station_vs30)
        meta = metadatas[ex_idx]
        print(f"[eval] test-split medians: M{med[0]:.1f}, R_hyp {med[1]:.0f} km, "
              f"Vs30 {med[2]:.0f} m/s")
    print(f"[eval] scenario record {ex_idx}: M{feats[0]:.1f}, "
          f"R_hyp {feats[1]:.0f} km, Vs30 {feats[2]:.0f} m/s, "
          f"station {meta['station_name']}")

    sampler = GwmSampler(args.checkpoint, args.ae_checkpoint)
    out_path = OUT_DIR / (f"realizations_{tag}_idx{ex_idx}_model{sampler.cache_tag}"
                          f"_n{args.n_realizations}.npz")
    if out_path.exists():
        print(f"[eval] realizations cache exists: {out_path.name}")
        return out_path

    station = meta["station_name"]
    code = f"{meta.get('channel_type', 'HH')}{channel}"
    event_id = str(meta.get("event_id", ""))

    pga = np.full(args.n_realizations, np.nan)
    pgv = np.full(args.n_realizations, np.nan)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for start in range(0, args.n_realizations, args.batch_size):
            b = min(args.batch_size, args.n_realizations - start)
            waves = sampler.generate([meta] * b, args.steps, args.gl_iters,
                                     args.seed + start, pool, channel_idx=channel_idx)
            for k, wave in zip(range(start, start + b), waves):
                if wave is None:
                    continue
                acc = _deconvolve(wave, station, code, event_id, "ACC")
                vel = _deconvolve(wave, station, code, event_id, "VEL")
                if acc is not None:
                    pga[k] = float(np.max(np.abs(acc)))
                if vel is not None:
                    pgv[k] = float(np.max(np.abs(vel)))
            print(f"[eval] realizations {min(start + b, args.n_realizations)}"
                  f"/{args.n_realizations}")

    # The real record: nature's single realization of the same scenario.
    from obspy import read as obspy_read

    p = Path(meta["file_path"])
    stream = obspy_read(str(p if p.is_absolute() else (DIFF_DIR / p).resolve()))
    stream.sort(keys=["channel"])
    trace = stream[channel_idx]
    if abs(trace.stats.sampling_rate - fo.FS) > 1e-6:
        trace.resample(fo.FS)
    data = _crop_pad(trace.data.astype(np.float64))
    acc = _deconvolve(data, station, code, event_id, "ACC")
    vel = _deconvolve(data, station, code, event_id, "VEL")
    real_pga = float(np.max(np.abs(acc))) if acc is not None else np.nan
    real_pgv = float(np.max(np.abs(vel))) if vel is not None else np.nan

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, pga=pga, pgv=pgv, real_pga=real_pga, real_pgv=real_pgv,
             example_index=ex_idx, mag=feats[0], r_hyp=feats[1], vs30=feats[2],
             station=station)
    print(f"[eval] realizations cache written: {out_path}")
    return out_path


# ── plot stage ────────────────────────────────────────────────────────────────
def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import shapiro

    if args.realizations_cache:
        path = Path(args.realizations_cache)
    else:
        caches = sorted(OUT_DIR.glob("realizations_*.npz"),
                        key=lambda p: p.stat().st_mtime)
        if not caches:
            raise FileNotFoundError(f"No cache in {OUT_DIR}; run `compute` first.")
        path = caches[-1]
    data = dict(np.load(path, allow_pickle=True))

    im = args.im
    unit = {"pga": "PGA [m/s$^2$]", "pgv": "PGV [m/s]"}[im]
    peaks = data[im]
    peaks = peaks[np.isfinite(peaks) & (peaks > 0)]
    real = float(data[f"real_{im}"])
    n_total = peaks.shape[0]
    print(f"[eval] {path.name}: {n_total} valid realizations, {im.upper()}")

    ns = np.arange(1, n_total + 1)
    med = np.array([np.median(peaks[:n]) for n in ns])
    std = np.array([peaks[:n].std() if n >= 2 else np.nan for n in ns])
    sw_ns = np.arange(3, n_total + 1)
    sw = np.array([shapiro(np.log10(peaks[:n])).statistic for n in sw_ns])

    log_sigma = np.log10(peaks).std()
    print(f"[eval] final: median {med[-1]:.4g}, std {std[-1]:.4g} "
          f"({log_sigma:.3f} log10 units), Shapiro-Wilk W={sw[-1]:.3f} "
          f"(p={shapiro(np.log10(peaks)).pvalue:.2f})")
    if np.isfinite(real):
        print(f"[eval] real record {im.upper()}: {real:.4g}")

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.patch.set_facecolor("white")

    ax_a.errorbar(ns[1:], med[1:], yerr=std[1:], fmt="o", ms=2.5,
                  color=INK, ecolor=GWM_COLOR, elinewidth=1.0, capsize=0)
    if np.isfinite(real):
        ax_a.axhline(real, color=MUTED, lw=1.2, ls="--")
        ax_a.text(0.99, real, "real record", ha="right", va="bottom",
                  transform=ax_a.get_yaxis_transform(), fontsize=8, color=MUTED)
    ax_a.set_ylabel(unit, color=INK)
    ax_a.text(0.98, 0.04,
              f"Scenario (record {int(data['example_index'])}):\n"
              f"M {float(data['mag']):.1f}\n"
              f"Hypocentral distance = {float(data['r_hyp']):.0f} km\n"
              f"V$_{{S30}}$ = {float(data['vs30']):.0f} m/s\n"
              f"Station {data['station']}",
              transform=ax_a.transAxes, fontsize=8, ha="right", va="bottom",
              color=INK, bbox=dict(facecolor="white", edgecolor=GRID))
    ax_a.text(-0.09, 1.02, "a)", transform=ax_a.transAxes, fontsize=12, color=INK)

    ax_b.plot(sw_ns, sw, color=INK, lw=1.2)
    ax_b.set_ylabel("Shapiro–Wilk Test Statistic", color=INK)
    ax_b.set_xlabel("Realizations", color=INK)
    ax_b.text(-0.09, 1.02, "b)", transform=ax_b.transAxes, fontsize=12, color=INK)

    for ax in (ax_a, ax_b):
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)

    fig.suptitle(f"Statistics of GWM realizations — {im.upper()}, "
                 f"E component", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUT_DIR / f"fig_{path.stem.removeprefix('realizations_')}_{im}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stage", choices=["compute", "plot", "all"], nargs="?",
                        default="all")
    parser.add_argument("--cache", type=str, default=None,
                        help="Synthetic cache defining the test set "
                             "(default: most recent in eval/first_order).")
    parser.add_argument("--realizations_cache", type=str, default=None,
                        help="Realizations .npz for `plot` (default: most recent).")
    parser.add_argument("--example_index", type=int, default=-1,
                        help="Metadata index of the scenario record (-1 = record "
                             "closest to the test split's median M/R_hyp/Vs30).")
    parser.add_argument("--n_realizations", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gl_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--im", choices=["pga", "pgv"], default="pga",
                        help="Intensity measure for the plot.")
    args = parser.parse_args()

    if args.stage in ("compute", "all"):
        path = run_compute(args)
        if args.stage == "all" and args.realizations_cache is None:
            args.realizations_cache = str(path)
    if args.stage in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
