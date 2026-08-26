"""
Browse candidate example events for the shaking-duration figure.

Given a target magnitude, lists the test-split records closest to it (ties
ordered by SNR, best first) and plots the real cumulative Arias Intensity
curve of each, with 5%/95% markers and the D5-95 duration - no synthetics,
no GPU. Pick a record by eye, then generate the figure with

    python eval/evaluate_shake_duration.py all --example_index <idx> ...

Run from the project root:
    python eval/pick_duration_example.py --mag 2.0 [--n 12]

The chart is saved to eval/shake_duration/candidates_M<mag>.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_peak_amplitudes as peak  # noqa: E402
from evaluate_peak_amplitudes import (  # noqa: E402
    CHANNEL_NAMES, DIFF_DIR, INK, MUTED, GRID,
    _crop_pad, _deconvolve, find_synth_cache,
)
from embedding_artifacts import artifact_path, resolve_embeddings_dir  # noqa: E402
from evaluate_shake_duration import (  # noqa: E402
    CAI_DECIMATE, OUT_DIR, REAL_COLOR, arias_curve, d595,
)
from ML.diffusion.waveform_domain import (  # noqa: E402
    INSTRUMENT_COUNTS,
    normalize_waveform_domain,
)


def real_cai(meta, channel_idx, waveform_domain):
    """Real record -> decimated cAI curve, or None on read/response failure."""
    from obspy import read as obspy_read

    p = Path(meta["file_path"])
    path = p if p.is_absolute() else (DIFF_DIR / p).resolve()
    try:
        stream = obspy_read(str(path))
        if len(stream) != 3:
            return None
        stream.sort(keys=["channel"])
        trace = stream[channel_idx]
        if abs(trace.stats.sampling_rate - peak.FS) > 1e-6:
            trace.resample(peak.FS)
        code = f"{meta.get('channel_type', 'HH')}{CHANNEL_NAMES[channel_idx]}"
        acc = _deconvolve(_crop_pad(trace.data.astype(np.float64)),
                          meta["station_name"], code,
                          str(meta.get("event_id", "")), "ACC", waveform_domain)
        if acc is None:
            return None
        return arias_curve(acc)[::CAI_DECIMATE]
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mag", type=float, required=True,
                        help="Target magnitude.")
    parser.add_argument("--n", type=int, default=12,
                        help="Number of candidates to show.")
    parser.add_argument("--cache", type=str, default=None,
                        help="Synthetic cache defining the test set "
                             "(default: most recent in eval/first_order).")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(resolve_embeddings_dir(None)),
        help="Embedding export directory used for metadata and station artifacts.",
    )
    args = parser.parse_args()
    embeddings_dir = peak.configure_embeddings_dir(args.embeddings_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from obspy.geodetics import gps2dist_azimuth

    with artifact_path(embeddings_dir, "metadata.json").open(encoding="utf-8") as handle:
        metadatas = json.load(handle)
    with artifact_path(embeddings_dir, "station_locations.json").open(encoding="utf-8") as handle:
        station_locations = json.load(handle)
    synth_cache = Path(args.cache) if args.cache else find_synth_cache()
    with np.load(synth_cache) as sc:
        indices = sc["indices"].astype(int).tolist()
        waveform_domain = normalize_waveform_domain(
            str(sc["waveform_domain"].item())
            if "waveform_domain" in sc.files else INSTRUMENT_COUNTS
        )
    channel = synth_cache.stem.rsplit("_ch", 1)[-1].split("_")[0]
    channel_idx = CHANNEL_NAMES.index(channel)

    picks = sorted(
        indices,
        key=lambda i: (abs(float(metadatas[i]["magnitude"]) - args.mag),
                       -float(metadatas[i].get("snr", 0.0))),
    )[:args.n]

    ncols = 4
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             sharex=True)
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).ravel()

    print(f"[pick] {len(picks)} candidates closest to M{args.mag:g} "
          f"(channel {channel}):")
    print(f"{'index':>7}  {'mag':>4}  {'station':<7} {'R_hyp':>6}  "
          f"{'SNR':>6}  {'D5-95':>6}")
    for ax, idx in zip(axes, picks):
        meta = metadatas[idx]
        sta = station_locations[meta["station_name"]]
        dist_m, _, _ = gps2dist_azimuth(meta["latitude"], meta["longitude"],
                                        sta["latitude"], sta["longitude"])
        r_hyp = float(np.hypot(dist_m / 1000.0, float(meta["depth"])))
        snr = float(meta.get("snr", np.nan))

        cai = real_cai(meta, channel_idx, waveform_domain)
        if cai is None or not np.isfinite(cai[-1]) or cai[-1] <= 0:
            dur = np.nan
            ax.text(0.5, 0.5, "failed", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED)
        else:
            dur, t5, t95 = d595(np.repeat(cai, CAI_DECIMATE))  # full-rate times
            t = np.arange(cai.shape[0]) * CAI_DECIMATE / peak.FS
            ax.semilogy(t, np.maximum(cai, 1e-14), color=REAL_COLOR, lw=1.2)
            for frac, marker in ((0.05, ">"), (0.95, "v")):
                i = min(int(np.searchsorted(cai, frac * cai[-1])), len(cai) - 1)
                ax.plot(t[i], cai[i], marker, mfc="white", mec=INK, ms=5)
        ax.set_title(
            f"idx {idx} — M{float(meta['magnitude']):.1f} "
            f"{meta['station_name']}\n"
            f"R={r_hyp:.0f} km, SNR={snr:.0f}, D$_{{5-95}}$={dur:.1f} s",
            fontsize=8, color=INK,
        )
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(colors=MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        print(f"{idx:>7}  {float(meta['magnitude']):>4.1f}  "
              f"{meta['station_name']:<7} {r_hyp:>5.0f}k  {snr:>6.0f}  "
              f"{dur:>5.1f}s")
    for ax in axes[len(picks):]:
        ax.axis("off")
    for ax in axes[max(0, len(picks) - ncols):len(picks)]:
        ax.set_xlabel("Time [s]", color=INK, fontsize=8)

    fig.suptitle(f"Example candidates near M{args.mag:g} — real cAI curves, "
                 f"{channel} component", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"candidates_M{args.mag:g}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[pick] figure saved: {out}")


if __name__ == "__main__":
    main()
