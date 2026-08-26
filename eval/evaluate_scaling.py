"""
SA scaling with magnitude and Vs30 (GWM paper, Fig. 9).

  a) SA(--period) vs magnitude: the GWM is sampled --n_realizations times at
     each magnitude grid point, at a fixed hypocentral distance and the
     scenario station's Vs30. Gray dots: real records from a narrow distance
     and Vs30 window (any magnitude).
  b) SA(--period) vs Vs30: same, sweeping the Vs30 conditioning feature at
     fixed magnitude and distance. Gray dots: real records from a narrow
     distance and magnitude window (any station), plotted at their station's
     Vs30 - only ~46 discrete values exist, so the dots form vertical
     stripes rather than the paper's continuum.

Black line: ensemble median per grid point; shaded band: log10-std.
No GMM in this figure (the paper's Fig. 9 has none either).

Adaptations vs the paper (documented, not corrected):
  - Magnitude sweep 1.5-5.0 and Vs30 sweep 350-650 m/s: the ranges the
    Marmara dataset actually covers (station Vs30 spans only ~400-605 m/s).
  - Period defaults to 0.3 s (in the 2-15 Hz waveform band).
  - E component. The legacy AmplitudeMLP variability caveat of
    evaluate_distributions.py applies only to per-event-normalized AEs.
  - Vs30 enters diffusion conditioning; for legacy AEs it also enters the
    AmplitudeMLP, while station identity is held fixed.

Run from the project root:
    python eval/evaluate_scaling.py compute [--n_realizations 50]
    python eval/evaluate_scaling.py plot
    python eval/evaluate_scaling.py all     [...]

Outputs (real-data cache, sweep cache, figure) go to eval/scaling/.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_peak_amplitudes import (  # noqa: E402
    DIFF_DIR, ROOT, VS30_RANGE,
    GWM_COLOR, SCATTER_COLOR, INK, MUTED, GRID,
)
from evaluate_attenuation import _sa_real, destination, event_azimuth  # noqa: E402
from embedding_artifacts import (  # noqa: E402
    artifact_path,
    deterministic_fraction_subset,
    deterministic_selection_tag,
    resolve_embeddings_dir,
)
from output_paths import evaluation_output_dir  # noqa: E402

OUT_DIR = evaluation_output_dir("scaling")


def _sa_waveform(args):
    """Worker: checkpoint-domain waveform -> physical-acceleration SA."""
    from evaluate_attenuation import _sa_synth

    return _sa_synth(args)


# ── compute stage ─────────────────────────────────────────────────────────────
def run_compute(args):
    from gwm_sampling import GwmSampler

    import evaluate_attenuation as attenuation

    embeddings_dir = attenuation.configure_embeddings_dir(args.embeddings_dir)
    sampler = GwmSampler(
        args.checkpoint, args.ae_checkpoint, args.waveform_domain, embeddings_dir
    )
    waveform_domain = sampler.waveform_domain
    from obspy.geodetics import gps2dist_azimuth

    metadatas = json.load(open(artifact_path(embeddings_dir, "metadata.json")))
    station_locations = json.load(open(artifact_path(embeddings_dir, "station_locations.json")))
    station_vs30 = json.load(open(artifact_path(embeddings_dir, "station_vs30.json")))
    period = args.period

    # Hypocentral distance for every record (cheap, a few seconds).
    r_hyp = np.empty(len(metadatas))
    for i, m in enumerate(metadatas):
        sta = station_locations[m["station_name"]]
        d, _, _ = gps2dist_azimuth(m["latitude"], m["longitude"],
                                   sta["latitude"], sta["longitude"])
        r_hyp[i] = np.hypot(d / 1000.0, float(m["depth"]))
    mags = np.asarray([float(m["magnitude"]) for m in metadatas])
    vs30s = np.asarray([float(station_vs30[m["station_name"]]) for m in metadatas])

    dist_lo = args.dist_center - args.dist_halfwidth
    dist_hi = args.dist_center + args.dist_halfwidth
    in_dist = (r_hyp >= dist_lo) & (r_hyp <= dist_hi)

    if args.station:
        sta_name = args.station
    else:  # most records inside the distance window
        counts = {}
        for i in np.flatnonzero(in_dist):
            name = metadatas[i]["station_name"]
            counts[name] = counts.get(name, 0) + 1
        sta_name = max(counts, key=counts.get)
    sta_vs30 = float(station_vs30[sta_name])

    in_mag_panel = in_dist & (np.abs(vs30s - sta_vs30) <= args.vs30_halfwidth)
    in_vs30_panel = (in_dist & (np.abs(mags - args.scenario_mag) <= args.mag_halfwidth)
                     & (vs30s >= VS30_RANGE[0]) & (vs30s <= VS30_RANGE[1]))
    union = sorted(set(np.flatnonzero(in_mag_panel)) | set(np.flatnonzero(in_vs30_panel)))
    print(f"[eval] distance {dist_lo:g}-{dist_hi:g} km, scenario station {sta_name} "
          f"(Vs30 {sta_vs30:.0f}): panel a {int(in_mag_panel.sum())} records, "
          f"panel b {int(in_vs30_panel.sum())}, union {len(union)}")

    # ── Real pass ────────────────────────────────────────────────────────────
    tag = (f"d{args.dist_center:g}_{sta_name}_T{period:g}"
           f"_M{args.scenario_mag:g}_Ma{args.mag_min:g}-{args.mag_max:g}")
    sel = deterministic_fraction_subset(union, args.fraction, args.limit)
    print(
        f"[eval] deterministic real-data sample: {len(sel)}/{len(union)} "
        f"records (fraction={args.fraction:g})"
    )
    selection_tag = deterministic_selection_tag(sel)
    real_path = OUT_DIR / (
        f"scal_real_{tag}_domain{waveform_domain}_n{len(sel)}_sel{selection_tag}.npz"
    )
    n = len(sel)
    real = None
    if real_path.exists():
        old = dict(np.load(real_path))
        if old["indices"].tolist() == sel:
            real = old
        else:
            print(f"[eval] {real_path.name} covers a different selection; rebuilding.")
    if real is None:
        real = {
            "indices": np.asarray(sel, dtype=np.int64),
            "waveform_domain": np.asarray(waveform_domain),
            "mag": mags[sel], "vs30": vs30s[sel], "r_hyp": r_hyp[sel],
            "in_mag_panel": in_mag_panel[sel], "in_vs30_panel": in_vs30_panel[sel],
            "sa": np.full(n, np.nan),
            "done": np.zeros(n, dtype=bool),
        }

    def resolve(m):
        p = Path(m["file_path"])
        return str(p if p.is_absolute() else (DIFF_DIR / p).resolve())

    todo = [j for j in range(n) if not real["done"][j]]
    if todo:
        print(f"[eval] real pass: {len(todo)} waveforms ({args.num_workers} workers)")
        t0, n_done = time.time(), 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for start in range(0, len(todo), args.chunk):
                rows = todo[start:start + args.chunk]
                jobs = []
                for j in rows:
                    m = metadatas[sel[j]]
                    jobs.append((resolve(m), m["station_name"],
                                 m.get("channel_type", "HH"),
                                 str(m.get("event_id", "")), 0, [period],
                                 waveform_domain))
                for j, out in zip(rows, pool.map(_sa_real, jobs, chunksize=8)):
                    if out is not None:
                        real["sa"][j] = out[0]
                    real["done"][j] = True
                n_done += len(rows)
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                tmp = real_path.with_name(real_path.stem + "_tmp.npz")
                np.savez(tmp, **real)
                tmp.replace(real_path)
                rate = n_done / max(time.time() - t0, 1e-9)
                eta = (len(todo) - n_done) / max(rate, 1e-9) / 60.0
                print(f"[eval] real {n_done}/{len(todo)} "
                      f"({rate:.1f} rec/s, ETA {eta:.0f} min)")
    print(f"[eval] real cache: {real_path.name}")

    sweep_path = OUT_DIR / (f"scal_sweep_{tag}_model{sampler.cache_tag}"
                            f"_n{args.n_realizations}.npz")
    if sweep_path.exists():
        print(f"[eval] sweep cache exists: {sweep_path.name}")
        return real_path, sweep_path

    # ── Scenario geometry (fixed for both sweeps) ────────────────────────────
    sel_meta = [metadatas[i] for i in union]
    depth = float(np.median([float(m["depth"]) for m in sel_meta]))
    snr = float(np.median([float(m.get("snr", 10.0)) for m in sel_meta]))
    sta = station_locations[sta_name]
    sta_records = [m for m in sel_meta if m["station_name"] == sta_name]
    azis = np.asarray([event_azimuth(m["latitude"], m["longitude"],
                                     sta["latitude"], sta["longitude"])
                       for m in sta_records])
    az = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(azis))),
                                     np.mean(np.cos(np.radians(azis)))))) % 360.0
    station_idx = int(sta_records[0]["station_idx"])
    channel_type = sta_records[0].get("channel_type", "HH")
    r_epi = float(np.sqrt(max(args.dist_center ** 2 - depth ** 2, 1.0)))
    ev_lat, ev_lon = destination(sta["latitude"], sta["longitude"],
                                 (az + 180.0) % 360.0, r_epi)
    print(f"[eval] scenario: R_hyp {args.dist_center:g} km, depth {depth:.1f} km, "
          f"snr {snr:.0f}, azimuth {az:.0f} deg")

    base_meta = {
        "latitude": ev_lat, "longitude": ev_lon, "depth": depth, "snr": snr,
        "station_name": sta_name, "station_idx": station_idx,
        "channel_idx": 0, "channel_type": channel_type, "event_id": "",
    }
    mag_grid = np.arange(args.mag_min, args.mag_max + args.mag_step / 2,
                         args.mag_step)
    vs30_grid = np.arange(args.vs30_min, args.vs30_max + args.vs30_step / 2,
                          args.vs30_step)
    points = ([("mag", gi, {**base_meta, "magnitude": float(v)})
               for gi, v in enumerate(mag_grid)]
              + [("vs30", gi, {**base_meta, "magnitude": args.scenario_mag,
                               "vs30_override": float(v)})
                 for gi, v in enumerate(vs30_grid)])
    pairs = [(p, ri) for p in points for ri in range(args.n_realizations)]

    # ── GWM sweeps ───────────────────────────────────────────────────────────
    sa_mag = np.full((len(mag_grid), args.n_realizations), np.nan)
    sa_vs30 = np.full((len(vs30_grid), args.n_realizations), np.nan)
    code = f"{channel_type}E"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            waves = sampler.generate([p[2] for p, _ in batch], args.steps,
                                     args.gl_iters, args.seed + start, pool)
            jobs = [(w, sta_name, code, "", [period], waveform_domain)
                    for w in waves if w is not None]
            keep = [k for k, w in enumerate(waves) if w is not None]
            for k, out in zip(keep, pool.map(_sa_waveform, jobs)):
                if out is None:
                    continue
                (kind, gi, _), ri = batch[k]
                (sa_mag if kind == "mag" else sa_vs30)[gi, ri] = out[0]
            done = min(start + len(batch), len(pairs))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(pairs) - done) / max(rate, 1e-9) / 60.0
            print(f"[eval] sweep {done}/{len(pairs)} "
                  f"({rate:.1f} samples/s, ETA {eta:.0f} min)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(sweep_path, mag_grid=mag_grid, vs30_grid=vs30_grid,
             sa_mag=sa_mag, sa_vs30=sa_vs30, period=period,
             station=sta_name, sta_vs30=sta_vs30, depth=depth,
             dist_center=args.dist_center, dist_halfwidth=args.dist_halfwidth,
             scenario_mag=args.scenario_mag,
             vs30_halfwidth=args.vs30_halfwidth,
             mag_halfwidth=args.mag_halfwidth,
             waveform_domain=waveform_domain)
    print(f"[eval] sweep cache written: {sweep_path.name}")
    return real_path, sweep_path


# ── plot stage ────────────────────────────────────────────────────────────────
def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def latest(pattern, override):
        if override:
            return Path(override)
        found = sorted(OUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"No {pattern} in {OUT_DIR}; run `compute` first.")
        return found[-1]

    real = dict(np.load(latest("scal_real_*.npz", args.real_cache)))
    sweep = dict(np.load(latest("scal_sweep_*.npz", args.sweep_cache),
                         allow_pickle=True))
    period = float(sweep["period"])
    dist_lo = float(sweep["dist_center"]) - float(sweep["dist_halfwidth"])
    dist_hi = float(sweep["dist_center"]) + float(sweep["dist_halfwidth"])

    good = np.isfinite(real["sa"]) & (real["sa"] > 0)
    if args.sa_floor > 0:
        n_bad = int((good & (real["sa"] < args.sa_floor)).sum())
        if n_bad:
            print(f"[eval] sa_floor {args.sa_floor:g}: excluding {n_bad} records")
        good &= real["sa"] >= args.sa_floor

    panels = [
        ("a)", "mag_grid", "sa_mag", "in_mag_panel", real["mag"], "Magnitude",
         f"Distance: {dist_lo:g}-{dist_hi:g} km,  "
         f"V$_{{S30}}$: {float(sweep['sta_vs30']):.0f}$\\pm$"
         f"{float(sweep['vs30_halfwidth']):.0f} m/s"),
        ("b)", "vs30_grid", "sa_vs30", "in_vs30_panel", real["vs30"],
         "V$_{S30}$ [m/s]",
         f"Distance: {dist_lo:g}-{dist_hi:g} km,  "
         f"M{float(sweep['scenario_mag']) - float(sweep['mag_halfwidth']):g}-"
         f"{float(sweep['scenario_mag']) + float(sweep['mag_halfwidth']):g}"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.patch.set_facecolor("white")

    for ax, (tag, grid_key, sa_key, sel_key, x_real, xlabel, title) in zip(
            axes, panels):
        grid = sweep[grid_key]
        # Show only data inside the sweep range (e.g. nothing above mag_max).
        sel = (real[sel_key].astype(bool) & good
               & (x_real >= grid.min()) & (x_real <= grid.max()))
        ax.scatter(x_real[sel], real["sa"][sel], s=4, color=SCATTER_COLOR,
                   alpha=0.35, lw=0, rasterized=True, label="Data")
        sa_s = sweep[sa_key]
        med = np.nanmedian(sa_s, axis=1)
        log_sd = np.nanstd(np.log10(np.maximum(sa_s, 1e-30)), axis=1)
        ax.fill_between(grid, 10 ** (np.log10(med) - log_sd),
                        10 ** (np.log10(med) + log_sd), color=GWM_COLOR,
                        alpha=0.25, lw=0, label="GWM-std.")
        ax.plot(grid, med, color=INK, lw=1.6, label="GWM-med.")

        ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylabel(f"SA({period:g}s) [m/s$^2$]", color=INK)
        ax.set_title(f"{title},  T={period:g}s,  N$_{{obs}}$={int(sel.sum())}",
                     fontsize=9, color=INK)
        ax.text(-0.09, 1.04, tag, transform=ax.transAxes, fontsize=12, color=INK)
        ax.legend(fontsize=8, loc="lower right", frameon=True, edgecolor=GRID)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)

    fig.suptitle(
        f"SA scaling with magnitude and V$_{{S30}}$ — GWM (station "
        f"{sweep['station']}, R={float(sweep['dist_center']):g} km) vs data, "
        f"E component", color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = OUT_DIR / (f"fig_scaling_{sweep['station']}"
                     f"_T{period:g}_R{float(sweep['dist_center']):g}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stage", choices=["compute", "plot", "all"], nargs="?",
                        default="all")
    parser.add_argument("--period", type=float, default=0.3,
                        help="SA period [s], inside the 2-15 Hz band.")
    parser.add_argument("--dist_center", type=float, default=60.0,
                        help="Fixed hypocentral distance for both sweeps [km].")
    parser.add_argument("--dist_halfwidth", type=float, default=10.0,
                        help="Data selection: distance window half-width [km].")
    parser.add_argument("--station", type=str, default=None,
                        help="Scenario station (default: most records in the "
                             "distance window).")
    parser.add_argument("--mag_min", type=float, default=1.5)
    parser.add_argument("--mag_max", type=float, default=3.5,
                        help="Sweep ceiling; data above ~M3.5 is too sparse "
                             "to constrain the model.")
    parser.add_argument("--mag_step", type=float, default=0.25)
    parser.add_argument("--scenario_mag", type=float, default=2.5,
                        help="Fixed magnitude for the Vs30 sweep.")
    parser.add_argument("--vs30_min", type=float, default=350.0)
    parser.add_argument("--vs30_max", type=float, default=650.0)
    parser.add_argument("--vs30_step", type=float, default=12.5)
    parser.add_argument("--vs30_halfwidth", type=float, default=75.0,
                        help="Data selection (panel a): Vs30 window half-width.")
    parser.add_argument("--mag_halfwidth", type=float, default=0.1,
                        help="Data selection (panel b): magnitude half-width.")
    parser.add_argument("--n_realizations", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--embeddings_dir", type=str, default=str(resolve_embeddings_dir(None)),
        help="Embedding export directory used for metadata and station artifacts.",
    )
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument(
        "--waveform_domain",
        choices=["auto", "instrument_counts", "physical_acceleration"],
        default="auto",
        help="Override checkpoint waveform-domain provenance for legacy checkpoints.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gl_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0,
                        help="Real pass: only N evenly-spaced records (0 = all).")
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Deterministic evenly-spaced fraction of selected real records in (0, 1].",
    )
    parser.add_argument("--real_cache", type=str, default=None)
    parser.add_argument("--sweep_cache", type=str, default=None)
    parser.add_argument("--sa_floor", type=float, default=1e-5,
                        help="Exclude records with SA below this [m/s^2] "
                             "(miscalibrated response epochs; 0 disables).")
    args = parser.parse_args()

    if args.stage in ("compute", "all"):
        real_path, sweep_path = run_compute(args)
        if args.stage == "all":
            args.real_cache = args.real_cache or str(real_path)
            args.sweep_cache = args.sweep_cache or str(sweep_path)
    if args.stage in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
