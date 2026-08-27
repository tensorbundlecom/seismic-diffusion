"""
Average model probabilities given the SA data (GWM paper, Fig. 10, Eq. 12).

For every magnitude-distance bin, both models predict a log-normal SA
distribution (mean mu, std sigma in log10 units):
  - GMM (--gmm, swappable): median and published sigma at the bin center;
  - GWM: mu/sigma of --n_realizations diffusion samples generated with the
    bin-center conditioning (scenario station, median depth/azimuth/snr).
The average probability density assigned to the N observed records in the
bin,  P = 1/N sum_i N(logSA_i | mu, sigma),  scores each model: it is high
only when both the mean AND the spread are right. The bottom row shows the
ratio P_GMM / P_GWM (red < 1: GWM explains the data better).

The GMM enters at plot time only, so swapping it (--gmm) or re-binning
reuses the expensive caches. Distances are capped by the bin grid (10-120
km by default); records beyond it are never used.

Adaptations vs the paper: single Vs30 column (the network only spans ~400-
600 m/s), T defaults to 0.3 s (2-15 Hz band), E component, ML magnitudes.
NOTE: the legacy per-event amplitude model can pin synthetic amplitudes (see
evaluate_distributions.py), under-estimating GWM sigma. Global-normalized AEs
do not apply that model.

Run from the project root:
    python eval/evaluate_model_probabilities.py compute [--n_realizations 20]
    python eval/evaluate_model_probabilities.py plot    [--gmm edwardsfah13]
    python eval/evaluate_model_probabilities.py all     [...]

Outputs (real cache, GWM-moments cache, figure) go to
eval/model_probabilities/.
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
    DIFF_DIR, ROOT, VS30_RANGE, INK, MUTED, GRID,
)
from evaluate_attenuation import (  # noqa: E402
    ATT_GMM_REGISTRY, _sa_real, _sa_synth, destination, event_azimuth,
)
from embedding_artifacts import (  # noqa: E402
    artifact_path,
    deterministic_fraction_subset,
    deterministic_selection_tag,
    resolve_embeddings_dir,
)
from output_paths import evaluation_output_dir  # noqa: E402

OUT_DIR = evaluation_output_dir("model_probabilities")
LN10 = np.log(10.0)


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

    r_hyp = np.empty(len(metadatas))
    r_epi = np.empty(len(metadatas))
    for i, m in enumerate(metadatas):
        sta = station_locations[m["station_name"]]
        d, _, _ = gps2dist_azimuth(m["latitude"], m["longitude"],
                                   sta["latitude"], sta["longitude"])
        r_epi[i] = d / 1000.0
        r_hyp[i] = np.hypot(d / 1000.0, float(m["depth"]))
    mags = np.asarray([float(m["magnitude"]) for m in metadatas])
    vs30s = np.asarray([float(station_vs30[m["station_name"]]) for m in metadatas])

    in_window = ((mags >= args.mag_min) & (mags <= args.mag_max)
                 & (r_hyp >= args.dist_min) & (r_hyp <= args.dist_max))
    if args.station:
        sta_name = args.station
    else:
        counts = {}
        for i in np.flatnonzero(in_window):
            name = metadatas[i]["station_name"]
            counts[name] = counts.get(name, 0) + 1
        sta_name = max(counts, key=counts.get)
    sta_vs30 = float(station_vs30[sta_name])
    sel_mask = (in_window & (np.abs(vs30s - sta_vs30) <= args.vs30_halfwidth)
                & (vs30s >= VS30_RANGE[0]) & (vs30s <= VS30_RANGE[1]))
    all_sel = sorted(np.flatnonzero(sel_mask))
    sel = deterministic_fraction_subset(all_sel, args.fraction, args.limit)
    print(f"[eval] selection: M{args.mag_min:g}-{args.mag_max:g}, "
          f"R {args.dist_min:g}-{args.dist_max:g} km, Vs30 "
          f"{sta_vs30:.0f}±{args.vs30_halfwidth:g} -> {len(sel)} records; "
          f"scenario station {sta_name}; deterministic real-data sample "
          f"{len(sel)}/{len(all_sel)} (fraction={args.fraction:g})")

    # ── Real pass: SA per record ─────────────────────────────────────────────
    tag = (f"{sta_name}_T{period:g}_M{args.mag_min:g}-{args.mag_max:g}"
           f"_R{args.dist_min:g}-{args.dist_max:g}")
    selection_tag = deterministic_selection_tag(sel)
    real_path = OUT_DIR / (
        f"prob_real_{tag}_domain{waveform_domain}_n{len(sel)}_sel{selection_tag}.npz"
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
            "mag": mags[sel], "r_hyp": r_hyp[sel],
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

    # ── GWM moments per bin center ───────────────────────────────────────────
    sweep_path = OUT_DIR / (f"prob_gwm_{tag}_model{sampler.cache_tag}"
                            f"_n{args.n_realizations}.npz")
    if sweep_path.exists():
        print(f"[eval] GWM cache exists: {sweep_path.name}")
        return real_path, sweep_path

    mag_edges = np.arange(args.mag_min, args.mag_max + args.mag_step / 2,
                          args.mag_step)
    dist_edges = np.arange(args.dist_min, args.dist_max + args.dist_step / 2,
                           args.dist_step)
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])

    # Scenario geometry must remain independent of the observed-data fraction
    # so cached GWM sweeps are comparable across full and sampled evaluations.
    sel_meta = [metadatas[i] for i in all_sel]
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
    print(f"[eval] scenario: depth {depth:.1f} km, snr {snr:.0f}, "
          f"azimuth {az:.0f} deg; grid {len(mag_centers)}x{len(dist_centers)} "
          f"bins x {args.n_realizations} realizations")

    points = []
    for mi, mag_c in enumerate(mag_centers):
        for di, dist_c in enumerate(dist_centers):
            d_epi = float(np.sqrt(max(dist_c ** 2 - depth ** 2, 1.0)))
            ev_lat, ev_lon = destination(sta["latitude"], sta["longitude"],
                                         (az + 180.0) % 360.0, d_epi)
            points.append((mi, di, {
                "magnitude": float(mag_c), "latitude": ev_lat,
                "longitude": ev_lon, "depth": depth, "snr": snr,
                "station_name": sta_name, "station_idx": station_idx,
                "channel_idx": 0, "channel_type": channel_type, "event_id": "",
            }))
    pairs = [(p, ri) for p in points for ri in range(args.n_realizations)]

    sa = np.full((len(mag_centers), len(dist_centers), args.n_realizations),
                 np.nan)
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
            for k, out in zip(keep, pool.map(_sa_synth, jobs)):
                if out is None:
                    continue
                (mi, di, _), ri = batch[k]
                sa[mi, di, ri] = out[0]
            done = min(start + len(batch), len(pairs))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(pairs) - done) / max(rate, 1e-9) / 60.0
            print(f"[eval] GWM sweep {done}/{len(pairs)} "
                  f"({rate:.1f} samples/s, ETA {eta:.0f} min)")

    log_sa = np.log10(np.maximum(sa, 1e-30))
    log_sa[~np.isfinite(sa)] = np.nan
    mu_gwm = np.nanmean(log_sa, axis=2)
    sigma_gwm = np.nanstd(log_sa, axis=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(sweep_path, mag_edges=mag_edges, dist_edges=dist_edges,
             mu_gwm=mu_gwm, sigma_gwm=sigma_gwm, period=period,
             station=sta_name, sta_vs30=sta_vs30, depth=depth,
             vs30_halfwidth=args.vs30_halfwidth,
             n_realizations=args.n_realizations,
             waveform_domain=waveform_domain)
    print(f"[eval] GWM cache written: {sweep_path.name}")
    return real_path, sweep_path


# ── plot stage ────────────────────────────────────────────────────────────────
def _avg_probability(log_obs, mu, sigma):
    """Eq. 12 in log10 space; NaN when the model is undefined."""
    if not (np.isfinite(mu) and np.isfinite(sigma)) or sigma <= 0:
        return np.nan
    dens = (np.exp(-0.5 * ((log_obs - mu) / sigma) ** 2)
            / (np.sqrt(2.0 * np.pi) * sigma))
    return float(np.mean(dens))


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    def latest(pattern, override):
        if override:
            return Path(override)
        found = sorted(OUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"No {pattern} in {OUT_DIR}; run `compute` first.")
        return found[-1]

    real = dict(np.load(latest("prob_real_*.npz", args.real_cache)))
    gwm = dict(np.load(latest("prob_gwm_*.npz", args.gwm_cache),
                       allow_pickle=True))
    period = float(gwm["period"])
    mag_edges, dist_edges = gwm["mag_edges"], gwm["dist_edges"]
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])
    sta_vs30 = float(gwm["sta_vs30"])
    depth = float(gwm["depth"])

    ok = np.isfinite(real["sa"]) & (real["sa"] > 0)
    if args.sa_floor > 0:
        n_bad = int((ok & (real["sa"] < args.sa_floor)).sum())
        if n_bad:
            print(f"[eval] sa_floor {args.sa_floor:g}: excluding {n_bad} records")
        ok &= real["sa"] >= args.sa_floor
    log_sa = np.log10(real["sa"][ok])
    mags, dists = real["mag"][ok], real["r_hyp"][ok]

    gmm = None if args.gmm == "none" else ATT_GMM_REGISTRY[args.gmm]
    mu_gmm = np.full((len(mag_centers), len(dist_centers)), np.nan)
    sg_gmm = np.full_like(mu_gmm, np.nan)
    if gmm is not None:
        for mi, mag_c in enumerate(mag_centers):
            d_epi = np.sqrt(np.maximum(dist_centers ** 2 - depth ** 2, 1.0))
            med, ln_std = gmm["predict"](float(mag_c), dist_centers, d_epi,
                                         sta_vs30, [period])
            mu_gmm[mi] = np.log10(med[0])
            sg_gmm[mi] = ln_std[0] / LN10

    n_obs = np.zeros_like(mu_gmm)
    p_gmm = np.full_like(mu_gmm, np.nan)
    p_gwm = np.full_like(mu_gmm, np.nan)
    for mi in range(len(mag_centers)):
        for di in range(len(dist_centers)):
            bsel = ((mags >= mag_edges[mi]) & (mags < mag_edges[mi + 1])
                    & (dists >= dist_edges[di]) & (dists < dist_edges[di + 1]))
            n_obs[mi, di] = bsel.sum()
            if bsel.sum() < args.min_bin_count:
                continue
            if gmm is not None:
                p_gmm[mi, di] = _avg_probability(log_sa[bsel], mu_gmm[mi, di],
                                                 sg_gmm[mi, di])
            p_gwm[mi, di] = _avg_probability(log_sa[bsel],
                                             gwm["mu_gwm"][mi, di],
                                             gwm["sigma_gwm"][mi, di])

    valid_gwm = np.isfinite(p_gwm)
    if gmm is None:
        print(f"[eval] bins with data: {int(valid_gwm.sum())} of {p_gwm.size}")
        panels = [("GWM", p_gwm, "viridis", (0, np.nanmax(p_gwm)), "$P_{GWM}$")]
    else:
        ratio = p_gmm / p_gwm
        both = np.isfinite(ratio)
        print(f"[eval] bins with data: {int(both.sum())} of {ratio.size}; "
              f"GWM better (ratio<1) in {int((ratio[both] < 1).sum())}, "
              f"{args.gmm} better in {int((ratio[both] > 1).sum())}; "
              f"median ratio {np.nanmedian(ratio):.2f}")
        pmax = np.nanmax([np.nanmax(p_gmm), np.nanmax(p_gwm)])
        panels = [
            (f"{gmm['label']}", p_gmm, "viridis", (0, pmax), "$P_{GMM}$"),
            ("GWM", p_gwm, "viridis", (0, pmax), "$P_{GWM}$"),
            (f"Prob. ratio $P_{{GMM}}/P_{{GWM}}$", ratio, "RdBu",
             None, "ratio"),
        ]

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(8.5, 3.8 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")
    for ax, (title, grid_vals, cmap, vlim, cbar_label) in zip(axes, panels):
        masked = np.ma.masked_invalid(grid_vals)
        if vlim is None:
            norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0,
                                vmax=max(2.0, float(np.nanmax(ratio))))
            mesh = ax.pcolormesh(dist_edges, mag_edges, masked, cmap=cmap,
                                 norm=norm, edgecolors="white", lw=0.3)
        else:
            mesh = ax.pcolormesh(dist_edges, mag_edges, masked, cmap=cmap,
                                 vmin=vlim[0], vmax=vlim[1],
                                 edgecolors="white", lw=0.3)
        cb = fig.colorbar(mesh, ax=ax, pad=0.015)
        cb.set_label(cbar_label, color=INK)
        cb.ax.tick_params(colors=MUTED)
        ax.set_ylabel("Magnitude", color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
    axes[-1].set_xlabel("Hypocentral Distance [km]", color=INK)

    subtitle = ("\n(bottom panel: red = GWM explains the data better)"
                if gmm is not None else "")
    fig.suptitle(
        f"Average model probabilities, SA(T={period:g}s) — station "
        f"{gwm['station']}, V$_{{S30}}$ {sta_vs30:.0f}$\\pm$"
        f"{float(gwm['vs30_halfwidth']):.0f} m/s, N$_{{obs}}$={int(n_obs.sum())}"
        f"{subtitle}",
        color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUT_DIR / (f"fig_probabilities_{gwm['station']}_T{period:g}"
                     f"_{args.gmm}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stage", choices=["compute", "plot", "all"], nargs="?",
                        default="all")
    parser.add_argument("--gmm", choices=["none", *sorted(ATT_GMM_REGISTRY)],
                        default="edwardsfah13",
                        help="GMM to score against (plot-time only), or 'none'.")
    parser.add_argument("--period", type=float, default=0.3)
    parser.add_argument("--station", type=str, default=None,
                        help="Scenario station (default: most records in window).")
    parser.add_argument("--vs30_halfwidth", type=float, default=75.0)
    parser.add_argument("--mag_min", type=float, default=1.5)
    parser.add_argument("--mag_max", type=float, default=3.5)
    parser.add_argument("--mag_step", type=float, default=0.25)
    parser.add_argument("--dist_min", type=float, default=10.0)
    parser.add_argument("--dist_max", type=float, default=120.0,
                        help="Records beyond this are excluded everywhere.")
    parser.add_argument("--dist_step", type=float, default=10.0)
    parser.add_argument("--n_realizations", type=int, default=20)
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Deterministic evenly-spaced fraction of selected real records in (0, 1].",
    )
    parser.add_argument("--real_cache", type=str, default=None)
    parser.add_argument("--gwm_cache", type=str, default=None)
    parser.add_argument("--min_bin_count", type=int, default=5)
    parser.add_argument("--sa_floor", type=float, default=1e-5,
                        help="Exclude records with SA below this [m/s^2] "
                             "(miscalibrated response epochs; 0 disables).")
    args = parser.parse_args()

    if args.stage in ("compute", "all"):
        real_path, gwm_path = run_compute(args)
        if args.stage == "all":
            args.real_cache = args.real_cache or str(real_path)
            args.gwm_cache = args.gwm_cache or str(gwm_path)
    if args.stage in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
