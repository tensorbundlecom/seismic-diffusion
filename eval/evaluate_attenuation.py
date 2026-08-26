"""
Spectral-acceleration attenuation curves: GWM ensembles vs a GMM vs data
(GWM paper, Fig. 8).

For a fixed scenario (magnitude, station, Vs30, depth, azimuth, SNR) the
diffusion model is sampled --n_realizations times at each point of a
hypocentral-distance grid; the median and log10-std of the resulting SA
values give a continuous attenuation curve with uncertainty. The curve is
compared against a published GMM (median +/- 1 sigma, --gmm, swappable) and
against the real records inside the magnitude/Vs30 selection (gray dots and
binned medians +/- std).

Scenario construction: conditioning distance is epicentral, so events are
placed on the sphere along the station's median back-azimuth at the chosen
epicentral distances (fixed depth), reproducing the requested hypocentral
distances exactly. Magnitude defaults to the center of --mag_range; depth,
SNR, and azimuth default to medians of the selected records; the station
defaults to the one with most records in the selection, and the Vs30 window
is centered on its Vs30.

Adaptations vs the paper (documented, not corrected):
  - Periods default to 0.1 s and 0.3 s: the waveforms are bandpass filtered
    2-15 Hz, so the paper's T = 1.0 s (1 Hz) is outside the usable band.
  - SA is computed on the E component (the GMMs predict horizontal-mean /
    RotD50 measures - a few-percent definition offset).
  - The paper's magnitude bins are +/-0.1 wide; here the data selection
    spans the full --mag_range (default 2-3) while the GWM/GMM curves are
    evaluated at its center, so part of the data scatter is magnitude
    scaling within the bin.
  - Legacy per-event-normalized AEs use an AmplitudeMLP that pins each
    synthetic's counts-domain peak deterministically. Globally normalized AEs
    instead recover amplitude from the decoded spectrogram itself.

SA implementation: relative-displacement transfer function in the frequency
domain (zero-padded FFT), PSA = omega_n^2 * max|u|, 5% damping.

Run from the project root:
    python eval/evaluate_attenuation.py compute [--n_realizations 50] [--dist_step 5]
    python eval/evaluate_attenuation.py plot    [--bin_km 10]
    python eval/evaluate_attenuation.py all     [...]

Outputs (real-data cache, sweep cache, figure) go to eval/attenuation/.
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
    CHANNEL_NAMES, DIFF_DIR, FS, ROOT, GRAVITY,
    GWM_COLOR, GMM_COLOR, SCATTER_COLOR, INK, MUTED, GRID,
    _crop_pad, _deconvolve, find_synth_cache,
)
from embedding_artifacts import (  # noqa: E402
    artifact_path,
    deterministic_fraction_subset,
    deterministic_selection_tag,
    resolve_embeddings_dir,
)
from output_paths import evaluation_output_dir  # noqa: E402

OUT_DIR = evaluation_output_dir("attenuation")
DATA_COLOR = "#008300"  # validated palette slot 4 (green), as in the paper
DAMPING = 0.05
EARTH_R = 6371.0


def configure_embeddings_dir(path: str | Path | None) -> Path:
    """Keep this evaluator's sample rate aligned with the selected export."""
    import evaluate_peak_amplitudes as peaks

    global FS
    directory = peaks.configure_embeddings_dir(path)
    FS = peaks.FS
    return directory


# ── Spectral acceleration ─────────────────────────────────────────────────────
def spectral_accel(acc: np.ndarray, periods) -> list:
    """5%-damped PSA [same units as acc] via frequency-domain SDOF response."""
    from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

    acc = np.asarray(acc, dtype=np.float64)
    nfft = next_fast_len(acc.shape[0] + 2048)  # pad: oscillator decay room
    spec = rfft(acc, nfft)
    omega = 2.0 * np.pi * rfftfreq(nfft, 1.0 / FS)
    out = []
    for period in periods:
        wn = 2.0 * np.pi / period
        h = -1.0 / (wn ** 2 - omega ** 2 + 2j * DAMPING * wn * omega)
        u = irfft(spec * h, nfft)
        out.append(float(wn ** 2 * np.max(np.abs(u))))
    return out


# ── Workers ───────────────────────────────────────────────────────────────────
def _sa_real(args):
    """Worker: real mseed -> SA(periods) of physical E acceleration."""
    path_str, station, channel_type, event_id, channel_idx, periods, waveform_domain = args
    try:
        from obspy import read as obspy_read

        stream = obspy_read(path_str)
        if len(stream) != 3:
            return None
        stream.sort(keys=["channel"])
        trace = stream[channel_idx]
        if abs(trace.stats.sampling_rate - FS) > 1e-6:
            trace.resample(FS)
        code = f"{channel_type}{CHANNEL_NAMES[channel_idx]}"
        acc = _deconvolve(_crop_pad(trace.data.astype(np.float64)),
                          station, code, event_id, "ACC", waveform_domain)
        if acc is None:
            return None
        return spectral_accel(acc, periods)
    except Exception:
        return None


def _sa_synth(args):
    """Worker: checkpoint-domain synthetic -> physical-acceleration SA."""
    wave, station, channel_code, event_id, periods, waveform_domain = args
    try:
        acc = _deconvolve(np.asarray(wave, dtype=np.float64),
                          station, channel_code, event_id, "ACC", waveform_domain)
        if acc is None:
            return None
        return spectral_accel(acc, periods)
    except Exception:
        return None


# ── GMM SA predictions (median + ln-sigma; swappable) ─────────────────────────
def _sa_gmm_bssa14(mag, r_hyp, r_epi, vs30, periods):
    import pygmm

    med = np.full((len(periods), r_hyp.shape[0]), np.nan)
    ln_std = np.full_like(med, np.nan)
    for k in range(r_hyp.shape[0]):
        s = pygmm.Scenario(mag=float(mag), dist_jb=float(r_epi[k]),
                           v_s30=float(vs30), mechanism="U")
        m = pygmm.BooreStewartSeyhanAtkinson2014(s)
        med[:, k] = np.asarray(m.interp_spec_accels(periods)) * GRAVITY
        ln_std[:, k] = np.asarray(m.interp_ln_stds(periods))
    return med, ln_std


def _sa_gmm_oq(cls_name, dist_field, site_fields):
    def predict(mag, r_hyp, r_epi, vs30, periods):
        from openquake.hazardlib.contexts import simple_cmaker
        from openquake.hazardlib.gsim import get_available_gsims

        gsim = get_available_gsims()[cls_name]()
        cmaker = simple_cmaker([gsim], [f"SA({p})" for p in periods])
        ctx = cmaker.new_ctx(r_hyp.shape[0])
        ctx["mag"] = mag
        ctx[dist_field] = r_hyp
        if "vs30" in site_fields:
            ctx["vs30"] = vs30
        if "rake" in site_fields:
            ctx["rake"] = 0.0
        mean, sig, _, _ = cmaker.get_mean_stds([ctx])
        return np.exp(mean[0]) * GRAVITY, sig[0]

    return predict


ATT_GMM_REGISTRY = {
    "bssa14": {"label": "Boore et al. (2014)", "predict": _sa_gmm_bssa14},
    "atkinson15": {"label": "Atkinson (2015)",
                   "predict": _sa_gmm_oq("Atkinson2015", "rhypo", ())},
    "edwardsfah13": {"label": "Edwards & Fäh (2013)",
                     "predict": _sa_gmm_oq("EdwardsFah2013Alpine60Bars", "rrup",
                                           ("vs30", "rake"))},
}


# ── Geometry ──────────────────────────────────────────────────────────────────
def destination(lat, lon, bearing_deg, dist_km):
    """Spherical destination point (same R=6371 sphere as the conditioning)."""
    lat1, lon1, b = map(np.radians, (lat, lon, bearing_deg))
    d = dist_km / EARTH_R
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(b))
    lon2 = lon1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


def event_azimuth(ev_lat, ev_lon, st_lat, st_lon):
    """Event->station forward azimuth, matching create_conditioning_vector."""
    ev_lat, ev_lon, st_lat, st_lon = map(np.radians,
                                         (ev_lat, ev_lon, st_lat, st_lon))
    dlon = st_lon - ev_lon
    y = np.sin(dlon) * np.cos(st_lat)
    x = (np.cos(ev_lat) * np.sin(st_lat)
         - np.sin(ev_lat) * np.cos(st_lat) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


# ── Selection ─────────────────────────────────────────────────────────────────
def select_records(metadatas, station_locations, station_vs30, args):
    """Records in the magnitude/Vs30 window; scenario defaults from them."""
    from obspy.geodetics import gps2dist_azimuth

    mag_lo, mag_hi = args.mag_lo, args.mag_hi
    if args.station:
        sta_name = args.station
    else:
        counts = {}
        for m in metadatas:
            if mag_lo <= float(m["magnitude"]) <= mag_hi:
                counts[m["station_name"]] = counts.get(m["station_name"], 0) + 1
        sta_name = max(counts, key=counts.get)
    sta_vs30 = float(station_vs30[sta_name])
    vs_half = args.vs30_halfwidth
    vs_lo, vs_hi = sta_vs30 - vs_half, sta_vs30 + vs_half

    sel, r_hyp, r_epi = [], [], []
    for i, m in enumerate(metadatas):
        if not (mag_lo <= float(m["magnitude"]) <= mag_hi):
            continue
        if not (vs_lo <= float(station_vs30[m["station_name"]]) <= vs_hi):
            continue
        sta = station_locations[m["station_name"]]
        dist_m, _, _ = gps2dist_azimuth(m["latitude"], m["longitude"],
                                        sta["latitude"], sta["longitude"])
        sel.append(i)
        r_epi.append(dist_m / 1000.0)
        r_hyp.append(float(np.hypot(dist_m / 1000.0, float(m["depth"]))))
    return sta_name, sta_vs30, (vs_lo, vs_hi), sel, np.asarray(r_hyp), np.asarray(r_epi)


# ── compute stage ─────────────────────────────────────────────────────────────
def run_compute(args):
    from gwm_sampling import GwmSampler

    embeddings_dir = configure_embeddings_dir(args.embeddings_dir)
    sampler = GwmSampler(
        args.checkpoint, args.ae_checkpoint, args.waveform_domain, embeddings_dir
    )
    waveform_domain = sampler.waveform_domain
    metadatas = json.load(open(artifact_path(embeddings_dir, "metadata.json")))
    station_locations = json.load(open(artifact_path(embeddings_dir, "station_locations.json")))
    station_vs30 = json.load(open(artifact_path(embeddings_dir, "station_vs30.json")))
    periods = [float(p) for p in args.periods.split(",")]

    sta_name, sta_vs30, vs_window, sel, r_hyp_sel, r_epi_sel = select_records(
        metadatas, station_locations, station_vs30, args)
    print(f"[eval] selection: M{args.mag_lo:g}-{args.mag_hi:g}, "
          f"Vs30 {vs_window[0]:.0f}-{vs_window[1]:.0f} m/s "
          f"-> {len(sel)} records; scenario station {sta_name} "
          f"(Vs30 {sta_vs30:.0f} m/s)")

    # ── Real pass ────────────────────────────────────────────────────────────
    tag = (f"M{args.mag_lo:g}-{args.mag_hi:g}_vs{vs_window[0]:.0f}-"
           f"{vs_window[1]:.0f}_T{'-'.join(str(p) for p in periods)}")
    sel_lim = deterministic_fraction_subset(sel, args.fraction, args.limit)
    selected_positions = {value: index for index, value in enumerate(sel)}
    r_hyp_sel = r_hyp_sel[[selected_positions[value] for value in sel_lim]]
    print(
        f"[eval] deterministic real-data sample: {len(sel_lim)}/{len(sel)} "
        f"records (fraction={args.fraction:g})"
    )
    selection_tag = deterministic_selection_tag(sel_lim)
    real_path = OUT_DIR / (
        f"att_real_{tag}_domain{waveform_domain}_n{len(sel_lim)}_sel{selection_tag}.npz"
    )
    n = len(sel_lim)
    real = None
    if real_path.exists():
        old = dict(np.load(real_path))
        if old["indices"].tolist() == sel_lim:
            real = old
        else:
            print(f"[eval] {real_path.name} covers a different selection; rebuilding.")
    if real is None:
        real = {
            "indices": np.asarray(sel_lim, dtype=np.int64),
            "waveform_domain": np.asarray(waveform_domain),
            "periods": np.asarray(periods),
            "r_hyp": r_hyp_sel.astype(np.float64),
            "sa": np.full((n, len(periods)), np.nan),
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
                    m = metadatas[sel_lim[j]]
                    jobs.append((resolve(m), m["station_name"],
                                 m.get("channel_type", "HH"),
                                 str(m.get("event_id", "")), 0, periods,
                                 waveform_domain))
                for j, out in zip(rows, pool.map(_sa_real, jobs, chunksize=8)):
                    if out is not None:
                        real["sa"][j] = out
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

    # ── Scenario definition ──────────────────────────────────────────────────
    scen_mag = args.scenario_mag if args.scenario_mag else 0.5 * (args.mag_lo + args.mag_hi)
    sel_meta = [metadatas[i] for i in sel]
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
    print(f"[eval] scenario: M{scen_mag:g}, depth {depth:.1f} km, snr {snr:.0f}, "
          f"azimuth {az:.0f} deg, station {sta_name}")

    r_hyp_grid = np.arange(args.dist_min, args.dist_max + args.dist_step,
                           args.dist_step, dtype=np.float64)
    r_hyp_grid = r_hyp_grid[r_hyp_grid > depth + 0.5]  # need r_epi > 0
    r_epi_grid = np.sqrt(r_hyp_grid ** 2 - depth ** 2)

    # Constructing the sampler also resolves the checkpoint-bound AE contract.
    # Its content-addressed tag prevents stale legacy/global reconstructions
    # from sharing an evaluation cache.
    sweep_path = OUT_DIR / (f"att_sweep_{tag}_M{scen_mag:g}_{sta_name}"
                            f"_model{sampler.cache_tag}"
                            f"_n{args.n_realizations}.npz")
    if sweep_path.exists():
        print(f"[eval] sweep cache exists: {sweep_path.name}")
        return real_path, sweep_path

    # ── GWM sweep ────────────────────────────────────────────────────────────
    # One fabricated meta per grid distance (event on the median back-azimuth).
    metas = []
    for d_epi in r_epi_grid:
        ev_lat, ev_lon = destination(sta["latitude"], sta["longitude"],
                                     (az + 180.0) % 360.0, float(d_epi))
        metas.append({
            "magnitude": scen_mag, "latitude": ev_lat, "longitude": ev_lon,
            "depth": depth, "snr": snr, "station_name": sta_name,
            "station_idx": station_idx, "channel_idx": 0,
            "channel_type": channel_type, "event_id": "",
        })

    pairs = [(di, ri) for di in range(len(r_hyp_grid))
             for ri in range(args.n_realizations)]
    sa_synth = np.full((len(r_hyp_grid), args.n_realizations, len(periods)), np.nan)
    code = f"{channel_type}{CHANNEL_NAMES[0]}"
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            waves = sampler.generate([metas[di] for di, _ in batch], args.steps,
                                     args.gl_iters, args.seed + start, pool)
            sa_jobs = [(w.astype(np.float64), sta_name, code, "", periods,
                        waveform_domain)
                       for w in waves if w is not None]
            keep = [k for k, w in enumerate(waves) if w is not None]
            for k, out in zip(keep, pool.map(_sa_synth, sa_jobs)):
                if out is not None:
                    di, ri = batch[k]
                    sa_synth[di, ri] = out
            done = min(start + len(batch), len(pairs))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(pairs) - done) / max(rate, 1e-9) / 60.0
            print(f"[eval] sweep {done}/{len(pairs)} "
                  f"({rate:.1f} samples/s, ETA {eta:.0f} min)")

    # ── GMM curves on the same grid ──────────────────────────────────────────
    gmm = ATT_GMM_REGISTRY[args.gmm]
    gmm_med, gmm_ln_std = gmm["predict"](scen_mag, r_hyp_grid, r_epi_grid,
                                         sta_vs30, periods)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(sweep_path, r_hyp=r_hyp_grid, periods=np.asarray(periods),
             sa_synth=sa_synth, gmm_med=gmm_med, gmm_ln_std=gmm_ln_std,
             gmm_name=args.gmm, scen_mag=scen_mag, station=sta_name,
             vs30=sta_vs30, depth=depth, mag_lo=args.mag_lo, mag_hi=args.mag_hi,
             vs_lo=vs_window[0], vs_hi=vs_window[1],
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

    real = dict(np.load(latest("att_real_*.npz", args.real_cache)))
    sweep = dict(np.load(latest("att_sweep_*.npz", args.sweep_cache),
                         allow_pickle=True))
    periods = sweep["periods"]
    gmm_label = ATT_GMM_REGISTRY[str(sweep["gmm_name"])]["label"]
    print(f"[eval] plotting: {len(real['indices'])} records, "
          f"{sweep['sa_synth'].shape[1]} realizations/distance, gmm={sweep['gmm_name']}")

    fig, axes = plt.subplots(1, len(periods), figsize=(6.0 * len(periods), 5.0),
                             sharex=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    # Ignore stray records beyond the sweep grid (unfiltered outliers).
    data_max = args.data_max_km if args.data_max_km else float(sweep["r_hyp"].max())
    # Drop records with implausibly small amplitudes (bad response epochs,
    # e.g. MRMT 2019-2020: gain off by ~2 orders of magnitude). Keyed on the
    # first period and applied record-wide so all panels stay consistent.
    good = np.ones(real["sa"].shape[0], dtype=bool)
    if args.sa_floor > 0:
        sa0 = real["sa"][:, 0]
        good = ~(np.isfinite(sa0) & (sa0 > 0) & (sa0 < args.sa_floor))
        if (~good).sum():
            print(f"[eval] sa_floor {args.sa_floor:g}: excluding "
                  f"{int((~good).sum())} records (bad response epochs)")

    for col, (ax, period) in enumerate(zip(axes, periods)):
        sa_r = real["sa"][:, col]
        ok = (np.isfinite(sa_r) & (sa_r > 0) & (real["r_hyp"] <= data_max)
              & good)
        ax.scatter(real["r_hyp"][ok], sa_r[ok], s=3, color=SCATTER_COLOR,
                   alpha=0.3, lw=0, rasterized=True, label="Data")

        edges = np.arange(0.0, real["r_hyp"][ok].max() + args.bin_km, args.bin_km)
        for lo, hi in zip(edges[:-1], edges[1:]):
            bsel = ok & (real["r_hyp"] >= lo) & (real["r_hyp"] < hi)
            if bsel.sum() < args.min_bin_count:
                continue
            logs = np.log10(real["sa"][bsel, col])
            c = 0.5 * (lo + hi)
            med, sd = np.median(logs), logs.std()
            ax.plot(c, 10 ** med, "s", ms=5, color=DATA_COLOR, zorder=4,
                    label="Data-bin med." if lo == edges[0] else None)
            ax.plot([c, c], [10 ** (med - sd), 10 ** (med + sd)], color=DATA_COLOR,
                    lw=1.4, zorder=3)

        sa_s = sweep["sa_synth"][:, :, col]
        med = np.nanmedian(sa_s, axis=1)
        log_sd = np.nanstd(np.log10(np.maximum(sa_s, 1e-30)), axis=1)
        ax.fill_between(sweep["r_hyp"], 10 ** (np.log10(med) - log_sd),
                        10 ** (np.log10(med) + log_sd), color=GWM_COLOR,
                        alpha=0.22, lw=0, label="GWM-std.")
        ax.plot(sweep["r_hyp"], med, color=INK, lw=1.6, label="GWM-med.", zorder=5)

        gm, gs = sweep["gmm_med"][col], sweep["gmm_ln_std"][col]
        ax.plot(sweep["r_hyp"], gm, color=GMM_COLOR, lw=1.6, label=gmm_label)
        for sgn in (-1, 1):
            ax.plot(sweep["r_hyp"], gm * np.exp(sgn * gs), color=GMM_COLOR,
                    lw=1.0, ls="--")

        ax.set_yscale("log")
        ax.set_xlim(0, data_max * 1.04)
        ax.set_xlabel("Hypocentral Distance [km]", color=INK)
        ax.set_ylabel(f"SA({period:g}s) [m/s$^2$]", color=INK)
        ax.set_title(
            f"V$_{{S30}}$: {float(sweep['vs_lo']):.0f}-{float(sweep['vs_hi']):.0f} m/s,  "
            f"M{float(sweep['mag_lo']):g}-{float(sweep['mag_hi']):g},  "
            f"T={period:g}s,  N$_{{obs}}$={int(ok.sum())}",
            fontsize=9, color=INK)
        ax.text(-0.08, 1.04, "ab cd"[col] + ")", transform=ax.transAxes,
                fontsize=12, color=INK)
        if col == 0:
            ax.legend(fontsize=7.5, loc="lower left", frameon=True, edgecolor=GRID)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)

    fig.suptitle(
        f"SA attenuation — GWM (M{float(sweep['scen_mag']):g}, station "
        f"{sweep['station']}) vs {gmm_label} vs data, E component",
        color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = OUT_DIR / (f"fig_attenuation_M{float(sweep['scen_mag']):g}"
                     f"_{sweep['station']}_{sweep['gmm_name']}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stage", choices=["compute", "plot", "all"], nargs="?",
                        default="all")
    parser.add_argument("--gmm", choices=sorted(ATT_GMM_REGISTRY),
                        default="edwardsfah13",
                        help="GMM overlay (median + 1 sigma), swappable.")
    parser.add_argument("--mag_lo", type=float, default=2.0)
    parser.add_argument("--mag_hi", type=float, default=3.0)
    parser.add_argument("--scenario_mag", type=float, default=None,
                        help="GWM/GMM curve magnitude (default: bin center).")
    parser.add_argument("--periods", type=str, default="0.1,0.3",
                        help="SA periods [s], comma separated (keep inside "
                             "the 2-15 Hz waveform band).")
    parser.add_argument("--station", type=str, default=None,
                        help="Scenario station (default: most records in "
                             "the magnitude window).")
    parser.add_argument("--vs30_halfwidth", type=float, default=75.0,
                        help="Data selection: Vs30 window half-width around "
                             "the scenario station's Vs30.")
    parser.add_argument("--dist_min", type=float, default=10.0)
    parser.add_argument("--dist_max", type=float, default=120.0)
    parser.add_argument("--dist_step", type=float, default=5.0)
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
    parser.add_argument("--bin_km", type=float, default=10.0)
    parser.add_argument("--min_bin_count", type=int, default=5)
    parser.add_argument("--data_max_km", type=float, default=None,
                        help="Ignore records beyond this hypocentral distance "
                             "in the plot (default: the sweep grid maximum).")
    parser.add_argument("--sa_floor", type=float, default=1e-5,
                        help="Exclude records whose SA at the first period is "
                             "below this [m/s^2] - filters miscalibrated "
                             "response epochs (0 disables).")
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
