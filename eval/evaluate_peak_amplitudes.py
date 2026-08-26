"""
Peak ground motion bias vs hypocentral distance (GWM paper, Fig. 4).

Produces a 2x2 figure of log10(observed / predicted) peak amplitudes against
hypocentral distance, with binned mean +/- std overlays:

  a,b) Generative waveform model (GWM): PGA and PGV measured on the diffusion
       synthetics from the evaluate_first_order.py cache (one synthetic per
       test-split record, matched conditioning, E component). Real and
       synthetic waveforms go through the same domain-aware physical-motion
       conversion as real data, so the ratio compares like with like.
  c,d) A ground motion model chosen with --gmm, for records with
       M >= --gmm_min_mag (default: the model's validity floor). Observed
       peaks are RotD50 of the two horizontal components.

Available GMMs (--gmm):
  bssa14        Boore et al. (2014), NGA-West2 (pygmm). Floor M3.0.
                R_JB ~ R_epi (point source), Vs30 150-1500 m/s, mechanism
                unspecified. Predicts RotD50.
  atkinson15    Atkinson (2015), small-to-moderate events at short hypocentral
                distances (openquake.hazardlib). Floor M3.0. No site term;
                calibrated for R_hyp < ~50-60 km, so most of this dataset
                extrapolates it in distance. Predicts RotD50.
  edwardsfah13  Edwards & Faeh (2013), Swiss shallow crustal model, Alpine
                60-bar branch (openquake.hazardlib). Floor M2.0 - the option
                that maximises overlap with the test split (~4,100 records vs
                ~590 at M3). R_rup ~ R_hyp, Vs30-dependent, rake fixed to 0
                (Marmara is dominantly strike-slip). Predicts the geometric
                mean of the horizontals (~ RotD50 to within a few percent).

Known approximations (documented, not corrected):
  - Catalog magnitudes are ML; the GMMs expect Mw. The scales diverge at
    small magnitudes, which maps directly into apparent bias.
  - All waveforms are bandpass filtered (2-15 Hz dataset), so observed peaks
    are band-limited while GMMs predict broadband motion (mainly depresses
    observed PGV).
  - Point-source distances: R_JB ~ R_epi, R_rup ~ R_hyp.
  - The GWM row uses the E component only: the first_order cache stores one
    channel, and re-sampling the diffusion model for N is expensive.
  - None of the GMMs is calibrated on Turkish data; the GMM row is a sanity
    reference, not a strict benchmark.

Run from the project root:
    python eval/evaluate_peak_amplitudes.py compute [--gmm bssa14] [--limit N]
    python eval/evaluate_peak_amplitudes.py plot    [--set_mode matched] [--bin_km 5]
    python eval/evaluate_peak_amplitudes.py all     [...]

`compute` is resumable and the cache is shared across GMMs: observed and
synthetic peaks are stored once, GMM predictions are added per model, and
the record set only ever grows (a --limit smoke test never discards
previously computed records). --scope val (default) restricts the GMM row to
test-split records, which is enough for --set_mode val/matched and avoids
processing tens of thousands of extra waveforms; use --scope all (with
--set_mode all) to evaluate the GMM on every eligible record instead.
`plot` only needs the cache. Synthetic waveforms are NOT regenerated here;
they retain the domain recorded in the evaluate_first_order.py cache.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ML.diffusion.waveform_domain import (
    INSTRUMENT_COUNTS,
    normalize_waveform_domain,
    physical_acceleration_to_motion,
)
from embedding_artifacts import artifact_path, resolve_embeddings_dir
from output_paths import evaluation_output_dir

DIFF_DIR = ROOT / "ML" / "diffusion"
FIRST_ORDER_DIR = evaluation_output_dir("first_order")
OUT_DIR = evaluation_output_dir("peak_amplitudes")
RESPONSES_XML = ROOT / "eval" / "station_responses.xml"
CHANNEL_NAMES = ["E", "N", "Z"]

# Response deconvolution (counts -> ground motion), same as evaluate_first_order.
RESPONSE_WATER_LEVEL = 60
GRAVITY = 9.80665          # GMM PGA outputs are in g
ROTD_ANGLES_DEG = 180      # RotD50: 1-degree rotation grid over [0, 180)
VS30_RANGE = (150.0, 1500.0)  # BSSA14 validity; excludes bad lookups (GADA: 3.3 m/s)

EMBEDDINGS_DIR = resolve_embeddings_dir(None)
_DEFAULT_STFT = {
    "nperseg": 256,
    "noverlap": 192,
    "nfft": 256,
    "resample_hz": 100.0,
    "target_seconds": 70.0,
}
_source = {"stft": dict(_DEFAULT_STFT)}
_default_source_path = artifact_path(EMBEDDINGS_DIR, "source.json")
if _default_source_path.is_file():
    with _default_source_path.open(encoding="utf-8") as handle:
        _source = json.load(handle)
STFT_CFG = _source["stft"]
FS = float(STFT_CFG.get("resample_hz", 100.0))
TARGET_SAMPLES = int(round(FS * float(STFT_CFG.get("target_seconds", 70.0))))


def configure_embeddings_dir(path: str | Path | None) -> Path:
    """Select the embedding export used for metadata and STFT geometry."""
    global EMBEDDINGS_DIR, _source, STFT_CFG, FS, TARGET_SAMPLES
    EMBEDDINGS_DIR = resolve_embeddings_dir(path)
    with artifact_path(EMBEDDINGS_DIR, "source.json").open(encoding="utf-8") as handle:
        _source = json.load(handle)
    STFT_CFG = _source["stft"]
    FS = float(STFT_CFG.get("resample_hz", 100.0))
    TARGET_SAMPLES = int(round(FS * float(STFT_CFG.get("target_seconds", 70.0))))
    return EMBEDDINGS_DIR


# ── Response deconvolution (runs in worker processes) ─────────────────────────
# Copied from evaluate_first_order.py so this script stays free of the
# torch/diffusers stack: everything here runs on cached or on-disk waveforms.
_INVENTORY = None


def _get_inventory():
    global _INVENTORY
    if _INVENTORY is None:
        from obspy import read_inventory

        _INVENTORY = read_inventory(str(RESPONSES_XML))
    return _INVENTORY


def _find_channel_epoch(station: str, channel_code: str, t):
    """Return (network, location, time) of the response epoch covering t,
    falling back to the most recent epoch with a valid response."""
    fallback = None
    for net in _get_inventory():
        for sta in net:
            if sta.code != station:
                continue
            for cha in sta:
                if cha.code != channel_code or cha.response is None:
                    continue
                if not cha.response.instrument_sensitivity:
                    continue
                start, end = cha.start_date, cha.end_date
                if (t is not None and start is not None and start <= t
                        and (end is None or t <= end)):
                    return net.code, cha.location_code, t
                epoch_time = (start + 86400.0) if start is not None else t
                fallback = (net.code, cha.location_code, epoch_time)
    return fallback


def _deconvolve(data: np.ndarray, station: str, channel_code: str,
                event_time_str: str, output: str,
                waveform_domain: str = INSTRUMENT_COUNTS):
    """Convert counts or physical acceleration to the requested ground motion."""
    if waveform_domain != INSTRUMENT_COUNTS:
        return physical_acceleration_to_motion(data, output, FS)
    from obspy import Trace, UTCDateTime

    try:
        t = UTCDateTime.strptime(event_time_str, "%Y%m%d%H%M%S")
    except Exception:
        t = None
    epoch = _find_channel_epoch(station, channel_code, t)
    if epoch is None:
        return None
    network, location, starttime = epoch

    trace = Trace(
        data=np.asarray(data, dtype=np.float64),
        header={
            "network": network,
            "station": station,
            "location": location,
            "channel": channel_code,
            "starttime": starttime,
            "sampling_rate": FS,
        },
    )
    try:
        trace.remove_response(
            inventory=_get_inventory(),
            output=output,
            water_level=RESPONSE_WATER_LEVEL,
        )
    except Exception:
        return None
    return trace.data


# ── Peak measures ─────────────────────────────────────────────────────────────
def _crop_pad(x: np.ndarray) -> np.ndarray:
    n = TARGET_SAMPLES
    return x[:n] if x.shape[0] >= n else np.pad(x, (0, n - x.shape[0]))


def _rotd50(e: np.ndarray, n: np.ndarray) -> float:
    """Median over rotation angles of the peak absolute rotated horizontal."""
    theta = np.deg2rad(np.arange(ROTD_ANGLES_DEG, dtype=np.float64))
    rotated = np.cos(theta)[:, None] * e[None, :] + np.sin(theta)[:, None] * n[None, :]
    return float(np.median(np.max(np.abs(rotated), axis=1)))


def _process_real(args):
    """Worker: real mseed -> (pga_e, pgv_e, pga_rotd50, pgv_rotd50)."""
    path_str, station, channel_type, event_id, waveform_domain = args
    out = [np.nan] * 4
    try:
        from obspy import read as obspy_read

        stream = obspy_read(path_str)
        if len(stream) != 3:
            return out
        stream.sort(keys=["channel"])
        counts = {}
        for comp in ("E", "N"):
            trace = stream[CHANNEL_NAMES.index(comp)]
            if abs(trace.stats.sampling_rate - FS) > 1e-6:
                trace.resample(FS)
            counts[comp] = _crop_pad(trace.data.astype(np.float64))

        motion = {}
        for kind in ("ACC", "VEL"):
            for comp in ("E", "N"):
                motion[kind, comp] = _deconvolve(
                    counts[comp], station, f"{channel_type}{comp}", event_id, kind,
                    waveform_domain,
                )

        if motion["ACC", "E"] is not None:
            out[0] = float(np.max(np.abs(motion["ACC", "E"])))
        if motion["VEL", "E"] is not None:
            out[1] = float(np.max(np.abs(motion["VEL", "E"])))
        if motion["ACC", "E"] is not None and motion["ACC", "N"] is not None:
            out[2] = _rotd50(motion["ACC", "E"], motion["ACC", "N"])
        if motion["VEL", "E"] is not None and motion["VEL", "N"] is not None:
            out[3] = _rotd50(motion["VEL", "E"], motion["VEL", "N"])
    except Exception:
        pass
    return out


def _process_synth(args):
    """Worker: cached counts-domain synthetic -> (pga, pgv)."""
    wave, station, channel_code, event_id, waveform_domain = args
    out = [np.nan, np.nan]
    try:
        data = np.asarray(wave, dtype=np.float64)
        acc = _deconvolve(data, station, channel_code, event_id, "ACC", waveform_domain)
        vel = _deconvolve(data, station, channel_code, event_id, "VEL", waveform_domain)
        if acc is not None:
            out[0] = float(np.max(np.abs(acc)))
        if vel is not None:
            out[1] = float(np.max(np.abs(vel)))
    except Exception:
        pass
    return out


# ── GMM predictions ───────────────────────────────────────────────────────────
def _predict_bssa14(mag, r_hyp, r_epi, vs30):
    """BSSA14 via pygmm. Median PGA (m/s^2) and PGV (m/s), NaN where the
    station Vs30 is outside the model's validity range."""
    import pygmm

    pga = np.full(mag.shape, np.nan)
    pgv = np.full(mag.shape, np.nan)
    for k in range(mag.shape[0]):
        if not (VS30_RANGE[0] <= vs30[k] <= VS30_RANGE[1]):
            continue
        scenario = pygmm.Scenario(
            mag=float(mag[k]), dist_jb=float(r_epi[k]),
            v_s30=float(vs30[k]), mechanism="U",
        )
        model = pygmm.BooreStewartSeyhanAtkinson2014(scenario)
        pga[k] = float(model.pga) * GRAVITY
        pgv[k] = float(model.pgv) / 100.0
    return pga, pgv


def _predict_oq(cls_name, dist_field, site_fields):
    """openquake.hazardlib GSIM -> vectorized median PGA (m/s^2), PGV (m/s)."""

    def predict(mag, r_hyp, r_epi, vs30):
        from openquake.hazardlib.contexts import simple_cmaker
        from openquake.hazardlib.gsim import get_available_gsims

        gsim = get_available_gsims()[cls_name]()
        cmaker = simple_cmaker([gsim], ["PGA", "PGV"])
        ctx = cmaker.new_ctx(mag.shape[0])
        ctx["mag"] = mag
        ctx[dist_field] = r_hyp
        if "vs30" in site_fields:
            ctx["vs30"] = vs30
        if "rake" in site_fields:
            ctx["rake"] = 0.0
        mean, _, _, _ = cmaker.get_mean_stds([ctx])
        pga = np.exp(mean[0, 0]) * GRAVITY
        pgv = np.exp(mean[0, 1]) / 100.0
        if "vs30" in site_fields:
            bad = (vs30 < VS30_RANGE[0]) | (vs30 > VS30_RANGE[1])
            pga[bad] = np.nan
            pgv[bad] = np.nan
        return pga, pgv

    return predict


GMM_REGISTRY = {
    "bssa14": {
        "label": "Boore et al. (2014)",
        "min_mag": 3.0,
        "predict": _predict_bssa14,
    },
    "atkinson15": {
        "label": "Atkinson (2015)",
        "min_mag": 3.0,
        "predict": _predict_oq("Atkinson2015", "rhypo", ()),
    },
    "edwardsfah13": {
        "label": "Edwards & Fäh (2013)",
        "min_mag": 2.0,
        "predict": _predict_oq("EdwardsFah2013Alpine60Bars", "rrup",
                               ("vs30", "rake")),
    },
}


# ── Cache ─────────────────────────────────────────────────────────────────────
PEAK_KEYS = ("pga_obs_e", "pgv_obs_e", "pga_obs_rot", "pgv_obs_rot",
             "pga_synth_e", "pgv_synth_e")


def find_synth_cache() -> Path:
    caches = sorted(FIRST_ORDER_DIR.glob("cache_*.npz"), key=lambda p: p.stat().st_mtime)
    if not caches:
        raise FileNotFoundError(
            f"No synthetic cache in {FIRST_ORDER_DIR}; run evaluate_first_order.py first."
        )
    return caches[-1]


def cache_path(synth_cache: Path) -> Path:
    return OUT_DIR / f"peaks_{synth_cache.stem.removeprefix('cache_')}.npz"


def init_or_load_cache(path: Path, indices, val_set):
    """Cache covering `indices`, unioned with any existing cache: the record
    set only ever grows, so previously computed peaks survive --limit smoke
    tests and scope/GMM changes."""
    old = dict(np.load(path)) if path.exists() else None
    if old is not None:
        old_indices = old["indices"].tolist()
        merged = sorted(set(indices) | set(old_indices))
        if len(merged) > len(indices):
            print(f"[eval] cache union: {len(indices)} requested + existing "
                  f"{path.name} -> {len(merged)} records")
        indices = merged
    n = len(indices)
    nanf = lambda: np.full(n, np.nan, dtype=np.float64)  # noqa: E731
    cache = {
        "indices": np.asarray(indices, dtype=np.int64),
        "in_val": np.asarray([i in val_set for i in indices], dtype=bool),
        "mag": nanf(), "r_hyp": nanf(), "r_epi": nanf(), "vs30": nanf(),
        **{k: nanf() for k in PEAK_KEYS},
        "done_real": np.zeros(n, dtype=bool),
        "done_synth": np.zeros(n, dtype=bool),
    }
    if old is None:
        return cache
    if old_indices == indices:
        old["in_val"] = cache["in_val"]
        return old
    pos = {idx: i for i, idx in enumerate(old_indices)}
    new_rows, old_rows = zip(*[(j, pos[idx]) for j, idx in enumerate(indices)
                               if idx in pos])
    new_rows, old_rows = np.asarray(new_rows), np.asarray(old_rows)
    for key, arr in old.items():
        if key in ("indices", "in_val", "waveform_domain"):
            continue
        if key not in cache:  # per-GMM prediction arrays
            cache[key] = np.full(n, np.nan, dtype=np.float64)
        cache[key][new_rows] = arr[old_rows]
    print(f"[eval] carried over {len(new_rows)} records from existing {path.name}")
    return cache


def save_cache(path: Path, cache: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp.npz")
    np.savez(tmp, **cache)
    tmp.replace(path)


# ── compute stage ─────────────────────────────────────────────────────────────
def run_compute(args):
    from obspy.geodetics import gps2dist_azimuth

    gmm = GMM_REGISTRY[args.gmm]
    min_mag = args.gmm_min_mag if args.gmm_min_mag is not None else gmm["min_mag"]

    metadatas = json.load(open(artifact_path(EMBEDDINGS_DIR, "metadata.json")))
    station_locations = json.load(open(artifact_path(EMBEDDINGS_DIR, "station_locations.json")))
    station_vs30 = json.load(open(artifact_path(EMBEDDINGS_DIR, "station_vs30.json")))

    synth_cache_path = Path(args.cache) if args.cache else find_synth_cache()
    with np.load(synth_cache_path) as sc:
        synth_indices = sc["indices"].astype(int).tolist()
        wave_synth = sc["wave_synth"]
        waveform_domain = normalize_waveform_domain(
            str(sc["waveform_domain"].item())
            if "waveform_domain" in sc.files else INSTRUMENT_COUNTS
        )
    if waveform_domain == INSTRUMENT_COUNTS and not RESPONSES_XML.exists():
        raise FileNotFoundError(
            f"Missing {RESPONSES_XML}. Run eval/fetch_station_responses.py first."
        )
    channel = synth_cache_path.stem.rsplit("_ch", 1)[-1].split("_")[0]
    print(f"[eval] synthetics: {synth_cache_path.name} "
          f"({len(synth_indices)} samples, channel {channel})")

    val_set = set(synth_indices)
    eligible = {i for i, m in enumerate(metadatas)
                if float(m["magnitude"]) >= min_mag}
    if args.scope == "val":
        eligible &= val_set
    indices = sorted(val_set | eligible)
    if args.limit and args.limit < len(indices):
        picks = np.linspace(0, len(indices) - 1, num=args.limit, dtype=int)
        indices = [indices[i] for i in picks]
    path = cache_path(synth_cache_path)
    cache = init_or_load_cache(path, indices, val_set)
    cache["waveform_domain"] = np.asarray(waveform_domain)
    indices = cache["indices"].tolist()
    n_gmm = sum(1 for i in indices if i in eligible)
    print(f"[eval] records: {len(indices)} total "
          f"(GWM row: {int(cache['in_val'].sum())}, "
          f"{args.gmm} row M>={min_mag:g}, scope={args.scope}: {n_gmm})")
    synth_row = {idx: j for j, idx in enumerate(synth_indices)}

    # Scalar metadata (cheap, recomputed every run).
    for j, idx in enumerate(indices):
        m = metadatas[idx]
        sta = station_locations[m["station_name"]]
        dist_m, _, _ = gps2dist_azimuth(
            m["latitude"], m["longitude"], sta["latitude"], sta["longitude"]
        )
        cache["mag"][j] = float(m["magnitude"])
        cache["r_epi"][j] = dist_m / 1000.0
        cache["r_hyp"][j] = float(np.hypot(dist_m / 1000.0, float(m["depth"])))
        cache["vs30"][j] = float(station_vs30[m["station_name"]])

    def resolve(m):
        p = Path(m["file_path"])
        return str(p if p.is_absolute() else (DIFF_DIR / p).resolve())

    def run_pool(todo, jobs, done_key, apply, label):
        t0, n_done = time.time(), 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for start in range(0, len(todo), args.chunk):
                rows = todo[start:start + args.chunk]
                for j, out in zip(rows, pool.map(
                        _process_real if label == "real" else _process_synth,
                        [jobs[j] for j in rows], chunksize=8)):
                    apply(j, out)
                    cache[done_key][j] = True
                n_done += len(rows)
                save_cache(path, cache)
                rate = n_done / max(time.time() - t0, 1e-9)
                eta_min = (len(todo) - n_done) / max(rate, 1e-9) / 60.0
                print(f"[eval] {label} {n_done}/{len(todo)} "
                      f"({rate:.1f} rec/s, ETA {eta_min:.0f} min)")

    # ── Real pass: observed peaks (E component and RotD50) ──────────────────
    todo_real = [j for j in range(len(indices)) if not cache["done_real"][j]]
    if todo_real:
        print(f"[eval] real pass: {len(todo_real)} waveforms ({args.num_workers} workers)")
        jobs = {
            j: (resolve(metadatas[indices[j]]),
                metadatas[indices[j]]["station_name"],
                metadatas[indices[j]].get("channel_type", "HH"),
                str(metadatas[indices[j]].get("event_id", "")), waveform_domain)
            for j in todo_real
        }

        def apply_real(j, out):
            (cache["pga_obs_e"][j], cache["pgv_obs_e"][j],
             cache["pga_obs_rot"][j], cache["pgv_obs_rot"][j]) = out

        run_pool(todo_real, jobs, "done_real", apply_real, "real")
        n_failed = int(np.isnan(cache["pga_obs_e"][todo_real]).sum())
        if n_failed:
            print(f"[eval] real pass: {n_failed} waveforms failed (read/response)")

    # ── Synthetic pass: peaks of the cached diffusion waveforms ─────────────
    todo_synth = [j for j in range(len(indices))
                  if cache["in_val"][j] and not cache["done_synth"][j]]
    if todo_synth:
        print(f"[eval] synthetic pass: {len(todo_synth)} waveforms")
        jobs = {}
        for j in todo_synth:
            m = metadatas[indices[j]]
            code = f"{m.get('channel_type', 'HH')}{channel}"
            jobs[j] = (wave_synth[synth_row[indices[j]]], m["station_name"],
                       code, str(m.get("event_id", "")), waveform_domain)

        def apply_synth(j, out):
            cache["pga_synth_e"][j], cache["pgv_synth_e"][j] = out

        run_pool(todo_synth, jobs, "done_synth", apply_synth, "synth")

    # ── GMM pass: median predictions (fast, vectorized, main process) ───────
    rows = np.asarray([j for j, idx in enumerate(indices) if idx in eligible])
    if rows.size:
        print(f"[eval] {args.gmm} pass: {rows.size} scenarios")
        pga, pgv = gmm["predict"](
            cache["mag"][rows], cache["r_hyp"][rows],
            cache["r_epi"][rows], cache["vs30"][rows],
        )
        pga_key, pgv_key = f"pga_gmm_{args.gmm}", f"pgv_gmm_{args.gmm}"
        for key in (pga_key, pgv_key):
            if key not in cache:
                cache[key] = np.full(len(indices), np.nan, dtype=np.float64)
        cache[pga_key][rows], cache[pgv_key][rows] = pga, pgv
        n_skipped = int(np.isnan(pga).sum())
        if n_skipped:
            print(f"[eval] {args.gmm} pass: {n_skipped} records skipped "
                  f"(outside model validity, e.g. Vs30)")
        save_cache(path, cache)

    print(f"[eval] cache written: {path}")
    return path


# ── plot stage ────────────────────────────────────────────────────────────────
# Panel colors from the validated categorical palette (dataviz reference):
# blue for the GWM rows, indigo for the GMM rows (paper: blue / violet).
GWM_COLOR, GMM_COLOR = "#2a78d6", "#4a3aa7"
SCATTER_COLOR = "#a6a49d"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _binned(x, y, bin_km, min_count):
    edges = np.arange(0.0, np.nanmax(x) + bin_km, bin_km)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < min_count:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(np.mean(y[sel]))
        stds.append(np.std(y[sel]))
    return np.asarray(centers), np.asarray(means), np.asarray(stds)


def run_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.peaks_cache:
        path = Path(args.peaks_cache)
    else:
        caches = sorted(OUT_DIR.glob("peaks_*.npz"), key=lambda p: p.stat().st_mtime)
        if not caches:
            raise FileNotFoundError(f"No cache in {OUT_DIR}; run `compute` first.")
        path = caches[-1]
    cache = dict(np.load(path))
    print(f"[eval] plotting from {path.name}  (gmm={args.gmm}, set_mode={args.set_mode})")

    gmm = GMM_REGISTRY[args.gmm]
    min_mag = args.gmm_min_mag if args.gmm_min_mag is not None else gmm["min_mag"]
    pga_key, pgv_key = f"pga_gmm_{args.gmm}", f"pgv_gmm_{args.gmm}"
    if pga_key not in cache:
        raise KeyError(f"No {args.gmm} predictions in {path.name}; "
                       f"run `compute --gmm {args.gmm}` first.")

    r_hyp = cache["r_hyp"]
    in_val = cache["in_val"].astype(bool)
    in_gmm = (np.isfinite(cache[pga_key]) & np.isfinite(cache[pgv_key])
              & (cache["mag"] >= min_mag))
    gwm_label = "Generative waveform model"
    gmm_label = f"{gmm['label']}, M$\\geq${min_mag:g}"
    if args.set_mode == "all":
        gwm_sel, gmm_sel = in_val, in_gmm
    elif args.set_mode == "val":
        gwm_sel, gmm_sel = in_val, in_gmm & in_val
        gmm_label += ", test split"
    else:  # matched: identical record set in every panel
        finite = np.ones(in_val.shape, dtype=bool)
        for key in PEAK_KEYS:
            finite &= np.isfinite(cache[key]) & (cache[key] > 0)
        gwm_sel = gmm_sel = in_val & in_gmm & finite
        gwm_label += f", M$\\geq${min_mag:g}"
        gmm_label += ", test split"

    gmm_bin_km = args.gmm_bin_km if args.gmm_bin_km else args.bin_km
    panels = [
        ("a)", "PGA", "pga_obs_e", "pga_synth_e", gwm_sel,
         gwm_label, GWM_COLOR, "GWM", args.bin_km),
        ("b)", "PGV", "pgv_obs_e", "pgv_synth_e", gwm_sel,
         gwm_label, GWM_COLOR, "GWM", args.bin_km),
        ("c)", "PGA", "pga_obs_rot", pga_key, gmm_sel,
         gmm_label, GMM_COLOR, "GMM", gmm_bin_km),
        ("d)", "PGV", "pgv_obs_rot", pgv_key, gmm_sel,
         gmm_label, GMM_COLOR, "GMM", gmm_bin_km),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (tag, im, obs_key, pred_key, base_sel, label, color, denom,
             bin_km) in zip(axes.ravel(), panels):
        obs, pred = cache[obs_key], cache[pred_key]
        sel = (base_sel & np.isfinite(obs) & np.isfinite(pred)
               & (obs > 0) & (pred > 0))
        x, y = r_hyp[sel], np.log10(obs[sel] / pred[sel])

        ax.scatter(x, y, s=2, color=SCATTER_COLOR, alpha=0.25, lw=0, rasterized=True)
        ax.axhline(0.0, color=INK, lw=1.0)
        centers, means, stds = _binned(x, y, bin_km, args.min_bin_count)
        ax.errorbar(centers, means, yerr=stds, fmt="none", ecolor=color,
                    elinewidth=1.3, capsize=0, alpha=0.85)
        ax.plot(centers, means, color=color, lw=1.6)

        mean_bias = float(np.mean(y))
        near = y[x < 20.0]
        near_txt = f", <20 km {near.mean():+.2f}" if near.size >= args.min_bin_count else ""
        print(f"[eval] {tag} {im} vs {denom}: n={sel.sum()}  "
              f"mean bias {mean_bias:+.2f} log10 units "
              f"({(10 ** mean_bias - 1) * 100:+.0f}% under/over-prediction{near_txt})")

        ax.set_ylim(-3, 3)
        ax.set_ylabel(f"Log$_{{10}}$({im}$_{{obs}}$/{im}$_{{{denom}}}$)", color=INK)
        ax.text(0.02, 0.04, f"{label}  (n={int(sel.sum()):,})",
                transform=ax.transAxes, fontsize=9, style="italic", color=INK)
        ax.text(-0.08, 1.03, tag, transform=ax.transAxes, fontsize=12, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(MUTED)
    for ax in axes[1]:
        ax.set_xlabel("Hypocentral Distance [km]", color=INK)

    fig.suptitle(
        f"Peak amplitude model bias vs hypocentral distance — "
        f"GWM (test split, E comp.) and {gmm['label']} (RotD50)",
        color=INK, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUT_DIR / (f"fig_{path.stem.removeprefix('peaks_')}"
                     f"_{args.gmm}_{args.set_mode}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[eval] figure saved: {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stage", choices=["compute", "plot", "all"], nargs="?",
                        default="all")
    parser.add_argument("--gmm", choices=sorted(GMM_REGISTRY), default="bssa14",
                        help="Ground motion model for panels c/d (see module "
                             "docstring for validity notes).")
    parser.add_argument("--gmm_min_mag", type=float, default=None,
                        help="Magnitude floor for the GMM rows "
                             "(default: the chosen model's validity floor).")
    parser.add_argument("--scope", choices=["val", "all"], default="val",
                        help="compute: which records get GMM predictions. 'val' "
                             "= test split only (enough for --set_mode val/"
                             "matched), 'all' = every eligible record in the "
                             "dataset (needed for --set_mode all).")
    parser.add_argument("--cache", type=str, default=None,
                        help="Synthetic-waveform cache from evaluate_first_order.py "
                            "(default: most recent in eval/first_order).")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(EMBEDDINGS_DIR),
        help="Embedding export directory used for metadata and station artifacts.",
    )
    parser.add_argument("--peaks_cache", type=str, default=None,
                        help="Peaks .npz for `plot` (default: most recent).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only N evenly-spaced records (0 = all).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=512,
                        help="Records per pool chunk between cache saves.")
    parser.add_argument("--set_mode", choices=["val", "all", "matched"], default="val",
                        help="Record set for the GMM panels: 'val' = test split "
                             "only (same population as the GWM panels), 'all' = every "
                             "M>=floor record in the dataset, 'matched' = restrict ALL "
                             "panels to the common record set (identical n everywhere).")
    parser.add_argument("--bin_km", type=float, default=2.5,
                        help="Distance bin width for the mean/std overlay.")
    parser.add_argument("--gmm_bin_km", type=float, default=None,
                        help="Bin width for the GMM panels (default: --bin_km). "
                             "Useful when the GMM record set is much smaller.")
    parser.add_argument("--min_bin_count", type=int, default=5)
    args = parser.parse_args()
    configure_embeddings_dir(args.embeddings_dir)

    if args.stage in ("compute", "all"):
        path = run_compute(args)
        if args.stage == "all" and args.peaks_cache is None:
            args.peaks_cache = str(path)
    if args.stage in ("plot", "all"):
        run_plot(args)


if __name__ == "__main__":
    main()
